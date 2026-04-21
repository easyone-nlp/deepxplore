import argparse
import json
import os
import random
from pathlib import Path

import torch
from PIL import Image
from torchvision import datasets, transforms

from deepxplore import deepXplore
from utils import constraint_black, constraint_light, constraint_occl

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "CIFAR-10"))
from resnet import resnet50  # noqa: E402


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DeepXplore on three CIFAR-10 ResNet50 checkpoints."
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[
            "./resnet50_cifar10_seed42.pth",
            "./resnet50_cifar10_seed123_ep20.pth",
            "./resnet50_cifar10_seed42_ep30.pth",
        ],
        help="Checkpoint paths for the ResNet50 models under test.",
    )
    parser.add_argument(
        "--transformation",
        choices=["light", "occl", "blackout"],
        default="occl",
        help="Constraint used on the input gradients.",
    )
    parser.add_argument("--dataset-root", default="./CIFAR-10/data", help="Root directory for CIFAR-10 data.")
    parser.add_argument("--output-dir", default="./results", help="Directory for generated results.")
    parser.add_argument("--seeds", type=int, default=20, help="Number of CIFAR-10 test samples to try.")
    parser.add_argument("--grad-iterations", type=int, default=300, help="Maximum gradient steps per seed.")
    parser.add_argument("--step", type=float, default=0.1, help="Gradient ascent step size.")
    parser.add_argument("--weight-diff", type=float, default=2.0, help="Weight for differential behavior objective.")
    parser.add_argument("--weight-nc", type=float, default=1.0, help="Weight for neuron coverage objective.")
    parser.add_argument("--threshold", type=float, default=0.75, help="Activation threshold for neuron coverage.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection and transformations.")
    parser.add_argument("--start-point", nargs=2, type=int, default=(0, 0), help="Occlusion start point (row col).")
    parser.add_argument("--occlusion-size", nargs=2, type=int, default=(50, 50), help="Occlusion size (h w).")
    parser.add_argument("--device", default=None, help="Torch device string. Defaults to cuda if available.")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = checkpoint.get("num_classes", 10)
    model = resnet50(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)
    return model, checkpoint


def build_dataset(dataset_root):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return datasets.CIFAR10(root=dataset_root, train=False, download=True, transform=transform)


def denormalize(tensor):
    return torch.clamp(tensor * 0.5 + 0.5, 0.0, 1.0)


def to_image(img_tensor):
    img = denormalize(img_tensor.detach().cpu().squeeze(0))
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255.0).round().clip(0, 255).astype("uint8")
    return Image.fromarray(img)


def labels_from_models(models, batch):
    with torch.no_grad():
        outputs = [model(batch).squeeze(0) for model in models]
    return [int(output.argmax().item()) for output in outputs]


def choose_constraint(args):
    if args.transformation == "light":
        return constraint_light
    if args.transformation == "occl":
        start_point = tuple(args.start_point)
        rect_shape = tuple(args.occlusion_size)

        def occl(grad):
            return constraint_occl(grad, start_point, rect_shape)

        return occl

    return constraint_black


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    checkpoint_paths = [str(Path(path).resolve()) for path in args.checkpoints]

    models = []
    checkpoint_info = []
    for checkpoint_path in checkpoint_paths:
        model, checkpoint = load_checkpoint_model(checkpoint_path, device)
        models.append(model)
        checkpoint_info.append(
            {
                "path": checkpoint_path,
                "epoch": checkpoint.get("epoch"),
                "seed": checkpoint.get("seed"),
                "dataset_name": checkpoint.get("dataset_name"),
                "num_classes": checkpoint.get("num_classes"),
            }
        )

    dataset = build_dataset(args.dataset_root)
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    dxp = deepXplore(
        models,
        itr_num=args.grad_iterations,
        lambda_1=args.weight_diff,
        lambda_2=args.weight_nc,
        threshold=args.threshold,
        s=args.step,
    )
    constraint = choose_constraint(args)

    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)

    records = []
    tried = 0
    disagreements = 0

    for dataset_index in indices[: args.seeds]:
        image_tensor, true_label = dataset[dataset_index]
        batch = image_tensor.unsqueeze(0).to(device)
        original = batch.detach().clone()

        initial_labels = labels_from_models(models, batch)
        if len(set(initial_labels)) > 1:
            final_batch = batch
            final_labels = initial_labels
        else:
            final_batch = dxp.generate(batch, constraint)
            final_labels = labels_from_models(models, final_batch)

        tried += 1
        record = {
            "dataset_index": dataset_index,
            "true_label": CIFAR10_CLASSES[true_label],
            "initial_predictions": [CIFAR10_CLASSES[label] for label in initial_labels],
            "final_predictions": [CIFAR10_CLASSES[label] for label in final_labels],
            "coverage": [float(value) for value in dxp.get_coverage()],
            "diverged": len(set(final_labels)) > 1,
        }

        if record["diverged"]:
            disagreements += 1
            stem = f"idx{dataset_index:05d}_{record['true_label']}_{args.transformation}"
            generated_path = output_dir / f"{stem}_generated.png"
            original_path = output_dir / f"{stem}_original.png"
            to_image(final_batch).save(generated_path)
            to_image(original).save(original_path)
            record["generated_path"] = str(generated_path)
            record["original_path"] = str(original_path)

        records.append(record)
        print(
            f"[{tried}/{args.seeds}] idx={dataset_index} true={record['true_label']} "
            f"initial={record['initial_predictions']} final={record['final_predictions']} "
            f"coverage={[round(value, 4) for value in record['coverage']]}"
        )

    summary = {
        "device": str(device),
        "checkpoints": checkpoint_info,
        "tried": tried,
        "disagreements": disagreements,
        "records": records,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"Saved summary to {summary_path}")
    print(f"Disagreement-inducing inputs: {disagreements}/{tried}")


if __name__ == "__main__":
    main()
