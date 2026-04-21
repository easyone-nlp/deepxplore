import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter DeepXplore results into newly induced disagreement cases."
    )
    parser.add_argument(
        "--summary",
        default="./results/summary.json",
        help="Path to the DeepXplore summary.json file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write filtered outputs. Defaults to the summary's parent directory.",
    )
    return parser.parse_args()


def all_same(items):
    return len(set(items)) == 1


def main():
    args = parse_args()
    summary_path = Path(args.summary).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else summary_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    records = data.get("records", [])

    induced = []
    already_disagreeing = []

    for record in records:
        initial = record.get("initial_predictions", [])
        final = record.get("final_predictions", [])
        diverged = record.get("diverged", False)

        if not diverged:
            continue

        if initial and all_same(initial) and not all_same(final):
            induced.append(record)
        else:
            already_disagreeing.append(record)

    induced_json = output_dir / "induced_only.json"
    induced_txt = output_dir / "induced_only.txt"
    already_json = output_dir / "already_disagreeing.json"

    induced_payload = {
        "summary_path": str(summary_path),
        "induced_count": len(induced),
        "records": induced,
    }
    induced_json.write_text(json.dumps(induced_payload, indent=2), encoding="utf-8")

    already_payload = {
        "summary_path": str(summary_path),
        "already_disagreeing_count": len(already_disagreeing),
        "records": already_disagreeing,
    }
    already_json.write_text(json.dumps(already_payload, indent=2), encoding="utf-8")

    with induced_txt.open("w", encoding="utf-8") as file:
        file.write(f"summary: {summary_path}\n")
        file.write(f"induced_count: {len(induced)}\n")
        file.write(f"already_disagreeing_count: {len(already_disagreeing)}\n")
        file.write("\n")
        for index, record in enumerate(induced, start=1):
            file.write(f"[{index}]\n")
            file.write(f"dataset_index: {record.get('dataset_index')}\n")
            file.write(f"true_label: {record.get('true_label')}\n")
            file.write(f"initial_predictions: {record.get('initial_predictions')}\n")
            file.write(f"final_predictions: {record.get('final_predictions')}\n")
            file.write(f"coverage: {record.get('coverage')}\n")
            file.write(f"generated_path: {record.get('generated_path')}\n")
            file.write(f"original_path: {record.get('original_path')}\n")
            file.write("\n")

    print(f"Induced-only cases: {len(induced)}")
    print(f"Already-disagreeing cases: {len(already_disagreeing)}")
    print(f"Saved {induced_json}")
    print(f"Saved {induced_txt}")
    print(f"Saved {already_json}")


if __name__ == "__main__":
    main()
