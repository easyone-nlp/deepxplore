# DeepXplore on CIFAR-10 ResNet50 Models

This repository adapts the original DeepXplore-style workflow to run on three ResNet50 models trained on CIFAR-10.

The project is organized in two stages:

1. Train multiple CIFAR-10 ResNet50 models with different training settings.
2. Run DeepXplore on those trained checkpoints through `test.py`.

## Environment Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Activate the environment before training or testing:

```bash
conda activate ...
```

## 1. Train the ResNet50 Models

The training code is under:

```bash
./CIFAR-10/
```

Main training entrypoint:

```bash
python CIFAR-10/main.py --help
```

### Training setup

We trained three ResNet50 models on CIFAR-10 with different seeds and epoch settings:

- `resnet50_cifar10_seed42.pth`
- `resnet50_cifar10_seed123_ep20.pth`
- `resnet50_cifar10_seed42_ep30.pth`

Example training commands:

```bash
python CIFAR-10/main.py \
  --model resnet50 \
  --dataset cifar10 \
  --epochs 20 \
  --seed 42 \
  --checkpoint-path ./resnet50_cifar10_seed42.pth \
  --plot-path ./resnet50_cifar10_seed42.png
```

```bash
python CIFAR-10/main.py \
  --model resnet50 \
  --dataset cifar10 \
  --epochs 20 \
  --seed 123 \
  --checkpoint-path ./resnet50_cifar10_seed123_ep20.pth \
  --plot-path ./resnet50_cifar10_seed123_ep20.png
```

```bash
python CIFAR-10/main.py \
  --model resnet50 \
  --dataset cifar10 \
  --epochs 30 \
  --seed 42 \
  --checkpoint-path ./resnet50_cifar10_seed42_ep30.pth \
  --plot-path ./resnet50_cifar10_seed42_ep30.png
```

### Training preprocessing

To keep preprocessing consistent between training and DeepXplore testing:

- CIFAR-10 images are resized from `32x32` to `224x224`
- the same normalization is used in both stages:
  - `mean=(0.5, 0.5, 0.5)`
  - `std=(0.5, 0.5, 0.5)`

This logic is implemented in:

- [CIFAR-10/train.py](./CIFAR-10/train.py)
- [cifar10_gen_diff.py](./cifar10_gen_diff.py)

## 2. Run DeepXplore with `test.py`

The submission entrypoint is:

```bash
python test.py
```

This script loads the trained CIFAR-10 ResNet50 checkpoints and runs the adapted DeepXplore workflow.

Example command:

```bash
python test.py \
  --transformation occl \
  --seeds 100 \
  --grad-iterations 200 \
  --step 0.1 \
  --threshold 0.5 \
  --output-dir ./results
```

### What `test.py` does

- loads the three trained ResNet50 checkpoints
- uses CIFAR-10 test images as seed inputs
- applies DeepXplore-style gradient-based input generation
- records disagreement-inducing inputs
- records neuron coverage
- saves generated results into `results/`

## Output Files

The main outputs are written to:

```bash
./results/
```

Important files:

- `results/summary.json`
- `results/final_metrics.txt`
- `results/final_metrics.json`
- `results/selected_cases/`

The `results/selected_cases/` directory contains five representative disagreement-inducing examples selected for analysis.

## Project-Specific Modifications

Compared with the original ImageNet-oriented prototype, this repository was modified to:

- use CIFAR-10 instead of the original ImageNet seed folder
- train and load custom CIFAR-10 ResNet50 checkpoints
- use a dedicated CIFAR-10 DeepXplore runner: `cifar10_gen_diff.py`
- provide a submission-friendly entrypoint: `test.py`
- separate already-disagreeing seeds from newly induced DeepXplore cases using `filter_induced_cases.py`

## Notes

- The original `seeds/` directory from the prototype is not required for the CIFAR-10 workflow.
- During gradient ascent, perturbed inputs are not clamped after every step; this follows the behavior of the original DeepXplore-style implementation more closely.
- PNG outputs are denormalized before saving for visualization.
