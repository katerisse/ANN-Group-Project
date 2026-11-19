# ANN Group Project

Phase 1 builds a reproducible ImageNet data hook for class-unlearning experiments.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Data location

ImageNet is stored outside this repository. Point the pipeline at it with:

```bash
export IMAGENET_ROOT="$HOME/datasets/imagenet"
```

Expected layout:

```text
$IMAGENET_ROOT/
├── raw/                 # original archives, never committed
├── train/<synset>/*.JPEG
├── val/<synset>/*.JPEG
└── devkit/              # meta.mat and validation ground truth
```

No images, archives, weights, or credentials are committed. Full data access
instructions are added in PR3.
