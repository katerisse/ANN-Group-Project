# ANN Group Project

Phase 1 builds a reproducible ImageNet data hook for class-unlearning experiments.
The repo stores code, configs, and manifests only. Images, archives, weights, and
credentials stay outside Git.

## Setup

```bash
python -m pip install -r requirements.txt
python -m pytest
```

## Data location

Point the pipeline at a local copy of ILSVRC2012:

```bash
export IMAGENET_ROOT="$HOME/datasets/imagenet"
```

Expected layout:

```text
$IMAGENET_ROOT/
├── raw/
│   ├── ILSVRC2012_img_val.tar
│   ├── ILSVRC2012_devkit_t12.tar.gz
│   └── ILSVRC2012_img_train.tar   # only after --include-train
├── train/<synset>/*.JPEG
├── val/<synset>/*.JPEG
└── devkit/data/
    ├── meta.mat
    └── ILSVRC2012_validation_ground_truth.txt
```

The val tar contains flat JPEGs; the hook uses the devkit ground truth to place
them into synset folders. Never point `IMAGENET_ROOT` at a mounted Google Drive
path: download once to local disk, then train from there.

## Acquiring the data

Choose one source:

```bash
# Verify files you already downloaded
python scripts/download_imagenet.py --source local --root "$IMAGENET_ROOT"

# Shared Google Drive via rclone
export IMAGENET_GDRIVE_REMOTE="ann-drive:ANN-Group-Project/imagenet"
python scripts/download_imagenet.py --source gdrive --root "$IMAGENET_ROOT"

# Kaggle (needs KAGGLE_USERNAME and KAGGLE_KEY)
python scripts/download_imagenet.py --source kaggle --root "$IMAGENET_ROOT"

# Hugging Face (needs HF_TOKEN)
python scripts/download_imagenet.py --source hf --root "$IMAGENET_ROOT"

# Manual TFDS download: place archives in raw/ first
python scripts/download_imagenet.py --source tfds-manual --root "$IMAGENET_ROOT"
```

The default is val + devkit only. Add `--include-train` for the 138 GB train
archive. Credentials are read from the environment only and are never printed
or committed. Add `--verify-only` to check existing files without downloading.

ImageNet access is for non-commercial research/education. Share the data only
with teammates who accept the official terms; do not make the Drive link public.

## Building and auditing manifests

```bash
python scripts/build_manifests.py --root "$IMAGENET_ROOT" --seed 2026

python scripts/audit_images.py \
  --root "$IMAGENET_ROOT" \
  --manifest manifests/phase1_splits.csv \
  --devkit "$IMAGENET_ROOT/devkit" \
  --apply
```

This writes:

- `manifests/phase1_splits.csv` — split, source split, relative path, class
  indices, synset, name, and group.
- `manifests/phase1_splits.meta.json` — seed, counts, and class-index hash.
- `manifests/phase1_splits.sha256` — manifest checksum.
- `reports/` — quarantine, exact-duplicate, near-duplicate, and summary reports.

The same seed and the same images reproduce the same manifest SHA256. Only the
CSV and hashes are committed; images are never committed.

## Loading batches

```python
from unlearning.data.tfdata import build_eval_dataset, build_tuning_dataset, build_train_dataset
from unlearning.splits import Split

train = build_train_dataset(
    manifest_path="manifests/phase1_splits.csv",
    root="$IMAGENET_ROOT",
    split=Split.TRAIN_FORGET,
    batch_size=32,
)
```

`build_tuning_dataset` accepts only `val_tune`; `build_eval_dataset` accepts
only `val_test`. Requesting the test split for tuning raises
`TestSplitLeakageError`, so tuning code cannot accidentally read the held-out
split.

ResNet50 preprocessing is `resnet50.preprocess_input` semantics: decode RGB,
resize the shorter side to 256, crop to 224, RGB→BGR, and subtract
`[103.939, 116.779, 123.68]` without scaling. Keep model-specific transforms
behind the `Preprocessor` interface so CLIP can replace them later.

## Current status

PR1 adds the class index and acquisition CLI, PR2 adds splits and auditing, and
PR3 adds preprocessing, the loader, and this documentation. The actual shared
Drive upload and real manifest generation are deliberately deferred until the
data is available.
