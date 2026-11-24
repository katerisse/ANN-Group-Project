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

## Model layer

Phase 1 uses the unmodified Keras ImageNet ResNet50: 25,636,712 parameters,
53 BatchNorm layers, a 224×224 RGB input, and 1000 logits
(`classifier_activation=None`). Verify the pinned weights before use:

```bash
python scripts/verify_weights.py
```

Expected file: `resnet50_weights_tf_dim_ordering_tf_kernels.h5`, 102,967,424
bytes, MD5 `2cb95161c43110f7111970584f804107`, SHA256
`7011d39ea4f61f4ddb8da99c4addf3fae4209bfda7828adb4698b16283258fbe`.
The loader fails closed on any mismatch and the weights are never committed.

Use `unlearning.seeding.set_seed(seed)` for every run; for full determinism
also export `TF_DETERMINISTIC_OPS=1`, `TF_CUDNN_DETERMINISTIC=1`, and
`PYTHONHASHSEED=0` before Python starts.

All methods freeze the 53 BatchNorm layers so neither moving statistics nor
`gamma`/`beta` change. `FrozenStatsBatchNormalization` is the ablation that
keeps statistics fixed while allowing `gamma`/`beta` to train.

## Local ImageNet copy and manifests

The on-device copy at `~/Documents/imagenet` has 1000 human-readable class
folders (539,826 images) rather than the ILSVRC synset/train/val layout.
`LocalLayout` maps those folders to the committed class index and resolves
ambiguous names with explicit aliases; it fails closed on unknown or duplicate
folders. Regenerate the split manifest with:

```bash
PYTHONPATH=src python scripts/build_local_manifest.py --root ~/Documents/imagenet
```

This writes `manifests/phase1_local.csv`, a metadata sidecar, a checksum, and
`manifests/phase1_local_short_classes.json`. Generated manifests are ignored by
git. All splits are drawn from one seeded per-class pool, so train, relearn,
`val_tune` and `val_test` are disjoint. Four dog classes have fewer than the
requested 170 images (`black_and_tan_coonhound` 135, `affenpinscher` 153,
`sealyham_terrier` 161, `otterhound` 169); their budgets are scaled
proportionally and recorded in the short-class report instead of being hidden.

## CPU proxy

`docs/phase1_cpu_proxy.md` records the 60-row, 2-step CPU smoke test across all
six methods and three seeds, with reproduction commands, key metrics and its
limitations. The committed `configs/phase1_cpu_proxy.yaml` and
`manifests/phase1_proxy.csv` make that run reproducible; generated outputs under
`results/` are ignored. The proxy is not an acceptance result and issue #7
remains open.

## Current status

PR1 adds the class index and acquisition CLI, PR2 adds splits and auditing,
PR3 adds preprocessing and the loader, PR4 adds the pinned model, PR5–PR7 add
metrics, methods and the training loop, PR8–PR16 add the local adapter,
manifests, harness, probes, MIA, relearning, runner, aggregation, mask fix and
proxy documentation. The full experiments remain outstanding.
