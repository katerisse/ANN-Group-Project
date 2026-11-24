# Phase 1 CPU proxy results

This page records a **smoke test** of the full Phase 1 pipeline on this CPU-only
machine. It is not an acceptance result and does not close issue #7. The full
1000-class, multi-seed ImageNet experiment remains a GPU-machine task.

## What was run

All six methods (`original`, `mask`, `head_only`, `retain_only`, `neggrad`,
`scrub`) were run for seeds 2026, 2027 and 2028: 18 runs total. Each run used
2 optimizer steps, batch size 4, learning rate `1e-4`, and `eval_every: 1`.
The pinned Keras ImageNet ResNet50 was used for the teacher, students and
probes; all 53 BatchNorm layers were frozen and trainable updates were limited
to `conv5 + head` (or the head only for `head_only`).

The proxy used `manifests/phase1_proxy.csv`: a 60-row balanced sample drawn
with `random.Random(11)` from the real `manifests/phase1_local.csv`. All paths
are relative to `~/Documents/imagenet`; no images are stored in Git.

| split | dog | canid | other |
| --- | ---: | ---: | ---: |
| `train_forget` | 8 | 0 | 0 |
| `train_retain` | 0 | 4 | 4 |
| `relearn` | 8 | 0 | 0 |
| `val_tune` | 8 | 4 | 4 |
| `val_test` | 8 | 4 | 8 |

`val_tune` was used for periodic curves and plausibility; `val_test` was passed
only to final evaluation. The `Purpose` guard in `build_dataset()` rejects
`val_test` for training or tuning.

## Environment

- Python 3.12, TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.4.4, Pillow 11.0.0,
  scikit-learn 1.8.0, PyYAML 6.0.2.
- Apple M3 Pro, but TensorFlow sees no GPU or Metal device; runs are CPU-only.
- ResNet50 weights were verified before use: 102,967,424 bytes, MD5
  `2cb95161c43110f7111970584f804107`, SHA256
  `7011d39ea4f61f4ddb8da99c4addf3fae4209bfda7828adb4698b16283258fbe`.

## Reproduction

From the repository root, with the local dataset at `~/Documents/imagenet`:

```bash
python -m pip install -r requirements.txt
python scripts/verify_weights.py

# Optional audit: rebuild the real manifest and verify its checksum.
PYTHONPATH=src python scripts/build_local_manifest.py \
  --root "$HOME/Documents/imagenet" \
  --output manifests/phase1_local.csv
shasum -a 256 manifests/phase1_local.csv
# d9ca46b4427ebbe5de0c9c7f2a7dc539659edf196b38e0d32f3ca16ac1a4f9e9

shasum -a 256 manifests/phase1_proxy.csv
# 2a6aadb39dd3a62f1937aa2d0bafa717fef2922ef34b207e8031a592c3b7da40

PYTHONPATH=src python scripts/run_experiment.py \
  --config configs/phase1_cpu_proxy.yaml \
  --manifest manifests/phase1_proxy.csv \
  --root "$HOME/Documents/imagenet" \
  --batch-size 4 --steps 2 --max-batches 0 \
  --output results/proxy

PYTHONPATH=src python scripts/evaluate.py \
  --results-dir results/proxy --bootstrap 1000 --seed 0
```

The runner writes `run.json`, per-run curve CSVs and metadata under
`results/proxy/<method>_s<seed>/`. The aggregator writes
`results/proxy/results_table.csv`, `results/proxy/aggregate.json`, and
`confusions_<method>_<seed>.csv`. These generated outputs are gitignored.

## Headline metrics

Mean ± standard deviation over the three seeds. Full-precision values are in
the generated `results_table.csv`.

| method | forget top-1 | any-dog top-1 | dog mass | retain top-1 | canid top-1 | plausibility | runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `original` | 1.0000 | 1.0000 | 0.9999 | 0.8333 | 1.0000 | 1.0000 | 25.61 ± 0.64 |
| `mask` | 0.0000 | 0.0000 | 0.0000 | 0.8333 | 1.0000 | 0.1250 | 25.39 ± 0.30 |
| `head_only` | 1.0000 | 1.0000 | 0.9999 | 0.8333 | 1.0000 | 1.0000 | 28.25 ± 0.53 |
| `retain_only` | 1.0000 | 1.0000 | 1.0000 | 0.9167 | 1.0000 | 1.0000 | 28.43 ± 0.69 |
| `neggrad` | 1.0000 | 1.0000 | 0.9999 | 0.9167 | 1.0000 | 1.0000 | 29.41 ± 0.10 |
| `scrub` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 32.55 ± 0.59 |

## Diagnostics

Probe AUCs are dog-vs-rest and canid-vs-rest logistic-regression probes on
`avg_pool` features, fit on train-split features and scored on `val_test`.
MIA AUC uses true-class confidence with train members vs held-out examples.
Relearning uses the disjoint `relearn` split, from the same unlearned weights
for every budget; values below are budget-20 `val_test` dog top-1.

| method | dog probe AUC | canid probe AUC | forget MIA AUC | retain MIA AUC | relearn start → end |
| --- | ---: | ---: | ---: | ---: | ---: |
| `original` | 0.9062 | 0.8698 ± 0.0868 | 0.5469 | 0.7604 | 1.0000 → 0.2500 |
| `mask` | 0.9062 | 0.8698 ± 0.0868 | 0.5469 | 0.7604 | 1.0000 → 0.2500 |
| `head_only` | 0.9062 | 0.8698 ± 0.0868 | 0.5469 | 0.7812 | 1.0000 → 0.2500 |
| `retain_only` | 0.8333 | 0.8073 ± 0.0295 | 0.5312 | 0.9792 | 1.0000 → 0.5000 |
| `neggrad` | 0.8750 | 0.8281 ± 0.0383 | 0.4062 | 0.9792 | 1.0000 → 0.7500 |
| `scrub` | 0.8646 | 0.7969 ± 0.0383 | 0.6719 | 0.7396 | 1.0000 → 0.3750 |

## Interpretation and limitations

- This proxy verifies that the code path runs end to end. It does **not**
  satisfy issue #7's acceptance criteria.
- `head_only`, `retain_only`, `neggrad` and `scrub` still have 100% forget
  top-1 and dog mass near 1.0 after two steps. Two steps at `1e-4` on only
  8–16 training images per step are far too small to demonstrate unlearning.
- `mask` reports 0% forget accuracy and 0 dog mass, but its dog probe AUC is
  unchanged at 0.9062 and its forget MIA AUC is unchanged at 0.5469. That is
  inference-time suppression, not real forgetting.
- `scrub` did not dominate the baselines: retain top-1 is 1.0, but forget top-1
  is still 1.0, dog probe AUC is 0.8646, and forget MIA AUC is 0.6719 (higher
  than the original 0.5469). No superiority claim is supported.
- Relearning starts at dog top-1 1.0 for every method, so `recovery_fraction`
  is 0 by definition and the curves cannot distinguish methods. The end values
  are reported for transparency, not as evidence.
- Runtime is CPU-only and varies with the tiny proxy's inference overhead; it
  is not a throughput estimate for full ImageNet.
- Full runs need the pinned 1000-class dataset, 118 dog breeds, 7 canids,
  100k-scale train/eval images and multiple seeds on a GPU machine. Those
  runs are blocked here.

## Status

Issue #7 stays open. The next diagnostic step is a larger CPU proxy with more
steps and a higher learning rate, tuned only on `val_tune`; `val_test` must
remain frozen until final evaluation.
