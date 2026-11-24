"""Run all Phase 1 methods and seeds from one committed config."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import yaml

from unlearning.classes import Group, load_class_index
from unlearning.data.paths import get_root
from unlearning.methods import mask
from unlearning.data.tfdata import Purpose, build_dataset
from unlearning.metrics import evaluate
from unlearning.mia import run_mia
from unlearning.models import load_resnet50
from unlearning.probes import evaluate_probes
from unlearning.relearn import run_relearning
from unlearning.seeding import set_seed
from unlearning.splits import DataConfig, Split, load_manifest, write_manifest
from unlearning.train import TrainConfig, _git_sha, collect_logits, train_method, run_name


def finalize_logits(method: str, logits, index):
    """Apply inference-only masking to final metrics for the mask baseline."""
    if method == "mask":
        return mask.mask_logits(logits, index.indices(Group.DOG))
    return logits


def _dataset(manifest, root, split, purpose, batch_size, max_batches, group=None):
    dataset = build_dataset(
        manifest, root, split, purpose, batch_size=batch_size, group=group
    )
    return dataset.take(max_batches) if max_batches else dataset


def build_datasets(manifest, root, batch_size, max_batches, tune_only):
    """Build the datasets needed for a full run or a val_tune-only run."""
    datasets = {
        "forget": _dataset(
            manifest, root, Split.TRAIN_FORGET, Purpose.TRAIN, batch_size, max_batches
        ),
        "retain": _dataset(
            manifest, root, Split.TRAIN_RETAIN, Purpose.TRAIN, batch_size, max_batches
        ),
        "tune": _dataset(
            manifest, root, Split.VAL_TUNE, Purpose.TUNE, batch_size, max_batches
        ),
    }
    if tune_only:
        return datasets

    datasets.update(
        {
            "test": _dataset(
                manifest, root, Split.VAL_TEST, Purpose.EVAL, batch_size, max_batches
            ),
            "forget_test": _dataset(
                manifest, root, Split.VAL_TEST, Purpose.EVAL,
                batch_size, max_batches, "dog",
            ),
            "retain_test": _dataset(
                manifest, root, Split.VAL_TEST, Purpose.EVAL,
                batch_size, max_batches, "canid",
            ).concatenate(
                _dataset(
                    manifest, root, Split.VAL_TEST, Purpose.EVAL,
                    batch_size, max_batches, "other",
                )
            ),
        }
    )
    return datasets


def eval_split(tune_only: bool) -> str:
    """Return the split used for final metrics in the current mode."""
    return Split.VAL_TUNE.value if tune_only else Split.VAL_TEST.value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase1_resnet50.yaml"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--tune-only",
        action="store_true",
        help="Evaluate only on val_tune; never build or read val_test.",
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    manifest = Path(args.manifest or config["dataset"]["manifest"])
    root = get_root(args.root or config["dataset"]["root"])
    index_path = Path(config["dataset"]["index"])
    index = load_class_index(index_path)
    batch_size = args.batch_size or config["batch_size"]
    steps = args.steps or config["steps"]
    max_batches = (
        config["max_batches"] if args.max_batches is None else args.max_batches
    )
    methods = (args.methods.split(",") if args.methods else config["methods"])
    seeds = (
        [int(value) for value in args.seeds.split(",")]
        if args.seeds
        else config["seeds"]
    )
    out_root = Path(args.output or config["results_dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(manifest)

    datasets = build_datasets(
        manifest, root, batch_size, max_batches, args.tune_only
    )
    forget_ds = datasets["forget"]
    retain_ds = datasets["retain"]
    tune_ds = datasets["tune"]

    teacher = load_resnet50()
    teacher_tune = None
    teacher_test = None

    if args.tune_only:
        teacher_tune, _ = collect_logits(teacher, tune_ds)
    else:
        test_ds = datasets["test"]
        forget_test_ds = datasets["forget_test"]
        retain_test_ds = datasets["retain_test"]
        probe_train_ds = forget_ds.concatenate(retain_ds)
        teacher_test, teacher_labels = collect_logits(teacher, test_ds)
        original_top1 = float(
            evaluate(teacher_test, teacher_labels, index)["forget_top1"]
        )

    for seed in seeds:
        set_seed(seed)
        for method in methods:
            run_cfg = TrainConfig(
                method=method,
                seed=seed,
                steps=steps,
                batch_size=batch_size,
                lr=config["lr"],
                weight_decay=config["weight_decay"],
                grad_clip=config["grad_clip"],
                eval_every=config["eval_every"],
                lambda_retain=config["lambda_retain"],
                scrub_alpha=config["scrub_alpha"],
                scrub_beta=config["scrub_beta"],
                patience=config["patience"],
                min_delta=config["min_delta"],
                save_checkpoints=config["save_checkpoints"],
            )
            student = load_resnet50()
            started = time.monotonic()
            if args.tune_only:
                out_dir = out_root / "tune" / f"{method}_s{seed}"
            else:
                out_dir = out_root / f"{method}_s{seed}"
            trained, curve = train_method(
                student, forget_ds, retain_ds, tune_ds, run_cfg, index,
                teacher=teacher, out_dir=out_dir,
            )
            for saved in out_dir.glob("*.keras"):
                saved.unlink()

            eval_ds = tune_ds if args.tune_only else test_ds
            eval_logits, eval_labels = collect_logits(trained, eval_ds)
            eval_logits = finalize_logits(method, eval_logits, index)
            teacher_logits = teacher_tune if args.tune_only else teacher_test
            record = {
                "method": method,
                "seed": seed,
                "git_sha": _git_sha(),
                "config": asdict(run_cfg),
                "run_name": run_name(run_cfg),
                "metrics": evaluate(eval_logits, eval_labels, index, teacher_logits),
                "curve": curve,
                "runtime_sec": round(time.monotonic() - started, 3),
                "tune_split": Split.VAL_TUNE.value,
                "eval_split": eval_split(args.tune_only),
                "tune_only": bool(args.tune_only),
            }
            if not args.tune_only:
                record["probes"] = evaluate_probes(
                    trained, probe_train_ds, test_ds, index,
                    max_samples=config["probe_max_samples"], seed=seed,
                )
                record["mia"] = run_mia(
                    trained, forget_ds, forget_test_ds, retain_ds, retain_test_ds
                )
                with tempfile.TemporaryDirectory() as tmp:
                    relearn_path = Path(tmp) / "relearn.csv"

                    def factory(selected):
                        write_manifest(selected, relearn_path, index_path, DataConfig())
                        return _dataset(
                            relearn_path, root, Split.RELEARN, Purpose.TRAIN,
                            batch_size, max_batches,
                        )

                    record["relearning"] = run_relearning(
                        trained, rows, index, factory, test_ds,
                        original_top1=original_top1,
                        budgets=tuple(config["relearning_budgets"]),
                        epochs=config["relearning_epochs"],
                        seed=seed,
                    )
            (out_dir / "run.json").write_text(json.dumps(record, indent=2) + "\n")
            print(f"finished {method} seed={seed} -> {out_dir / 'run.json'}")


if __name__ == "__main__":
    main()
