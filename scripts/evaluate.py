"""Aggregate Phase 1 runs into a results table with bootstrap CIs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SKIP_KEYS = {"config", "curve", "git_sha", "run_name", "top_dog_confusions"}


def load_runs(results_dir: Path) -> list[dict]:
    paths = sorted(results_dir.glob("*/run.json"))
    if not paths:
        raise SystemExit(f"no run.json files under {results_dir}")
    return [json.loads(path.read_text()) for path in paths]


def assert_no_test_tuning(records: list[dict]) -> None:
    """Reject runs that used val_test for tuning or val_tune for final metrics."""
    for record in records:
        if record.get("eval_split") != "val_test" or record.get("tune_split") != "val_tune":
            raise ValueError(
                f"{record.get('method')} seed={record.get('seed')} used an invalid split"
            )


def flatten(record: dict, prefix: str = "") -> dict[str, float]:
    """Flatten nested metric dictionaries, skipping curves and text fields."""
    values: dict[str, float] = {}
    for key, value in record.items():
        if key in SKIP_KEYS or key.endswith("_curve"):
            continue
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            values.update(flatten(value, f"{name}."))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            values[name] = float(value)
    return values


def bootstrap_ci(values: np.ndarray, seed: int, samples: int = 1000) -> tuple[float, float]:
    if len(values) < 2:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def aggregate(records: list[dict], seed: int = 0, samples: int = 1000) -> list[dict]:
    """Group runs by method and compute mean, std and bootstrap CI per metric."""
    by_method: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_method[record["method"]].append(flatten(record))

    rows: list[dict] = []
    for method in sorted(by_method):
        runs = by_method[method]
        metrics = sorted(set().union(*(set(run) for run in runs)))
        for metric in metrics:
            values = np.array([run[metric] for run in runs if metric in run], dtype=float)
            low, high = bootstrap_ci(values, seed, samples)
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "n": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return rows


def write_confusions(records: list[dict], results_dir: Path) -> None:
    for record in records:
        confusions = record.get("metrics", {}).get("top_dog_confusions", {})
        if not confusions:
            continue
        path = results_dir / f"confusions_{record['method']}_s{record['seed']}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["predicted_class", "count"])
            writer.writerows(sorted(confusions.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/phase1"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    records = load_runs(args.results_dir)
    assert_no_test_tuning(records)
    rows = aggregate(records, args.seed, args.bootstrap)
    write_confusions(records, args.results_dir)

    table = args.results_dir / "results_table.csv"
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["method", "metric", "n", "mean", "std", "ci_low", "ci_high"]
        )
        writer.writeheader()
        writer.writerows(rows)
    (args.results_dir / "aggregate.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {len(rows)} rows to {table}")


if __name__ == "__main__":
    main()
