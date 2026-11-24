import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from evaluate import aggregate, assert_no_test_tuning, flatten  # noqa: E402


def _record(method: str, seed: int, value: float) -> dict:
    return {
        "method": method,
        "seed": seed,
        "tune_split": "val_tune",
        "eval_split": "val_test",
        "metrics": {"forget_top1": value, "top_dog_confusions": {"tench": 2}},
        "curve": [{"step": 1}],
    }


def test_flatten_skips_curves_and_confusions():
    values = flatten(_record("scrub", 1, 0.1))

    assert values == {"seed": 1.0, "metrics.forget_top1": 0.1}


def test_aggregate_computes_mean_std_and_ci():
    rows = aggregate([_record("scrub", 1, 0.1), _record("scrub", 2, 0.3)])

    row = next(row for row in rows if row["metric"] == "metrics.forget_top1")
    assert row["n"] == 2
    assert row["mean"] == pytest.approx(0.2)
    assert row["std"] == pytest.approx(0.1)
    assert row["ci_low"] <= row["mean"] <= row["ci_high"]


def test_test_split_guard_rejects_tuning_on_test():
    bad = _record("scrub", 1, 0.1)
    bad["tune_split"] = "val_test"

    with pytest.raises(ValueError, match="invalid split"):
        assert_no_test_tuning([bad])


def test_evaluate_cli_writes_results_table(tmp_path):
    for seed, value in ((1, 0.1), (2, 0.3)):
        directory = tmp_path / f"scrub_s{seed}"
        directory.mkdir()
        (directory / "run.json").write_text(json.dumps(_record("scrub", seed, value)))

    result = subprocess.run(
        [sys.executable, "scripts/evaluate.py", "--results-dir", str(tmp_path)],
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO / "scripts")},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "results_table.csv").is_file()
    assert (tmp_path / "confusions_scrub_s1.csv").is_file()
