import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def test_phase1_config_is_complete():
    config = yaml.safe_load((REPO / "configs/phase1_resnet50.yaml").read_text())

    assert set(config["methods"]) == {
        "original",
        "mask",
        "head_only",
        "retain_only",
        "neggrad",
        "scrub",
    }
    assert len(config["seeds"]) >= 3
    assert config["dataset"]["manifest"].startswith("manifests/")
    assert config["max_batches"] == 0


def test_run_experiment_help_lists_proxy_controls():
    result = subprocess.run(
        [sys.executable, "scripts/run_experiment.py", "--help"],
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    for flag in ("--config", "--steps", "--max-batches", "--methods", "--seeds"):
        assert flag in result.stdout
