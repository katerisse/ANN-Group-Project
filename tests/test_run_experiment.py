import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from unlearning.classes import Group, load_class_index

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from run_experiment import finalize_logits  # noqa: E402


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


def test_finalize_logits_masks_only_the_mask_method():
    index = load_class_index()
    logits = np.ones((2, 1000))

    masked = finalize_logits("mask", logits, index)

    assert (masked[:, index.indices(Group.DOG)] == -1e9).all()
    assert np.array_equal(finalize_logits("scrub", logits, index), logits)


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
