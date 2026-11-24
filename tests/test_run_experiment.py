import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from unlearning.classes import Group, load_class_index
from unlearning.data.tfdata import Purpose
from unlearning.splits import Split

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import run_experiment  # noqa: E402
from run_experiment import build_datasets, eval_split, finalize_logits  # noqa: E402


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
    for flag in (
        "--config",
        "--steps",
        "--max-batches",
        "--methods",
        "--seeds",
        "--tune-only",
    ):
        assert flag in result.stdout


def _record_split_requests(monkeypatch):
    requests = []

    class FakeDataset:
        def concatenate(self, other):
            return self

    def fake_dataset(manifest, root, split, purpose, batch_size, max_batches, group=None):
        requests.append((split, purpose, group))
        return FakeDataset()

    monkeypatch.setattr(run_experiment, "_dataset", fake_dataset)
    return requests


def test_tune_only_datasets_never_request_val_test(monkeypatch):
    requests = _record_split_requests(monkeypatch)

    datasets = build_datasets(Path("manifest.csv"), Path("/tmp"), 8, 0, True)

    assert set(datasets) == {"forget", "retain", "tune"}
    assert requests
    assert all(split != Split.VAL_TEST for split, _, _ in requests)
    assert all(purpose != Purpose.EVAL for _, purpose, _ in requests)


def test_full_datasets_request_val_test(monkeypatch):
    requests = _record_split_requests(monkeypatch)

    datasets = build_datasets(Path("manifest.csv"), Path("/tmp"), 8, 0, False)

    assert set(datasets) == {
        "forget",
        "retain",
        "tune",
        "test",
        "forget_test",
        "retain_test",
    }
    assert any(split == Split.VAL_TEST for split, _, _ in requests)
    assert any(purpose == Purpose.EVAL for _, purpose, _ in requests)


def test_eval_split_matches_mode():
    assert eval_split(True) == Split.VAL_TUNE.value
    assert eval_split(False) == Split.VAL_TEST.value
