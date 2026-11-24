import json

import keras
import numpy as np
import tensorflow as tf

from unlearning.classes import load_class_index
from unlearning.train import TrainConfig, collect_logits, train_method


def _model():
    return keras.Sequential(
        [
            keras.Input((3,)),
            keras.layers.Dense(8, name="conv5_dense"),
            keras.layers.Dense(1000, name="predictions"),
        ]
    )


def _dataset(seed: int):
    rng = np.random.default_rng(seed)
    images = rng.normal(size=(8, 3)).astype(np.float32)
    labels = np.array([0, 0, 150, 150, 268, 268, 1, 1])
    return tf.data.Dataset.from_tensor_slices((images, labels)).batch(2)


def test_collect_logits_stacks_batches():
    logits, labels = collect_logits(_model(), _dataset(0))

    assert logits.shape == (8, 1000)
    assert labels.shape == (8,)


def test_retain_only_trains_for_a_fixed_step_budget():
    model = _model()
    before = model.get_weights()[0].copy()

    trained, log = train_method(
        model,
        forget_ds=_dataset(1),
        retain_ds=_dataset(2),
        tune_ds=_dataset(3),
        config=TrainConfig(method="retain_only", steps=2, eval_every=1, lr=0.01),
        index=load_class_index(),
    )

    assert len(log) == 2
    assert log[0]["step"] == 1
    assert not np.allclose(trained.get_weights()[0], before)


def test_original_method_does_not_change_weights():
    model = _model()
    before = model.get_weights()[0].copy()

    trained, log = train_method(
        model,
        forget_ds=_dataset(1),
        retain_ds=_dataset(2),
        tune_ds=_dataset(3),
        config=TrainConfig(method="original", steps=2),
        index=load_class_index(),
    )

    assert len(log) == 1
    assert log[0]["step"] == 0
    np.testing.assert_array_equal(trained.get_weights()[0], before)


def test_mask_baseline_removes_all_dog_predictions():
    _, log = train_method(
        _model(),
        forget_ds=_dataset(1),
        retain_ds=_dataset(2),
        tune_ds=_dataset(3),
        config=TrainConfig(method="mask"),
        index=load_class_index(),
    )

    assert log[0]["any_dog_top1"] == 0.0


def test_early_stopping_respects_patience():
    _, log = train_method(
        _model(),
        forget_ds=_dataset(1),
        retain_ds=_dataset(2),
        tune_ds=_dataset(3),
        config=TrainConfig(
            method="retain_only",
            steps=10,
            eval_every=1,
            patience=1,
            min_delta=1.0,
            lr=0.01,
        ),
        index=load_class_index(),
    )

    assert len(log) == 2
    assert log[-1]["step"] < 10


def test_run_metadata_and_curve_csv_are_written(tmp_path):
    out_dir = tmp_path / "runs"
    train_method(
        _model(),
        forget_ds=_dataset(1),
        retain_ds=_dataset(2),
        tune_ds=_dataset(3),
        config=TrainConfig(method="retain_only", steps=1, eval_every=1),
        index=load_class_index(),
        out_dir=out_dir,
    )

    metadata_path = next(out_dir.glob("run_metadata_*.json"))
    metadata = json.loads(metadata_path.read_text())
    assert metadata["method"] == "retain_only"
    assert metadata["run_name"].startswith("retain_only_s2026_")

    curve_path = next(out_dir.glob("metrics_*.csv"))
    header = curve_path.read_text().splitlines()[0]
    for field in ("step", "objective", "forget_loss", "retain_loss", "lr", "dog_mass"):
        assert field in header
