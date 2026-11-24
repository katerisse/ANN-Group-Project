import keras
import numpy as np
import tensorflow as tf

from unlearning.classes import load_class_index
from unlearning.relearn import relearn_curve, run_relearning, select_relearn_rows, summarize
from unlearning.splits import ManifestRow

INDEX = load_class_index()


def _row(path: str, class_index: int) -> ManifestRow:
    return ManifestRow(
        split="relearn",
        source_split="flat",
        relative_path=path,
        class_index_1b=class_index + 1,
        class_index_0b=class_index,
        synset="n00000000",
        name="test",
        group="dog",
    )


def test_select_relearn_rows_caps_each_dog_class():
    rows = [_row(f"dog/{i}.jpg", 150) for i in range(5)]
    rows += [_row(f"other/{i}.jpg", 0) for i in range(5)]

    selected = select_relearn_rows(rows, 2, INDEX, seed=3)

    assert len(selected) == 2
    assert all(row.class_index_0b == 150 for row in selected)
    assert selected == select_relearn_rows(rows, 2, INDEX, seed=3)


def test_summarize_recovery_and_steps_to_half():
    result = summarize([0.0, 0.4, 0.6], original_top1=0.8)

    assert np.isclose(result["recovery_fraction"], 0.75)
    assert result["steps_to_50pct_original"] == 1
    assert result["dog_top1_end"] == 0.6


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
    labels = np.full((8,), 150, dtype=np.int32)
    return tf.data.Dataset.from_tensor_slices((images, labels)).batch(4)


def test_relearn_curve_records_one_value_per_epoch():
    curve = relearn_curve(
        _model(), _dataset(0), _dataset(1), INDEX, epochs=2, lr=0.01
    )

    assert len(curve) == 3
    assert all(0.0 <= value <= 1.0 for value in curve)


def test_run_relearning_uses_each_budget_from_the_same_model():
    model = _model()
    before = model.get_weights()[0].copy()
    rows = [_row(f"dog/{i}.jpg", 150) for i in range(25)]

    results = run_relearning(
        model,
        rows,
        INDEX,
        lambda selected: _dataset(0),
        _dataset(1),
        original_top1=0.8,
        budgets=(1, 5),
        epochs=1,
    )

    assert set(results) == {
        "budget_1",
        "budget_1_curve",
        "budget_5",
        "budget_5_curve",
    }
    assert len(results["budget_1_curve"]) == 2
    np.testing.assert_array_equal(model.get_weights()[0], before)
