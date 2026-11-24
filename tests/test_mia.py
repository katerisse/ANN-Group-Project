import keras
import numpy as np
import tensorflow as tf

from unlearning.mia import confidence_scores, evaluate_membership, run_mia, score_dataset


def test_confidence_scores_are_true_class_probabilities():
    logits = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    scores = confidence_scores(logits, np.array([0, 1]))

    np.testing.assert_allclose(scores, [np.exp(2) / (np.exp(2) + 2)] * 2)


def test_membership_auc_and_gap_on_separated_scores():
    result = evaluate_membership(
        np.array([0.9, 0.8, 0.7]), np.array([0.2, 0.3, 0.4])
    )

    assert result["auc"] == 1.0
    assert np.isclose(result["gap"], 0.5)


def test_empty_membership_scores_are_neutral():
    assert evaluate_membership(np.array([]), np.array([0.2])) == {
        "auc": 0.5,
        "gap": 0.0,
    }


def _model():
    return keras.Sequential(
        [keras.Input((3,)), keras.layers.Dense(1000, name="predictions")]
    )


def _dataset(seed: int):
    rng = np.random.default_rng(seed)
    images = rng.normal(size=(10, 3)).astype(np.float32)
    labels = rng.integers(0, 1000, size=(10,))
    return tf.data.Dataset.from_tensor_slices((images, labels)).batch(2)


def test_score_dataset_returns_one_confidence_per_example():
    scores = score_dataset(_model(), _dataset(0))

    assert scores.shape == (10,)
    assert ((scores >= 0) & (scores <= 1)).all()


def test_run_mia_reports_forget_and_retain_control():
    results = run_mia(_model(), _dataset(1), _dataset(2), _dataset(3), _dataset(4))

    assert set(results) == {
        "forget_mia_auc",
        "forget_mia_gap",
        "retain_mia_auc",
        "retain_mia_gap",
    }
    assert all(0.0 <= results[key] <= 1.0 for key in results if key.endswith("auc"))
    assert all(np.isfinite(value) for value in results.values())
