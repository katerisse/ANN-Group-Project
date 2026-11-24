import keras
import numpy as np
import tensorflow as tf

from unlearning.classes import Group, load_class_index
from unlearning.probes import evaluate_probes, extract_features, fit_probe, score_probe

INDEX = load_class_index()


def _model():
    return keras.Sequential(
        [
            keras.Input((4, 4, 3)),
            keras.layers.GlobalAveragePooling2D(name="avg_pool"),
            keras.layers.Dense(1000, name="predictions"),
        ]
    )


def _dataset(seed: int):
    rng = np.random.default_rng(seed)
    images = rng.normal(size=(12, 4, 4, 3)).astype(np.float32)
    labels = np.array([150, 151, 268, 269, 0, 1, 2, 3, 0, 1, 2, 3])
    return tf.data.Dataset.from_tensor_slices((images, labels)).batch(4)


def test_extract_features_uses_the_named_penultimate_layer():
    features, labels = extract_features(_model(), _dataset(0))

    assert features.shape == (12, 3)
    assert labels.shape == (12,)
    assert np.array_equal(labels[:4], [150, 151, 268, 269])


def test_probe_on_separable_features_has_perfect_auc():
    rng = np.random.default_rng(0)
    positives = rng.normal(-3, 0.2, size=(50, 2))
    negatives = rng.normal(3, 0.2, size=(50, 2))
    features = np.vstack([positives, negatives])
    labels = np.array([150] * 50 + [0] * 50)

    dog = INDEX.indices(Group.DOG)
    probe = fit_probe(features, labels, dog, seed=1)
    scores = score_probe(probe, features, labels, dog)

    assert scores["auc"] == 1.0
    assert scores["accuracy"] == 1.0


def test_probe_fit_is_reproducible_for_a_seed():
    rng = np.random.default_rng(2)
    features = rng.normal(size=(40, 3))
    labels = np.array([150 if i % 2 else 0 for i in range(40)])
    dog = (150, 151)

    first = fit_probe(features, labels, dog, seed=5).coef_
    second = fit_probe(features, labels, dog, seed=5).coef_
    np.testing.assert_array_equal(first, second)


def test_evaluate_probes_reports_dog_and_canid_auc():
    results = evaluate_probes(
        _model(), _dataset(1), _dataset(2), INDEX, max_samples=20
    )

    assert set(results) == {
        "dog_probe_auc",
        "dog_probe_accuracy",
        "canid_probe_auc",
        "canid_probe_accuracy",
    }
    assert all(0.0 <= value <= 1.0 for value in results.values())
