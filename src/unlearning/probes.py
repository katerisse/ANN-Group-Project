"""Linear probes on penultimate features for dog and canid recognition."""

from __future__ import annotations

import keras
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from unlearning.classes import Group


def extract_features(model, dataset, layer_name: str = "avg_pool") -> tuple[np.ndarray, np.ndarray]:
    """Return ``(features, labels)`` from a dataset using one internal layer."""
    inputs = model.inputs
    if isinstance(inputs, (list, tuple)):
        inputs = inputs[0]
    extractor = keras.Model(inputs, model.get_layer(layer_name).output)
    features, labels = [], []
    for images, batch_labels in dataset:
        features.append(extractor(images, training=False).numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(features), np.concatenate(labels).astype(int)


def fit_probe(
    features: np.ndarray,
    labels: np.ndarray,
    positive: tuple[int, ...] | list[int],
    seed: int = 0,
    max_samples: int = 4000,
):
    """Fit a class-balanced logistic-regression probe."""
    positive = np.asarray(positive, dtype=int)
    is_positive = np.isin(labels, positive)
    pos = np.flatnonzero(is_positive)
    neg = np.flatnonzero(~is_positive)
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("probe needs both positive and negative samples")

    count = min(len(pos), len(neg), max_samples // 2)
    rng = np.random.default_rng(seed)
    chosen = np.concatenate(
        [rng.choice(pos, count, replace=False), rng.choice(neg, count, replace=False)]
    )
    probe = LogisticRegression(max_iter=1000, random_state=seed)
    probe.fit(features[chosen], is_positive[chosen].astype(int))
    return probe


def score_probe(probe, features: np.ndarray, labels: np.ndarray, positive) -> dict[str, float]:
    """Score a fitted probe on features it was not trained on."""
    target = np.isin(labels, np.asarray(positive, dtype=int)).astype(int)
    if len(np.unique(target)) < 2:
        return {"auc": 0.0, "accuracy": float(np.mean(target == 0))}
    probability = probe.predict_proba(features)[:, 1]
    return {
        "auc": float(roc_auc_score(target, probability)),
        "accuracy": float(accuracy_score(target, probability >= 0.5)),
    }


def evaluate_probes(
    model,
    train_ds,
    test_ds,
    index,
    *,
    layer_name: str = "avg_pool",
    seed: int = 0,
    max_samples: int = 4000,
) -> dict[str, float]:
    """Train on train-split features, evaluate on frozen test-split features."""
    train_features, train_labels = extract_features(model, train_ds, layer_name)
    test_features, test_labels = extract_features(model, test_ds, layer_name)

    results: dict[str, float] = {}
    for name, group in (("dog", Group.DOG), ("canid", Group.CANID)):
        indices = index.indices(group)
        probe = fit_probe(train_features, train_labels, indices, seed, max_samples)
        scores = score_probe(probe, test_features, test_labels, indices)
        results[f"{name}_probe_auc"] = scores["auc"]
        results[f"{name}_probe_accuracy"] = scores["accuracy"]
    return results
