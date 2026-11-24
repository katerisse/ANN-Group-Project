"""Membership-inference check for forget and retain examples.

The score is the model's true-class probability (confidence). Higher scores for
train members than for held-out test examples indicate memorisation; a strong
unlearning method should reduce the forget-set gap toward zero while leaving
the retain control unchanged.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def confidence_scores(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """True-class softmax probability for each example."""
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
    rows = np.arange(len(values))
    return probabilities[rows, np.asarray(labels, dtype=int)]


def score_dataset(model, dataset) -> np.ndarray:
    """Collect confidence scores for one labelled dataset."""
    scores, labels = [], []
    for images, batch_labels in dataset:
        logits = model(images, training=False).numpy()
        scores.append(confidence_scores(logits, batch_labels.numpy()))
        labels.append(batch_labels.numpy())
    return np.concatenate(scores)


def evaluate_membership(train_scores: np.ndarray, test_scores: np.ndarray) -> dict[str, float]:
    """AUC (members=1) and the mean train-minus-test score gap."""
    train_scores = np.asarray(train_scores, dtype=float)
    test_scores = np.asarray(test_scores, dtype=float)
    if len(train_scores) == 0 or len(test_scores) == 0:
        return {"auc": 0.5, "gap": 0.0}
    members = np.concatenate(
        [np.ones(len(train_scores)), np.zeros(len(test_scores))]
    )
    scores = np.concatenate([train_scores, test_scores])
    return {
        "auc": float(roc_auc_score(members, scores)),
        "gap": float(train_scores.mean() - test_scores.mean()),
    }


def run_mia(
    model,
    forget_train_ds,
    forget_test_ds,
    retain_train_ds,
    retain_test_ds,
) -> dict[str, float]:
    """Score the forget set and the retain-set control."""
    forget = evaluate_membership(
        score_dataset(model, forget_train_ds), score_dataset(model, forget_test_ds)
    )
    retain = evaluate_membership(
        score_dataset(model, retain_train_ds), score_dataset(model, retain_test_ds)
    )
    return {
        "forget_mia_auc": forget["auc"],
        "forget_mia_gap": forget["gap"],
        "retain_mia_auc": retain["auc"],
        "retain_mia_gap": retain["gap"],
    }
