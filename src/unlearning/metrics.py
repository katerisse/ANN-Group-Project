"""Evaluation metrics for dog unlearning on the frozen val-test split."""

from __future__ import annotations

from collections import Counter

import numpy as np

from unlearning.classes import ClassIndex, Group


def probabilities(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def top1(logits: np.ndarray) -> np.ndarray:
    return np.asarray(logits).argmax(axis=-1)


def top5(logits: np.ndarray) -> np.ndarray:
    return np.asarray(logits).argsort(axis=-1)[:, -5:]


def _accuracy(predicted: np.ndarray, labels: np.ndarray) -> float:
    return _safe_mean((predicted == labels).astype(float))


def _top5_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    return _safe_mean((top5(logits) == np.asarray(labels)[:, None]).any(axis=-1))


def _group_mask(labels: np.ndarray, index: ClassIndex, group: Group) -> np.ndarray:
    return np.isin(labels, index.indices(group))


def evaluate(
    logits: np.ndarray,
    labels: np.ndarray,
    index: ClassIndex,
    teacher_logits: np.ndarray | None = None,
) -> dict[str, float | dict[str, int]]:
    """Compute the Phase 1 metrics from one batch of logits and labels."""
    labels = np.asarray(labels, dtype=int)
    probs = probabilities(logits)
    dog = _group_mask(labels, index, Group.DOG)
    canid = _group_mask(labels, index, Group.CANID)
    retain = ~dog

    dog_logits, dog_labels = logits[dog], labels[dog]
    retain_logits, retain_labels = logits[retain], labels[retain]
    canid_logits, canid_labels = logits[canid], labels[canid]

    dog_indices = index.indices(Group.DOG)
    canid_indices = index.indices(Group.CANID)

    metrics: dict[str, float | dict[str, int]] = {
        "forget_top1": _accuracy(top1(dog_logits), dog_labels) if dog.any() else 0.0,
        "forget_top5": _top5_accuracy(dog_logits, dog_labels) if dog.any() else 0.0,
        "any_dog_top1": (
            _safe_mean(np.isin(top1(dog_logits), dog_indices).astype(float))
            if dog.any()
            else 0.0
        ),
        "dog_mass": _safe_mean(probs[dog][:, dog_indices].sum(axis=-1)),
        "retain_micro_top1": (
            _accuracy(top1(retain_logits), retain_labels) if retain.any() else 0.0
        ),
        "retain_micro_top5": (
            _top5_accuracy(retain_logits, retain_labels) if retain.any() else 0.0
        ),
        "canid_top1": (
            _accuracy(top1(canid_logits), canid_labels) if canid.any() else 0.0
        ),
        "canid_top5": (
            _top5_accuracy(canid_logits, canid_labels) if canid.any() else 0.0
        ),
        "canid_mass": _safe_mean(probs[canid][:, canid_indices].sum(axis=-1)),
        "entropy": _safe_mean(-(probs * np.log(probs + 1e-12)).sum(axis=-1)),
        "max_softmax": _safe_mean(probs.max(axis=-1)),
    }

    per_class = {
        int(cls): _accuracy(top1(retain_logits[retain_labels == cls]), retain_labels[retain_labels == cls])
        for cls in np.unique(retain_labels)
    }
    metrics["retain_macro_recall"] = _safe_mean(np.array(list(per_class.values())))
    metrics["worst_class_retain_recall"] = (
        float(min(per_class.values())) if per_class else 0.0
    )

    if dog.any():
        confusions = Counter(
            index.get(int(cls)).name for cls in top1(dog_logits)
        )
        metrics["top_dog_confusions"] = dict(confusions.most_common(10))

    if teacher_logits is not None and dog.any():
        teacher_top5 = top5(teacher_logits[dog])
        student_top1 = top1(dog_logits)
        metrics["plausibility_rate"] = _safe_mean(
            np.array(
                [pred in allowed for pred, allowed in zip(student_top1, teacher_top5)]
            ).astype(float)
        )

    return metrics
