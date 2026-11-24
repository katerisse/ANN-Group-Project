"""Brief relearning on the disjoint dog holdout after unlearning."""

from __future__ import annotations

import random
from collections import defaultdict

import keras
import numpy as np
import tensorflow as tf

from unlearning.classes import Group
from unlearning.methods.base import sparse_ce
from unlearning.metrics import evaluate
from unlearning.scope import set_trainable_scope
from unlearning.splits import ManifestRow

BUDGETS = (1, 5, 20)


def select_relearn_rows(
    rows: list[ManifestRow], per_class: int, index, seed: int = 0
) -> list[ManifestRow]:
    """Pick up to ``per_class`` disjoint relearn images per dog class."""
    by_class: dict[int, list[ManifestRow]] = defaultdict(list)
    for row in rows:
        if row.split != "relearn" or index.group(row.class_index_0b) != Group.DOG:
            continue
        by_class[row.class_index_0b].append(row)

    rng = random.Random(seed)
    selected: list[ManifestRow] = []
    for class_index in sorted(by_class):
        candidates = sorted(by_class[class_index], key=lambda row: row.relative_path)
        selected.extend(rng.sample(candidates, min(per_class, len(candidates))))
    return selected


def dog_top1(model, test_ds, index) -> float:
    """Dog accuracy on a frozen test dataset."""
    logits, labels = [], []
    for images, batch_labels in test_ds:
        logits.append(model(images, training=False).numpy())
        labels.append(batch_labels.numpy())
    if not labels:
        return 0.0
    return float(evaluate(np.concatenate(logits), np.concatenate(labels), index)["forget_top1"])


def relearn_curve(
    model, train_ds, test_ds, index, *, epochs: int = 3, lr: float = 1e-4
) -> list[float]:
    """Fine-tune `conv5 + head`, recording dog-test top-1 after each epoch."""
    set_trainable_scope(model, "conv5_head")
    optimizer = keras.optimizers.AdamW(learning_rate=lr)
    variables = [
        variable
        for layer in model.layers
        if layer.trainable
        for variable in layer.trainable_variables
    ]

    curve = [dog_top1(model, test_ds, index)]
    for _ in range(epochs):
        for images, labels in train_ds:
            with tf.GradientTape() as tape:
                loss = sparse_ce(model(images, training=True), labels)
            gradients = tape.gradient(loss, variables)
            pairs = [
                (gradient, variable)
                for gradient, variable in zip(gradients, variables)
                if gradient is not None
            ]
            optimizer.apply_gradients(pairs)
        curve.append(dog_top1(model, test_ds, index))
    return curve


def summarize(curve: list[float], original_top1: float) -> dict[str, float]:
    """Recovery fraction and first epoch reaching half the original score."""
    start, end = curve[0], curve[-1]
    headroom = original_top1 - start
    recovery = (end - start) / headroom if headroom > 1e-6 else 0.0
    target = 0.5 * original_top1
    steps = next((step for step, score in enumerate(curve) if score >= target), None)
    return {
        "dog_top1_start": start,
        "dog_top1_end": end,
        "recovery_fraction": float(recovery),
        "steps_to_50pct_original": -1 if steps is None else float(steps),
    }


def run_relearning(
    model,
    rows: list[ManifestRow],
    index,
    dataset_factory,
    test_ds,
    *,
    original_top1: float,
    budgets=BUDGETS,
    epochs: int = 3,
    seed: int = 0,
) -> dict:
    """Run every budget from the same unlearned weights."""
    initial_weights = model.get_weights()
    results: dict = {}
    for budget in budgets:
        model.set_weights(initial_weights)
        selected = select_relearn_rows(rows, budget, index, seed)
        curve = relearn_curve(model, dataset_factory(selected), test_ds, index, epochs=epochs)
        results[f"budget_{budget}"] = summarize(curve, original_top1)
        results[f"budget_{budget}_curve"] = curve
    model.set_weights(initial_weights)
    return results
