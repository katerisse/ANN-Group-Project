"""Trainable-scope policy shared by the Phase 1 unlearning methods."""

from __future__ import annotations

from unlearning.bn import freeze_batch_norm


def set_trainable_scope(model, scope: str) -> int:
    """Freeze everything except the requested conv5/head layers."""
    if scope == "none":
        trainable = set()
    elif scope == "head":
        trainable = {"predictions"}
    elif scope == "conv5_head":
        trainable = {
            layer.name
            for layer in model.layers
            if layer.name.startswith("conv5_")
            or layer.name in {"avg_pool", "predictions"}
        }
    else:
        raise ValueError(f"unknown trainable scope: {scope}")

    for layer in model.layers:
        layer.trainable = layer.name in trainable
    freeze_batch_norm(model)
    return sum(1 for layer in model.layers if layer.trainable)
