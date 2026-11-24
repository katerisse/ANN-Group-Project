"""BatchNorm policy shared by all Phase 1 unlearning methods."""

from __future__ import annotations

import keras
import numpy as np


@keras.saving.register_keras_serializable(package="unlearning")
class FrozenStatsBatchNormalization(keras.layers.BatchNormalization):
    """BatchNorm whose moving statistics never update during training.

    ``gamma`` and ``beta`` stay trainable; only the statistics are frozen.
    """

    def call(self, inputs, training=None, **kwargs):
        return super().call(inputs, training=False, **kwargs)


def iter_layers(model):
    """Yield every layer, including layers inside nested models."""
    for layer in model.layers:
        yield layer
        if isinstance(layer, keras.Model):
            yield from iter_layers(layer)


def iter_batch_norms(model):
    return (
        layer
        for layer in iter_layers(model)
        if isinstance(layer, keras.layers.BatchNormalization)
    )


def freeze_batch_norm(model) -> int:
    """Freeze all BatchNorm layers and return how many were frozen."""
    count = 0
    for layer in iter_batch_norms(model):
        layer.trainable = False
        count += 1
    return count


def bn_snapshot(model) -> dict[str, np.ndarray]:
    """Copy every BatchNorm weight for before/after comparisons."""
    return {
        f"{layer.name}.{weight.name}": weight.numpy().copy()
        for layer in iter_batch_norms(model)
        for weight in layer.weights
    }


def max_bn_delta(before: dict[str, np.ndarray], model) -> float:
    """Largest absolute change in any BatchNorm weight since ``before``."""
    deltas = []
    for layer in iter_batch_norms(model):
        for weight in layer.weights:
            key = f"{layer.name}.{weight.name}"
            deltas.append(float(np.max(np.abs(weight.numpy() - before[key]))))
    return max(deltas, default=0.0)


def replace_batch_norms(model) -> keras.Model:
    """Clone a model with gamma/beta-only BatchNorm in place of normal BN."""

    def clone_layer(layer):
        if isinstance(layer, keras.layers.BatchNormalization):
            return FrozenStatsBatchNormalization.from_config(layer.get_config())
        return layer.__class__.from_config(layer.get_config())

    clone = keras.models.clone_model(model, clone_function=clone_layer)
    for source, target in zip(iter_layers(model), iter_layers(clone)):
        weights = source.get_weights()
        if weights and len(weights) == len(target.get_weights()):
            target.set_weights(weights)
    return clone
