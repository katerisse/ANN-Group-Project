"""Reproducible seeds and deterministic TensorFlow settings."""

from __future__ import annotations

import os

import keras
import tensorflow as tf

# PYTHONHASHSEED is read at interpreter startup, so set it in the shell too.
DETERMINISM_ENV = {
    "TF_DETERMINISTIC_OPS": "1",
    "TF_CUDNN_DETERMINISTIC": "1",
    "PYTHONHASHSEED": "0",
}

_determinism_enabled = False


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow, then enable op determinism."""
    for name, value in DETERMINISM_ENV.items():
        os.environ.setdefault(name, value)
    keras.utils.set_random_seed(seed)

    global _determinism_enabled
    if not _determinism_enabled:
        tf.config.experimental.enable_op_determinism()
        _determinism_enabled = True
