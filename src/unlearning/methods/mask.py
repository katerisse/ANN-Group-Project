"""Inference-only dog-logit mask (sanity baseline, no training)."""

from __future__ import annotations

import numpy as np


def mask_logits(logits: np.ndarray, dog_indices) -> np.ndarray:
    """Replace dog logits with a very small value so they cannot win."""
    output = np.array(logits, copy=True)
    output[..., np.asarray(dog_indices, dtype=int)] = -1e9
    return output
