"""Retain-only cross-entropy on conv5 + head."""

from __future__ import annotations

from unlearning.methods.base import sparse_ce


def loss(model, retain_batch, **kwargs):
    images, labels = retain_batch
    return sparse_ce(model(images, training=True), labels)
