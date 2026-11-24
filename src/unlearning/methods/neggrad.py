"""NegGrad+: maximise forget CE while keeping retain CE low."""

from __future__ import annotations

from unlearning.methods.base import sparse_ce


def loss(model, forget_batch, retain_batch, lambda_retain: float = 1.0, **kwargs):
    forget_images, forget_labels = forget_batch
    retain_images, retain_labels = retain_batch
    forget_ce = sparse_ce(model(forget_images, training=True), forget_labels)
    retain_ce = sparse_ce(model(retain_images, training=True), retain_labels)
    return -forget_ce + lambda_retain * retain_ce
