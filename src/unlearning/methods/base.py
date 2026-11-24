"""Shared loss helpers for the Phase 1 unlearning methods."""

from __future__ import annotations

import keras


def sparse_ce(logits, labels):
    """Mean sparse cross-entropy from logits."""
    return keras.ops.mean(
        keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    )


def kl_divergence(teacher_logits, student_logits):
    """Mean KL(teacher || student), with logits on the last axis."""
    teacher = keras.ops.softmax(teacher_logits, axis=-1)
    log_teacher = keras.ops.log_softmax(teacher_logits, axis=-1)
    log_student = keras.ops.log_softmax(student_logits, axis=-1)
    return keras.ops.mean(keras.ops.sum(teacher * (log_teacher - log_student), axis=-1))
