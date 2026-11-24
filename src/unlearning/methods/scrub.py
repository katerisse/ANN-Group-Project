"""SCRUB: reverse teacher-student KL on forget, KL + CE on retain."""

from __future__ import annotations

from unlearning.methods.base import kl_divergence, sparse_ce


def loss(
    model,
    forget_batch,
    retain_batch,
    teacher,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    **kwargs,
):
    forget_images, _ = forget_batch
    retain_images, retain_labels = retain_batch

    student_forget = model(forget_images, training=True)
    student_retain = model(retain_images, training=True)
    teacher_forget = teacher(forget_images, training=False)
    teacher_retain = teacher(retain_images, training=False)

    return (
        -alpha * kl_divergence(teacher_forget, student_forget)
        + beta * kl_divergence(teacher_retain, student_retain)
        + gamma * sparse_ce(student_retain, retain_labels)
    )
