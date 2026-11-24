"""Config-driven training loop shared by all Phase 1 methods."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from unlearning.methods import METHODS
from unlearning.metrics import evaluate
from unlearning.scope import set_trainable_scope
from unlearning.seeding import set_seed


@dataclass(frozen=True)
class TrainConfig:
    method: str
    seed: int = 2026
    steps: int = 100
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    lambda_retain: float = 1.0
    scrub_alpha: float = 1.0
    scrub_beta: float = 1.0
    eval_every: int = 25
    scope: str | None = None
    save_checkpoints: bool = False


def collect_logits(model, dataset) -> tuple[np.ndarray, np.ndarray]:
    """Run a dataset once and stack logits and labels."""
    logits, labels = [], []
    for images, batch_labels in dataset:
        logits.append(model(images, training=False).numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _save(model, config: TrainConfig, log: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "final.keras")
    if config.save_checkpoints and log:
        best = max(log, key=lambda row: row["retain_micro_top1"])
        model.save(out_dir / "best_retain.keras")
        model.save(out_dir / "last.keras")
        _ = best
    config_json = json.dumps(asdict(config), sort_keys=True)
    metadata = {
        "method": config.method,
        "seed": config.seed,
        "config": asdict(config),
        "config_sha256": hashlib.sha256(config_json.encode()).hexdigest()[:12],
        "git_sha": _git_sha(),
        "log": log,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def train_method(
    model,
    forget_ds,
    retain_ds,
    tune_ds,
    config: TrainConfig,
    index,
    teacher=None,
    out_dir: Path | None = None,
):
    """Train one method for a fixed step budget and return (model, log)."""
    spec = METHODS[config.method]
    set_seed(config.seed)
    scope = config.scope or spec.scope
    set_trainable_scope(model, scope)

    if spec.loss is None:
        log = []
        if out_dir is not None:
            _save(model, config, log, out_dir)
        return model, log

    optimizer = keras.optimizers.AdamW(
        learning_rate=config.lr, weight_decay=config.weight_decay
    )
    variables = [
        variable
        for layer in model.layers
        if layer.trainable
        for variable in layer.trainable_variables
    ]

    retain_iter = iter(retain_ds.repeat())
    forget_iter = iter(forget_ds.repeat()) if spec.uses_forget else None

    def build_loss():
        retain_batch = next(retain_iter)
        if not spec.uses_forget:
            return spec.loss(model, retain_batch)
        forget_batch = next(forget_iter)
        kwargs = {"lambda_retain": config.lambda_retain}
        if spec.uses_teacher:
            kwargs.update(
                teacher=teacher,
                alpha=config.scrub_alpha,
                beta=config.scrub_beta,
                gamma=config.lambda_retain,
            )
        return spec.loss(model, forget_batch, retain_batch, **kwargs)

    log = []
    for step in range(1, config.steps + 1):
        with tf.GradientTape() as tape:
            loss = build_loss()
        gradients = tape.gradient(loss, variables)
        pairs = [
            (gradient, variable)
            for gradient, variable in zip(gradients, variables)
            if gradient is not None
        ]
        clipped, _ = tf.clip_by_global_norm(
            [pair[0] for pair in pairs], config.grad_clip
        )
        optimizer.apply_gradients(zip(clipped, [pair[1] for pair in pairs]))

        if step == 1 or step % config.eval_every == 0:
            logits, labels = collect_logits(model, tune_ds)
            row = evaluate(logits, labels, index)
            row.update(step=step, loss=float(loss.numpy()))
            log.append(row)

    if out_dir is not None:
        _save(model, config, log, out_dir)
    return model, log
