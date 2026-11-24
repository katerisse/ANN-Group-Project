"""Config-driven training loop shared by all Phase 1 unlearning methods."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from unlearning.classes import Group
from unlearning.methods import METHODS, mask
from unlearning.methods.base import sparse_ce
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
    patience: int = 0
    min_delta: float = 0.0


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


def run_name(config: TrainConfig) -> str:
    """Stable checkpoint stem: method, seed, config hash and git SHA."""
    payload = json.dumps(asdict(config), sort_keys=True)
    config_hash = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return f"{config.method}_s{config.seed}_{config_hash}_{_git_sha()[:8]}"


def _learning_rate(optimizer) -> float:
    value = optimizer.learning_rate
    return float(value.numpy() if hasattr(value, "numpy") else value)


def _cross_entropy(logits, labels) -> float:
    if len(labels) == 0:
        return 0.0
    return float(keras.ops.convert_to_numpy(sparse_ce(logits, labels)))


def _write_csv(log: list[dict], path: Path) -> None:
    if not log:
        return
    fields = list(log[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in log:
            writer.writerow(
                {key: value for key, value in row.items() if not isinstance(value, dict)}
            )


def _save(model, config: TrainConfig, log: list[dict], out_dir: Path) -> None:
    name = run_name(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / f"final_{name}.keras")
    if config.save_checkpoints:
        model.save(out_dir / f"last_{name}.keras")
    metadata = {
        "method": config.method,
        "seed": config.seed,
        "run_name": name,
        "config": asdict(config),
        "git_sha": _git_sha(),
        "log": log,
    }
    (out_dir / f"run_metadata_{name}.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    _write_csv(log, out_dir / f"metrics_{name}.csv")


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
    set_trainable_scope(model, config.scope or spec.scope)

    teacher_tune = collect_logits(teacher, tune_ds)[0] if teacher is not None else None
    dog_indices = index.indices(Group.DOG)

    def evaluate_tune(raw_logits, step, objective=0.0, lr=0.0) -> dict:
        logits = (
            mask.mask_logits(raw_logits, dog_indices)
            if config.method == "mask"
            else raw_logits
        )
        labels = tune_labels
        row = evaluate(logits, labels, index, teacher_logits=teacher_tune)
        dog = np.isin(labels, dog_indices)
        row.update(
            step=step,
            objective=float(objective),
            forget_loss=_cross_entropy(raw_logits[dog], labels[dog]),
            retain_loss=_cross_entropy(raw_logits[~dog], labels[~dog]),
            lr=float(lr),
        )
        return row

    tune_logits, tune_labels = collect_logits(model, tune_ds)

    if spec.loss is None:
        log = [evaluate_tune(tune_logits, 0)]
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
        # One batch from each stream per step: balanced forget/retain composition.
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

    log: list[dict] = []
    best_score = -np.inf
    best_weights = None
    stale = 0
    name = run_name(config)

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
            current_logits, _ = collect_logits(model, tune_ds)
            row = evaluate_tune(current_logits, step, loss, _learning_rate(optimizer))
            log.append(row)
            score = row["retain_micro_top1"]
            if score > best_score + config.min_delta:
                best_score, stale = score, 0
                best_weights = model.get_weights()
                if config.save_checkpoints and out_dir is not None:
                    model.save(out_dir / f"best_retain_{name}.keras")
            elif config.patience:
                stale += 1
                if stale >= config.patience:
                    if best_weights is not None:
                        model.set_weights(best_weights)
                    break

    if out_dir is not None:
        _save(model, config, log, out_dir)
    return model, log
