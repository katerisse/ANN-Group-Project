"""Manifest-driven tf.data pipelines with a test-split access guard."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import tensorflow as tf

from unlearning.data.preprocess import ResNet50Preprocessor
from unlearning.splits import Split, load_manifest


class Purpose(str, Enum):
    TRAIN = "train"
    TUNE = "tune"
    EVAL = "eval"


ALLOWED_SPLITS = {
    Purpose.TRAIN: {
        Split.TRAIN_FORGET.value,
        Split.TRAIN_RETAIN.value,
        Split.RELEARN.value,
    },
    Purpose.TUNE: {Split.VAL_TUNE.value},
    Purpose.EVAL: {Split.VAL_TEST.value},
}


class TestSplitLeakageError(ValueError):
    """Raised when the held-out val-test split is requested for tuning/training."""

    __test__ = False  # Not a pytest test class despite the name.


def build_dataset(
    manifest_path: Path,
    root: Path,
    split: Split | str,
    purpose: Purpose | str,
    *,
    preprocessor: ResNet50Preprocessor | None = None,
    batch_size: int = 32,
    shuffle: bool = False,
    seed: int = 0,
    group: str | None = None,
) -> tf.data.Dataset:
    purpose = Purpose(purpose)
    split_value = split.value if isinstance(split, Split) else split
    if split_value not in ALLOWED_SPLITS[purpose]:
        raise TestSplitLeakageError(
            f"{purpose.value} cannot use split {split_value}; "
            f"allowed: {sorted(ALLOWED_SPLITS[purpose])}"
        )

    rows = [
        row
        for row in load_manifest(manifest_path)
        if row.split == split_value and (group is None or row.group == group)
    ]
    if not rows:
        raise ValueError(f"no manifest rows for split={split_value}, group={group}")

    paths = [str(root / row.relative_path) for row in rows]
    labels = [row.class_index_0b for row in rows]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(rows), seed=seed, reshuffle_each_iteration=False
        )

    processor = preprocessor or ResNet50Preprocessor()

    def load_one(path, label):
        image = processor.decode(path)
        if purpose == Purpose.TRAIN:
            image = processor.train(image, seed=(seed, seed + 1))
        else:
            image = processor.eval(image)
        return image, label

    return (
        dataset.map(
            load_one,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def build_train_dataset(**kwargs) -> tf.data.Dataset:
    kwargs.setdefault("shuffle", True)
    return build_dataset(purpose=Purpose.TRAIN, **kwargs)


def build_tuning_dataset(**kwargs) -> tf.data.Dataset:
    return build_dataset(purpose=Purpose.TUNE, **kwargs)


def build_eval_dataset(**kwargs) -> tf.data.Dataset:
    return build_dataset(purpose=Purpose.EVAL, **kwargs)
