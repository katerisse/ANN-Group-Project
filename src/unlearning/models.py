"""Pinned Keras ImageNet ResNet50 and verification of its weights file."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import keras

WEIGHTS_URL = (
    "https://storage.googleapis.com/tensorflow/keras-applications/resnet/"
    "resnet50_weights_tf_dim_ordering_tf_kernels.h5"
)
WEIGHTS_FILENAME = "resnet50_weights_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_SIZE = 102_967_424
WEIGHTS_MD5 = "2cb95161c43110f7111970584f804107"
WEIGHTS_SHA256 = (
    "7011d39ea4f61f4ddb8da99c4addf3fae4209bfda7828adb4698b16283258fbe"
)
PARAM_COUNT = 25_636_712
BN_COUNT = 53


def default_weights_path() -> Path:
    """Path Keras uses for the cached ImageNet weights (no download)."""
    cache = Path(os.environ.get("KERAS_HOME", Path.home() / ".keras"))
    return cache / "models" / WEIGHTS_FILENAME


def verify_weights_file(path: Path) -> str:
    """Check size, MD5 and SHA256; raise ValueError on any mismatch."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"missing weights file: {path}")

    size = path.stat().st_size
    if size != WEIGHTS_SIZE:
        raise ValueError(f"weights size mismatch: {size} != {WEIGHTS_SIZE}")

    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            md5.update(chunk)
            sha256.update(chunk)

    if md5.hexdigest() != WEIGHTS_MD5:
        raise ValueError(f"weights MD5 mismatch: {md5.hexdigest()}")
    if sha256.hexdigest() != WEIGHTS_SHA256:
        raise ValueError(f"weights SHA256 mismatch: {sha256.hexdigest()}")
    return sha256.hexdigest()


def load_resnet50(*, verify: bool = True) -> keras.Model:
    """Load unmodified ResNet50 with 1000 logits and no activation."""
    path = default_weights_path()
    if verify and path.is_file():
        verify_weights_file(path)

    model = keras.applications.ResNet50(
        weights="imagenet",
        include_top=True,
        classes=1000,
        classifier_activation=None,
    )

    if verify:
        verify_weights_file(default_weights_path())
    return model


def softmax_logits(logits):
    """Convert logits to probabilities along the class axis."""
    return keras.ops.softmax(logits, axis=-1)
