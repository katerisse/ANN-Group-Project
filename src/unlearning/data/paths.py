"""Resolve ImageNet paths relative to a configurable root directory."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_ROOT = Path.home() / "datasets" / "imagenet"


def get_root(cli_root: str | Path | None = None) -> Path:
    value = cli_root or os.environ.get("IMAGENET_ROOT", DEFAULT_ROOT)
    return Path(value).expanduser().resolve()


def relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()
