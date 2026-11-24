"""Verify the pinned ResNet50 ImageNet weights file before use."""

from __future__ import annotations

import argparse
from pathlib import Path

from unlearning.models import default_weights_path, verify_weights_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=None)
    args = parser.parse_args()

    path = args.path or default_weights_path()
    try:
        checksum = verify_weights_file(path)
    except ValueError as error:
        raise SystemExit(f"ERROR: {error}") from error

    print(f"verified {path}")
    print(f"sha256: {checksum}")


if __name__ == "__main__":
    main()
