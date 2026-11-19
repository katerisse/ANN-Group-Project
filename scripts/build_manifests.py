"""Build manifests/phase1_splits.csv from a local ImageNet copy."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from unlearning.classes import default_class_index_path, load_class_index
from unlearning.config import DataConfig
from unlearning.data.paths import get_root
from unlearning.splits import build_rows, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--index", type=Path, default=default_class_index_path())
    parser.add_argument(
        "--output", type=Path, default=Path("manifests/phase1_splits.csv")
    )
    args = parser.parse_args()

    root = get_root(args.root)
    index = load_class_index(args.index)
    config = DataConfig(seed=args.seed)
    rows = build_rows(root, index, config)
    digest = write_manifest(rows, args.output, args.index, config)

    for split, count in sorted(Counter(row.split for row in rows).items()):
        print(f"{split}: {count}")
    print(f"manifest sha256: {digest}")


if __name__ == "__main__":
    main()
