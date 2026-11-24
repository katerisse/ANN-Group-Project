"""Build manifests/phase1_local.csv from the on-device flat ImageNet copy."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from unlearning.classes import default_class_index_path, load_class_index
from unlearning.config import DataConfig
from unlearning.data.local_layout import LocalLayout
from unlearning.data.local_manifest import build_local_rows
from unlearning.data.paths import get_root
from unlearning.splits import write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-per-class", type=int, default=100)
    parser.add_argument("--relearn-per-class", type=int, default=20)
    parser.add_argument("--val-per-split-per-class", type=int, default=25)
    parser.add_argument("--index", type=Path, default=default_class_index_path())
    parser.add_argument(
        "--output", type=Path, default=Path("manifests/phase1_local.csv")
    )
    args = parser.parse_args()

    config = DataConfig(
        seed=args.seed,
        train_per_class=args.train_per_class,
        relearn_per_class=args.relearn_per_class,
        val_per_split_per_class=args.val_per_split_per_class,
    )
    index = load_class_index(args.index)
    layout = LocalLayout(get_root(args.root), index)
    rows, short_classes = build_local_rows(layout, config)

    digest = write_manifest(rows, args.output, args.index, config)
    report_path = args.output.with_name(args.output.stem + "_short_classes.json")
    report_path.write_text(json.dumps(short_classes, indent=2) + "\n")

    for split, count in sorted(Counter(row.split for row in rows).items()):
        print(f"{split}: {count}")
    print(f"short classes: {len(short_classes)}")
    print(f"manifest sha256: {digest}")


if __name__ == "__main__":
    main()
