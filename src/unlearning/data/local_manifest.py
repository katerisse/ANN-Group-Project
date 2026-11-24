"""Seeded split allocation for the flat local ImageNet copy.

The local copy has no separate ILSVRC train/val sources, so all splits are
sampled from one per-class pool. When a class has fewer images than the
configured budget, every requested split is scaled proportionally and the
shortfall is reported rather than silently ignored.
"""

from __future__ import annotations

import random
from pathlib import Path

from unlearning.classes import Group
from unlearning.config import DataConfig
from unlearning.data.local_layout import LocalLayout
from unlearning.splits import ManifestRow


def allocate(
    files: list[Path], counts: dict[str, int], seed: str
) -> dict[str, list[Path]]:
    """Assign distinct files to splits in ``counts`` insertion order."""
    shuffled = random.Random(seed).sample(sorted(files), len(files))
    needed = sum(counts.values())
    if not counts:
        raise ValueError("no splits requested")
    if len(shuffled) >= needed:
        sizes = dict(counts)
    else:
        scale = len(shuffled) / needed
        sizes = {key: max(1, int(value * scale)) for key, value in counts.items()}
        while sum(sizes.values()) > len(shuffled):
            reducible = [key for key in sizes if sizes[key] > 1]
            if reducible:
                sizes[max(reducible, key=sizes.get)] -= 1
            else:
                del sizes[list(sizes)[-1]]  # drop the lowest-priority split
        keys = list(sizes)
        cursor = 0
        while sum(sizes.values()) < len(shuffled):
            sizes[keys[cursor % len(keys)]] += 1
            cursor += 1

    result: dict[str, list[Path]] = {}
    cursor = 0
    for key, size in sizes.items():
        result[key] = shuffled[cursor : cursor + size]
        cursor += size
    return result


def build_local_rows(
    layout: LocalLayout, config: DataConfig
) -> tuple[list[ManifestRow], list[dict]]:
    """Build manifest rows and a short-class report from a flat dataset."""
    index = layout.index
    rows: list[ManifestRow] = []
    report: list[dict] = []

    for index_0b in range(1000):
        info = index.get(index_0b)
        group = index.group(index_0b)
        # Evaluation splits are listed first so they survive extreme shortfalls.
        counts: dict[str, int] = {
            "val_test": config.val_per_split_per_class,
            "val_tune": config.val_per_split_per_class,
        }
        if group == Group.DOG:
            counts["train_forget"] = config.train_per_class
            counts["relearn"] = config.relearn_per_class
        else:
            counts["train_retain"] = config.train_per_class

        files = layout.images(index_0b)
        allocated = allocate(files, counts, f"{config.seed}:local:{info.synset}")
        if len(files) < sum(counts.values()):
            report.append(
                {
                    "model_index_0b": index_0b,
                    "name": info.name,
                    "requested": sum(counts.values()),
                    "available": len(files),
                    "allocated": {key: len(value) for key, value in allocated.items()},
                }
            )

        for split, paths in allocated.items():
            for path in paths:
                rows.append(
                    ManifestRow(
                        split=split,
                        source_split="flat",
                        relative_path=path.relative_to(layout.root).as_posix(),
                        class_index_1b=info.imagenet_index_1b,
                        class_index_0b=index_0b,
                        synset=info.synset,
                        name=info.name,
                        group=group.value,
                    )
                )

    rows.sort(key=lambda row: (row.split, row.class_index_0b, row.relative_path))
    return rows, report
