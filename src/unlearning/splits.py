"""Build the deterministic, manifest-based Phase 1 data splits."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from unlearning.classes import ClassIndex, Group
from unlearning.config import DataConfig
from unlearning.data.paths import relative_path

FIELDS = [
    "split",
    "source_split",
    "relative_path",
    "class_index_1b",
    "class_index_0b",
    "synset",
    "name",
    "group",
]


class Split(str, Enum):
    TRAIN_FORGET = "train_forget"
    TRAIN_RETAIN = "train_retain"
    RELEARN = "relearn"
    VAL_TUNE = "val_tune"
    VAL_TEST = "val_test"


class DataError(ValueError):
    """Raised when the dataset cannot satisfy the requested split counts."""


@dataclass(frozen=True)
class ManifestRow:
    split: str
    source_split: str
    relative_path: str
    class_index_1b: int
    class_index_0b: int
    synset: str
    name: str
    group: str


def _sample(files: list[Path], count: int, rng: random.Random, label: str) -> list[Path]:
    if len(files) < count:
        raise DataError(f"{label}: need {count} images, found {len(files)}")
    return rng.sample(files, count)


def build_rows(root: Path, index: ClassIndex, config: DataConfig) -> list[ManifestRow]:
    """Select images with a per-class seeded RNG and sorted input files."""
    indices = config.class_indices or tuple(range(1000))
    rows: list[ManifestRow] = []

    for index_0b in indices:
        info = index.get(index_0b)
        group = index.group(index_0b)
        synset = info.synset

        train_files = sorted((root / "train" / synset).iterdir())
        train_rng = random.Random(f"{config.seed}:train:{synset}")
        need_train = config.train_per_class
        if group == Group.DOG:
            need_train += config.relearn_per_class
        chosen = _sample(train_files, need_train, train_rng, synset)

        train_rows: list[tuple[str, list[Path]]] = []
        if group == Group.DOG:
            train_rows.append((Split.TRAIN_FORGET, chosen[: config.train_per_class]))
            train_rows.append((Split.RELEARN, chosen[config.train_per_class :]))
        else:
            train_rows.append((Split.TRAIN_RETAIN, chosen))

        val_files = sorted((root / "val" / synset).iterdir())
        val_rng = random.Random(f"{config.seed}:val:{synset}")
        need_val = 2 * config.val_per_split_per_class
        val_chosen = _sample(val_files, need_val, val_rng, f"{synset}/val")
        half = config.val_per_split_per_class
        train_rows.extend(
            [
                (Split.VAL_TUNE, val_chosen[:half]),
                (Split.VAL_TEST, val_chosen[half:]),
            ]
        )

        for split, files in train_rows:
            for path in files:
                rows.append(
                    ManifestRow(
                        split=split.value,
                        source_split=(
                            "val"
                            if split in (Split.VAL_TUNE, Split.VAL_TEST)
                            else "train"
                        ),
                        relative_path=relative_path(path, root),
                        class_index_1b=info.imagenet_index_1b,
                        class_index_0b=index_0b,
                        synset=synset,
                        name=info.name,
                        group=group.value,
                    )
                )

    rows.sort(key=lambda row: (row.split, row.class_index_0b, row.relative_path))
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    rows: list[ManifestRow],
    path: Path,
    index_path: Path,
    config: DataConfig,
) -> str:
    """Write the CSV plus a metadata sidecar and a .sha256 file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    digest = sha256_file(path)
    meta = {
        "schema_version": 1,
        "seed": config.seed,
        "train_per_class": config.train_per_class,
        "relearn_per_class": config.relearn_per_class,
        "val_per_split_per_class": config.val_per_split_per_class,
        "class_index_sha256": sha256_file(index_path),
        "manifest_sha256": digest,
    }
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    path.with_suffix(".sha256").write_text(digest + "\n")
    return digest


def load_manifest(path: Path) -> list[ManifestRow]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            ManifestRow(
                split=row["split"],
                source_split=row["source_split"],
                relative_path=row["relative_path"],
                class_index_1b=int(row["class_index_1b"]),
                class_index_0b=int(row["class_index_0b"]),
                synset=row["synset"],
                name=row["name"],
                group=row["group"],
            )
            for row in csv.DictReader(handle)
        ]
