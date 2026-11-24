"""ImageNet class index and the dog/canid group definitions used by Phase 1."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

CLASS_COUNT = 1000
DOG_1B = range(151, 269)
CANID_1B = range(269, 276)


class Group(str, Enum):
    DOG = "dog"
    CANID = "canid"
    OTHER = "other"


@dataclass(frozen=True)
class ClassInfo:
    imagenet_index_1b: int
    model_index_0b: int
    synset: str
    name: str


class ClassIndex:
    """An immutable 1000-class index keyed by the model's 0-based index."""

    def __init__(self, infos: list[ClassInfo]):
        self._by_index = {info.model_index_0b: info for info in infos}
        self.validate()

    def get(self, model_index_0b: int) -> ClassInfo:
        return self._by_index[model_index_0b]

    def group(self, model_index_0b: int) -> Group:
        index_1b = self.get(model_index_0b).imagenet_index_1b
        if index_1b in DOG_1B:
            return Group.DOG
        if index_1b in CANID_1B:
            return Group.CANID
        return Group.OTHER

    def indices(self, group: Group) -> tuple[int, ...]:
        return tuple(i for i in sorted(self._by_index) if self.group(i) == group)

    def validate(self) -> None:
        assert len(self._by_index) == CLASS_COUNT, "expected 1000 classes"
        assert set(self._by_index) == set(range(CLASS_COUNT))
        assert {i.imagenet_index_1b for i in self._by_index.values()} == set(
            range(1, CLASS_COUNT + 1)
        )
        assert len(self.indices(Group.DOG)) == 118
        assert self.indices(Group.DOG) == tuple(range(150, 268))
        assert len(self.indices(Group.CANID)) == 7
        assert self.indices(Group.CANID) == tuple(range(268, 275))
        assert len(self.indices(Group.OTHER)) == 875
        assert len(self.indices(Group.CANID) + self.indices(Group.OTHER)) == 882


def default_class_index_path() -> Path:
    """Path to the committed CSV, relative to this source file."""
    return Path(__file__).resolve().parents[2] / "manifests" / "class_index.csv"


def load_class_index(path: Path | None = None) -> ClassIndex:
    path = path or default_class_index_path()
    with path.open(newline="", encoding="utf-8") as handle:
        infos = [
            ClassInfo(
                imagenet_index_1b=int(row["imagenet_index_1b"]),
                model_index_0b=int(row["model_index_0b"]),
                synset=row["synset"],
                name=row["name"],
            )
            for row in csv.DictReader(handle)
        ]
    return ClassIndex(infos)


def check_synset_mapping(index: ClassIndex, synsets: set[str]) -> list[str]:
    """Return synsets present in one source but missing from the other."""
    expected = {info.synset for info in index._by_index.values()}
    return sorted(expected.symmetric_difference(synsets))
