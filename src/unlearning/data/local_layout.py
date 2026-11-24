"""Adapter for the local flat ImageNet copy with human-readable folders.

The on-device copy has ``<class name>/<image>.jpg`` directories instead of the
ILSVRC ``train/<synset>/`` and ``val/<synset>/`` layout. Folders are mapped to
the committed class index by a normalised name; a small alias table resolves
names that are ambiguous or use local wording.
"""

from __future__ import annotations

import re
from pathlib import Path

from unlearning.classes import ClassIndex

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class LocalLayoutError(ValueError):
    """Raised when a local folder cannot be mapped unambiguously."""


# Values are 0-based model indices. Each entry was checked by classifying
# sample images from both colliding folders with the pinned ImageNet ResNet50.
ALIASES_0B = {
    "cardigan": 474,  # the knitted garment
    "cardigan_welsh_corgi": 264,  # the dog breed
    "crane": 517,  # the machine (second 'crane' synset)
    "crane_bird": 134,  # the bird (first 'crane' synset)
    "jack_o_lantern": 607,
    "maillot": 638,
    "tank_suit": 639,
}


def normalize_name(name: str) -> str:
    """Lower-case a class name and collapse punctuation to single underscores."""
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", name.lower())).strip("_")


class LocalLayout:
    """Maps one flat class folder per 0-based class index."""

    def __init__(self, root: Path, index: ClassIndex):
        self.root = Path(root)
        self.index = index
        self._by_folder = self._build()

    def _build(self) -> dict[int, Path]:
        if not self.root.is_dir():
            raise LocalLayoutError(f"dataset root is not a directory: {self.root}")

        by_name: dict[str, list[int]] = {}
        for index_0b in range(1000):
            by_name.setdefault(normalize_name(self.index.get(index_0b).name), []).append(
                index_0b
            )

        result: dict[int, Path] = {}
        for folder in sorted(path for path in self.root.iterdir() if path.is_dir()):
            key = normalize_name(folder.name)
            if key in ALIASES_0B:
                candidates = [ALIASES_0B[key]]
            else:
                candidates = by_name.get(key, [])
            if len(candidates) != 1:
                raise LocalLayoutError(
                    f"folder {folder.name!r} maps to {len(candidates)} classes "
                    f"(normalised {key!r}); add an explicit alias"
                )
            target = candidates[0]
            if target in result:
                raise LocalLayoutError(
                    f"both {result[target].name!r} and {folder.name!r} map to index {target}"
                )
            result[target] = folder

        missing = sorted(set(range(1000)) - set(result))
        if missing:
            names = ", ".join(self.index.get(i).name for i in missing[:10])
            raise LocalLayoutError(
                f"missing {len(missing)} class folders, first: {names}"
            )
        return result

    def folder(self, index_0b: int) -> Path:
        return self._by_folder[index_0b]

    def images(self, index_0b: int) -> list[Path]:
        """Sorted image files for one class; sorting makes seeded samples stable."""
        return sorted(
            path
            for path in self.folder(index_0b).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )

    def counts(self) -> dict[int, int]:
        return {index_0b: len(self.images(index_0b)) for index_0b in self._by_folder}
