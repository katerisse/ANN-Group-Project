"""Integrity checks for ImageNet archives and the extracted directory layout."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

ARCHIVE_MD5 = {
    "ILSVRC2012_img_val.tar": "29b22e2961454d5413ddabcf34fc5622",
    "ILSVRC2012_devkit_t12.tar.gz": "fa75699e90414af021442c21a62c3abf",
    "ILSVRC2012_img_train.tar": "1d675b47d978889d74fa0da5fadfb00e",
}

TRAIN_IMAGES = 1_281_167
VAL_IMAGES = 50_000
VAL_PER_CLASS = 50
CLASS_DIRS = 1000


@dataclass
class VerificationReport:
    checked: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, message: str) -> None:
        self.errors.append(message)


def md5_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archives(raw_dir: Path, include_train: bool = False) -> VerificationReport:
    report = VerificationReport()
    filenames = list(ARCHIVE_MD5)
    if not include_train:
        filenames.remove("ILSVRC2012_img_train.tar")

    for name in filenames:
        path = raw_dir / name
        if not path.is_file():
            report.add_error(f"missing archive: {path}")
            continue
        actual = md5_of(path)
        report.checked[name] = path.stat().st_size
        if actual != ARCHIVE_MD5[name]:
            report.add_error(f"MD5 mismatch for {name}: {actual}")
    return report


def _count_images(directory: Path) -> int:
    return sum(1 for path in directory.iterdir() if path.is_file())


def verify_layout(root: Path, require_train: bool = False) -> VerificationReport:
    report = VerificationReport()

    val_root = root / "val"
    if not val_root.is_dir():
        report.add_error(f"missing validation directory: {val_root}")
    else:
        synsets = sorted(p.name for p in val_root.iterdir() if p.is_dir())
        report.checked["val_classes"] = len(synsets)
        if len(synsets) != CLASS_DIRS:
            report.add_error(f"expected {CLASS_DIRS} val classes, found {len(synsets)}")
        total = 0
        for synset in synsets:
            count = _count_images(val_root / synset)
            total += count
            if count != VAL_PER_CLASS:
                report.add_error(f"{synset}: expected 50 val images, found {count}")
        report.checked["val_images"] = total
        if total != VAL_IMAGES:
            report.add_error(f"expected {VAL_IMAGES} val images, found {total}")

    devkit = root / "devkit"
    ground_truth = devkit / "data" / "ILSVRC2012_validation_ground_truth.txt"
    if not ground_truth.is_file():
        report.add_error(f"missing validation ground truth: {ground_truth}")
    else:
        lines = ground_truth.read_text().splitlines()
        report.checked["ground_truth_labels"] = len(lines)
        if len(lines) != VAL_IMAGES:
            report.add_error(f"expected {VAL_IMAGES} labels, found {len(lines)}")

    if require_train:
        train_root = root / "train"
        if not train_root.is_dir():
            report.add_error(f"missing training directory: {train_root}")
        else:
            synsets = sorted(p.name for p in train_root.iterdir() if p.is_dir())
            report.checked["train_classes"] = len(synsets)
            if len(synsets) != CLASS_DIRS:
                report.add_error(
                    f"expected {CLASS_DIRS} train classes, found {len(synsets)}"
                )
            total = 0
            for synset in synsets:
                count = _count_images(train_root / synset)
                total += count
                for path in (train_root / synset).iterdir():
                    if path.is_file() and not path.name.startswith(f"{synset}_"):
                        report.add_error(f"unexpected train filename: {path}")
                        break
            report.checked["train_images"] = total
            if total != TRAIN_IMAGES:
                report.add_error(f"expected {TRAIN_IMAGES} train images, found {total}")

    return report


def load_synset_mapping(path: Path) -> set[str]:
    """Read the synsets listed in LOC_synset_mapping.txt."""
    synsets = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            synsets.add(line.split()[0])
    return synsets


def load_synset_mapping_from_devkit(devkit_dir: Path) -> set[str]:
    """Read synsets from the development kit's meta.mat."""
    from scipy.io import loadmat

    mat = loadmat(devkit_dir / "data" / "meta.mat")
    return {str(row[1][0]) for row in mat["synsets"]}
