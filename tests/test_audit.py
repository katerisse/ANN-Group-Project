from PIL import Image

from unlearning.data.audit import audit_manifest, write_reports
from unlearning.splits import ManifestRow


def _row(split: str, relative_path: str) -> ManifestRow:
    return ManifestRow(
        split=split,
        source_split="val" if split.startswith("val") else "train",
        relative_path=relative_path,
        class_index_1b=1,
        class_index_0b=0,
        synset="n01440764",
        name="tench",
        group="other",
    )


def test_audit_quarantines_corrupt_and_drops_train_duplicate(tmp_path):
    root = tmp_path / "imagenet"
    train_dir = root / "train" / "n01440764"
    val_dir = root / "val" / "n01440764"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    good = train_dir / "good.JPEG"
    Image.new("RGB", (32, 32), (1, 2, 3)).save(good)
    duplicate = val_dir / "duplicate.JPEG"
    duplicate.write_bytes(good.read_bytes())
    corrupt = train_dir / "corrupt.JPEG"
    corrupt.write_bytes(b"not an image")

    rows = [
        _row("train_forget", "train/n01440764/good.JPEG"),
        _row("val_test", "val/n01440764/duplicate.JPEG"),
        _row("train_retain", "train/n01440764/corrupt.JPEG"),
    ]
    audit = audit_manifest(rows, root)

    assert [row.relative_path for row in audit.kept] == [
        "train/n01440764/good.JPEG"
    ]
    assert audit.quarantine[0]["reason"] != "missing"
    assert audit.exact_duplicates[0]["reason"] == "drop_val_duplicate_of_train"


def test_audit_flags_near_duplicates(tmp_path):
    root = tmp_path / "imagenet"
    directory = root / "train" / "n01440764"
    directory.mkdir(parents=True)
    first = directory / "a.JPEG"
    second = directory / "b.JPEG"
    Image.new("RGB", (32, 32), (10, 20, 30)).save(first)
    Image.new("RGB", (32, 32), (11, 20, 30)).save(second)

    rows = [
        _row("train_retain", "train/n01440764/a.JPEG"),
        _row("train_retain", "train/n01440764/b.JPEG"),
    ]
    audit = audit_manifest(rows, root)

    assert len(audit.near_duplicates) >= 1


def test_reports_are_written(tmp_path):
    audit = audit_manifest([], tmp_path)
    summary = write_reports(audit, tmp_path / "reports")

    assert summary["kept"] == 0
    assert (tmp_path / "reports" / "audit_summary.json").is_file()
    assert (tmp_path / "reports" / "quarantine.csv").is_file()
