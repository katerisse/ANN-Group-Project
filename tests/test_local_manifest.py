from pathlib import Path

from unlearning.classes import load_class_index
from unlearning.config import DataConfig
from unlearning.data.local_layout import LocalLayout, normalize_name
from unlearning.data.local_manifest import allocate, build_local_rows

INDEX = load_class_index()
LOCAL_NAMES = {
    134: "crane_bird",
    264: "cardigan_welsh_corgi",
    474: "cardigan",
    517: "crane",
    638: "maillot",
    639: "tank_suit",
}


def make_root(tmp_path, images_per_class=4):
    for index_0b in range(1000):
        name = LOCAL_NAMES.get(index_0b, normalize_name(INDEX.get(index_0b).name))
        folder = tmp_path / name
        folder.mkdir()
        for number in range(images_per_class):
            (folder / f"{number:03d}.jpg").write_bytes(b"x")
    return tmp_path


def test_allocate_is_reproducible_for_a_seed(tmp_path):
    files = [tmp_path / f"{i}.jpg" for i in range(10)]
    counts = {"train_forget": 4, "relearn": 2, "val_test": 1}

    first = allocate(files, counts, "seed:class")
    assert first == allocate(files, counts, "seed:class")
    assert allocate(files, counts, "other:class") != first


def test_short_class_keeps_at_least_one_image_per_split(tmp_path):
    files = [tmp_path / f"{i}.jpg" for i in range(3)]
    result = allocate(files, {"train_forget": 2, "relearn": 2, "val_test": 2}, "short")

    assert {key: len(value) for key, value in result.items()} == {
        "train_forget": 1,
        "relearn": 1,
        "val_test": 1,
    }
    assert len({path for paths in result.values() for path in paths}) == 3


def test_build_local_rows_reports_short_classes(tmp_path):
    root = make_root(tmp_path)
    for name in ("black_and_tan_coonhound", "affenpinscher"):
        next(iter(sorted((root / name).iterdir()))).unlink()

    config = DataConfig(
        seed=7,
        train_per_class=1,
        relearn_per_class=1,
        val_per_split_per_class=1,
    )
    layout = LocalLayout(root, INDEX)
    rows, report = build_local_rows(layout, config)

    assert len(rows) == 118 * 4 + 882 * 3 - 2
    assert sum(row.split == "val_test" for row in rows) == 1000
    assert sum(row.split == "train_forget" for row in rows) == 118
    assert sum(row.split == "train_retain" for row in rows) == 882
    assert {entry["name"] for entry in report} == {
        "black-and-tan_coonhound",
        "affenpinscher",
    }
    assert all(entry["allocated"]["val_test"] == 1 for entry in report)


def test_manifest_rows_use_flat_relative_paths(tmp_path):
    root = make_root(tmp_path, images_per_class=4)
    config = DataConfig(
        train_per_class=1, relearn_per_class=1, val_per_split_per_class=1
    )
    rows, report = build_local_rows(LocalLayout(root, INDEX), config)

    assert report == []
    assert all(row.source_split == "flat" for row in rows)
    assert all("/" in row.relative_path for row in rows)
    assert all((root / row.relative_path).is_file() for row in rows)

    dog_paths = {
        row.relative_path for row in rows if row.class_index_0b == 150
    }
    assert len(dog_paths) == 4
