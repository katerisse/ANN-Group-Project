from collections import Counter

import pytest

from fixtures.make_synthetic_dataset import make_dataset
from unlearning.classes import Group, load_class_index, default_class_index_path
from unlearning.config import DataConfig
from unlearning.splits import (
    Split,
    build_rows,
    load_manifest,
    sha256_file,
    write_manifest,
)

INDEX = load_class_index()
INDICES = (0, 150, 268)
CONFIG = DataConfig(
    seed=7,
    train_per_class=5,
    relearn_per_class=2,
    val_per_split_per_class=3,
    class_indices=INDICES,
)


@pytest.fixture
def root(tmp_path):
    counts = {0: 5, 150: 9, 268: 5}
    return make_dataset(tmp_path / "imagenet", INDICES, counts, val_count=6)


def test_split_counts(root):
    rows = build_rows(root, INDEX, CONFIG)
    counts = Counter(row.split for row in rows)

    assert counts[Split.TRAIN_FORGET.value] == 5
    assert counts[Split.TRAIN_RETAIN.value] == 10
    assert counts[Split.RELEARN.value] == 2
    assert counts[Split.VAL_TUNE.value] == 9
    assert counts[Split.VAL_TEST.value] == 9


def test_group_and_source_labels(root):
    rows = build_rows(root, INDEX, CONFIG)

    forget = [r for r in rows if r.split == Split.TRAIN_FORGET.value]
    retain = [r for r in rows if r.split == Split.TRAIN_RETAIN.value]
    relearn = [r for r in rows if r.split == Split.RELEARN.value]

    assert {r.group for r in forget} == {Group.DOG.value}
    assert {r.group for r in relearn} == {Group.DOG.value}
    assert {r.group for r in retain} == {Group.OTHER.value, Group.CANID.value}
    assert {r.source_split for r in forget + retain + relearn} == {"train"}
    assert {
        r.source_split
        for r in rows
        if r.split in (Split.VAL_TUNE.value, Split.VAL_TEST.value)
    } == {"val"}


def test_splits_are_disjoint(root):
    rows = build_rows(root, INDEX, CONFIG)
    paths = {row.relative_path: row.split for row in rows}

    assert len(paths) == len(rows)
    forget = {r.relative_path for r in rows if r.split == Split.TRAIN_FORGET.value}
    relearn = {r.relative_path for r in rows if r.split == Split.RELEARN.value}
    tune = {r.relative_path for r in rows if r.split == Split.VAL_TUNE.value}
    test = {r.relative_path for r in rows if r.split == Split.VAL_TEST.value}
    assert not forget & relearn
    assert not tune & test
    assert not forget & tune


def test_same_seed_reproduces_manifest(root, tmp_path):
    first = build_rows(root, INDEX, CONFIG)
    second = build_rows(root, INDEX, CONFIG)
    assert first == second

    output = tmp_path / "manifest.csv"
    digest = write_manifest(first, output, default_class_index_path(), CONFIG)
    assert sha256_file(output) == digest
    assert (tmp_path / "manifest.sha256").read_text().strip() == digest
    assert load_manifest(output) == first


def test_different_seed_changes_sample(root):
    rows_a = build_rows(root, INDEX, CONFIG)
    rows_b = build_rows(root, INDEX, DataConfig(**{**CONFIG.__dict__, "seed": 8}))
    assert rows_a != rows_b
