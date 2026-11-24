import numpy as np
import pytest

from fixtures.make_synthetic_dataset import make_dataset
from unlearning.classes import default_class_index_path, load_class_index
from unlearning.config import DataConfig
from unlearning.data.tfdata import (
    TestSplitLeakageError,
    build_eval_dataset,
    build_tuning_dataset,
    build_train_dataset,
)
from unlearning.splits import Split, build_rows, write_manifest

INDEX = load_class_index()
INDICES = (0, 150)
CONFIG = DataConfig(
    seed=5,
    train_per_class=4,
    relearn_per_class=2,
    val_per_split_per_class=2,
    class_indices=INDICES,
)


@pytest.fixture
def data(tmp_path):
    root = make_dataset(
        tmp_path / "imagenet", INDICES, {0: 4, 150: 6}, val_count=4
    )
    manifest = tmp_path / "manifest.csv"
    rows = build_rows(root, INDEX, CONFIG)
    write_manifest(rows, manifest, default_class_index_path(), CONFIG)
    return root, manifest


def test_train_dataset_yields_correct_batches(data):
    root, manifest = data
    dataset = build_train_dataset(
        manifest_path=manifest,
        root=root,
        split=Split.TRAIN_FORGET,
        batch_size=2,
        seed=1,
    )

    images, labels = next(iter(dataset))

    assert images.shape == (2, 224, 224, 3)
    assert labels.shape == (2,)
    assert np.isfinite(images.numpy()).all()


def test_eval_and_tuning_are_separate(data):
    root, manifest = data

    images, _ = next(
        iter(
            build_eval_dataset(
                manifest_path=manifest,
                root=root,
                split=Split.VAL_TEST,
                batch_size=4,
            )
        )
    )
    assert images.shape == (4, 224, 224, 3)

    with pytest.raises(TestSplitLeakageError):
        build_tuning_dataset(
            manifest_path=manifest,
            root=root,
            split=Split.VAL_TEST,
            batch_size=2,
        )
    with pytest.raises(TestSplitLeakageError):
        build_eval_dataset(
            manifest_path=manifest,
            root=root,
            split=Split.VAL_TUNE,
            batch_size=2,
        )


def test_train_dataset_is_reproducible(data):
    root, manifest = data

    def first_batch():
        return next(
            iter(
                build_train_dataset(
                    manifest_path=manifest,
                    root=root,
                    split=Split.TRAIN_FORGET,
                    batch_size=2,
                    seed=11,
                )
            )
        )

    images_a, labels_a = first_batch()
    images_b, labels_b = first_batch()
    np.testing.assert_array_equal(images_a.numpy(), images_b.numpy())
    np.testing.assert_array_equal(labels_a.numpy(), labels_b.numpy())
