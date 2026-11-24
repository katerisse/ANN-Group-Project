import hashlib

import keras
import numpy as np
import pytest

from unlearning import models
from unlearning.bn import iter_batch_norms
from unlearning.seeding import set_seed

requires_weights = pytest.mark.skipif(
    not models.default_weights_path().is_file(),
    reason="ImageNet weights are not cached",
)


def test_verify_weights_accepts_the_pinned_file():
    path = models.default_weights_path()
    if not path.is_file():
        pytest.skip("ImageNet weights are not cached")
    assert models.verify_weights_file(path) == models.WEIGHTS_SHA256


def test_verify_weights_fails_closed(tmp_path, monkeypatch):
    fake = tmp_path / "weights.h5"
    fake.write_bytes(b"not weights")
    with pytest.raises(ValueError, match="size"):
        models.verify_weights_file(fake)

    data = b"x" * 16
    fake.write_bytes(data)
    monkeypatch.setattr(models, "WEIGHTS_SIZE", len(data))
    monkeypatch.setattr(models, "WEIGHTS_MD5", hashlib.md5(data).hexdigest())
    monkeypatch.setattr(models, "WEIGHTS_SHA256", hashlib.sha256(data).hexdigest())
    assert models.verify_weights_file(fake) == hashlib.sha256(data).hexdigest()

    monkeypatch.setattr(models, "WEIGHTS_MD5", "0" * 32)
    with pytest.raises(ValueError, match="MD5"):
        models.verify_weights_file(fake)


@requires_weights
def test_resnet50_shape_counts_and_logits():
    model = models.load_resnet50()

    assert model.count_params() == 25_636_712
    assert len(list(iter_batch_norms(model))) == 53
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 1000)

    image = np.zeros((1, 224, 224, 3), dtype=np.float32)
    logits = model(image, training=False).numpy()
    assert logits.shape == (1, 1000)
    assert not np.isclose(logits.sum(axis=-1), 1.0).all()

    probs = keras.ops.convert_to_numpy(models.softmax_logits(logits))
    assert np.isclose(probs.sum(axis=-1), 1.0).all()


@requires_weights
def test_fixed_seed_gives_identical_logits():
    image = np.random.default_rng(0).random((1, 224, 224, 3), dtype=np.float32)

    set_seed(0)
    first = models.load_resnet50()(image, training=False).numpy()
    set_seed(0)
    second = models.load_resnet50()(image, training=False).numpy()

    np.testing.assert_array_equal(first, second)
