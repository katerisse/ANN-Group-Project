import numpy as np
import keras
import tensorflow as tf

from unlearning.seeding import DETERMINISM_ENV, set_seed


def _tiny_output(seed: int) -> np.ndarray:
    set_seed(seed)
    model = keras.Sequential([keras.Input((3,)), keras.layers.Dense(4)])
    return model(tf.ones((1, 3)), training=False).numpy()


def test_same_seed_gives_identical_tiny_model():
    np.testing.assert_array_equal(_tiny_output(42), _tiny_output(42))


def test_different_seed_changes_initialization():
    assert not np.allclose(_tiny_output(42), _tiny_output(43))


def test_determinism_variables_are_documented():
    assert set(DETERMINISM_ENV) == {
        "TF_DETERMINISTIC_OPS",
        "TF_CUDNN_DETERMINISTIC",
        "PYTHONHASHSEED",
    }
