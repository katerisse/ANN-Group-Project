import keras
import numpy as np

from unlearning.methods import METHODS, mask
from unlearning.methods.base import kl_divergence, sparse_ce
from unlearning.methods.neggrad import loss as neggrad_loss
from unlearning.scope import set_trainable_scope


def _tiny_model():
    return keras.Sequential(
        [
            keras.Input((3,)),
            keras.layers.Dense(4, name="conv5_dense"),
            keras.layers.Dense(5, name="predictions"),
        ]
    )


def test_mask_disables_only_dog_logits():
    logits = np.ones((2, 5))
    masked = mask.mask_logits(logits, [1, 3])

    assert (masked[:, [1, 3]] == -1e9).all()
    assert (masked[:, [0, 2, 4]] == 1.0).all()
    assert (logits == 1.0).all()


def test_kl_is_zero_for_identical_logits():
    logits = np.array([[0.1, 0.2, 0.7]])
    value = kl_divergence(
        keras.ops.convert_to_tensor(logits), keras.ops.convert_to_tensor(logits)
    )
    assert np.isclose(float(value), 0.0)


def test_neggrad_flips_the_forget_term():
    model = _tiny_model()
    forget = (np.zeros((2, 3), dtype=np.float32), np.array([0, 1]))
    retain = (np.ones((2, 3), dtype=np.float32), np.array([2, 3]))

    value = neggrad_loss(
        model, forget, retain, lambda_retain=0.0
    )
    expected = -sparse_ce(model(forget[0], training=False), forget[1])
    np.testing.assert_allclose(float(value), float(expected), atol=1e-4)


def test_scope_selection_counts_trainable_layers():
    model = _tiny_model()

    assert set_trainable_scope(model, "none") == 0
    assert set_trainable_scope(model, "head") == 1
    assert [layer.name for layer in model.layers if layer.trainable] == [
        "predictions"
    ]
    assert set_trainable_scope(model, "conv5_head") == 2


def test_method_registry_covers_all_six_methods():
    assert set(METHODS) == {
        "original",
        "mask",
        "head_only",
        "retain_only",
        "neggrad",
        "scrub",
    }
