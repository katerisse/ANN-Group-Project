import keras
import numpy as np
import pytest

from unlearning import bn, models

requires_weights = pytest.mark.skipif(
    not models.default_weights_path().is_file(),
    reason="ImageNet weights are not cached",
)


def test_freeze_batch_norm_marks_all_layers():
    inputs = keras.Input((8,))
    x = bn.FrozenStatsBatchNormalization()(inputs)
    x = keras.layers.BatchNormalization()(x)
    model = keras.Model(inputs, x)

    assert bn.freeze_batch_norm(model) == 2
    assert all(not layer.trainable for layer in bn.iter_batch_norms(model))


@requires_weights
def test_frozen_batch_norm_has_exactly_zero_delta_after_a_step():
    model = models.load_resnet50()
    assert bn.freeze_batch_norm(model) == 53

    before = bn.bn_snapshot(model)
    model.compile(optimizer=keras.optimizers.SGD(0.01), loss="mse")
    image = np.zeros((2, 224, 224, 3), dtype=np.float32)
    labels = np.zeros((2, 1000), dtype=np.float32)
    model.train_on_batch(image, labels)

    assert bn.max_bn_delta(before, model) == 0.0


def test_gamma_beta_ablation_trains_affine_only():
    layer = bn.FrozenStatsBatchNormalization()
    inputs = keras.Input((4,))
    outputs = keras.layers.Dense(2)(layer(inputs))
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.SGD(0.5), loss="mse")

    mean_before = layer.moving_mean.numpy().copy()
    variance_before = layer.moving_variance.numpy().copy()
    gamma_before = layer.gamma.numpy().copy()

    data = np.random.default_rng(0).normal(size=(8, 4)).astype(np.float32)
    model.train_on_batch(data, np.zeros((8, 2), dtype=np.float32))

    np.testing.assert_array_equal(layer.moving_mean.numpy(), mean_before)
    np.testing.assert_array_equal(layer.moving_variance.numpy(), variance_before)
    assert not np.allclose(layer.gamma.numpy(), gamma_before)


def test_custom_layer_round_trips_through_keras(tmp_path):
    model = keras.Sequential(
        [keras.Input((4,)), bn.FrozenStatsBatchNormalization()]
    )
    path = tmp_path / "bn.keras"
    model.save(path)

    loaded = keras.saving.load_model(path)
    assert isinstance(loaded.layers[-1], bn.FrozenStatsBatchNormalization)


@requires_weights
def test_replace_batch_norms_preserves_outputs():
    original = models.load_resnet50()
    replaced = bn.replace_batch_norms(original)

    assert sum(
        isinstance(layer, bn.FrozenStatsBatchNormalization)
        for layer in bn.iter_layers(replaced)
    ) == 53

    image = np.zeros((1, 224, 224, 3), dtype=np.float32)
    expected = original(image, training=False).numpy()
    actual = replaced(image, training=False).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-4)
