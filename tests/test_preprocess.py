import numpy as np
import tensorflow as tf
from keras.applications.resnet50 import preprocess_input as keras_preprocess
from PIL import Image

from unlearning.data.preprocess import ResNet50Preprocessor


def test_caffe_preprocess_matches_keras():
    rng = np.random.default_rng(0)
    batch = rng.uniform(0, 255, size=(4, 224, 224, 3)).astype(np.float32)

    ours = ResNet50Preprocessor().preprocess(tf.constant(batch)).numpy()
    expected = keras_preprocess(batch.copy())

    np.testing.assert_allclose(ours, expected, atol=1e-4)


def test_eval_pipeline_produces_correct_shape(tmp_path):
    path = tmp_path / "sample.JPEG"
    Image.new("RGB", (320, 240), (10, 20, 30)).save(path)

    processor = ResNet50Preprocessor()
    image = processor.decode(str(path))
    output = processor.eval(image).numpy()

    assert output.shape == (224, 224, 3)
    assert np.isfinite(output).all()


def test_train_pipeline_is_stateless(tmp_path):
    path = tmp_path / "sample.JPEG"
    Image.new("RGB", (320, 240), (10, 20, 30)).save(path)

    processor = ResNet50Preprocessor()
    image = processor.decode(str(path))

    first = processor.train(image, seed=(3, 4)).numpy()
    second = processor.train(image, seed=(3, 4)).numpy()

    np.testing.assert_array_equal(first, second)
