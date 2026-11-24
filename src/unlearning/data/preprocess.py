"""ResNet50 preprocessing, kept behind a small swappable interface."""

from __future__ import annotations

import tensorflow as tf

CAFFE_MEAN = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)


class ResNet50Preprocessor:
    """Caffe-mode preprocessing: RGB to BGR, then subtract channel means."""

    def __init__(self, crop_size: int = 224, resize_side: int = 256):
        self.crop_size = crop_size
        self.resize_side = resize_side

    def decode(self, path: str | tf.Tensor) -> tf.Tensor:
        image = tf.io.decode_image(
            tf.io.read_file(path), channels=3, expand_animations=False
        )
        image.set_shape([None, None, 3])
        return tf.cast(image, tf.float32)

    def resize(self, image: tf.Tensor) -> tf.Tensor:
        shape = tf.cast(tf.shape(image), tf.float32)
        scale = self.resize_side / tf.minimum(shape[0], shape[1])
        height = tf.cast(tf.round(shape[0] * scale), tf.int32)
        width = tf.cast(tf.round(shape[1] * scale), tf.int32)
        return tf.image.resize(image, [height, width], method="bilinear", antialias=True)

    def preprocess(self, image: tf.Tensor) -> tf.Tensor:
        """Apply only the ResNet50 caffe normalization to a 224x224 RGB image."""
        image = tf.cast(image, tf.float32)
        return image[..., ::-1] - CAFFE_MEAN

    def eval(self, image: tf.Tensor) -> tf.Tensor:
        image = self.resize(image)
        crop = self.crop_size
        top = (tf.shape(image)[0] - crop) // 2
        left = (tf.shape(image)[1] - crop) // 2
        image = image[top : top + crop, left : left + crop, :]
        image.set_shape([self.crop_size, self.crop_size, 3])
        return self.preprocess(image)

    def train(self, image: tf.Tensor, seed: tuple[int, int] = (0, 0)) -> tf.Tensor:
        image = self.resize(image)
        crop = self.crop_size
        ratios = tf.random.stateless_uniform([2], seed=seed)
        top = tf.cast(ratios[0] * tf.cast(tf.shape(image)[0] - crop, tf.float32), tf.int32)
        left = tf.cast(
            ratios[1] * tf.cast(tf.shape(image)[1] - crop, tf.float32), tf.int32
        )
        image = image[top : top + crop, left : left + crop, :]
        image.set_shape([self.crop_size, self.crop_size, 3])
        flip = tf.random.stateless_uniform([], seed=(seed[0] + 1, seed[1])) < 0.5
        image = tf.cond(
            flip, lambda: tf.reverse(image, axis=[1]), lambda: tf.identity(image)
        )
        return self.preprocess(image)
