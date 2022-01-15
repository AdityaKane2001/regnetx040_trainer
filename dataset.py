import math

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp

from typing import Union, Callable, Tuple, List, Type
from datetime import datetime

tfd = tfp.distributions


AUTO = tf.data.AUTOTUNE

_TFRECS_FORMAT = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "height": tf.io.FixedLenFeature([], tf.int64),
    "width": tf.io.FixedLenFeature([], tf.int64),
    "filename": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
    "synset": tf.io.FixedLenFeature([], tf.string),
}


class ImageNet:
    """
    Class for all ImageNet data-related functions, including TFRecord
    parsing along with augmentation transforms. TFRecords must follow the format
    given in _TFRECS_FORMAT. 

    Augmentations are followed according to the flags in `cfg`.

    Args:
       cfg: The preprocessing ml_collections.ConfigDict instance
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.eigen_vecs, self.eigen_vals = self._get_eigen_vecs_and_vals()

    def _get_eigen_vecs_and_vals(self):
        eigen_vals = tf.constant(
            [[0.2175, 0.0188, 0.0045],
             [0.2175, 0.0188, 0.0045],
             [0.2175, 0.0188, 0.0045],
             ]
        )
        eigen_vals = tf.stack([eigen_vals] * self.batch_size, axis=0)

        eigen_vecs = tf.constant(
            [
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, 0.4203],
            ]
        )
        eigen_vecs = tf.stack([eigen_vecs] * self.batch_size, axis=0)

        return eigen_vecs, eigen_vals
    
    def decode_example(self, example_: tf.Tensor) -> dict:
        """Decodes an example to its individual attributes.

        Args:
            example: A TFRecord dataset example.

        Returns:
            Dict containing attributes from a single example. Follows
            the same names as _TFRECS_FORMAT.
        """

        example = tf.io.parse_example(example_, _TFRECS_FORMAT)
        image = tf.reshape(
            tf.io.decode_jpeg(
                example["image"]), (example["height"], example["width"], 3)
        )
        height = example["height"]
        width = example["width"]
        filename = example["filename"]
        label = example["label"]
        synset = example["synset"]
        return {
            "image": image,
            "height": height,
            "width": width,
            "filename": filename,
            "label": label,
            "synset": synset,
        }

    def _read_tfrecs(self) -> Type[tf.data.Dataset]:
        """Function for reading and loading TFRecords into a tf.data.Dataset.
        Returns dataset without batching or one-hot encoding.

        Args: None.

        Returns:
            A tf.data.Dataset instance.
        """

        files = tf.data.Dataset.list_files(self.tfrecs_filepath)
        ds = files.interleave(
            tf.data.TFRecordDataset, num_parallel_calls=AUTO, deterministic=False
        )

        ds = ds.map(self.decode_example, num_parallel_calls=AUTO)

        # ds = ds.map(self._one_hot_encode_example, num_parallel_calls=AUTO)
        # ds = ds.batch(self.batch_size, drop_remainder=True)
        ds = ds.prefetch(AUTO)

        
        return ds

    def _pca_jitter(self, image: tf.Tensor, target: tf.Tensor) -> tuple:
        """
        Applies PCA jitter to images.

        Args:
            image: Batch of images to perform random rotation on.
            target: Target tensor.

        Returns:
            Augmented example with batch of images and targets with same dimensions.
        """

        aug_images = tf.cast(image, tf.float32) / 255.
        alpha = tf.random.normal((self.cfg.preproc.batch_size, 3), stddev=0.1)
        alpha = tf.stack([alpha, alpha, alpha], axis=1)
        rgb = tf.math.reduce_sum(
            alpha * self.eigen_vals * self.eigen_vecs, axis=2)
        rgb = tf.expand_dims(rgb, axis=1)
        rgb = tf.expand_dims(rgb, axis=1)

        aug_images = aug_images + rgb
        aug_images = aug_images * 255.

        aug_images = tf.cast(tf.clip_by_value(aug_images, 0, 255), tf.uint8)

        return aug_images, target

    def random_flip(self, image: tf.Tensor, target: tf.Tensor) -> tuple:
        """
        Returns randomly flipped batch of images. Only horizontal flip
        is available

        Args:
            image: Batch of images to perform random rotation on.
            target: Target tensor.

        Returns:
            Augmented example with batch of images and targets with same dimensions.
        """

        aug_images = tf.image.random_flip_left_right(image)
        return aug_images, target

    def random_rotate(self, image: tf.Tensor, target: tf.Tensor) -> tuple:
        """
        Returns randomly rotated batch of images.

        Args:
            image: Batch of images to perform random rotation on.
            target: Target tensor.

        Returns:
            Augmented example with batch of images and targets with same dimensions.
        """

        angles = tf.random.uniform((self.batch_size,)) * (math.pi / 2.0)
        rotated = tfa.image.rotate(image, angles, fill_value=128.0)
        return rotated, target
    
    def cutmix(self, image: tf.Tensor, target: tf.Tensor) -> tuple:
        """
        Credits: https://keras.io/examples/vision/cutmix/
        Applies cutmix augmentation to a batch of images.

        Args:
            image: batch of images
            target: batch of targets corresponding the targets
        
        Returns:
            Tuple containing (images, targets)
        """
