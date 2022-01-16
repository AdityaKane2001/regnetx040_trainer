import math

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp

from official.vision.image_classification.augment import RandAugment

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
        self.augmenter = self._get_augmenter()
    
    def init_randaugment(self):
        if self.cfg.randaugment.apply:
            augmenter = RandAugment(
                num_layers=self.cfg.randaugment.num_layers,
                magnitude=self.cfg.randaugment.magnitude)
        else:
            augmenter = None
        return augmenter

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
    
    def sample_beta_distribution(self, size, concentration_0=0.2, concentration_1=0.2):
        gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
        gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
        return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


    @tf.function
    def get_box(self, lambda_value):
        # Code credits: https://keras.io/examples/vision/cutmix/
        IMG_SIZE = self.cfg.image_size
        cut_rat = tf.math.sqrt(1.0 - lambda_value)

        cut_w = IMG_SIZE * cut_rat  # rw
        cut_w = tf.cast(cut_w, tf.int32)

        cut_h = IMG_SIZE * cut_rat  # rh
        cut_h = tf.cast(cut_h, tf.int32)

        cut_x = tf.random.uniform(
            (1,), minval=0, maxval=IMG_SIZE, dtype=tf.int32)  # rx
        cut_y = tf.random.uniform(
            (1,), minval=0, maxval=IMG_SIZE, dtype=tf.int32)  # ry

        boundaryx1 = tf.clip_by_value(cut_x[0] - cut_w // 2, 0, IMG_SIZE)
        boundaryy1 = tf.clip_by_value(cut_y[0] - cut_h // 2, 0, IMG_SIZE)
        bbx2 = tf.clip_by_value(cut_x[0] + cut_w // 2, 0, IMG_SIZE)
        bby2 = tf.clip_by_value(cut_y[0] + cut_h // 2, 0, IMG_SIZE)

        target_h = bby2 - boundaryy1
        if target_h == 0:
            target_h += 1

        target_w = bbx2 - boundaryx1
        if target_w == 0:
            target_w += 1

        return boundaryx1, boundaryy1, target_h, target_w


    @tf.function
    def cutmix(self, image: tf.Tensor, target: tf.Tensor) -> tuple:
        """
        Code credits: https://keras.io/examples/vision/cutmix/
        Applies cutmix augmentation to a batch of images.

        Args:
            image: batch of images
            target: batch of targets corresponding the targets

        Returns:
            Tuple containing (images, targets)
        """
        IMG_SIZE = self.cfg.image_size
        
        (image1, label1), (image2, label2) = train_ds_one, train_ds_two

        alpha = [0.25]
        beta = [0.25]

        # Get a sample from the Beta distribution
        lambda_value = self.sample_beta_distribution(1, alpha, beta)

        # Define Lambda
        lambda_value = lambda_value[0][0]

        # Get the bounding box offsets, heights and widths
        boundaryx1, boundaryy1, target_h, target_w = self.get_box(lambda_value)

        # Get a patch from the second image (`image2`)
        crop2 = tf.image.crop_to_bounding_box(
            image2, boundaryy1, boundaryx1, target_h, target_w
        )
        # Pad the `image2` patch (`crop2`) with the same offset
        image2 = tf.image.pad_to_bounding_box(
            crop2, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE
        )
        # Get a patch from the first image (`image1`)
        crop1 = tf.image.crop_to_bounding_box(
            image1, boundaryy1, boundaryx1, target_h, target_w
        )
        # Pad the `image1` patch (`crop1`) with the same offset
        img1 = tf.image.pad_to_bounding_box(
            crop1, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE
        )

        # Modify the first image by subtracting the patch from `image1`
        # (before applying the `image2` patch)
        image1 = image1 - img1
        # Add the modified `image1` and `image2`  together to get the CutMix image
        image = image1 + image2

        # Adjust Lambda in accordance to the pixel ration
        lambda_value = 1 - (target_w * target_h) / (IMG_SIZE * IMG_SIZE)
        lambda_value = tf.cast(lambda_value, tf.float32)

        # Combine the labels of both images
        label = lambda_value * label1 + (1 - lambda_value) * label2
        return image, label

    # Cutmix and Mixup
    # RandAugment
    # * Random Erasing
    def randaugment(self, image: tf.Tensor, target: tf.Tensor) -> tuple:
        """
        Applies RandAugment using implementation from tensorflow/models.
        processes a single image at a time.
        
        Args:
            image: single image
            target: single target

        Returns:
            Tuple containing (images, targets)
        """
        img = self.augmenter.distort(image)
        return img, target

    def random_erasing(self, image: tf.Tensor, target: tf.Tensor) -> tuple:
        pass
