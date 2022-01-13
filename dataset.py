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
       cfg: The main ml_collections.ConfigDict instance
    """

    def __
