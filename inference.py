"""Model building and inference-related utilities.

Refs:
    - https://www.tensorflow.org/hub/tutorials/movenet
    - https://tfhub.dev/google/movenet/singlepose/lightning/4
    - https://tfhub.dev/google/movenet/singlepose/thunder/4
"""

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import utils
from typing import Tuple


MODELS = {
    "lightning": {
        "model_path": "https://tfhub.dev/google/movenet/singlepose/lightning/4",
        "image_size": 192,
    },
    "thunder": {
        # "model_path": "https://tfhub.dev/google/movenet/singlepose/thunder/4",
        "model_path": "models/movenet_singlepose_thunder",
        "image_size": 256,
    },
}


def disable_gpu_preallocation():
    """Disable GPU pre-allocation so TensorFlow doesn't use all the GPU memory."""
    for gpu in tf.config.get_visible_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


def use_cpu_only():
    """Disable GPU usage."""
    tf.config.set_visible_devices([], "GPU")


def load_model(model_name: str) -> tf.keras.Model:
    """Load a MoveNet model by name.

    Args:
        model_name: Name of the model ("lightning" or "thunder")

    Returns:
        A tf.keras.Model ready for inference.
    """
    model_path = MODELS[model_name]["model_path"]
    image_size = MODELS[model_name]["image_size"]

    x_in = tf.keras.layers.Input([image_size, image_size, 3], name="image")
    x = tf.keras.layers.Lambda(
        lambda x: tf.cast(x, dtype=tf.int32), name="cast_to_int32"
    )(x_in)
    layer = hub.KerasLayer(
        model_path,
        signature="serving_default",
        output_key="output_0",
        name="movenet_layer",
    )
    x = layer(x)

    def split_outputs(x):
        x_ = tf.reshape(x, [17, 3])
        keypoints = tf.gather(x_, [1, 0], axis=-1)
        keypoints *= image_size
        scores = tf.squeeze(tf.gather(x_, [2], axis=-1), axis=1)
        return keypoints, scores

    x = tf.keras.layers.Lambda(split_outputs, name="keypoints_and_scores")(x)
    model = tf.keras.Model(x_in, x)
    return model


def predict(
    model: tf.keras.Model, img: np.ndarray, pad: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict on a single image with preprocessing.

    Args:
        model: A MoveNet tf.keras.Model (see load_model()).
        img: A single image of shape (height, width, 3).
        pad: If True (the default), pad instead of center cropping image.

    Returns:
        A tuple of (points, scores).

        points: Predicted keypoints of the pose as array of shape (17, 2).
        scores: Predicted scores of the keypoints as array of shape (17,).
    """
    image_size = model.input.shape[1]
    if pad:
        img_, scale, off_x, off_y = utils.center_pad(img, image_size)
    else:
        img_, scale, off_x, off_y = utils.center_crop(img, image_size)
    points, scores = model.predict_on_batch(img_)
    points /= scale
    points[:, 0] -= off_x
    points[:, 1] -= off_y
    return points, scores
