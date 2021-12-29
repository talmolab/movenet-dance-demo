import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from typing import Union, Tuple


KEYPOINT_NODES = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

KEYPOINT_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def disable_gpu_preallocation():
    """Disable GPU pre-allocation so TensorFlow doesn't use all the GPU memory."""
    for gpu in tf.config.get_visible_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)


def center_pad(img: np.ndarray, image_size: int) -> np.ndarray:
    """Center pad an image to its longest dimension and resize.

    Args:
        img: (1, height, width, 3) or (height, width, 3) array.
        image_size: Length of square to resize to.

    Returns:
        Padded image of size (1, image_size, image_size, 3).
    """
    if img.ndim == 4:
        img = np.squeeze(img, axis=0)
    height, width, _ = img.shape
    if height > width:
        pad_pre = (height - width) // 2
        pad_post = (height - width) - pad_pre
        off_x, off_y = pad_pre, 0
        img = np.pad(img, ((0, 0), (pad_pre, pad_post), (0, 0)))
    else:
        pad_pre = (width - height) // 2
        pad_post = (width - height) - pad_pre
        off_x, off_y = 0, pad_pre
        img = np.pad(img, ((pad_pre, pad_post), (0, 0), (0, 0)))

    img = cv2.resize(img, (image_size, image_size))
    scale = image_size / max(height, width)
    img = np.expand_dims(img, axis=0)
    return img, scale, off_x, off_y


def center_crop(img: np.ndarray, image_size: int) -> np.ndarray:
    """Center crop an image to its shortest dimension and resize.

    Args:
        img: (1, height, width, 3) or (height, width, 3) array.
        image_size: Length of square to resize to.

    Returns:
        Cropped image of size (1, image_size, image_size, 3).
    """
    if img.ndim == 4:
        img = np.squeeze(img, axis=0)
    height, width, _ = img.shape
    if height > width:
        crop_pre = (height - width) // 2
        off_x, off_y = 0, -crop_pre
        img = img[crop_pre : (crop_pre + width), :]
    else:
        crop_pre = (width - height) // 2
        off_x, off_y = -crop_pre, 0
        img = img[:, crop_pre : (crop_pre + height)]

    img = cv2.resize(img, (image_size, image_size))
    scale = image_size / min(height, width)
    img = np.expand_dims(img, axis=0)
    return img, scale, off_x, off_y


def get_num_frames(video_path: str) -> int:
    """Return the number of frames in a video."""
    vr = cv2.VideoCapture(video_path)
    return int(vr.get(cv2.CAP_PROP_FRAME_COUNT))


def read_frame(video_path: str, fidx: int) -> np.ndarray:
    """Read a single frame from a video.

    Args:
        video_path: Path to a video file.
        fidx: 0-based frame index.

    Returns:
        Image as numpy array of shape (height, width, 3) in RGB channel order.
    """
    vr = cv2.VideoCapture(video_path)
    vr.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    frame = vr.read()[1][..., ::-1]
    return frame


def imgfig(
    size: Union[float, Tuple] = 6, dpi: int = 72, scale: float = 1.0
) -> matplotlib.figure.Figure:
    """Create a tight figure for image plotting.

    Args:
        size: Scalar or 2-tuple specifying the (width, height) of the figure in inches.
            If scalar, will assume equal width and height.
        dpi: Dots per inch, controlling the resolution of the image.
        scale: Factor to scale the size of the figure by. This is a convenience for
            increasing the size of the plot at the same DPI.

    Returns:
        A matplotlib.figure.Figure to use for plotting.
    """
    if not isinstance(size, (tuple, list)):
        size = (size, size)
    fig = plt.figure(figsize=(scale * size[0], scale * size[1]), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    return fig


def plot_img(
    img: np.ndarray, dpi: int = 72, scale: float = 1.0
) -> matplotlib.figure.Figure:
    """Plot an image in a tight figure."""
    if hasattr(img, "numpy"):
        img = img.numpy()

    if img.shape[0] == 1:
        # Squeeze out batch singleton dimension.
        img = img.squeeze(axis=0)

    # Check if image is grayscale (single channel).
    grayscale = img.shape[-1] == 1
    if grayscale:
        # Squeeze out singleton channel.
        img = img.squeeze(axis=-1)

    # Normalize the range of pixel values.
    img_min = img.min()
    img_max = img.max()
    if img_min < 0.0 or img_max > 1.0:
        img = (img - img_min) / (img_max - img_min)

    fig = imgfig(
        size=(float(img.shape[1]) / dpi, float(img.shape[0]) / dpi),
        dpi=dpi,
        scale=scale,
    )

    ax = fig.gca()
    ax.imshow(
        img,
        cmap="gray" if grayscale else None,
        origin="upper",
        extent=[-0.5, img.shape[1] - 0.5, img.shape[0] - 0.5, -0.5],
    )
    return fig


def plot_pose(
    keypoints: np.ndarray,
    lw: int = 6,
    ms: int = 30,
    cmap: str = "tab20",
    scale: float = 1.0,
    **kwargs
):
    """Plot a set of keypoints colored by edges."""
    colors = sns.color_palette(cmap, len(KEYPOINT_EDGES))

    for k in range(len(KEYPOINT_EDGES)):
        s, d = KEYPOINT_EDGES[k]
        src, dst = keypoints[s] * scale, keypoints[d] * scale

        plt.plot(
            [src[0], dst[0]],
            [src[1], dst[1]],
            ".-",
            lw=lw,
            ms=ms,
            c=colors[k],
            **kwargs
        )
