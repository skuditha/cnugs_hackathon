"""Data loading and preprocessing utilities.

The functions here are intentionally small and notebook-friendly. They are meant
for quick debugging in a hackathon, especially for image arrays where shape and
scale mistakes are common.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ArrayLike = Any


def set_global_seed(seed: int = 42) -> None:
    """Set Python, NumPy, and TensorFlow seeds when TensorFlow is available."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # type: ignore

        tf.random.set_seed(seed)
    except Exception:
        # TensorFlow may not be installed in a lightweight local environment.
        pass


def describe_array(name: str, x: ArrayLike, max_unique: int = 20) -> None:
    """Print type, shape, dtype, min, max, and small unique-value summaries."""
    arr = np.asarray(x)
    print(f"{name}")
    print(f"  type:  {type(x)}")
    print(f"  shape: {arr.shape}")
    print(f"  dtype: {arr.dtype}")
    if arr.size == 0:
        print("  empty array")
        return
    if np.issubdtype(arr.dtype, np.number):
        print(f"  min:   {np.nanmin(arr)}")
        print(f"  max:   {np.nanmax(arr)}")
        print(f"  mean:  {np.nanmean(arr):.6g}")
        print(f"  std:   {np.nanstd(arr):.6g}")
    unique = np.unique(arr)
    if unique.size <= max_unique:
        print(f"  unique values: {unique}")
    else:
        print(f"  unique count: {unique.size}")


def normalize_images(images: ArrayLike, dtype: str = "float32") -> np.ndarray:
    """Convert image arrays to floats and scale to [0, 1] when needed.

    If the max is greater than 1, the array is divided by 255. This matches the
    MNIST/CIFAR bootcamp pattern while still being safe for already-normalized
    arrays.
    """
    x = np.asarray(images).astype(dtype)
    if x.size and np.nanmax(x) > 1.0:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def ensure_channel_dim(images: ArrayLike) -> np.ndarray:
    """Ensure images have a channel dimension.

    - `(N, H, W)` becomes `(N, H, W, 1)`.
    - `(H, W)` becomes `(H, W, 1)`.
    - Existing `(N, H, W, C)` arrays are unchanged.
    """
    x = np.asarray(images)
    if x.ndim == 2:
        return x[..., np.newaxis]
    if x.ndim == 3:
        return x[..., np.newaxis]
    if x.ndim == 4:
        return x
    raise ValueError(
        f"Expected 2D, 3D, or 4D image array, got shape {x.shape}. "
        "Common shapes are (H,W), (N,H,W), or (N,H,W,C)."
    )


def prepare_images(images: ArrayLike, add_channel: bool = True) -> np.ndarray:
    """Normalize image data and optionally add a channel dimension."""
    x = normalize_images(images)
    return ensure_channel_dim(x) if add_channel else x


def add_gaussian_noise(
    clean_images: ArrayLike,
    sigma: float = 0.3,
    clip: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Add Gaussian noise to normalized images."""
    rng = np.random.default_rng(seed)
    x = np.asarray(clean_images).astype("float32")
    noisy = x + sigma * rng.normal(size=x.shape)
    if clip:
        noisy = np.clip(noisy, 0.0, 1.0)
    return noisy.astype("float32")


def load_mnist_prepared(add_channel: bool = False) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load MNIST and normalize images to `[0, 1]`."""
    from tensorflow.keras.datasets import mnist  # type: ignore

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = prepare_images(train_images, add_channel=add_channel)
    test_images = prepare_images(test_images, add_channel=add_channel)
    return (train_images, train_labels), (test_images, test_labels)


def load_cifar10_prepared() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load CIFAR-10 and normalize images to `[0, 1]`."""
    from tensorflow import keras  # type: ignore

    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
    return (normalize_images(train_images), train_labels), (normalize_images(test_images), test_labels)


def split_train_val(
    x: ArrayLike,
    y: ArrayLike,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False,
):
    """Wrapper around sklearn's train_test_split with optional stratification."""
    y_arr = np.asarray(y)
    stratify_arg = y_arr if stratify else None
    return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=stratify_arg)


def read_table(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV/TSV/Excel/pickle table by extension."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", **kwargs)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, **kwargs)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path, **kwargs)
    raise ValueError(f"Unsupported table type: {suffix}")
