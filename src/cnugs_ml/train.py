"""Training helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def make_early_stopping(monitor: str = "val_loss", patience: int = 10, restore_best_weights: bool = True):
    """Create a Keras EarlyStopping callback."""
    from tensorflow.keras.callbacks import EarlyStopping  # type: ignore

    return EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=restore_best_weights)


def make_model_checkpoint(path: str | Path, monitor: str = "val_loss", save_best_only: bool = True):
    """Create a Keras ModelCheckpoint callback."""
    from tensorflow.keras.callbacks import ModelCheckpoint  # type: ignore

    return ModelCheckpoint(str(path), monitor=monitor, save_best_only=save_best_only)


def predict_classes(model, x):
    """Return class labels from a softmax classifier."""
    import numpy as np

    return np.asarray(model.predict(x)).argmax(axis=1)
