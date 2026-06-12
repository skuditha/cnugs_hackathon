"""Plotting utilities for fast model debugging."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def set_plot_style(height: float = 5.0) -> None:
    """Set a simple wide figure size inspired by the bootcamp notebooks."""
    plt.rcParams["figure.figsize"] = [1.618 * height, height]


def plot_image_grid(
    images,
    labels: Optional[Sequence] = None,
    n: int = 6,
    rows: int = 2,
    cmap: str = "gray",
    title_prefix: str = "Label",
) -> None:
    """Plot the first `n` images in a grid."""
    images = np.asarray(images)
    cols = int(np.ceil(n / rows))
    plt.figure(figsize=(3 * cols, 3 * rows))
    for i in range(min(n, len(images))):
        plt.subplot(rows, cols, i + 1)
        img = np.squeeze(images[i])
        plt.imshow(img, cmap=cmap if img.ndim == 2 else None)
        if labels is not None:
            plt.title(f"{title_prefix}: {labels[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_loss_history(history, skip_first: int = 0) -> None:
    """Plot train/validation loss curves from a Keras History object."""
    hist = history.history if hasattr(history, "history") else history
    plt.figure()
    if "loss" in hist:
        plt.plot(hist["loss"][skip_first:], label="Training Loss")
    if "val_loss" in hist:
        plt.plot(hist["val_loss"][skip_first:], label="Validation Loss")
    plt.title("Loss vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_accuracy_history(history, skip_first: int = 0) -> None:
    """Plot train/validation accuracy curves from a Keras History object."""
    hist = history.history if hasattr(history, "history") else history
    plt.figure()
    if "accuracy" in hist:
        plt.plot(hist["accuracy"][skip_first:], label="Training Accuracy")
    if "val_accuracy" in hist:
        plt.plot(hist["val_accuracy"][skip_first:], label="Validation Accuracy")
    plt.title("Accuracy vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def plot_prediction_triplets(inputs, predictions, targets, n: int = 5, cmap: str = "gray") -> None:
    """Plot input/prediction/target rows for image-to-image problems."""
    inputs = np.asarray(inputs)
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    plt.figure(figsize=(2.4 * n, 7))
    for i in range(min(n, len(inputs))):
        for row, arr, title in [
            (0, inputs, "Input"),
            (1, predictions, "Prediction"),
            (2, targets, "Target"),
        ]:
            plt.subplot(3, n, row * n + i + 1)
            img = np.squeeze(arr[i])
            plt.imshow(img, cmap=cmap if img.ndim == 2 else None)
            plt.title(title)
            plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names: Optional[Sequence[str]] = None) -> None:
    """Plot a confusion matrix using matplotlib only."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm)
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    ticks = np.arange(cm.shape[0])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if class_names is not None:
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)

    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.show()
