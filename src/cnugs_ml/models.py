"""Keras model builders for quick hackathon baselines."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple


def _keras_layers():
    from tensorflow import keras  # type: ignore
    from tensorflow.keras import layers  # type: ignore

    return keras, layers


def build_mlp_classifier(
    input_shape: Tuple[int, ...],
    num_classes: int,
    hidden_units: Sequence[int] = (512, 512, 512),
    learning_rate: float = 1e-3,
):
    """Build a dense classifier similar to the MNIST bootcamp baseline."""
    keras, layers = _keras_layers()
    model = keras.Sequential(name="mlp_classifier")
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Flatten())
    for units in hidden_units:
        model.add(layers.Dense(units, activation="relu"))
    model.add(layers.Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn_classifier(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    conv_filters: Sequence[int] = (32, 64, 64),
    dense_units: Sequence[int] = (128,),
    learning_rate: float = 1e-3,
    padding: str = "same",
):
    """Build a small CNN classifier for MNIST/CIFAR-style problems."""
    keras, layers = _keras_layers()
    model = keras.Sequential(name="cnn_classifier")
    model.add(layers.Input(shape=input_shape))
    for i, filters in enumerate(conv_filters):
        model.add(layers.Conv2D(filters, 3, padding=padding, activation="relu"))
        if i < len(conv_filters) - 1:
            model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    for units in dense_units:
        model.add(layers.Dense(units, activation="relu"))
    model.add(layers.Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_denoising_autoencoder(
    input_shape: Tuple[int, int, int],
    filters: Sequence[int] = (32, 32),
    learning_rate: float = 1e-3,
    output_activation: str = "sigmoid",
):
    """Build a simple image-to-image denoiser.

    For normalized images in `[0, 1]`, `sigmoid` is usually a safer output
    activation than `linear` because it keeps the prediction in image range.
    """
    keras, layers = _keras_layers()
    channels = input_shape[-1]
    model = keras.Sequential(name="denoising_autoencoder")
    model.add(layers.Input(shape=input_shape))
    for f in filters:
        model.add(layers.Conv2D(f, 3, padding="same", activation="relu"))
    model.add(layers.Conv2D(channels, 3, padding="same", activation=output_activation))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


def build_image_regressor(
    input_shape: Tuple[int, int, int],
    hidden_units: Sequence[int] = (64, 32),
    learning_rate: float = 1e-3,
):
    """Build a simple image-to-scalar regression model."""
    keras, layers = _keras_layers()
    model = keras.Sequential(name="image_regressor")
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Flatten())
    for units in hidden_units:
        model.add(layers.Dense(units, activation="relu"))
    model.add(layers.Dense(1, activation="linear"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model
