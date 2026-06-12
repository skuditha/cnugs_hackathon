"""Classical ML helpers: Iris/kNN-style baselines."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def load_iris_dataframe() -> pd.DataFrame:
    """Load Iris as a DataFrame with a numeric target column named `type`."""
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df = df.rename(columns={"target": "type"})
    return df


def train_knn_classifier(x_train, y_train, n_neighbors: int = 3) -> KNeighborsClassifier:
    """Train a kNN classifier."""
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(x_train, y_train)
    return model


def sweep_k_neighbors(
    x_train,
    y_train,
    x_val,
    y_val,
    k_values: Iterable[int] = range(1, 21),
) -> pd.DataFrame:
    """Evaluate kNN validation accuracy over a range of k values."""
    rows = []
    for k in k_values:
        model = train_knn_classifier(x_train, y_train, n_neighbors=k)
        score = model.score(x_val, y_val)
        rows.append({"n_neighbors": k, "accuracy": score})
    return pd.DataFrame(rows)
