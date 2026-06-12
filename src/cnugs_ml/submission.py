"""Submission helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def save_classification_submission(
    predictions,
    output_path: str | Path,
    id_column: str = "id",
    prediction_column: str = "label",
    ids: Optional[Sequence] = None,
) -> pd.DataFrame:
    """Save a simple classification submission CSV."""
    predictions = np.asarray(predictions).reshape(-1)
    if ids is None:
        ids = np.arange(len(predictions))
    df = pd.DataFrame({id_column: ids, prediction_column: predictions})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def save_regression_submission(
    predictions,
    output_path: str | Path,
    id_column: str = "id",
    prediction_column: str = "prediction",
    ids: Optional[Sequence] = None,
) -> pd.DataFrame:
    """Save a simple regression submission CSV."""
    predictions = np.asarray(predictions).reshape(-1)
    if ids is None:
        ids = np.arange(len(predictions))
    df = pd.DataFrame({id_column: ids, prediction_column: predictions})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
