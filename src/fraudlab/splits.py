"""Explicit split strategies shared by every model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)


def make_splits(
    frame: pd.DataFrame,
    target: str,
    method: str,
    folds: int,
    seed: int,
    group_column: str | None = None,
    time_column: str | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if folds < 2:
        raise ValueError("At least two folds are required.")
    y = frame[target]
    if method == "stratified":
        if y.value_counts().min() < folds:
            raise ValueError("Every class needs at least one row per fold.")
        splits = list(
            StratifiedKFold(folds, shuffle=True, random_state=seed).split(frame, y)
        )
    elif method == "group":
        if not group_column or group_column not in frame:
            raise ValueError("Group validation requires an existing --group-column.")
        groups = frame[group_column]
        if groups.isna().any() or groups.nunique() < folds:
            raise ValueError(
                "Group column must be non-null and contain at least one group per fold."
            )
        splits = list(
            StratifiedGroupKFold(folds, shuffle=True, random_state=seed).split(
                frame, y, groups
            )
        )
    elif method == "time":
        if not time_column or time_column not in frame:
            raise ValueError("Time validation requires an existing --time-column.")
        values = frame[time_column]
        if not pd.api.types.is_numeric_dtype(values):
            values = pd.to_datetime(values, errors="raise", utc=True)
        if values.isna().any():
            raise ValueError("Time column must not contain missing values.")
        # Split unique timestamps, so simultaneous events never cross a boundary.
        times = np.sort(values.unique())
        if len(times) <= folds:
            raise ValueError("Time validation needs more unique timestamps than folds.")
        splits = [
            (
                np.flatnonzero(values.isin(times[a])),
                np.flatnonzero(values.isin(times[b])),
            )
            for a, b in TimeSeriesSplit(folds).split(times)
        ]
    else:
        raise ValueError(f"Unknown validation method: {method}")
    seen: set[int] = set()
    for training, validation in splits:
        if set(training) & set(validation) or seen & set(validation):
            raise ValueError(
                "Training and validation rows must be disjoint; each OOF row is predicted once."
            )
        seen.update(validation)
        if y.iloc[training].nunique() != 2 or y.iloc[validation].nunique() != 2:
            raise ValueError(
                "Every training and validation fold needs both classes for ROC-AUC."
            )
    return splits
