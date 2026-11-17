"""Pure weighted score combinations; ranking never reads target labels."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

SPACES = ("probability", "logit", "rank", "rank_logit")
LOGIT_CLIP = 1e-6


def blend_scores(columns, weights=None, *, space="probability") -> np.ndarray:
    """Blend whole aligned vectors; rank OOF and test in separate calls."""
    if space not in SPACES:
        raise ValueError(f"Unknown score space: {space}")
    values = [np.asarray(column, dtype=np.float64) for column in columns]
    if not values or any(
        column.ndim != 1
        or not len(column)
        or column.shape != values[0].shape
        or not np.isfinite(column).all()
        or ((column < 0) | (column > 1)).any()
        for column in values
    ):
        raise ValueError("Scores must be aligned nonempty finite vectors in [0, 1].")
    coefficients = np.asarray(
        weights if weights is not None else [1.0] * len(values), dtype=np.float64
    ).copy()
    if (
        coefficients.shape != (len(values),)
        or not np.isfinite(coefficients).all()
        or (coefficients < 0).any()
        or not np.isfinite(coefficients.sum())
        or coefficients.sum() <= 0
    ):
        raise ValueError("Provide finite nonnegative weights with a positive total.")
    coefficients /= coefficients.sum()
    if space in ("rank", "rank_logit"):
        values = [
            rankdata(column, method="average") / (len(column) + 1) for column in values
        ]
    if space in ("logit", "rank_logit"):
        values = [np.clip(column, LOGIT_CLIP, 1 - LOGIT_CLIP) for column in values]
        values = [np.log(column / (1 - column)) for column in values]
    result = sum(weight * column for weight, column in zip(coefficients, values))
    return (
        np.exp(-np.logaddexp(0.0, -result))
        if space in ("logit", "rank_logit")
        else result
    )
