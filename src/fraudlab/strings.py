"""Object string arrays whose repeated values reference the same Python string."""

import numpy as np


def categorical_strings(values: np.ndarray) -> np.ndarray:
    """Preserve integer/Unicode text while converting only distinct values.

    This is storage sharing, not a learned vocabulary or pandas categorical dtype.
    Float/object inputs are excluded because their formatting is not interchangeable
    with pandas' existing raw-value conversion.
    """
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iuU":
        raise ValueError(
            "Shared categorical strings require integer or Unicode arrays."
        )
    unique, inverse = np.unique(values, return_inverse=True)
    return unique.astype(str).astype(object)[inverse]
