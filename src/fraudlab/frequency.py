"""Label-free category counts: fold-local by default, optionally fixed."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class FrequencyEncoder(TransformerMixin, BaseEstimator):
    """Encode log1p support from training or explicit reference; unknowns are zero."""

    # Older persisted encoders have no instance attribute for this new option.
    reference_counts = None

    def __init__(self, reference_counts=None):
        self.reference_counts = reference_counts

    def fit(self, frame: pd.DataFrame, y=None):
        if (
            not isinstance(frame, pd.DataFrame)
            or not frame.columns.is_unique
            or not all(isinstance(name, str) for name in frame.columns)
        ):
            raise ValueError(
                "FrequencyEncoder requires unique string DataFrame columns."
            )
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        if self.reference_counts is None:
            self.counts_ = {
                name: frame[name].value_counts(dropna=False, sort=False)
                for name in self.feature_names_in_
            }
        else:
            if not isinstance(self.reference_counts, dict) or set(
                self.reference_counts
            ) != set(frame):
                raise ValueError(
                    "Reference counts must match frequency columns exactly."
                )
            self.counts_ = {}
            for name in self.feature_names_in_:
                counts = self.reference_counts[name]
                if (
                    not isinstance(counts, pd.Series)
                    or not len(counts)
                    or not all(isinstance(key, str) for key in counts.index)
                    or not counts.index.is_unique
                    or counts.index.hasnans
                    or not pd.api.types.is_numeric_dtype(counts)
                    or pd.api.types.is_bool_dtype(counts)
                    or pd.api.types.is_complex_dtype(counts)
                ):
                    raise ValueError(
                        "Reference counts need unique string keys and numeric counts."
                    )
                values = counts.to_numpy(dtype=np.float64)
                if (
                    not np.isfinite(values).all()
                    or (values <= 0).any()
                    or (values != np.floor(values)).any()
                    or (values >= 2**63).any()
                ):
                    raise ValueError(
                        "Reference counts must be finite positive integers."
                    )
                observed = frame[name].value_counts(dropna=False, sort=False)
                available = counts.reindex(observed.index)
                if (
                    available.isna().any()
                    or (available.to_numpy() < observed.to_numpy()).any()
                ):
                    raise ValueError("Reference counts must cover the fitting rows.")
                self.counts_[name] = counts.astype("int64").copy()
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "counts_")
        if (
            not isinstance(frame, pd.DataFrame)
            or frame.columns.tolist() != self.feature_names_in_.tolist()
        ):
            raise ValueError("Frequency columns must match training columns and order.")
        encoded = np.empty((len(frame), self.n_features_in_), dtype=np.float64)
        for index, name in enumerate(self.feature_names_in_):
            counts = frame[name].map(self.counts_[name]).astype(np.float64).fillna(0)
            encoded[:, index] = np.log1p(counts.to_numpy(dtype=np.float64))
        return encoded

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "counts_")
        if input_features is not None and not np.array_equal(
            input_features, self.feature_names_in_
        ):
            raise ValueError("Frequency feature names must match training columns.")
        return np.asarray(
            [f"{name}__log_count" for name in self.feature_names_in_], dtype=object
        )
