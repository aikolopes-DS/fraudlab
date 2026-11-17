"""Column frequencies learned from training features, without target labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class FrequencyEncoder(TransformerMixin, BaseEstimator):
    """Encode each value as log1p of its training count; unknown values become zero."""

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
        self.counts_ = {
            name: frame[name].value_counts(dropna=False, sort=False)
            for name in self.feature_names_in_
        }
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
