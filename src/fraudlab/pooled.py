"""Hierarchical decimal/token target rates with honest inner cross-fitting.

Concept: megayak's token pooling, credited in goodpjw2008's Part 12 notebook
(version 16, gboost.py cell). This implementation is independently written;
no third-party code is copied or executed. The hypothesis concerns observed
numeric patterns, not a verified data-generating mechanism.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted

POOL2 = (
    "amount_usd",
    "hours_since_last_txn",
    "distance_from_home_km",
    "account_balance_usd",
)
POOL1 = ("velocity_score", "merchant_risk_score")
COLUMNS = POOL2 + POOL1
GROUPS = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9), (10, 11))
OUTPUT_NAMES = (
    tuple(f"{name}__pooled_decimal_rate" for name in POOL2)
    + tuple(f"{name}__pooled_first_decimal_rate" for name in POOL2)
    + tuple(f"{name}__pooled_decimal_rate" for name in POOL1)
    + tuple(f"{name}__pooled_last_integer_token_rate" for name in POOL1)
    + tuple(f"{name}__pooled_markov_rate" for name in POOL2)
)


def canonical_decimal(value: float) -> str:
    """Represent a finite nonnegative float64 without exponent or new rounding."""
    if np.iscomplexobj(value):
        raise ValueError("Decimal token inputs must be real numbers.")
    number = float(value)
    if not np.isfinite(number) or number < 0:
        raise ValueError("Decimal token inputs must be finite and nonnegative.")
    text = format(Decimal(str(number if number else 0.0)), "f")
    return text if "." in text else text + ".0"


def _keys(frame: pd.DataFrame) -> list[np.ndarray]:
    from .bpe import _encoding

    decimals, first, last, pairs = {}, {}, {}, {}
    tokenizer = _encoding()
    integer_tokens = {}
    for name in COLUMNS:
        series = frame[name]
        if (
            not pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_bool_dtype(series)
            or pd.api.types.is_complex_dtype(series)
        ):
            raise ValueError(f"{name} requires numeric decimal token inputs.")
        values = series.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{name} requires finite nonnegative values.")
        unique, inverse = np.unique(values, return_inverse=True)
        parts = [canonical_decimal(value).split(".", 1) for value in unique]
        integer = [part[0] for part in parts]
        for text in set(integer) - integer_tokens.keys():
            integer_tokens[text] = tokenizer.encode_ordinary(" " + text)[-1]
        dec = [part[1] for part in parts]
        token = [integer_tokens[text] for text in integer]
        decimals[name] = np.asarray(dec, dtype=object)[inverse]
        first[name] = np.asarray([text[0] for text in dec], dtype=object)[inverse]
        last[name] = np.asarray(token, dtype=np.int32)[inverse]
        if name in POOL2:
            pairs[name] = np.asarray(
                [f"{key}:{text}" for key, text in zip(token, dec)], dtype=object
            )[inverse]
    return (
        [decimals[name] for name in POOL2]
        + [first[name] for name in POOL2]
        + [decimals[name] for name in POOL1]
        + [last[name] for name in POOL1]
        + [pairs[name] for name in POOL2]
    )


def manifest(random_state: int = 42) -> dict:
    from .bpe import ENCODING, ENCODING_FILES, TOKENIZER_VERSION

    return {
        "algorithm": "hierarchical_pooled_decimal_token_rates",
        "input_columns": list(COLUMNS),
        "output_features": list(OUTPUT_NAMES),
        "output_count": 16,
        "groups": [
            {"fields": list(POOL2), "key": "complete decimal text"},
            {"fields": list(POOL2), "key": "first decimal digit"},
            {"fields": list(POOL1), "key": "complete decimal text"},
            {"fields": list(POOL1), "key": "last GPT-2 integer token ID"},
        ],
        "markov_fields": list(POOL2),
        "markov_key": "last GPT-2 integer token ID + complete decimal text",
        "smoothing": {
            "pooled_to_prior": 50.0,
            "field_to_pool": 2000.0,
            "markov_to_field_decimal": 200.0,
        },
        "crossfit": {
            "n_splits": 5,
            "shuffle": True,
            "random_state": random_state,
            "seed_scope": "model seed; CV runner passes run seed + outer fold index",
            "prior": "inner-training mean only",
            "inference": "outer-training mappings only",
        },
        "canonical_decimal": "Decimal(str(float64)), fixed notation, at least .0; normalize negative zero; no cent rounding or decimal padding",
        "integer_token": "last token of one ASCII space + canonical integer part",
        "tokenizer": {
            "tiktoken": TOKENIZER_VERSION,
            "encoding": ENCODING,
            "encoding_files": list(ENCODING_FILES),
        },
        "pipeline": "rates -> clipped log_odds -> outer-training StandardScaler",
        "concept_sources": [
            {
                "author": "megayak",
                "url": "https://www.kaggle.com/code/megayak/s6e11-per-value-digit-te-shallow-tree-stack",
                "scope": "token pooling concept credited as v7 in the Part 12 source",
            },
            {
                "author": "goodpjw2008",
                "url": "https://www.kaggle.com/code/goodpjw2008/s6e11-part-12-wide-stack-2nd-lm-lb-0-95298",
                "version": 16,
                "cell": 74,
                "source_body_sha256": "835772526d7efcc70f771aa446690cb1956676a457f40b08d48760bbf3336baf",
            },
        ],
        "implementation": "independently written; concept credit only, no source code copied or executed",
        "interpretation": "numeric-pattern hypothesis; generator not verified",
    }


class PooledTokenEncoder(TransformerMixin, BaseEstimator):
    """Pool target rates across fields, then shrink field and conditional rates.

    fit_transform returns inner-held-out rates and retains full-training maps
    only for later transform calls. IDs and any extra columns are ignored.
    """

    def __init__(
        self,
        *,
        n_splits=5,
        random_state=42,
        pool_smoothing=50.0,
        field_smoothing=2000.0,
        markov_smoothing=200.0,
    ):
        self.n_splits = n_splits
        self.random_state = random_state
        self.pool_smoothing = pool_smoothing
        self.field_smoothing = field_smoothing
        self.markov_smoothing = markov_smoothing

    def _validate(self, frame, y=None):
        if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
            raise ValueError("PooledTokenEncoder requires unique DataFrame columns.")
        missing = set(COLUMNS) - set(frame)
        if missing:
            raise ValueError(f"Missing pooled-token columns: {sorted(missing)}")
        if y is None:
            return None
        labels = np.asarray(y, dtype=np.float64)
        if (
            labels.shape != (len(frame),)
            or not len(labels)
            or not np.isin(labels, [0, 1]).all()
        ):
            raise ValueError("PooledTokenEncoder requires one binary label per row.")
        if (
            not isinstance(self.n_splits, int)
            or isinstance(self.n_splits, bool)
            or self.n_splits < 2
        ):
            raise ValueError("n_splits must be an integer of at least two.")
        for value in (self.pool_smoothing, self.field_smoothing, self.markov_smoothing):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Pooled-token smoothing values must be positive.")
        return labels

    def _learn(self, keys, labels, rows):
        prior = float(labels[rows].mean())
        fields, pools, markov = [], [], []
        for group in GROUPS:
            tables = [
                pd.DataFrame({"key": keys[column][rows], "y": labels[rows]})
                .groupby("key", sort=False)["y"]
                .agg(["sum", "count"])
                for column in group
            ]
            pooled = pd.concat(tables).groupby(level=0, sort=False).sum()
            rate = (pooled["sum"] + self.pool_smoothing * prior) / (
                pooled["count"] + self.pool_smoothing
            )
            pools.append(rate)
            for table in tables:
                fields.append(
                    (table["sum"] + self.field_smoothing * rate.reindex(table.index))
                    / (table["count"] + self.field_smoothing)
                )
        for column in range(12, 16):
            markov.append(
                pd.DataFrame({"key": keys[column][rows], "y": labels[rows]})
                .groupby("key", sort=False)["y"]
                .agg(["sum", "count"])
            )
        return {"prior": prior, "pools": pools, "fields": fields, "markov": markov}

    def _encode(self, keys, maps, rows):
        result = np.empty((len(rows), 16), dtype=np.float64)
        for index, group in enumerate(GROUPS):
            for column in group:
                query = pd.Series(keys[column][rows], copy=False)
                fallback = query.map(maps["pools"][index]).fillna(maps["prior"])
                result[:, column] = query.map(maps["fields"][column]).fillna(fallback)
        for index in range(4):
            counts = maps["markov"][index].reindex(keys[index + 12][rows], fill_value=0)
            rate = (
                counts["sum"].to_numpy() + self.markov_smoothing * result[:, index]
            ) / (counts["count"].to_numpy() + self.markov_smoothing)
            result[:, index + 12] = np.where(
                counts["count"].to_numpy() == 0, result[:, index], rate
            )
        return result

    def _retain(self, maps):
        self.maps_ = maps
        self.feature_names_in_ = np.asarray(COLUMNS, dtype=object)
        self.n_features_in_ = len(COLUMNS)

    def fit(self, frame, y):
        labels = self._validate(frame, y)
        keys = _keys(frame)
        self._retain(self._learn(keys, labels, np.arange(len(frame))))
        return self

    def fit_transform(self, frame, y, **fit_params):
        labels = self._validate(frame, y)
        if (
            min(np.count_nonzero(labels == 0), np.count_nonzero(labels == 1))
            < self.n_splits
        ):
            raise ValueError("Cross-fitting requires at least n_splits rows per class.")
        keys = _keys(frame)
        result = np.empty((len(frame), 16), dtype=np.float64)
        splitter = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        for training, validation in splitter.split(np.zeros(len(frame)), labels):
            maps = self._learn(keys, labels, training)
            result[validation] = self._encode(keys, maps, validation)
        self._retain(self._learn(keys, labels, np.arange(len(frame))))
        return result

    def transform(self, frame):
        check_is_fitted(self, "maps_")
        self._validate(frame)
        return self._encode(_keys(frame), self.maps_, np.arange(len(frame)))

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "maps_")
        if input_features is not None and list(input_features) != list(COLUMNS):
            raise ValueError(
                "Pooled-token feature names must match the six input columns."
            )
        return np.asarray(OUTPUT_NAMES, dtype=object)
