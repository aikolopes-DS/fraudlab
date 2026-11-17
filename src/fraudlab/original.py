"""Validate explicitly pinned original data for outer-training-only augmentation."""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .data import fingerprint, validate_ids

NUMERIC = (
    "amount_usd",
    "hours_since_last_txn",
    "txn_count_last_24h",
    "distance_from_home_km",
    "card_age_months",
    "customer_age",
    "account_balance_usd",
    "cvv_retry_count",
    "velocity_score",
    "time_of_day_hour",
    "day_of_week",
    "merchant_risk_score",
    "prior_disputes",
)
NOMINAL = ("merchant_category", "card_type", "auth_method", "channel", "device_type")
FLAGS = (
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
)
COLUMNS = NUMERIC + NOMINAL + FLAGS
AUDITED_SHA256 = "4e32b1ad4880cae452d1f2e5aa53224d38eb913c185455c00a3759df5eb3da61"
SOURCE_URL = "https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026/versions/5"


def validate_config(config: dict) -> None:
    path, digest = config.get("original_data"), config.get("original_sha256")
    if not path:
        if (
            digest
            or config.get("original_id_column", "transaction_id") != "transaction_id"
        ):
            raise ValueError("Original-data options require --original-data.")
        return
    if not isinstance(digest, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", digest):
        raise ValueError("--original-data requires a valid --original-sha256.")
    if (
        config.get("phase") != "exploratory_full_data_cv"
        or config.get("split") != "stratified"
        or config.get("holdout_fraction", 0)
        or config.get("folds") != 5
        or config.get("seed") != 42
        or config.get("refit")
    ):
        raise ValueError(
            "Original data requires exploratory stratified CV5/42, no holdout/refit."
        )
    if config.get("model") in ("catboost", "lightgbm", "mlp") and not config.get(
        "fixed_training"
    ):
        raise ValueError(
            "Original augmentation cannot use outer-validation early stopping."
        )


def verify_unchanged(path: Path, expected_sha256: str) -> None:
    if fingerprint(path) != expected_sha256.lower():
        raise ValueError(
            "Original CSV bytes changed; SHA-256 no longer matches the pin."
        )


def load_original(
    path: Path,
    expected_sha256: str,
    competition: pd.DataFrame,
    test: pd.DataFrame,
    *,
    original_id_column="transaction_id",
    id_column="id",
    target="is_fraud",
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Return only the 24 ordered features and labels; never carry source IDs."""
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[a-fA-F0-9]{64}", expected_sha256
    ):
        raise ValueError("A valid original SHA-256 pin is required.")
    if original_id_column == target or {original_id_column, id_column, target} & set(
        COLUMNS
    ):
        raise ValueError("Original IDs and target must not be feature columns.")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256.lower():
        raise ValueError("Original CSV SHA-256 does not match the expected pin.")
    original = pd.read_csv(BytesIO(raw))
    if set(original) != {*COLUMNS, original_id_column, target}:
        raise ValueError(
            "Original schema must contain exactly 24 common features, ID and target."
        )
    order = [name for name in competition if name not in (id_column, target)]
    if (
        len(order) != 24
        or set(order) != set(COLUMNS)
        or set(test) != {*COLUMNS, id_column}
    ):
        raise ValueError(
            "Competition schema must contain the same 24 original features."
        )
    validate_ids(original, original_id_column)
    if original.isna().any().any():
        raise ValueError("Original CSV must not contain missing values.")
    labels = original[target]
    if not pd.api.types.is_numeric_dtype(labels) or set(labels.unique()) != {0, 1}:
        raise ValueError("Original target must contain binary numeric labels 0 and 1.")
    for name in order:
        values = original[name]
        if values.dtype != competition[name].dtype or values.dtype != test[name].dtype:
            raise ValueError(f"Original dtype differs from competition: {name}")
        if name in NUMERIC:
            if (
                not pd.api.types.is_numeric_dtype(values)
                or pd.api.types.is_bool_dtype(values)
                or pd.api.types.is_complex_dtype(values)
                or not np.isfinite(values.to_numpy()).all()
            ):
                raise ValueError(f"Original numeric feature is invalid: {name}")
        elif name in FLAGS:
            if not pd.api.types.is_bool_dtype(values):
                raise ValueError(f"Original flag must be boolean: {name}")
        elif not values.map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"Original nominal feature must contain strings: {name}")
    features = original[order].copy()
    if features.duplicated().any():
        raise ValueError("Original CSV contains duplicate feature rows.")
    hashes = pd.util.hash_pandas_object(features, index=False)
    for name, reference in (("train", competition), ("test", test)):
        reference_features = reference[order]
        candidates = reference_features.loc[
            pd.util.hash_pandas_object(reference_features, index=False).isin(hashes)
        ]
        if (
            not candidates.empty
            and not features.merge(candidates.drop_duplicates(), on=order).empty
        ):
            raise ValueError(f"Original features overlap competition {name} rows.")
    audited = digest == AUDITED_SHA256
    metadata = {
        "source": str(path),
        "sha256": digest,
        "source_url": SOURCE_URL if audited else None,
        "version": 5 if audited else None,
        "license": "CC0-1.0" if audited else None,
        "provenance": (
            "audited public original"
            if audited
            else "explicit local CSV; public provenance not asserted"
        ),
        "rows": len(original),
        "positives": int(labels.sum()),
        "usage": "outer-training only",
        "excluded_columns": [original_id_column],
        "competition_validation_rows_added": 0,
        "competition_oof_rows_added": 0,
        "feature_overlap": "zero exact feature rows against competition train and test",
    }
    verify_unchanged(path, digest)
    return features, labels.to_numpy(dtype=np.int64), metadata
