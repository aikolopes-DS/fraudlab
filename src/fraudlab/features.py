"""Small deterministic hypotheses; every feature set is compared by ablation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .strings import categorical_strings

CONTINUOUS = (
    "amount_usd",
    "hours_since_last_txn",
    "distance_from_home_km",
    "account_balance_usd",
    "velocity_score",
    "merchant_risk_score",
)


def transform_features(
    frame: pd.DataFrame,
    feature_set: str = "raw",
    excluded: tuple[str, ...] = ("id", "is_fraud"),
) -> pd.DataFrame:
    if feature_set == "joint_bpe":
        from .joints import joint_keys

        keys = joint_keys(frame, excluded)
        base = transform_features(frame, "bpe_digits", excluded)
        return pd.concat([base, keys], axis=1)
    if feature_set == "token_digits":
        from .tokens import token_keys

        keys = token_keys(frame, excluded)
        base = transform_features(frame, "value_digits", excluded)
        return pd.concat([base, keys], axis=1)
    if feature_set == "bpe_digits":
        from .bpe import bpe_keys

        base = transform_features(frame, "value_digits", excluded)
        return pd.concat([base, bpe_keys(frame, excluded)], axis=1)
    if feature_set == "raw":
        return frame.copy()
    if feature_set not in ("behavior", "digits", "value_digits", "paired_digits"):
        raise ValueError(f"Unknown feature set: {feature_set}")
    result = frame.copy()
    for name in CONTINUOUS:
        if name in result and name not in excluded:
            values = result[name].to_numpy(dtype=float)
            result[name + "__log1p"] = np.sign(values) * np.log1p(np.abs(values))
    for name, period in (("time_of_day_hour", 24), ("day_of_week", 7)):
        if name in result and name not in excluded:
            angle = 2 * np.pi * result[name] / period
            result[name + "__sin"] = np.sin(angle)
            result[name + "__cos"] = np.cos(angle)
    if {"amount_usd", "account_balance_usd"}.issubset(result) and not {
        "amount_usd",
        "account_balance_usd",
    }.intersection(excluded):
        result["amount_balance_ratio"] = result["amount_usd"] / (
            np.abs(result["account_balance_usd"]) + 1
        )
    if feature_set in ("digits", "value_digits", "paired_digits"):
        # These views test synthetic rounding patterns; they are not banking rules.
        for name in CONTINUOUS:
            if name not in result or name in excluded:
                continue
            values = np.abs(result[name].to_numpy(dtype=float))
            if not np.isfinite(values).all():
                raise ValueError("Digit views require finite numeric inputs.")
            cents = np.rint(values * 100).astype(np.int64)
            magnitude = np.floor(np.log10(np.maximum(values, 1e-12))).astype(int)
            leading = np.floor(values / np.power(10.0, magnitude)).astype(int)
            result[name + "__value_key"] = categorical_strings(cents)
            result[name + "__cents_key"] = categorical_strings(cents % 100)
            result[name + "__leading_key"] = categorical_strings(leading)
            if feature_set in ("value_digits", "paired_digits"):
                leading_two = np.floor(values / np.power(10.0, magnitude - 1)).astype(
                    int
                )
                result[name + "__integer_key"] = categorical_strings(cents // 100)
                result[name + "__tenths_key"] = categorical_strings(cents // 10)
                result[name + "__leading_two_key"] = categorical_strings(leading_two)
        if feature_set in ("value_digits", "paired_digits"):
            for name in frame.select_dtypes(include="number"):
                if name not in CONTINUOUS and name not in excluded:
                    result[name + "__value_key"] = frame[name].astype(str)
    if feature_set == "paired_digits":
        # Four domain/audit hypotheses fixed before CV; no label-based pair search.
        pairs = (
            ("cvv_retry_count", None, "auth_method", None),
            ("velocity_score", 5, "distance_from_home_km", 5),
            ("velocity_score", 5, "txn_count_last_24h", 3),
            ("velocity_score", 5, "hours_since_last_txn", 2),
        )
        for left, left_width, right, right_width in pairs:
            if not {left, right}.issubset(frame) or {left, right}.intersection(
                excluded
            ):
                continue
            left_key = frame[left] // left_width if left_width else frame[left]
            right_key = frame[right] // right_width if right_width else frame[right]
            # Tuple text escapes embedded separators and keeps the pair unambiguous.
            result[f"{left}__{right}__pair"] = pd.Series(
                list(zip(left_key, right_key)), index=frame.index
            ).astype(str)
    return result
