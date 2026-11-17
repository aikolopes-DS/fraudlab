"""Eight fixed interaction hypotheses, with no learned selection or statistics.

Public-source inspiration: megayak's joint-key ideas as credited in goodpjw2008's
Part 12, version 16. This extraction is independently written. No G-statistic
screen, quantile fitting, third-party code or prediction arrays are used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

JOINTS = (
    ("foreign_ip", "is_foreign_transaction", "exact", "ip_country_mismatch", "exact"),
    ("category_risk10", "merchant_category", "exact", "merchant_risk_score", "floor10"),
    ("category_amount_log2", "merchant_category", "exact", "amount_usd", "log2"),
    (
        "scam_velocity5",
        "is_ai_generated_scam_attempt",
        "exact",
        "velocity_score",
        "floor5",
    ),
    ("newmerchant_weekday", "is_new_merchant", "exact", "day_of_week", "exact"),
    ("hour_weekday", "time_of_day_hour", "exact", "day_of_week", "exact"),
    ("age_cardyears", "customer_age", "floor10", "card_age_months", "floor12"),
    ("auth_channel", "auth_method", "exact", "channel", "exact"),
)
OUTPUT_NAMES = tuple(name + "__joint_key" for name, *_ in JOINTS)
MAX_VALUE = 2**53 - 1
# Exact float64 boundaries avoid log2(1+x) rounding up immediately below a tie.
LOG2_BOUNDARIES = np.asarray([2**power - 1 for power in range(54)], dtype=np.float64)
FORMULAS = {
    "exact": "unchanged string; numeric value normalized to float64",
    "floor5": "floor(float64(x)/5)",
    "floor10": "floor(float64(x)/10)",
    "floor12": "floor(float64(x)/12)",
    "log2": "floor(log2(1+x)); interval k starts at exact boundary 2**k-1",
}


def _numeric(series: pd.Series, *, allow_bool: bool = False) -> np.ndarray:
    if (
        not pd.api.types.is_numeric_dtype(series)
        or pd.api.types.is_complex_dtype(series)
        or (pd.api.types.is_bool_dtype(series) and not allow_bool)
    ):
        raise ValueError(f"{series.name} requires real numeric joint inputs.")
    values = series.to_numpy(dtype=np.float64, na_value=np.nan)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{series.name} requires finite nonnegative joint inputs.")
    if (values > MAX_VALUE).any():
        raise ValueError(f"{series.name} exceeds the supported joint integer range.")
    return np.where(values == 0, 0.0, values)


def _view(series: pd.Series, mode: str) -> list:
    if mode == "exact":
        if pd.api.types.is_numeric_dtype(series):
            return _numeric(series, allow_bool=True).tolist()
        values = series.tolist()
        if not all(isinstance(value, str) for value in values):
            raise ValueError(f"{series.name} requires nonmissing string categories.")
        return values
    values = _numeric(series)
    if mode == "log2":
        bins = np.searchsorted(LOG2_BOUNDARIES, values, side="right") - 1
    else:
        bins = np.floor(values / int(mode.removeprefix("floor")))
    return bins.astype(np.int64).tolist()


def joint_keys(frame: pd.DataFrame, excluded: tuple[str, ...] = ()) -> pd.DataFrame:
    """Return fixed categorical tuple keys; skip a pair if either field is excluded.

    Only the declared fields are read. IDs/targets/other columns have no effect.
    There is no fitted state, category vocabulary or dependence on other rows.
    """
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("Joint keys require unique DataFrame columns.")
    result = pd.DataFrame(index=frame.index)
    views = {}
    for name, left, left_mode, right, right_mode in JOINTS:
        if {left, right}.intersection(excluded):
            continue
        for column, mode in ((left, left_mode), (right, right_mode)):
            if column not in frame:
                raise ValueError(f"Missing joint input column: {column}")
            if (column, mode) not in views:
                views[column, mode] = _view(frame[column], mode)
        pairs = zip(views[left, left_mode], views[right, right_mode])
        result[name + "__joint_key"] = pd.Series(
            [repr(pair) for pair in pairs], index=frame.index, dtype=object
        )
    return result


def manifest() -> dict:
    from .bpe import manifest as bpe_manifest

    return {
        "algorithm": "bpe_digits_plus_eight_fixed_joint_categories",
        "base_feature_set": "bpe_digits",
        "bpe": bpe_manifest(),
        "joint_encoding": {
            "output_count": 8,
            "pairs": [
                {
                    "output": name + "__joint_key",
                    "left": left,
                    "left_view": FORMULAS[left_mode],
                    "right": right,
                    "right_view": FORMULAS[right_mode],
                }
                for name, left, left_mode, right, right_mode in JOINTS
            ],
            "category_representation": "repr((left, right)); strings escaped, numbers canonicalized as above; categorical labels, not ordinal magnitudes",
            "numeric_domain": "finite nonnegative float64 at most 2**53-1; negative zero normalized; bins use no cent rounding",
            "log2_boundaries": "2**k-1 for integer k=0..53, left-inclusive via searchsorted(side='right')-1",
            "excluded_fields": "Exclude the entire pair if either input is excluded",
            "selection": "Eight hypotheses fixed before fitting; public-source-informed exploratory selection; no G scan, label ranking or quantile fitting",
            "label_usage": "none in extraction; downstream TE/frequency/scaling remain fitted inside outer training only",
            "concept_source": {
                "url": "https://www.kaggle.com/code/goodpjw2008/s6e11-part-12-wide-stack-2nd-lm-lb-0-95298",
                "version": 16,
                "source_cell_zero_based": 74,
                "source_body_sha256": "835772526d7efcc70f771aa446690cb1956676a457f40b08d48760bbf3336baf",
                "credit": "megayak's joint-key ideas; independent extraction, no source code copied or executed",
            },
        },
    }
