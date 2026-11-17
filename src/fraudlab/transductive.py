"""Label-free categorical counts over official training and test features."""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from .models import clean_features, feature_types


def validate_config(config: dict) -> None:
    if not config.get("glm_transductive_frequency"):
        return
    if (
        not config.get("glm_frequency")
        or config.get("model") not in ("glm", "offset")
        or config.get("phase") != "exploratory_full_data_cv"
        or config.get("split") != "stratified"
        or config.get("folds") != 5
        or config.get("seed") != 42
        or config.get("holdout_fraction", 0)
        or config.get("original_data")
        or config.get("refit")
    ):
        raise ValueError(
            "Transductive frequency requires GLM/offset, glm_frequency and exploratory CV5/42 without original data, holdout or refit."
        )


def counts_sha256(counts: dict[str, pd.Series]) -> str:
    """Hash sorted UTF-8 category/count records with unambiguous JSON framing."""
    digest = hashlib.sha256()
    for name in sorted(counts):
        column = counts[name]
        digest.update((json.dumps([name], ensure_ascii=False) + "\n").encode("utf-8"))
        for key, value in column.sort_index().items():
            digest.update(
                (json.dumps([key, int(value)], ensure_ascii=False) + "\n").encode(
                    "utf-8"
                )
            )
    return digest.hexdigest()


def build_reference_counts(
    training: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    corpus_hashes: dict,
    excluded: tuple[str, ...] = ("id", "is_fraud"),
) -> tuple[dict[str, pd.Series], dict]:
    """Count one column at a time; never concatenate a million-row wide frame."""
    columns = [name for name in feature_columns if name not in excluded]
    if (
        training.empty
        or test.empty
        or not training.columns.is_unique
        or not test.columns.is_unique
        or len(columns) != len(set(columns))
        or not all(isinstance(name, str) for name in columns)
        or not set(columns).issubset(training)
        or not set(columns).issubset(test)
    ):
        raise ValueError(
            "Transductive counts require nonempty frames and matching feature columns."
        )
    # Zero-row selection preserves exactly feature_types' dtype semantics.
    _, categorical = feature_types(training.iloc[:0][columns])
    if not categorical:
        raise ValueError("Transductive frequency requires categorical features.")
    reference = {}
    for name in categorical:
        left = clean_features(training[[name]], [], [name])[name].value_counts(
            sort=False
        )
        right = clean_features(test[[name]], [], [name])[name].value_counts(sort=False)
        reference[name] = left.add(right, fill_value=0).astype("int64")
    metadata = {
        "scope": "transductive official train + test feature counts",
        "label_usage": "none",
        "rows": {
            "train": len(training),
            "test": len(test),
            "total": len(training) + len(test),
        },
        "corpus_hashes": dict(corpus_hashes),
        "columns": categorical,
        "counts_sha256": counts_sha256(reference),
        "normalization": "clean_features categorical string conversion and __MISSING__",
        "supervised_statistics": "target encoding, pooled target rates and all scalers fit on outer training only",
    }
    return reference, metadata
