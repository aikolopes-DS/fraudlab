"""Input contracts and aggregate inspection. Raw data stays local."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def validate_ids(frame: pd.DataFrame, id_column: str) -> None:
    if (
        frame.empty
        or id_column not in frame
        or frame[id_column].isna().any()
        or frame[id_column].duplicated().any()
    ):
        raise ValueError(
            f"{id_column}: rows and IDs must exist; IDs must be non-null and unique."
        )


def validate_submission(
    frame: pd.DataFrame,
    expected: pd.DataFrame,
    id_column: str = "id",
    target: str = "is_fraud",
) -> None:
    if list(frame.columns) != [id_column, target]:
        raise ValueError(
            f"Submission columns must be exactly {[id_column, target]} in that order."
        )
    validate_ids(frame, id_column)
    validate_ids(expected, id_column)
    if frame[id_column].tolist() != expected[id_column].tolist():
        raise ValueError(
            "Submission IDs must match the expected IDs and their order exactly."
        )
    try:
        probabilities = frame[target].to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Submission probabilities must be numeric.") from error
    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError(
            "Submission probabilities must be finite values between 0 and 1."
        )


def load_data(
    directory: Path, id_column: str = "id", target: str = "is_fraud"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(directory / "train.csv")
    test = pd.read_csv(directory / "test.csv")
    sample = pd.read_csv(directory / "sample_submission.csv")
    if train.empty or test.empty:
        raise ValueError("Training and test data must not be empty.")
    validate_ids(train, id_column)
    validate_ids(test, id_column)
    if set(train[id_column]) & set(test[id_column]):
        raise ValueError("Training and test row IDs must be disjoint.")
    if (
        target not in train
        or train[target].isna().any()
        or set(train[target].unique()) != {0, 1}
    ):
        raise ValueError(
            f"{target}: expected a non-null binary target containing both 0 and 1."
        )
    features = [column for column in train if column not in (id_column, target)]
    if not features or set(test.columns) != {id_column, *features}:
        raise ValueError(
            "Training and test feature schemas must match, excluding target and ID."
        )
    if list(sample.columns) != [id_column, target]:
        raise ValueError("sample_submission.csv has an unexpected schema.")
    validate_ids(sample, id_column)
    if set(sample[id_column]) != set(test[id_column]):
        raise ValueError("Sample submission IDs must cover the test IDs exactly.")
    return train, test[[id_column, *features]], sample


def submission(
    test_ids: pd.Series, predictions: np.ndarray, sample: pd.DataFrame, target: str
) -> pd.DataFrame:
    values = pd.Series(np.asarray(predictions), index=test_ids)
    result = sample.copy()
    result[target] = values.reindex(sample[test_ids.name]).to_numpy()
    validate_submission(result, sample, test_ids.name, target)
    return result


def inspect_data(
    directory: Path, id_column: str = "id", target: str = "is_fraud"
) -> dict:
    train, test, _ = load_data(directory, id_column, target)
    features = [column for column in test if column != id_column]
    columns = {}
    for name in features:
        columns[name] = {
            "dtype": str(train[name].dtype),
            "train_missing": int(train[name].isna().sum()),
            "test_missing": int(test[name].isna().sum()),
            "train_unique": int(train[name].nunique(dropna=False)),
            "test_unique": int(test[name].nunique(dropna=False)),
        }
    duplicate_mask = train.duplicated(features, keep=False)
    return {
        "rows": {"train": len(train), "test": len(test)},
        "target": {
            "name": target,
            "positive_rate": float(train[target].mean()),
            "counts": {str(k): int(v) for k, v in train[target].value_counts().items()},
        },
        "features": columns,
        "duplicate_feature_rows": int(duplicate_mask.sum()),
        "train_test_feature_overlap": int(
            pd.util.hash_pandas_object(train[features], index=False)
            .isin(pd.util.hash_pandas_object(test[features], index=False))
            .sum()
        ),
        "potential_split_columns": [
            name
            for name in features
            if any(
                word in name.lower()
                for word in (
                    "date",
                    "time",
                    "customer",
                    "account",
                    "user",
                    "card_id",
                    "merchant_id",
                )
            )
        ],
        "validation_warning": "Inspect time, entities and duplicate rows before choosing a split. Candidate names alone do not establish the required split.",
        "data_hashes": {
            name: fingerprint(directory / name)
            for name in ("train.csv", "test.csv", "sample_submission.csv")
        },
    }
