"""Contract tests use manufactured data, never Kaggle test labels."""

import numpy as np
import pandas as pd
import pytest

from fraudlab.data import load_data, submission, validate_submission
from fraudlab.models import NeuralPreprocessor, clean_features
from fraudlab.features import transform_features
from fraudlab.splits import make_splits


def test_submission_requires_exact_ids_schema_and_probabilities():
    expected = pd.DataFrame({"id": [10, 11], "is_fraud": [0, 0]})
    valid = pd.DataFrame({"id": [10, 11], "is_fraud": [0.2, 0.8]})
    validate_submission(valid, expected)
    for invalid in (
        valid.iloc[::-1],
        valid.assign(id=[10, 10]),
        valid.iloc[:1],
        valid.iloc[:0],
        valid.assign(is_fraud=[0.2, np.nan]),
        valid.assign(is_fraud=[0.2, np.inf]),
        valid.assign(is_fraud=[0.2, 1.1]),
        valid.assign(extra=0),
    ):
        with pytest.raises(ValueError):
            validate_submission(invalid, expected)


def test_submission_reorders_predictions_by_sample_ids():
    ids = pd.Series([10, 11], name="id")
    sample = pd.DataFrame({"id": [11, 10], "is_fraud": [0, 0]})
    result = submission(ids, np.array([0.2, 0.8]), sample, "is_fraud")
    assert result["id"].tolist() == [11, 10]
    assert result["is_fraud"].tolist() == [0.8, 0.2]


def test_group_folds_never_share_entities():
    frame = pd.DataFrame(
        {"entity": np.repeat(np.arange(12), 4), "is_fraud": np.tile([0, 1], 24)}
    )
    folds = make_splits(frame, "is_fraud", "group", 3, 42, group_column="entity")
    seen = []
    for train, validation in folds:
        assert set(frame.iloc[train]["entity"]).isdisjoint(
            frame.iloc[validation]["entity"]
        )
        seen.extend(validation)
    assert sorted(seen) == list(range(len(frame)))


def test_time_folds_never_train_on_future_or_simultaneous_events():
    frame = pd.DataFrame(
        {"time": np.repeat(np.arange(12), 4), "is_fraud": np.tile([0, 1], 24)}
    )
    folds = make_splits(frame, "is_fraud", "time", 3, 42, time_column="time")
    for train, validation in folds:
        assert frame.iloc[train]["time"].max() < frame.iloc[validation]["time"].min()
    assert sum(len(validation) for _, validation in folds) < len(frame)


def test_time_folds_parse_dates_without_crossing_boundaries():
    frame = pd.DataFrame(
        {
            "time": np.repeat(pd.date_range("2026-01-01", periods=12).astype(str), 4),
            "is_fraud": np.tile([0, 1], 24),
        }
    )
    for train, validation in make_splits(
        frame, "is_fraud", "time", 3, 42, time_column="time"
    ):
        assert frame.iloc[train]["time"].max() < frame.iloc[validation]["time"].min()


def test_stratified_folds_predict_each_row_once_and_reject_undefined_auc():
    frame = pd.DataFrame({"is_fraud": [0, 1] * 12})
    folds = make_splits(frame, "is_fraud", "stratified", 3, 42)
    assert sorted(np.concatenate([validation for _, validation in folds])) == list(
        range(len(frame))
    )
    with pytest.raises(ValueError, match="class"):
        make_splits(
            pd.DataFrame({"is_fraud": [0] * 10 + [1]}), "is_fraud", "stratified", 3, 42
        )


def test_neural_preprocessing_uses_training_statistics_and_unknown_categories():
    training = clean_features(
        pd.DataFrame({"value": [0.0, 2.0, np.nan], "category": ["a", "b", "a"]})
    )
    held_out = clean_features(
        pd.DataFrame({"value": [1000.0], "category": ["never-seen"]})
    )
    preprocessing = NeuralPreprocessor().fit(training)
    numeric, categories = preprocessing.transform(held_out)
    assert preprocessing.scaler.mean_[0] == pytest.approx(1.0)
    assert numeric[0, 0] > 100
    assert categories[0, 0] == 0
    assert "never-seen" not in preprocessing.vocabularies["category"]


def test_loader_rejects_target_leakage_in_test_schema(tmp_path):
    pd.DataFrame({"id": [1, 2], "x": [0.1, 0.2], "is_fraud": [0, 1]}).to_csv(
        tmp_path / "train.csv", index=False
    )
    pd.DataFrame({"id": [3], "x": [0.3], "is_fraud": [1]}).to_csv(
        tmp_path / "test.csv", index=False
    )
    pd.DataFrame({"id": [3], "is_fraud": [0]}).to_csv(
        tmp_path / "sample_submission.csv", index=False
    )
    with pytest.raises(ValueError, match="schemas"):
        load_data(tmp_path)


def test_engineered_features_are_independent_of_target_and_batch_composition():
    frame = pd.DataFrame(
        {
            "amount_usd": [12.34, 50.0],
            "account_balance_usd": [100, 200],
            "time_of_day_hour": [23, 0],
            "day_of_week": [1, 2],
            "is_fraud": [0, 1],
        }
    )
    transformed = transform_features(frame, "digits")
    flipped = transform_features(frame.assign(is_fraud=1 - frame["is_fraud"]), "digits")
    assert transformed.drop(columns="is_fraud").equals(flipped.drop(columns="is_fraud"))
    assert transformed.iloc[:1].equals(transform_features(frame.iloc[:1], "digits"))
    assert transformed["amount_usd__cents_key"].tolist() == ["34", "0"]


def test_neural_target_encoding_never_uses_an_examples_own_unique_label():
    frame = pd.DataFrame(
        {"value": np.zeros(20), "category": [f"unique-{index}" for index in range(20)]}
    )
    labels = np.tile([0, 1], 10)
    preprocessing = NeuralPreprocessor(categorical_transform="target", smoothing=20)
    training, categories = preprocessing.fit_transform(frame, labels)
    assert np.allclose(training, 0)
    assert categories.shape == (20, 0)
    # Inference mappings use all training labels, which is deliberately different
    # from the internally cross-fitted values presented to the optimizer.
    inference, _ = preprocessing.transform(frame)
    assert not np.allclose(inference[:, -1], training[:, -1])


def test_feature_views_respect_custom_id_and_target_names():
    frame = pd.DataFrame(
        {"record_key": [1, 2], "amount_usd": [0, 1], "customer_age": [20, 30]}
    )
    transformed = transform_features(
        frame, "value_digits", excluded=("record_key", "amount_usd")
    )
    assert not any(
        name.startswith("record_key__") or name.startswith("amount_usd__")
        for name in transformed
    )
