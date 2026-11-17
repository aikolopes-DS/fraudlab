"""Frequency encoding must use training support and retain its column contract."""

import joblib
import numpy as np
import pandas as pd
import pytest

from fraudlab.frequency import FrequencyEncoder


def test_counts_are_local_to_training_and_each_column():
    training = pd.DataFrame({"key": ["a", "a", "b"], "other": ["a", "b", "b"]})
    validation = pd.DataFrame({"key": ["a", "unseen"], "other": ["b", "a"]})
    encoder = FrequencyEncoder().fit(training, y=[0, 1, 0])
    expected = np.log1p([[2, 2], [0, 1]])
    np.testing.assert_allclose(encoder.transform(validation), expected)
    repeated = pd.concat([validation] * 20, ignore_index=True)
    np.testing.assert_allclose(encoder.transform(repeated)[:2], expected)
    np.testing.assert_array_equal(
        encoder.get_feature_names_out(), ["key__log_count", "other__log_count"]
    )
    assert encoder.counts_["key"]["a"] == 2
    categorical = pd.DataFrame(
        {
            "key": pd.Categorical(["a", None], categories=["a", "b"]),
            "other": pd.Categorical(["b", "a"], categories=["a", "b"]),
        }
    )
    np.testing.assert_allclose(encoder.transform(categorical), expected)


def test_reload_preserves_counts_unknowns_and_feature_names(tmp_path):
    training = pd.DataFrame({"key": ["a", "a", "b", None]})
    inference = pd.DataFrame({"key": [None, "b", "unknown"]})
    encoder = FrequencyEncoder().fit(training)
    path = tmp_path / "frequency.joblib"
    joblib.dump(encoder, path)
    restored = joblib.load(path)
    np.testing.assert_allclose(restored.transform(inference), np.log1p([[1], [1], [0]]))
    np.testing.assert_allclose(
        restored.transform(inference), encoder.transform(inference)
    )
    np.testing.assert_array_equal(
        restored.get_feature_names_out(["key"]), ["key__log_count"]
    )


def test_changed_column_order_is_rejected():
    frame = pd.DataFrame({"left": ["a", "a"], "right": ["x", "y"]})
    encoder = FrequencyEncoder().fit(frame)
    with pytest.raises(ValueError, match="columns and order"):
        encoder.transform(frame[["right", "left"]])
    with pytest.raises(ValueError, match="feature names"):
        encoder.get_feature_names_out(["right", "left"])
