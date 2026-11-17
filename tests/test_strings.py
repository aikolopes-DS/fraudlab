"""Storage sharing must preserve legacy features and fitted predictions exactly."""

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from fraudlab import features
from fraudlab.models import make_glm_pipeline
from fraudlab.scores import blend_scores
from fraudlab.strings import categorical_strings


@pytest.mark.parametrize(
    "values",
    [
        np.array([-12345678, 0, 12345678, -12345678], dtype=np.int64),
        np.array([-(2**63), 2**63 - 1, -(2**63)], dtype=np.int64),
        np.array([0, 2**64 - 1, 2**64 - 1], dtype=np.uint64),
        np.array(["-1:99", "3388:35", "-1:99"]),
        np.array([], dtype=np.int32),
    ],
)
def test_shared_object_strings_preserve_text_and_identity(values):
    actual = categorical_strings(values)
    expected = values.astype(str).astype(object)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == object and all(type(value) is str for value in actual)
    for value in np.unique(values):
        repeated = actual[values == value]
        assert all(item is repeated[0] for item in repeated)
    if values.dtype.kind in "iu":
        np.testing.assert_array_equal(actual, pd.Series(values).astype(str).to_numpy())
    with pytest.raises(ValueError, match="integer or Unicode"):
        categorical_strings(np.array([1.0, np.nan]))


def source_frame(size):
    values = np.resize(
        [0.0, 0.005, 0.015, 9.999, 61.16, 1269.35, 99701.39, 61.16], size
    )
    frame = pd.DataFrame({name: values for name in features.CONTINUOUS})
    frame["time_of_day_hour"] = np.arange(size) % 24
    frame["day_of_week"] = np.arange(size) % 7
    frame["customer_age"] = 20 + np.arange(size) % 50
    frame["card_age_months"] = np.arange(size) % 36
    frame["cvv_retry_count"] = np.arange(size) % 3
    frame["txn_count_last_24h"] = np.arange(size) % 10
    for name in (
        "is_foreign_transaction",
        "ip_country_mismatch",
        "is_new_merchant",
        "is_ai_generated_scam_attempt",
    ):
        frame[name] = np.arange(size) % 2
    frame["auth_method"] = np.resize(["a',b", "pin"], size)
    frame["channel"] = np.resize(["online", "store"], size)
    frame["merchant_category"] = np.resize(["retail", "food"], size)
    # These raw numeric string views must retain pandas' formatting unchanged.
    frame["raw_float"] = pd.Series(
        np.resize([1.0, -0.0, 1e-9, np.nan], size), dtype="Float64"
    )
    frame["raw_nullable_integer"] = pd.Series(
        np.resize([1, 2, None, 1], size), dtype="Int64"
    )
    frame["id"] = np.arange(size)
    frame["is_fraud"] = np.arange(size) % 2
    return frame


def legacy_transform(frame, feature_set, monkeypatch):
    from fraudlab import bpe

    # Frozen conversion formulas used before the storage-only implementation fix.
    with monkeypatch.context() as legacy:
        legacy.setattr(
            features,
            "categorical_strings",
            lambda values: pd.Series(values).astype(str).to_numpy(),
        )
        legacy.setattr(
            bpe, "categorical_strings", lambda values: np.asarray(values).astype(str)
        )
        return features.transform_features(frame, feature_set)


@pytest.mark.parametrize(
    "feature_set",
    [
        "digits",
        "value_digits",
        "paired_digits",
        "bpe_digits",
        "token_digits",
        "joint_bpe",
    ],
)
def test_features_equal_legacy_values_dtypes_order_and_index(feature_set, monkeypatch):
    pytest.importorskip("tiktoken")
    source = source_frame(24).set_axis(np.resize([9, 9, 81, 4], 24))
    legacy = legacy_transform(source, feature_set, monkeypatch)
    actual = features.transform_features(source, feature_set)
    pd.testing.assert_frame_equal(actual, legacy, check_exact=True)
    keys = [name for name in actual if name.endswith("_key")]
    assert all(actual[name].dtype == object for name in keys)
    # A repeated multi-character amount key and the combined BPE key share objects.
    value = actual["amount_usd__value_key"].to_numpy()
    assert value[4] is value[7]
    if feature_set in ("bpe_digits", "joint_bpe"):
        combined = actual["amount_usd__gpt2_second_cents_key"].to_numpy()
        assert combined[4] is combined[7]
    pd.testing.assert_series_equal(actual["raw_float"], source["raw_float"])


def test_two_glm_fits_and_probability_blend_equal_legacy_bit_for_bit(monkeypatch):
    pytest.importorskip("tiktoken")
    raw = source_frame(320).drop(
        columns=["id", "is_fraud", "raw_float", "raw_nullable_integer"]
    )
    labels = np.tile([0, 1], 160)
    old = legacy_transform(raw, "bpe_digits", monkeypatch)
    new = features.transform_features(raw, "bpe_digits")
    config = {
        "logistic_c": 0.3,
        "target_smoothing": 50.0,
        "glm_inner_folds": 5,
        "glm_frequency": True,
        "glm_pooled_tokens": True,
    }
    predictions = []
    coefficients = []
    for encoded in (old, new):
        glm = make_glm_pipeline(encoded.iloc[:240], config, 42)
        with threadpool_limits(limits=1):
            glm.fit(encoded.iloc[:240], labels[:240])
            predictions.append(glm.predict_proba(encoded.iloc[240:])[:, 1])
        coefficients.append(glm.named_steps["classifier"].coef_)
    np.testing.assert_array_equal(coefficients[0], coefficients[1])
    np.testing.assert_array_equal(predictions[0], predictions[1])
    baseline = np.linspace(0.05, 0.95, len(predictions[0]))
    old_blend = blend_scores([baseline, predictions[0]], [0.9, 0.1])
    new_blend = blend_scores([baseline, predictions[1]], [0.9, 0.1])
    np.testing.assert_array_equal(old_blend, new_blend)
