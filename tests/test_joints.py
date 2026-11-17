"""Fixed-bin and categorical interaction contracts, with synthetic persistence."""

import ast
import json
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from fraudlab.features import CONTINUOUS, transform_features
from fraudlab.joints import MAX_VALUE, OUTPUT_NAMES, joint_keys, manifest


def frame(size=4):
    return pd.DataFrame(
        {
            "is_foreign_transaction": [0] * size,
            "ip_country_mismatch": [1] * size,
            "merchant_category": ["grocery"] * size,
            "merchant_risk_score": [21.0] * size,
            "amount_usd": [16.0] * size,
            "is_ai_generated_scam_attempt": [0] * size,
            "velocity_score": [11.0] * size,
            "is_new_merchant": [1] * size,
            "day_of_week": [2] * size,
            "time_of_day_hour": [9] * size,
            "customer_age": [39] * size,
            "card_age_months": [23] * size,
            "auth_method": ["pin"] * size,
            "channel": ["online"] * size,
        }
    )


def pairs(keys, name):
    return [ast.literal_eval(value) for value in keys[name + "__joint_key"]]


def test_eight_predeclared_pairs_have_expected_categories():
    result = joint_keys(frame(1))
    assert tuple(result) == OUTPUT_NAMES and result.shape == (1, 8)
    assert all(dtype == object for dtype in result.dtypes)
    assert [ast.literal_eval(value) for value in result.iloc[0]] == [
        (0, 1),
        ("grocery", 2),
        ("grocery", 4),
        (0, 2),
        (1, 2),
        (9, 2),
        (3, 1),
        ("pin", "online"),
    ]
    assert not any("cvv" in name or "distance" in name for name in result)


def test_fixed_width_bins_floor_and_do_not_round_halfway_values():
    source = frame(5)
    source["merchant_risk_score"] = [np.nextafter(10.0, 0), 10, 10.5, 19.99, 20]
    source["velocity_score"] = [np.nextafter(5.0, 0), 5, 7.5, 9.99, 10]
    source["customer_age"] = [np.nextafter(20.0, 0), 20, 25, 29.99, 30]
    source["card_age_months"] = [np.nextafter(12.0, 0), 12, 18, 23.99, 24]
    result = joint_keys(source)
    assert [right for _, right in pairs(result, "category_risk10")] == [0, 1, 1, 1, 2]
    assert [right for _, right in pairs(result, "scam_velocity5")] == [0, 1, 1, 1, 2]
    assert pairs(result, "age_cardyears") == [(1, 0), (2, 1), (2, 1), (2, 1), (3, 2)]


def test_log2_bins_distinguish_predecessor_exact_boundary_and_successor():
    values, expected = [0.0], [0]
    for power in (1, 2, 3, 4, 20, 53):
        boundary = float(2**power - 1)
        values.extend([np.nextafter(boundary, 0), boundary])
        expected.extend([power - 1, power])
        if power < 53:
            values.append(np.nextafter(boundary, np.inf))
            expected.append(power)
    source = frame(len(values)).assign(amount_usd=values)
    assert [
        right for _, right in pairs(joint_keys(source), "category_amount_log2")
    ] == expected


def test_tuple_repr_escapes_categories_and_numeric_zero_is_canonical():
    source = frame(3)
    source["auth_method"] = ["a', 'b", "a", "a\\'b"]
    source["channel"] = ["c", "b', 'c", "c\\d"]
    source["is_foreign_transaction"] = [-0.0, 0.0, 0.0]
    result = joint_keys(source)
    assert len(set(result.auth_channel__joint_key)) == 3
    assert pairs(result, "auth_channel") == list(
        zip(source.auth_method, source.channel)
    )
    assert result.foreign_ip__joint_key.nunique() == 1
    boolean = joint_keys(source.assign(is_foreign_transaction=False))
    pd.testing.assert_series_equal(
        boolean.foreign_ip__joint_key, result.foreign_ip__joint_key
    )


@pytest.mark.parametrize(
    "value", [-1, np.nan, np.inf, -np.inf, MAX_VALUE * 2, "5", True, 1 + 0j]
)
def test_bins_reject_invalid_numeric_input_without_silent_coercion(value):
    source = frame(1).assign(merchant_risk_score=value)
    with pytest.raises(ValueError, match="joint"):
        joint_keys(source)


def test_excluded_inputs_labels_ids_empty_frames_and_batches_are_independent():
    source = frame(4).set_axis([8, 8, 41, 3])
    source["id"], source["is_fraud"] = range(4), [1, 0, 1, 0]
    original = source.copy(deep=True)
    keys = joint_keys(source)
    changed = source.assign(id=-1, is_fraud=[0, 1, 0, 1], unknown=np.nan)
    pd.testing.assert_frame_equal(joint_keys(changed), keys)
    pd.testing.assert_frame_equal(source, original)
    pd.testing.assert_frame_equal(joint_keys(source.iloc[[3, 0]]), keys.iloc[[3, 0]])
    skipped = joint_keys(
        source.drop(columns="merchant_category"), excluded=("merchant_category",)
    )
    assert len(skipped.columns) == 6
    assert (
        "category_risk10__joint_key" not in skipped
        and "category_amount_log2__joint_key" not in skipped
    )
    with pytest.raises(ValueError, match="Missing joint input"):
        joint_keys(source.drop(columns="merchant_category"))
    empty = joint_keys(source.iloc[:0].rename_axis("original_row"))
    assert empty.shape == (0, 8) and empty.index.name == "original_row"
    excluded = tuple(source.columns)
    assert joint_keys(pd.DataFrame(index=[2, 4]), excluded=excluded).shape == (2, 0)
    with pytest.raises(ValueError, match="unique"):
        joint_keys(pd.concat([source, source[["amount_usd"]]], axis=1))


def test_joint_bpe_adds_only_eight_keys_and_preserves_old_bpe_manifest():
    pytest.importorskip("tiktoken", reason="Install the optional bpe extra.")
    from fraudlab.bpe import manifest as bpe_manifest
    from fraudlab.experiment import source_provenance

    source = frame().assign(account_balance_usd=15.0, id=0, is_fraud=1)
    before = transform_features(source, "bpe_digits")
    extended = transform_features(source, "joint_bpe")
    pd.testing.assert_frame_equal(extended[before.columns], before)
    pd.testing.assert_frame_equal(transform_features(source, "bpe_digits"), before)
    assert set(extended) - set(before) == set(OUTPUT_NAMES)
    assert len(extended.columns) == len(before.columns) + 8
    assert source_provenance("joint_bpe")["feature_encoding"] == manifest()
    assert (
        manifest()["bpe"]
        == source_provenance("bpe_digits")["feature_encoding"]
        == bpe_manifest()
    )
    assert manifest()["joint_encoding"]["output_count"] == 8


def test_synthetic_cli_offset_persists_joints_unknown_prior_and_predictions(
    tmp_path, monkeypatch
):
    pytest.importorskip("tiktoken", reason="Install the optional bpe extra.")
    pytest.importorskip("lightgbm", reason="Install the optional boosting extra.")
    from fraudlab.__main__ import main
    from fraudlab.data import validate_submission
    from fraudlab.models import predict_saved

    rng = np.random.default_rng(18)
    source = frame(240)
    for name in CONTINUOUS:
        source[name] = rng.uniform(0, 100, len(source)).round(3)
    source["customer_age"] = rng.integers(18, 82, len(source))
    source["card_age_months"] = rng.integers(0, 75, len(source))
    source["merchant_category"] = rng.choice(["retail", "food"], len(source))
    source["auth_method"] = rng.choice(["pin", "password"], len(source))
    source["channel"] = rng.choice(["online", "store"], len(source))
    source["id"] = np.arange(len(source))
    training, test = source.iloc[:200].copy(), source.iloc[200:].copy()
    training["is_fraud"] = np.tile([0, 1], 100)
    test["auth_method"] = "never-seen"
    test["channel"] = "never-seen-channel"
    data, output = tmp_path / "data", tmp_path / "runs"
    data.mkdir()
    training.to_csv(data / "train.csv", index=False)
    test.to_csv(data / "test.csv", index=False)
    sample = pd.DataFrame({"id": test.id.iloc[::-1], "is_fraud": 0})
    sample.to_csv(data / "sample_submission.csv", index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fraudlab",
            "train",
            "--data",
            str(data),
            "--output",
            str(output),
            "--name",
            "joint",
            "--model",
            "offset",
            "--feature-set",
            "joint_bpe",
            "--folds",
            "2",
            "--fixed-training",
            "--glm-frequency",
            "--glm-pooled-tokens",
            "--target-smoothing",
            "50",
            "--logistic-c",
            "0.3",
            "--iterations",
            "8",
            "--num-leaves",
            "4",
            "--learning-rate",
            "0.03",
            "--l2-leaf-reg",
            "5",
            "--min-child-samples",
            "5",
            "--threads",
            "1",
            "--validation-note",
            "Manufactured rows only; persistence contract, no competition fit.",
        ],
    )
    main()
    run = output / "joint"
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert (
        metrics["feature_encoding"] == manifest()
        and "joints.py" in metrics["source_hashes"]
    )
    assert metrics["model_details"]["pooled_tokens"]["output_count"] == 16
    oof = pd.read_csv(run / "oof.csv")
    test_features = transform_features(test, "joint_bpe").drop(columns="id")
    averages = np.zeros(len(test))
    splits = StratifiedKFold(2, shuffle=True, random_state=42).split(
        training, training.is_fraud
    )
    for fold, (train_rows, valid_rows) in enumerate(splits):
        valid = training.iloc[valid_rows[:16]]
        features = transform_features(valid, "joint_bpe").drop(
            columns=["id", "is_fraud"]
        )
        actual = predict_saved("offset", run / f"fold-{fold}", features, "cpu")
        np.testing.assert_allclose(
            actual, oof.prediction.iloc[valid_rows[:16]], atol=1e-12, rtol=0
        )
        averages += (
            predict_saved("offset", run / f"fold-{fold}", test_features, "cpu") / 2
        )
        model = joblib.load(run / f"fold-{fold}.joblib")
        preprocessing = model.glm.named_steps["preprocessing"]
        categorical = next(
            columns
            for name, _, columns in preprocessing.transformers_
            if name == "target_logits"
        )
        encoder = preprocessing.named_transformers_["target_logits"].named_steps[
            "target"
        ]
        index = list(categorical).index("auth_channel__joint_key")
        np.testing.assert_allclose(
            encoder.transform(test_features[categorical])[:, index],
            training.is_fraud.iloc[train_rows].mean(),
        )
        nominal = next(
            columns
            for name, _, columns in preprocessing.transformers_
            if name == "one_hot"
        )
        assert not set(OUTPUT_NAMES).intersection(nominal)
    submission = pd.read_csv(run / "submission.csv")
    validate_submission(submission, sample)
    expected = pd.Series(averages, index=test.id).reindex(sample.id).to_numpy()
    np.testing.assert_allclose(submission.is_fraud, expected, atol=1e-12, rtol=0)
