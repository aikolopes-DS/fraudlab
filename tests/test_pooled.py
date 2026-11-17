"""Honest hierarchical pooling, numeric keys and real toy model persistence."""

import builtins
import json
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from fraudlab.__main__ import main, parser, validate_training_config
from fraudlab.features import transform_features
from fraudlab.models import fit_predict, make_glm_pipeline, predict_saved
from fraudlab.pooled import (
    COLUMNS,
    OUTPUT_NAMES,
    PooledTokenEncoder,
    canonical_decimal,
    manifest,
)

pytest.importorskip("tiktoken", reason="Install the optional bpe extra.")


def frame(values):
    return pd.DataFrame({name: values for name in COLUMNS})


def toy(size=160):
    rng = np.random.default_rng(123)
    result = pd.DataFrame(
        {
            name: np.round(rng.uniform(1, 1000, size), 2 if index < 4 else 1)
            for index, name in enumerate(COLUMNS)
        }
    )
    result["merchant_category"] = np.resize(["a", "b", "c"], size)
    return result, np.tile([0, 1], size // 2)


def config(**overrides):
    return {
        "glm_pooled_tokens": True,
        "glm_inner_folds": 5,
        "logistic_c": 0.3,
        "target_smoothing": 50.0,
        "threads": 1,
        "iterations": 8,
        "num_leaves": 4,
        "learning_rate": 0.03,
        "l2_leaf_reg": 5.0,
        "min_child_samples": 5,
        **overrides,
    }


def test_hierarchy_pools_across_fields_and_conditional_unknown_uses_field():
    source = frame([1.11, 2.22, 3.11, 4.22])
    source["hours_since_last_txn"] = [5.22, 6.22, 7.33, 8.33]
    labels = np.array([1, 0, 0, 0])
    encoder = PooledTokenEncoder(
        pool_smoothing=1, field_smoothing=2, markov_smoothing=3
    ).fit(source, labels)
    outside = frame([9.33, 1.11, 9.44])
    encoded = encoder.transform(outside)
    # Decimal33 is absent from amount, but hours contributes two negatives.
    decimal33_pool = (0 + 0.25) / (2 + 1)
    assert encoded[0, 0] == pytest.approx(decimal33_pool)
    # Decimal11: three fields contribute (sum1,count2) each, then amount shrinks.
    decimal11_pool = (3 + 0.25) / (6 + 1)
    amount11 = (1 + 2 * decimal11_pool) / (2 + 2)
    assert encoded[1, 0] == pytest.approx(amount11)
    assert encoded[1, 12] == pytest.approx((1 + 3 * amount11) / 4)
    # New integer+decimal33 pair backs off to the field's decimal33 rate.
    assert encoded[0, 12] == pytest.approx(encoded[0, 0])
    assert encoded[2, 0] == pytest.approx(labels.mean())
    assert encoded[2, 12] == pytest.approx(labels.mean())
    assert encoded.shape == (3, 16) and np.isfinite(encoded).all()
    np.testing.assert_array_equal(encoder.get_feature_names_out(COLUMNS), OUTPUT_NAMES)


def test_inner_held_out_labels_do_not_affect_their_encoded_rows(monkeypatch):
    import fraudlab.pooled as pooled

    source, labels = toy(60)
    splits = list(
        StratifiedKFold(5, shuffle=True, random_state=42).split(source, labels)
    )

    class FixedSplits:
        def __init__(self, **kwargs):
            pass

        def split(self, X, y):
            return iter(splits)

    monkeypatch.setattr(pooled, "StratifiedKFold", FixedSplits)
    original = PooledTokenEncoder().fit_transform(source, labels)
    held_out = splits[0][1]
    changed = labels.copy()
    changed[held_out] = 1 - changed[held_out]
    modified = PooledTokenEncoder().fit_transform(source, changed)
    np.testing.assert_array_equal(original[held_out], modified[held_out])
    assert not np.array_equal(original, modified)


def test_unknowns_use_each_inner_training_prior_not_full_prior(monkeypatch):
    import fraudlab.pooled as pooled

    labels = np.array([0] * 20 + [1] * 10)
    splits = list(
        StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(30), labels)
    )
    source = frame(np.ones(30))
    # Distinct decimal texts and first digits for each held-out partition.
    for index, (_, validation) in enumerate(splits, 1):
        source.iloc[validation] = 100 + index / 10

    class FixedSplits:
        def __init__(self, **kwargs):
            pass

        def split(self, X, y):
            return iter(splits)

    # Shift labels while holding membership fixed to make inner priors unequal.
    labels = labels.copy()
    labels[splits[0][1]] = 1
    monkeypatch.setattr(pooled, "StratifiedKFold", FixedSplits)
    encoded = PooledTokenEncoder().fit_transform(source, labels)
    for training, validation in splits:
        # The first ten outputs and conditional pairs have never-seen decimals.
        columns = list(range(10)) + list(range(12, 16))
        np.testing.assert_allclose(
            encoded[np.ix_(validation, columns)],
            labels[training].mean(),
            rtol=0,
            atol=0,
        )
    assert labels[splits[0][0]].mean() != labels.mean()
    fitted = PooledTokenEncoder().fit(source, labels)
    unknown = frame([99701.987654])
    np.testing.assert_allclose(fitted.transform(unknown)[:, :10], labels.mean())
    np.testing.assert_allclose(fitted.transform(unknown)[:, 12:], labels.mean())


def test_decimal_canonicalization_token_boundary_and_ids_are_not_inputs():
    from fraudlab.bpe import _encoding
    from fraudlab.pooled import _keys

    assert [canonical_decimal(value) for value in [61.10, 3233, 1e-7, -0.0]] == [
        "61.1",
        "3233.0",
        "0.0000001",
        "0.0",
    ]
    with pytest.raises(ValueError, match="real numbers"):
        canonical_decimal(np.complex128(1))
    source = frame([61.1, 61.10, 99701.39])
    source["id"], source["is_fraud"] = [99, 100, 101], [1, 0, 1]
    keys = _keys(source)
    assert keys[0].tolist() == ["1", "1", "39"]
    assert keys[4].tolist() == ["1", "1", "3"]
    tokens = _encoding().encode_ordinary(" 99701.39")
    dot = _encoding().encode_ordinary(".")[0]
    assert keys[10][-1] == tokens[tokens.index(dot) - 1]
    assert keys[12][-1] == f"{keys[10][-1]}:39"
    source["id"], source["is_fraud"] = [-1, 1234, 0], [0, 1, 0]
    for before, after in zip(keys, _keys(source)):
        np.testing.assert_array_equal(before, after)


def test_decimal_text_is_computed_per_unique_value_not_per_row(monkeypatch):
    import fraudlab.pooled as pooled

    calls = []
    original = pooled.canonical_decimal

    def counted(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(pooled, "canonical_decimal", counted)
    pooled._keys(frame(np.resize([10.1, 20.2], 1000)))
    assert len(calls) == 2 * len(COLUMNS)


@pytest.mark.parametrize(
    "value", [np.nan, np.inf, -1.0, "1.10", True, np.complex128(1 + 2j)]
)
def test_invalid_inputs_fail_instead_of_silently_changing_keys(value):
    with pytest.raises(ValueError):
        PooledTokenEncoder().fit(frame([value] * 10), np.tile([0, 1], 5))


@pytest.mark.parametrize("model_name", ["glm", "offset"])
def test_real_toy_checkpoint_and_metadata_preserve_pooled_rates(tmp_path, model_name):
    source, labels = toy()
    source = transform_features(source, "bpe_digits")
    training, validation, test = (
        source.iloc[:120],
        source.iloc[120:140],
        source.iloc[140:],
    )
    path = tmp_path / model_name
    p_valid, p_test, details = fit_predict(
        model_name,
        training,
        labels[:120],
        validation,
        labels[120:140],
        test,
        config(glm_frequency=True),
        42,
        path,
    )
    saved = joblib.load(path.with_suffix(".joblib"))
    glm = saved if model_name == "glm" else saved.glm
    preprocessing = glm.named_steps["preprocessing"]
    assert (
        preprocessing.output_indices_["pooled_token_logits"].stop
        - preprocessing.output_indices_["pooled_token_logits"].start
        == 16
    )
    assert details["pooled_tokens"] == manifest(42)
    encoder = preprocessing.named_transformers_["pooled_token_logits"].named_steps[
        "pooled"
    ]
    assert encoder.n_features_in_ == 6
    assert not set(encoder.feature_names_in_) & {"id", "is_fraud"}
    np.testing.assert_allclose(
        predict_saved(model_name, path, validation), p_valid, rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(
        predict_saved(model_name, path, test), p_test, rtol=0, atol=1e-14
    )


def test_flag_off_is_identical_and_does_not_import_optional_encoder(monkeypatch):
    source, labels = toy(60)
    source = transform_features(source, "bpe_digits")
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in ("pooled", "tiktoken") or name.startswith("tiktoken."):
            raise AssertionError("Disabled pooling imported an optional component.")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    options = vars(
        parser().parse_args(
            [
                "train",
                "--model",
                "glm",
                "--name",
                "toy",
                "--validation-note",
                "Toy only.",
            ]
        )
    )
    options["patience"] = 100
    validate_training_config(options)
    assert "pooled_token_encoding" not in options
    omitted = config(glm_frequency=True)
    omitted.pop("glm_pooled_tokens")
    a = make_glm_pipeline(source, omitted, 42)
    b = make_glm_pipeline(
        source, config(glm_pooled_tokens=False, glm_frequency=True), 42
    )
    a.fit(source, labels)
    b.fit(source, labels)
    np.testing.assert_array_equal(a.predict_proba(source), b.predict_proba(source))
    assert (
        "pooled_token_logits" not in a.named_steps["preprocessing"].named_transformers_
    )


@pytest.mark.parametrize("model,inner", [("lightgbm", "5"), ("glm", "10")])
def test_cli_rejects_ignored_or_changed_pooled_protocol(model, inner):
    options = vars(
        parser().parse_args(
            [
                "train",
                "--model",
                model,
                "--name",
                "toy",
                "--glm-pooled-tokens",
                "--glm-inner-folds",
                inner,
                "--validation-note",
                "Toy only.",
            ]
        )
    )
    with pytest.raises(ValueError, match="glm_pooled_tokens requires"):
        validate_training_config(options)


def test_cli_records_config_provenance_and_real_offset_cv(tmp_path, monkeypatch):
    source, labels = toy(100)
    train_frame, test = source.iloc[:80].copy(), source.iloc[80:].copy()
    train_frame.insert(0, "id", np.arange(80))
    train_frame["is_fraud"] = labels[:80]
    test.insert(0, "id", np.arange(80, 100))
    data = tmp_path / "data"
    data.mkdir()
    train_frame.to_csv(data / "train.csv", index=False)
    test.to_csv(data / "test.csv", index=False)
    pd.DataFrame({"id": test.id, "is_fraud": 0}).to_csv(
        data / "sample_submission.csv", index=False
    )
    output = tmp_path / "runs"
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
            "toy-pooled",
            "--model",
            "offset",
            "--feature-set",
            "bpe_digits",
            "--glm-pooled-tokens",
            "--glm-frequency",
            "--logistic-c",
            "0.3",
            "--target-smoothing",
            "50",
            "--folds",
            "2",
            "--iterations",
            "4",
            "--min-child-samples",
            "5",
            "--threads",
            "1",
            "--fixed-training",
            "--validation-note",
            "Manufactured rows only; no competition score.",
        ],
    )
    main()
    run = output / "toy-pooled"
    stored_config = json.loads((run / "config.json").read_text())
    metrics = json.loads((run / "metrics.json").read_text())
    assert stored_config["glm_pooled_tokens"] is True
    assert stored_config["pooled_token_encoding"] == manifest(42)
    assert "pooled.py" in metrics["source_hashes"]
    assert all(
        fold["pooled_tokens"] == manifest(42 + index)
        for index, fold in enumerate(metrics["model_details"]["folds"])
    )
