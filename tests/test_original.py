"""Pinned external labels join only outer training, never OOF or test rows."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraudlab import experiment
from fraudlab.__main__ import parser, validate_training_config
from fraudlab.data import fingerprint
from fraudlab.models import predict_saved
from fraudlab.original import (
    COLUMNS,
    FLAGS,
    NOMINAL,
    NUMERIC,
    load_original,
    validate_config,
)
from fraudlab.refit import refit


def raw_frame(size, start):
    continuous = {
        "amount_usd",
        "hours_since_last_txn",
        "distance_from_home_km",
        "account_balance_usd",
        "velocity_score",
        "merchant_risk_score",
    }
    frame = pd.DataFrame(
        {
            name: (
                start + np.arange(size) + index + 0.13
                if name in continuous
                else start + np.arange(size) + index
            )
            for index, name in enumerate(NUMERIC)
        }
    )
    for name in NOMINAL:
        frame[name] = np.resize(["a", "b", "c"], size)
    for name in FLAGS:
        frame[name] = np.arange(size) % 2 == 0
    return frame[list(COLUMNS)]


@pytest.fixture
def sources(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    train = raw_frame(100, 0)
    train.insert(0, "id", np.arange(100))
    train["is_fraud"] = np.tile([0, 1], 50)
    test = raw_frame(10, 1000)
    test.insert(0, "id", np.arange(100, 110))
    original = raw_frame(12, 10000)
    # Numeric identifiers may coincide in different namespaces; never use them.
    original.insert(0, "transaction_id", np.arange(12))
    original["is_fraud"] = np.tile([0, 1], 6)
    for name, frame in (("train", train), ("test", test)):
        frame.to_csv(data / f"{name}.csv", index=False)
    pd.DataFrame({"id": test.id.iloc[::-1], "is_fraud": 0}).to_csv(
        data / "sample_submission.csv", index=False
    )
    path = tmp_path / "original.csv"
    original.to_csv(path, index=False)
    return data, train, test, original, path


def training_config(data, output, name, original=None):
    args = [
        "train",
        "--data",
        str(data),
        "--output",
        str(output),
        "--name",
        name,
        "--model",
        "glm",
        "--phase",
        "exploratory_full_data_cv",
        "--folds",
        "5",
        "--threads",
        "1",
        "--fixed-training",
        "--logistic-c",
        "0.3",
        "--target-smoothing",
        "50",
        "--validation-note",
        "Manufactured original-only augmentation test.",
    ]
    if original is not None:
        args += [
            "--original-data",
            str(original),
            "--original-sha256",
            fingerprint(original),
        ]
    config = vars(parser().parse_args(args))
    config.pop("command")
    config["patience"] = 100
    validate_training_config(config)
    return config


def test_outer_fold_membership_and_oof_never_include_original_rows(
    sources, tmp_path, monkeypatch
):
    data, reference, test, original, path = sources
    captured = []

    def observe(
        name,
        training,
        labels,
        validation,
        validation_y,
        test_features,
        config,
        seed,
        checkpoint,
    ):
        assert (
            "id" not in training
            and "transaction_id" not in training
            and "is_fraud" not in training
        )
        assert list(training) == list(COLUMNS)
        assert len(training) == 92 and training.index.tolist() == list(range(92))
        pd.testing.assert_frame_equal(
            training.iloc[-12:].reset_index(drop=True), original[list(COLUMNS)]
        )
        np.testing.assert_array_equal(labels[-12:], original.is_fraud)
        assert not set(validation.amount_usd) & set(training.amount_usd)
        assert not set(test_features.amount_usd) & set(training.amount_usd)
        expected = dict(zip(reference.amount_usd, reference.is_fraud))
        np.testing.assert_array_equal(
            labels[:80], [expected[value] for value in training.amount_usd[:80]]
        )
        captured.append(validation.amount_usd.to_numpy())
        return (
            np.linspace(0.2, 0.8, len(validation)),
            np.full(len(test_features), 0.4),
            {"algorithm": "toy-boundary-observer"},
        )

    monkeypatch.setattr(experiment, "fit_predict", observe)
    output = tmp_path / "runs"
    metrics = experiment.train(training_config(data, output, "augmented", path))
    oof = pd.read_csv(output / "augmented/oof.csv")
    np.testing.assert_array_equal(oof.id, reference.id)
    np.testing.assert_array_equal(oof.target, reference.is_fraud)
    assert len(np.unique(np.concatenate(captured))) == 100
    assert metrics["rows"] == {"train": 100, "test": 10, "oof": 100}
    assert set(metrics["data_hashes"]) == {
        "train.csv",
        "test.csv",
        "sample_submission.csv",
    }
    metadata = metrics["additional_training_data"]
    assert metadata["rows"] == 12 and metadata["positives"] == 6
    assert metadata["fold_training_rows"] == [92] * 5
    assert metadata["sha256"] == fingerprint(path)
    assert metadata["license"] is None  # A toy file must not inherit public CC0 credit.
    assert metadata["competition_oof_rows_added"] == 0
    assert metadata["component_run"] == str(output / "augmented")


@pytest.mark.parametrize(
    "change",
    [
        "unknown",
        "missing_column",
        "missing_value",
        "invalid_dtype",
        "invalid_label",
        "duplicate_id",
        "duplicate_features",
        "overlap_train",
        "overlap_test",
        "label_as_feature",
    ],
)
def test_bad_schema_dtypes_labels_ids_and_feature_overlap_fail_before_fit(
    sources, change
):
    _, train, test, original, path = sources
    if change == "unknown":
        original["id"] = np.arange(len(original))
    if change == "missing_column":
        original = original.drop(columns="amount_usd")
    if change == "missing_value":
        original.loc[0, "amount_usd"] = np.nan
    if change == "invalid_dtype":
        original["amount_usd"] = "invalid"
    if change == "invalid_label":
        original.loc[0, "is_fraud"] = 2
    if change == "duplicate_id":
        original.loc[1, "transaction_id"] = original.loc[0, "transaction_id"]
    if change == "duplicate_features":
        for name in COLUMNS:
            original.loc[1, name] = original.loc[0, name]
    if change in ("overlap_train", "overlap_test"):
        source = train if change == "overlap_train" else test
        for name in COLUMNS:
            original.loc[0, name] = source.loc[0, name]
    if change == "label_as_feature":
        original["is_fraud__value_key"] = original.is_fraud
    original.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_original(path, fingerprint(path), train, test)


def test_sha_pin_and_id_target_feature_guards(sources):
    _, train, test, _, path = sources
    with pytest.raises(ValueError, match="SHA-256"):
        load_original(path, "0" * 64, train, test)
    with pytest.raises(ValueError, match="must not be feature"):
        load_original(
            path, fingerprint(path), train, test, original_id_column="amount_usd"
        )


@pytest.mark.parametrize(
    "change",
    [
        {"original_sha256": None},
        {"holdout_fraction": 0.15},
        {"split": "group"},
        {"phase": "development"},
        {"refit": True},
        {"model": "lightgbm", "fixed_training": False},
    ],
)
def test_original_protocol_cannot_silently_use_another_validation_contract(
    sources, tmp_path, change
):
    data, _, _, _, path = sources
    config = training_config(data, tmp_path / "runs", "unused", path)
    config.update(change)
    with pytest.raises(ValueError):
        validate_config(config)
    with pytest.raises(ValueError, match="require --original-data"):
        validate_config({"original_sha256": "0" * 64})


@pytest.mark.parametrize("when", ["before", "after"])
def test_changed_original_bytes_invalidate_run_before_oof_is_written(
    sources, tmp_path, monkeypatch, when
):
    data, _, _, _, path = sources
    called = []
    split = experiment.make_splits

    def changed_splits(*args, **kwargs):
        result = split(*args, **kwargs)
        if when == "before":
            path.write_bytes(path.read_bytes() + b"\n")
        return result

    def fitted(
        name, training, labels, validation, validation_y, test, config, seed, checkpoint
    ):
        called.append(True)
        if when == "after":
            path.write_bytes(path.read_bytes() + b"\n")
        return np.full(len(validation), 0.5), np.full(len(test), 0.5), {}

    monkeypatch.setattr(experiment, "make_splits", changed_splits)
    monkeypatch.setattr(experiment, "fit_predict", fitted)
    config = training_config(data, tmp_path / "runs", "changed", path)
    with pytest.raises(ValueError, match="bytes changed"):
        experiment.train(config)
    run = tmp_path / "runs/changed"
    assert json.loads((run / "status.json").read_text())["status"] == "failed"
    assert not (run / "oof.csv").exists() and not (run / "metrics.json").exists()
    assert len(called) == (1 if when == "after" else 0)


def test_disabled_option_is_bit_identical_and_augmentation_checkpoint_reloads(
    sources, tmp_path
):
    data, _, test, _, path = sources
    output = tmp_path / "runs"
    base = training_config(data, output, "base")
    experiment.train(base)
    omitted = training_config(data, output, "omitted")
    for key in ("original_data", "original_sha256", "original_id_column"):
        omitted.pop(key)
    off = experiment.train(omitted)
    assert "additional_training_data" not in off
    for name in ("oof.csv", "predictions.csv", "submission.csv"):
        assert (output / "base" / name).read_bytes() == (
            output / "omitted" / name
        ).read_bytes()
    augmented = training_config(data, output, "real-toy", path)
    augmented.update(
        model="offset",
        feature_set="token_digits",
        glm_pooled_tokens=True,
        glm_frequency=True,
        iterations=3,
        min_child_samples=5,
    )
    metrics = experiment.train(augmented)
    assert metrics["additional_training_data"]["fold_training_rows"] == [92] * 5
    from fraudlab.features import transform_features

    features = transform_features(test.drop(columns="id"), "token_digits")
    restored = np.mean(
        [
            predict_saved("offset", output / f"real-toy/fold-{fold}", features)
            for fold in range(5)
        ],
        axis=0,
    )
    saved = pd.read_csv(output / "real-toy/predictions.csv")
    expected = pd.Series(restored, index=test.id).reindex(saved.id)
    np.testing.assert_allclose(saved.is_fraud, expected, rtol=0, atol=1e-14)


def test_refit_refuses_augmented_config_before_reading_holdout(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"original_data": "pinned.csv", "model": "offset"})
    )
    with pytest.raises(ValueError, match="Original-data augmentation"):
        refit(source, tmp_path / "nonexistent-holdout", tmp_path / "runs", "forbidden")
