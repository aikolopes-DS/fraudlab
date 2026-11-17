"""Label-free reference counts, supervised fold locality and toy persistence."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from fraudlab import experiment, transductive
from fraudlab.__main__ import parser, validate_training_config
from fraudlab.features import CONTINUOUS, transform_features
from fraudlab.frequency import FrequencyEncoder
from fraudlab.models import clean_features, make_glm_pipeline, predict_saved
from fraudlab.refit import refit
from fraudlab.transductive import build_reference_counts, counts_sha256, validate_config


def test_counts_are_per_column_normalized_like_model_and_ignore_ids_labels(monkeypatch):
    training = pd.DataFrame(
        {
            "id": ["secret-a", "b", "c", "d"],
            "is_fraud": [0, 1, 1, 0],
            "category": ["a", None, "b", "a"],
            "flag": [True, False, True, False],
            "numeric": [1.0, 2.0, 3.0, 4.0],
        }
    )
    test = pd.DataFrame(
        {
            "id": ["z", "x", "y"],
            "is_fraud": [1, 0, 1],
            "category": ["test-only", "a", None],
            "flag": [False, False, True],
            "numeric": [9.0, 8.0, 7.0],
        }
    )
    cleaner = transductive.clean_features
    widths = []

    def one_column(frame, numeric, categorical):
        widths.append(frame.shape[1])
        return cleaner(frame, numeric, categorical)

    monkeypatch.setattr(transductive, "clean_features", one_column)
    reference, metadata = build_reference_counts(
        training, test, list(training), {"train.csv": "toy"}
    )
    assert reference["category"].to_dict() == {
        "__MISSING__": 2,
        "a": 3,
        "b": 1,
        "test-only": 1,
    }
    assert reference["flag"].to_dict() == {"False": 4, "True": 3}
    assert set(reference) == {"category", "flag"} and widths == [1] * 4
    assert metadata["rows"] == {"train": 4, "test": 3, "total": 7}
    assert metadata["label_usage"] == "none"
    changed, changed_meta = build_reference_counts(
        training.assign(id="other", is_fraud=1 - training.is_fraud),
        test.assign(is_fraud=0),
        list(training),
        {"train.csv": "toy"},
    )
    assert (
        counts_sha256(changed)
        == metadata["counts_sha256"]
        == changed_meta["counts_sha256"]
    )
    # Row ordering and count-table index ordering do not affect the content hash.
    reordered, _ = build_reference_counts(
        training.iloc[::-1], test.iloc[::-1], list(training), {}
    )
    assert counts_sha256(reordered) == counts_sha256(reference)
    assert counts_sha256(dict(reversed(list(reference.items())))) == counts_sha256(
        reference
    )


def test_reference_encoder_is_cloneable_owned_persistent_and_uses_unlabeled_support(
    tmp_path,
):
    whole = pd.DataFrame({"category": ["a", "a", "b", "validation-only"]})
    test = pd.DataFrame({"category": ["test-only", "a"]})
    reference, _ = build_reference_counts(whole, test, list(whole), {})
    encoder = clone(FrequencyEncoder(reference_counts=reference)).fit(
        whole.iloc[:3], y=[0, 1, 0]
    )
    outside = pd.DataFrame(
        {"category": ["validation-only", "test-only", "never-seen", "a"]}
    )
    np.testing.assert_array_equal(
        encoder.transform(outside), np.log1p([[1], [1], [0], [3]])
    )
    repeated = FrequencyEncoder(reference_counts=reference).fit(
        whole.iloc[:3], y=[1, 0, 1]
    )
    np.testing.assert_array_equal(
        repeated.transform(outside), encoder.transform(outside)
    )
    reference["category"].loc["a"] = 99
    assert encoder.counts_["category"].loc["a"] == 3
    path = tmp_path / "counts.joblib"
    joblib.dump(encoder, path)
    np.testing.assert_array_equal(
        joblib.load(path).transform(outside), encoder.transform(outside)
    )


def test_legacy_checkpoint_without_reference_option_keeps_default_behavior(tmp_path):
    encoder = FrequencyEncoder().fit(pd.DataFrame({"key": ["a", "a", "b"]}))
    del encoder.reference_counts
    path = tmp_path / "legacy.joblib"
    joblib.dump(encoder, path)
    restored = joblib.load(path)
    assert clone(restored).reference_counts is None
    np.testing.assert_array_equal(
        restored.transform(pd.DataFrame({"key": ["a", "unknown"]})),
        np.log1p([[2], [0]]),
    )


@pytest.mark.parametrize(
    "counts",
    [
        pd.Series([2, -1], index=["a", "b"]),
        pd.Series([2, 1.5], index=["a", "b"]),
        pd.Series([2, np.nan], index=["a", "b"]),
        pd.Series([2, 1], index=["a", "a"]),
        pd.Series([2, 1], index=[1, 2]),
        pd.Series([2], index=["a"]),
        pd.Series([1, 1], index=["a", "b"]),
        pd.Series([2, np.inf], index=["a", "b"]),
    ],
)
def test_invalid_or_incomplete_reference_counts_cannot_be_fitted(counts):
    with pytest.raises(ValueError):
        FrequencyEncoder(reference_counts={"key": counts}).fit(
            pd.DataFrame({"key": ["a", "a", "b"]})
        )


def model_frame(size):
    frame = pd.DataFrame(
        {
            name: np.arange(size, dtype=float) + index + 0.13
            for index, name in enumerate(CONTINUOUS)
        }
    )
    frame["category"] = np.resize(["a", "b", "c"], size)
    frame["flag"] = np.arange(size) % 2 == 0
    return frame


def test_only_frequency_reference_is_global_scalers_targets_and_pool_prior_are_outer_only():
    full, test = model_frame(100), model_frame(20)
    test["category"] = "test-only"
    references, metadata = build_reference_counts(full, test, list(full), {})
    training = clean_features(full.iloc[:60])
    labels = np.array([0] * 45 + [1] * 15)
    config = {
        "glm_frequency": True,
        "glm_transductive_frequency": True,
        "glm_pooled_tokens": True,
        "_frequency_reference": references,
        "transductive_frequency": metadata,
        "target_smoothing": 50,
        "logistic_c": 0.3,
    }
    preprocessing = make_glm_pipeline(training, config, 42).named_steps["preprocessing"]
    preprocessing.fit_transform(training, labels)
    frequency = preprocessing.named_transformers_["category_frequency"]
    expected = np.log1p(
        np.column_stack([training[name].map(references[name]) for name in references])
    )
    np.testing.assert_allclose(
        frequency.named_steps["scale"].mean_, expected.mean(axis=0)
    )
    target = preprocessing.named_transformers_["target_logits"].named_steps["target"]
    assert target.target_mean_ == labels.mean() == 0.25
    pooled = preprocessing.named_transformers_["pooled_token_logits"].named_steps[
        "pooled"
    ]
    assert pooled.maps_["prior"] == 0.25
    numeric = preprocessing.named_transformers_["numeric"].named_steps["scale"]
    np.testing.assert_allclose(numeric.mean_, training[list(CONTINUOUS)].mean())


def configuration(**changes):
    config = {
        "glm_transductive_frequency": True,
        "glm_frequency": True,
        "model": "offset",
        "phase": "exploratory_full_data_cv",
        "split": "stratified",
        "folds": 5,
        "seed": 42,
        "holdout_fraction": 0,
    }
    return {**config, **changes}


@pytest.mark.parametrize(
    "changes",
    [
        {"glm_frequency": False},
        {"model": "neural_offset"},
        {"phase": "development"},
        {"split": "group"},
        {"folds": 3},
        {"seed": 99},
        {"holdout_fraction": 0.1},
        {"original_data": "original.csv"},
        {"refit": True},
    ],
)
def test_flag_cannot_silently_change_protocol_or_mix_original_rows(changes):
    with pytest.raises(ValueError, match="Transductive frequency requires"):
        validate_config(configuration(**changes))


def test_refit_and_serialized_runtime_maps_are_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"glm_transductive_frequency": True, "model": "offset"})
    )
    with pytest.raises(ValueError, match="no full-data refit contract"):
        refit(source, tmp_path / "absent", tmp_path / "runs", "invalid")
    with pytest.raises(ValueError, match="runtime-only"):
        experiment.train({"_frequency_reference": {}})


def test_real_toy_cv_serializes_only_metadata_and_reload_keeps_fixed_maps(tmp_path):
    pytest.importorskip("tiktoken")
    data = tmp_path / "data"
    data.mkdir()
    full, test = model_frame(100), model_frame(10)
    full.insert(0, "id", np.arange(100))
    full["is_fraud"] = np.tile([0, 1], 50)
    test.insert(0, "id", np.arange(100, 110))
    test["category"] = "test-only"
    full.to_csv(data / "train.csv", index=False)
    test.to_csv(data / "test.csv", index=False)
    pd.DataFrame({"id": test.id, "is_fraud": 0}).to_csv(
        data / "sample_submission.csv", index=False
    )
    output = tmp_path / "runs"
    args = [
        "train",
        "--data",
        str(data),
        "--output",
        str(output),
        "--name",
        "transductive",
        "--model",
        "offset",
        "--phase",
        "exploratory_full_data_cv",
        "--folds",
        "5",
        "--threads",
        "1",
        "--feature-set",
        "bpe_digits",
        "--glm-frequency",
        "--glm-pooled-tokens",
        "--glm-transductive-frequency",
        "--fixed-training",
        "--iterations",
        "3",
        "--min-child-samples",
        "5",
        "--target-smoothing",
        "50",
        "--logistic-c",
        "0.3",
        "--validation-note",
        "Manufactured transductive feature counts only.",
    ]
    config = vars(parser().parse_args(args))
    config.pop("command")
    config["patience"] = 100
    validate_training_config(config)
    metrics = experiment.train(config)
    run = output / "transductive"
    stored = json.loads((run / "config.json").read_text())
    assert (
        "_frequency_reference" not in stored
        and "_frequency_reference" not in json.dumps(metrics)
    )
    metadata = metrics["transductive_frequency"]
    assert stored["transductive_frequency"] == metadata
    assert metadata["rows"] == {"train": 100, "test": 10, "total": 110}
    assert metrics["validation"]["transductive_feature_counts"]["rows"] == 110
    assert metadata["corpus_hashes"] == metrics["data_hashes"]
    infer = transform_features(
        pd.read_csv(data / "test.csv").drop(columns="id"), "bpe_digits"
    )
    prediction = []
    for fold in range(5):
        checkpoint = run / f"fold-{fold}"
        model = joblib.load(checkpoint.with_suffix(".joblib"))
        frequency = (
            model.glm.named_steps["preprocessing"]
            .named_transformers_["category_frequency"]
            .named_steps["frequency"]
        )
        assert counts_sha256(frequency.counts_) == metadata["counts_sha256"]
        prediction.append(predict_saved("offset", checkpoint, infer))
    saved = pd.read_csv(run / "predictions.csv")
    np.testing.assert_allclose(
        saved.is_fraud, np.mean(prediction, axis=0), rtol=0, atol=1e-14
    )
    # Removing the opt-in and explicitly setting false follow the identical old path.
    base = {**config, "name": "base"}
    base.pop("glm_transductive_frequency")
    off = {**base, "name": "off", "glm_transductive_frequency": False}
    experiment.train(base)
    experiment.train(off)
    for name in ("oof.csv", "predictions.csv", "submission.csv"):
        assert (output / "base" / name).read_bytes() == (
            output / "off" / name
        ).read_bytes()
