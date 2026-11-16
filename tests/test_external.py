"""Manufactured arrays exercise import contracts; they are not public-model scores."""

import json
import shutil
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from fraudlab import external
from fraudlab.__main__ import main
from fraudlab.data import fingerprint
from fraudlab.splits import make_splits
from fraudlab.experiment import blend


@pytest.fixture
def prediction_release(tmp_path, monkeypatch):
    source, data = tmp_path / "release", tmp_path / "data"
    source.mkdir()
    data.mkdir()
    train = pd.DataFrame(
        {
            "id": np.arange(100),
            "feature": np.arange(100) % 7,
            "is_fraud": np.tile([0, 1], 50),
        }
    )
    test = pd.DataFrame({"id": np.arange(100, 120), "feature": np.arange(20) % 7})
    train.to_csv(data / "train.csv", index=False)
    test.to_csv(data / "test.csv", index=False)
    pd.DataFrame({"id": test["id"].iloc[::-1], "is_fraud": 0}).to_csv(
        data / "sample_submission.csv", index=False
    )
    fold_ids = np.full(len(train), -1, dtype=np.int8)
    for fold, (_, valid) in enumerate(
        make_splits(train, "is_fraud", "stratified", 5, 42)
    ):
        fold_ids[valid] = fold
    logits = np.linspace(-3, 3, len(train), dtype=np.float64)
    np.save(source / "oof.npy", logits)
    np.save(source / "test.npy", np.linspace(-2, 2, len(test), dtype=np.float64))
    np.save(source / "fold_ids.npy", fold_ids)
    manifest = {
        "inputs": {
            "train_csv": {"sha256": fingerprint(data / "train.csv")},
            "test_csv": {"sha256": fingerprint(data / "test.csv")},
            "fold_ids": {"sha256": fingerprint(source / "fold_ids.npy")},
        },
        "sha256": {
            filename: fingerprint(source / filename)
            for filename in ("oof.npy", "test.npy")
        },
        "auc": {"pooled": roc_auc_score(train["is_fraud"], logits)},
        "settings": {
            "prior": "training fold rate",
            "orders": 10,
            "combine": "log-sum-exp per label",
            "test": "mean fold logits",
        },
    }

    def seal():
        (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        # A tiny manufactured release replaces the production pin only in this test.
        monkeypatch.setattr(
            external, "MANIFEST_SHA256", fingerprint(source / "manifest.json")
        )

    seal()
    return source, data, manifest, seal


def test_import_converts_logits_preserves_row_alignment_and_external_credits(
    tmp_path, prediction_release
):
    source, data, _, _ = prediction_release
    metrics = external.import_public_gpt2(source, data, tmp_path / "runs", "public")
    result = pd.read_csv(tmp_path / "runs" / "public" / "submission.csv")
    expected = (1 / (1 + np.exp(-np.linspace(-2, 2, 20))))[::-1]
    assert np.allclose(result["is_fraud"], expected)
    assert result["id"].tolist() == list(range(119, 99, -1))
    assert metrics["phase"] == "exploratory_full_data_cv"
    assert metrics["validation"]["independent_holdout"] is False
    component = metrics["model_details"]["external_components"][0]
    assert component["training_executed_locally"] is False
    assert component["predictions_license"] == "CC BY 4.0"
    assert component["inference"]["import_transform"].startswith("sigmoid")


def test_import_rejects_corrupt_array_before_creating_a_run(
    tmp_path, prediction_release
):
    source, data, _, _ = prediction_release
    np.save(source / "test.npy", np.zeros(20, dtype=np.float64))
    with pytest.raises(ValueError, match="hash mismatch"):
        external.import_public_gpt2(source, data, tmp_path / "runs", "bad")
    assert not (tmp_path / "runs" / "bad").exists()


def test_import_rejects_fold_mismatch_even_with_matching_file_hashes(
    tmp_path, prediction_release
):
    source, data, manifest, seal = prediction_release
    fold_ids = np.load(source / "fold_ids.npy")
    np.save(source / "fold_ids.npy", (4 - fold_ids).astype(np.int8))
    manifest["inputs"]["fold_ids"]["sha256"] = fingerprint(source / "fold_ids.npy")
    seal()
    with pytest.raises(ValueError, match="Public folds differ"):
        external.import_public_gpt2(source, data, tmp_path / "runs", "bad")
    assert not (tmp_path / "runs" / "bad").exists()


def test_blend_retains_external_author_license_and_effective_probability_weight(
    tmp_path, prediction_release
):
    source, data, _, _ = prediction_release
    output = tmp_path / "runs"
    metadata = external.import_public_gpt2(source, data, output, "public")
    own = output / "own"
    own.mkdir()
    for filename in ("oof.csv", "predictions.csv"):
        shutil.copyfile(output / "public" / filename, own / filename)
    metadata.update(model="glm", model_details={"external_components": []})
    (own / "metrics.json").write_text(json.dumps(metadata), encoding="utf-8")
    mixed = blend([own, output / "public"], [0.75, 0.25], output, "mixed")
    component = mixed["model_details"]["external_components"][0]
    assert component["weight"] == 0.25 and component["mixture_space"] == "probability"
    assert component["trained_by_us"] is False
    assert component["predictions_license"] == "CC BY 4.0"
    assert mixed["phase"] == "exploratory_full_data_cv"
    assert mixed["external_components"] == mixed["model_details"]["external_components"]


def test_import_cli_returns_success_after_writing_the_validated_contract(
    tmp_path, prediction_release, monkeypatch, capsys
):
    source, data, _, _ = prediction_release
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fraudlab",
            "import-public-gpt2",
            "--source",
            str(source),
            "--data",
            str(data),
            "--output",
            str(tmp_path / "runs"),
            "--name",
            "public",
        ],
    )
    main()
    assert '"model": "external_gpt2"' in capsys.readouterr().out
    assert (tmp_path / "runs" / "public" / "metrics.json").is_file()
