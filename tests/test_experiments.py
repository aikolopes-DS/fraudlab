"""End-to-end manufactured experiments verify artifacts, not competition performance."""

import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from fraudlab.__main__ import main
from fraudlab.data import validate_submission
from fraudlab.experiment import blend
from fraudlab.holdout import evaluate


@pytest.fixture
def synthetic_data(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    x, y = make_classification(
        n_samples=220,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        class_sep=3,
        random_state=7,
    )
    frame = pd.DataFrame(x, columns=["amount", "distance", "frequency", "age"])
    frame["category"] = np.where(frame["amount"] > 0, "retail", "fuel")
    frame.insert(0, "id", np.arange(len(frame)))
    training, test = frame.iloc[:180].copy(), frame.iloc[180:].copy()
    training["is_fraud"] = y[:180]
    test.loc[test.index[:3], "category"] = "unseen"
    sample = pd.DataFrame({"id": test["id"].iloc[::-1], "is_fraud": 0})
    training.to_csv(data / "train.csv", index=False)
    test.to_csv(data / "test.csv", index=False)
    sample.to_csv(data / "sample_submission.csv", index=False)
    return data, sample


@pytest.mark.parametrize(
    "model",
    [
        "logistic",
        "catboost",
        "lightgbm",
        pytest.param(
            "mlp",
            marks=pytest.mark.skipif(
                importlib.util.find_spec("torch") is None,
                reason="Optional PyTorch is not installed.",
            ),
        ),
    ],
)
def test_training_cli_writes_auditable_oof_and_submission(
    tmp_path, synthetic_data, monkeypatch, model
):
    data, sample = synthetic_data
    output = tmp_path / "runs"
    arguments = [
        "fraudlab",
        "train",
        "--data",
        str(data),
        "--output",
        str(output),
        "--model",
        model,
        "--name",
        "simulation",
        "--folds",
        "3",
        "--validation-note",
        "Manufactured independent rows for an integration test; no competition score.",
        "--device",
        "cpu",
        "--threads",
        "1",
        "--iterations",
        "15",
        "--epochs",
        "3",
        "--batch-size",
        "64",
        "--hidden-dims",
        "16",
        "8",
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    main()
    run = output / "simulation"
    metrics = json.loads((run / "metrics.json").read_text())
    oof = pd.read_csv(run / "oof.csv")
    result = pd.read_csv(run / "submission.csv")
    assert metrics["status"] == "complete"
    assert metrics["rows"] == {"train": 180, "test": 40, "oof": 180}
    assert len(metrics["fold_scores"]) == 3
    assert set(metrics["features"]["excluded"]) == {"id", "is_fraud"}
    assert "id" not in metrics["features"]["names"]
    assert "is_fraud" not in metrics["features"]["names"]
    assert oof["id"].is_unique and len(oof) == 180
    assert oof["prediction"].between(0, 1).all()
    assert set(oof["fold"]) == {0, 1, 2}
    validate_submission(result, sample)
    assert set(metrics["data_hashes"]) == {
        "train.csv",
        "test.csv",
        "sample_submission.csv",
    }
    if model == "logistic":
        assert metrics["roc_auc"] > 0.85
    if model == "mlp":
        assert metrics["model_details"]["architecture"]["hidden_dims"] == [16, 8]
        assert (run / "fold-0.pt").exists()
    with pytest.raises(SystemExit, match="exists"):
        main()


def test_blend_rejects_misaligned_oof_even_when_ids_and_scores_look_valid(tmp_path):
    runs = [tmp_path / "a", tmp_path / "b"]
    for run in runs:
        run.mkdir()
        metrics = {
            "status": "complete",
            "data_hashes": {"train.csv": "same"},
            "validation": {"method": "stratified", "n_splits": 2},
        }
        (run / "metrics.json").write_text(json.dumps(metrics))
        pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "fold": [0, 0, 1, 1],
                "target": [0, 1, 0, 1],
                "prediction": [0.1, 0.8, 0.3, 0.7],
            }
        ).to_csv(run / "oof.csv", index=False)
        pd.DataFrame({"id": [10], "is_fraud": [0.2]}).to_csv(
            run / "predictions.csv", index=False
        )
    second = pd.read_csv(runs[1] / "oof.csv")
    second["fold"] = [1, 1, 0, 0]
    second.to_csv(runs[1] / "oof.csv", index=False)
    with pytest.raises(ValueError, match="OOF rows"):
        blend(runs, [0.5, 0.5], tmp_path / "runs", "bad")
    assert not (tmp_path / "runs" / "bad").exists()


def test_blend_writes_normalized_explicit_weights_and_valid_predictions(tmp_path):
    runs = [tmp_path / "a", tmp_path / "b"]
    for index, run in enumerate(runs):
        run.mkdir()
        metrics = {
            "status": "complete",
            "data_hashes": {"train.csv": "same"},
            "validation": {"method": "stratified", "n_splits": 2},
        }
        (run / "metrics.json").write_text(json.dumps(metrics))
        pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "fold": [0, 0, 1, 1],
                "target": [0, 1, 0, 1],
                "prediction": [0.1, 0.8, 0.3, 0.7],
            }
        ).to_csv(run / "oof.csv", index=False)
        pd.DataFrame(
            {"id": [10, 11], "is_fraud": [0.2 + index * 0.4, 0.8 - index * 0.4]}
        ).to_csv(run / "predictions.csv", index=False)
    result = blend(runs, [1, 3], tmp_path / "runs", "mixed")
    submission = pd.read_csv(tmp_path / "runs" / "mixed" / "submission.csv")
    assert result["model_details"]["weights"] == [0.25, 0.75]
    assert submission["is_fraud"].to_numpy() == pytest.approx([0.5, 0.5])
    assert result["roc_auc"] == 1.0


def test_reserved_holdout_is_absent_from_every_oof_fold_and_evaluated_once(
    tmp_path, synthetic_data, monkeypatch
):
    data, _ = synthetic_data
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
            "--model",
            "logistic",
            "--name",
            "development",
            "--folds",
            "3",
            "--holdout-fraction",
            "0.2",
            "--validation-note",
            "Manufactured IID rows; independent holdout reserved before CV.",
            "--threads",
            "1",
        ],
    )
    main()
    run = output / "development"
    oof = pd.read_csv(run / "oof.csv")
    report = tmp_path / "holdout" / "evaluation.json"
    result = evaluate([run], [1], report, "cpu")
    held_out = pd.read_csv(report.with_suffix(".csv"))
    assert len(oof) == 144 and len(held_out) == 36
    assert set(oof["id"]).isdisjoint(held_out["id"])
    assert len(set(oof["id"]) | set(held_out["id"])) == 180
    assert result["roc_auc"] > 0.85
    assert result["selection"]["frozen_at"] <= result["evaluated_at"]
    with pytest.raises(FileExistsError, match="already exists"):
        evaluate([run], [1], report, "cpu")
