"""Synthetic checks for TabM loss, nested selection, persistence and CLI artifacts."""

import inspect
import json
import sys

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tabm")
pytest.importorskip("rtdl_num_embeddings")

from fraudlab import tabm as candidate
from fraudlab.__main__ import main, parser, validate_training_config
from fraudlab.data import validate_submission
from fraudlab.experiment import versions
from fraudlab.models import clean_features, fit_predict, predict_saved


def config(**changes):
    result = {
        "dropout": 0.1,
        "neural_learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 64,
        "epochs": 2,
        "patience": 1,
        "target_smoothing": 20.0,
        "neural_numeric_transform": "standard",
        "neural_categorical_transform": "target",
        "hidden_dims": [256, 256],
        "device": "cpu",
        "threads": 1,
    }
    result.update(changes)
    return result


def test_tabm_optimizes_each_member_and_averages_probabilities():
    logits = torch.tensor([[[4.0], [-1.0]], [[-3.0], [2.0]]], requires_grad=True)
    labels = torch.tensor([1.0, 0.0])
    loss = candidate.member_loss(logits, labels)
    manual = torch.stack(
        [
            torch.nn.functional.binary_cross_entropy_with_logits(
                logits[:, i, 0], labels
            )
            for i in range(2)
        ]
    ).mean()
    torch.testing.assert_close(loss, manual)
    loss.backward()
    assert torch.all(logits.grad != 0)
    probabilities = candidate.mean_probability(logits.detach())
    torch.testing.assert_close(
        probabilities, logits.detach().sigmoid().mean(dim=1)[:, 0]
    )
    assert not torch.allclose(
        probabilities, logits.detach().mean(dim=1)[:, 0].sigmoid()
    )
    with pytest.raises(ValueError, match="batch, k, 1"):
        candidate.member_loss(logits[..., 0], labels)


def test_tabm_nested_preprocessing_fresh_refit_flags_and_reload(tmp_path, monkeypatch):
    rng = np.random.default_rng(100)
    frame = pd.DataFrame(
        {
            "value": rng.normal(size=240),
            "constant": np.ones(240),
            "category": np.tile(["a", "b", "c", "d"], 60),
        },
        index=np.arange(1000, 1240),
    )
    y = np.tile([0, 1], 120)
    external = pd.DataFrame(
        {
            "value": [1e10, -1e10, np.nan, 0.0],
            "constant": [7.0] * 4,
            "category": ["unseen"] * 4,
        },
        index=np.arange(9000, 9004),
    )
    seen_pre, seen_bins, seen_training, created_models, seen_optimizers = (
        [],
        [],
        [],
        [],
        [],
    )
    original_prepare, original_bins = candidate.prepare_training, candidate.compute_bins
    original_train, original_build = candidate._train, candidate.build_tabm
    original_optimizer = torch.optim.AdamW

    def prepare(frame, y, seed, config):
        seen_pre.append(frame.index.to_numpy())
        return original_prepare(frame, y, seed, config)

    def bins(x, *args, **kwargs):
        seen_bins.append((len(x), float(x.abs().max())))
        return original_bins(x, *args, **kwargs)

    def train(numbers, y, bins, config, seed, device, **kwargs):
        seen_training.append(
            (len(numbers), kwargs.get("epochs"), kwargs.get("validation"))
        )
        return original_train(numbers, y, bins, config, seed, device, **kwargs)

    def build(bins, config):
        model = original_build(bins, config)
        created_models.append(model)
        return model

    def optimizer(parameters, **kwargs):
        seen_optimizers.append(kwargs)
        return original_optimizer(parameters, **kwargs)

    monkeypatch.setattr(candidate, "prepare_training", prepare)
    monkeypatch.setattr(candidate, "compute_bins", bins)
    monkeypatch.setattr(candidate, "_train", train)
    monkeypatch.setattr(candidate, "build_tabm", build)
    monkeypatch.setattr(torch.optim, "AdamW", optimizer)
    settings = config(
        neural_learning_rate=0.002,
        weight_decay=0.003,
        target_smoothing=7.0,
        dropout=0.15,
        batch_size=48,
    )
    checkpoint = tmp_path / "fold-0"
    # An unusable outer-label object proves that the dispatch does not read it.
    outer_labels = object()
    valid, test, details = fit_predict(
        "tabm",
        frame,
        y,
        external,
        outer_labels,
        external.copy(),
        settings,
        42,
        checkpoint,
    )
    assert [len(index) for index in seen_pre] == [204, 240]
    assert all(set(index).issubset(frame.index) for index in seen_pre)
    assert all(not set(index).intersection(external.index) for index in seen_pre)
    assert [rows for rows, _ in seen_bins] == [204, 240]
    assert all(maximum < 1e9 for _, maximum in seen_bins)
    assert seen_training[0][1] is None and len(seen_training[0][2][0]) == 36
    assert seen_training[1][1] == details["selected_epoch"]
    assert seen_training[1][2] is None
    assert len(created_models) == 2 and created_models[0] is not created_models[1]
    assert details["refit_epochs_completed"] == details["selected_epoch"]
    assert details["input_features"] == 3 and details["active_features"] == 2
    assert details["target_encoder"]["smooth"] == 7
    assert details["learning_rate"] == 0.002 and details["weight_decay"] == 0.003
    assert details["batch_size"] == 48 and details["architecture"]["dropout"] == 0.15
    assert details["threads"] == 1 and details["device"] == "cpu"
    assert seen_optimizers == [{"lr": 0.002, "weight_decay": 0.003}] * 2
    assert valid.shape == test.shape == (4,) and np.isfinite(valid).all()
    checkpoint_data = torch.load(checkpoint.with_suffix(".pt"), weights_only=True)
    assert len(checkpoint_data["bins"]) == details["active_features"]
    assert checkpoint_data["config"]["target_smoothing"] == 7
    reloaded = predict_saved("tabm", checkpoint, external, "cpu")
    np.testing.assert_allclose(valid, reloaded, rtol=0, atol=1e-7)
    # A serving batch can infer object dtype when a numeric column is all None.
    all_missing = external.iloc[:2].assign(value=None, constant=None)
    missing_predictions = predict_saved("tabm", checkpoint, all_missing, "cpu")
    numeric_missing = all_missing.assign(value=np.nan, constant=np.nan)
    np.testing.assert_allclose(
        missing_predictions,
        predict_saved("tabm", checkpoint, numeric_missing, "cpu"),
        rtol=0,
        atol=1e-7,
    )
    assert np.isfinite(missing_predictions).all()
    assert "validation_y" not in inspect.signature(candidate.fit_tabm).parameters


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--fixed-training"], "inner epoch selection"),
        (["--hidden-dims", "16", "8"], "256 256"),
        (["--neural-categorical-transform", "embedding"], "target"),
    ],
)
def test_tabm_rejects_incompatible_flags(arguments, message):
    settings = vars(
        parser().parse_args(
            [
                "train",
                "--model",
                "tabm",
                "--name",
                "toy",
                "--validation-note",
                "Toy",
                "--patience",
                "6",
                *arguments,
            ]
        )
    )
    with pytest.raises(ValueError, match=message):
        validate_training_config(settings)


def test_tabm_cli_writes_artifacts_with_inference_only_outer_validation(
    tmp_path, monkeypatch
):
    data, output = tmp_path / "data", tmp_path / "runs"
    data.mkdir()
    rng = np.random.default_rng(7)
    x = rng.normal(size=270)
    frame = pd.DataFrame(
        {
            "id": np.arange(270),
            "value": x,
            "category": np.where(x > 0, "retail", "fuel"),
        }
    )
    training, test = frame.iloc[:240].copy(), frame.iloc[240:].copy()
    training["is_fraud"] = np.tile([0, 1], 120)
    sample = pd.DataFrame({"id": test["id"].iloc[::-1], "is_fraud": 0})
    training.to_csv(data / "train.csv", index=False)
    test.to_csv(data / "test.csv", index=False)
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
            "--model",
            "tabm",
            "--name",
            "toy",
            "--folds",
            "3",
            "--device",
            "cpu",
            "--threads",
            "1",
            "--epochs",
            "1",
            "--batch-size",
            "64",
            "--validation-note",
            "Fabricated rows; no competition performance measured.",
        ],
    )
    main()
    run = output / "toy"
    saved_config = json.loads((run / "config.json").read_text())
    metrics = json.loads((run / "metrics.json").read_text())
    oof = pd.read_csv(run / "oof.csv")
    assert saved_config["patience"] == 6
    assert saved_config["hidden_dims"] == [256, 256]
    assert saved_config["neural_categorical_transform"] == "target"
    assert metrics["validation"]["early_stopping_uses_validation"] is False
    assert metrics["validation"]["inner_epoch_selection"]["fresh_outer_refit"] is True
    assert len(metrics["model_details"]["folds"]) == 3
    assert len(oof) == 240 and oof["prediction"].between(0, 1).all()
    for index, details in enumerate(metrics["model_details"]["folds"]):
        assert details["selected_epoch"] == details["refit_epochs_completed"] == 1
        assert details["architecture"]["k"] == 16
        assert (run / f"fold-{index}.pt").exists()
        assert (run / f"fold-{index}.preprocessing.joblib").exists()
    validate_submission(pd.read_csv(run / "submission.csv"), sample)
    assert versions()["tabm"] == "0.0.3"
    assert versions()["rtdl_num_embeddings"] == "0.0.12"


def test_tabm_refuses_undefined_full_data_refit_before_creating_run(tmp_path):
    from fraudlab.refit import refit

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model": "tabm"}))
    with pytest.raises(ValueError, match="CV checkpoint ensemble"):
        refit(source, tmp_path / "absent.json", tmp_path / "output", "refit")
    assert not (tmp_path / "output").exists()
