"""Small real-library checks of the RealMLP data boundary and saved inference."""

import inspect
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, train_test_split

pytest.importorskip("pytabkit")

from fraudlab import models, realmlp as candidate
from fraudlab.__main__ import parser, validate_training_config
from fraudlab.refit import refit


def toy(n=160):
    rng = np.random.default_rng(123)
    frame = pd.DataFrame({name: rng.uniform(0, 100, n) for name in candidate.NUMERIC})
    frame["hours_since_last_txn"] = np.arange(n) + 0.25
    frame["amount_usd"] = np.arange(n) + 0.75
    for name in candidate.CATEGORICAL:
        frame[name] = np.where(np.arange(n) % 3, "seen_a", "seen_b")
    return frame, (np.arange(n) % 4 == 0).astype(np.int64)


def config():
    arguments = parser().parse_args(
        [
            "train",
            "--name",
            "toy-realmlp",
            "--model",
            "realmlp",
            "--feature-set",
            "raw",
            "--validation-note",
            "synthetic contract only",
            "--epochs",
            "1",
            "--batch-size",
            "32",
            "--dropout",
            ".05",
            "--neural-learning-rate",
            ".073",
            "--weight-decay",
            ".038",
            "--target-smoothing",
            "auto",
            "--device",
            "cpu",
            "--threads",
            "2",
        ]
    )
    values = vars(arguments).copy()
    values.pop("command")
    validate_training_config(values)
    return values


def test_preprocessing_is_train_only_crossfitted_and_unknown_safe():
    frame, y = toy()
    train_idx, valid_idx = train_test_split(
        np.arange(len(frame)), test_size=0.15, stratify=y, random_state=42
    )
    prep = candidate.RealMLPPreprocessor(42)
    train = prep.fit_transform(frame.iloc[train_idx], y[train_idx])
    original_edges = [x.bin_edges_[0].copy() for x in prep.bins]
    original_categories = [
        x.copy()
        for x in prep.converter.cat_tf.named_transformers_["categorical"].categories_
    ]
    outliers = frame.iloc[valid_idx].copy()
    outliers.loc[:, list(candidate.NUMERIC)] = 1e12
    outliers.loc[:, list(candidate.CATEGORICAL)] = "validation_only"
    external = prep.transform(outliers)
    assert train.tensors["x_cont"].shape == (len(train_idx), 20)
    assert train.tensors["x_cat"].shape == (len(train_idx), 33)
    assert (external.tensors["x_cat"][:, :11].numpy() == 0).all()
    for old, encoder in zip(original_edges, prep.bins):
        np.testing.assert_array_equal(old, encoder.bin_edges_[0])
    for old, new in zip(
        original_categories,
        prep.converter.cat_tf.named_transformers_["categorical"].categories_,
    ):
        np.testing.assert_array_equal(old, new)
    # The exact hours value occurs once: its training TE must be the prior of the
    # other four encoder folds, rather than its own label/full-map target value.
    actual = train.tensors["x_cont"][:, -1].numpy()
    for fit, valid in StratifiedKFold(5, shuffle=True, random_state=42).split(
        frame.iloc[train_idx], y[train_idx]
    ):
        np.testing.assert_allclose(actual[valid], y[train_idx][fit].mean(), atol=1e-7)
    missing = outliers.iloc[:2].copy()
    for name in candidate.NUMERIC:
        missing[name] = pd.Series(
            [None] * len(missing), index=missing.index, dtype=object
        )
    assert np.isfinite(prep.transform(missing).tensors["x_cont"].numpy()).all()


def test_inner_selection_fresh_outer_refit_and_reload_in_new_process(
    tmp_path, monkeypatch
):
    frame, y = toy()
    calls, models_created = [], []
    real_train, real_interface = candidate._train, candidate.NNAlgInterface

    def traced_train(numbers, labels, validation, *args, **kwargs):
        calls.append(
            (
                numbers.tensors["x_cont"][:, 0].numpy().copy(),
                labels.copy(),
                validation,
                kwargs.get("stop_epoch"),
            )
        )
        return real_train(numbers, labels, validation, *args, **kwargs)

    def fresh_interface(*args, **kwargs):
        model = real_interface(*args, **kwargs)
        models_created.append(model)
        return model

    monkeypatch.setattr(candidate, "_train", traced_train)
    monkeypatch.setattr(candidate, "NNAlgInterface", fresh_interface)
    validation = frame.iloc[:7].copy()
    validation.loc[:, "amount_usd"] = -10000
    path = tmp_path / "model"
    valid, test, details = models.fit_predict(
        "realmlp", frame, y, validation, object(), validation, config(), 42, path
    )
    expected_train, expected_valid = train_test_split(
        np.arange(len(frame)), test_size=0.15, stratify=y, random_state=42
    )
    np.testing.assert_array_equal(
        calls[0][0], frame.iloc[expected_train]["amount_usd"].to_numpy(dtype=np.float32)
    )
    np.testing.assert_array_equal(calls[0][1], y[expected_train])
    np.testing.assert_array_equal(calls[0][2][1], y[expected_valid])
    np.testing.assert_array_equal(
        calls[1][0], frame["amount_usd"].to_numpy(dtype=np.float32)
    )
    assert calls[1][2] is None and calls[1][3] == details["selected_epoch"] == 1
    assert len(models_created) == 2 and models_created[0] is not models_created[1]
    assert models_created[0].model is not models_created[1].model
    # Repeat the inner fit with validation-only outliers and opposite labels.
    # With E fixed to1, every fitted tensor must remain bit-identical: neither
    # the library normalization nor gradients may depend on this held-out set.
    prep = candidate.RealMLPPreprocessor(42)
    inner = prep.fit_transform(frame.iloc[expected_train], y[expected_train])
    perturbed = frame.iloc[expected_valid].copy()
    perturbed.loc[:, list(candidate.NUMERIC)] = 1e9
    perturbed.loc[:, list(candidate.CATEGORICAL)] = "inner_validation_only"
    other, _ = real_train(
        inner,
        y[expected_train],
        (prep.transform(perturbed), 1 - y[expected_valid]),
        candidate.RealMLPConfig.from_training_config(config()),
        42,
        "cpu",
        2,
    )
    for key, value in models_created[0].model.state_dict().items():
        np.testing.assert_array_equal(
            value.cpu().numpy(), other.model.state_dict()[key].cpu().numpy()
        )
    assert details["refit"]["completed_epochs"] == 1
    assert "validation_y" not in inspect.signature(candidate.fit_realmlp).parameters
    np.testing.assert_array_equal(valid, test)
    loaded = models.predict_saved("realmlp", path, validation, "cpu")
    np.testing.assert_array_equal(loaded, valid)
    validation.to_pickle(tmp_path / "input.pkl")
    command = [
        sys.executable,
        "-c",
        "import numpy as np,pandas as pd; from pathlib import Path; from fraudlab.models import predict_saved; import sys; p=Path(sys.argv[1]); np.save(p/'loaded.npy',predict_saved('realmlp',p/'model',pd.read_pickle(p/'input.pkl'),'cpu'))",
        str(tmp_path),
    ]
    subprocess.run(
        command, check=True, capture_output=True, text=True, env=os.environ.copy()
    )
    np.testing.assert_array_equal(np.load(tmp_path / "loaded.npy"), valid)


def test_effective_library_config_and_flags_are_explicit(monkeypatch):
    values = config()
    resolved = candidate.official_config(
        candidate.RealMLPConfig.from_training_config(values), True
    )
    assert (resolved["n_ens"], resolved["hidden_sizes"], resolved["act"]) == (
        16,
        [512, 256, 128],
        "silu",
    )
    assert resolved["num_emb_type"] == "pbld" and resolved["plr_act_name"] == "linear"
    assert resolved["plr_use_densenet"] and resolved["use_parametric_act"]
    for key, value in [
        ("target_smoothing", 20),
        ("epochs", 4),
        ("feature_set", "value_digits"),
        ("fixed_training", True),
        ("neural_numeric_transform", "standard"),
    ]:
        with pytest.raises(ValueError, match="RealMLP"):
            candidate.RealMLPConfig.from_training_config({**values, key: value})
    monkeypatch.setattr(candidate.importlib.metadata, "version", lambda _: "1.9.0")
    with pytest.raises(ValueError, match="1.8.0"):
        candidate.official_config(candidate.RealMLPConfig(), True)


def test_full_data_refit_is_rejected_before_reading_holdout(tmp_path):
    import json

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model": "realmlp"}))
    with pytest.raises(ValueError, match="RealMLP.*full-data refit"):
        refit(source, tmp_path / "missing.json", tmp_path / "runs", "forbidden")
    assert not (tmp_path / "runs").exists()
