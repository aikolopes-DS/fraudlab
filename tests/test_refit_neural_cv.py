"""Frozen CV-selection receipts and one tiny real all-row CPU neural refit."""

import copy
import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "refit_neural_cv", Path(__file__).parents[1] / "scripts/refit_neural_cv.py"
)
candidate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(candidate)


def test_epoch_requires_the_exact_audited_cv_selection_and_all_gates():
    epochs = [3, 2, 3, 3, 3]
    weights = [0.421875, 0.140625, 0.1875, 0.25]
    plan = {
        "cv_epochs": epochs.copy(),
        "reference_cv_auc": 0.9518,
        "inputs": {"selection": {"path": "artifacts/runs/selected-neural-cv"}},
    }
    own = {
        "status": "complete",
        "phase": "exploratory_full_data_cv",
        "model_details": {"folds": [{"selected_epoch": e} for e in epochs]},
    }
    selected = {
        "status": "complete",
        "phase": "exploratory_full_data_cv",
        "roc_auc": 0.9518,
        "model_details": {"weights": weights.copy()},
    }
    audit = {
        "status": "passed",
        "selected_within_round": {
            "run": "selected-neural-cv",
            "eligible": True,
            "weights": weights.copy(),
            "delta_vs_r18": 0.00005,
            "delta_vs_r10": 0.0002,
            "improved_folds_vs_r18": 4,
            "roc_auc": 0.9518,
        },
    }
    originals = copy.deepcopy((plan, own, selected, audit))
    assert candidate.chosen_epoch(plan, own, selected, audit) == 3
    assert (plan, own, selected, audit) == originals

    # Each variation breaks a different part of the frozen selection contract.
    mutations = [
        lambda p, o, s, a: p["cv_epochs"].__setitem__(1, 3),
        lambda p, o, s, a: o["model_details"]["folds"][0].update(selected_epoch=2),
        lambda p, o, s, a: o.update(status="failed"),
        lambda p, o, s, a: s.update(phase="full_data_refit"),
        lambda p, o, s, a: a.update(status="failed"),
        lambda p, o, s, a: a["selected_within_round"].update(run="another-run"),
        lambda p, o, s, a: a["selected_within_round"].update(eligible=False),
        lambda p, o, s, a: a["selected_within_round"].update(
            weights=[0.5625, 0, 0.1875, 0.25]
        ),
        lambda p, o, s, a: a["selected_within_round"].update(delta_vs_r18=0),
        lambda p, o, s, a: a["selected_within_round"].update(delta_vs_r10=0.0001),
        lambda p, o, s, a: a["selected_within_round"].update(improved_folds_vs_r18=3),
        lambda p, o, s, a: s.update(roc_auc=0.9517),
        lambda p, o, s, a: s["model_details"].update(weights=[0.5625, 0, 0.1875, 0.25]),
    ]
    for mutate in mutations:
        changed = copy.deepcopy(originals)
        mutate(*changed)
        with pytest.raises(ValueError):
            candidate.chosen_epoch(*changed)


def test_full_refit_uses_all_rows_once_without_validation_and_reloads(
    tmp_path, monkeypatch
):
    pytest.importorskip("pytabkit")
    import cloudpickle
    from fraudlab import realmlp as api

    rng = np.random.default_rng(230042)

    def raw(n):
        frame = pd.DataFrame({name: rng.uniform(0, 100, n) for name in api.NUMERIC})
        frame["hours_since_last_txn"] = np.arange(n) + 0.25
        frame["amount_usd"] = np.arange(n) + 0.75
        for name in api.CATEGORICAL:
            frame[name] = np.where(np.arange(n) % 3, "seen_a", "seen_b")
        return frame

    frame, test = raw(160), raw(1027)
    y = (np.arange(len(frame)) % 4 == 0).astype(np.int64)
    for name in api.CATEGORICAL:
        test[name] = "test_only_category"
    config = {
        "model": "realmlp",
        "feature_set": "raw",
        "phase": "exploratory_full_data_cv",
        "hidden_dims": [512, 256, 128],
        "neural_categorical_transform": "embedding",
        "neural_numeric_transform": "realmlp",
        "target_smoothing": "auto",
        "epochs": 3,
        "patience": 3,
        "batch_size": 288,
        "dropout": 0.05,
        "neural_learning_rate": 0.073,
        "weight_decay": 0.038,
        "seed": 42,
        "device": "cpu",
        "threads": 2,
    }
    calls, prep_seeds = [], []
    original_train = api._train
    original_init = api.RealMLPPreprocessor.__init__

    def trace_train(*args, **kwargs):
        bound = inspect.signature(original_train).bind(*args, **kwargs).arguments
        calls.append(bound)
        return original_train(*args, **kwargs)

    def trace_init(self, seed):
        prep_seeds.append(seed)
        original_init(self, seed)

    monkeypatch.setattr(api, "_train", trace_train)
    monkeypatch.setattr(api.RealMLPPreprocessor, "__init__", trace_init)
    path = tmp_path / "model"
    prediction, details, reload = candidate.fit_full_neural(
        api, config, frame, y, test, 3, path
    )
    assert prep_seeds == [42]
    assert len(calls) == 1
    call = calls[0]
    assert call["validation"] is None and call["numbers"].n_samples == len(frame)
    np.testing.assert_array_equal(call["y"], y)
    assert (call["seed"], call["device"], call["stop_epoch"]) == (10042, "cpu", 3)
    assert call["settings"].epochs == 3
    assert prediction.dtype == np.float64 and prediction.shape == (len(test),)
    assert (
        np.isfinite(prediction).all() and ((prediction >= 0) & (prediction <= 1)).all()
    )
    assert details["refit_rows"] == len(frame) and details["validation_rows"] == 0
    assert details["refit"]["completed_epochs"] == details["epochs"] == 3
    assert details["preprocessing"]["training_rows"] == len(frame)
    assert "selected_epoch" not in details and "inner_preprocessing" not in details
    assert reload["rows"] == 1024 and reload["max_abs_error"] <= reload["atol"] == 1e-6
    assert reload["rtol"] == 0
    assert reload["checkpoint_sha256"] == candidate.digest(path.with_suffix(".pkl"))
    with path.with_suffix(".pkl").open("rb") as stream:
        saved = cloudpickle.load(stream)
    assert all(
        tensor.device.type == "cpu" for tensor in saved["model"].model.parameters()
    )
    assert all(
        tensor.device.type == "cpu"
        for tensor in saved["model"].model.creator.static_model.parameters()
    )
    loaded = api.predict_saved_realmlp(path, test.iloc[:1024], "cpu")
    np.testing.assert_array_equal(loaded, prediction[:1024])
