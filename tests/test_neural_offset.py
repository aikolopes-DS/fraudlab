"""CPU-only synthetic checks of the residual logit and nested fitting boundary."""

import inspect
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

torch = pytest.importorskip("torch")

from fraudlab import models
from fraudlab import neural_offset as candidate
from fraudlab.__main__ import parser, validate_training_config


def config(**changes):
    settings = {
        "feature_set": "raw",
        "glm_pooled_tokens": False,
        "glm_frequency": False,
        "target_smoothing": 20.0,
        "logistic_c": 0.3,
        "epochs": 2,
        "patience": 1,
        "batch_size": 64,
        "device": "cpu",
        "threads": 1,
    }
    settings.update(changes)
    return settings


def toy_frame(size=320):
    rng = np.random.default_rng(8)
    x = rng.normal(size=size)
    frame = pd.DataFrame(
        {"value": x, "category": rng.choice(["a", "b", "c"], size)},
        index=np.arange(1000, 1000 + size),
    )
    y = (x**2 + rng.normal(scale=0.2, size=size) > 0.7).astype(int)
    return frame, y


class TrainingTransformGuard(BaseEstimator, TransformerMixin):
    """A unique key memorizes its own target if training is transformed again."""

    def fit_transform(self, frame, y):
        self.training_index_ = frame.index.copy()
        self.encoder_ = TargetEncoder(
            target_type="binary", smooth=0, cv=5, shuffle=True, random_state=42
        )
        self.encoded_training_ = self.encoder_.fit_transform(frame, y)
        self.labels_ = np.array(y)
        return self.encoded_training_

    def transform(self, frame):
        if frame.index.equals(self.training_index_):
            raise AssertionError("Training rows re-encoded using full target maps.")
        return self.encoder_.transform(frame)


def test_zero_head_starts_at_glm_and_bce_uses_sum_of_logits():
    settings = candidate.NeuralOffsetConfig()
    defaults = candidate.NeuralOffsetConfig.from_training_config(
        {"patience": None, "hidden_dims": None}
    )
    assert defaults.patience == 6 and defaults.hidden_width == 256
    cli = vars(
        parser().parse_args(
            [
                "train",
                "--model",
                "neural_offset",
                "--name",
                "toy",
                "--validation-note",
                "Synthetic CPU test",
                "--glm-frequency",
                "--glm-pooled-tokens",
                "--feature-set",
                "bpe_digits",
            ]
        )
    )
    validate_training_config(cli)
    assert cli["hidden_dims"] == [256, 256] and cli["patience"] == 6
    assert cli["neural_categorical_transform"] == "target"
    assert (
        cli["pooled_token_encoding"]["algorithm"]
        == "hierarchical_pooled_decimal_token_rates"
    )
    with pytest.raises(ValueError, match="inner epoch"):
        validate_training_config({**cli, "fixed_training": True})
    model = candidate.build_residual(3, settings)
    x = torch.tensor([[1.0, 0.0, 2.0], [-1.0, 2.0, 0.0]])
    base = torch.tensor([2.0, -1.0])
    labels = torch.tensor([0.0, 1.0])
    residual = model(x).squeeze(-1)
    torch.testing.assert_close(residual, torch.zeros(2), rtol=0, atol=0)
    logits = base + residual
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    manual = (torch.nn.functional.softplus(logits) - labels * logits).mean()
    torch.testing.assert_close(loss, manual)
    loss.backward()
    assert torch.count_nonzero(model[-1].weight.grad) > 0
    assert torch.count_nonzero(model[0].weight.grad) == 0
    assert not np.isclose(loss.item(), np.log(2))
    with pytest.raises(ValueError, match="inner epoch"):
        candidate.NeuralOffsetConfig.from_training_config({"fixed_training": True})
    with pytest.raises(ValueError, match="1..40"):
        candidate.NeuralOffsetConfig.from_training_config({"epochs": 41})
    with pytest.raises(ValueError, match="numeric scaling"):
        candidate.NeuralOffsetConfig.from_training_config(
            {"neural_numeric_transform": "quantile"}
        )


def test_training_never_retransforms_self_target_keys(tmp_path, monkeypatch):
    created = []

    def guarded_pipeline(training, settings, seed):
        model = Pipeline(
            [
                ("preprocessing", TrainingTransformGuard()),
                ("classifier", LogisticRegression(C=0.3, solver="newton-cholesky")),
            ]
        )
        created.append(model)
        return model

    monkeypatch.setattr(models, "make_glm_pipeline", guarded_pipeline)
    training = pd.DataFrame({"key": [f"unique-{i}" for i in range(120)]})
    y = np.tile([0, 1], 60)
    external = pd.DataFrame({"key": ["unseen"]}, index=[9000])
    candidate.fit_neural_offset(
        training, y, external, external.copy(), config(epochs=1), 42, tmp_path / "guard"
    )
    assert len(created) == 2 and created[0] is not created[1]
    for model in created:
        guard = model.named_steps["preprocessing"]
        # All category keys are unique. Full-map inference would reveal y exactly.
        assert not np.array_equal(guard.encoded_training_[:, 0], guard.labels_)
        keys = training.loc[guard.training_index_]
        np.testing.assert_array_equal(
            guard.encoder_.transform(keys)[:, 0], guard.labels_
        )


def test_inner_only_preprocessing_and_fresh_outer_weights_optimizer(
    tmp_path, monkeypatch
):
    frame, y = toy_frame()
    external = pd.DataFrame({"value": [1e8], "category": ["unseen"]}, index=[9000])
    stages, glms, networks, optimizers, training_calls = [], [], [], [], []
    prepare, build, train = (
        candidate.prepare_training,
        candidate.build_residual,
        candidate._train,
    )
    adam = torch.optim.AdamW

    def traced_prepare(frame, labels, settings, seed):
        result = prepare(frame, labels, settings, seed)
        stages.append(frame.index.to_numpy())
        glms.append(result[0])
        return result

    def traced_build(width, settings):
        model = build(width, settings)
        networks.append(
            (model, {k: v.detach().clone() for k, v in model.state_dict().items()})
        )
        return model

    def traced_adam(parameters, **kwargs):
        optimizer = adam(parameters, **kwargs)
        optimizers.append(optimizer)
        return optimizer

    def traced_train(*args, **kwargs):
        training_calls.append(
            (len(args[1]), kwargs.get("epochs"), kwargs.get("validation"))
        )
        return train(*args, **kwargs)

    monkeypatch.setattr(candidate, "prepare_training", traced_prepare)
    monkeypatch.setattr(candidate, "build_residual", traced_build)
    monkeypatch.setattr(candidate, "_train", traced_train)
    monkeypatch.setattr(torch.optim, "AdamW", traced_adam)
    _, _, details = models.fit_predict(
        "neural_offset",
        frame,
        y,
        external,
        object(),
        external.copy(),
        config(),
        42,
        tmp_path / "nested",
    )
    inner, inner_valid = train_test_split(
        np.arange(len(y)), test_size=0.15, random_state=42, stratify=y
    )
    np.testing.assert_array_equal(stages[0], frame.iloc[inner].index)
    np.testing.assert_array_equal(stages[1], frame.index)
    assert not set(stages[0]) & set(frame.iloc[inner_valid].index)
    assert not set(stages[1]) & set(external.index)
    for glm, rows in zip(glms, (inner, np.arange(len(y)))):
        scaler = (
            glm.named_steps["preprocessing"]
            .named_transformers_["numeric"]
            .named_steps["scale"]
        )
        np.testing.assert_allclose(scaler.mean_, [frame.iloc[rows].value.mean()])
    assert glms[0] is not glms[1]
    assert networks[0][0] is not networks[1][0]
    assert optimizers[0] is not optimizers[1]
    # Both networks start from the same fresh seed, rather than selected weights.
    for key in networks[0][1]:
        torch.testing.assert_close(
            networks[0][1][key], networks[1][1][key], rtol=0, atol=0
        )
    assert training_calls[0][1] is None and len(training_calls[0][2][1]) == len(
        inner_valid
    )
    assert training_calls[1][1] == details["selected_epoch"]
    assert training_calls[1][2] is None
    assert details["refit_epochs_completed"] == details["selected_epoch"]
    assert details["refit_rows"] == len(frame)
    assert details["device"] == "cpu" and details["threads"] == 1
    assert (
        "validation_y" not in inspect.signature(candidate.fit_neural_offset).parameters
    )


def test_reload_preserves_glm_plus_neural_logit_and_missing_types(tmp_path):
    frame, y = toy_frame()
    external = pd.DataFrame(
        {"value": [0.0, 1.0, -1.0, np.nan], "category": ["unseen", "a", "b", "c"]},
        index=np.arange(9000, 9004),
    )
    path = tmp_path / "fold-0"
    valid, test, details = candidate.fit_neural_offset(
        frame, y, external, external.copy(), config(), 42, path
    )
    reloaded = models.predict_saved("neural_offset", path, external, "cpu")
    np.testing.assert_allclose(valid, reloaded, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(valid, test)
    saved = joblib.load(path.with_suffix(".glm.joblib"))
    payload = torch.load(path.with_suffix(".pt"), weights_only=True)
    model = candidate.build_residual(
        payload["architecture"]["input_features"],
        candidate.NeuralOffsetConfig(**payload["config"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    x, margin = candidate.transform_frame(
        saved["glm"],
        models.clean_features(external, saved["numeric"], saved["categorical"]),
    )
    with torch.no_grad():
        residual = model(torch.from_numpy(x)).squeeze(-1)
        expected = (torch.from_numpy(margin) + residual).sigmoid().numpy()
    np.testing.assert_allclose(valid, expected, rtol=0, atol=1e-7)
    assert residual.abs().max().item() > 1e-6
    assert not np.allclose(valid, residual.sigmoid().numpy())
    missing = external.assign(value=None)
    result = candidate.predict_saved_neural_offset(path, missing)
    np.testing.assert_allclose(
        result,
        candidate.predict_saved_neural_offset(path, missing.assign(value=np.nan)),
        rtol=0,
        atol=1e-7,
    )
    assert np.isfinite(result).all()
    assert payload["epoch_metadata"]["selected_epoch"] == details["selected_epoch"]


@pytest.mark.parametrize("warning_stage", [1, 2])
def test_either_glm_convergence_warning_invalidates_candidate(
    tmp_path, monkeypatch, warning_stage
):
    frame, y = toy_frame()
    external = frame.iloc[:2].set_axis([9000, 9001])
    original_factory = models.make_glm_pipeline
    calls = []

    def limited_pipeline(training, settings, seed):
        glm = original_factory(training, settings, seed)
        calls.append(glm)
        if len(calls) == warning_stage:
            # A real sklearn fit must emit its real convergence warning.
            glm.named_steps["classifier"].set_params(max_iter=1)
        return glm

    monkeypatch.setattr(models, "make_glm_pipeline", limited_pipeline)
    path = tmp_path / "invalid"
    with (
        pytest.warns(ConvergenceWarning),
        pytest.raises(RuntimeError, match="did not converge"),
    ):
        candidate.fit_neural_offset(
            frame, y, external, external, config(epochs=1), 42, path
        )
    assert len(calls) == warning_stage
    assert not path.with_suffix(".pt").exists()
    assert not path.with_suffix(".glm.joblib").exists()


def test_refit_rejects_neural_offset_before_reading_holdout(tmp_path):
    from fraudlab.refit import refit

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model": "neural_offset"}))
    with pytest.raises(ValueError, match="full-data refit protocol is not defined"):
        refit(source, tmp_path / "absent-holdout.json", tmp_path / "runs", "invalid")
    assert not (tmp_path / "runs").exists()
