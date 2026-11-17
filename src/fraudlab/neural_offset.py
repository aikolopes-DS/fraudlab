"""A fresh neural correction to a frozen GLM logit, selected inside outer training.

The outer validation labels are deliberately absent from this module's API.
Both models learn from the identical, internally cross-fitted training matrix.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from .models import clean_features, feature_types


@dataclass(frozen=True)
class NeuralOffsetConfig:
    hidden_width: int = 256
    dropout: float = 0.1
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 4096
    epochs: int = 40
    patience: int = 6
    gradient_clip: float = 1.0
    inner_fraction: float = 0.15
    inner_seed: int = 42

    @classmethod
    def from_training_config(cls, config: dict):
        if config.get("fixed_training") or config.get("refit"):
            raise ValueError("Neural offset requires inner epoch selection.")
        hidden_dims = config.get("hidden_dims")
        if hidden_dims is not None and hidden_dims != [256, 256]:
            raise ValueError("Neural offset requires --hidden-dims 256 256.")
        if config.get("glm_value_one_hot"):
            raise ValueError("Neural offset requires the dense GLM representation.")
        if config.get("neural_numeric_transform", "standard") != "standard":
            raise ValueError("Neural offset uses the GLM's standard numeric scaling.")
        if config.get("neural_categorical_transform") not in (None, "target"):
            raise ValueError("Neural offset uses the GLM's cross-fitted target logits.")
        settings = cls(
            dropout=config.get("dropout", 0.1),
            learning_rate=config.get("neural_learning_rate", 0.001),
            weight_decay=config.get("weight_decay", 0.0001),
            batch_size=config.get("batch_size", 4096),
            epochs=config.get("epochs", 40),
            patience=6 if config.get("patience") is None else config["patience"],
        )
        if not 1 <= settings.epochs <= 40 or settings.patience < 1:
            raise ValueError("Neural offset needs 1..40 epochs and positive patience.")
        if settings.batch_size < 1 or settings.learning_rate <= 0:
            raise ValueError("Batch size and learning rate must be positive.")
        if not 0 <= settings.dropout < 1 or settings.weight_decay < 0:
            raise ValueError("Invalid dropout or weight decay.")
        return settings


def build_residual(input_features: int, settings: NeuralOffsetConfig):
    width = settings.hidden_width
    model = torch.nn.Sequential(
        torch.nn.Linear(input_features, width),
        torch.nn.SiLU(),
        torch.nn.Dropout(settings.dropout),
        torch.nn.Linear(width, width),
        torch.nn.SiLU(),
        torch.nn.Dropout(settings.dropout),
        torch.nn.Linear(width, 1),
    )
    # The initial combined predictor is the GLM with its margin cast to float32.
    torch.nn.init.zeros_(model[-1].weight)
    torch.nn.init.zeros_(model[-1].bias)
    return model


def prepare_training(frame, y, config, seed):
    # Local import permits later lazy dispatch without a circular import.
    from .models import make_glm_pipeline

    glm = make_glm_pipeline(frame, config, seed)
    preprocessing = glm.named_steps["preprocessing"]
    classifier = glm.named_steps["classifier"]
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        with threadpool_limits(limits=config["threads"]):
            encoded = preprocessing.fit_transform(frame, y)
            classifier.fit(encoded, y)
            margin = classifier.decision_function(encoded)
    for warning in captured:
        warnings.warn(warning.message, warning.category, stacklevel=2)
    if (
        any(issubclass(w.category, ConvergenceWarning) for w in captured)
        or int(classifier.n_iter_[0]) >= classifier.max_iter
    ):
        raise RuntimeError(
            "Neural offset GLM did not converge; candidate invalid under fixed protocol."
        )
    # NEVER transform training again with the full target maps.
    return glm, _float_matrix(encoded), np.asarray(margin, dtype=np.float32)


def _float_matrix(encoded):
    if hasattr(encoded, "toarray"):
        raise ValueError("Neural offset requires a dense GLM training matrix.")
    matrix = np.ascontiguousarray(encoded, dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Neural offset received nonfinite encoded features.")
    return matrix


def transform_frame(glm, frame):
    encoded = glm.named_steps["preprocessing"].transform(frame)
    margin = glm.named_steps["classifier"].decision_function(encoded)
    return _float_matrix(encoded), np.asarray(margin, dtype=np.float32)


@torch.no_grad()
def predict_array(model, numbers, margin, batch_size):
    model.eval()
    return np.concatenate(
        [
            (offset + model(batch).squeeze(-1)).sigmoid().cpu().numpy()
            for batch, offset in zip(
                numbers.split(batch_size), margin.split(batch_size)
            )
        ]
    )


def _train(numbers, y, margin, settings, seed, device, *, validation=None, epochs=None):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = build_residual(numbers.shape[1], settings).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    x = torch.as_tensor(numbers, device=device)
    labels = torch.as_tensor(y, dtype=torch.float32, device=device)
    offsets = torch.as_tensor(margin, dtype=torch.float32, device=device).detach()
    best_auc, best_epoch, stale, history = -np.inf, 0, 0, []
    for epoch in range(1, (epochs or settings.epochs) + 1):
        model.train()
        for selected in torch.randperm(len(y), device=device).split(
            settings.batch_size
        ):
            optimizer.zero_grad(set_to_none=True)
            logits = offsets[selected] + model(x[selected]).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, labels[selected]
            )
            if not torch.isfinite(loss):
                raise ValueError("Neural offset loss became nonfinite.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip)
            optimizer.step()
        if validation is None:
            best_epoch = epoch
            continue
        vx, vy, vm = validation
        score = roc_auc_score(vy, predict_array(model, vx, vm, settings.batch_size))
        history.append({"epoch": epoch, "inner_auc": float(score)})
        print(
            f"neural_offset seed={seed} inner_epoch={epoch} roc_auc={score:.8f}",
            flush=True,
        )
        if score > best_auc + 1e-6:
            best_auc, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
        if stale >= settings.patience:
            break
    return model, best_epoch, history, epoch


def fit_neural_offset(training, y, validation, test, config, seed, path: Path):
    """Choose E internally, discard that stage, and refit both models from scratch."""
    settings = NeuralOffsetConfig.from_training_config(config)
    config = {
        "feature_set": "bpe_digits",
        "glm_frequency": True,
        "glm_pooled_tokens": True,
        "glm_inner_folds": 5,
        "glm_spline_knots": 0,
        "target_smoothing": 50.0,
        "logistic_c": 0.3,
        "threads": 4,
        **config,
    }
    torch.set_num_threads(config["threads"])
    requested_device = config.get("device", "cpu")
    if requested_device not in ("cpu", "cuda", "auto"):
        raise ValueError("Unknown neural offset device.")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    device = torch.device(
        "cuda" if requested_device != "cpu" and torch.cuda.is_available() else "cpu"
    )
    numeric, categorical = feature_types(training)
    training = clean_features(training, numeric, categorical)
    y = np.asarray(y)
    inner_train, inner_valid = train_test_split(
        np.arange(len(y)),
        test_size=settings.inner_fraction,
        random_state=settings.inner_seed,
        stratify=y,
    )
    inner_glm, inner_x, inner_margin = prepare_training(
        training.iloc[inner_train], y[inner_train], config, seed
    )
    inner_classifier = inner_glm.named_steps["classifier"]
    inner_glm_iterations = int(inner_classifier.n_iter_[0])
    inner_glm_max_iter = inner_classifier.max_iter
    inner_vx, inner_vm = transform_frame(inner_glm, training.iloc[inner_valid])
    selection_model, selected_epoch, history, selection_epochs = _train(
        inner_x,
        y[inner_train],
        inner_margin,
        settings,
        seed,
        device,
        validation=(
            torch.as_tensor(inner_vx, device=device),
            y[inner_valid],
            torch.as_tensor(inner_vm, device=device),
        ),
    )
    # No preprocessing, GLM, network weights or optimizer crosses this boundary.
    del (
        selection_model,
        inner_glm,
        inner_classifier,
        inner_x,
        inner_margin,
        inner_vx,
        inner_vm,
    )
    print(
        f"neural_offset seed={seed} fresh_outer_refit_epochs={selected_epoch}",
        flush=True,
    )
    glm, numbers, margin = prepare_training(training, y, config, seed)
    model, _, _, refit_epochs = _train(
        numbers, y, margin, settings, seed, device, epochs=selected_epoch
    )
    predictions = []
    for frame in (validation, test):
        vx, vm = transform_frame(glm, clean_features(frame, numeric, categorical))
        predictions.append(
            predict_array(
                model,
                torch.as_tensor(vx, device=device),
                torch.as_tensor(vm, device=device),
                settings.batch_size,
            )
        )
    epoch_metadata = {
        "selected_epoch": selected_epoch,
        "selection_epochs_completed": selection_epochs,
        "refit_epochs_completed": refit_epochs,
        "inner_history": history,
        "inner_training_rows": len(inner_train),
        "inner_validation_rows": len(inner_valid),
        "refit_rows": len(training),
        "inner_glm_iterations_used": inner_glm_iterations,
        "inner_glm_converged": inner_glm_iterations < inner_glm_max_iter,
    }
    architecture = {
        "name": "GLM_logit_plus_neural_residual",
        "input_features": numbers.shape[1],
        "hidden_dims": [256, 256],
        "activation": "SiLU",
        "dropout": settings.dropout,
        "zero_initialized_head": True,
        "glm_margin_dtype": "float32",
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"glm": glm, "numeric": numeric, "categorical": categorical},
        path.with_suffix(".glm.joblib"),
    )
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "config": asdict(settings),
            "architecture": architecture,
            "epoch_metadata": epoch_metadata,
        },
        path.with_suffix(".pt"),
    )
    return *predictions, {
        "architecture": architecture,
        "config": asdict(settings),
        **epoch_metadata,
        "glm": {
            key: config[key]
            for key in (
                "feature_set",
                "logistic_c",
                "target_smoothing",
                "glm_inner_folds",
                "glm_frequency",
                "glm_pooled_tokens",
                "glm_spline_knots",
            )
        },
        "glm_iterations_used": int(glm.named_steps["classifier"].n_iter_[0]),
        "glm_converged": int(glm.named_steps["classifier"].n_iter_[0])
        < glm.named_steps["classifier"].max_iter,
        "optimizer": "AdamW",
        "loss": "unweighted mean BCEWithLogits",
        "prediction": "sigmoid(frozen GLM decision_function + neural residual)",
        "training_encoding": "same cross-fitted matrix for GLM and residual; no full-map training retransform",
        "epoch_selection": "inner 15% AUC; fresh GLM, preprocessing, network and AdamW for full outer refit",
        "outer_validation_usage": "inference only; no labels accepted",
        "seed": seed,
        "threads": torch.get_num_threads(),
        "device": str(device),
    }


def predict_saved_neural_offset(path: Path, frame: pd.DataFrame, device="cpu"):
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    device = "cuda" if device != "cpu" and torch.cuda.is_available() else "cpu"
    payload = torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=True)
    settings = NeuralOffsetConfig(**payload["config"])
    model = build_residual(payload["architecture"]["input_features"], settings).to(
        device
    )
    model.load_state_dict(payload["state_dict"])
    saved = joblib.load(path.with_suffix(".glm.joblib"))
    numbers, margin = transform_frame(
        saved["glm"], clean_features(frame, saved["numeric"], saved["categorical"])
    )
    return predict_array(
        model,
        torch.as_tensor(numbers, device=device),
        torch.as_tensor(margin, device=device),
        settings.batch_size,
    )
