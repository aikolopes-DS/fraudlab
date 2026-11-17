"""TabM with inner epoch selection and fresh outer-training refits.

The outer validation labels are deliberately absent from this module's API.
Architecture: https://github.com/yandex-research/tabm (Apache-2.0).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .models import NeuralPreprocessor, clean_features
from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins
from tabm import TabM


@dataclass(frozen=True)
class TabMConfig:
    n_blocks: int = 2
    d_block: int = 256
    k: int = 16
    dropout: float = 0.1
    d_embedding: int = 12
    n_bins: int = 48
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    epochs: int = 40
    patience: int = 6
    gradient_clip: float = 1.0
    inner_fraction: float = 0.15
    inner_seed: int = 42
    numeric_transform: str = "standard"
    target_smoothing: float = 20.0

    @classmethod
    def from_training_config(cls, config: dict):
        if config.get("fixed_training") or config.get("refit"):
            raise ValueError(
                "TabM requires inner epoch selection; fixed-training/full-data refit is not defined."
            )
        if config.get("neural_categorical_transform", "target") != "target":
            raise ValueError("TabM requires --neural-categorical-transform target.")
        if config.get("hidden_dims", [256, 256]) != [256, 256]:
            raise ValueError("TabM architecture is fixed: --hidden-dims 256 256.")
        return cls(
            dropout=config["dropout"],
            learning_rate=config["neural_learning_rate"],
            weight_decay=config["weight_decay"],
            batch_size=config["batch_size"],
            epochs=config["epochs"],
            patience=config["patience"],
            numeric_transform=config.get("neural_numeric_transform", "standard"),
            target_smoothing=config["target_smoothing"],
        )


def build_tabm(bins: list[torch.Tensor], config: TabMConfig) -> TabM:
    return TabM.make(
        n_num_features=len(bins),
        d_out=1,
        num_embeddings=PiecewiseLinearEmbeddings(
            bins, config.d_embedding, activation=False, version="B"
        ),
        arch_type="tabm",
        n_blocks=config.n_blocks,
        d_block=config.d_block,
        k=config.k,
        dropout=config.dropout,
    )


def member_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """One BCE per row AND member. Never BCE on the average prediction."""
    if logits.ndim != 3 or logits.shape[-1] != 1:
        raise ValueError("Expected [batch, k, 1] logits.")
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits[..., 0], y[:, None].expand_as(logits[..., 0])
    )


def mean_probability(logits: torch.Tensor) -> torch.Tensor:
    return logits[..., 0].sigmoid().mean(dim=1)


def prepare_training(frame: pd.DataFrame, y: np.ndarray, seed: int, config: TabMConfig):
    preprocessing = NeuralPreprocessor(
        numeric_transform=config.numeric_transform,
        seed=seed,
        categorical_transform="target",
        smoothing=config.target_smoothing,
    )
    numbers, categories = preprocessing.fit_transform(frame, y)
    assert categories.shape == (len(frame), 0)
    # The official binning API rejects constants. Select using training only;
    # the same fixed mask is applied to every later partition and checkpoint.
    active = np.any(numbers != numbers[0], axis=0)
    if not active.any():
        raise ValueError("TabM needs at least one nonconstant training feature.")
    numbers = np.ascontiguousarray(numbers[:, active])
    bins = compute_bins(torch.from_numpy(numbers), n_bins=config.n_bins)
    return preprocessing, active, numbers, bins


@torch.no_grad()
def predict_array(model, numbers: torch.Tensor, batch_size: int) -> np.ndarray:
    model.eval()
    return np.concatenate(
        [
            mean_probability(model(batch)).cpu().numpy()
            for batch in numbers.split(batch_size)
        ]
    )


def _train(numbers, y, bins, config, seed, device, *, validation=None, epochs=None):
    """Select E on inner validation, or train exactly E from a new seed state."""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = build_tabm(bins, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    x = torch.as_tensor(numbers, device=device)
    labels = torch.as_tensor(y, dtype=torch.float32, device=device)
    best_auc, best_epoch, stale, history = -np.inf, 0, 0, []
    for epoch in range(1, (epochs or config.epochs) + 1):
        model.train()
        for selected in torch.randperm(len(y), device=device).split(config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = member_loss(model(x[selected]), labels[selected])
            if not torch.isfinite(loss):
                raise ValueError("TabM loss became nonfinite.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
        if validation is None:
            best_epoch = epoch
            continue
        vx, vy = validation
        score = roc_auc_score(vy, predict_array(model, vx, config.batch_size))
        history.append({"epoch": epoch, "inner_auc": float(score)})
        print(f"tabm seed={seed} inner_epoch={epoch} roc_auc={score:.8f}", flush=True)
        if score > best_auc + 1e-6:
            best_auc, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
        if stale >= config.patience:
            break
    return model, best_epoch, history, epoch


def fit_tabm(
    training: pd.DataFrame,
    y: np.ndarray,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    config: dict,
    seed: int,
    path: Path,
):
    """Outer validation is inference-only: its labels cannot enter this API."""
    settings = TabMConfig.from_training_config(config)
    torch.set_num_threads(config["threads"])
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA was requested but is not available in this PyTorch installation."
        )
    device = torch.device(
        "cuda" if config["device"] != "cpu" and torch.cuda.is_available() else "cpu"
    )
    inner_train, inner_valid = train_test_split(
        np.arange(len(y)),
        test_size=settings.inner_fraction,
        random_state=settings.inner_seed,
        stratify=y,
    )
    inner_pre, inner_active, inner_x, inner_bins = prepare_training(
        training.iloc[inner_train], y[inner_train], seed, settings
    )
    inner_vx, _ = inner_pre.transform(training.iloc[inner_valid])
    selection_model, selected_epoch, history, selection_epochs_completed = _train(
        inner_x,
        y[inner_train],
        inner_bins,
        settings,
        seed,
        device,
        validation=(
            torch.as_tensor(inner_vx[:, inner_active], device=device),
            y[inner_valid],
        ),
    )
    # Discard ALL inner preprocessing, bins, weights and optimizer state.
    del selection_model, inner_pre, inner_active, inner_x, inner_bins, inner_vx
    print(f"tabm seed={seed} fresh_outer_refit_epochs={selected_epoch}", flush=True)
    preprocessing, active, outer_x, bins = prepare_training(training, y, seed, settings)
    model, _, _, refit_epochs_completed = _train(
        outer_x, y, bins, settings, seed, device, epochs=selected_epoch
    )
    predictions = []
    for frame in (validation, test):
        numbers, _ = preprocessing.transform(frame)
        predictions.append(
            predict_array(
                model,
                torch.as_tensor(numbers[:, active], device=device),
                settings.batch_size,
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"preprocessing": preprocessing, "active": active},
        path.with_suffix(".preprocessing.joblib"),
    )
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "bins": [b.cpu() for b in bins],
            "config": asdict(settings),
            "selected_epoch": selected_epoch,
        },
        path.with_suffix(".pt"),
    )
    return *predictions, {
        "architecture": {
            "name": "TabM",
            "n_blocks": settings.n_blocks,
            "d_block": settings.d_block,
            "k": settings.k,
            "dropout": settings.dropout,
            "d_out": 1,
            "num_embeddings": {
                "name": "PiecewiseLinearEmbeddings",
                "version": "B",
                "activation": False,
                "d_embedding": settings.d_embedding,
                "n_bins": settings.n_bins,
            },
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        "config": asdict(settings),
        "selected_epoch": selected_epoch,
        "selection_epochs_completed": selection_epochs_completed,
        "refit_epochs_completed": refit_epochs_completed,
        "inner_history": history,
        "inner_training_rows": len(inner_train),
        "inner_validation_rows": len(inner_valid),
        "refit_rows": len(training),
        "input_features": len(active),
        "active_features": int(active.sum()),
        "active_columns": np.flatnonzero(active).tolist(),
        "numeric_features": preprocessing.numeric,
        "categorical_features": preprocessing.categorical,
        "bin_edges_per_active_feature": [len(b) for b in bins],
        "optimizer": "AdamW",
        "loss": "mean BCEWithLogits over rows and members",
        "inference": "mean sigmoid probability over members",
        "learning_rate": settings.learning_rate,
        "weight_decay": settings.weight_decay,
        "batch_size": settings.batch_size,
        "device": str(device),
        "seed": seed,
        "threads": torch.get_num_threads(),
        "target_encoder": {
            "internal_folds": 5,
            "smooth": settings.target_smoothing,
            "log_odds": True,
            "training_values": "crossfit",
        },
        "numeric_transform": settings.numeric_transform,
        "bins_fit": "training only per stage; quantiles, no labels",
        "epoch_selection": "inner 15% AUC; fresh full outer-training refit",
        "outer_validation_usage": "inference only; no labels accepted",
    }


def predict_saved_tabm(path: Path, frame: pd.DataFrame, device: str = "cpu"):
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA was requested but is not available in this PyTorch installation."
        )
    device = "cuda" if device != "cpu" and torch.cuda.is_available() else "cpu"
    payload = torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=True)
    config = TabMConfig(**payload["config"])
    model = build_tabm(payload["bins"], config).to(device)
    model.load_state_dict(payload["state_dict"])
    saved = joblib.load(path.with_suffix(".preprocessing.joblib"))
    preprocessing = saved["preprocessing"]
    numbers, _ = preprocessing.transform(
        clean_features(frame, preprocessing.numeric, preprocessing.categorical)
    )
    return predict_array(
        model,
        torch.as_tensor(numbers[:, saved["active"]], device=device),
        config.batch_size,
    )
