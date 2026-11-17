"""Official RealMLP with train-only statistics and fresh outer refits.

RealMLP: https://github.com/dholzmueller/pytabkit (Apache-2.0).
Fixed representation/recipe adapted from yekenot's public PyTabKit notebook v15:
https://www.kaggle.com/code/yekenot/ps-s6-e11-realmlp-pytabkit
Outer validation labels are deliberately absent from this module's API.
"""

from __future__ import annotations

import gc
import importlib.metadata
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
import torch
from pytabkit.models.alg_interfaces.base import InterfaceResources, SplitIdxs
from pytabkit.models.alg_interfaces.nn_interfaces import NNAlgInterface
from pytabkit.models.data.conversion import ToDictDatasetConverter
from pytabkit.models.data.data import DictDataset, TensorInfo
from pytabkit.models.sklearn.default_params import DefaultParams
from pytabkit.models.training.logging import StdoutLogger
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer, TargetEncoder

from .models import clean_features
from .strings import categorical_strings

NUMERIC = (
    "amount_usd",
    "hours_since_last_txn",
    "txn_count_last_24h",
    "distance_from_home_km",
    "card_age_months",
    "customer_age",
    "account_balance_usd",
    "cvv_retry_count",
    "velocity_score",
    "time_of_day_hour",
    "day_of_week",
    "merchant_risk_score",
    "prior_disputes",
)
CATEGORICAL = (
    "merchant_category",
    "card_type",
    "auth_method",
    "channel",
    "device_type",
    "is_foreign_transaction",
    "is_new_merchant",
    "used_vpn",
    "ip_country_mismatch",
    "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt",
)
DECIMALS = (
    "amount_usd",
    "hours_since_last_txn",
    "velocity_score",
    "distance_from_home_km",
)
BINS = (
    ("merchant_risk_score", 7),
    ("hours_since_last_txn", 500),
    ("hours_since_last_txn", 1000),
)
COMBOS = (
    ("merchant_category", "card_type", "txn_count_last_24h"),
    ("merchant_category", "customer_age"),
)
TE_COLUMNS = ("combo_0", "combo_1", "hours_since_last_txn")
METRIC = "1-auc_ovr"


@dataclass(frozen=True)
class RealMLPConfig:
    epochs: int = 3
    batch_size: int = 288
    learning_rate: float = 0.073
    weight_decay: float = 0.038
    dropout: float = 0.05
    inner_fraction: float = 0.15
    inner_seed: int = 42

    @classmethod
    def from_training_config(cls, config: dict):
        if config.get("fixed_training") or config.get("refit"):
            raise ValueError(
                "RealMLP requires inner selection; full-data refit is undefined."
            )
        required = {
            "feature_set": "raw",
            "hidden_dims": [512, 256, 128],
            "neural_categorical_transform": "embedding",
            "neural_numeric_transform": "realmlp",
            "target_smoothing": "auto",
        }
        for key, expected in required.items():
            if config.get(key) != expected:
                raise ValueError(f"RealMLP requires {key}={expected!r}.")
        epochs = config.get("epochs", 3)
        if not 1 <= epochs <= 3 or config.get("patience", 3) != 3:
            raise ValueError(
                "RealMLP requires epochs in 1..3 and patience=3 (no early termination)."
            )
        return cls(
            epochs=epochs,
            batch_size=config["batch_size"],
            learning_rate=config["neural_learning_rate"],
            weight_decay=config["weight_decay"],
            dropout=config["dropout"],
        )


def official_config(settings: RealMLPConfig, selection: bool) -> dict:
    """Resolve the actual 1.8.0 defaults, including PBLD's linear second layer."""
    if importlib.metadata.version("pytabkit") != "1.8.0":
        raise ValueError("RealMLP requires exactly pytabkit==1.8.0.")
    return {
        **DefaultParams.RealMLP_TD_CLASS,
        "n_ens": 16,
        "n_epochs": settings.epochs,
        "batch_size": settings.batch_size,
        "predict_batch_size": 1024,
        "use_early_stopping": False,
        "ens_av_before_softmax": False,
        "use_best_epoch": selection,
        "val_metric_name": METRIC,
        "lr": settings.learning_rate,
        "wd": settings.weight_decay,
        "sq_mom": 0.988,
        "lr_sched": "flat_anneal",
        "wd_sched": "cos_log_15",
        "first_layer_lr_factor": 0.25,
        "embedding_size": 5,
        "max_one_hot_cat_size": 16,
        "hidden_sizes": [512, 256, 128],
        "act": "silu",
        "p_drop": settings.dropout,
        "p_drop_sched": "pow5",
        "num_emb_type": "pbld",
        "use_plr_embeddings": True,
        "plr_hidden_1": 16,
        "plr_hidden_2": 8,
        "plr_act_name": "linear",
        "plr_use_densenet": True,
        "plr_use_cos_bias": True,
        "plr_lr_factor": 0.1151,
        "plr_sigma": 0.75,
        "ls_eps": 0.01,
        "ls_eps_sched": "sqrt_cos",
        "use_ls_prior": False,
        "smooth_clip_max_abs_value": 3.0,
        "robust_scale_eps": 1e-30,
        "l2_normalize_eps": 1e-8,
        "add_front_scale": False,
        "bias_init_mode": "neg-uniform-dynamic-2",
        "tfms": [
            "one_hot",
            "median_center",
            "robust_scale",
            "smooth_clip",
            "embedding",
            "l2_normalize",
        ],
    }


class RealMLPPreprocessor:
    """53 deterministic views; every learned object is fitted on this stage's train."""

    def __init__(self, seed: int):
        self.seed = seed

    def _base(self, frame: pd.DataFrame) -> pd.DataFrame:
        if set(frame) != set(NUMERIC + CATEGORICAL):
            raise ValueError(
                "RealMLP requires exactly the 24 raw features, without ID/target."
            )
        return clean_features(
            frame.loc[:, list(NUMERIC + CATEGORICAL)], list(NUMERIC), list(CATEGORICAL)
        )

    def _views(self, frame: pd.DataFrame, fit: bool) -> pd.DataFrame:
        result = self._base(frame)
        if fit:
            self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            values = self.imputer.fit_transform(result[list(NUMERIC)])
            self.bins = []
        else:
            values = self.imputer.transform(result[list(NUMERIC)])
        result[list(NUMERIC)] = values
        for name in DECIMALS:
            result[f"decimal_{name}"] = (result[name] % 1).round(2).astype(np.float32)
        for name in NUMERIC:
            floors = np.floor(result[name].to_numpy())
            if np.any(np.abs(floors) >= 2**63):
                raise ValueError("A numerical floor exceeds int64 support.")
            result[f"category_{name}"] = categorical_strings(floors.astype(np.int64))
        for name in DECIMALS:
            cents = np.rint(
                result[f"decimal_{name}"].to_numpy(dtype=np.float64) * 100
            ).astype(np.int64)
            result[f"category_decimal_{name}"] = categorical_strings(cents)
        for index, (name, count) in enumerate(BINS):
            if fit:
                encoder = KBinsDiscretizer(
                    n_bins=count,
                    encode="ordinal",
                    strategy="quantile",
                    subsample=None,
                    quantile_method="linear",
                )
                encoded = encoder.fit_transform(result[[name]])
                self.bins.append(encoder)
            else:
                encoded = self.bins[index].transform(result[[name]])
            result[f"bin_{index}"] = categorical_strings(
                encoded.ravel().astype(np.int64)
            )
        for index, columns in enumerate(COMBOS):
            # Tuple representation avoids ambiguous delimiter concatenations.
            codes, uniques = pd.factorize(
                pd.MultiIndex.from_frame(result[list(columns)]), sort=False
            )
            keys = np.array([repr(tuple(row)) for row in uniques], dtype=object)
            result[f"combo_{index}"] = keys[codes]
        return result

    def fit_transform(self, frame: pd.DataFrame, y: np.ndarray) -> DictDataset:
        self.fit_rows = len(frame)
        result = self._views(frame, fit=True)
        self.target_encoder = TargetEncoder(
            cv=5, smooth="auto", shuffle=True, random_state=self.seed
        )
        result[["te_0", "te_1", "te_2"]] = self.target_encoder.fit_transform(
            result[list(TE_COLUMNS)], y
        )
        self.columns = result.columns.tolist()
        self.categorical = [
            name
            for name in self.columns
            if name not in NUMERIC and not name.startswith(("decimal_", "te_"))
        ]
        self.converter = ToDictDatasetConverter(
            cat_features=[name in self.categorical for name in self.columns]
        )
        return self.converter.fit_transform(result)

    def transform(self, frame: pd.DataFrame) -> DictDataset:
        result = self._views(frame, fit=False)
        result[["te_0", "te_1", "te_2"]] = self.target_encoder.transform(
            result[list(TE_COLUMNS)]
        )
        return self.converter.transform(result[self.columns])

    def metadata(self) -> dict:
        return {
            "columns": self.columns,
            "total_features": 53,
            "numeric_features": 20,
            "categorical_features": 33,
            "training_rows": self.fit_rows,
            "categorical_cardinalities": self.converter.tensor_infos["x_cat"]
            .get_cat_sizes()
            .tolist(),
            "quantile_bin_edges": [
                encoder.bin_edges_[0].tolist() for encoder in self.bins
            ],
            "target_encoding": {
                "columns": list(TE_COLUMNS),
                "smooth": "auto",
                "cv": 5,
                "shuffle": True,
                "seed": self.seed,
                "training": "crossfit fit_transform once",
            },
        }


def _train(
    numbers: DictDataset,
    y: np.ndarray,
    validation,
    settings: RealMLPConfig,
    seed: int,
    device: str,
    threads: int,
    stop_epoch: int | None = None,
):
    """The official implementation owns architecture, optimization and epoch callbacks."""
    started = time.perf_counter()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    tensors = numbers.tensors.copy()
    labels = np.asarray(y)
    if validation is not None:
        valid_numbers, valid_y = validation
        tensors = {
            key: torch.cat((value, valid_numbers.tensors[key]))
            for key, value in tensors.items()
        }
        labels = np.concatenate((labels, valid_y))
    tensors["y"] = torch.as_tensor(labels[:, None], dtype=torch.long)
    ds = DictDataset(tensors, {**numbers.tensor_infos, "y": TensorInfo(cat_sizes=[2])})
    split = SplitIdxs(
        train_idxs=torch.arange(len(y))[None, :],
        val_idxs=(
            torch.arange(len(y), len(labels))[None, :]
            if validation is not None
            else None
        ),
        test_idxs=None,
        split_seed=seed,
        sub_split_seeds=[seed],
        split_id=0,
    )
    parameters = official_config(settings, selection=validation is not None)
    fit_params = None if stop_epoch is None else [{"stop_epoch": {METRIC: stop_epoch}}]
    model = NNAlgInterface(fit_params=fit_params, **parameters)
    model.fit(
        ds,
        [split],
        InterfaceResources(threads, [device] if device.startswith("cuda") else []),
        StdoutLogger(verbosity_level=2),
        [None],
        "FraudLab RealMLP",
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    selected = (
        int(model.fit_params[0]["stop_epoch"][METRIC])
        if validation is not None
        else stop_epoch
    )
    details = {
        "elapsed_seconds": time.perf_counter() - started,
        "parameters": sum(p.numel() for p in model.model.model.parameters()),
        "completed_epochs": int(round(model.model.progress.epoch_float)),
        "selected_epoch": selected,
        "resolved_parameters": parameters,
        "cuda_memory_scope": "reset before official model construction; synchronized after fit; includes train, inner validation and optimizer; excludes outer/test prediction",
        "cuda_max_memory_allocated_bytes": (
            torch.cuda.max_memory_allocated() if device.startswith("cuda") else None
        ),
        "cuda_max_memory_reserved_bytes": (
            torch.cuda.max_memory_reserved() if device.startswith("cuda") else None
        ),
    }
    return model, details


def _predict(model: NNAlgInterface, numbers: DictDataset) -> np.ndarray:
    # NNAlgInterface already returns log(mean(softmax(member logits))).
    return torch.softmax(model.predict(numbers), dim=-1).mean(dim=0)[:, 1].numpy()


def fit_realmlp(
    training: pd.DataFrame,
    y: np.ndarray,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    config: dict,
    seed: int,
    path: Path,
):
    settings = RealMLPConfig.from_training_config(config)
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    device = (
        "cuda:0" if config["device"] != "cpu" and torch.cuda.is_available() else "cpu"
    )
    torch.set_num_threads(config["threads"])
    inner_train, inner_valid = train_test_split(
        np.arange(len(training)),
        test_size=settings.inner_fraction,
        stratify=y,
        random_state=settings.inner_seed,
    )
    inner_started = time.perf_counter()
    preprocessing = RealMLPPreprocessor(seed)
    numbers = preprocessing.fit_transform(training.iloc[inner_train], y[inner_train])
    inner_metadata = preprocessing.metadata()
    model, selection = _train(
        numbers,
        y[inner_train],
        (preprocessing.transform(training.iloc[inner_valid]), y[inner_valid]),
        settings,
        seed,
        device,
        config["threads"],
    )
    epoch = selection["selected_epoch"]
    if not 1 <= epoch <= settings.epochs:
        raise ValueError("RealMLP did not select a finite inner validation epoch.")
    selection["stage_elapsed_seconds"] = time.perf_counter() - inner_started
    del preprocessing, numbers, model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    print(f"RealMLP selected_epoch={epoch}; fresh outer refit", flush=True)
    outer_started = time.perf_counter()
    preprocessing = RealMLPPreprocessor(seed)
    numbers = preprocessing.fit_transform(training, y)
    model, refit = _train(
        numbers,
        y,
        None,
        settings,
        seed + 10000,
        device,
        config["threads"],
        stop_epoch=epoch,
    )
    refit["stage_elapsed_seconds"] = time.perf_counter() - outer_started
    del numbers
    predictions = [
        _predict(model, preprocessing.transform(frame)) for frame in (validation, test)
    ]
    details = {
        "architecture": {
            "library": "pytabkit==1.8.0",
            "name": "RealMLP_TD",
            "n_ens": 16,
            "hidden_dims": [512, 256, 128],
            "activation": "parametric SiLU",
            "numeric_embedding": "PBLD: periodic16, linear7 + original-value skip =8",
            "categorical_embedding_dim": 5,
            "parameters": sum(p.numel() for p in model.model.model.parameters()),
        },
        "config": asdict(settings),
        "preprocessing": preprocessing.metadata(),
        "inner_preprocessing": inner_metadata,
        "selection": selection,
        "refit": refit,
        "selected_epoch": epoch,
        "inner_training_rows": len(inner_train),
        "inner_validation_rows": len(inner_valid),
        "refit_rows": len(training),
        "epoch_selection": f"inner 15% AUC, cap{settings.epochs}; discard all state, fresh outer refit for selected E; LR schedules retain cap{settings.epochs} horizon",
        "outer_validation_usage": "inference only; labels absent from API",
        "prediction": "mean of member softmax probabilities",
        "loss": "sum across16 members of batch-mean cross entropy; label smoothing .01 sqrt_cos",
        "seed": seed,
        "refit_seed": seed + 10000,
        "device": device,
        "threads": config["threads"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _move(model, "cpu")
    # The official scheduler contains lambdas; ordinary pickle/joblib cannot save it.
    # Only locally trained, hash-audited bundles are loaded, as for other model files.
    with path.with_suffix(".pkl").open("wb") as stream:
        cloudpickle.dump(
            {"model": model, "preprocessing": preprocessing, "details": details}, stream
        )
    return *predictions, details


def predict_saved_realmlp(
    path: Path, frame: pd.DataFrame, device: str = "cpu"
) -> np.ndarray:
    if importlib.metadata.version("pytabkit") != "1.8.0":
        raise ValueError("RealMLP requires exactly pytabkit==1.8.0.")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    with path.with_suffix(".pkl").open("rb") as stream:
        saved = cloudpickle.load(stream)
    _move(
        saved["model"],
        "cuda:0" if device != "cpu" and torch.cuda.is_available() else "cpu",
    )
    return _predict(saved["model"], saved["preprocessing"].transform(frame))


def _move(model: NNAlgInterface, device: str) -> None:
    model.to(device)
    # This fitted preprocessing layer belongs to the non-Module creator object.
    model.model.creator.static_model.to(device)
    model.model.creator.device_info = device
