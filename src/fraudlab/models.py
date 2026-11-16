"""Small tabular models, each fitted inside its training fold."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    QuantileTransformer,
    StandardScaler,
    TargetEncoder,
)
from threadpoolctl import threadpool_limits


def feature_types(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = frame.select_dtypes(include="number").columns.tolist()
    return numeric, [name for name in frame if name not in numeric]


def clean_features(
    frame: pd.DataFrame,
    numeric: list[str] | None = None,
    categorical: list[str] | None = None,
) -> pd.DataFrame:
    result = frame.copy()
    if numeric is None or categorical is None:
        numeric, categorical = feature_types(result)
    for name in numeric:
        result[name] = pd.to_numeric(result[name], errors="raise")
    result[numeric] = result[numeric].replace([np.inf, -np.inf], np.nan)
    for name in categorical:
        result[name] = result[name].astype("string").fillna("__MISSING__").astype(str)
    return result


def log_odds(values: np.ndarray) -> np.ndarray:
    probabilities = np.clip(values, 1e-6, 1 - 1e-6)
    return np.log(probabilities / (1 - probabilities))


def fit_logistic(
    training: pd.DataFrame,
    y: np.ndarray,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    config: dict,
    seed: int,
    path: Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    numeric, categorical = feature_types(training)
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="median", keep_empty_features=True),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore", min_frequency=5, dtype=np.float32
                ),
                categorical,
            ),
        ],
        sparse_threshold=1.0,
    )
    model = Pipeline(
        [
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    C=config["logistic_c"], max_iter=1000, random_state=seed
                ),
            ),
        ]
    )
    with threadpool_limits(limits=config["threads"]):
        model.fit(training, y)
    joblib.dump(model, path.with_suffix(".joblib"))
    return (
        model.predict_proba(validation)[:, 1],
        model.predict_proba(test)[:, 1],
        {
            "algorithm": "logistic_regression",
            "C": config["logistic_c"],
            "solver": "lbfgs",
            "max_iter": 1000,
            "iterations_used": int(model.named_steps["classifier"].n_iter_[0]),
            "preprocessing": "Fold-local median imputation, standard scaling, one-hot encoding; unknown categories ignored; category min_frequency=5.",
        },
    )


def fit_catboost(
    training: pd.DataFrame,
    y: np.ndarray,
    validation: pd.DataFrame,
    validation_y: np.ndarray,
    test: pd.DataFrame,
    config: dict,
    seed: int,
    path: Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    from catboost import CatBoostClassifier

    _, categorical = feature_types(training)
    device = "GPU" if config["device"] == "cuda" else "CPU"
    parameters = {
        "iterations": config["iterations"],
        "depth": config["depth"],
        "learning_rate": config["learning_rate"],
        "l2_leaf_reg": config["l2_leaf_reg"],
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": seed,
        "thread_count": config["threads"],
        "task_type": device,
        "allow_writing_files": False,
        "verbose": False,
    }
    model = CatBoostClassifier(**parameters)
    if config.get("refit"):
        model.fit(training, y, cat_features=categorical)
    else:
        model.fit(
            training,
            y,
            cat_features=categorical,
            eval_set=(validation, validation_y),
            early_stopping_rounds=config["patience"],
            use_best_model=True,
        )
    model.save_model(str(path.with_suffix(".cbm")))
    details = {
        **parameters,
        "best_iteration": int(model.get_best_iteration()),
        "trees_used": int(model.tree_count_),
        "early_stopping_patience": None if config.get("refit") else config["patience"],
        "preprocessing": "Numeric missing values handled by CatBoost; categorical missing values mapped to __MISSING__; native ordered categorical statistics.",
    }
    return (
        model.predict_proba(validation)[:, 1],
        model.predict_proba(test)[:, 1],
        details,
    )


class NeuralPreprocessor:
    """Training-fold-only numeric statistics and categorical vocabularies."""

    def __init__(
        self,
        numeric_transform: str = "standard",
        seed: int = 42,
        categorical_transform: str = "embedding",
        smoothing: float = 20.0,
    ):
        self.numeric_transform = numeric_transform
        self.seed = seed
        self.categorical_transform = categorical_transform
        self.smoothing = smoothing

    def fit(self, frame: pd.DataFrame) -> "NeuralPreprocessor":
        self.numeric, self.categorical = feature_types(frame)
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler = (
            QuantileTransformer(
                n_quantiles=min(1000, len(frame)),
                output_distribution="normal",
                subsample=min(100000, len(frame)),
                random_state=self.seed,
            )
            if self.numeric_transform == "quantile"
            else StandardScaler()
        )
        if self.numeric:
            self.scaler.fit(self.imputer.fit_transform(frame[self.numeric]))
        self.vocabularies = {
            name: {
                value: index + 1
                for index, value in enumerate(sorted(frame[name].unique()))
            }
            for name in self.categorical
        }
        self.cardinalities = [
            len(self.vocabularies[name]) + 1 for name in self.categorical
        ]
        return self

    def fit_transform(
        self, frame: pd.DataFrame, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        self.fit(frame)
        if self.categorical_transform == "embedding":
            return self.transform(frame)
        numbers = (
            self.scaler.transform(self.imputer.transform(frame[self.numeric])).astype(
                np.float32
            )
            if self.numeric
            else np.empty((len(frame), 0), dtype=np.float32)
        )
        self.cardinalities = []
        self.target_encoder = TargetEncoder(
            target_type="binary",
            smooth=self.smoothing,
            cv=5,
            shuffle=True,
            random_state=self.seed,
        )
        self.target_scaler = StandardScaler()
        if self.categorical:
            encoded = self.target_encoder.fit_transform(frame[self.categorical], y)
            encoded = self.target_scaler.fit_transform(log_odds(encoded)).astype(
                np.float32
            )
            numbers = np.column_stack([numbers, encoded])
        return numbers, np.empty((len(frame), 0), dtype=np.int64)

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        numbers = (
            self.scaler.transform(self.imputer.transform(frame[self.numeric])).astype(
                np.float32
            )
            if self.numeric
            else np.empty((len(frame), 0), dtype=np.float32)
        )
        if getattr(self, "categorical_transform", "embedding") == "target":
            if self.categorical:
                encoded = self.target_encoder.transform(frame[self.categorical])
                encoded = self.target_scaler.transform(log_odds(encoded)).astype(
                    np.float32
                )
                numbers = np.column_stack([numbers, encoded])
            return numbers, np.empty((len(frame), 0), dtype=np.int64)
        categories = (
            np.column_stack(
                [
                    frame[name]
                    .map(self.vocabularies[name])
                    .fillna(0)
                    .to_numpy(dtype=np.int64)
                    for name in self.categorical
                ]
            )
            if self.categorical
            else np.empty((len(frame), 0), dtype=np.int64)
        )
        return numbers, categories


def build_network(
    torch,
    numeric_count: int,
    cardinalities: list[int],
    hidden_dims: list[int],
    dropout: float,
):
    nn = torch.nn
    embedding_dims = [
        min(32, max(2, int(np.sqrt(count)) + 1)) for count in cardinalities
    ]

    class TabularMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = nn.ModuleList(
                [
                    nn.Embedding(count, width, padding_idx=0)
                    for count, width in zip(cardinalities, embedding_dims)
                ]
            )
            width = numeric_count + sum(embedding_dims)
            layers = []
            for next_width in hidden_dims:
                layers.extend(
                    [nn.Linear(width, next_width), nn.ReLU(), nn.Dropout(dropout)]
                )
                width = next_width
            layers.append(nn.Linear(width, 1))
            self.layers = nn.Sequential(*layers)

        def forward(self, numbers, categories):
            pieces = [
                numbers,
                *[
                    embedding(categories[:, i])
                    for i, embedding in enumerate(self.embeddings)
                ],
            ]
            return self.layers(torch.cat(pieces, dim=1)).squeeze(1)

    return TabularMLP(), embedding_dims


def fit_neural(
    training: pd.DataFrame,
    y: np.ndarray,
    validation: pd.DataFrame,
    validation_y: np.ndarray,
    test: pd.DataFrame,
    config: dict,
    seed: int,
    path: Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Neural training requires PyTorch: install fraudlab[neural]."
        ) from error

    torch.set_num_threads(config["threads"])
    torch.manual_seed(seed)
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA was requested but is not available in this PyTorch installation."
        )
    device = torch.device(
        "cuda" if config["device"] != "cpu" and torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    preprocessing = NeuralPreprocessor(
        config.get("neural_numeric_transform", "standard"),
        seed,
        config.get("neural_categorical_transform", "embedding"),
        config.get("target_smoothing", 20.0),
    )
    arrays = [
        preprocessing.fit_transform(training, y),
        preprocessing.transform(validation),
        preprocessing.transform(test),
    ]
    tensors = [
        tuple(torch.as_tensor(value, device=device) for value in pair)
        for pair in arrays
    ]
    labels = torch.as_tensor(y, dtype=torch.float32, device=device)
    model, embedding_dims = build_network(
        torch,
        arrays[0][0].shape[1],
        preprocessing.cardinalities,
        config["hidden_dims"],
        config["dropout"],
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["neural_learning_rate"],
        weight_decay=config["weight_decay"],
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    def predict(pair) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            return np.concatenate(
                [
                    torch.sigmoid(
                        model(
                            pair[0][start : start + config["batch_size"]],
                            pair[1][start : start + config["batch_size"]],
                        )
                    )
                    .cpu()
                    .numpy()
                    for start in range(0, len(pair[0]), config["batch_size"])
                ]
            )

    best_auc, stale, best_epoch, best_state = -np.inf, 0, 0, None
    epochs_completed = 0
    for epoch in range(config["epochs"]):
        model.train()
        order = torch.randperm(len(y), device=device)
        for start in range(0, len(y), config["batch_size"]):
            selected = order[start : start + config["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(
                model(tensors[0][0][selected], tensors[0][1][selected]),
                labels[selected],
            )
            if not torch.isfinite(loss):
                raise ValueError(
                    "Neural loss became non-finite; check features or lower the learning rate."
                )
            loss.backward()
            optimizer.step()
        epochs_completed = epoch + 1
        if config.get("refit"):
            best_epoch = epochs_completed
            if epochs_completed == config["epochs"]:
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
            continue
        score = roc_auc_score(validation_y, predict(tensors[1]))
        if score > best_auc + 1e-6:
            best_auc, best_epoch, stale = score, epoch + 1, 0
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    if best_state is None:
        raise ValueError("Neural training did not produce a usable checkpoint.")
    model.load_state_dict(best_state)
    details = {
        "architecture": {
            "numeric_features": int(arrays[0][0].shape[1]),
            "raw_numeric_features": len(preprocessing.numeric),
            "categorical_transform": config.get(
                "neural_categorical_transform", "embedding"
            ),
            "categorical_cardinalities": preprocessing.cardinalities,
            "embedding_dims": embedding_dims,
            "hidden_dims": config["hidden_dims"],
            "dropout": config["dropout"],
            "hidden_activation": "ReLU",
            "normalization": "none",
            "output_activation": "sigmoid",
        },
        "optimizer": "AdamW",
        "loss": "BCEWithLogitsLoss",
        "learning_rate": config["neural_learning_rate"],
        "weight_decay": config["weight_decay"],
        "batch_size": config["batch_size"],
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "early_stopping_patience": None if config.get("refit") else config["patience"],
        "device": str(device),
        "numeric_transform": config.get("neural_numeric_transform", "standard"),
        "target_encoder": (
            {
                "internal_folds": 5,
                "smooth": config.get("target_smoothing", 20.0),
                "unknown": "training prior",
                "log_odds": True,
            }
            if config.get("neural_categorical_transform") == "target"
            else None
        ),
        "preprocessing": "Fold-local numeric median imputation and "
        + (
            "quantile-to-normal transformation"
            if config.get("neural_numeric_transform") == "quantile"
            else "standard scaling"
        )
        + (
            "; categorical target-encoding logits cross-fitted into five inner partitions, then scaled using training-only statistics."
            if config.get("neural_categorical_transform") == "target"
            else "; categorical training vocabularies with index 0 reserved for unknown categories."
        ),
    }
    torch.save(
        {"state_dict": best_state, "architecture": details["architecture"]},
        path.with_suffix(".pt"),
    )
    joblib.dump(preprocessing, path.with_suffix(".preprocessing.joblib"))
    return predict(tensors[1]), predict(tensors[2]), details


def fit_predict(
    name: str,
    training: pd.DataFrame,
    y: np.ndarray,
    validation: pd.DataFrame,
    validation_y: np.ndarray,
    test: pd.DataFrame,
    config: dict,
    seed: int,
    path: Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    numeric, categorical = feature_types(training)
    training, validation, test = [
        clean_features(frame, numeric, categorical)
        for frame in (training, validation, test)
    ]
    if name == "logistic":
        return fit_logistic(training, y, validation, test, config, seed, path)
    if name == "glm":
        return fit_glm(training, y, validation, test, config, seed, path)
    if name == "catboost":
        return fit_catboost(
            training, y, validation, validation_y, test, config, seed, path
        )
    if name == "mlp":
        return fit_neural(
            training, y, validation, validation_y, test, config, seed, path
        )
    if name == "lightgbm":
        return fit_lightgbm(
            training, y, validation, validation_y, test, config, seed, path
        )
    raise ValueError(f"Unknown model: {name}")


def predict_saved(
    name: str, path: Path, frame: pd.DataFrame, device: str = "auto"
) -> np.ndarray:
    """Load only locally trained checkpoints; never download executable artifacts."""
    if name in ("logistic", "glm", "lightgbm"):
        return joblib.load(path.with_suffix(".joblib")).predict_proba(
            clean_features(frame)
        )[:, 1]
    if name == "catboost":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier()
        model.load_model(str(path.with_suffix(".cbm")))
        return model.predict_proba(clean_features(frame))[:, 1]
    if name != "mlp":
        raise ValueError(f"Unknown saved model: {name}")
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    selected_device = torch.device(
        "cuda" if device != "cpu" and torch.cuda.is_available() else "cpu"
    )
    preprocessing = joblib.load(path.with_suffix(".preprocessing.joblib"))
    checkpoint = torch.load(
        path.with_suffix(".pt"), map_location="cpu", weights_only=True
    )
    architecture = checkpoint["architecture"]
    model, _ = build_network(
        torch,
        architecture["numeric_features"],
        architecture["categorical_cardinalities"],
        architecture["hidden_dims"],
        architecture["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(selected_device).eval()
    prepared = preprocessing.transform(
        clean_features(frame, preprocessing.numeric, preprocessing.categorical)
    )
    predictions = []
    with torch.no_grad():
        for start in range(0, len(frame), 8192):
            tensors = [
                torch.as_tensor(values[start : start + 8192], device=selected_device)
                for values in prepared
            ]
            predictions.append(torch.sigmoid(model(*tensors)).cpu().numpy())
    return np.concatenate(predictions)


def fit_lightgbm(
    training: pd.DataFrame,
    y: np.ndarray,
    validation: pd.DataFrame,
    validation_y: np.ndarray,
    test: pd.DataFrame,
    config: dict,
    seed: int,
    path: Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    from lightgbm import LGBMClassifier, early_stopping

    numeric, categorical = feature_types(training)
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(strategy="median", keep_empty_features=True),
                numeric,
            ),
            (
                "categorical",
                TargetEncoder(
                    target_type="binary",
                    smooth=config.get("target_smoothing", "auto"),
                    cv=5,
                    shuffle=True,
                    random_state=seed,
                ),
                categorical,
            ),
        ]
    )
    # fit_transform cross-fits encodings inside this outer training fold. Validation
    # and test use mappings learned only from that training fold.
    encoded_training = preprocessing.fit_transform(training, y)
    encoded_validation = preprocessing.transform(validation)
    parameters = {
        "n_estimators": config["iterations"],
        "num_leaves": config["num_leaves"],
        "learning_rate": config["learning_rate"],
        "min_child_samples": config["min_child_samples"],
        "reg_lambda": config["l2_leaf_reg"],
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "random_state": seed,
        "n_jobs": config["threads"],
        "deterministic": True,
        "force_col_wise": True,
    }
    classifier = LGBMClassifier(**parameters)
    if config.get("refit"):
        classifier.fit(encoded_training, y)
    else:
        classifier.fit(
            encoded_training,
            y,
            eval_X=encoded_validation,
            eval_y=validation_y,
            eval_metric="auc",
            callbacks=[
                early_stopping(
                    config["patience"], first_metric_only=True, verbose=False
                )
            ],
        )
    model = Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])
    joblib.dump(model, path.with_suffix(".joblib"))
    details = {
        **parameters,
        "best_iteration": int(classifier.best_iteration_ or classifier.n_estimators_),
        "target_encoder": {
            "internal_folds": 5,
            "smooth": config.get("target_smoothing", "auto"),
            "unknown": "training prior",
        },
        "preprocessing": "Fold-local median imputation and supervised categorical target encoding, internally cross-fitted into five partitions.",
    }
    return (
        model.predict_proba(validation)[:, 1],
        model.predict_proba(test)[:, 1],
        details,
    )


def fit_glm(
    training: pd.DataFrame,
    y: np.ndarray,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    config: dict,
    seed: int,
    path: Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    numeric, categorical = feature_types(training)
    nominal = [name for name in categorical if "__" not in name]
    encoding = Pipeline(
        [
            (
                "target",
                TargetEncoder(
                    target_type="binary",
                    smooth=config["target_smoothing"],
                    cv=5,
                    shuffle=True,
                    random_state=seed,
                ),
            ),
            ("log_odds", FunctionTransformer(log_odds)),
            ("scale", StandardScaler()),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="median", keep_empty_features=True),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            ("target_logits", encoding, categorical),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, dtype=np.float32
                ),
                nominal,
            ),
        ]
    )
    classifier = LogisticRegression(
        C=config["logistic_c"],
        solver="newton-cholesky",
        max_iter=100,
        tol=1e-7,
        random_state=seed,
    )
    model = Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])
    with threadpool_limits(limits=config["threads"]):
        model.fit(training, y)
    joblib.dump(model, path.with_suffix(".joblib"))
    details = {
        "algorithm": "logistic_regression_with_cross_fitted_target_logits",
        "C": config["logistic_c"],
        "solver": "newton-cholesky",
        "iterations_used": int(classifier.n_iter_[0]),
        "target_encoder": {
            "internal_folds": 5,
            "smooth": config["target_smoothing"],
            "unknown": "training prior",
            "log_odds": True,
        },
        "preprocessing": "Fold-local numeric imputation and scaling, cross-fitted target-encoding logits with scaling, plus one-hot encoding of original categorical features.",
    }
    return (
        model.predict_proba(validation)[:, 1],
        model.predict_proba(test)[:, 1],
        details,
    )
