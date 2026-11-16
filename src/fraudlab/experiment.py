"""Cross-validation, immutable run artifacts and explicit-weight blending."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .data import fingerprint, load_data, submission, validate_ids, validate_submission, write_json
from .models import feature_types, fit_predict
from .splits import make_splits


def versions() -> dict:
    result = {"python": platform.python_version()}
    for package in ("fraudlab", "numpy", "pandas", "scikit-learn", "catboost", "torch"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return result


def git_revision() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def create_run(output: Path, name: str) -> Path:
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Run name must be a single directory name.")
    directory = output / name
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(f"Run already exists: {directory}. Choose a new run name.") from error
    return directory


def train(config: dict) -> dict:
    started = time.perf_counter()
    data_dir = Path(config["data"])
    frame, test, sample = load_data(data_dir, config["id_column"], config["target"])
    target, id_column = config["target"], config["id_column"]
    total_rows = len(frame)
    holdout_fraction = config.get("holdout_fraction", 0.0)
    if holdout_fraction:
        development, _ = train_test_split(np.arange(len(frame)), test_size=holdout_fraction, stratify=frame[target], random_state=config["holdout_seed"])
        frame = frame.iloc[np.sort(development)].reset_index(drop=True)
    excluded = [id_column, target]
    if config["split"] == "group":
        excluded.append(config["group_column"])
    if config["split"] == "time":
        excluded.append(config["time_column"])
    features = [name for name in frame if name not in excluded]
    if not features:
        raise ValueError("No model features remain after excluding ID, target and split columns.")
    splits = make_splits(frame, target, config["split"], config["folds"], config["seed"], config["group_column"], config["time_column"])
    run = create_run(Path(config["output"]), config["name"])
    write_json(run / "config.json", config)
    write_json(run / "status.json", {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()})
    hashes = {name: fingerprint(data_dir / name) for name in ("train.csv", "test.csv", "sample_submission.csv")}
    oof = np.full(len(frame), np.nan)
    fold_ids = np.full(len(frame), -1, dtype=int)
    test_predictions = np.zeros(len(test))
    scores, details = [], []
    try:
        for fold, (training_rows, validation_rows) in enumerate(splits):
            valid_prediction, test_prediction, model_details = fit_predict(config["model"], frame.iloc[training_rows][features], frame.iloc[training_rows][target].to_numpy(), frame.iloc[validation_rows][features], frame.iloc[validation_rows][target].to_numpy(), test[features], config, config["seed"] + fold, run / f"fold-{fold}")
            if not np.isfinite(valid_prediction).all() or not np.isfinite(test_prediction).all():
                raise ValueError("Model produced non-finite predictions.")
            oof[validation_rows] = valid_prediction
            fold_ids[validation_rows] = fold
            test_predictions += test_prediction / len(splits)
            score = float(roc_auc_score(frame.iloc[validation_rows][target], valid_prediction))
            scores.append(score)
            details.append(model_details)
            print(f"fold={fold} roc_auc={score:.8f} train={len(training_rows)} validation={len(validation_rows)}", flush=True)
        covered = np.isfinite(oof)
        oof_frame = pd.DataFrame({id_column: frame.loc[covered, id_column], "fold": fold_ids[covered], "target": frame.loc[covered, target], "prediction": oof[covered]})
        result = submission(test[id_column], test_predictions, sample, target)
        oof_frame.to_csv(run / "oof.csv", index=False)
        result.to_csv(run / "predictions.csv", index=False)
        result.to_csv(run / "submission.csv", index=False)
        numeric, categorical = feature_types(frame[features])
        metrics = {
            "status": "complete", "model": config["model"], "roc_auc": float(roc_auc_score(frame.loc[covered, target], oof[covered])),
            "fold_scores": scores, "fold_std": float(np.std(scores)),
            "rows": {"train": total_rows, "test": len(test), "oof": int(covered.sum()), **({"development": len(frame), "holdout": total_rows - len(frame)} if holdout_fraction else {})},
            "features": {"count": len(features), "numeric": len(numeric), "categorical": len(categorical), "names": features, "excluded": list(dict.fromkeys(excluded))},
            "target": {"name": target, "positive_rate": float(frame[target].mean())},
            "seed": config["seed"], "validation": {"method": config["split"], "n_splits": len(splits), "note": config["validation_note"], "group_column": config["group_column"], "time_column": config["time_column"], "holdout_fraction": holdout_fraction, "holdout_seed": config.get("holdout_seed"), "early_stopping_uses_validation": config["model"] != "logistic"},
            "elapsed_seconds": round(time.perf_counter() - started, 3), "versions": versions(), "data_hashes": hashes,
            "source_hashes": {file.name: fingerprint(file) for file in sorted(Path(__file__).parent.glob("*.py"))},
            "model_details": {**details[0], "folds": details}, "created_at": datetime.now(timezone.utc).isoformat(), "git_revision": git_revision(),
            "submission_sha256": fingerprint(run / "submission.csv"),
            "limitations": ["OOF scores used for model selection are not an independent final assessment.", "Public leaderboard and final private leaderboard scores must be obtained from Kaggle.", "Temporal OOF, when selected, excludes the initial training-only window."],
        }
        write_json(run / "metrics.json", metrics)
        write_json(run / "status.json", {"status": "complete"})
        return metrics
    except Exception as error:
        write_json(run / "status.json", {"status": "failed", "error": str(error)})
        raise


def blend(runs: list[Path], weights: list[float] | None, output: Path, name: str, id_column: str = "id", target: str = "is_fraud") -> dict:
    started = time.perf_counter()
    if len(runs) < 2:
        raise ValueError("Blending requires at least two runs.")
    coefficients = np.asarray(weights if weights is not None else [1.0] * len(runs), dtype=float)
    if len(coefficients) != len(runs) or not np.isfinite(coefficients).all() or (coefficients < 0).any() or not np.isfinite(coefficients.sum()) or coefficients.sum() <= 0:
        raise ValueError("Provide one finite nonnegative weight per run, with a positive total.")
    coefficients /= coefficients.sum()
    metrics = [json.loads((run / "metrics.json").read_text(encoding="utf-8")) for run in runs]
    oofs = [pd.read_csv(run / "oof.csv") for run in runs]
    predictions = [pd.read_csv(run / "predictions.csv") for run in runs]
    reference, expected = oofs[0], predictions[0]
    validate_ids(reference, id_column)
    for index, (score, oof, prediction) in enumerate(zip(metrics, oofs, predictions)):
        if score.get("status") != "complete" or score["data_hashes"] != metrics[0]["data_hashes"]:
            raise ValueError("Blend sources must be complete runs with identical data fingerprints.")
        if not reference[[id_column, "fold", "target"]].equals(oof[[id_column, "fold", "target"]]):
            raise ValueError("Blend sources must use exactly the same OOF rows, order, folds and labels.")
        if score["validation"] != metrics[0]["validation"]:
            # Models can differ in early stopping; the actual split contract cannot.
            fields = ("method", "n_splits", "group_column", "time_column", "holdout_fraction", "holdout_seed")
            if any(score["validation"].get(field) != metrics[0]["validation"].get(field) for field in fields):
                raise ValueError("Blend sources must use the same validation strategy.")
        validate_submission(prediction, expected, id_column, target)
        if not np.isfinite(oof["prediction"]).all() or not oof["prediction"].between(0, 1).all():
            raise ValueError(f"Source {index} has invalid OOF probabilities.")
    oof_result, prediction_result = reference.copy(), expected.copy()
    oof_result["prediction"] = sum(weight * frame["prediction"].to_numpy() for weight, frame in zip(coefficients, oofs))
    prediction_result[target] = sum(weight * frame[target].to_numpy() for weight, frame in zip(coefficients, predictions))
    fold_scores = [float(roc_auc_score(group["target"], group["prediction"])) for _, group in oof_result.groupby("fold")]
    run = create_run(output, name)
    oof_result.to_csv(run / "oof.csv", index=False)
    prediction_result.to_csv(run / "predictions.csv", index=False)
    prediction_result.to_csv(run / "submission.csv", index=False)
    result = {**metrics[0], "model": "blend", "roc_auc": float(roc_auc_score(oof_result["target"], oof_result["prediction"])), "fold_scores": fold_scores, "fold_std": float(np.std(fold_scores)), "elapsed_seconds": round(time.perf_counter() - started, 3), "created_at": datetime.now(timezone.utc).isoformat(), "git_revision": git_revision(), "model_details": {"method": "weighted_probability_average", "sources": [str(source) for source in runs], "weights": coefficients.tolist()}, "submission_sha256": fingerprint(run / "submission.csv")}
    write_json(run / "config.json", {"sources": [str(source) for source in runs], "weights": coefficients.tolist()})
    write_json(run / "metrics.json", result)
    write_json(run / "status.json", {"status": "complete"})
    return result
