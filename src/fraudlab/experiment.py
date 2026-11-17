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

from .data import (
    fingerprint,
    load_data,
    submission,
    validate_ids,
    validate_submission,
    write_json,
)
from .features import transform_features
from .models import feature_types, fit_predict
from .splits import make_splits


def versions() -> dict:
    result = {"python": platform.python_version()}
    for package in (
        "fraudlab",
        "numpy",
        "pandas",
        "scikit-learn",
        "catboost",
        "lightgbm",
        "torch",
        "tabm",
        "rtdl_num_embeddings",
        "tiktoken",
        "regex",
    ):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return result


def git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def source_provenance(feature_set: str | None = None) -> dict:
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    result = {
        "source_hashes": {
            file.name: fingerprint(file)
            for file in sorted(Path(__file__).parent.glob("*.py"))
        },
        "git_revision": git_revision(),
        "git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "source_hashes_timing": "captured before reading data or fitting models",
    }
    if feature_set == "bpe_digits":
        from .bpe import manifest

        result["feature_encoding"] = manifest()
    elif feature_set == "token_digits":
        from .tokens import manifest

        result["feature_encoding"] = manifest()
    return result


def create_run(output: Path, name: str) -> Path:
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("Run name must be a single directory name.")
    directory = output / name
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(
            f"Run already exists: {directory}. Choose a new run name."
        ) from error
    return directory


def train(config: dict) -> dict:
    started = time.perf_counter()
    original_features = original_labels = original_metadata = None
    if (
        config.get("original_data")
        or config.get("original_sha256")
        or config.get("original_id_column", "transaction_id") != "transaction_id"
    ):
        from .original import load_original, validate_config, verify_unchanged

        validate_config(config)
    provenance = source_provenance(config.get("feature_set"))
    data_dir = Path(config["data"])
    frame, test, sample = load_data(data_dir, config["id_column"], config["target"])
    target, id_column = config["target"], config["id_column"]
    total_rows = len(frame)
    if config.get("original_data"):
        original_path = Path(config["original_data"])
        original_features, original_labels, original_metadata = load_original(
            original_path,
            config["original_sha256"],
            frame,
            test,
            original_id_column=config.get("original_id_column", "transaction_id"),
            id_column=id_column,
            target=target,
        )
        original_features = transform_features(
            original_features, config.get("feature_set", "raw"), (id_column, target)
        )
    holdout_fraction = config.get("holdout_fraction", 0.0)
    if holdout_fraction:
        development, _ = train_test_split(
            np.arange(len(frame)),
            test_size=holdout_fraction,
            stratify=frame[target],
            random_state=config["holdout_seed"],
        )
        frame = frame.iloc[np.sort(development)].reset_index(drop=True)
    frame = transform_features(
        frame, config.get("feature_set", "raw"), (id_column, target)
    )
    test = transform_features(
        test, config.get("feature_set", "raw"), (id_column, target)
    )
    excluded = [id_column, target]
    if config["split"] == "group":
        excluded.append(config["group_column"])
    if config["split"] == "time":
        excluded.append(config["time_column"])
    features = [name for name in frame if name not in excluded]
    if not features:
        raise ValueError(
            "No model features remain after excluding ID, target and split columns."
        )
    if original_features is not None and original_features.columns.tolist() != features:
        raise ValueError(
            "Engineered original feature columns must match competition order."
        )
    splits = make_splits(
        frame,
        target,
        config["split"],
        config["folds"],
        config["seed"],
        config["group_column"],
        config["time_column"],
    )
    run = create_run(Path(config["output"]), config["name"])
    write_json(run / "config.json", config)
    config_hash = fingerprint(run / "config.json")
    write_json(
        run / "status.json",
        {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()},
    )
    hashes = {
        name: fingerprint(data_dir / name)
        for name in ("train.csv", "test.csv", "sample_submission.csv")
    }
    oof = np.full(len(frame), np.nan)
    fold_ids = np.full(len(frame), -1, dtype=int)
    test_predictions = np.zeros(len(test))
    scores, details = [], []
    try:
        for fold, (training_rows, validation_rows) in enumerate(splits):
            training = frame.iloc[training_rows][features]
            labels = frame.iloc[training_rows][target].to_numpy()
            if original_features is not None:
                verify_unchanged(original_path, config["original_sha256"])
                training = pd.concat([training, original_features], ignore_index=True)
                labels = np.concatenate([labels, original_labels])
            valid_prediction, test_prediction, model_details = fit_predict(
                config["model"],
                training,
                labels,
                frame.iloc[validation_rows][features],
                frame.iloc[validation_rows][target].to_numpy(),
                test[features],
                config,
                config["seed"] + fold,
                run / f"fold-{fold}",
            )
            if original_features is not None:
                verify_unchanged(original_path, config["original_sha256"])
                model_details.update(
                    training_rows=len(training),
                    competition_training_rows=len(training_rows),
                    additional_training_rows=len(original_labels),
                )
            if (
                not np.isfinite(valid_prediction).all()
                or not np.isfinite(test_prediction).all()
            ):
                raise ValueError("Model produced non-finite predictions.")
            oof[validation_rows] = valid_prediction
            fold_ids[validation_rows] = fold
            test_predictions += test_prediction / len(splits)
            score = float(
                roc_auc_score(frame.iloc[validation_rows][target], valid_prediction)
            )
            scores.append(score)
            details.append(model_details)
            print(
                f"fold={fold} roc_auc={score:.8f} train={len(training)} validation={len(validation_rows)}",
                flush=True,
            )
        covered = np.isfinite(oof)
        oof_frame = pd.DataFrame(
            {
                id_column: frame.loc[covered, id_column],
                "fold": fold_ids[covered],
                "target": frame.loc[covered, target],
                "prediction": oof[covered],
            }
        )
        result = submission(test[id_column], test_predictions, sample, target)
        if original_features is not None:
            verify_unchanged(original_path, config["original_sha256"])
        oof_frame.to_csv(run / "oof.csv", index=False)
        result.to_csv(run / "predictions.csv", index=False)
        result.to_csv(run / "submission.csv", index=False)
        numeric, categorical = feature_types(frame[features])
        metrics = {
            "status": "complete",
            "phase": config.get("phase", "development"),
            "config_sha256": config_hash,
            "model": config["model"],
            "roc_auc": float(roc_auc_score(frame.loc[covered, target], oof[covered])),
            "fold_scores": scores,
            "fold_std": float(np.std(scores)),
            "rows": {
                "train": total_rows,
                "test": len(test),
                "oof": int(covered.sum()),
                **(
                    {"development": len(frame), "holdout": total_rows - len(frame)}
                    if holdout_fraction
                    else {}
                ),
            },
            "features": {
                "count": len(features),
                "numeric": len(numeric),
                "categorical": len(categorical),
                "names": features,
                "excluded": list(dict.fromkeys(excluded)),
            },
            "target": {"name": target, "positive_rate": float(frame[target].mean())},
            "seed": config["seed"],
            "validation": {
                "method": config["split"],
                "n_splits": len(splits),
                "note": config["validation_note"],
                "group_column": config["group_column"],
                "time_column": config["time_column"],
                "holdout_fraction": holdout_fraction,
                "holdout_seed": (
                    config.get("holdout_seed") if holdout_fraction else None
                ),
                "independent_holdout": bool(holdout_fraction),
                "early_stopping_uses_validation": config["model"]
                in ("catboost", "lightgbm", "mlp")
                and not config.get("fixed_training"),
                **(
                    {
                        "inner_epoch_selection": {
                            "fraction": 0.15,
                            "seed": 42,
                            "metric": "roc_auc",
                            "fresh_outer_refit": True,
                        }
                    }
                    if config["model"] in ("tabm", "neural_offset")
                    else {}
                ),
            },
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "versions": versions(),
            "data_hashes": hashes,
            **provenance,
            "model_details": {
                **details[0],
                "feature_set": config.get("feature_set", "raw"),
                "folds": details,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "submission_sha256": fingerprint(run / "submission.csv"),
            "limitations": [
                "OOF scores used for model selection are not an independent final assessment.",
                "Public leaderboard and final private leaderboard scores must be obtained from Kaggle.",
                "Temporal OOF, when selected, excludes the initial training-only window.",
                *(
                    [
                        "Exploratory full-data CV uses all labels, including the historical holdout; no independent holdout remains for this stage."
                    ]
                    if config.get("phase") == "exploratory_full_data_cv"
                    else []
                ),
            ],
        }
        if original_metadata is not None:
            metrics["additional_training_data"] = {
                **original_metadata,
                "component_run": str(run),
                "component_model": config["model"],
                "component_scope": "this fitted component only; other blend sources are not retrained",
                "fold_training_rows": [fold["training_rows"] for fold in details],
            }
        write_json(run / "metrics.json", metrics)
        write_json(run / "status.json", {"status": "complete"})
        return metrics
    except Exception as error:
        write_json(run / "status.json", {"status": "failed", "error": str(error)})
        raise


def blend(
    runs: list[Path],
    weights: list[float] | None,
    output: Path,
    name: str,
    id_column: str = "id",
    target: str = "is_fraud",
    space: str = "probability",
) -> dict:
    from .scores import LOGIT_CLIP, SPACES, blend_scores

    started = time.perf_counter()
    provenance = source_provenance()
    if space not in SPACES:
        raise ValueError(f"Unknown score space: {space}")
    if len(runs) < 2:
        raise ValueError("Blending requires at least two runs.")
    coefficients = np.asarray(
        weights if weights is not None else [1.0] * len(runs), dtype=float
    ).copy()
    if (
        coefficients.shape != (len(runs),)
        or not np.isfinite(coefficients).all()
        or (coefficients < 0).any()
        or not np.isfinite(coefficients.sum())
        or coefficients.sum() <= 0
    ):
        raise ValueError(
            "Provide one finite nonnegative weight per run, with a positive total."
        )
    coefficients /= coefficients.sum()
    metrics = [
        json.loads((run / "metrics.json").read_text(encoding="utf-8")) for run in runs
    ]
    phases = [metric.get("phase") == "full_data_refit" for metric in metrics]
    if any(phases):
        if not all(phases):
            raise ValueError("Cannot mix development OOF runs with full-data refits.")
        if space != "probability":
            raise ValueError("Full-data refits support probability blending only.")
        from .refit import blend_refits

        return blend_refits(runs, weights, output, name, id_column, target)
    for coefficient, metric in zip(coefficients, metrics):
        components = metric.get("model_details", {}).get("external_components", [])
        if not components or coefficient <= 0:
            continue
        source_space = metric.get("model_details", {}).get(
            "mixture_space", "probability"
        )
        if (
            source_space != "probability"
            or any(
                component.get("mixture_space", "probability") != "probability"
                for component in components
            )
            or (space != "probability" and metric.get("model") == "blend")
        ):
            raise ValueError(
                "Cannot flatten nested external mixtures across nonlinear score spaces."
            )
        if space != "probability" and (
            len(components) != 1 or components[0].get("weight", 1.0) != 1.0
        ):
            raise ValueError(
                "Nonlinear external blending requires an atomic unit-weight source."
            )
    oofs = [pd.read_csv(run / "oof.csv") for run in runs]
    predictions = [pd.read_csv(run / "predictions.csv") for run in runs]
    reference, expected = oofs[0], predictions[0]
    validate_ids(reference, id_column)
    for index, (score, oof, prediction) in enumerate(zip(metrics, oofs, predictions)):
        if (
            score.get("status") != "complete"
            or score["data_hashes"] != metrics[0]["data_hashes"]
        ):
            raise ValueError(
                "Blend sources must be complete runs with identical data fingerprints."
            )
        if not reference[[id_column, "fold", "target"]].equals(
            oof[[id_column, "fold", "target"]]
        ):
            raise ValueError(
                "Blend sources must use exactly the same OOF rows, order, folds and labels."
            )
        if score["validation"] != metrics[0]["validation"]:
            # Models can differ in early stopping; the actual split contract cannot.
            fields = (
                "method",
                "n_splits",
                "group_column",
                "time_column",
                "holdout_fraction",
            )
            if any(
                score["validation"].get(field) != metrics[0]["validation"].get(field)
                for field in fields
            ):
                raise ValueError("Blend sources must use the same validation strategy.")
            if metrics[0]["validation"].get("holdout_fraction", 0) and score[
                "validation"
            ].get("holdout_seed") != metrics[0]["validation"].get("holdout_seed"):
                raise ValueError("Blend sources must reserve exactly the same holdout.")
        validate_submission(prediction, expected, id_column, target)
        if (
            not np.isfinite(oof["prediction"]).all()
            or not oof["prediction"].between(0, 1).all()
        ):
            raise ValueError(f"Source {index} has invalid OOF probabilities.")
    oof_result, prediction_result = reference.copy(), expected.copy()
    if space == "probability":
        # Keep the original arithmetic and its summation order exactly.
        oof_result["prediction"] = sum(
            weight * frame["prediction"].to_numpy()
            for weight, frame in zip(coefficients, oofs)
        )
        prediction_result[target] = sum(
            weight * frame[target].to_numpy()
            for weight, frame in zip(coefficients, predictions)
        )
    else:
        oof_result["prediction"] = blend_scores(
            [frame["prediction"].to_numpy() for frame in oofs], weights, space=space
        )
        prediction_result[target] = blend_scores(
            [frame[target].to_numpy() for frame in predictions], weights, space=space
        )
    fold_scores = [
        float(roc_auc_score(group["target"], group["prediction"]))
        for _, group in oof_result.groupby("fold")
    ]
    run = create_run(output, name)
    oof_result.to_csv(run / "oof.csv", index=False)
    prediction_result.to_csv(run / "predictions.csv", index=False)
    prediction_result.to_csv(run / "submission.csv", index=False)
    external_components = {}
    for weight, metric, source in zip(coefficients, metrics, runs):
        if weight <= 0:
            continue
        for component in metric.get("model_details", {}).get("external_components", []):
            entry = {**component, "mixture_space": space}
            if space == "probability":
                entry["weight"] = float(weight) * component.get("weight", 1.0)
                key = entry["predictions_source_url"]
            else:
                entry.pop("weight", None)
                entry.update(
                    coefficient=float(weight),
                    coefficient_space=space,
                    coefficient_interpretation="direct source coefficient, not a probability share",
                    source_run=str(source),
                    source_prediction_space="probability",
                )
                key = (entry["predictions_source_url"], str(source))
            if key in external_components:
                field = "weight" if space == "probability" else "coefficient"
                external_components[key][field] += entry[field]
            else:
                external_components[key] = entry
    result = {
        **metrics[0],
        "model": "blend",
        "external_components": list(external_components.values()),
        "roc_auc": float(roc_auc_score(oof_result["target"], oof_result["prediction"])),
        "fold_scores": fold_scores,
        "fold_std": float(np.std(fold_scores)),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "created_at": datetime.now(timezone.utc).isoformat(),
        **provenance,
        "model_details": {
            "method": {
                "probability": "weighted_probability_average",
                "logit": "sigmoid_weighted_logit_average",
                "rank": "weighted_normalized_rank_average",
                "rank_logit": "sigmoid_weighted_rank_logit_average",
            }[space],
            "sources": [str(source) for source in runs],
            "weights": coefficients.tolist(),
            "external_components": list(external_components.values()),
            "mixture_space": space,
        },
        "submission_sha256": fingerprint(run / "submission.csv"),
    }
    blend_config = {
        "sources": [str(source) for source in runs],
        "weights": coefficients.tolist(),
    }
    if space != "probability":
        blend_config["space"] = space
        result["model_details"]["score_transform"] = {
            "input_space": "probability",
            "coefficient_space": space,
            "logit_clip": LOGIT_CLIP if space in ("logit", "rank_logit") else None,
            "rank_method": "average" if space in ("rank", "rank_logit") else None,
            "rank_normalization": (
                "rank/(n+1)" if space in ("rank", "rank_logit") else None
            ),
            "rank_scope": (
                "full OOF vector and full test vector separately; no labels"
                if space in ("rank", "rank_logit")
                else None
            ),
        }
    write_json(
        run / "config.json",
        blend_config,
    )
    result["config_sha256"] = fingerprint(run / "config.json")
    write_json(run / "metrics.json", result)
    write_json(run / "status.json", {"status": "complete"})
    return result
