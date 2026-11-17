"""Refit a frozen choice once on all labeled rows, without another selection loop."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .data import fingerprint, load_data, submission, validate_submission, write_json
from .experiment import create_run, source_provenance, versions
from .features import transform_features
from .models import feature_types, fit_predict


def refit(source: Path, holdout_report: Path, output: Path, name: str) -> dict:
    started = time.perf_counter()
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if config.get("original_data"):
        raise ValueError("Original-data augmentation has no full-data refit contract.")
    if config.get("glm_transductive_frequency"):
        raise ValueError("Transductive frequency has no full-data refit contract.")
    if config["model"] == "neural_offset":
        raise ValueError(
            "Neural offset uses the CV checkpoint ensemble; a full-data refit protocol is not defined."
        )
    if config["model"] == "tabm":
        raise ValueError(
            "TabM uses the CV checkpoint ensemble; a full-data refit protocol is not defined."
        )
    provenance = source_provenance(config.get("feature_set"))
    selected = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    holdout = json.loads(holdout_report.read_text(encoding="utf-8"))
    if (
        selected.get("status") != "complete"
        or holdout.get("kind") != "independent_holdout"
        or holdout.get("status") != "complete"
    ):
        raise ValueError(
            "A completed development run and frozen independent holdout evaluation are required."
        )
    sources = [Path(path).resolve() for path in holdout["selection"]["sources"]]
    if source.resolve() not in sources:
        raise ValueError("This model was not part of the frozen holdout selection.")
    index = sources.index(source.resolve())
    if holdout["selection"]["weights"][index] <= 0:
        raise ValueError("A model with zero frozen weight does not need a final refit.")
    if (
        fingerprint(source / "metrics.json")
        != holdout["selection"]["metrics_hashes"][index]
        or fingerprint(source / "config.json")
        != holdout["selection"]["config_hashes"][index]
    ):
        raise ValueError(
            "Development metrics or configuration changed after the selection was frozen."
        )
    data_dir = Path(config["data"])
    if any(
        fingerprint(data_dir / filename) != expected
        for filename, expected in selected["data_hashes"].items()
    ):
        raise ValueError("Input data changed after development training.")
    folds = selected["model_details"]["folds"]
    if config["model"] == "mlp":
        config["epochs"] = int(
            round(float(np.median([fold["best_epoch"] for fold in folds])))
        )
    elif config["model"] in ("catboost", "lightgbm"):
        config["iterations"] = int(
            round(
                float(
                    np.median(
                        [
                            (
                                fold["trees_used"]
                                if config["model"] == "catboost"
                                else fold["best_iteration"]
                            )
                            for fold in folds
                        ]
                    )
                )
            )
        )
    config.update(
        refit=True,
        holdout_fraction=0.0,
        name=name,
        output=str(output),
        selection_source=str(source),
        holdout_report=str(holdout_report),
    )
    frame, test, sample = load_data(data_dir, config["id_column"], config["target"])
    target, id_column = config["target"], config["id_column"]
    frame = transform_features(
        frame, config.get("feature_set", "raw"), (id_column, target)
    )
    test = transform_features(
        test, config.get("feature_set", "raw"), (id_column, target)
    )
    features = selected["features"]["names"]
    run = create_run(output, name)
    write_json(run / "config.json", config)
    write_json(run / "status.json", {"status": "running"})
    try:
        _, predictions, details = fit_predict(
            config["model"],
            frame[features],
            frame[target].to_numpy(),
            test[features],
            None,
            test[features],
            config,
            config["seed"],
            run / "model",
        )
        result = submission(test[id_column], predictions, sample, target)
        result.to_csv(run / "predictions.csv", index=False)
        result.to_csv(run / "submission.csv", index=False)
        numeric, categorical = feature_types(frame[features])
        metrics = {
            "status": "complete",
            "model": config["model"],
            "phase": "full_data_refit",
            "roc_auc": None,
            "fold_scores": [],
            "fold_std": None,
            "rows": {"train": len(frame), "test": len(test), "oof": 0},
            "features": {
                "count": len(features),
                "numeric": len(numeric),
                "categorical": len(categorical),
                "names": features,
                "excluded": selected["features"]["excluded"],
            },
            "target": {"name": target, "positive_rate": float(frame[target].mean())},
            "seed": config["seed"],
            "validation": {
                "method": "no_new_selection_full_data_refit",
                "n_splits": 0,
                "early_stopping_uses_validation": False,
            },
            "selection": {
                "source_dev_run": str(source),
                "development_roc_auc": selected["roc_auc"],
                "holdout_report_sha256": fingerprint(holdout_report),
                "frozen_iterations_rule": "median best epoch/tree count across development folds; model hyperparameters unchanged",
            },
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "versions": versions(),
            "data_hashes": selected["data_hashes"],
            **provenance,
            "model_details": {
                **details,
                "feature_set": config.get("feature_set", "raw"),
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "submission_sha256": fingerprint(run / "submission.csv"),
            "limitations": [
                "All labels were used to refit a previously frozen choice. No new independent score was measured for this final model.",
                "Wall-clock training times from concurrent runs are not controlled performance benchmarks.",
            ],
        }
        write_json(run / "metrics.json", metrics)
        write_json(run / "status.json", {"status": "complete"})
        return metrics
    except Exception as error:
        write_json(run / "status.json", {"status": "failed", "error": str(error)})
        raise


def blend_refits(
    runs: list[Path],
    weights: list[float] | None,
    output: Path,
    name: str,
    id_column: str,
    target: str,
) -> dict:
    """Combine final predictions with exactly the pre-holdout frozen weights."""
    started = time.perf_counter()
    provenance = source_provenance()
    configs = [
        json.loads((run / "config.json").read_text(encoding="utf-8")) for run in runs
    ]
    metrics = [
        json.loads((run / "metrics.json").read_text(encoding="utf-8")) for run in runs
    ]
    report_path = Path(configs[0]["holdout_report"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_hash = fingerprint(report_path)
    frozen = {
        Path(source).resolve(): weight
        for source, weight in zip(
            report["selection"]["sources"], report["selection"]["weights"]
        )
        if weight > 0
    }
    source_paths = [Path(config["selection_source"]).resolve() for config in configs]
    if set(source_paths) != set(frozen) or len(set(source_paths)) != len(source_paths):
        raise ValueError(
            "Final blend must contain every nonzero frozen source exactly once."
        )
    coefficients = np.array([frozen[source] for source in source_paths], dtype=float)
    if weights is not None:
        requested = np.array(weights, dtype=float)
        if len(requested) != len(coefficients) or not np.allclose(
            requested / requested.sum(), coefficients, rtol=0, atol=1e-12
        ):
            raise ValueError(
                "Final blend weights cannot change after independent holdout evaluation."
            )
    for config, metric in zip(configs, metrics):
        if (
            metric.get("status") != "complete"
            or metric.get("phase") != "full_data_refit"
            or metric["data_hashes"] != metrics[0]["data_hashes"]
            or metric["selection"]["holdout_report_sha256"] != report_hash
            or fingerprint(Path(config["holdout_report"])) != report_hash
        ):
            raise ValueError(
                "Final sources must be completed refits of the same frozen selection and input data."
            )
    predictions = [pd.read_csv(run / "predictions.csv") for run in runs]
    result = predictions[0].copy()
    for frame in predictions:
        validate_submission(frame, result, id_column, target)
    result[target] = sum(
        weight * frame[target].to_numpy()
        for weight, frame in zip(coefficients, predictions)
    )
    validate_submission(result, predictions[0], id_column, target)
    run = create_run(output, name)
    result.to_csv(run / "predictions.csv", index=False)
    result.to_csv(run / "submission.csv", index=False)
    details = {
        "method": "weighted_probability_average",
        "sources": [str(source) for source in runs],
        "weights": coefficients.tolist(),
    }
    metadata = {
        **metrics[0],
        **provenance,
        "model": "blend",
        "phase": "full_data_refit",
        "roc_auc": None,
        "fold_scores": [],
        "fold_std": None,
        "model_details": details,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "submission_sha256": fingerprint(run / "submission.csv"),
        "selection": {
            "source_dev_run": report["selection"].get("source_dev_run"),
            "development_roc_auc": report["selection"].get("development_roc_auc"),
            "development_metrics_sha256": report["selection"].get(
                "development_metrics_sha256"
            ),
            "holdout_report_sha256": report_hash,
            "weights_frozen_before_holdout": True,
        },
    }
    write_json(run / "config.json", details)
    write_json(run / "metrics.json", metadata)
    write_json(run / "status.json", {"status": "complete"})
    return metadata
