"""One final evaluation after development choices have been frozen."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .data import fingerprint, load_data, write_json
from .features import transform_features
from .models import predict_saved


def evaluate(
    runs: list[Path], weights: list[float] | None, output: Path, device: str = "auto"
) -> dict:
    if output.exists():
        raise FileExistsError(
            "Holdout evaluation already exists; do not reuse it for model selection."
        )
    if not runs:
        raise ValueError("At least one frozen source run is required.")
    coefficients = np.asarray(
        weights if weights is not None else [1.0] * len(runs), dtype=float
    )
    if (
        len(coefficients) != len(runs)
        or not np.isfinite(coefficients).all()
        or not np.isfinite(coefficients.sum())
        or (coefficients < 0).any()
        or coefficients.sum() <= 0
    ):
        raise ValueError(
            "Provide finite, nonnegative weights with a finite positive total."
        )
    coefficients /= coefficients.sum()
    configs = [
        json.loads((run / "config.json").read_text(encoding="utf-8")) for run in runs
    ]
    metrics = [
        json.loads((run / "metrics.json").read_text(encoding="utf-8")) for run in runs
    ]
    reference = configs[0]
    if not reference.get("holdout_fraction"):
        raise ValueError("Sources must have reserved a holdout before training.")
    fields = ("data", "target", "id_column", "holdout_fraction", "holdout_seed")
    for config, metric in zip(configs, metrics):
        if (
            any(config.get(field) != reference.get(field) for field in fields)
            or metric.get("status") != "complete"
            or metric["data_hashes"] != metrics[0]["data_hashes"]
        ):
            raise ValueError(
                "Sources must be complete runs sharing the same untouched holdout and input data."
            )
    data_dir = Path(reference["data"])
    for filename, expected in metrics[0]["data_hashes"].items():
        if fingerprint(data_dir / filename) != expected:
            raise ValueError("Input files changed after training.")
    frame, _, _ = load_data(data_dir, reference["id_column"], reference["target"])
    _, held_out = train_test_split(
        np.arange(len(frame)),
        test_size=reference["holdout_fraction"],
        stratify=frame[reference["target"]],
        random_state=reference["holdout_seed"],
    )
    frame = frame.iloc[np.sort(held_out)].reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Persist the chosen configuration before the first holdout prediction is evaluated.
    selection = {
        "sources": [str(run) for run in runs],
        "weights": coefficients.tolist(),
        "metrics_hashes": [fingerprint(run / "metrics.json") for run in runs],
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output.with_name(output.stem + "-frozen-selection.json"), selection)
    predictions = np.zeros(len(frame))
    individual = []
    for weight, run, config, metric in zip(coefficients, runs, configs, metrics):
        model_predictions = np.zeros(len(frame))
        inputs = transform_features(frame, config.get("feature_set", "raw"))[
            metric["features"]["names"]
        ]
        for fold in range(metric["validation"]["n_splits"]):
            model_predictions += (
                predict_saved(config["model"], run / f"fold-{fold}", inputs, device)
                / metric["validation"]["n_splits"]
            )
        predictions += weight * model_predictions
        individual.append(
            float(roc_auc_score(frame[reference["target"]], model_predictions))
        )
    if (
        not np.isfinite(predictions).all()
        or ((predictions < 0) | (predictions > 1)).any()
    ):
        raise ValueError("Invalid holdout predictions.")
    result = {
        "status": "complete",
        "kind": "independent_holdout",
        "roc_auc": float(roc_auc_score(frame[reference["target"]], predictions)),
        "individual_roc_auc": individual,
        "rows": len(frame),
        "positive_rows": int(frame[reference["target"]].sum()),
        "holdout_seed": reference["holdout_seed"],
        "holdout_fraction": reference["holdout_fraction"],
        "selection": selection,
        "data_hashes": metrics[0]["data_hashes"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "limitation": "The selection was frozen before this evaluation. Using this result to retune would make the holdout part of model selection.",
    }
    pd.DataFrame(
        {
            reference["id_column"]: frame[reference["id_column"]],
            "target": frame[reference["target"]],
            "prediction": predictions,
        }
    ).to_csv(output.with_suffix(".csv"), index=False)
    write_json(output, result)
    return result
