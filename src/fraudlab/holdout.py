"""One final evaluation after development choices have been frozen."""

from __future__ import annotations

import json
import hashlib
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
    runs: list[Path],
    weights: list[float] | None,
    output: Path,
    device: str = "auto",
    development_run: Path | None = None,
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
    for run, config, metric in zip(runs, configs, metrics):
        if (
            metric.get("config_sha256")
            and fingerprint(run / "config.json") != metric["config_sha256"]
        ):
            raise ValueError("Source configuration changed after training.")
        if config["model"] != metric["model"] or config.get(
            "feature_set", "raw"
        ) != metric["model_details"].get("feature_set", "raw"):
            raise ValueError(
                "Source model or feature set no longer matches its training metadata."
            )
        if any(
            config.get(field) != metric["validation"].get(field)
            for field in ("holdout_fraction", "holdout_seed")
        ):
            raise ValueError(
                "Source config no longer matches the holdout recorded at training time."
            )
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
    held_out_ids = set(frame[reference["id_column"]])
    for run in runs:
        if held_out_ids.intersection(
            pd.read_csv(run / "oof.csv", usecols=[reference["id_column"]])[
                reference["id_column"]
            ]
        ):
            raise ValueError("Holdout rows were found in development OOF predictions.")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Persist the chosen configuration before the first holdout prediction is evaluated.
    selection = {
        "sources": [str(run) for run in runs],
        "weights": coefficients.tolist(),
        "metrics_hashes": [fingerprint(run / "metrics.json") for run in runs],
        "config_hashes": [fingerprint(run / "config.json") for run in runs],
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    if development_run is not None:
        dev_config = json.loads(
            (development_run / "config.json").read_text(encoding="utf-8")
        )
        dev_metric = json.loads(
            (development_run / "metrics.json").read_text(encoding="utf-8")
        )
        if (
            [Path(source).resolve() for source in dev_config["sources"]]
            != [source.resolve() for source in runs]
            or not np.allclose(dev_config["weights"], coefficients, rtol=0, atol=1e-12)
            or dev_metric.get("status") != "complete"
            or dev_metric["data_hashes"] != metrics[0]["data_hashes"]
        ):
            raise ValueError(
                "Development blend does not match the frozen sources, weights and input data."
            )
        selection.update(
            source_dev_run=str(development_run),
            development_roc_auc=dev_metric["roc_auc"],
            development_metrics_sha256=fingerprint(development_run / "metrics.json"),
        )
    partition_key = hashlib.sha256(
        json.dumps(
            {
                "train_sha256": metrics[0]["data_hashes"]["train.csv"],
                "target": reference["target"],
                "seed": reference["holdout_seed"],
                "fraction": reference["holdout_fraction"],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    ledger_dir = data_dir / ".holdout-evaluations"
    ledger_dir.mkdir(exist_ok=True)
    ledger = ledger_dir / f"{partition_key}.json"
    selection_key = hashlib.sha256(
        json.dumps(
            {key: value for key, value in selection.items() if key != "frozen_at"},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if ledger.exists():
        prior = json.loads(ledger.read_text(encoding="utf-8"))
        if prior["status"] == "complete" or prior["selection_key"] != selection_key:
            raise ValueError(
                "This independent holdout partition is already reserved or evaluated; do not retune from it."
            )
    else:
        # The dataset-local ledger follows this partition even if --output changes.
        # A failed evaluation may resume only with the identical frozen sources.
        with ledger.open("x", encoding="utf-8") as stream:
            json.dump(
                {
                    "status": "started",
                    "selection_key": selection_key,
                    "selection": selection,
                },
                stream,
                indent=2,
            )
    write_json(output.with_name(output.stem + "-frozen-selection.json"), selection)
    predictions = np.zeros(len(frame))
    individual = []
    for weight, run, config, metric in zip(coefficients, runs, configs, metrics):
        model_predictions = np.zeros(len(frame))
        inputs = transform_features(
            frame,
            config.get("feature_set", "raw"),
            (config["id_column"], config["target"]),
        )[metric["features"]["names"]]
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
        "evaluation_source_hashes": {
            file.name: fingerprint(file)
            for file in sorted(Path(__file__).parent.glob("*.py"))
        },
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
    write_json(
        ledger,
        {
            "status": "complete",
            "selection_key": selection_key,
            "report_sha256": fingerprint(output),
            "evaluated_at": result["evaluated_at"],
        },
    )
    return result
