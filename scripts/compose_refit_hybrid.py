"""Compose the selected recipe's existing full-data refits for inference only.

This specialized operation neither fits models nor measures AUC nor submits to Kaggle.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fraudlab.data import fingerprint, submission, validate_submission, write_json
from fraudlab.experiment import create_run, source_provenance, versions

WEIGHTS = [0.75, 0.25]
EFFECTIVE_WEIGHTS = {"glm": 0.5625, "lightgbm": 0.1875, "external_gpt2": 0.25}
REFITS = ("refit", "glm", "lightgbm")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def source_names(values: list[str]) -> list[str]:
    return [value.replace("\\", "/").rstrip("/").split("/")[-1] for value in values]


def checked_run(root: Path, spec: dict) -> tuple[dict, dict]:
    directory = root / spec["path"]
    if not {"config.json", "metrics.json", "status.json"} <= spec["sha256"].keys():
        raise ValueError(
            "Every input needs pinned configuration, metrics and status hashes."
        )
    for name, expected in spec["sha256"].items():
        if fingerprint(directory / name) != expected:
            raise ValueError(f"Input SHA-256 mismatch: {spec['path']}/{name}")
    metrics, config = read(directory / "metrics.json"), read(directory / "config.json")
    if (
        metrics.get("status") != "complete"
        or read(directory / "status.json").get("status") != "complete"
    ):
        raise ValueError(f"Input run is not complete: {spec['path']}")
    return metrics, config


def check_recipe(metrics: dict, config: dict, names: list[str]) -> None:
    details = metrics["model_details"]
    if (
        metrics["model"] != "blend"
        or details.get("method") != "weighted_probability_average"
        or details.get("weights") != WEIGHTS
        or config.get("weights") != WEIGHTS
        or source_names(details.get("sources", [])) != names
        or source_names(config.get("sources", [])) != names
    ):
        raise ValueError(
            "The input recipe differs from the fixed selected probability weights."
        )


def compose(
    plan_path: Path, root: Path = Path("."), output: Path = Path("artifacts/runs")
) -> dict:
    provenance = source_provenance()
    script_hash = fingerprint(Path(__file__))
    plan_hash, plan = fingerprint(plan_path), read(plan_path)
    if plan["weights"] != WEIGHTS or plan["effective_weights"] != EFFECTIVE_WEIGHTS:
        raise ValueError("This operation permits only the one declared refit recipe.")
    specs = plan["inputs"]
    loaded = {role: checked_run(root, spec) for role, spec in specs.items()}
    metrics = {role: pair[0] for role, pair in loaded.items()}
    configs = {role: pair[1] for role, pair in loaded.items()}
    data_hashes = plan["data_hashes"]
    if set(data_hashes) != {"train.csv", "test.csv", "sample_submission.csv"}:
        raise ValueError("All three input data fingerprints are required.")
    for name, expected in data_hashes.items():
        if fingerprint(root / plan["data"] / name) != expected:
            raise ValueError(f"Data SHA-256 mismatch: {name}")
    for role, metric in metrics.items():
        if metric["data_hashes"] != data_hashes or any(
            metric["rows"][name] != count for name, count in plan["rows"].items()
        ):
            raise ValueError(f"Input data or row counts differ: {role}")
        expected_phase = (
            "full_data_refit" if role in REFITS else "exploratory_full_data_cv"
        )
        if metric.get("phase") != expected_phase:
            raise ValueError(f"Unexpected input phase: {role}")
        if role in REFITS and (
            metric.get("roc_auc") is not None
            or metric.get("fold_scores") != []
            or metric["rows"].get("oof") != 0
            or (root / specs[role]["path"] / "oof.csv").exists()
        ):
            raise ValueError(
                "Full-data refits must not claim OOF predictions or scores."
            )
    if any("model.joblib" not in specs[role]["sha256"] for role in ("glm", "lightgbm")):
        raise ValueError(
            "Both audited refit checkpoints need pinned provenance hashes."
        )
    check_recipe(metrics["refit"], configs["refit"], ["final-glm", "final-lgbm"])
    check_recipe(
        metrics["selection"],
        configs["selection"],
        ["explore5-own", "explore5-public-gpt2-v2"],
    )
    check_recipe(
        metrics["selected_own"],
        configs["selected_own"],
        ["explore5-glm", "explore5-lgbm"],
    )
    if metrics["glm"]["model"] != "glm" or metrics["lightgbm"]["model"] != "lightgbm":
        raise ValueError("The refits must be the selected GLM and LightGBM families.")
    selected_auc = metrics["selection"]["roc_auc"]
    if (
        type(selected_auc) not in (int, float)
        or not math.isfinite(selected_auc)
        or not 0 <= selected_auc <= 1
        or not math.isclose(
            selected_auc, plan["development_roc_auc"], rel_tol=0, abs_tol=1e-12
        )
    ):
        raise ValueError(
            "The historical development reference changed after selection."
        )
    imported = metrics["external"]["model_details"]["external_components"]
    credited = metrics["selection"].get("external_components", [])

    def without_weight(item):
        return {
            key: value
            for key, value in item.items()
            if key not in ("weight", "mixture_space")
        }

    if (
        len(imported) != 1
        or len(credited) != 1
        or without_weight(imported[0]) != without_weight(credited[0])
    ):
        raise ValueError(
            "External release and attribution differ from the selected recipe."
        )
    component = imported[0]
    if (
        any(
            component.get(key) != expected
            for key, expected in plan["external_release"].items()
        )
        or credited[0].get("weight") != 0.25
    ):
        raise ValueError(
            "External author, version, license or selected weight is inconsistent."
        )
    id_column, target = plan["id_column"], plan["target"]
    sample = pd.read_csv(root / plan["data"] / "sample_submission.csv")
    predictions = {}
    for role in (*REFITS, "external"):
        spec, metric = specs[role], metrics[role]
        if (
            not {"predictions.csv", "submission.csv"} <= spec["sha256"].keys()
            or metric.get("submission_sha256") != spec["sha256"]["submission.csv"]
        ):
            raise ValueError("Every prediction input needs its recorded CSV hashes.")
        directory = root / spec["path"]
        frame, submitted = pd.read_csv(directory / "predictions.csv"), pd.read_csv(
            directory / "submission.csv"
        )
        validate_submission(frame, sample, id_column, target)
        validate_submission(submitted, sample, id_column, target)
        if len(frame) != plan["rows"]["test"] or not frame.equals(submitted):
            raise ValueError("Prediction and submission CSVs disagree.")
        predictions[role] = frame[target].to_numpy()
    if not np.allclose(
        predictions["refit"],
        0.75 * predictions["glm"] + 0.25 * predictions["lightgbm"],
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("The own refit CSV does not reproduce its recorded recipe.")
    result = submission(
        sample[id_column],
        0.75 * predictions["refit"] + 0.25 * predictions["external"],
        sample,
        target,
    )
    external = {key: value for key, value in component.items() if key != "oof_auc"}
    external.update(weight=0.25, mixture_space="probability")
    inputs = {
        role: {"path": spec["path"], "sha256": spec["sha256"]}
        for role, spec in specs.items()
    }
    config = {
        "operation": "compose_existing_refit_predictions",
        "plan": str(plan_path),
        "plan_sha256": plan_hash,
        "weights": WEIGHTS,
        "effective_weights": EFFECTIVE_WEIGHTS,
        "inputs": inputs,
    }
    run = create_run(root / output, plan["name"])
    write_json(run / "status.json", {"status": "running"})
    try:
        write_json(run / "config.json", config)
        result.to_csv(run / "predictions.csv", index=False)
        result.to_csv(run / "submission.csv", index=False)
        metadata = {
            "status": "complete",
            "model": "blend",
            "phase": "full_data_refit",
            "roc_auc": None,
            "fold_scores": [],
            "fold_std": None,
            "rows": {**plan["rows"], "oof": 0},
            "features": metrics["refit"]["features"],
            "validation": {
                "method": "inference_only_composition_after_selection",
                "n_splits": 0,
                "independent_holdout": False,
                "early_stopping_uses_validation": False,
            },
            "data_hashes": data_hashes,
            "versions": versions(),
            **provenance,
            "composition_script_sha256": script_hash,
            "plan_sha256": plan_hash,
            "config_sha256": fingerprint(run / "config.json"),
            "inputs": inputs,
            "selection": {
                "source_dev_run": (root / specs["selection"]["path"]).as_posix(),
                "development_metrics_sha256": specs["selection"]["sha256"][
                    "metrics.json"
                ],
                "development_roc_auc": selected_auc,
                "development_roc_auc_scope": "Historical full-data CV of explore5-public25 used to select the recipe; not an AUC of this refit output.",
                "inference_protocol_change": "One existing own model per family fitted on 100%, seed42, replaces the five-model average fitted on80%, seeds42..46. External GPT-2 v2 is unchanged.",
            },
            "refit_provenance": {
                role: {
                    key: metrics[role].get(key)
                    for key in (
                        "git_revision",
                        "git_dirty",
                        "source_hashes",
                        "selection",
                        "versions",
                        "rows",
                        "model_details",
                    )
                }
                for role in REFITS
            },
            "effective_weights": EFFECTIVE_WEIGHTS,
            "external_components": [external],
            "model_details": {
                "method": "weighted_probability_average",
                "sources": [specs[role]["path"] for role in ("refit", "external")],
                "weights": WEIGHTS,
                "effective_weights": EFFECTIVE_WEIGHTS,
                "mixture_space": "probability",
                "models_fitted_for_this_run": False,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "submission_sha256": fingerprint(run / "submission.csv"),
            "limitations": [
                "This inference-only composition has no OOF or independent local AUC. Historical CV explains selection only.",
                "The old holdout remains historical and is not independent for this hybrid.",
                "Changing from fold averages to full-data refits is not evidence of improved performance and does not relax failed experiment gates.",
                "No training or Kaggle submission was performed by this composition script; any submission decision is separate.",
            ],
        }
        write_json(run / "metrics.json", metadata)
        write_json(run / "status.json", {"status": "complete"})
        return metadata
    except Exception as error:
        write_json(run / "status.json", {"status": "failed", "error": str(error)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan", type=Path, default=Path("evidence/refit-hybrid-plan.json")
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("artifacts/runs"))
    arguments = parser.parse_args()
    metrics = compose(arguments.plan, arguments.root, arguments.output)
    print(
        json.dumps(
            {
                key: metrics[key]
                for key in (
                    "phase",
                    "rows",
                    "roc_auc",
                    "effective_weights",
                    "submission_sha256",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
