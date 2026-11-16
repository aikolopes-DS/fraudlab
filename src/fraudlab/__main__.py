"""Run with python -m fraudlab; defaults never download or submit data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import inspect_data, validate_submission, write_json
from .experiment import blend, train


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Auditable Kaggle fraud experiments.")
    commands = root.add_subparsers(dest="command", required=True)
    inspection = commands.add_parser("inspect", help="Write aggregate data checks and SHA-256 fingerprints.")
    inspection.add_argument("--data", default="data/raw")
    inspection.add_argument("--output", default="artifacts/profile.json")
    inspection.add_argument("--id-column", default="id")
    inspection.add_argument("--target", default="is_fraud")

    training = commands.add_parser("train", help="Fit cross-validation folds and write OOF, models and test predictions.")
    training.add_argument("--data", default="data/raw")
    training.add_argument("--output", default="artifacts/runs")
    training.add_argument("--name", required=True)
    training.add_argument("--model", choices=("logistic", "catboost", "mlp"), required=True)
    training.add_argument("--id-column", default="id")
    training.add_argument("--target", default="is_fraud")
    training.add_argument("--split", choices=("stratified", "group", "time"), default="stratified")
    training.add_argument("--validation-note", required=True, help="Explain why this split matches the data and intended generalization.")
    training.add_argument("--group-column")
    training.add_argument("--time-column")
    training.add_argument("--folds", type=int, default=3)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--holdout-fraction", type=float, default=0.0, help="Reserve this fraction before CV; never evaluate it during model selection.")
    training.add_argument("--holdout-seed", type=int, default=2026)
    training.add_argument("--threads", type=int, default=8)
    training.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Auto uses CPU for CatBoost and CUDA when available for the MLP.")
    training.add_argument("--logistic-c", type=float, default=1.0)
    training.add_argument("--iterations", type=int, default=2000)
    training.add_argument("--depth", type=int, default=6)
    training.add_argument("--learning-rate", type=float, default=0.05)
    training.add_argument("--l2-leaf-reg", type=float, default=3.0)
    training.add_argument("--epochs", type=int, default=30)
    training.add_argument("--hidden-dims", type=int, nargs="+", default=[128, 64])
    training.add_argument("--dropout", type=float, default=0.1)
    training.add_argument("--neural-learning-rate", type=float, default=0.001)
    training.add_argument("--weight-decay", type=float, default=0.0001)
    training.add_argument("--batch-size", type=int, default=4096)
    training.add_argument("--patience", type=int, help="Defaults: CatBoost 100 iterations; MLP 5 epochs.")

    blending = commands.add_parser("blend", help="Average compatible runs with explicit, normalized weights.")
    blending.add_argument("--runs", type=Path, nargs="+", required=True)
    blending.add_argument("--weights", type=float, nargs="+")
    blending.add_argument("--name", required=True)
    blending.add_argument("--output", type=Path, default=Path("artifacts/runs"))
    blending.add_argument("--id-column", default="id")
    blending.add_argument("--target", default="is_fraud")

    validation = commands.add_parser("validate-submission", help="Require exact schema, ID order and finite probabilities.")
    validation.add_argument("--file", type=Path, required=True)
    validation.add_argument("--sample", type=Path, default=Path("data/raw/sample_submission.csv"))
    validation.add_argument("--id-column", default="id")
    validation.add_argument("--target", default="is_fraud")
    return root


def validate_training_config(config: dict) -> None:
    if config["id_column"] == config["target"]:
        raise ValueError("ID and target must be different columns.")
    if not config["validation_note"].strip():
        raise ValueError("A non-empty validation note is required.")
    if not 0 <= config["holdout_fraction"] < 0.5:
        raise ValueError("holdout_fraction must be in [0, 0.5).")
    if config["holdout_fraction"] and config["split"] != "stratified":
        raise ValueError("The independent random holdout is supported only for stratified IID experiments.")
    for name in ("threads", "iterations", "epochs", "batch_size", "patience", "depth"):
        if config[name] < 1:
            raise ValueError(f"{name} must be positive.")
    for name in ("learning_rate", "neural_learning_rate", "logistic_c"):
        if config[name] <= 0:
            raise ValueError(f"{name} must be positive.")
    if not 0 <= config["dropout"] < 1 or any(width < 1 for width in config["hidden_dims"]):
        raise ValueError("Dropout must be in [0, 1) and hidden layer widths must be positive.")
    if config["l2_leaf_reg"] < 0 or config["weight_decay"] < 0:
        raise ValueError("Regularization values must be nonnegative.")
    if config["split"] != "group" and config["group_column"]:
        raise ValueError("--group-column requires --split group.")
    if config["split"] != "time" and config["time_column"]:
        raise ValueError("--time-column requires --split time.")
    if config["group_column"] in (config["target"], config["id_column"]) or config["time_column"] == config["target"]:
        raise ValueError("The target cannot define a split; entity grouping must use an actual entity column.")


def main() -> None:
    arguments = parser().parse_args()
    try:
        if arguments.command == "inspect":
            profile = inspect_data(Path(arguments.data), arguments.id_column, arguments.target)
            write_json(Path(arguments.output), profile)
            print(json.dumps({"profile": arguments.output, "rows": profile["rows"], "target": profile["target"], "duplicate_feature_rows": profile["duplicate_feature_rows"], "potential_split_columns": profile["potential_split_columns"]}, indent=2))
        elif arguments.command == "train":
            config = vars(arguments).copy()
            config.pop("command")
            if config["patience"] is None:
                config["patience"] = 5 if config["model"] == "mlp" else 100
            validate_training_config(config)
            metrics = train(config)
            print(json.dumps({key: metrics[key] for key in ("model", "roc_auc", "fold_scores", "fold_std", "elapsed_seconds")}, indent=2))
        elif arguments.command == "blend":
            metrics = blend(arguments.runs, arguments.weights, arguments.output, arguments.name, arguments.id_column, arguments.target)
            print(json.dumps({key: metrics[key] for key in ("model", "roc_auc", "fold_scores", "fold_std")}, indent=2))
        else:
            frame, sample = pd.read_csv(arguments.file), pd.read_csv(arguments.sample)
            validate_submission(frame, sample, arguments.id_column, arguments.target)
            print(json.dumps({"valid": True, "rows": len(frame), "file": str(arguments.file)}))
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
