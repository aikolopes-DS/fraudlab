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
    inspection = commands.add_parser(
        "inspect", help="Write aggregate data checks and SHA-256 fingerprints."
    )
    inspection.add_argument("--data", default="data/raw")
    inspection.add_argument("--output", default="artifacts/profile.json")
    inspection.add_argument("--id-column", default="id")
    inspection.add_argument("--target", default="is_fraud")

    training = commands.add_parser(
        "train",
        help="Fit cross-validation folds and write OOF, models and test predictions.",
    )
    training.add_argument("--data", default="data/raw")
    training.add_argument("--output", default="artifacts/runs")
    training.add_argument("--name", required=True)
    training.add_argument(
        "--phase",
        choices=("development", "exploratory_full_data_cv"),
        default="development",
        help="Label the validation protocol; exploratory full-data CV has no independent holdout.",
    )
    training.add_argument(
        "--model",
        choices=("logistic", "glm", "catboost", "lightgbm", "mlp", "offset", "tabm"),
        required=True,
    )
    training.add_argument(
        "--feature-set",
        choices=(
            "raw",
            "behavior",
            "digits",
            "value_digits",
            "paired_digits",
            "bpe_digits",
            "token_digits",
        ),
        default="raw",
    )
    training.add_argument("--id-column", default="id")
    training.add_argument("--target", default="is_fraud")
    training.add_argument(
        "--split", choices=("stratified", "group", "time"), default="stratified"
    )
    training.add_argument(
        "--validation-note",
        required=True,
        help="Explain why this split matches the data and intended generalization.",
    )
    training.add_argument("--group-column")
    training.add_argument("--time-column")
    training.add_argument("--folds", type=int, default=3)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.0,
        help="Reserve this fraction before CV; never evaluate it during model selection.",
    )
    training.add_argument("--holdout-seed", type=int, default=2026)
    training.add_argument("--threads", type=int, default=8)
    training.add_argument(
        "--fixed-training",
        action="store_true",
        help="Use the configured tree/epoch count without early stopping; useful for preselected full-data CV.",
    )
    training.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Auto uses CPU for CatBoost and CUDA when available for neural models.",
    )
    training.add_argument("--logistic-c", type=float, default=1.0)
    training.add_argument(
        "--glm-inner-folds",
        type=int,
        choices=(5, 10),
        default=5,
        help="Inner target-encoding partitions for GLM/offset; outer CV is unchanged.",
    )
    training.add_argument(
        "--glm-value-one-hot",
        action="store_true",
        help="One-hot encode every categorical GLM input with fold-local rare pooling and sparse L-BFGS.",
    )
    training.add_argument(
        "--glm-frequency",
        action="store_true",
        help="Add training-fold log category counts to GLM/offset inputs.",
    )
    training.add_argument(
        "--glm-pooled-tokens",
        action="store_true",
        help="Add 16 inner-cross-fitted hierarchical decimal/token logits to GLM/offset.",
    )
    training.add_argument(
        "--glm-spline-knots",
        type=int,
        default=0,
        help="Add cubic quantile splines of continuous features to GLM/offset; 0 disables them.",
    )
    training.add_argument("--iterations", type=int, default=2000)
    training.add_argument("--depth", type=int, default=6)
    training.add_argument("--learning-rate", type=float, default=0.05)
    training.add_argument("--l2-leaf-reg", type=float, default=3.0)
    training.add_argument("--num-leaves", type=int, default=4)
    training.add_argument("--min-child-samples", type=int, default=100)
    training.add_argument("--target-smoothing", type=float, default=20.0)
    training.add_argument("--epochs", type=int, default=30)
    training.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        help="Defaults: MLP 128 64; TabM requires the fixed architecture 256 256.",
    )
    training.add_argument("--dropout", type=float, default=0.1)
    training.add_argument("--neural-learning-rate", type=float, default=0.001)
    training.add_argument("--weight-decay", type=float, default=0.0001)
    training.add_argument(
        "--neural-numeric-transform",
        choices=("standard", "quantile"),
        default="standard",
    )
    training.add_argument("--batch-size", type=int, default=4096)
    training.add_argument(
        "--neural-categorical-transform",
        choices=("embedding", "target"),
        help="Defaults: MLP embedding; TabM requires target.",
    )
    training.add_argument(
        "--patience",
        type=int,
        help="Defaults: CatBoost 100 iterations; MLP 5 epochs; TabM 6 inner epochs.",
    )

    blending = commands.add_parser(
        "blend", help="Average compatible runs with explicit, normalized weights."
    )
    blending.add_argument("--runs", type=Path, nargs="+", required=True)
    blending.add_argument("--weights", type=float, nargs="+")
    blending.add_argument(
        "--space",
        choices=("probability", "logit", "rank", "rank_logit"),
        default="probability",
        help="Combine probabilities, logits or whole-vector normalized ranks.",
    )
    blending.add_argument("--name", required=True)
    blending.add_argument("--output", type=Path, default=Path("artifacts/runs"))
    blending.add_argument("--id-column", default="id")
    blending.add_argument("--target", default="is_fraud")

    validation = commands.add_parser(
        "validate-submission",
        help="Require exact schema, ID order and finite probabilities.",
    )
    validation.add_argument("--file", type=Path, required=True)
    validation.add_argument(
        "--sample", type=Path, default=Path("data/raw/sample_submission.csv")
    )
    validation.add_argument("--id-column", default="id")
    validation.add_argument("--target", default="is_fraud")

    holdout = commands.add_parser(
        "evaluate-holdout",
        help="Evaluate frozen development choices once on the reserved independent holdout.",
    )
    holdout.add_argument("--runs", type=Path, nargs="+", required=True)
    holdout.add_argument("--weights", type=float, nargs="+")
    holdout.add_argument(
        "--output", type=Path, default=Path("artifacts/holdout/evaluation.json")
    )
    holdout.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    holdout.add_argument(
        "--development-run",
        type=Path,
        help="The matching pre-holdout development blend, for an explicit provenance chain.",
    )
    final = commands.add_parser(
        "refit",
        help="Refit one frozen selected model on all labeled data, without early stopping.",
    )
    final.add_argument("--run", type=Path, required=True)
    final.add_argument(
        "--holdout-report", type=Path, default=Path("artifacts/holdout/evaluation.json")
    )
    final.add_argument("--output", type=Path, default=Path("artifacts/runs"))
    final.add_argument("--name", required=True)
    public = commands.add_parser(
        "import-public-gpt2",
        help="Validate the approved licensed public GPT-2 arrays and convert logits to the OOF contract.",
    )
    public.add_argument("--source", type=Path, required=True)
    public.add_argument("--data", type=Path, default=Path("data/raw"))
    public.add_argument("--output", type=Path, default=Path("artifacts/runs"))
    public.add_argument("--name", default="explore5-public-gpt2")
    return root


def validate_training_config(config: dict) -> None:
    if config.get("hidden_dims") is None:
        config["hidden_dims"] = [256, 256] if config["model"] == "tabm" else [128, 64]
    if config.get("neural_categorical_transform") is None:
        config["neural_categorical_transform"] = (
            "target" if config["model"] == "tabm" else "embedding"
        )
    if config["model"] == "tabm":
        if config.get("fixed_training") or config.get("refit"):
            raise ValueError(
                "TabM requires inner epoch selection; --fixed-training/refit is unsupported."
            )
        if config["hidden_dims"] != [256, 256]:
            raise ValueError("TabM requires --hidden-dims 256 256.")
        if config["neural_categorical_transform"] != "target":
            raise ValueError("TabM requires --neural-categorical-transform target.")
    inner_folds = config.get("glm_inner_folds", 5)
    if inner_folds not in (5, 10) or (
        inner_folds != 5 and config["model"] not in ("glm", "offset")
    ):
        raise ValueError("glm_inner_folds must be 5 or 10; 10 requires GLM/offset.")
    if config.get("glm_value_one_hot") and config["model"] != "glm":
        raise ValueError("glm_value_one_hot is supported only for GLM.")
    if config.get("glm_frequency") and config["model"] not in ("glm", "offset"):
        raise ValueError("glm_frequency is supported only for GLM/offset.")
    if config.get("glm_pooled_tokens"):
        if config["model"] not in ("glm", "offset") or inner_folds != 5:
            raise ValueError(
                "glm_pooled_tokens requires GLM/offset with glm_inner_folds=5."
            )
        from .pooled import manifest

        config["pooled_token_encoding"] = manifest(random_state=config["seed"])
    knots = config.get("glm_spline_knots", 0)
    if knots < 0 or knots == 1 or (knots and config["model"] not in ("glm", "offset")):
        raise ValueError(
            "glm_spline_knots must be 0 or at least 2, for GLM/offset only."
        )
    if config.get("phase") == "exploratory_full_data_cv" and (
        config["holdout_fraction"]
        or config["split"] != "stratified"
        or config["folds"] != 5
        or config["seed"] != 42
    ):
        raise ValueError(
            "Exploratory full-data comparison requires stratified 5-fold/seed42 CV with no independent holdout."
        )
    if config["id_column"] == config["target"]:
        raise ValueError("ID and target must be different columns.")
    if not config["validation_note"].strip():
        raise ValueError("A non-empty validation note is required.")
    if not 0 <= config["holdout_fraction"] < 0.5:
        raise ValueError("holdout_fraction must be in [0, 0.5).")
    if config["holdout_fraction"] and config["split"] != "stratified":
        raise ValueError(
            "The independent random holdout is supported only for stratified IID experiments."
        )
    for name in (
        "threads",
        "iterations",
        "epochs",
        "batch_size",
        "patience",
        "depth",
        "num_leaves",
        "min_child_samples",
    ):
        if config[name] < 1:
            raise ValueError(f"{name} must be positive.")
    if config["num_leaves"] < 2:
        raise ValueError("num_leaves must be at least two.")
    import math

    for name in ("learning_rate", "neural_learning_rate", "logistic_c"):
        if not math.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be positive.")
    if not 0 <= config["dropout"] < 1 or any(
        width < 1 for width in config["hidden_dims"]
    ):
        raise ValueError(
            "Dropout must be in [0, 1) and hidden layer widths must be positive."
        )
    if (
        not math.isfinite(config["l2_leaf_reg"])
        or not math.isfinite(config["weight_decay"])
        or config["l2_leaf_reg"] < 0
        or config["weight_decay"] < 0
        or not math.isfinite(config["target_smoothing"])
        or config["target_smoothing"] < 0
    ):
        raise ValueError("Regularization values must be nonnegative.")
    if config["split"] != "group" and config["group_column"]:
        raise ValueError("--group-column requires --split group.")
    if config["split"] != "time" and config["time_column"]:
        raise ValueError("--time-column requires --split time.")
    if (
        config["group_column"] in (config["target"], config["id_column"])
        or config["time_column"] == config["target"]
    ):
        raise ValueError(
            "The target cannot define a split; entity grouping must use an actual entity column."
        )


def main() -> None:
    arguments = parser().parse_args()
    try:
        if arguments.command == "inspect":
            profile = inspect_data(
                Path(arguments.data), arguments.id_column, arguments.target
            )
            write_json(Path(arguments.output), profile)
            print(
                json.dumps(
                    {
                        "profile": arguments.output,
                        "rows": profile["rows"],
                        "target": profile["target"],
                        "duplicate_feature_rows": profile["duplicate_feature_rows"],
                        "potential_split_columns": profile["potential_split_columns"],
                    },
                    indent=2,
                )
            )
        elif arguments.command == "train":
            config = vars(arguments).copy()
            config.pop("command")
            if config["patience"] is None:
                config["patience"] = {"mlp": 5, "tabm": 6}.get(config["model"], 100)
            validate_training_config(config)
            metrics = train(config)
            print(
                json.dumps(
                    {
                        key: metrics[key]
                        for key in (
                            "model",
                            "roc_auc",
                            "fold_scores",
                            "fold_std",
                            "elapsed_seconds",
                        )
                    },
                    indent=2,
                )
            )
        elif arguments.command == "blend":
            metrics = blend(
                arguments.runs,
                arguments.weights,
                arguments.output,
                arguments.name,
                arguments.id_column,
                arguments.target,
                arguments.space,
            )
            print(
                json.dumps(
                    {
                        key: metrics[key]
                        for key in ("model", "roc_auc", "fold_scores", "fold_std")
                    },
                    indent=2,
                )
            )
        elif arguments.command == "evaluate-holdout":
            from .holdout import evaluate

            print(
                json.dumps(
                    evaluate(
                        arguments.runs,
                        arguments.weights,
                        arguments.output,
                        arguments.device,
                        arguments.development_run,
                    ),
                    indent=2,
                )
            )
        elif arguments.command == "refit":
            from .refit import refit

            metrics = refit(
                arguments.run,
                arguments.holdout_report,
                arguments.output,
                arguments.name,
            )
            print(
                json.dumps(
                    {
                        key: metrics[key]
                        for key in ("model", "phase", "rows", "elapsed_seconds")
                    },
                    indent=2,
                )
            )
        elif arguments.command == "import-public-gpt2":
            from .external import import_public_gpt2

            metrics = import_public_gpt2(
                arguments.source, arguments.data, arguments.output, arguments.name
            )
            print(
                json.dumps(
                    {
                        key: metrics[key]
                        for key in ("model", "phase", "roc_auc", "fold_scores")
                    },
                    indent=2,
                )
            )
        else:
            frame, sample = pd.read_csv(arguments.file), pd.read_csv(arguments.sample)
            validate_submission(frame, sample, arguments.id_column, arguments.target)
            print(
                json.dumps(
                    {"valid": True, "rows": len(frame), "file": str(arguments.file)}
                )
            )
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(f"Error: {error}") from error


if __name__ == "__main__":
    main()
