"""Recover completed neural folds and fit only the missing folds in a pinned core.

This entrypoint does not implement or change a model. It imports the reviewed
core from --source-worktree only after checking its revision and clean state.
The public recovery plan must be reviewed and published before execution.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ENTRYPOINT = Path(__file__).resolve()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def verify_source(worktree, revision):
    def git(*arguments):
        return subprocess.check_output(
            ["git", "-C", str(worktree), *arguments], text=True
        ).strip()

    require(
        git("rev-parse", "HEAD") == revision,
        "Core revision differs from the recovery plan.",
    )
    require(not git("status", "--porcelain"), "Core worktree must remain clean.")
    return {
        path.name: digest(path)
        for path in sorted((worktree / "src/fraudlab").glob("*.py"))
    }


def resumed_config(original, name, plan):
    require(original["name"] == plan["original_run"], "Original run name differs.")
    require(
        name == plan["continuation_run"], "Continuation name differs from the plan."
    )
    require(
        Path(name).name == name and name not in (".", ".."),
        "Run name must be a directory name.",
    )
    require(
        original["model"] == "neural_offset"
        and original["feature_set"] == "bpe_digits",
        "Unsupported original model/representation.",
    )
    require(
        original["folds"] == 5
        and original["seed"] == 42
        and original["split"] == "stratified",
        "Recovery requires the original CV5/42.",
    )
    require(
        not original.get("original_data") and not original["holdout_fraction"],
        "Recovery must not add data or a holdout.",
    )
    require(
        original["device"] == "cpu" and original["threads"] == 4,
        "Original execution must remain CPU/four threads.",
    )
    require(
        plan["recovery"]["reuse"] == [0, 1, 2] and plan["recovery"]["fit"] == [3, 4],
        "Recovery fold partition differs.",
    )
    result = dict(original)
    result["name"] = name
    return result


def copy_checkpoint(source, destination, expected):
    require(digest(source) == expected, "Original checkpoint changed before copying.")
    require(not destination.exists(), "Destination checkpoint already exists.")
    shutil.copy2(source, destination)
    require(digest(destination) == expected, "Copied checkpoint SHA-256 differs.")


def logged_folds(text):
    rows = re.findall(
        r"^fold=(\d+) roc_auc=([0-9.]+) train=(\d+) validation=(\d+)\s*$",
        text,
        re.MULTILINE,
    )
    require(
        [int(row[0]) for row in rows] == [0, 1, 2],
        "Log must document exactly the three completed folds.",
    )
    return {
        int(fold): {"auc": float(auc), "train": int(train), "validation": int(valid)}
        for fold, auc, train, valid in rows
    }


def collect_predictions(labels, test_rows, splits, recover, fit, completed):
    """Keep the original float32 division before float64 accumulation, in order."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    oof = np.full(len(labels), np.nan)
    fold_ids = np.full(len(labels), -1, dtype=int)
    test_average = np.zeros(test_rows)
    scores, details = [], []
    for fold, (training, validation) in enumerate(splits):
        callback = recover if fold in (0, 1, 2) else fit
        valid_prediction, test_prediction, metadata = callback(
            fold, training, validation
        )
        for prediction, size in (
            (valid_prediction, len(validation)),
            (test_prediction, test_rows),
        ):
            require(
                prediction.shape == (size,) and prediction.dtype == np.float32,
                "Frozen neural predictor must return one float32 probability per row.",
            )
            require(
                np.isfinite(prediction).all()
                and ((prediction >= 0) & (prediction <= 1)).all(),
                "Invalid recovered prediction.",
            )
        require((fold_ids[validation] == -1).all(), "Validation folds overlap.")
        oof[validation] = valid_prediction
        fold_ids[validation] = fold
        test_average += test_prediction / len(splits)
        score = float(roc_auc_score(labels[validation], valid_prediction))
        scores.append(score)
        details.append(metadata)
        completed(fold, score, metadata)
    require(
        np.isfinite(oof).all() and (fold_ids >= 0).all(),
        "Incomplete competition OOF coverage.",
    )
    return oof, fold_ids, test_average, scores, details


def reconstruct_details(base, config, seed, training_rows, names, log_record):
    import joblib
    import numpy as np
    import torch
    from fraudlab.neural_offset import NeuralOffsetConfig
    from fraudlab.pooled import manifest as pooled_manifest

    payload = torch.load(base.with_suffix(".pt"), map_location="cpu", weights_only=True)
    saved = joblib.load(base.with_suffix(".glm.joblib"))
    require(
        set(payload) == {"state_dict", "config", "architecture", "epoch_metadata"},
        "Unexpected saved payload schema.",
    )
    require(
        payload["config"] == asdict(NeuralOffsetConfig.from_training_config(config)),
        "Saved neural settings differ.",
    )
    epoch = payload["epoch_metadata"]
    require(
        1 <= epoch["selected_epoch"] <= config["epochs"]
        and epoch["refit_epochs_completed"] == epoch["selected_epoch"],
        "Saved epoch selection/refit differs.",
    )
    require(
        epoch["refit_rows"] == len(training_rows)
        and epoch["inner_training_rows"] + epoch["inner_validation_rows"]
        == len(training_rows),
        "Saved training counts differ.",
    )
    require(
        epoch["inner_validation_rows"] == int(np.ceil(len(training_rows) * 0.15)),
        "Saved internal split differs.",
    )
    require(epoch["inner_glm_converged"] is True, "Saved inner GLM did not converge.")
    glm = saved["glm"]
    classifier = glm.named_steps["classifier"]
    preprocessing = glm.named_steps["preprocessing"]
    require(
        list(preprocessing.feature_names_in_) == names, "Saved feature schema differs."
    )
    require(
        classifier.C == config["logistic_c"] and classifier.solver == "newton-cholesky",
        "Saved GLM settings differ.",
    )
    require(
        int(classifier.n_iter_[0]) < classifier.max_iter,
        "Saved outer GLM reached the cap.",
    )
    architecture = payload["architecture"]
    require(
        architecture["input_features"] == classifier.n_features_in_
        and architecture["hidden_dims"] == [256, 256]
        and architecture["activation"] == "SiLU",
        "Saved neural architecture differs.",
    )
    require(
        architecture["dropout"] == config["dropout"]
        and architecture["zero_initialized_head"] is True,
        "Saved neural head/dropout differs.",
    )
    pooled = preprocessing.named_transformers_["pooled_token_logits"].named_steps[
        "pooled"
    ]
    require(
        pooled.random_state == seed and pooled.n_splits == 5,
        "Saved pooled seed/folds differ.",
    )
    require(
        log_record["train"] == len(training_rows),
        "Original logged training count differs.",
    )
    details = {
        "architecture": architecture,
        "config": payload["config"],
        **epoch,
        "glm": {
            key: config[key]
            for key in (
                "feature_set",
                "logistic_c",
                "target_smoothing",
                "glm_inner_folds",
                "glm_frequency",
                "glm_pooled_tokens",
                "glm_spline_knots",
            )
        },
        "glm_iterations_used": int(classifier.n_iter_[0]),
        "glm_converged": True,
        "optimizer": "AdamW",
        "loss": "unweighted mean BCEWithLogits",
        "prediction": "sigmoid(frozen GLM decision_function + neural residual)",
        "training_encoding": "same cross-fitted matrix for GLM and residual; no full-map training retransform",
        "epoch_selection": "inner 15% AUC; fresh GLM, preprocessing, network and AdamW for full outer refit",
        "outer_validation_usage": "inference only; no labels accepted",
        "seed": seed,
        "threads": config["threads"],
        "device": config["device"],
        "pooled_tokens": pooled_manifest(random_state=seed),
        "recovery_metadata": {
            "origin": "reused complete checkpoint; no retraining",
            "payload_facts": ["architecture", "config", "epoch_metadata"],
            "outer_glm_facts": "Saved fitted parameters and iterations; convergence also implied by pinned abort-on-warning code, not a recovered warning log",
            "execution_fields": "Seed matches the saved pooled encoder; device/threads and optimizer/protocol derive from original config and pinned source, not separate runtime telemetry",
            "not_persisted": [
                "original per-fold elapsed time",
                "original package-version snapshot",
                "warning messages/list",
                "original OOF/test arrays",
                "inner models and optimizer states",
            ],
        },
    }
    return details


def execute(args):
    started = time.perf_counter()
    original_run, worktree = args.source_run.resolve(), args.source_worktree.resolve()
    plan_path, output = args.plan.resolve(), args.output.resolve()
    requested_evidence = (
        args.recovery_evidence.resolve() if args.recovery_evidence else None
    )
    plan = read_json(plan_path)
    require(
        plan.get("entrypoint_sha256") == digest(ENTRYPOINT),
        "Recovery plan must pin the reviewed entrypoint SHA-256 before execution.",
    )
    require(
        args.revision == plan["source"]["revision"], "Requested core revision differs."
    )
    source_hashes = verify_source(worktree, args.revision)
    original = read_json(original_run / "config.json")
    config = resumed_config(original, args.name, plan)
    require(
        output == Path(original["output"]).resolve(),
        "Output must preserve the original configuration.",
    )
    require(
        original_run.name == original["name"], "Source directory/config name mismatch."
    )
    status = read_json(original_run / "status.json")
    require(
        status["status"] == "failed"
        and "Unable to allocate" in status.get("error", ""),
        "Source run is not the documented memory failure.",
    )
    invocation = read_json(args.source_invocation)
    require(
        invocation["source_commit"] == args.revision
        and Path(invocation["working_directory"]).resolve() == worktree,
        "Original invocation source differs.",
    )
    original_plan = (plan_path.parent.parent / plan["original_plan"]).resolve()
    require(
        digest(original_plan) == invocation["plan_sha256"],
        "Original R17 plan bytes differ.",
    )
    for key, value in read_json(original_plan)["candidate"].items():
        require(
            original[key] == value,
            "Original config differs from the R17 candidate: " + key,
        )
    log_text = args.source_log.read_text(encoding="utf-8-sig")
    logged = logged_folds(log_text)
    require("_ArrayMemoryError" in log_text, "Original OOM traceback is missing.")
    receipts = {
        "original-config.json": original_run / "config.json",
        "original-status.json": original_run / "status.json",
        "original-invocation.json": args.source_invocation.resolve(),
        "original-training.log": args.source_log.resolve(),
        "original-plan.json": original_plan,
        "recovery-plan.json": plan_path,
    }
    frozen = {str(path): digest(path) for path in receipts.values()}
    for fold in (0, 1, 2):
        for suffix in (".pt", ".glm.joblib"):
            path = original_run / f"fold-{fold}{suffix}"
            frozen[str(path)] = digest(path)
    require(
        not list(original_run.glob("fold-3.*"))
        and not list(original_run.glob("fold-4.*")),
        "Unexpected partial later-fold artifacts; review them before recovery.",
    )
    sys.path.insert(0, str(worktree / "src"))
    require(
        "fraudlab" not in sys.modules, "Core was imported before pinning its source."
    )
    core = importlib.import_module("fraudlab")
    require(
        Path(core.__file__).resolve().is_relative_to(worktree / "src"),
        "Wrong core import.",
    )
    os.chdir(worktree)
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import roc_auc_score
    from fraudlab.__main__ import parser as core_parser, validate_training_config
    from fraudlab.data import load_data, submission
    from fraudlab.experiment import create_run, source_provenance, versions
    from fraudlab.features import transform_features
    from fraudlab.models import feature_types, fit_predict, predict_saved
    from fraudlab.splits import make_splits

    for name in ("fraudlab.models", "fraudlab.features", "fraudlab.neural_offset"):
        module = importlib.import_module(name)
        require(
            Path(module.__file__).resolve().is_relative_to(worktree / "src"),
            "Core submodule imported outside pinned source: " + name,
        )
    parsed = vars(core_parser().parse_args(invocation["command"][3:]))
    require(parsed.pop("command") == "train", "Original invocation is not training.")
    if parsed["patience"] is None:
        parsed["patience"] = 6
    validate_training_config(parsed)
    require(
        parsed == original, "Original config differs from frozen invocation/defaults."
    )
    provenance = source_provenance(config["feature_set"])
    require(
        provenance["source_hashes"] == source_hashes
        and provenance["git_revision"] == args.revision
        and provenance["git_dirty"] is False,
        "Core provenance mismatch.",
    )
    data_dir = Path(config["data"])
    baseline = read_json(
        output / read_json(original_plan)["baseline_run"] / "metrics.json"
    )
    hashes = {
        name: digest(data_dir / name)
        for name in ("train.csv", "test.csv", "sample_submission.csv")
    }
    require(
        hashes == baseline["data_hashes"],
        "Official data differs from the frozen baseline.",
    )
    frozen.update({str(data_dir / name): value for name, value in hashes.items()})
    frozen[str(ENTRYPOINT)] = digest(ENTRYPOINT)
    evidence = (
        requested_evidence
        if requested_evidence
        else output / args.name / "recovery-evidence"
    )
    require(
        not evidence.is_relative_to(original_run),
        "Recovery evidence must not modify the failed run.",
    )
    require(not evidence.exists(), "Recovery evidence directory already exists.")
    run = create_run(output, args.name)
    evidence.mkdir(parents=True, exist_ok=False)
    write_json(run / "config.json", config)
    write_json(
        run / "status.json",
        {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()},
    )
    recovery = {
        "original_run": str(original_run),
        "source_revision": args.revision,
        "entrypoint_sha256": digest(ENTRYPOINT),
        "plan_sha256": digest(plan_path),
        "original_input_sha256": dict(frozen),
        "reuse_folds": [0, 1, 2],
        "fit_folds": [3, 4],
        "original_elapsed_seconds": None,
        "original_versions": None,
        "old_run_preserved": True,
        "predictions_reconstructed_from_weights": True,
        "comparison_to_original_prediction_arrays": "unavailable; they were not persisted",
        "data_binding": "Current official CSV hashes match frozen baseline; original failed process did not persist a standalone pre-fit data-hash receipt",
    }
    for name, source in receipts.items():
        copy_checkpoint(source, evidence / name, frozen[str(source)])
    for fold in (0, 1, 2):
        for suffix in (".pt", ".glm.joblib"):
            source = original_run / f"fold-{fold}{suffix}"
            copy_checkpoint(source, run / source.name, frozen[str(source)])
    write_json(run / "recovery.json", recovery)
    torch.set_num_threads(config["threads"])
    try:
        frame, test, sample = load_data(data_dir, config["id_column"], config["target"])
        frame = transform_features(
            frame, config["feature_set"], (config["id_column"], config["target"])
        )
        test = transform_features(
            test, config["feature_set"], (config["id_column"], config["target"])
        )
        features = [
            name
            for name in frame
            if name not in (config["id_column"], config["target"])
        ]
        labels = frame[config["target"]].to_numpy()
        splits = make_splits(
            frame,
            config["target"],
            config["split"],
            config["folds"],
            config["seed"],
            config["group_column"],
            config["time_column"],
        )

        def recover(fold, training, validation):
            base = run / f"fold-{fold}"
            details = reconstruct_details(
                base, config, config["seed"] + fold, training, features, logged[fold]
            )
            require(
                logged[fold]["validation"] == len(validation),
                "Original logged validation count differs.",
            )
            valid = predict_saved(
                config["model"], base, frame.iloc[validation][features], "cpu"
            )
            predicted = predict_saved(config["model"], base, test[features], "cpu")
            return valid, predicted, details

        def fit(fold, training, validation):
            return fit_predict(
                config["model"],
                frame.iloc[training][features],
                labels[training],
                frame.iloc[validation][features],
                labels[validation],
                test[features],
                config,
                config["seed"] + fold,
                run / f"fold-{fold}",
            )

        def completed(fold, score, details):
            if fold in logged:
                require(
                    abs(score - logged[fold]["auc"]) <= 5.1e-9,
                    "Reconstructed AUC differs from original eight-decimal log.",
                )
            write_json(
                run / f"fold-{fold}-recovery.json",
                {
                    "fold": fold,
                    "roc_auc": score,
                    "origin": "recovered" if fold < 3 else "new continuation fit",
                    "model_details": details,
                },
            )
            print(
                json.dumps({"fold": fold, "roc_auc": score, "reused": fold < 3}),
                flush=True,
            )

        oof, fold_ids, test_average, scores, details = collect_predictions(
            labels, len(test), splits, recover, fit, completed
        )
        pd.DataFrame(
            {
                config["id_column"]: frame[config["id_column"]],
                "fold": fold_ids,
                "target": labels,
                "prediction": oof,
            }
        ).to_csv(run / "oof.csv", index=False)
        result = submission(
            test[config["id_column"]], test_average, sample, config["target"]
        )
        result.to_csv(run / "predictions.csv", index=False)
        result.to_csv(run / "submission.csv", index=False)
        for path, expected in frozen.items():
            require(digest(path) == expected, "Recovery input changed: " + path)
        require(
            verify_source(worktree, args.revision) == source_hashes,
            "Core source changed during recovery.",
        )
        for fold in (0, 1, 2):
            for suffix in (".pt", ".glm.joblib"):
                source = original_run / f"fold-{fold}{suffix}"
                require(
                    digest(run / source.name) == frozen[str(source)],
                    "Reused checkpoint changed.",
                )
        numeric, categorical = feature_types(frame[features])
        recovery["resume_elapsed_seconds"] = round(time.perf_counter() - started, 3)
        recovery["versions_at_resume"] = versions()
        recovery["checkpoint_sha256"] = {
            path.name: digest(path)
            for path in sorted(run.glob("fold-*"))
            if path.suffix in (".pt", ".joblib")
        }
        metrics = {
            "status": "complete",
            "phase": config["phase"],
            "config_sha256": digest(run / "config.json"),
            "model": config["model"],
            "roc_auc": float(roc_auc_score(labels, oof)),
            "fold_scores": scores,
            "fold_std": float(np.std(scores)),
            "rows": {"train": len(frame), "test": len(test), "oof": len(frame)},
            "features": {
                "count": len(features),
                "numeric": len(numeric),
                "categorical": len(categorical),
                "names": features,
                "excluded": [config["id_column"], config["target"]],
            },
            "target": {"name": config["target"], "positive_rate": float(labels.mean())},
            "seed": config["seed"],
            "validation": {
                "method": config["split"],
                "n_splits": 5,
                "note": config["validation_note"],
                "group_column": None,
                "time_column": None,
                "holdout_fraction": 0.0,
                "holdout_seed": None,
                "independent_holdout": False,
                "early_stopping_uses_validation": False,
                "inner_epoch_selection": {
                    "fraction": 0.15,
                    "seed": 42,
                    "metric": "roc_auc",
                    "fresh_outer_refit": True,
                },
            },
            "elapsed_seconds": None,
            "elapsed_seconds_note": "Original total was not persisted; recovery.resume_elapsed_seconds covers this continuation only.",
            "versions": versions(),
            "versions_scope": "Continuation environment; original package snapshot was not persisted",
            "data_hashes": hashes,
            **provenance,
            "model_details": {
                **details[0],
                "feature_set": config["feature_set"],
                "folds": details,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "submission_sha256": digest(run / "submission.csv"),
            "recovery": recovery,
            "limitations": [
                "Exploratory CV reuses all labels; no independent holdout.",
                "Three folds reconstructed from complete saved checkpoints, two newly fitted under the original protocol.",
                "Prior timing/version telemetry and original OOF/test arrays were not persisted; no comparison with nonexistent arrays is claimed.",
            ],
        }
        write_json(run / "metrics.json", metrics)
        write_json(run / "recovery.json", recovery)
        write_json(run / "status.json", {"status": "complete"})
        return metrics
    except Exception as error:
        write_json(run / "status.json", {"status": "failed", "error": str(error)})
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "source-run",
        "source-worktree",
        "output",
        "plan",
        "source-invocation",
        "source-log",
    ):
        result.add_argument("--" + flag, type=Path, required=True)
    result.add_argument("--revision", required=True)
    result.add_argument("--name", required=True)
    result.add_argument("--recovery-evidence", type=Path)
    return result


if __name__ == "__main__":
    metrics = execute(parser().parse_args())
    print(
        json.dumps(
            {
                key: metrics[key]
                for key in ("roc_auc", "fold_scores", "submission_sha256")
            }
        ),
        flush=True,
    )
