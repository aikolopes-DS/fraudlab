"""Refit one audited RealMLP CV recipe; reuse the audited R20 full models.

An explicit all-label contract, separate from the independent-holdout refit.
No new AUC, OOF, epoch search, tree fit or submission is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys
import time

ENTRYPOINT = Path(__file__).resolve()
WEIGHTS = [0.421875, 0.140625, 0.1875, 0.25]
ROLES = ("offset", "realmlp", "lightgbm", "external_gpt2")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    Path(path).write_bytes(
        (
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        ).encode()
    )


def chosen_epoch(plan, own_metric, selected_metric, audit22):
    require(
        own_metric["status"] == selected_metric["status"] == "complete"
        and audit22["status"] == "passed",
        "Incomplete CV or independent audit.",
    )
    require(
        own_metric["phase"] == selected_metric["phase"] == "exploratory_full_data_cv",
        "Selection must be exploratory CV.",
    )
    epochs = [item["selected_epoch"] for item in own_metric["model_details"]["folds"]]
    require(epochs == plan["cv_epochs"] == [3, 2, 3, 3, 3], "CV epochs changed.")
    selected = audit22.get("selected_within_round") or {}
    require(
        selected.get("run") == Path(plan["inputs"]["selection"]["path"]).name
        and selected.get("eligible") is True
        and selected.get("weights") == WEIGHTS
        and selected.get("delta_vs_r18", 0) > 0
        and selected.get("delta_vs_r10", 0) > 0.0001
        and selected.get("improved_folds_vs_r18", 0) >= 4
        and selected.get("roc_auc")
        == selected_metric["roc_auc"]
        == plan["reference_cv_auc"],
        "Audited CV selection/gate/weights changed.",
    )
    require(
        selected_metric["model_details"]["weights"] == WEIGHTS,
        "Selected mixture differs.",
    )
    epoch = statistics.median(epochs)
    require(epoch == 3, "Expected median epoch3.")
    return int(epoch)


def fit_full_neural(api, config, frame, y, test, epoch, path):
    """Use the frozen official implementation once, without any validation set."""
    import cloudpickle
    import numpy as np
    import torch

    settings = api.RealMLPConfig.from_training_config(config)
    require(
        settings.epochs == epoch == 3,
        "The fixed horizon and median epoch must both be3.",
    )
    require(config["device"] in ("cpu", "cuda"), "Choose an explicit device.")
    require(
        config["device"] != "cuda" or torch.cuda.is_available(),
        "CUDA unavailable; no fallback permitted.",
    )
    torch.set_num_threads(config["threads"])
    started = time.perf_counter()
    prep = api.RealMLPPreprocessor(42)
    numbers = prep.fit_transform(frame, y)
    model, detail = api._train(
        numbers,
        y,
        None,
        settings,
        10042,
        "cuda:0" if config["device"] == "cuda" else "cpu",
        config["threads"],
        stop_epoch=epoch,
    )
    del numbers
    require(
        detail["completed_epochs"] == epoch and model.model.progress.max_epochs == 3,
        "Full fit did not complete the fixed horizon.",
    )
    prediction = api._predict(model, prep.transform(test)).astype(np.float64)
    details = {
        "architecture": {
            "library": "pytabkit==1.8.0",
            "name": "RealMLP_TD",
            "n_ens": 16,
            "hidden_dims": [512, 256, 128],
            "activation": "parametric SiLU",
            "numeric_embedding": "PBLD: periodic16, linear7 + original-value skip =8",
            "categorical_embedding_dim": 5,
            "parameters": detail["parameters"],
        },
        "preprocessing": prep.metadata(),
        "refit": detail,
        "refit_rows": len(frame),
        "epochs": epoch,
        "preprocessing_seed": 42,
        "network_seed": 10042,
        "validation_rows": 0,
        "epoch_selection": "No selection in this fit; fixed median of the five audited CV-selected epochs.",
        "elapsed_seconds": time.perf_counter() - started,
    }
    api._move(model, "cpu")
    with path.with_suffix(".pkl").open("wb") as stream:
        cloudpickle.dump(
            {"model": model, "preprocessing": prep, "details": details}, stream
        )
    count = min(1024, len(test))
    reloaded = api.predict_saved_realmlp(path, test.iloc[:count], "cpu")
    error = float(np.max(np.abs(reloaded - prediction[:count])))
    require(
        np.isfinite(prediction).all()
        and ((prediction >= 0) & (prediction <= 1)).all()
        and error <= 1e-6,
        "Full neural probability/reload check failed.",
    )
    return (
        prediction,
        details,
        {
            "rows": count,
            "max_abs_error": error,
            "atol": 1e-6,
            "rtol": 0,
            "checkpoint_sha256": digest(path.with_suffix(".pkl")),
        },
    )


def execute(plan_path, root, worktree):
    started = time.perf_counter()
    root, worktree, plan_path = root.resolve(), worktree.resolve(), plan_path.resolve()
    plan, frozen, loaded = read(plan_path), {}, {}
    require(
        plan["weights"] == WEIGHTS and digest(ENTRYPOINT) == plan["entrypoint_sha256"],
        "Entrypoint/weights changed.",
    )
    for role, spec in plan["inputs"].items():
        directory = (root / spec["path"]).resolve()
        require(directory.is_relative_to(root), "Input escaped workspace.")
        for name, expected in spec["sha256"].items():
            require(Path(name).name == name, "Pinned input must be a basename.")
            path = directory / name
            require(digest(path) == expected, "Input changed: " + str(path))
            frozen[str(path)] = expected
        loaded[role] = read(directory / "metrics.json"), read(directory / "config.json")
        require(
            loaded[role][0]["status"]
            == read(directory / "status.json")["status"]
            == "complete"
            and loaded[role][0]["data_hashes"] == plan["data_hashes"],
            "Incomplete or different-data input.",
        )
    receipts = {}
    for role, spec in plan["proofs"].items():
        path = (root / spec["path"]).resolve()
        require(
            path.is_relative_to(root) and digest(path) == spec["sha256"],
            "Proof changed: " + role,
        )
        frozen[str(path)] = spec["sha256"]
        receipts[role] = read(path)
    epoch = chosen_epoch(
        plan, loaded["neural_cv"][0], loaded["selection"][0], receipts["r22_audit"]
    )
    require(
        receipts["r22_audit"]["core_revision"] == plan["source"]["revision"]
        and receipts["r22_audit"]["plan_sha256"]
        == plan["proofs"]["r22_plan"]["sha256"],
        "CV audit source/plan differs.",
    )
    audited = {
        key.replace("\\", "/"): value
        for key, value in receipts["r22_audit"]["input_sha256"].items()
    }
    for role in ("neural_cv", "selection"):
        spec = plan["inputs"][role]
        for name in ("config.json", "metrics.json"):
            require(
                audited.get(spec["path"] + "/" + name) == spec["sha256"][name],
                "CV source differs from independent audit.",
            )
    require(
        receipts["r20_audit"]["status"] == "passed", "R20 full-data audit did not pass."
    )
    for role in ("offset", "lightgbm"):
        metric, _ = loaded[role]
        require(
            metric["phase"] == "full_data_refit"
            and metric["roc_auc"] is None
            and metric["rows"] == {**plan["rows"], "oof": 0},
            "Reuse requires complete all-label R20 models.",
        )
        proof = next(
            item
            for item in receipts["r20_audit"]["checkpoints"]
            if item["role"] == role
        )
        require(
            proof["model_sha256"] == plan["inputs"][role]["sha256"]["model.joblib"]
            and proof["parameters_unchanged"],
            "Reused model was not audited.",
        )
    helper = ENTRYPOINT.with_name("refit_selected_cv.py")
    require(
        digest(helper) == plan["support_script_sha256"],
        "Source pinning helper changed.",
    )
    spec = importlib.util.spec_from_file_location("_selected_refit_contracts", helper)
    contracts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contracts)
    for role in ("offset", "lightgbm"):
        require(
            contracts.parameters_sha256(loaded[role][1])
            == loaded[role][0]["source_parameters_sha256"],
            "Reused R20 parameters changed.",
        )
    contracts.pin_core(worktree, plan["source"])
    import numpy as np
    import pandas as pd
    from fraudlab import realmlp
    from fraudlab.data import load_data, submission, validate_submission
    from fraudlab.experiment import create_run, source_provenance, versions
    from fraudlab.features import transform_features
    from fraudlab.models import predict_saved

    require(
        all(
            importlib.metadata.version(name) == version
            for name, version in plan["dependency_versions"].items()
        ),
        "Pinned dependency versions changed.",
    )

    frozen.update(
        {
            str(plan_path): digest(plan_path),
            str(ENTRYPOINT): digest(ENTRYPOINT),
            str(helper): digest(helper),
        }
    )
    data, output = root / plan["data"], root / plan["output"]
    require(
        data.resolve().is_relative_to(root) and output.resolve().is_relative_to(root),
        "Data/output escaped workspace.",
    )
    for name, expected in plan["data_hashes"].items():
        require(digest(data / name) == expected, "Raw data changed.")
        frozen[str(data / name)] = expected
    raw, test_raw, sample = load_data(data, "id", "is_fraud")
    require(
        len(raw) == plan["rows"]["train"] and len(test_raw) == plan["rows"]["test"],
        "Wrong data sizes.",
    )
    config = loaded["neural_cv"][1]
    require(
        loaded["neural_cv"][0]["config_sha256"]
        == plan["inputs"]["neural_cv"]["sha256"]["config.json"],
        "Neural source config differs from its receipt.",
    )
    require(
        config["seed"] == 42 and config["device"] == "cuda" and config["threads"] == 2,
        "Frozen neural execution settings differ.",
    )
    require(
        not (output / plan["neural_name"]).exists()
        and not (output / plan["name"]).exists(),
        "Output exists; no retry/overwrite permitted.",
    )
    nn_run = create_run(output, plan["neural_name"])
    hybrid_run = create_run(output, plan["name"])
    selection = {
        "source_dev_run": str(root / plan["inputs"]["selection"]["path"]),
        "reference_cv_run": Path(plan["inputs"]["selection"]["path"]).name,
        "development_metrics_sha256": plan["inputs"]["selection"]["sha256"][
            "metrics.json"
        ],
        "development_roc_auc": plan["reference_cv_auc"],
        "development_roc_auc_scope": "Historical exploratory CV of the selected R22 hybrid; no score of these full-data predictions.",
        "independent_holdout": False,
    }
    common = {
        **source_provenance(),
        "phase": "full_data_refit",
        "roc_auc": None,
        "fold_scores": [],
        "fold_std": None,
        "rows": {**plan["rows"], "oof": 0},
        "data_hashes": plan["data_hashes"],
        "selection": selection,
        "versions": versions(),
        "plan_sha256": digest(plan_path),
        "refit_script_sha256": digest(ENTRYPOINT),
        "validation": {
            "method": "fixed_median_epoch_refit_of_audited_exploratory_cv",
            "n_splits": 0,
            "independent_holdout": False,
            "early_stopping_uses_validation": False,
        },
    }
    for run in (nn_run, hybrid_run):
        write(run / "status.json", {"status": "running"})
    try:
        nn_config = {
            **config,
            "name": plan["neural_name"],
            "phase": "full_data_refit",
            "refit": True,
            "epochs": epoch,
            "network_seed": 10042,
            "preprocessing_seed": 42,
            "selection": selection,
            "validation_note": "Fixed median E3 from audited R22 CV; fresh all-label refit, no validation and no new local score.",
        }
        write(nn_run / "config.json", nn_config)
        values, details, reload = fit_full_neural(
            realmlp,
            config,
            raw.drop(columns=["id", "is_fraud"]),
            raw.is_fraud.to_numpy(),
            test_raw.drop(columns="id"),
            epoch,
            nn_run / "model",
        )
        nn_frame = submission(test_raw.id, values, sample, "is_fraud")
        for filename in ("predictions.csv", "submission.csv"):
            nn_frame.to_csv(nn_run / filename, index=False)
        nn_metric = {
            **common,
            "status": "complete",
            "model": "realmlp",
            "features": loaded["neural_cv"][0]["features"],
            "model_details": details,
            "checkpoint_reload": reload,
            "config_sha256": digest(nn_run / "config.json"),
            "submission_sha256": digest(nn_run / "submission.csv"),
        }
        write(nn_run / "metrics.json", nn_metric)
        predictions, reuse = {
            "realmlp": pd.read_csv(nn_run / "predictions.csv").is_fraud.to_numpy()
        }, {}
        for role in ("offset", "lightgbm", "external_gpt2"):
            path = root / plan["inputs"][role]["path"]
            frame = pd.read_csv(path / "predictions.csv")
            validate_submission(frame, sample, "id", "is_fraud")
            predictions[role] = frame.is_fraud.to_numpy()
            if role != "external_gpt2":
                metric, cfg = loaded[role]
                transformed = transform_features(
                    test_raw.iloc[:1024], cfg["feature_set"], ("id", "is_fraud")
                )
                result = predict_saved(
                    role,
                    path / "model",
                    transformed[metric["features"]["names"]],
                    "cpu",
                )
                error = float(np.max(np.abs(result - predictions[role][: len(result)])))
                require(error <= 1e-12, "Reused R20 checkpoint reload differs.")
                reuse[role] = {
                    "path": str(path),
                    "checkpoint_sha256": digest(path / "model.joblib"),
                    "reload_rows": len(result),
                    "max_abs_error": error,
                    "new_fit": False,
                }
        formula = sum(
            weight * predictions[role] for role, weight in zip(ROLES, WEIGHTS)
        )
        final = submission(test_raw.id, formula, sample, "is_fraud")
        for filename in ("predictions.csv", "submission.csv"):
            final.to_csv(hybrid_run / filename, index=False)
        written = pd.read_csv(hybrid_run / "submission.csv")
        validate_submission(written, sample, "id", "is_fraud")
        error = float(np.max(np.abs(written.is_fraud.to_numpy() - formula)))
        require(error <= 1e-15, "Full-test serialized formula differs.")
        for path, expected in frozen.items():
            require(digest(path) == expected, "Input changed during fit: " + path)
        provenance = source_provenance()
        require(
            provenance["git_revision"] == plan["source"]["revision"]
            and provenance["git_dirty"] is False
            and provenance["source_hashes"] == plan["source"]["sha256"],
            "Frozen core changed.",
        )
        cfg = {
            "operation": "full_data_neural_refit_reuse_audited_trees",
            "effective_weights": dict(zip(ROLES, WEIGHTS)),
            "plan_sha256": digest(plan_path),
            "inputs": plan["inputs"],
            "neural_run": str(nn_run),
        }
        write(hybrid_run / "config.json", cfg)
        source_credit = (
            loaded["selection"][0].get("external_components")
            or loaded["selection"][0]["model_details"]["external_components"]
        )
        credited = [
            {key: value for key, value in item.items() if key != "oof_auc"}
            for item in source_credit
        ]
        metric = {
            **common,
            "status": "complete",
            "model": "blend",
            "features": loaded["selection"][0]["features"],
            "effective_weights": dict(zip(ROLES, WEIGHTS)),
            "external_components": credited,
            "model_details": {
                "method": "weighted_probability_average",
                "sources": [
                    (
                        str(nn_run)
                        if role == "realmlp"
                        else str(root / plan["inputs"][role]["path"])
                    )
                    for role in ROLES
                ],
                "weights": WEIGHTS,
            },
            "config_sha256": digest(hybrid_run / "config.json"),
            "submission_sha256": digest(hybrid_run / "submission.csv"),
            "reused_models": reuse,
            "neural_checkpoint_reload": reload,
            "formula_rows_verified": len(final),
            "formula_max_absolute_error": error,
            "elapsed_seconds": time.perf_counter() - started,
            "new_fits": 1,
            "new_submissions": 0,
        }
        write(hybrid_run / "metrics.json", metric)
        for run in (nn_run, hybrid_run):
            write(run / "status.json", {"status": "complete"})
        return metric
    except Exception as error:
        for run in (nn_run, hybrid_run):
            if (run / "metrics.json").exists():
                failed = read(run / "metrics.json")
                failed.update(status="failed", error=str(error))
                write(run / "metrics.json", failed)
            write(run / "status.json", {"status": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--core-worktree", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(execute(args.plan, args.workspace, args.core_worktree), indent=2))
