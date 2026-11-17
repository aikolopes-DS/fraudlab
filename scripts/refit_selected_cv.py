"""Refit one audited exploratory-CV recipe on all labels; no new AUC or submission.

This explicit contract is separate from the independent-holdout refit command.
Only the pinned core imports after verification; fit parameters come unchanged
from the audited CV configurations, and the external predictions remain fixed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ENTRYPOINT = Path(__file__).resolve()
ROLES = ("offset", "lightgbm", "external")
WEIGHTS = [0.5625, 0.1875, 0.25]
EXECUTION = {"data", "output", "name", "phase", "refit", "validation_note"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def parameters_sha256(config):
    values = {key: value for key, value in config.items() if key not in EXECUTION}
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def checked_inputs(root, plan):
    require(plan["weights"] == WEIGHTS, "Wrong probability weights.")
    require(set(plan["inputs"]) == {"selection", *ROLES}, "Wrong source roles.")
    loaded, frozen = {}, {}
    for role, spec in plan["inputs"].items():
        directory = (root / spec["path"]).resolve()
        require(directory.is_relative_to(root), "Source outside the workspace.")
        needed = {"config.json", "metrics.json", "status.json"}
        if role in ROLES[:2]:
            needed |= {f"fold-{i}.joblib" for i in range(5)}
        require(needed <= spec["sha256"].keys(), "Missing source pins.")
        for name, expected in spec["sha256"].items():
            require(Path(name).name == name, "Pinned file must be a basename.")
            path = directory / name
            require(digest(path) == expected, "Input changed: " + str(path))
            frozen[str(path)] = expected
        metric, cfg = read(directory / "metrics.json"), read(directory / "config.json")
        require(
            metric["status"] == read(directory / "status.json")["status"] == "complete",
            "Incomplete source.",
        )
        require(metric["phase"] == "exploratory_full_data_cv", "Not exploratory CV.")
        require(
            metric["rows"] == {**plan["rows"], "oof": plan["rows"]["train"]},
            "Wrong source rows.",
        )
        require(metric["data_hashes"] == plan["data_hashes"], "Wrong source data.")
        if role in ROLES[:2]:
            require(
                cfg["model"] == role
                and cfg["phase"] == "exploratory_full_data_cv"
                and cfg["folds"] == 5
                and cfg["split"] == "stratified"
                and cfg["holdout_fraction"] == 0
                and cfg["seed"] == 42
                and cfg["fixed_training"],
                "Wrong refit protocol.",
            )
            require(
                not cfg.get("original_data")
                and not cfg.get("glm_transductive_frequency"),
                "Extra/transductive data excluded.",
            )
            require(
                metric["config_sha256"] == spec["sha256"]["config.json"],
                "Config receipt differs.",
            )
            require(
                parameters_sha256(cfg) == spec["parameters_sha256"],
                "Parameters changed.",
            )
            require(
                all(
                    cfg[key] == value
                    for key, value in spec.get("parameters_summary", {}).items()
                ),
                "Fixed recipe summary differs.",
            )
        loaded[role] = metric, cfg
    receipts = {}
    for kind in ("audit", "public_proof"):
        spec, path = plan[kind], root / plan[kind]["path"]
        require(path.resolve().is_relative_to(root), "Receipt outside the workspace.")
        require(digest(path) == spec["sha256"], "Receipt changed: " + kind)
        receipts[kind], frozen[str(path)] = read(path), spec["sha256"]
    metric = loaded["selection"][0]
    name = Path(plan["inputs"]["selection"]["path"]).name
    audit, proof = receipts["audit"], receipts["public_proof"]
    require(
        audit["status"] == "passed" and audit["source_clean_at_start_and_finish"],
        "Audit did not pass.",
    )
    require(
        audit["source_revision"] == plan["source"]["revision"]
        and audit["competition_data_hashes"] == plan["data_hashes"],
        "Audit source/data differ.",
    )
    require(
        audit["selected_within_round"]["run"] == name
        and audit["selected_within_round"]["eligible_to_submit"] is True
        and audit["selected_within_round"]["oof_auc"]
        == metric["roc_auc"]
        == plan["reference_cv_auc"],
        "Audited selection differs.",
    )
    audited = {
        k.replace("\\", "/").removeprefix("./"): v
        for k, v in audit["inputs_sha256"].items()
    }
    for role, spec in plan["inputs"].items():
        for filename in ("config.json", "metrics.json"):
            require(
                audited.get(spec["path"] + "/" + filename) == spec["sha256"][filename],
                "Unaudited source: " + role,
            )
    require(
        proof["run"] == name
        and proof["submission_file_verified"]
        and proof["ref"] == plan["public_proof"]["ref"],
        "Wrong public proof.",
    )
    require(
        proof["submission_sha256"]
        == proof["downloaded_submission_sha256"]
        == metric["submission_sha256"],
        "Public CSV differs.",
    )
    recipe = metric["model_details"]
    actual = {
        Path(p.replace("\\", "/")).name: w
        for p, w in zip(recipe["sources"], recipe["weights"])
        if w
    }
    expected = {Path(plan["inputs"][r]["path"]).name: w for r, w in zip(ROLES, WEIGHTS)}
    require(actual == expected, "Selected probability recipe differs.")
    component = metric["external_components"]
    require(
        len(component) == 1
        and component[0]["predictions_source_version"] == 2
        and component[0]["weight"] == 0.25,
        "External release differs.",
    )
    require(
        {"predictions.csv", "submission.csv"}
        <= plan["inputs"]["external"]["sha256"].keys(),
        "External CSVs unpinned.",
    )
    return loaded, frozen


def final_config(original, name, data, output):
    config = {
        **original,
        "name": name,
        "data": str(data),
        "output": str(output),
        "phase": "full_data_refit",
        "refit": True,
        "validation_note": "All-label refit of a fixed exploratory-CV selection; no new local score or independent holdout.",
    }
    require(
        parameters_sha256(config) == parameters_sha256(original),
        "Refit changed training parameters.",
    )
    return config


def pin_core(worktree, spec):
    git = lambda *args: subprocess.check_output(
        ["git", "-C", str(worktree), *args], text=True
    ).strip()
    require(
        git("rev-parse", "HEAD") == spec["revision"]
        and not git("status", "--porcelain"),
        "Core must be clean at its pinned revision.",
    )
    require(
        {p.name: digest(p) for p in sorted((worktree / "src/fraudlab").glob("*.py"))}
        == spec["sha256"],
        "Core source hashes differ.",
    )
    require("fraudlab" not in sys.modules, "Core imported before source pinning.")
    sys.path.insert(0, str(worktree / "src"))
    core = importlib.import_module("fraudlab")
    require(
        Path(core.__file__).resolve().is_relative_to(worktree / "src"),
        "Wrong core import.",
    )
    os.chdir(worktree)


def fit_and_reload(config, training, y, test, path, fit_predict, predict_saved):
    import numpy as np

    _, prediction, details = fit_predict(
        config["model"], training, y, test, None, test, config, config["seed"], path
    )
    require(
        len(prediction) == len(test)
        and np.isfinite(prediction).all()
        and ((prediction >= 0) & (prediction <= 1)).all(),
        "Invalid full-data predictions.",
    )
    count = min(1024, len(test))
    reloaded = predict_saved(config["model"], path, test.iloc[:count], "cpu")
    error = float(np.max(np.abs(reloaded - prediction[:count])))
    require(error <= 1e-12, "Full-data checkpoint reload differs.")
    return (
        prediction,
        details,
        {
            "test_rows": count,
            "max_absolute_error": error,
            "atol": 1e-12,
            "rtol": 0,
            "checkpoint_sha256": digest(path.with_suffix(".joblib")),
        },
    )


def execute(plan_path, root, worktree):
    started = time.perf_counter()
    plan_path, root, worktree = plan_path.resolve(), root.resolve(), worktree.resolve()
    plan = read(plan_path)
    require(
        digest(ENTRYPOINT) == plan["entrypoint_sha256"],
        "Entrypoint differs from the published plan.",
    )
    loaded, frozen = checked_inputs(root, plan)
    frozen.update(
        {str(plan_path): digest(plan_path), str(ENTRYPOINT): digest(ENTRYPOINT)}
    )
    pin_core(worktree, plan["source"])
    import numpy as np
    import pandas as pd
    from fraudlab.data import load_data, submission, validate_submission
    from fraudlab.experiment import create_run, source_provenance, versions
    from fraudlab.features import transform_features
    from fraudlab.models import fit_predict, predict_saved

    data, output = root / plan["data"], root / plan["output"]
    require(
        data.resolve().is_relative_to(root) and output.resolve().is_relative_to(root),
        "Data/output outside the workspace.",
    )
    for filename, expected in plan["data_hashes"].items():
        require(digest(data / filename) == expected, "Official data changed.")
        frozen[str(data / filename)] = expected
    raw, raw_test, sample = load_data(data, plan["id_column"], plan["target"])
    require(
        len(raw) == plan["rows"]["train"] and len(raw_test) == plan["rows"]["test"],
        "Row counts differ.",
    )
    reference = root / plan["inputs"]["selection"]["path"]
    selection = {
        "source_dev_run": str(reference),
        "reference_cv_run": reference.name,
        "development_metrics_sha256": plan["inputs"]["selection"]["sha256"][
            "metrics.json"
        ],
        "development_roc_auc": plan["reference_cv_auc"],
        "development_roc_auc_scope": "Historical exploratory full-data CV of the selected hybrid; not an AUC of this refit.",
        "independent_holdout": False,
    }
    provenance = source_provenance()
    provenance["source_hashes_timing"] = (
        "Pinned core checked before reading CSVs; hashes captured again before fitting."
    )
    common = {
        **provenance,
        "status": "complete",
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
            "method": "all_labels_refit_of_audited_exploratory_cv",
            "n_splits": 0,
            "independent_holdout": False,
            "early_stopping_uses_validation": False,
        },
        "limitations": [
            "No OOF or independent local score exists for these full-data fits.",
            "Replacing five folds by one all-label model changes inference; improved performance is not established.",
        ],
    }
    predictions, bundles = {}, {}
    for role in ROLES[:2]:
        metric, original = loaded[role]
        spec = plan["inputs"][role]
        config = final_config(original, plan["refit_names"][role], data, output)
        run = create_run(output, config["name"])
        write(run / "config.json", config)
        write(run / "status.json", {"status": "running"})
        try:
            train = transform_features(
                raw, config["feature_set"], (config["id_column"], config["target"])
            )
            test = transform_features(
                raw_test, config["feature_set"], (config["id_column"], config["target"])
            )
            names = metric["features"]["names"]
            require(
                [
                    name
                    for name in train
                    if name not in (config["id_column"], config["target"])
                ]
                == names,
                "Engineered schema differs from the selected CV.",
            )
            prediction, details, reload = fit_and_reload(
                config,
                train[names],
                train[config["target"]].to_numpy(),
                test[names],
                run / "model",
                fit_predict,
                predict_saved,
            )
            if role == "offset":
                require(
                    details["encoded_features"] == spec["encoded_features"],
                    "Full-data encoded feature count differs.",
                )
            frame = submission(
                test[config["id_column"]], prediction, sample, config["target"]
            )
            for filename in ("predictions.csv", "submission.csv"):
                frame.to_csv(run / filename, index=False)
            result = {
                **common,
                "model": role,
                "config_sha256": digest(run / "config.json"),
                "features": metric["features"],
                "model_details": details,
                "checkpoint_reload": reload,
                "source_cv_run": spec["path"],
                "source_parameters_sha256": spec["parameters_sha256"],
                "submission_sha256": digest(run / "submission.csv"),
            }
            if metric.get("feature_encoding") is not None:
                result["feature_encoding"] = metric["feature_encoding"]
            write(run / "metrics.json", result)
            write(run / "status.json", {"status": "complete"})
            predictions[role] = frame[config["target"]].to_numpy()
            bundles[role] = {
                "path": str(run),
                "sha256": {
                    name: digest(run / name)
                    for name in (
                        "config.json",
                        "metrics.json",
                        "model.joblib",
                        "submission.csv",
                    )
                },
            }
            del train, test, prediction
            gc.collect()
        except Exception as error:
            write(run / "status.json", {"status": "failed", "error": str(error)})
            raise
    external = root / plan["inputs"]["external"]["path"]
    external_frame = pd.read_csv(external / "predictions.csv")
    validate_submission(external_frame, sample, plan["id_column"], plan["target"])
    predictions["external"] = external_frame[plan["target"]].to_numpy()
    credited = [
        {k: v for k, v in item.items() if k != "oof_auc"}
        for item in loaded["selection"][0]["external_components"]
    ]
    values = sum(weight * predictions[role] for role, weight in zip(ROLES, WEIGHTS))
    final = submission(sample[plan["id_column"]], values, sample, plan["target"])
    require(
        np.array_equal(final[plan["target"]].to_numpy(), values),
        "Final probability formula differs.",
    )
    for path, expected in frozen.items():
        require(digest(path) == expected, "Input changed during refit: " + path)
    require(
        source_provenance()["source_hashes"] == plan["source"]["sha256"]
        and source_provenance()["git_revision"] == plan["source"]["revision"]
        and source_provenance()["git_dirty"] is False,
        "Pinned core changed.",
    )
    run = create_run(output, plan["name"])
    config = {
        "operation": "refit_selected_exploratory_cv",
        "plan_sha256": digest(plan_path),
        "effective_weights": dict(zip(ROLES, WEIGHTS)),
        "inputs": plan["inputs"],
        "refits": bundles,
    }
    write(run / "config.json", config)
    for filename in ("predictions.csv", "submission.csv"):
        final.to_csv(run / filename, index=False)
    written = pd.read_csv(run / "submission.csv")
    validate_submission(written, sample, plan["id_column"], plan["target"])
    part_paths = [Path(bundles[r]["path"]) for r in ROLES[:2]] + [external]
    parts = [pd.read_csv(p / "submission.csv")[plan["target"]] for p in part_paths]
    expected = sum(w * values.to_numpy() for w, values in zip(WEIGHTS, parts))
    formula_error = float(np.max(np.abs(written[plan["target"]] - expected)))
    require(formula_error <= 1e-15, "Serialized full-test formula differs.")
    metric = {
        **common,
        "model": "blend",
        "feature_encoding": loaded["offset"][0].get("feature_encoding"),
        "features": loaded["offset"][0]["features"],
        "config_sha256": digest(run / "config.json"),
        "effective_weights": dict(zip(ROLES, WEIGHTS)),
        "model_details": {
            "method": "weighted_probability_average",
            "sources": [bundles[role]["path"] for role in ROLES[:2]] + [str(external)],
            "weights": WEIGHTS,
        },
        "refits": bundles,
        "external_components": credited,
        "formula_rows_verified": len(final),
        "formula_max_absolute_error": formula_error,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "submission_sha256": digest(run / "submission.csv"),
    }
    write(run / "metrics.json", metric)
    write(run / "status.json", {"status": "complete"})
    return metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-worktree", type=Path, required=True)
    args = parser.parse_args()
    result = execute(args.plan, args.root, args.source_worktree)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("phase", "roc_auc", "rows", "submission_sha256")
            }
        )
    )
