"""R21: honest inner-holdout meta fitting over three pinned owned members.

Only explicit execution fits models. Importing this module is side-effect free.
The external teacher is never a meta input; it enters one fixed final blend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
BASE_RUNS = [
    "round18-joint-bpe-pooled-offset",
    "round19-transductive-pooled-offset",
    "explore5-lgbm",
]
META = {
    "C": 0.01,
    "solver": "newton-cholesky",
    "tol": 1e-7,
    "max_iter": 200,
    "fit_intercept": True,
    "class_weight": None,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def probabilities(values, rows):
    values = np.asarray(values, dtype=np.float64)
    require(
        values.shape == (rows,)
        and np.isfinite(values).all()
        and ((values >= 0) & (values <= 1)).all(),
        "Invalid member probabilities",
    )
    return values


def logits(values):
    values = np.asarray(values, dtype=np.float64)
    require(
        values.ndim == 2
        and values.shape[1] == 3
        and np.isfinite(values).all()
        and ((values >= 0) & (values <= 1)).all(),
        "Meta input must be exactly three owned probability columns",
    )
    clipped = np.clip(values, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def row_hash(values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def fit_meta(predictions, labels, path, seed, *, max_iter=200):
    labels = np.asarray(labels)
    require(
        labels.shape == (len(predictions),) and set(np.unique(labels)) == {0, 1},
        "Meta labels must be binary and aligned",
    )
    model = LogisticRegression(**{**META, "max_iter": max_iter}, random_state=seed)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        with threadpool_limits(limits=4):
            model.fit(logits(predictions), labels)
    for warning in captured:
        warnings.warn(warning.message, warning.category, stacklevel=2)
    if (
        any(issubclass(item.category, ConvergenceWarning) for item in captured)
        or int(model.n_iter_[0]) >= max_iter
    ):
        raise RuntimeError("Nested meta learner did not converge; no automatic retry.")
    joblib.dump(model, path)
    return model


def fit_fold(
    frame,
    test,
    labels,
    ids,
    outer_train,
    outer_valid,
    members,
    fold,
    directory,
    fit_base,
    predict_outer,
):
    """Callbacks receive only their allowed partitions; no teacher argument exists."""
    require(
        len(members) == 3 and not set(outer_train) & set(outer_valid),
        "Invalid outer membership",
    )
    a, b = train_test_split(
        np.arange(len(outer_train)),
        test_size=0.25,
        stratify=labels[outer_train],
        random_state=2026,
    )
    inner, meta = outer_train[np.sort(a)], outer_train[np.sort(b)]
    scores = np.empty((len(meta), 3))
    validation = np.empty((len(outer_valid), 3))
    testing = np.empty((len(test), 3))
    checkpoints = []
    for index, member in enumerate(members):
        names = member["names"]
        require(
            not {"id", "is_fraud", "transaction_id"}.intersection(names),
            "Protected input in member features",
        )
        stem = directory / f"fold-{fold}-inner-{index}"
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", ConvergenceWarning)
            predicted, dummy, details = fit_base(
                index,
                frame.iloc[inner][names],
                labels[inner],
                frame.iloc[meta][names],
                test.iloc[:1][names],
                42 + fold,
                stem,
            )
        for warning in captured:
            warnings.warn(warning.message, warning.category, stacklevel=2)
        if any(issubclass(item.category, ConvergenceWarning) for item in captured):
            raise RuntimeError(
                "Nested base learner did not converge; no automatic retry."
            )
        scores[:, index] = probabilities(predicted, len(meta))
        probabilities(dummy, 1)
        validation[:, index], testing[:, index] = predict_outer(
            index, fold, frame.iloc[outer_valid][names], test[names]
        )
        probabilities(validation[:, index], len(outer_valid))
        probabilities(testing[:, index], len(test))
        checkpoints.append(
            {
                "member": member["run"],
                "path": str(stem.with_suffix(".joblib")),
                "sha256": sha(stem.with_suffix(".joblib")),
                "details": details,
            }
        )
    scores_path = directory / f"fold-{fold}-meta-scores.npy"
    np.save(scores_path, scores)
    meta_path = directory / f"fold-{fold}-meta.joblib"
    model = fit_meta(scores, labels[meta], meta_path, 42 + fold)
    record = {
        "fold": fold,
        "seed": 42 + fold,
        "inner_training_rows": len(inner),
        "meta_training_rows": len(meta),
        "outer_validation_rows": len(outer_valid),
        "inner_ids_sha256": row_hash(ids[inner]),
        "meta_ids_sha256": row_hash(ids[meta]),
        "outer_validation_ids_sha256": row_hash(ids[outer_valid]),
        "inner_checkpoints": checkpoints,
        "meta_scores_sha256": sha(scores_path),
        "meta_checkpoint_sha256": sha(meta_path),
        "meta_iterations": int(model.n_iter_[0]),
        "coefficients_logit": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "outer_checkpoint_refit": False,
    }
    return (
        model.predict_proba(logits(validation))[:, 1],
        model.predict_proba(logits(testing))[:, 1],
        record,
    )


def validate_plan(plan):
    require(
        [item["run"] for item in plan["sources"]] == BASE_RUNS,
        "The three owned sources/order are fixed",
    )
    require(
        {key: plan["meta"][key] for key in META} == META
        and plan["meta"]["clip"] == 1e-6,
        "Meta configuration changed",
    )
    require(
        plan["outer"]
        == {
            "folds": 5,
            "shuffle": True,
            "seed": 42,
            "phase": "exploratory_full_data_cv",
            "independent_holdout": False,
        },
        "Outer protocol changed",
    )
    require(
        plan["inner"]["seed"] == 2026
        and plan["inner"]["meta_fraction"] == 0.25
        and plan["inference"]["teacher_weight"] == 0.25,
        "Inner/teacher protocol changed",
    )


def clean_core(source, plan):
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip()
    require(
        head == plan["core_revision"] and not dirty,
        "Core must be clean at the pinned revision",
    )
    actual = {
        path.name: sha(path) for path in sorted((source / "src/fraudlab").glob("*.py"))
    }
    require(actual == plan["core_source_hashes"], "Pinned core files changed")


def audit_run(path, pin, data_hashes, ids, labels, folds, test_ids):
    for filename, key in [
        ("config.json", "config_sha256"),
        ("metrics.json", "metrics_sha256"),
        ("oof.csv", "oof_sha256"),
        ("predictions.csv", "predictions_sha256"),
    ]:
        require(
            sha(path / filename) == pin[key],
            f"Source pin changed: {path.name}/{filename}",
        )
    config, metric = read(path / "config.json"), read(path / "metrics.json")
    require(
        metric["status"] == "complete" and metric["data_hashes"] == data_hashes,
        "Source status/data mismatch",
    )
    oof, prediction = pd.read_csv(path / "oof.csv"), pd.read_csv(
        path / "predictions.csv"
    )
    require(
        np.array_equal(
            oof[["id", "fold", "target"]].to_numpy(),
            np.column_stack([ids, folds, labels]),
        ),
        "Source OOF ID/order/fold/labels mismatch",
    )
    require(np.array_equal(prediction.id, test_ids), "Source test IDs/order mismatch")
    probabilities(oof.prediction, len(ids))
    probabilities(prediction.is_fraud, len(test_ids))
    if "outer_checkpoints_sha256" in pin:
        require(
            metric["model"] in ("offset", "lightgbm")
            and not metric.get("model_details", {}).get("external_components"),
            "Meta base must be owned",
        )
        require(
            config.get("fixed_training")
            and not config.get("original_data")
            and not config.get("holdout_fraction"),
            "Source training contract mismatch",
        )
        require(
            config["model"] == pin["model"]
            and config["feature_set"] == pin["feature_set"]
            and config["seed"] == 42
            and config["folds"] == 5
            and config["split"] == "stratified",
            "Source model/fold protocol changed",
        )
        for fold, expected in pin["outer_checkpoints_sha256"].items():
            require(
                sha(path / f"fold-{fold}.joblib") == expected, "Outer model pin changed"
            )
    return config, metric, oof.prediction.to_numpy(), prediction.is_fraud.to_numpy()


def run(plan_path, source, output):
    plan_path, source, output = (
        Path(plan_path).resolve(),
        Path(source).resolve(),
        Path(output).resolve(),
    )
    plan = read(plan_path)
    validate_plan(plan)
    clean_core(source, plan)
    plan_hash = sha(plan_path)
    require(
        sha(__file__) == plan["implementation_sha256"], "Published runner pin changed"
    )
    require(
        not any((output / plan[key]).exists() for key in ("own_run", "hybrid_run")),
        "R21 output already exists; refuse a partial or automatic retry",
    )
    sys.path.insert(0, str(source / "src"))
    import fraudlab
    from fraudlab.data import load_data, submission, write_json
    from fraudlab.experiment import blend, create_run, versions
    from fraudlab.features import transform_features
    from fraudlab.models import clean_features, fit_predict
    from fraudlab.transductive import build_reference_counts

    require(
        Path(fraudlab.__file__).resolve().is_relative_to(source / "src"),
        "Another core was imported",
    )
    data = ROOT / "data/raw"
    hashes = {name: sha(data / name) for name in plan["data_hashes"]}
    require(hashes == plan["data_hashes"], "Official data changed")
    raw, test, sample = load_data(data, "id", "is_fraud")
    require(
        (len(raw), len(test))
        == (plan["expected_rows"]["train"], plan["expected_rows"]["test"]),
        "Row counts changed",
    )
    ids, labels, test_ids = (
        raw.id.to_numpy(),
        raw.is_fraud.to_numpy(),
        test.id.to_numpy(),
    )
    splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(ids, labels))
    folds = np.full(len(raw), -1)
    for fold, (_, validation) in enumerate(splits):
        folds[validation] = fold
    runs = ROOT / "artifacts/runs"
    members, source_oof, source_test = [], [], []
    for pin in plan["sources"]:
        config, metric, p, t = audit_run(
            runs / pin["run"], pin, hashes, ids, labels, folds, test_ids
        )
        members.append({**pin, "config": config, "names": metric["features"]["names"]})
        source_oof.append(p)
        source_test.append(t)
    teacher = audit_run(
        runs / plan["teacher"]["run"],
        plan["teacher"],
        hashes,
        ids,
        labels,
        folds,
        test_ids,
    )
    teacher_component = teacher[1]["model_details"]["external_components"][0]
    require(
        teacher_component["predictions_source_version"] == 2
        and teacher_component["predictions_license"] == "CC BY 4.0"
        and teacher_component["trained_by_us"] is False,
        "Teacher provenance mismatch",
    )
    baselines = {}
    for prefix in ("r18", "r10"):
        baseline = runs / plan["gate"][prefix + "_run"]
        require(
            sha(baseline / "metrics.json") == plan["gate"][prefix + "_metrics_sha256"],
            "Gate baseline changed",
        )
        baselines[prefix] = read(baseline / "metrics.json")
        require(
            baselines[prefix]["status"] == "complete"
            and baselines[prefix]["data_hashes"] == hashes
            and baselines[prefix]["roc_auc"] == plan["gate"][prefix + "_oof_auc"],
            "Gate baseline contract changed",
        )
    frame = transform_features(raw.drop(columns=["id", "is_fraud"]), "joint_bpe")
    test = transform_features(test.drop(columns="id"), "joint_bpe")
    del raw
    reference, reference_meta = build_reference_counts(
        frame, test, members[1]["names"], hashes
    )
    require(
        reference_meta == members[1]["config"]["transductive_frequency"],
        "R19 global feature reference changed",
    )
    directory = create_run(output, plan["own_run"])
    write(
        directory / "config.json",
        {
            "plan": plan,
            "plan_path": str(plan_path),
            "plan_sha256": sha(plan_path),
            "model": "nested_stack",
        },
    )
    write(directory / "status.json", {"status": "running"})
    predictions, oof, details = np.zeros(len(test)), np.full(len(ids), np.nan), []
    outer_test = np.zeros((3, len(test)))

    def fit_base(index, training, y, meta, dummy, seed, path):
        config = dict(members[index]["config"])
        if index == 1:
            config["_frequency_reference"] = reference
        result = fit_predict(
            config["model"], training, y, meta, None, dummy, config, seed, path
        )
        if index < 2:
            classifier = joblib.load(path.with_suffix(".joblib")).glm.named_steps[
                "classifier"
            ]
            require(
                int(classifier.n_iter_[0]) < classifier.max_iter,
                "Nested base GLM reached its iteration cap; no automatic retry",
            )
            result[2]["glm"].update(converged=True, max_iter=classifier.max_iter)
        return result

    def predict_outer(index, fold, validation, testing):
        member = members[index]
        path = runs / member["run"] / f"fold-{fold}.joblib"
        require(
            sha(path) == member["outer_checkpoints_sha256"][str(fold)],
            "Outer checkpoint changed before inference",
        )
        model = joblib.load(path)
        a = model.predict_proba(clean_features(validation))[:, 1]
        b = model.predict_proba(clean_features(testing))[:, 1]
        require(
            np.allclose(a, source_oof[index][splits[fold][1]], rtol=0, atol=1e-12),
            "Outer checkpoint differs from its own complete saved OOF fold",
        )
        outer_test[index] += b / 5
        require(
            sha(path) == member["outer_checkpoints_sha256"][str(fold)],
            "Outer checkpoint changed during inference",
        )
        return a, b

    try:
        for fold, (training, validation) in enumerate(splits):
            p, t, record = fit_fold(
                frame,
                test,
                labels,
                ids,
                training,
                validation,
                members,
                fold,
                directory,
                fit_base,
                predict_outer,
            )
            oof[validation] = p
            predictions += t / 5
            details.append(record)
            write(directory / f"fold-{fold}.json", record)
            print(
                json.dumps(
                    {
                        "event": "outer_fold_complete",
                        "fold": fold,
                        "inner_rows": record["inner_training_rows"],
                        "meta_rows": record["meta_training_rows"],
                    }
                ),
                flush=True,
            )
        probabilities(oof, len(ids))
        probabilities(predictions, len(test))
        clean_core(source, plan)
        require(
            sha(plan_path) == plan_hash
            and sha(__file__) == plan["implementation_sha256"],
            "Runner or plan changed during fitting",
        )
        for actual, expected in zip(outer_test, source_test):
            require(
                np.allclose(actual, expected, rtol=0, atol=1e-12),
                "Outer checkpoint ensemble differs from saved member test scores",
            )
        require(
            {name: sha(data / name) for name in hashes} == hashes,
            "Official data changed during fitting",
        )
        for pin in plan["sources"] + [plan["teacher"]]:
            audit_run(runs / pin["run"], pin, hashes, ids, labels, folds, test_ids)
        oof_frame = pd.DataFrame(
            {"id": ids, "fold": folds, "target": labels, "prediction": oof}
        )
        oof_frame.to_csv(directory / "oof.csv", index=False)
        oof = pd.read_csv(directory / "oof.csv").prediction.to_numpy()
        final = submission(
            pd.Series(test_ids, name="id"), predictions, sample, "is_fraud"
        )
        final.to_csv(directory / "predictions.csv", index=False)
        final.to_csv(directory / "submission.csv", index=False)
        architecture = {
            "name": "NestedOwnedLogitStack",
            "base_runs": BASE_RUNS,
            "meta_input": "three clipped probability logits",
            "meta": META,
            "external_teacher_in_meta": False,
        }
        metric = {
            "status": "complete",
            "phase": "exploratory_full_data_cv",
            "model": "nested_stack",
            "roc_auc": float(roc_auc_score(labels, oof)),
            "fold_scores": [
                float(roc_auc_score(labels[folds == i], oof[folds == i]))
                for i in range(5)
            ],
            "rows": {"train": len(ids), "test": len(test), "oof": len(ids)},
            "data_hashes": hashes,
            "seed": 42,
            "validation": {
                "method": "stratified",
                "n_splits": 5,
                "holdout_fraction": 0.0,
                "independent_holdout": False,
                "early_stopping_uses_validation": False,
                "inner_fraction": 0.25,
                "base_training_fraction": 0.75,
                "meta_training_fraction": 0.25,
                "inner_seed": 2026,
            },
            "versions": versions(),
            "config_sha256": sha(directory / "config.json"),
            "source_hashes": plan["core_source_hashes"],
            "git_revision": plan["core_revision"],
            "git_dirty": False,
            "git_provenance_scope": "Pinned core worktree; runner is pinned separately by SHA256",
            "runner_sha256": sha(__file__),
            "plan_sha256": sha(plan_path),
            "model_details": {
                "algorithm": "nested_owned_logit_holdout_stack",
                "architecture": architecture,
                "base_runs": BASE_RUNS,
                "inner_fraction": 0.25,
                "inner_seed": 2026,
                "folds": details,
                "stages": [
                    "new inner-base fits",
                    "inner-holdout meta fit",
                    "existing outer-checkpoint inference; no outer refit",
                ],
                "coefficient_space": "base logits; not probability percentages",
                "external_components": [],
            },
            "submission_sha256": sha(directory / "submission.csv"),
            "artifact_hashes": {
                name: sha(directory / name)
                for name in ("oof.csv", "predictions.csv", "submission.csv")
            },
            "limitations": plan["limitations"],
        }
        metric["fold_std"] = float(np.std(metric["fold_scores"]))
        write_json(directory / "metrics.json", metric)
        hybrid = blend(
            [directory, runs / plan["teacher"]["run"]],
            [0.75, 0.25],
            output,
            plan["hybrid_run"],
        )
        hybrid_oof = pd.read_csv(
            output / plan["hybrid_run"] / "oof.csv"
        ).prediction.to_numpy()
        require(
            np.allclose(hybrid_oof, 0.75 * oof + 0.25 * teacher[2], rtol=0, atol=1e-15),
            "Fixed teacher blend differs",
        )
        saved_own_test = pd.read_csv(directory / "predictions.csv").is_fraud.to_numpy()
        saved_hybrid_test = pd.read_csv(
            output / plan["hybrid_run"] / "predictions.csv"
        ).is_fraud.to_numpy()
        require(
            np.allclose(
                saved_hybrid_test,
                0.75 * saved_own_test + 0.25 * teacher[3],
                rtol=0,
                atol=1e-15,
            ),
            "Fixed teacher test blend differs",
        )
        for prefix in ("r18", "r10"):
            baseline = runs / plan["gate"][prefix + "_run"]
            require(
                sha(baseline / "metrics.json")
                == plan["gate"][prefix + "_metrics_sha256"],
                "Gate baseline changed",
            )
        improved = int(
            np.count_nonzero(
                np.array(hybrid["fold_scores"]) > baselines["r18"]["fold_scores"]
            )
        )
        eligible = (
            hybrid["roc_auc"] > plan["gate"]["r18_oof_auc"]
            and hybrid["roc_auc"] - plan["gate"]["r10_oof_auc"] > 0.0001
            and improved >= 4
        )
        result = {
            "status": "complete",
            "plan_sha256": sha(plan_path),
            "own_run": directory.name,
            "hybrid_run": plan["hybrid_run"],
            "own_auc": metric["roc_auc"],
            "hybrid_auc": hybrid["roc_auc"],
            "improved_folds_vs_r18": improved,
            "eligible_for_independent_review": bool(eligible),
            "new_base_fits": 15,
            "meta_fits": 5,
            "new_kaggle_submissions": 0,
        }
        write(directory / "results.json", result)
        write(directory / "status.json", {"status": "complete"})
        return result
    except Exception as error:
        for candidate in (directory, output / plan["hybrid_run"]):
            if candidate.exists():
                failed = {"status": "failed", "error": str(error)}
                write(candidate / "status.json", failed)
                if (candidate / "metrics.json").exists():
                    write(
                        candidate / "metrics.json",
                        {**read(candidate / "metrics.json"), **failed},
                    )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-tree", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/runs")
    args = parser.parse_args()
    print(json.dumps(run(args.plan, args.source_tree, args.output), indent=2))


if __name__ == "__main__":
    main()
