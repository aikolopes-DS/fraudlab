"""Manufactured orchestration proves base/meta/outer membership separation."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold

SPEC = importlib.util.spec_from_file_location(
    "nested_stack",
    Path(__file__).resolve().parents[1] / "scripts/train_nested_stack.py",
)
stack = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stack)


class ToyOuter:
    def __init__(self, member):
        self.member = member
        self.glm = SimpleNamespace(
            named_steps={"classifier": SimpleNamespace(n_iter_=[2], max_iter=100)}
        )

    def predict_proba(self, frame):
        number = frame.number.to_numpy()
        p = 0.15 + 0.2 * (number % 2) + 0.02 * self.member + 0.0001 * number
        return np.column_stack((1 - p, p))


def test_inner_bases_exclude_meta_and_outer_labels_and_inference_uses_outer_models(
    tmp_path,
):
    frame = pd.DataFrame({"number": np.arange(100, dtype=float)})
    test = pd.DataFrame({"number": np.arange(100, 110, dtype=float)})
    labels = np.tile([0, 1], 50)
    members = [{"run": name, "names": ["number"]} for name in stack.BASE_RUNS]
    seen, outer = [], []

    def fit_base(index, training, y, meta, dummy, seed, checkpoint):
        a = training.number.to_numpy(dtype=int)
        b = meta.number.to_numpy(dtype=int)
        assert not set(a) & set(b)
        assert not set(a) & set(range(80, 100))
        assert not set(b) & set(range(80, 100))
        np.testing.assert_array_equal(y, labels[a])
        assert len(a) == 60 and len(b) == 20 and len(dummy) == 1
        assert seed == 42 and set(training) == {"number"}
        checkpoint.with_suffix(".joblib").write_bytes(
            f"new-inner-member-{index}".encode()
        )
        seen.append((a, b))
        return 0.1 + 0.7 * labels[b] + 0.01 * index, np.array([0.2]), {"toy": True}

    def predict_outer(index, fold, validation, testing):
        outer.append((index, fold, validation.index.tolist()))
        return np.full(len(validation), 0.2 + 0.1 * index), np.full(
            len(testing), 0.3 + 0.1 * index
        )

    a, b, record = stack.fit_fold(
        frame,
        test,
        labels,
        np.arange(100),
        np.arange(80),
        np.arange(80, 100),
        members,
        0,
        tmp_path,
        fit_base,
        predict_outer,
    )
    assert len(seen) == len(outer) == 3
    assert all(np.array_equal(seen[0][i], item[i]) for item in seen for i in (0, 1))
    assert record["inner_training_rows"] == 60 and record["meta_training_rows"] == 20
    assert (
        record["outer_validation_rows"] == 20
        and record["outer_checkpoint_refit"] is False
    )
    assert (
        len(record["inner_checkpoints"]) == 3 and len(record["coefficients_logit"]) == 3
    )
    model = joblib.load(tmp_path / "fold-0-meta.joblib")
    expected = np.tile([0.2, 0.3, 0.4], (20, 1))
    np.testing.assert_allclose(
        a, model.predict_proba(stack.logits(expected))[:, 1], rtol=0, atol=1e-14
    )
    assert len(b) == 10
    # Validation labels do not influence either split, any base fit or meta fit.
    changed = labels.copy()
    changed[80:] = 1 - changed[80:]
    again = tmp_path / "again"
    again.mkdir()
    modified, _, other = stack.fit_fold(
        frame,
        test,
        changed,
        np.arange(100),
        np.arange(80),
        np.arange(80, 100),
        members,
        0,
        again,
        fit_base,
        predict_outer,
    )
    np.testing.assert_array_equal(a, modified)
    assert record["meta_ids_sha256"] == other["meta_ids_sha256"]


def test_meta_uses_only_three_clipped_logits_and_persists_actual_coefficients(tmp_path):
    probabilities = np.tile([[0.0, 0.2, 0.3], [1.0, 0.8, 0.7]], (30, 1))
    labels = np.tile([0, 1], 30)
    x = stack.logits(probabilities)
    assert x.shape == (60, 3) and np.isfinite(x).all()
    path = tmp_path / "meta.joblib"
    model = stack.fit_meta(probabilities, labels, path, 42)
    assert model.C == 0.01 and model.solver == "newton-cholesky"
    assert model.tol == 1e-7 and model.max_iter == 200
    np.testing.assert_array_equal(
        joblib.load(path).predict_proba(x), model.predict_proba(x)
    )
    for invalid in [np.zeros((60, 4)), np.full((60, 3), np.nan), np.full((60, 3), 1.1)]:
        with pytest.raises(ValueError):
            stack.logits(invalid)


def test_nonconverged_real_meta_fit_warns_and_does_not_create_checkpoint(tmp_path):
    rng = np.random.default_rng(23)
    predictions = rng.uniform(0.001, 0.999, (400, 3))
    labels = (predictions[:, 0] + predictions[:, 1] > 1).astype(int)
    path = tmp_path / "invalid.joblib"
    with pytest.warns(ConvergenceWarning):
        with pytest.raises(RuntimeError, match="did not converge"):
            stack.fit_meta(predictions, labels, path, 42, max_iter=1)
    assert not path.exists()


def test_base_convergence_warning_stops_before_outer_inference(tmp_path):
    frame = pd.DataFrame({"number": np.arange(100, dtype=float)})
    members = [{"run": name, "names": ["number"]} for name in stack.BASE_RUNS]

    def failed_fit(*args):
        warnings.warn("Manufactured base failure", ConvergenceWarning)
        return np.zeros(20), np.zeros(1), {}

    def forbidden(*args):
        pytest.fail("A nonconverged member must not reach inference")

    with pytest.warns(ConvergenceWarning, match="Manufactured"):
        with pytest.raises(RuntimeError, match="base learner did not converge"):
            stack.fit_fold(
                frame,
                frame.iloc[:1],
                np.tile([0, 1], 50),
                np.arange(100),
                np.arange(80),
                np.arange(80, 100),
                members,
                0,
                tmp_path,
                failed_fit,
                forbidden,
            )


def test_tiny_complete_run_persists_provenance_and_only_fixed_teacher_blend(
    tmp_path,
    monkeypatch,
):
    import fraudlab.features
    import fraudlab.models
    import fraudlab.transductive

    repository = Path(__file__).resolve().parents[1]
    plan = json.loads((repository / "evidence/round21-plan.json").read_text())
    data = tmp_path / "data/raw"
    runs = tmp_path / "artifacts/runs"
    data.mkdir(parents=True)
    runs.mkdir(parents=True)
    ids, test_ids = np.arange(200), np.arange(200, 220)
    y = ids % 2
    pd.DataFrame({"id": ids, "number": ids.astype(float), "is_fraud": y}).to_csv(
        data / "train.csv", index=False
    )
    pd.DataFrame({"id": test_ids, "number": test_ids.astype(float)}).to_csv(
        data / "test.csv", index=False
    )
    pd.DataFrame({"id": test_ids, "is_fraud": np.zeros(20)}).to_csv(
        data / "sample_submission.csv", index=False
    )
    hashes = {name: stack.sha(data / name) for name in plan["data_hashes"]}
    folds = np.full(200, -1)
    memberships = list(StratifiedKFold(5, shuffle=True, random_state=42).split(ids, y))
    for k, (_, validation) in enumerate(memberships):
        folds[validation] = k
    validation = {"method": "stratified", "n_splits": 5, "holdout_fraction": 0.0}
    for index, pin in enumerate(plan["sources"] + [plan["teacher"]]):
        directory = runs / pin["run"]
        directory.mkdir()
        model = ToyOuter(index)
        p = model.predict_proba(pd.DataFrame({"number": ids}))[:, 1]
        t = model.predict_proba(pd.DataFrame({"number": test_ids}))[:, 1]
        config = {
            "model": pin.get("model", "external"),
            "feature_set": pin.get("feature_set", "raw"),
            "fixed_training": True,
            "folds": 5,
            "seed": 42,
            "split": "stratified",
            "holdout_fraction": 0.0,
            "transductive_frequency": {},
        }
        components = (
            []
            if index < 3
            else [
                {
                    "predictions_source_version": 2,
                    "predictions_license": "CC BY 4.0",
                    "predictions_source_url": "https://example.invalid/toy-teacher",
                    "trained_by_us": False,
                    "weight": 1.0,
                }
            ]
        )
        metric = {
            "status": "complete",
            "phase": "exploratory_full_data_cv",
            "data_hashes": hashes,
            "model": config["model"],
            "features": {"names": ["number"]},
            "validation": validation,
            "model_details": {"external_components": components},
        }
        stack.write(directory / "config.json", config)
        stack.write(directory / "metrics.json", metric)
        pd.DataFrame({"id": ids, "fold": folds, "target": y, "prediction": p}).to_csv(
            directory / "oof.csv", index=False
        )
        pd.DataFrame({"id": test_ids, "is_fraud": t}).to_csv(
            directory / "predictions.csv", index=False
        )
        for filename, key in [
            ("config.json", "config_sha256"),
            ("metrics.json", "metrics_sha256"),
            ("oof.csv", "oof_sha256"),
            ("predictions.csv", "predictions_sha256"),
        ]:
            pin[key] = stack.sha(directory / filename)
        if index < 3:
            for k in range(5):
                checkpoint = directory / f"fold-{k}.joblib"
                joblib.dump(model, checkpoint)
                pin["outer_checkpoints_sha256"][str(k)] = stack.sha(checkpoint)
    for prefix in ("r18", "r10"):
        directory = runs / plan["gate"][prefix + "_run"]
        directory.mkdir()
        stack.write(
            directory / "metrics.json",
            {
                "status": "complete",
                "data_hashes": hashes,
                "roc_auc": 0.5,
                "fold_scores": [0.5] * 5,
            },
        )
        plan["gate"][prefix + "_metrics_sha256"] = stack.sha(directory / "metrics.json")
        plan["gate"][prefix + "_oof_auc"] = 0.5
    plan.update(
        data_hashes=hashes,
        expected_rows={"train": 200, "test": 20},
        implementation_sha256=stack.sha(stack.__file__),
    )
    plan_path = tmp_path / "plan.json"
    stack.write(plan_path, plan)
    calls = []
    cap = False

    def fit_base(name, training, labels, meta, meta_y, dummy, config, seed, path):
        outer = seed - 42
        allowed, excluded = memberships[outer]
        assert len(training) == 120 and len(meta) == 40 and len(dummy) == 1
        assert set(training.index) <= set(allowed)
        assert not set(training.index) & set(meta.index)
        assert not (set(training.index) | set(meta.index)) & set(excluded)
        np.testing.assert_array_equal(labels, y[training.index])
        assert meta_y is None and set(training) == {"number"}
        member = len(calls) % 3
        assert ("_frequency_reference" in config) == (member == 1)
        model = ToyOuter(member)
        if cap:
            model.glm.named_steps["classifier"].n_iter_ = [100]
        joblib.dump(model, path.with_suffix(".joblib"))
        calls.append((outer, member))
        return (
            model.predict_proba(meta)[:, 1],
            model.predict_proba(dummy)[:, 1],
            {"glm": {}},
        )

    monkeypatch.setattr(stack, "ROOT", tmp_path)
    monkeypatch.setattr(stack, "clean_core", lambda *args: None)
    monkeypatch.setattr(fraudlab.features, "transform_features", lambda frame, _: frame)
    monkeypatch.setattr(fraudlab.models, "fit_predict", fit_base)
    monkeypatch.setattr(
        fraudlab.transductive, "build_reference_counts", lambda *args: ({}, {})
    )
    result = stack.run(plan_path, repository, runs)
    own = runs / plan["own_run"]
    hybrid = runs / plan["hybrid_run"]
    metric = stack.read(own / "metrics.json")
    assert len(calls) == result["new_base_fits"] == 15
    assert metric["model_details"]["base_runs"] == stack.BASE_RUNS
    assert metric["model_details"]["inner_fraction"] == 0.25
    assert metric["model_details"]["inner_seed"] == 2026
    assert metric["model_details"]["external_components"] == []
    assert len(metric["model_details"]["folds"]) == 5
    assert len(list(own.glob("fold-*-inner-*.joblib"))) == 15
    assert len(list(own.glob("fold-*-meta.joblib"))) == 5
    own_oof = pd.read_csv(own / "oof.csv")
    hybrid_oof = pd.read_csv(hybrid / "oof.csv")
    teacher = runs / plan["teacher"]["run"]
    np.testing.assert_allclose(
        hybrid_oof.prediction,
        0.75 * own_oof.prediction + 0.25 * pd.read_csv(teacher / "oof.csv").prediction,
        rtol=0,
        atol=1e-15,
    )
    components = stack.read(hybrid / "metrics.json")["model_details"][
        "external_components"
    ]
    assert len(components) == 1 and components[0]["weight"] == 0.25
    assert result["new_kaggle_submissions"] == 0
    with pytest.raises(ValueError, match="already exists"):
        stack.run(plan_path, repository, runs)
    teacher_oof = teacher / "oof.csv"
    teacher_oof.write_bytes(teacher_oof.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Source pin changed"):
        stack.audit_run(teacher, plan["teacher"], hashes, ids, y, folds, test_ids)
    teacher_oof.write_bytes(teacher_oof.read_bytes()[:-1])
    cap = True
    plan["own_run"] += "-failed"
    plan["hybrid_run"] += "-failed"
    stack.write(plan_path, plan)
    with pytest.raises(ValueError, match="iteration cap"):
        stack.run(plan_path, repository, runs)
    failed = runs / plan["own_run"]
    assert stack.read(failed / "status.json")["status"] == "failed"
    assert not (failed / "oof.csv").exists()


def test_public_plan_fixes_members_protocol_and_teacher_outside_learning():
    path = Path(__file__).resolve().parents[1] / "evidence/round21-plan.json"
    plan = json.loads(path.read_text())
    stack.validate_plan(plan)
    assert plan["implementation_sha256"] == stack.sha(stack.__file__)
    assert plan["base_training"]["new_fits"] == 15
    assert len(plan["sources"]) == 3
    assert all(
        len(member["outer_checkpoints_sha256"]) == 5 for member in plan["sources"]
    )
    assert plan["teacher"]["run"] not in stack.BASE_RUNS
    assert plan["inference"]["teacher_weight"] == 0.25
    plan["sources"][-1]["run"] = plan["teacher"]["run"]
    with pytest.raises(ValueError, match="owned sources"):
        stack.validate_plan(plan)
