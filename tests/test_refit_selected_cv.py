"""Manufactured CV receipts and two tiny real CPU fits; never competition fitting."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from fraudlab import experiment, models
from fraudlab.refit import refit

SPEC = importlib.util.spec_from_file_location(
    "refit_selected_cv", Path(__file__).parents[1] / "scripts/refit_selected_cv.py"
)
candidate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(candidate)


@pytest.fixture
def chain(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    rng = np.random.default_rng(42)
    raw = pd.DataFrame(
        {
            "id": range(72),
            "value": rng.normal(size=72),
            "category": np.resize(["a", "b"], 72),
        }
    )
    training, test = raw.iloc[:60].copy(), raw.iloc[60:].copy()
    training["is_fraud"] = np.tile([0, 1], 30)
    sample = pd.DataFrame({"id": test.id.iloc[::-1], "is_fraud": 0})
    for name, frame in (
        ("train", training),
        ("test", test),
        ("sample_submission", sample),
    ):
        frame.to_csv(data / (name + ".csv"), index=False)
    hashes = {p.name: candidate.digest(p) for p in data.iterdir()}
    names = {role: "cv-" + role for role in ("selection", *candidate.ROLES)}
    configs, metrics, specs = {}, {}, {}
    component = {
        "predictions_source_version": 2,
        "predictions_license": "CC BY 4.0",
        "trained_by_us": False,
    }
    for role, name in names.items():
        directory = tmp_path / "runs" / name
        directory.mkdir(parents=True)
        cfg = dict(
            name=name,
            model=role,
            phase="exploratory_full_data_cv",
            feature_set="raw",
            folds=5,
            split="stratified",
            seed=42,
            holdout_fraction=0,
            fixed_training=True,
            id_column="id",
            target="is_fraud",
            threads=1,
            device="cpu",
            iterations=8,
            num_leaves=4,
            learning_rate=0.03,
            l2_leaf_reg=5,
            min_child_samples=5,
            logistic_c=0.3,
            target_smoothing=20,
            glm_frequency=False,
            glm_pooled_tokens=False,
            data="data",
            output="runs",
        )
        if role == "selection":
            cfg = {
                "sources": [str(tmp_path / "runs" / names[r]) for r in candidate.ROLES],
                "weights": candidate.WEIGHTS,
            }
        candidate.write(directory / "config.json", cfg)
        candidate.write(directory / "status.json", {"status": "complete"})
        metric = {
            "status": "complete",
            "phase": "exploratory_full_data_cv",
            "model": role,
            "config_sha256": candidate.digest(directory / "config.json"),
            "roc_auc": 0.91,
            "rows": {"train": 60, "test": 12, "oof": 60},
            "data_hashes": hashes,
            "features": {
                "count": 2,
                "numeric": 1,
                "categorical": 1,
                "names": ["value", "category"],
                "excluded": ["id", "is_fraud"],
            },
            "model_details": {},
        }
        predictions = sample.assign(is_fraud=np.linspace(0.1, 0.9, len(sample)))
        predictions.to_csv(directory / "predictions.csv", index=False)
        predictions.to_csv(directory / "submission.csv", index=False)
        metric["submission_sha256"] = candidate.digest(directory / "submission.csv")
        if role in candidate.ROLES[:2]:
            for fold in range(5):
                (directory / f"fold-{fold}.joblib").write_bytes(
                    b"Fabricated CV receipt, never deserialized"
                )
        if role == "selection":
            metric["model_details"] = cfg.copy()
            metric["external_components"] = [{**component, "weight": 0.25}]
        if role == "external":
            metric["model_details"]["external_components"] = [component]
        candidate.write(directory / "metrics.json", metric)
        spec = {
            "path": directory.relative_to(tmp_path).as_posix(),
            "sha256": {p.name: candidate.digest(p) for p in directory.iterdir()},
        }
        if role in candidate.ROLES[:2]:
            spec["parameters_sha256"] = candidate.parameters_sha256(cfg)
        if role == "offset":
            spec["encoded_features"] = 4
        configs[role], metrics[role], specs[role] = cfg, metric, spec
    audit = {
        "status": "passed",
        "source_clean_at_start_and_finish": True,
        "source_revision": "toy-core",
        "competition_data_hashes": hashes,
        "selected_within_round": {
            "run": names["selection"],
            "oof_auc": 0.91,
            "eligible_to_submit": True,
        },
        "inputs_sha256": {
            spec["path"] + "/" + name: spec["sha256"][name]
            for spec in specs.values()
            for name in ("config.json", "metrics.json")
        },
    }
    proof = {
        "ref": 123,
        "run": names["selection"],
        "submission_file_verified": True,
        "promotion_gate_passed": True,
        "submission_sha256": metrics["selection"]["submission_sha256"],
        "downloaded_submission_sha256": metrics["selection"]["submission_sha256"],
    }
    candidate.write(tmp_path / "audit.json", audit)
    candidate.write(tmp_path / "public.json", proof)
    plan = {
        "name": "final-hybrid",
        "weights": candidate.WEIGHTS,
        "inputs": specs,
        "rows": {"train": 60, "test": 12},
        "data_hashes": hashes,
        "source": {"revision": "toy-core", "sha256": {}},
        "audit": {
            "path": "audit.json",
            "sha256": candidate.digest(tmp_path / "audit.json"),
        },
        "public_proof": {
            "path": "public.json",
            "sha256": candidate.digest(tmp_path / "public.json"),
            "ref": 123,
        },
        "reference_cv_auc": 0.91,
        "entrypoint_sha256": candidate.digest(candidate.ENTRYPOINT),
        "data": "data",
        "output": "runs",
        "id_column": "id",
        "target": "is_fraud",
        "refit_names": {"offset": "final-offset", "lightgbm": "final-lightgbm"},
    }
    candidate.write(tmp_path / "plan.json", plan)
    return tmp_path, plan, configs, metrics


@pytest.mark.parametrize("failure", ["parameter", "bytes", "failed", "audit"])
def test_reject_changed_params_bytes_failed_cv_or_missing_audit(chain, failure):
    root, plan, configs, _ = chain
    candidate.checked_inputs(root, plan)
    if failure == "parameter":
        plan["inputs"]["offset"]["parameters_sha256"] = "0" * 64
    elif failure == "bytes":
        (root / plan["inputs"]["offset"]["path"] / "fold-0.joblib").write_bytes(
            b"changed"
        )
    elif failure == "failed":
        path = root / plan["inputs"]["offset"]["path"] / "status.json"
        candidate.write(path, {"status": "failed"})
        plan["inputs"]["offset"]["sha256"]["status.json"] = candidate.digest(path)
    else:
        (root / "audit.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        candidate.checked_inputs(root, plan)


def test_two_full_fits_reload_formula_and_metadata_without_new_auc(chain, monkeypatch):
    root, plan, configs, _ = chain
    calls = []
    original_fit = models.fit_predict

    def traced(name, training, y, validation, validation_y, test, config, seed, path):
        calls.append((name, len(training), config.copy(), seed, validation_y))
        return original_fit(
            name, training, y, validation, validation_y, test, config, seed, path
        )

    monkeypatch.setattr(models, "fit_predict", traced)
    # Core pinning is tested separately; this manufactured chain exercises its API.
    monkeypatch.setattr(candidate, "pin_core", lambda *_: None)
    monkeypatch.setattr(
        experiment,
        "source_provenance",
        lambda *_: {
            "source_hashes": {},
            "git_revision": "toy-core",
            "git_dirty": False,
        },
    )
    metric = candidate.execute(root / "plan.json", root, root)
    assert [(name, rows, seed, valid_y) for name, rows, _, seed, valid_y in calls] == [
        ("offset", 60, 42, None),
        ("lightgbm", 60, 42, None),
    ]
    assert all(cfg["refit"] and cfg["fixed_training"] for _, _, cfg, _, _ in calls)
    assert metric["phase"] == "full_data_refit" and metric["roc_auc"] is None
    assert metric["fold_scores"] == [] and metric["rows"]["oof"] == 0
    assert metric["selection"]["development_roc_auc"] == 0.91
    assert metric["selection"]["source_dev_run"] == str(
        root / plan["inputs"]["selection"]["path"]
    )
    assert metric["validation"]["independent_holdout"] is False
    paths = [root / "runs" / plan["refit_names"][role] for role in candidate.ROLES[:2]]
    paths += [root / plan["inputs"]["external"]["path"]]
    parts = [pd.read_csv(path / "submission.csv").is_fraud.to_numpy() for path in paths]
    expected = sum(w * values for w, values in zip(candidate.WEIGHTS, parts))
    final = pd.read_csv(root / "runs/final-hybrid/submission.csv")
    np.testing.assert_allclose(final.is_fraud, expected, rtol=0, atol=1e-15)
    assert not list((root / "runs/final-hybrid").glob("oof*"))
    for path in paths[:2]:
        saved = candidate.read(path / "metrics.json")
        assert saved["checkpoint_reload"]["test_rows"] == 12
        assert saved["checkpoint_reload"]["max_absolute_error"] <= 1e-12
        assert saved["roc_auc"] is None and not (path / "oof.csv").exists()


def test_old_refit_still_requires_independent_holdout(chain):
    root, plan, _, _ = chain
    source = root / plan["inputs"]["offset"]["path"]
    report = root / "wrong-holdout.json"
    candidate.write(report, {"status": "complete", "kind": "exploratory_full_data_cv"})
    with pytest.raises(ValueError, match="independent holdout"):
        refit(source, report, root / "old-final", "invalid")


def test_pin_core_uses_real_clean_git_and_imports_only_pinned_package(tmp_path):
    package = tmp_path / "src/fraudlab"
    package.mkdir(parents=True)
    source = package / "__init__.py"
    source.write_text("MARKER = 'pinned-fixture'\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True
    ).strip()
    spec = {"revision": revision, "sha256": {"__init__.py": candidate.digest(source)}}
    code = "import importlib.util,json,sys; from pathlib import Path; s=importlib.util.spec_from_file_location('candidate',sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.pin_core(Path(sys.argv[2]),json.loads(sys.argv[3])); import fraudlab; assert fraudlab.MARKER=='pinned-fixture'"
    command = [
        sys.executable,
        "-c",
        code,
        str(candidate.ENTRYPOINT),
        str(tmp_path),
        json.dumps(spec),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    source.write_text("MARKER = 'changed'\n", encoding="utf-8")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and "Core must be clean" in result.stderr
