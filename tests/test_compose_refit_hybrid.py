"""Toy artifact chains verify inference-only composition, never a competition score."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

from fraudlab.data import fingerprint, write_json
from fraudlab.experiment import blend

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compose_refit_hybrid", ROOT / "scripts/compose_refit_hybrid.py"
)
composer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(composer)


@pytest.fixture
def chain(tmp_path):
    data = tmp_path / "data/raw"
    data.mkdir(parents=True)
    pd.DataFrame({"id": range(6), "amount": range(6), "is_fraud": [0, 1] * 3}).to_csv(
        data / "train.csv", index=False
    )
    pd.DataFrame({"id": [20, 10, 30], "amount": [1, 2, 3]}).to_csv(
        data / "test.csv", index=False
    )
    sample = pd.DataFrame({"id": [20, 10, 30], "is_fraud": 0})
    sample.to_csv(data / "sample_submission.csv", index=False)
    hashes = {
        name: fingerprint(data / name)
        for name in ("train.csv", "test.csv", "sample_submission.csv")
    }
    component = {
        "name": "GPT-2 target-first",
        "kind": "gpt2_target_first",
        "source_type": "external_predictions",
        "trained_by": "Masaya Kawamata",
        "predictions_source_version": 2,
        "predictions_license": "CC BY 4.0",
        "code_license": "MIT",
        "predictions_source_url": "https://example.com/versioned-predictions",
        "training_code_url": "https://example.com/training",
        "trained_by_us": False,
        "training_executed_locally": False,
    }
    names = {
        "refit": "final-blend",
        "glm": "final-glm",
        "lightgbm": "final-lgbm",
        "external": "explore5-public-gpt2-v2",
        "selection": "explore5-public25",
        "selected_own": "explore5-own",
    }
    recipes = {
        "refit": ["final-glm", "final-lgbm"],
        "selection": ["explore5-own", "explore5-public-gpt2-v2"],
        "selected_own": ["explore5-glm", "explore5-lgbm"],
    }
    values = {
        "glm": np.array([0.1, 0.6, 0.8]),
        "lightgbm": np.array([0.4, 0.5, 0.9]),
        "external": np.array([0.9, 0.3, 0.2]),
    }
    values["refit"] = 0.75 * values["glm"] + 0.25 * values["lightgbm"]
    plan = {
        "name": "toy-refit-hybrid",
        "data": "data/raw",
        "id_column": "id",
        "target": "is_fraud",
        "rows": {"train": 6, "test": 3},
        "data_hashes": hashes,
        "weights": [0.75, 0.25],
        "effective_weights": composer.EFFECTIVE_WEIGHTS,
        "development_roc_auc": 0.91,
        "external_release": component.copy(),
        "inputs": {},
    }
    run_metrics = {}
    for role, name in names.items():
        directory = tmp_path / "artifacts/runs" / name
        directory.mkdir(parents=True)
        refit = role in composer.REFITS
        metric = {
            "status": "complete",
            "phase": "full_data_refit" if refit else "exploratory_full_data_cv",
            "model": {
                "glm": "glm",
                "lightgbm": "lightgbm",
                "external": "external_gpt2",
            }.get(role, "blend"),
            "roc_auc": None if refit else 0.91,
            "fold_scores": [] if refit else [0.90, 0.92],
            "rows": {"train": 6, "test": 3, "oof": 0 if refit else 6},
            "data_hashes": hashes,
            "features": {"names": ["amount"], "excluded": ["id", "is_fraud"]},
            "git_revision": "historical-source",
            "git_dirty": True,
            "source_hashes": {"models.py": "a" * 64},
            "selection": {
                "source_dev_run": "historical-development",
                "holdout_report_sha256": "b" * 64,
            },
            "model_details": {},
        }
        config = {"model": metric["model"], "refit": refit}
        if role in recipes:
            details = {
                "method": "weighted_probability_average",
                "weights": [0.75, 0.25],
                "sources": ["artifacts\\runs\\" + source for source in recipes[role]],
            }
            metric["model_details"] = details.copy()
            config = details.copy()
        if role == "external":
            metric["model_details"]["external_components"] = [component.copy()]
        if role == "selection":
            metric["external_components"] = [
                {**component, "weight": 0.25, "mixture_space": "probability"}
            ]
        if role in values:
            prediction = sample.copy()
            prediction["is_fraud"] = values[role]
            for filename in ("predictions.csv", "submission.csv"):
                prediction.to_csv(directory / filename, index=False)
            metric["submission_sha256"] = fingerprint(directory / "submission.csv")
        if role in ("glm", "lightgbm"):
            (directory / "model.joblib").write_bytes(
                b"Toy checkpoint provenance; not an executable model."
            )
        write_json(directory / "config.json", config)
        write_json(directory / "status.json", {"status": "complete"})
        write_json(directory / "metrics.json", metric)
        run_metrics[role] = metric
        plan["inputs"][role] = {
            "path": directory.relative_to(tmp_path).as_posix(),
            "sha256": {file.name: fingerprint(file) for file in directory.iterdir()},
        }
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, plan)

    def repin(role):
        directory = tmp_path / plan["inputs"][role]["path"]
        plan["inputs"][role]["sha256"] = {
            file.name: fingerprint(file) for file in directory.iterdir()
        }
        write_json(plan_path, plan)

    def change_predictions(role, transform):
        directory = tmp_path / plan["inputs"][role]["path"]
        frame = transform(pd.read_csv(directory / "predictions.csv"))
        for filename in ("predictions.csv", "submission.csv"):
            frame.to_csv(directory / filename, index=False)
        run_metrics[role]["submission_sha256"] = fingerprint(
            directory / "submission.csv"
        )
        write_json(directory / "metrics.json", run_metrics[role])
        repin(role)

    return {
        "root": tmp_path,
        "plan": plan,
        "plan_path": plan_path,
        "values": values,
        "metrics": run_metrics,
        "repin": repin,
        "change_predictions": change_predictions,
        "output": tmp_path / "artifacts/runs/toy-refit-hybrid",
    }


def test_compose_preserves_effective_weights_and_has_no_new_auc_or_oof(chain):
    metrics = composer.compose(chain["plan_path"], chain["root"])
    result = pd.read_csv(chain["output"] / "submission.csv")
    expected = (
        0.5625 * chain["values"]["glm"]
        + 0.1875 * chain["values"]["lightgbm"]
        + 0.25 * chain["values"]["external"]
    )
    np.testing.assert_allclose(result.is_fraud, expected, rtol=0, atol=1e-15)
    assert result.id.tolist() == [20, 10, 30]
    assert metrics["phase"] == "full_data_refit" and metrics["roc_auc"] is None
    assert metrics["fold_scores"] == [] and metrics["rows"]["oof"] == 0
    assert not (chain["output"] / "oof.csv").exists()
    assert metrics["selection"]["development_roc_auc"] == 0.91
    assert (
        "not an AUC of this refit output"
        in metrics["selection"]["development_roc_auc_scope"]
    )
    assert metrics["effective_weights"] == {
        "glm": 0.5625,
        "lightgbm": 0.1875,
        "external_gpt2": 0.25,
    }
    assert metrics["external_components"][0]["trained_by_us"] is False
    assert metrics["external_components"][0]["predictions_source_version"] == 2
    assert metrics["external_components"][0]["weight"] == 0.25
    assert all(
        provenance["git_dirty"] is True
        for provenance in metrics["refit_provenance"].values()
    )
    assert metrics["composition_script_sha256"] == fingerprint(
        ROOT / "scripts/compose_refit_hybrid.py"
    )
    assert metrics["submission_sha256"] == fingerprint(
        chain["output"] / "submission.csv"
    )
    assert "model.joblib" in metrics["inputs"]["glm"]["sha256"]


@pytest.mark.parametrize("difference", ["order", "duplicate", "missing", "probability"])
def test_composition_rejects_bad_prediction_alignment_before_creating_output(
    chain, difference
):
    def transform(frame):
        if difference == "order":
            return frame.iloc[::-1]
        if difference == "missing":
            return frame.iloc[:-1]
        if difference == "duplicate":
            frame.loc[1, "id"] = frame.loc[0, "id"]
        else:
            frame.loc[0, "is_fraud"] = np.inf
        return frame

    chain["change_predictions"]("external", transform)
    with pytest.raises(ValueError, match="IDs|unique|probabilities"):
        composer.compose(chain["plan_path"], chain["root"])
    assert not chain["output"].exists()


def test_pinned_csv_hash_mismatch_is_rejected_before_use(chain):
    source = (
        chain["root"] / chain["plan"]["inputs"]["external"]["path"] / "predictions.csv"
    )
    source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Input SHA-256 mismatch"):
        composer.compose(chain["plan_path"], chain["root"])
    assert not chain["output"].exists()


@pytest.mark.parametrize(
    "role,phase",
    [("refit", "exploratory_full_data_cv"), ("external", "full_data_refit")],
)
def test_mislabelled_input_phase_is_rejected(chain, role, phase):
    chain["metrics"][role]["phase"] = phase
    write_json(
        chain["root"] / chain["plan"]["inputs"][role]["path"] / "metrics.json",
        chain["metrics"][role],
    )
    chain["repin"](role)
    with pytest.raises(ValueError, match="input phase"):
        composer.compose(chain["plan_path"], chain["root"])


def test_external_version_must_match_the_selected_release(chain):
    chain["metrics"]["external"]["model_details"]["external_components"][0][
        "predictions_source_version"
    ] = 3
    write_json(
        chain["root"] / chain["plan"]["inputs"]["external"]["path"] / "metrics.json",
        chain["metrics"]["external"],
    )
    chain["repin"]("external")
    with pytest.raises(ValueError, match="External release"):
        composer.compose(chain["plan_path"], chain["root"])


def test_script_has_one_fixed_effective_recipe(chain):
    chain["plan"]["effective_weights"] = {
        "glm": 0.6,
        "lightgbm": 0.15,
        "external_gpt2": 0.25,
    }
    write_json(chain["plan_path"], chain["plan"])
    with pytest.raises(ValueError, match="one declared refit recipe"):
        composer.compose(chain["plan_path"], chain["root"])


def test_general_blend_still_rejects_mixed_refit_and_cv_phases(chain):
    runs = [
        chain["root"] / chain["plan"]["inputs"][role]["path"]
        for role in ("refit", "external")
    ]
    with pytest.raises(
        ValueError, match="Cannot mix development OOF runs with full-data refits"
    ):
        blend(runs, [0.75, 0.25], chain["root"] / "runs", "forbidden")


def test_manifest_keeps_development_scope_through_receipt_and_site(chain, monkeypatch):
    metrics = composer.compose(chain["plan_path"], chain["root"])
    modules = {}
    for name, relative in (
        ("refit_receipt", "scripts/record_submission.py"),
        ("refit_site", "site/build.py"),
    ):
        spec = importlib.util.spec_from_file_location(name, ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    evidence_path = chain["root"] / "public.json"
    write_json(
        evidence_path,
        {
            "competition": {"url": "https://www.kaggle.com/competitions/toy"},
            "results": {},
            "checks": {},
        },
    )
    payload = (chain["output"] / "submission.csv").read_bytes()

    def official_cli(command, **kwargs):
        arguments = command[1:]
        if arguments[:2] == ["competitions", "submissions"]:
            output = json.dumps(
                [
                    {
                        "ref": 123,
                        "status": "SubmissionStatus.COMPLETE",
                        "publicScore": "0.93",
                    }
                ]
            )
        elif arguments[:2] == ["competitions", "submission-download"]:
            directory = Path(arguments[arguments.index("-p") + 1])
            (directory / "123.raw").write_bytes(payload)
            output = "Downloaded toy bytes."
        elif arguments[:2] == ["competitions", "leaderboard"]:
            directory = Path(arguments[arguments.index("-p") + 1])
            with zipfile.ZipFile(directory / "toy.zip", "w") as archive:
                archive.writestr(
                    "leaderboard.csv",
                    "Rank,Score,TeamMemberUserNames\n1,0.93,toy-user\n",
                )
            output = "Downloaded manufactured leaderboard."
        else:
            raise AssertionError(arguments)
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(modules["refit_receipt"].subprocess, "run", official_cli)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "record_submission",
            "--run",
            str(chain["output"]),
            "--ref",
            "123",
            "--user",
            "toy-user",
            "--evidence",
            str(evidence_path),
            "--kaggle",
            "toy-kaggle",
        ],
    )
    modules["refit_receipt"].main()
    recorded = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert recorded["results"]["local_auc"] == 0.91
    assert recorded["results"]["local_auc_scope"] == "development_oof"
    assert recorded["results"]["local_auc_run"] == "explore5-public25"
    assert recorded["run"]["rows"]["oof"] == 0
    assert recorded["run"]["effective_weights"] == composer.EFFECTIVE_WEIGHTS
    normalized = modules["refit_site"].normalize(recorded, metrics)
    assert normalized["results"]["local_auc"] == 0.91
    assert normalized["results"]["local_auc_scope"] == "development_oof"
    assert normalized["run"]["rows"]["oof"] == 0
    assert normalized["external_components"][0]["predictions_source_version"] == 2
