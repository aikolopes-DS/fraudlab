"""Recovery bookkeeping and arithmetic only; no CV fit or model deserialization."""

import importlib.util
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "resume_neural_cv", Path(__file__).parents[1] / "scripts/resume_neural_cv.py"
)
resume = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resume)


def test_execution_refuses_an_unpinned_entrypoint_before_core_import(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({}), encoding="utf-8")
    args = argparse.Namespace(
        source_run=tmp_path / "failed",
        source_worktree=tmp_path / "untrusted-core",
        plan=plan,
        output=tmp_path / "runs",
        recovery_evidence=None,
    )
    with pytest.raises(ValueError, match="entrypoint SHA-256"):
        resume.execute(args)
    assert not (tmp_path / "runs").exists()


def test_configuration_changes_only_name_and_rejects_other_protocols():
    original = {
        "name": "failed",
        "model": "neural_offset",
        "feature_set": "bpe_digits",
        "folds": 5,
        "seed": 42,
        "split": "stratified",
        "original_data": None,
        "holdout_fraction": 0,
        "device": "cpu",
        "threads": 4,
        "learning_rate": 0.123,
    }
    plan = {
        "original_run": "failed",
        "continuation_run": "continued",
        "recovery": {"reuse": [0, 1, 2], "fit": [3, 4]},
    }
    actual = resume.resumed_config(original, "continued", plan)
    assert actual == {**original, "name": "continued"}
    assert original["name"] == "failed"
    with pytest.raises(ValueError, match="CV5/42"):
        resume.resumed_config({**original, "seed": 43}, "continued", plan)
    with pytest.raises(ValueError, match="CPU/four"):
        resume.resumed_config({**original, "threads": 2}, "continued", plan)


def test_checkpoint_copy_is_exact_and_rejects_changed_or_existing_files(tmp_path):
    original, copied = tmp_path / "original.pt", tmp_path / "copied.pt"
    original.write_bytes(b"fabricated checkpoint bytes")
    expected = resume.digest(original)
    resume.copy_checkpoint(original, copied, expected)
    assert original.read_bytes() == copied.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        resume.copy_checkpoint(original, copied, expected)
    original.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        resume.copy_checkpoint(original, tmp_path / "other.pt", expected)


def test_source_requires_real_clean_pinned_git_worktree(tmp_path):
    package = tmp_path / "src/fraudlab"
    package.mkdir(parents=True)
    source = package / "__init__.py"
    source.write_text("# fabricated source\n", encoding="utf-8")
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
    assert resume.verify_source(tmp_path, revision) == {
        "__init__.py": resume.digest(source)
    }
    with pytest.raises(ValueError, match="revision"):
        resume.verify_source(tmp_path, "0" * 40)
    source.write_text("# modified\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean"):
        resume.verify_source(tmp_path, revision)


def test_only_pending_folds_fit_and_test_average_keeps_float32_division():
    labels = np.tile([0, 1], 5)
    splits = [
        (np.setdiff1d(np.arange(10), [2 * f, 2 * f + 1]), np.array([2 * f, 2 * f + 1]))
        for f in range(5)
    ]
    reuse_calls, fit_calls, completed = [], [], []
    test_predictions = [
        np.array([0.10000001 + f * 0.031, 0.33333334 + f * 0.07], dtype=np.float32)
        for f in range(5)
    ]

    def result(fold, training, validation, calls):
        calls.append(fold)
        assert not set(training).intersection(validation)
        return (
            np.array([0.1, 0.9], dtype=np.float32),
            test_predictions[fold],
            {"fold": fold},
        )

    actual = resume.collect_predictions(
        labels,
        2,
        splits,
        lambda *args: result(*args, reuse_calls),
        lambda *args: result(*args, fit_calls),
        lambda fold, score, details: completed.append((fold, score)),
    )
    assert reuse_calls == [0, 1, 2] and fit_calls == [3, 4]
    assert len(completed) == 5 and all(score == 1 for _, score in completed)
    np.testing.assert_array_equal(actual[1], np.repeat(np.arange(5), 2))
    expected = np.zeros(2)
    for prediction in test_predictions:
        expected += prediction / 5
    np.testing.assert_array_equal(actual[2], expected)
    wrong = sum(prediction.astype(np.float64) / 5 for prediction in test_predictions)
    assert not np.array_equal(actual[2], wrong)


def test_original_log_must_have_exactly_three_complete_folds():
    log = "\n".join(
        f"fold={fold} roc_auc=0.95000000 train=559968 validation=139992"
        for fold in range(3)
    )
    assert set(resume.logged_folds(log)) == {0, 1, 2}
    with pytest.raises(ValueError, match="exactly"):
        resume.logged_folds(
            log + "\nfold=3 roc_auc=0.95 train=559968 validation=139992"
        )
