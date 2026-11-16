"""Submission receipts require the exact official CSV, not a matching score alone."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from fraudlab.data import fingerprint


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "record_submission.py"
SPEC = importlib.util.spec_from_file_location("record_submission", SCRIPT)
recorder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recorder)


@pytest.fixture
def recording(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    payload = b"id,is_fraud\n10,0.2\n11,0.8\n"
    (run / "submission.csv").write_bytes(payload)
    digest = fingerprint(run / "submission.csv")
    metrics = {
        "status": "complete",
        "phase": "exploratory_full_data_cv",
        "roc_auc": 0.91,
        "fold_scores": [0.90, 0.92],
        "submission_sha256": digest,
        "model": "blend",
        "validation": {"method": "StratifiedKFold", "n_splits": 2},
        "rows": {"train": 4, "test": 2},
        "features": ["amount"],
    }
    (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    evidence_path = tmp_path / "public.json"
    evidence = {
        "competition": {"url": "https://www.kaggle.com/competitions/test-competition"},
        "results": {"observed_at": "2026-11-15T12:00:00+00:00", "public_auc": 0.89},
        "checks": {"submission_verified": False},
        "run": {
            "path": run.as_posix(),
            "submission_sha256": digest,
            "effective_weights": {"glm": 0.75, "external_gpt2": 0.25},
            "selection_reason": "Seleção anterior à consulta do score público.",
        },
    }
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False), encoding="utf-8"
    )
    receipt_path = run / "submission-receipt.json"
    receipt_path.write_text('{"existing": true}\n', encoding="utf-8")
    state = {
        "run": run,
        "metrics": metrics,
        "evidence": evidence,
        "evidence_path": evidence_path,
        "receipt_path": receipt_path,
        "payload": payload,
        "remote_payload": payload,
        "download_ref": "123",
        "calls": [],
    }

    def official_cli(command, **kwargs):
        assert command[0] == "fake-kaggle"
        assert kwargs["check"] is True
        arguments = command[1:]
        state["calls"].append(arguments)
        if arguments[:2] == ["competitions", "submissions"]:
            assert arguments[2] == "test-competition"
            output = json.dumps(
                [{"ref": 123, "status": "SubmissionStatus.COMPLETE", "publicScore": "0.93"}]
            )
        elif arguments[:2] == ["competitions", "submission-download"]:
            assert arguments[2] == "123"
            directory = Path(arguments[arguments.index("-p") + 1])
            (directory / f"{state['download_ref']}.raw").write_bytes(
                state["remote_payload"]
            )
            output = "Downloaded submission."
        elif arguments[:2] == ["competitions", "leaderboard"]:
            directory = Path(arguments[arguments.index("-p") + 1])
            with zipfile.ZipFile(directory / "test-competition.zip", "w") as archive:
                archive.writestr(
                    "leaderboard.csv",
                    "Rank,Score,TeamMemberUserNames\n1,0.95,other\n2,0.93,test-user\n",
                )
            output = "Downloaded leaderboard."
        else:
            raise AssertionError(f"Unexpected CLI request: {arguments}")
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(recorder.subprocess, "run", official_cli)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT), "--kaggle", "fake-kaggle", "--run", str(run),
            "--ref", "123", "--user", "test-user", "--evidence", str(evidence_path),
        ],
    )
    return state


@pytest.mark.parametrize("mismatch", ["bytes", "reference"])
def test_remote_mismatch_preserves_snapshot_and_receipt(recording, mismatch):
    if mismatch == "bytes":
        recording["remote_payload"] = b"id,is_fraud\n10,0.8\n11,0.2\n"
    else:
        recording["download_ref"] = "124"
    before = recording["evidence_path"].read_bytes()
    receipt_before = recording["receipt_path"].read_bytes()

    with pytest.raises(ValueError, match="requested|SHA-256"):
        recorder.main()

    assert recording["evidence_path"].read_bytes() == before
    assert recording["receipt_path"].read_bytes() == receipt_before
    assert not any(call[1] == "leaderboard" for call in recording["calls"])


def test_exact_official_file_records_score_and_preserves_same_run_metadata(recording):
    recorder.main()

    evidence = json.loads(recording["evidence_path"].read_text(encoding="utf-8"))
    receipt = json.loads(recording["receipt_path"].read_text(encoding="utf-8"))
    assert evidence["checks"]["submission_verified"] is True
    assert evidence["results"]["submission_id"] == "123"
    assert evidence["results"]["public_auc"] == 0.93
    assert evidence["results"]["ranking"] == 2
    assert evidence["results"]["observed_at"] != recording["evidence"]["results"]["observed_at"]
    assert evidence["results"]["observed_at"].endswith("+00:00")
    for key in ("effective_weights", "selection_reason"):
        assert evidence["run"][key] == recording["evidence"]["run"][key]
    assert receipt["submission_file_verified"] is True
    assert receipt["downloaded_submission_sha256"] == receipt["submission_sha256"]
    assert receipt["submission_sha256"] == recording["metrics"]["submission_sha256"]
    download = next(call for call in recording["calls"] if call[1] == "submission-download")
    assert download[2:4] == ["123", "-p"]
    assert not Path(download[4]).exists()


@pytest.mark.parametrize("difference", ["path", "sha"])
@pytest.mark.parametrize("explicit_weights", [None, "top_level", "model_details"])
def test_new_run_never_inherits_old_selection(recording, difference, explicit_weights):
    if difference == "path":
        recording["evidence"]["run"]["path"] = str(recording["run"].parent / "other")
    else:
        recording["evidence"]["run"]["submission_sha256"] = "0" * 64
    recording["evidence_path"].write_text(
        json.dumps(recording["evidence"]), encoding="utf-8"
    )
    weights = {"glm": 0.5, "lightgbm": 0.5}
    if explicit_weights:
        destination = recording["metrics"]
        if explicit_weights == "model_details":
            destination = destination.setdefault("model_details", {})
        destination["effective_weights"] = weights
        (recording["run"] / "metrics.json").write_text(
            json.dumps(recording["metrics"]), encoding="utf-8"
        )

    recorder.main()

    public_run = json.loads(recording["evidence_path"].read_text(encoding="utf-8"))["run"]
    assert "selection_reason" not in public_run
    if explicit_weights:
        assert public_run["effective_weights"] == weights
    else:
        assert "effective_weights" not in public_run
