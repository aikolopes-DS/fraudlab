"""Record a processed submission and a fresh, matching public leaderboard row."""

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile

from fraudlab.data import fingerprint, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kaggle", default="kaggle", help="Official Kaggle CLI 2.x executable"
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--ref", type=int, required=True)
    parser.add_argument(
        "--user", required=True, help="Public Kaggle username of a team member"
    )
    parser.add_argument("--holdout-report", type=Path)
    parser.add_argument("--evidence", type=Path, default=Path("evidence/public.json"))
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    competition = evidence["competition"]["url"].rstrip("/").split("/")[-1]
    metrics = json.loads((args.run / "metrics.json").read_text(encoding="utf-8"))
    if metrics.get("status") != "complete":
        raise ValueError("The local run is not complete.")
    digest = fingerprint(args.run / "submission.csv")
    if digest != metrics.get("submission_sha256"):
        raise ValueError("The submission file changed after the run was recorded.")
    development = metrics
    development_path = args.run
    if metrics.get("phase") == "full_data_refit":
        reference = metrics["selection"].get("source_dev_run") or metrics[
            "selection"
        ].get("source")
        if not reference:
            raise ValueError(
                "The final run is missing its frozen development reference."
            )
        development_path = Path(reference)
        development = json.loads(
            (development_path / "metrics.json").read_text(encoding="utf-8")
        )
        if (
            development.get("status") != "complete"
            or development.get("data_hashes") != metrics["data_hashes"]
        ):
            raise ValueError(
                "The frozen development reference does not match the final run."
            )
        if not math.isclose(
            development["roc_auc"],
            metrics["selection"]["development_roc_auc"],
            abs_tol=1e-12,
        ):
            raise ValueError("The development score changed after selection.")
    local_auc = development["roc_auc"]
    if not isinstance(local_auc, (int, float)) or not math.isfinite(local_auc):
        raise ValueError(
            "A finite, explicitly identified development OOF score is required."
        )
    holdout = None
    if args.holdout_report:
        if fingerprint(args.holdout_report) != metrics.get("selection", {}).get(
            "holdout_report_sha256"
        ):
            raise ValueError("The holdout report does not match the frozen final run.")
        holdout = json.loads(args.holdout_report.read_text(encoding="utf-8"))
        if (
            holdout.get("kind") != "independent_holdout"
            or holdout.get("status") != "complete"
        ):
            raise ValueError("A completed independent holdout report is required.")

    def kaggle(*arguments: str) -> str:
        result = subprocess.run(
            [args.kaggle, *arguments],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return result.stdout

    history = json.loads(
        kaggle(
            "competitions",
            "submissions",
            competition,
            "--page-size",
            "200",
            "--format",
            "json",
        )
    )
    submission = next((item for item in history if item["ref"] == args.ref), None)
    if (
        not submission
        or submission["status"] != "SubmissionStatus.COMPLETE"
        or not submission["publicScore"]
    ):
        raise ValueError(
            "The requested submission has not completed with a public score."
        )
    public_score = float(submission["publicScore"])
    if not math.isfinite(public_score) or not 0 <= public_score <= 1:
        raise ValueError("Unexpected ROC-AUC returned by Kaggle.")

    with tempfile.TemporaryDirectory(prefix="fraudlab-leaderboard-") as directory:
        kaggle(
            "competitions", "leaderboard", competition, "--download", "-p", directory
        )
        with zipfile.ZipFile(Path(directory) / f"{competition}.zip") as archive:
            names = [name for name in archive.namelist() if name.endswith(".csv")]
            if len(names) != 1:
                raise ValueError("Expected exactly one leaderboard CSV.")
            rows = list(
                csv.DictReader(io.StringIO(archive.read(names[0]).decode("utf-8-sig")))
            )
        own = [
            row
            for row in rows
            if args.user in re.split(r"[,;\s]+", row["TeamMemberUserNames"])
        ]
        if len(own) != 1 or not math.isclose(
            float(own[0]["Score"]), public_score, abs_tol=1e-8
        ):
            raise ValueError(
                "The fresh leaderboard does not yet match this user's submission score."
            )
    observed = datetime.now(timezone.utc).isoformat()
    evidence["results"].update(
        local_auc=local_auc,
        local_auc_scope="development_oof",
        local_auc_run=development_path.name,
        fold_scores=development["fold_scores"],
        public_auc=public_score,
        ranking=int(own[0]["Rank"]),
        total_teams=len(rows),
        submission_id=str(args.ref),
        observed_at=observed,
    )
    evidence["run"] = {
        "path": args.run.as_posix(),
        "phase": metrics.get("phase", "development"),
        "submission_sha256": digest,
        **{key: metrics[key] for key in ("model", "validation", "rows", "features")},
    }
    evidence["checks"].update(
        data_verified=True, training_verified=True, submission_verified=True
    )
    if holdout:
        evidence["results"].update(
            holdout_auc=holdout["roc_auc"], holdout_observed_at=holdout["evaluated_at"]
        )
        evidence["checks"]["holdout_verified"] = True
    evidence["updated_at"] = observed
    write_json(args.evidence, evidence)
    record = {
        "ref": args.ref,
        "run": args.run.name,
        "submission_sha256": digest,
        "public_auc": public_score,
        "ranking": int(own[0]["Rank"]),
        "total_teams": len(rows),
        "observed_at": observed,
        "source": evidence["competition"]["url"] + "/leaderboard",
    }
    write_json(args.run / "submission-receipt.json", record)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
