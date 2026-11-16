"""Build the public case study from explicitly verified evidence.

Only allowlisted evidence fields and presentation files enter ``dist``.
Training datasets, predictions, model weights, and local paths stay local.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from urllib.parse import urlsplit

SITE_DIR = Path(__file__).resolve().parent
ROOT = SITE_DIR.parent
ASSETS = ("index.html", "styles.css", "app.js", "favicon.svg")
PRESENTATION_ASSET = re.compile(r"(?:slides\.html|script\.json|captions\.(?:srt|vtt)|video\.mp4|(?:slide|formula)-\d+\.png)")


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def score(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and 0 <= value <= 1 else None


def positive_integer(value: object) -> int | None:
    return value if type(value) is int and value > 0 else None


def public_link(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    if "\\" in value or any(ord(char) < 32 for char in value):
        raise ValueError("Public links cannot contain backslashes or control characters")
    link = urlsplit(value)
    if link.username or link.password or value.startswith("//"):
        raise ValueError("Public links cannot contain credentials or protocol-relative URLs")
    if link.scheme:
        if link.scheme != "https" or not link.netloc:
            raise ValueError("External public links must use HTTPS")
    elif ".." in Path(link.path).parts:
        raise ValueError("Public file links cannot traverse parent directories")
    return value


def normalize(evidence: dict, metrics: dict | None = None) -> dict:
    competition = evidence.get("competition") or {}
    checks = evidence.get("checks") or {}
    results = dict(evidence.get("results") or {})
    run = dict(evidence.get("run") or {})
    if metrics and checks.get("training_verified") is True:
        results.update(local_auc=metrics.get("roc_auc"), fold_scores=metrics.get("fold_scores", []))
        run.update({key: metrics.get(key) for key in ("model", "validation", "rows", "features")})
    training_verified = checks.get("training_verified") is True
    submission_verified = checks.get("submission_verified") is True
    ranking = positive_integer(results.get("ranking")) if submission_verified and results.get("observed_at") else None
    total_teams = positive_integer(results.get("total_teams"))
    if ranking and total_teams and ranking > total_teams:
        raise ValueError("Observed ranking cannot exceed total teams")
    folds = results.get("fold_scores") or []
    if not isinstance(folds, list) or any(score(item) is None for item in folds):
        raise ValueError("fold_scores must be a list of finite numbers between 0 and 1")
    limitations = evidence.get("limitations") or []
    if not isinstance(limitations, list) or any(not isinstance(item, str) for item in limitations):
        raise ValueError("limitations must be a list of text entries")
    return {
        "competition": {key: competition.get(key) for key in ("title", "url", "deadline_utc")},
        "stage": evidence.get("stage", "implementation"),
        "updated_at": evidence.get("updated_at"),
        "checks": {key: checks.get(key) is True for key in ("data_verified", "training_verified", "submission_verified")},
        "results": {
            "local_auc": score(results.get("local_auc")) if training_verified else None,
            "fold_scores": folds if training_verified else [],
            "public_auc": score(results.get("public_auc")) if submission_verified else None,
            "ranking": ranking,
            "total_teams": total_teams if ranking else None,
            "submission_id": results.get("submission_id") if submission_verified else None,
            "observed_at": results.get("observed_at") if submission_verified else None,
        },
        "links": {key: public_link((evidence.get("links") or {}).get(key)) for key in ("github", "kaggle", "presentation", "video", "captions")},
        "run": {key: run.get(key) for key in ("model", "validation", "rows", "features")},
        "limitations": limitations,
    }


def presentation_assets(directory: Path) -> tuple[dict, list[Path]]:
    manifest = read_json(directory / "manifest.json")
    if manifest.get("status") not in ("draft", "evidence-backed"):
        raise ValueError("Presentation manifest must identify a draft or evidence-backed version")
    files = manifest.get("files")
    if not isinstance(files, dict) or "slides.html" not in files:
        raise ValueError("Presentation manifest must include slides.html and file hashes")
    assets = []
    for name, expected_hash in files.items():
        if not PRESENTATION_ASSET.fullmatch(name):
            continue
        source = (directory / name).resolve()
        if source.parent != directory.resolve() or not source.is_file():
            raise ValueError(f"Presentation asset is missing or outside its directory: {name}")
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            raise ValueError(f"Presentation asset differs from its manifest: {name}")
        assets.append(source)
    names = {path.name for path in assets}
    slides_html = (directory / "slides.html").read_text(encoding="utf-8")
    references = re.findall(r'(?:src|href)=[\"\']([^\"\']+)[\"\']', slides_html)
    for reference in references:
        parsed = urlsplit(reference)
        if not parsed.scheme and not reference.startswith("#") and parsed.path not in names:
            raise ValueError(f"Presentation references an asset absent from its manifest: {reference}")
    return manifest, assets


def build(evidence_path: Path, run_dir: Path | None = None, output_dir: Path | None = None, presentation_dir: Path | None = None) -> Path:
    output = (output_dir or SITE_DIR / "dist").resolve()
    if output.name != "dist" or not output.is_relative_to(SITE_DIR):
        raise ValueError("Build output must be a dist/ subdirectory inside site/")
    evidence = read_json(evidence_path)
    metrics = read_json(run_dir / "metrics.json") if run_dir else None
    public_evidence = normalize(evidence, metrics)
    if presentation_dir and presentation_dir.resolve().is_relative_to(output):
        raise ValueError("Presentation source cannot be inside the generated build output")
    manifest, media = presentation_assets(presentation_dir) if presentation_dir else ({}, [])
    # Only the validated, generated dist/ target is replaced. This removes stale
    # media and unexpected files from earlier builds before public packaging.
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ASSETS:
        shutil.copyfile(SITE_DIR / name, output / name)
    if presentation_dir:
        destination = output / "presentation"
        destination.mkdir()
        for source in media:
            shutil.copyfile(source, destination / source.name)
        public_manifest = {key: manifest.get(key) for key in ("status", "slides", "duration_seconds", "video_generated", "verification")}
        public_manifest["files"] = {source.name: manifest["files"][source.name] for source in media}
        (destination / "manifest.json").write_text(json.dumps(public_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        public_evidence["links"]["presentation"] = "presentation/slides.html"
        public_evidence["materials"] = {"presentation_status": manifest["status"], "video_status": None}
        names = {source.name for source in media}
        if "video.mp4" in names and manifest["status"] == "evidence-backed" and manifest.get("video_generated") is True:
            public_evidence["links"]["video"] = "presentation/video.mp4"
            verification = manifest.get("verification") or {}
            public_evidence["materials"]["video_status"] = "reviewed" if verification.get("watched_in_full") is True and verification.get("audio_reviewed") is True else "review"
            if "captions.vtt" in names:
                public_evidence["links"]["captions"] = "presentation/captions.vtt"
        elif manifest["status"] == "draft":
            public_evidence["links"]["video"] = None
            public_evidence["links"]["captions"] = None
    for value in public_evidence["links"].values():
        if not value or urlsplit(value).scheme:
            continue
        target = (output / urlsplit(value).path.lstrip("/")).resolve()
        if not target.is_relative_to(output) or not target.is_file():
            raise ValueError(f"Public local link has no packaged file: {value}")
    (output / "evidence.json").write_text(json.dumps(public_evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=ROOT / "evidence" / "public.json")
    parser.add_argument("--run-dir", type=Path, help="Verified run whose metrics will be displayed")
    parser.add_argument("--presentation-dir", type=Path, help="Generated presentation with a verified file manifest")
    args = parser.parse_args()
    output = build(args.evidence, args.run_dir, presentation_dir=args.presentation_dir)
    print(f"Static site built: {output}")


if __name__ == "__main__":
    main()
