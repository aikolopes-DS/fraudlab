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
ASSETS = (
    "index.html", "styles.css", "app.js", "favicon.svg",
    "assets/payment-detail.jpg", "assets/payment-context.jpg",
    "assets/space-grotesk.woff2", "assets/dm-sans.woff2",
    "assets/space-grotesk-OFL.txt", "assets/dm-sans-OFL.txt",
    "assets/credits.json", "assets/font-sources.json",
)
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


def video_review_status(verification: dict) -> str:
    if verification.get("watched_in_full") is True and verification.get("audio_reviewed") is True:
        return "reviewed"
    if (
        verification.get("technical_checks") == "passed"
        and verification.get("whole_audio_asr") is True
        and verification.get("all_slides_visually_reviewed") is True
    ):
        return "checked"
    return "review"


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


def public_weights(value: object) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("effective_weights must be a model-to-weight object")
    allowed = {"glm", "offset", "realmlp", "lightgbm", "external_gpt2"}
    if set(value) - allowed or any(score(weight) is None for weight in value.values()):
        raise ValueError("Composition needs known model names and finite weights in [0, 1]")
    if value and not math.isclose(sum(value.values()), 1.0, abs_tol=1e-8):
        raise ValueError("Composition weights must sum to one")
    return {name: float(weight) for name, weight in value.items()}


def public_components(value: object) -> list:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("external_components must be a list")
    result = []
    text_fields = ("name", "kind", "source_type", "trained_by", "model_architecture", "training_code_commit", "code_license", "predictions_license", "mixture_space")
    for item in value:
        if not isinstance(item, dict) or item.get("source_type") != "external_predictions" or item.get("trained_by_us") is not False:
            raise ValueError("External prediction components must retain their external training attribution")
        component = {key: item.get(key) for key in text_fields if isinstance(item.get(key), str)}
        credits = item.get("method_credits") or []
        if not isinstance(credits, list) or any(not isinstance(credit, str) for credit in credits):
            raise ValueError("External method credits must be text entries")
        component.update(
            trained_by_us=False,
            training_executed_locally=item.get("training_executed_locally") is True,
            method_credits=credits,
            weight=score(item.get("weight")),
            predictions_source_version=positive_integer(item.get("predictions_source_version")),
        )
        for key in ("training_code_url", "predictions_source_url"):
            component[key] = public_link(item.get(key))
        result.append(component)
    return result


def normalize(evidence: dict, metrics: dict | None = None) -> dict:
    competition = evidence.get("competition") or {}
    checks = evidence.get("checks") or {}
    results = dict(evidence.get("results") or {})
    run = dict(evidence.get("run") or {})
    if metrics and checks.get("training_verified") is True:
        run.update({key: metrics.get(key) for key in ("model", "phase", "validation", "rows", "features")})
        if metrics.get("phase") != "full_data_refit":
            results.update(local_auc=metrics.get("roc_auc"), fold_scores=metrics.get("fold_scores", []), local_auc_scope="development_oof")
    training_verified = checks.get("training_verified") is True
    submission_verified = checks.get("submission_verified") is True
    is_refit = run.get("phase") == "full_data_refit"
    local_scope = results.get("local_auc_scope") or (None if is_refit else "development_oof")
    if is_refit and results.get("local_auc") is not None and local_scope != "development_oof":
        raise ValueError("A final refit's local AUC must be explicitly attributed to development_oof")
    holdout_verified = checks.get("holdout_verified") is True and training_verified and checks.get("data_verified") is True
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
    weights = public_weights(run.get("effective_weights"))
    components = public_components(evidence.get("external_components"))
    if weights.get("external_gpt2", 0) > 0:
        gpt2 = [item for item in components if item.get("kind") == "gpt2_target_first"]
        required = ("name", "trained_by", "predictions_license", "code_license", "predictions_source_url", "predictions_source_version")
        if len(gpt2) != 1 or any(not gpt2[0].get(key) for key in required):
            raise ValueError("A submitted GPT-2 weight needs its author, licenses and versioned prediction source")
        if gpt2[0]["weight"] is None or not math.isclose(gpt2[0]["weight"], weights["external_gpt2"], abs_tol=1e-8):
            raise ValueError("External attribution weight differs from the submitted composition")
    return {
        "competition": {key: competition.get(key) for key in ("title", "url", "deadline_utc")},
        "stage": evidence.get("stage", "implementation"),
        "updated_at": evidence.get("updated_at"),
        "checks": {**{key: checks.get(key) is True for key in ("data_verified", "training_verified", "submission_verified")}, "holdout_verified": holdout_verified},
        "results": {
            "local_auc": score(results.get("local_auc")) if training_verified else None,
            "local_auc_scope": local_scope if training_verified else None,
            "fold_scores": folds if training_verified else [],
            "holdout_auc": score(results.get("holdout_auc")) if holdout_verified else None,
            "public_auc": score(results.get("public_auc")) if submission_verified else None,
            "ranking": ranking,
            "total_teams": total_teams if ranking else None,
            "submission_id": results.get("submission_id") if submission_verified else None,
            "observed_at": results.get("observed_at") if submission_verified else None,
        },
        "links": {key: public_link((evidence.get("links") or {}).get(key)) for key in ("github", "kaggle", "notebook", "presentation", "video", "captions", "delivery")},
        "run": {**{key: run.get(key) for key in ("model", "phase", "validation", "rows", "features")}, "effective_weights": weights, "selection_reason": run.get("selection_reason") if isinstance(run.get("selection_reason"), str) else None},
        "external_components": components,
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


def build(evidence_path: Path, run_dir: Path | None = None, output_dir: Path | None = None, presentation_dir: Path | None = None, video_url: str | None = None) -> Path:
    output = (output_dir or SITE_DIR / "dist").resolve()
    if output.name != "dist" or not output.is_relative_to(SITE_DIR):
        raise ValueError("Build output must be a dist/ subdirectory inside site/")
    evidence = read_json(evidence_path)
    metrics = read_json(run_dir / "metrics.json") if run_dir else None
    public_evidence = normalize(evidence, metrics)
    if presentation_dir and presentation_dir.resolve().is_relative_to(output):
        raise ValueError("Presentation source cannot be inside the generated build output")
    external_video = public_link(video_url) if video_url is not None else None
    if video_url is not None and (
        not external_video
        or urlsplit(external_video).scheme != "https"
        or not urlsplit(external_video).path.lower().endswith(".mp4")
    ):
        raise ValueError("--video-url must be a public HTTPS MP4 URL")
    manifest, media = presentation_assets(presentation_dir) if presentation_dir else ({}, [])
    names = {source.name for source in media}
    if external_video and (
        manifest.get("status") != "evidence-backed"
        or manifest.get("video_generated") is not True
        or "video.mp4" not in names
    ):
        raise ValueError("An external video requires a generated evidence-backed MP4 verified by its local manifest")
    packaged_media = [source for source in media if not (external_video and source.name == "video.mp4")]
    # Only the validated, generated dist/ target is replaced. This removes stale
    # media and unexpected files from earlier builds before public packaging.
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ASSETS:
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SITE_DIR / name, destination)
    if presentation_dir:
        destination = output / "presentation"
        destination.mkdir()
        for source in packaged_media:
            shutil.copyfile(source, destination / source.name)
        public_manifest = {key: manifest.get(key) for key in ("status", "slides", "duration_seconds", "video_generated")}
        verification = manifest.get("verification") or {}
        verification_fields = ("rendered", "watched_in_full", "audio_reviewed", "human_listening_review", "technical_checks", "whole_audio_asr", "all_slides_visually_reviewed")
        public_manifest["verification"] = {key: verification.get(key) for key in verification_fields if key in verification}
        public_manifest["files"] = {source.name: manifest["files"][source.name] for source in packaged_media}
        if external_video:
            public_manifest["external_video"] = {"url": external_video, "sha256": manifest["files"]["video.mp4"]}
        (destination / "manifest.json").write_text(json.dumps(public_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        public_evidence["links"]["presentation"] = "presentation/slides.html"
        public_evidence["materials"] = {"presentation_status": manifest["status"], "video_status": None}
        if "video.mp4" in names and manifest["status"] == "evidence-backed" and manifest.get("video_generated") is True:
            public_evidence["links"]["video"] = external_video or "presentation/video.mp4"
            public_evidence["materials"]["video_status"] = video_review_status(verification)
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
    parser.add_argument("--video-url", help="Public HTTPS URL for the same MP4 hosted externally; its local manifest is still verified")
    args = parser.parse_args()
    output = build(args.evidence, args.run_dir, presentation_dir=args.presentation_dir, video_url=args.video_url)
    print(f"Static site built: {output}")


if __name__ == "__main__":
    main()
