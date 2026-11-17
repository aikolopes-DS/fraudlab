"""Tiny fabricated bundles verify packaging/provenance; they are not audio QA."""

import hashlib
from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path
import subprocess
import zipfile

import pytest

SPEC = importlib.util.spec_from_file_location(
    "package_delivery", Path(__file__).resolve().parents[1] / "scripts/package_delivery.py"
)
delivery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(delivery)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(delivery, "ROOT", source)
    subprocess.run(["git", "init", "--quiet", str(source)], check=True)
    subprocess.run(["git", "config", "user.name", "Package Test"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.email", "package-test@example.invalid"], cwd=source, check=True)
    subprocess.run(["git", "config", "core.autocrlf", "false"], cwd=source, check=True)
    (source / "README.md").write_bytes(b"Committed source\n")
    (source / "docs").mkdir()
    (source / "docs/reproduce.md").write_text("Fabricated test fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "Manufactured source"], cwd=source, check=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    media = source / "artifacts/media"
    media.mkdir(parents=True)
    snapshot = source / "artifacts/snapshot/evidence.json"
    write_json(snapshot, {"results": {"submission_id": "123", "public_auc": 0.9,
                                     "ranking": 12, "total_teams": 100,
                                     "observed_at": "2026-01-01T00:00:00Z"}})
    script = {"slides": [{"chapter": "A < B", "narration": "One"},
                         {"chapter": "Second", "narration": "Two"}]}
    write_json(media / "script.json", script)
    write_json(media / "timeline.json", [{"slide": 1, "start": 0, "duration": 1},
                                         {"slide": 2, "start": 1, "duration": 1}])
    (media / "slides.html").write_text("<!doctype html><title>Two slides</title>", encoding="utf-8")
    (media / "video.mp4").write_bytes(b"fabricated bytes, not a real video")
    for name in ("captions.srt", "captions.vtt"):
        (media / name).write_text("Fabricated caption fixture\n", encoding="utf-8")
    video_sha = delivery.digest((media / "video.mp4").read_bytes())
    review = {"video_sha256": video_sha, "slides": 2, "technical_checks": "passed",
              "full_av_decode": {"exit_code": 0, "error_output": "", "scope": "entire video and audio streams"},
              "asr": {"input_media_sha256": video_sha, "whole_input_processed": True},
              "visually_reviewed_slides": [1, 2]}
    proofs = {"review.json": review,
              "asr.json": {"input_media": "video.mp4", "input_media_sha256": video_sha,
                           "whole_input_processed": True, "segments": [{"text": "One Two"}]},
              "visual-review.json": {"status": "passed", "reviewed_slides": [1, 2]},
              "asr-by-slide.json": {"slides": [{"slide": 1, "asr_text": "One"},
                                                {"slide": 2, "asr_text": "Two"}]}}
    verification = {"technical_checks": "passed", "whole_audio_asr": True,
                    "all_slides_visually_reviewed": True, "visual_reviewed_slides": [1, 2]}
    for name, value in proofs.items():
        write_json(media / "review" / name, value)
    for key, name in (("review_report_sha256", "review.json"), ("asr_report_sha256", "asr.json"),
                      ("visual_report_sha256", "visual-review.json"), ("asr_alignment_sha256", "asr-by-slide.json")):
        verification[key] = delivery.digest((media / "review" / name).read_bytes())
    manifest = {"status": "evidence-backed", "video_generated": True, "slides": 2,
                "duration_seconds": 2, "verification": verification,
                "inputs": {snapshot.relative_to(source).as_posix(): delivery.digest(snapshot.read_bytes())},
                "files": {path.name: delivery.digest(path.read_bytes()) for path in media.iterdir() if path.is_file()}}
    write_json(media / "manifest.json", manifest)
    return source, media, snapshot, revision, tmp_path / "delivery.zip"


def test_fabricated_delivery_has_verified_hashes_immutable_source_and_working_links(bundle):
    source, media, snapshot, revision, output = bundle
    (source / "README.md").write_text("Uncommitted changes must stay out\n", encoding="utf-8")
    result = delivery.package(media, snapshot, revision, output)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("MANIFEST.json"))
        assert manifest["source_commit"] == revision and manifest["presentation_slides"] == 2
        assert archive.read("codigo/README.md") == b"Committed source\n"
        assert archive.read("apresentacao/video.srt") == archive.read("apresentacao/captions.srt")
        assert set(manifest["files"]) == set(archive.namelist()) - {"MANIFEST.json"}
        for name, expected in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
        class Links(HTMLParser):
            links = []
            def handle_starttag(self, tag, attributes):
                self.links.extend(value for key, value in attributes if key == "href")
        document = archive.read("ENTREGA.html").decode("utf-8")
        parsed = Links()
        parsed.feed(document)
        assert parsed.links and all(link in archive.namelist() for link in parsed.links)
        assert "A &lt; B" in document
    assert result["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert output.with_suffix(".zip.sha256").read_text().split()[0] == result["sha256"]
    with pytest.raises(FileExistsError):
        delivery.package(media, snapshot, revision, output)


@pytest.mark.parametrize("tampering", ["video", "review_bytes", "other_video_review", "other_video_asr", "visual_gap", "snapshot_binding"])
def test_delivery_rejects_tampered_or_unrelated_proof_bytes(bundle, tampering):
    _, media, snapshot, revision, output = bundle
    manifest = delivery.read(media / "manifest.json")
    if tampering == "video":
        (media / "video.mp4").write_bytes(b"changed video")
    elif tampering == "review_bytes":
        (media / "review/review.json").write_bytes(b"{}")
    elif tampering in ("other_video_review", "other_video_asr"):
        name, key, hash_key = (("review.json", "video_sha256", "review_report_sha256")
                               if tampering == "other_video_review" else
                               ("asr.json", "input_media_sha256", "asr_report_sha256"))
        path = media / "review" / name
        proof = delivery.read(path)
        proof[key] = "0" * 64
        write_json(path, proof)
        manifest["verification"][hash_key] = delivery.digest(path.read_bytes())
    elif tampering == "visual_gap":
        manifest["verification"]["visual_reviewed_slides"] = [1]
    else:
        expected = manifest["inputs"].pop(next(iter(manifest["inputs"])))
        manifest["inputs"]["some-other-input.json"] = expected
    write_json(media / "manifest.json", manifest)
    with pytest.raises(ValueError):
        delivery.package(media, snapshot, revision, output)
    assert not output.exists() and not output.with_suffix(".zip.tmp").exists()


@pytest.mark.parametrize("private_path", ["data/raw/train.csv", "nested/data/rows.csv", ".env.production",
                                         "config/.kaggle/kaggle.json", "private-key.pem"])
def test_delivery_refuses_tracked_data_or_credentials(bundle, private_path):
    source, media, snapshot, _, output = bundle
    private = source / private_path
    private.parent.mkdir(parents=True, exist_ok=True)
    private.write_text("must not be packaged\n", encoding="utf-8")
    subprocess.run(["git", "add", private_path], cwd=source, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "Manufactured unsafe tracked file"], cwd=source, check=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    with pytest.raises(ValueError, match="Private data, credentials"):
        delivery.package(media, snapshot, revision, output)
    assert not output.exists()
