"""Public-evidence and asset-boundary tests; no competition data is used."""

import importlib.util
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("site_build", Path(__file__).with_name("build.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PublicBuildTests(unittest.TestCase):
    def external_fixture(self):
        return {
            "run": {"effective_weights": {"glm": 0.5625, "lightgbm": 0.1875, "external_gpt2": 0.25}},
            "external_components": [{
                "name": "GPT-2 target-first", "kind": "gpt2_target_first", "source_type": "external_predictions",
                "trained_by": "Masaya Kawamata", "trained_by_us": False, "training_executed_locally": False,
                "predictions_license": "CC BY 4.0", "code_license": "MIT", "predictions_source_version": 2,
                "predictions_source_url": "https://www.kaggle.com/datasets/masayakawamata/s6e11-gpt2-target-first-oof",
                "method_credits": ["Chris Deotte", "Borisov et al.", "Udit Jain"], "weight": 0.25,
            }],
        }

    def test_public_composition_preserves_credit_and_excludes_private_payloads(self):
        evidence = self.external_fixture()
        evidence["run"]["path"] = "C:/private/run"
        evidence["external_components"][0].update(path="C:/private/gpt2", logits=[1, 2], token="fixture-secret")
        public = MODULE.normalize(evidence)
        self.assertEqual(public["run"]["effective_weights"], evidence["run"]["effective_weights"])
        component = public["external_components"][0]
        self.assertEqual(component["trained_by"], "Masaya Kawamata")
        self.assertFalse(component["trained_by_us"])
        self.assertEqual(component["predictions_license"], "CC BY 4.0")
        self.assertEqual(component["predictions_source_version"], 2)
        serialized = json.dumps(public)
        self.assertNotIn("C:/private", serialized)
        self.assertNotIn("fixture-secret", serialized)
        self.assertNotIn("logits", serialized)

    def test_composition_rejects_invalid_weights_or_missing_external_credit(self):
        for weights in ({"glm": True}, {"glm": float("nan")}, {"glm": 1, "lightgbm": 1}, {"unknown": 1}):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                MODULE.public_weights(weights)
        evidence = self.external_fixture()
        evidence["external_components"] = []
        with self.assertRaisesRegex(ValueError, "author"):
            MODULE.normalize(evidence)
        evidence = self.external_fixture()
        evidence["external_components"][0]["trained_by_us"] = True
        with self.assertRaisesRegex(ValueError, "external training attribution"):
            MODULE.normalize(evidence)

    def test_residual_glm_retains_its_own_family_and_external_credit(self):
        evidence = self.external_fixture()
        evidence["run"]["effective_weights"] = {"offset": 0.5625, "lightgbm": 0.1875, "external_gpt2": 0.25}
        public = MODULE.normalize(evidence)
        self.assertEqual(public["run"]["effective_weights"]["offset"], 0.5625)
        self.assertNotIn("glm", public["run"]["effective_weights"])
        self.assertEqual(public["external_components"][0]["weight"], 0.25)

    def test_loading_metrics_does_not_discard_canonical_composition(self):
        evidence = self.external_fixture()
        evidence["checks"] = {"training_verified": True}
        public = MODULE.normalize(evidence, {"model": "blend", "phase": "exploratory_full_data_cv", "roc_auc": 0.91})
        self.assertEqual(public["run"]["effective_weights"]["external_gpt2"], 0.25)
        self.assertFalse(public["external_components"][0]["trained_by_us"])

    def test_technical_asr_and_slide_checks_do_not_claim_full_listening(self):
        verification = {"technical_checks": "passed", "whole_audio_asr": True, "all_slides_visually_reviewed": True, "watched_in_full": False, "audio_reviewed": False, "human_listening_review": False}
        self.assertEqual(MODULE.video_review_status(verification), "checked")
        for field in ("technical_checks", "whole_audio_asr", "all_slides_visually_reviewed"):
            incomplete = {**verification, field: False}
            self.assertEqual(MODULE.video_review_status(incomplete), "review")
        self.assertEqual(MODULE.video_review_status({"watched_in_full": True, "audio_reviewed": True}), "reviewed")

    def test_build_keeps_checked_video_distinct_from_human_review(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent, prefix=".test-build-") as directory:
            path = Path(directory)
            evidence = path / "source.json"
            evidence.write_text("{}", encoding="utf-8")
            media = path / "presentation"
            media.mkdir()
            for name, value in {"slides.html": "<h1>Contract fixture</h1>", "video.mp4": "contract fixture, not playable media"}.items():
                (media / name).write_text(value, encoding="utf-8")
            manifest = {"status": "evidence-backed", "video_generated": True, "verification": {"technical_checks": "passed", "whole_audio_asr": True, "all_slides_visually_reviewed": True, "watched_in_full": False, "audio_reviewed": False, "human_listening_review": False, "private_report_path": "C:/private/report.json"}, "files": {item.name: hashlib.sha256(item.read_bytes()).hexdigest() for item in media.iterdir()}}
            (media / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            output = MODULE.build(evidence, output_dir=path / "dist", presentation_dir=media)
            public = MODULE.read_json(output / "evidence.json")
            self.assertEqual(public["materials"]["video_status"], "checked")
            self.assertEqual(public["links"]["video"], "presentation/video.mp4")
            self.assertTrue((output / "presentation" / "video.mp4").is_file())
            packaged = MODULE.read_json(output / "presentation" / "manifest.json")
            self.assertFalse(packaged["verification"]["audio_reviewed"])
            self.assertFalse(packaged["verification"]["watched_in_full"])
            self.assertNotIn("C:/private", json.dumps(packaged))

    def test_external_video_preserves_hash_check_captions_and_review_status(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent, prefix=".test-build-") as directory:
            path = Path(directory)
            evidence = path / "source.json"
            evidence.write_text("{}", encoding="utf-8")
            media = path / "presentation"
            media.mkdir()
            for name, value in {
                "slides.html": "<h1>Contract fixture</h1>",
                "video.mp4": "contract fixture, not playable media",
                "captions.vtt": "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nExemplo.\n",
            }.items():
                (media / name).write_text(value, encoding="utf-8")
            manifest = {
                "status": "evidence-backed", "video_generated": True,
                "verification": {"technical_checks": "passed", "whole_audio_asr": True, "all_slides_visually_reviewed": True},
                "files": {item.name: hashlib.sha256(item.read_bytes()).hexdigest() for item in media.iterdir()},
            }
            (media / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            url = "https://github.com/example/repository/releases/download/v1/video.mp4"
            output = MODULE.build(evidence, output_dir=path / "dist", presentation_dir=media, video_url=url)
            public = MODULE.read_json(output / "evidence.json")
            packaged = MODULE.read_json(output / "presentation" / "manifest.json")
            self.assertEqual(public["links"]["video"], url)
            self.assertEqual(public["links"]["captions"], "presentation/captions.vtt")
            self.assertEqual(public["materials"]["video_status"], "checked")
            self.assertTrue((output / "presentation" / "captions.vtt").is_file())
            self.assertTrue((output / "presentation" / "slides.html").is_file())
            self.assertFalse((output / "presentation" / "video.mp4").exists())
            self.assertNotIn("video.mp4", packaged["files"])
            self.assertEqual(packaged["external_video"], {"url": url, "sha256": manifest["files"]["video.mp4"]})
            (media / "video.mp4").write_bytes(b"changed local bytes")
            with self.assertRaisesRegex(ValueError, "differs from its manifest"):
                MODULE.build(evidence, output_dir=path / "dist", presentation_dir=media, video_url=url)

    def test_external_video_requires_https_mp4_and_verified_local_source(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent, prefix=".test-build-") as directory:
            path = Path(directory)
            evidence = path / "source.json"
            evidence.write_text("{}", encoding="utf-8")
            for url in ("http://example.com/video.mp4", "presentation/video.mp4", "https://example.com/video.html", ""):
                with self.subTest(url=url), self.assertRaises(ValueError):
                    MODULE.build(evidence, output_dir=path / "dist", video_url=url)
            with self.assertRaisesRegex(ValueError, "verified by its local manifest"):
                MODULE.build(evidence, output_dir=path / "dist", video_url="https://example.com/video.mp4")

    def test_unverified_scores_are_not_published(self):
        public = MODULE.normalize({"results": {"local_auc": 0.99, "public_auc": 0.98, "ranking": 1, "observed_at": "2026-11-16T18:00:00Z"}})
        self.assertIsNone(public["results"]["local_auc"])
        self.assertIsNone(public["results"]["public_auc"])
        self.assertIsNone(public["results"]["ranking"])

    def test_local_metrics_do_not_prove_a_submission(self):
        public = MODULE.normalize({"checks": {"training_verified": True}}, {"roc_auc": 0.92, "fold_scores": [0.91, 0.93], "model": "reference"})
        self.assertEqual(public["results"]["local_auc"], 0.92)
        self.assertIsNone(public["results"]["public_auc"])
        self.assertFalse(public["checks"]["submission_verified"])

    def test_final_refit_preserves_development_oof_without_inventing_a_new_score(self):
        evidence = {"checks": {"data_verified": True, "training_verified": True, "holdout_verified": True}, "results": {"local_auc": 0.91, "local_auc_scope": "development_oof", "fold_scores": [0.90, 0.92], "holdout_auc": 0.905}}
        final = {"phase": "full_data_refit", "model": "blend", "roc_auc": None, "fold_scores": [], "rows": {"oof": 0}, "selection": {"development_roc_auc": 0.91}}
        public = MODULE.normalize(evidence, final)
        self.assertEqual(public["results"]["local_auc"], 0.91)
        self.assertEqual(public["results"]["fold_scores"], [0.90, 0.92])
        self.assertEqual(public["results"]["local_auc_scope"], "development_oof")
        self.assertEqual(public["run"]["phase"], "full_data_refit")
        self.assertEqual(public["run"]["rows"]["oof"], 0)
        self.assertEqual(public["results"]["holdout_auc"], 0.905)
        self.assertIsNone(public["results"]["public_auc"])

    def test_final_refit_requires_explicit_development_scope(self):
        with self.assertRaisesRegex(ValueError, "development_oof"):
            MODULE.normalize({"checks": {"training_verified": True}, "results": {"local_auc": 0.91}}, {"phase": "full_data_refit", "roc_auc": None})

    def test_holdout_needs_its_own_verification_and_verified_training_data(self):
        evidence = {"checks": {"data_verified": True, "training_verified": True}, "results": {"holdout_auc": 0.91}}
        self.assertIsNone(MODULE.normalize(evidence)["results"]["holdout_auc"])
        evidence["checks"]["holdout_verified"] = True
        evidence["checks"]["data_verified"] = False
        self.assertIsNone(MODULE.normalize(evidence)["results"]["holdout_auc"])

    def test_utf8_evidence_roundtrip_preserves_portuguese_text(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent, prefix=".test-build-") as directory:
            path = Path(directory)
            source = {"competition": {"title": "Detecção de fraude · apresentação"}, "limitations": ["Seleção e avaliação independentes; posição provisória."]}
            evidence = path / "source.json"
            evidence.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")
            output = MODULE.build(evidence, output_dir=path / "dist")
            public = MODULE.read_json(output / "evidence.json")
            self.assertEqual(public["competition"]["title"], source["competition"]["title"])
            self.assertEqual(public["limitations"], source["limitations"])

    def test_repository_evidence_has_no_known_encoding_corruption(self):
        evidence = MODULE.read_json(MODULE.ROOT / "evidence" / "public.json")
        serialized = json.dumps(evidence, ensure_ascii=False)
        self.assertIsNone(re.search(r"Ã[\u0080-\u00ff\u0192\u2010-\u20ff]|Â[\u0080-\u00ff]|�", serialized), "Evidence contains a mojibake or replacement-character sequence")

    def test_private_fields_and_local_paths_are_excluded(self):
        public = MODULE.normalize({"token": "not-a-real-secret", "run": {"path": "C:/private/run", "model": "reference"}, "raw_rows": [1, 2]})
        serialized = json.dumps(public)
        self.assertNotIn("not-a-real-secret", serialized)
        self.assertNotIn("C:/private", serialized)
        self.assertNotIn("raw_rows", serialized)

    def test_unsafe_links_are_rejected(self):
        for link in ("javascript:alert(1)", "//example.com", "https://user:password@example.com", "../data/train.csv", "http://example.com", "C:\\private\\file"):
            with self.subTest(link=link), self.assertRaises(ValueError):
                MODULE.public_link(link)

    def test_rank_requires_observation_and_cannot_exceed_teams(self):
        public = MODULE.normalize({"checks": {"submission_verified": True}, "results": {"ranking": 2}})
        self.assertIsNone(public["results"]["ranking"])
        with self.assertRaises(ValueError):
            MODULE.normalize({"checks": {"submission_verified": True}, "results": {"ranking": 8, "total_teams": 4, "observed_at": "2026-11-16T18:00:00Z"}})

    def test_non_finite_scores_do_not_reach_json(self):
        public = MODULE.normalize({"checks": {"training_verified": True}, "results": {"local_auc": float("nan")}})
        self.assertIsNone(public["results"]["local_auc"])
        with self.assertRaises(ValueError):
            MODULE.normalize({"results": {"fold_scores": [float("inf")]}})

    def test_build_copies_only_the_allowlisted_assets(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent, prefix=".test-build-") as directory:
            path = Path(directory)
            evidence = path / "source.json"
            evidence.write_text('{"stage":"implementation"}', encoding="utf-8")
            output = MODULE.build(evidence, output_dir=path / "dist")
            self.assertEqual({item.name for item in output.iterdir()}, set(MODULE.ASSETS) | {"evidence.json"})

    def test_rebuild_removes_stale_files_inside_validated_dist(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent, prefix=".test-build-") as directory:
            path = Path(directory)
            evidence = path / "source.json"
            evidence.write_text("{}", encoding="utf-8")
            output = MODULE.build(evidence, output_dir=path / "dist")
            (output / "stale-unexpected-file.csv").write_text("private fixture", encoding="utf-8")
            MODULE.build(evidence, output_dir=output)
            self.assertFalse((output / "stale-unexpected-file.csv").exists())
            self.assertTrue(evidence.exists())

    def test_presentation_hashes_and_draft_label_are_preserved(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent, prefix=".test-build-") as directory:
            path = Path(directory)
            evidence = path / "source.json"
            evidence.write_text("{}", encoding="utf-8")
            media = path / "presentation"
            media.mkdir()
            slides = media / "slides.html"
            slides.write_text("<h1>Test fixture</h1>", encoding="utf-8")
            manifest = {"status": "draft", "files": {"slides.html": hashlib.sha256(slides.read_bytes()).hexdigest()}, "inputs": {"private_path": "C:/private"}}
            (media / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            output = MODULE.build(evidence, output_dir=path / "dist", presentation_dir=media)
            public = MODULE.read_json(output / "evidence.json")
            self.assertEqual(public["materials"]["presentation_status"], "draft")
            self.assertEqual(public["links"]["presentation"], "presentation/slides.html")
            self.assertNotIn("private_path", (output / "presentation" / "manifest.json").read_text(encoding="utf-8"))
            slides.write_text("changed after manifest", encoding="utf-8")
            with self.assertRaises(ValueError):
                MODULE.build(evidence, output_dir=path / "dist", presentation_dir=media)

    def test_missing_local_media_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent, prefix=".test-build-") as directory:
            path = Path(directory)
            evidence = path / "source.json"
            evidence.write_text(json.dumps({"links": {"video": "presentation/missing.mp4"}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                MODULE.build(evidence, output_dir=path / "dist")


if __name__ == "__main__":
    unittest.main()
