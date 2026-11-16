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
