"""Evidence and teaching requirements; all metrics below are explicit test fixtures."""

import copy
import importlib.util
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import subprocess

from presentation.build import encode_video, formula_image, insert_code_slides, main, speech_chunks, srt_time, word_captions, write_html, write_json
from presentation.content import Slide, create_slides, ensure_pairs, validate_evidence
from presentation.external import components_from, gpt2_slides


def fixture():
    evidence = {
        "competition": {"url": "https://example.test/competition", "title": "TEST FIXTURE"},
        "checks": {"data_verified": True, "training_verified": True, "submission_verified": True},
        "links": {"github": "https://example.test/repo"},
        "results": {"public_auc": 0.7, "ranking": 8, "total_teams": 10,
                    "submission_id": "fixture-only", "observed_at": "2026-11-16T00:00:00Z"},
        "run": {"submission_sha256": "a" * 64},
    }
    run = {
        "status": "complete", "model": "mlp", "roc_auc": 0.7, "data_hashes": {"train.csv": "fixture-only"},
        "submission_sha256": "a" * 64,
        "features": {"count": 3}, "rows": {"train": 100, "test": 20},
        "validation": {"method": "stratified", "n_splits": 3},
        "model_details": {"architecture": {"hidden_dims": [8, 4], "numeric_features": 3,
            "dropout": 0.1, "hidden_activation": "ReLU", "output_activation": "sigmoid",
            "categorical_cardinalities": []}, "optimizer": "AdamW", "batch_size": 16,
            "early_stopping_patience": 3},
    }
    return evidence, run


class EvidenceTests(unittest.TestCase):
    def test_changed_input_during_narration_is_rejected(self):
        evidence, metrics = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, run = root / "evidence.json", root / "metrics.json"
            write_json(source, evidence)
            write_json(run, metrics)
            def change_input(*args):
                source.write_text('{"changed":true}', encoding="utf-8")
                return [{"slide": 1, "start": 0, "duration": 1}]
            with patch("presentation.build.create_slides", return_value=[Slide("test", "test", [], "test")]), \
                 patch("presentation.build.insert_code_slides", return_value=[]), \
                 patch("presentation.build.render_slides"), \
                 patch("presentation.build.narrate", side_effect=change_input), \
                 patch("presentation.build.encode_video"):
                with self.assertRaisesRegex(ValueError, "Inputs changed during generation"):
                    main(["--evidence", str(source), "--metrics", str(run), "--output", str(root / "out"), "--video"])
            self.assertFalse((root / "out/manifest.json").exists())

    def test_failed_encoding_preserves_previous_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "video.mp4"
            target.write_bytes(b"previous verified video")
            def encode_stub(command, **kwargs):
                Path(command[-1]).write_bytes(b"partial candidate")
                raise subprocess.CalledProcessError(1, command)
            with patch("presentation.build.shutil.which", return_value="ffmpeg"), \
                 patch("presentation.build.subprocess.run", side_effect=encode_stub):
                with self.assertRaises(subprocess.CalledProcessError):
                    encode_video(root, [{"slide": 1, "start": 0, "duration": 1}])
            self.assertEqual(target.read_bytes(), b"previous verified video")
            self.assertEqual(list(root.glob(".video-*.mp4")), [])

    def test_only_validated_unique_candidate_replaces_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "video.mp4"
            target.write_bytes(b"previous verified video")
            candidates = []
            def encode_stub(command, **kwargs):
                candidate = Path(command[-1])
                candidates.append(candidate)
                candidate.write_bytes(b"new video")
                return subprocess.CompletedProcess(command, 0)
            with patch("presentation.build.shutil.which", return_value="ffmpeg"), \
                 patch("presentation.build.subprocess.run", side_effect=encode_stub), \
                 patch("presentation.build.validate_encoded_video", side_effect=RuntimeError("decode failed")):
                with self.assertRaisesRegex(RuntimeError, "decode failed"):
                    encode_video(root, [{"slide": 1, "start": 0, "duration": 1}])
            self.assertEqual(target.read_bytes(), b"previous verified video")
            with patch("presentation.build.shutil.which", return_value="ffmpeg"), \
                 patch("presentation.build.subprocess.run", side_effect=encode_stub), \
                 patch("presentation.build.validate_encoded_video"):
                encode_video(root, [{"slide": 1, "start": 0, "duration": 1}])
            self.assertEqual(target.read_bytes(), b"new video")
            self.assertNotEqual(candidates[0], candidates[1])
            self.assertTrue(all(path != target for path in candidates))

    def test_final_rejects_unverified_submission(self):
        evidence, run = fixture()
        evidence["checks"]["submission_verified"] = False
        with self.assertRaisesRegex(ValueError, "submission_verified"):
            validate_evidence(evidence, [run], draft=False)

    def test_final_cannot_use_score_from_another_submission_file(self):
        evidence, run = fixture()
        evidence["run"]["submission_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "submission_sha256"):
            validate_evidence(evidence, [run], draft=False)
        slides = create_slides(evidence, [run], draft=True)
        result = next(slide for slide in slides if slide.title == "Submissão e posição observadas")
        self.assertTrue(all("não verificad" in point for point in result.points))

    def test_final_requires_executed_neural_architecture(self):
        evidence, run = fixture()
        run["model"] = "catboost"
        with self.assertRaisesRegex(ValueError, "trained MLP"):
            validate_evidence(evidence, [run], draft=False)

    def test_final_rejects_incomplete_or_incompatible_experiments(self):
        evidence, run = fixture()
        run["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "status complete"):
            validate_evidence(evidence, [run], draft=False)
        run["status"] = "complete"
        comparison = copy.deepcopy(run)
        comparison["data_hashes"]["train.csv"] = "other-data"
        with self.assertRaisesRegex(ValueError, "compatible data_hashes"):
            validate_evidence(evidence, [run, comparison], draft=False)

    def test_final_rejects_nonfinite_or_invalid_scores(self):
        for score in (float("nan"), float("inf"), -0.1, 1.1, None, True):
            evidence, run = fixture()
            evidence["results"]["public_auc"] = score
            with self.assertRaisesRegex(ValueError, "public_auc"):
                validate_evidence(evidence, [run], draft=False)

    def test_draft_keeps_unknown_results_unknown(self):
        slides = create_slides({}, [], draft=True)
        ranking = next(slide for slide in slides if slide.title == "Submissão e posição observadas")
        self.assertTrue(all("não verificada" in point or "não verificado" in point for point in ranking.points))
        self.assertFalse(any("AUC OOF 0" in point for slide in slides for point in slide.points))

    def test_every_complex_slide_is_followed_immediately_by_analogy(self):
        evidence, run = fixture()
        logistic = copy.deepcopy(run)
        logistic["model"] = "logistic"
        catboost = copy.deepcopy(run)
        catboost["model"] = "catboost"
        slides = create_slides(evidence, [run, logistic, catboost])
        ensure_pairs(slides)
        self.assertGreaterEqual(sum(slide.kind == "technical" for slide in slides), 10)
        with self.assertRaisesRegex(ValueError, "sem analogia"):
            ensure_pairs([Slide("test", "complex", [], "", kind="technical")])

    def test_unexecuted_methods_are_not_taught_as_used(self):
        evidence, run = fixture()
        run["model_details"]["architecture"]["dropout"] = 0
        run["model_details"]["optimizer"] = "SGD"
        slides = create_slides(evidence, [run])
        titles = [slide.title for slide in slides]
        self.assertNotIn("Dropout: reduzir dependências internas", titles)
        self.assertNotIn("AdamW: passos adaptativos e decaimento", titles)
        self.assertNotIn("Comparação: árvores em sequência", titles)
        self.assertNotIn("Target encoding com suavização", titles)
        self.assertNotIn("Quantis: outra preparação numérica", titles)

    def test_neural_baseline_difference_is_reported_honestly(self):
        evidence, neural = fixture()
        logistic = copy.deepcopy(neural)
        logistic.update(model="logistic", roc_auc=0.71)
        slides = create_slides(evidence, [neural, logistic])
        comparison = next(slide for slide in slides if slide.title == "A rede básica superou a referência?")
        self.assertIn("Não superou nesta execução", comparison.points)
        self.assertIn("-0,01000", " ".join(comparison.points))

    def test_refit_is_not_presented_as_a_new_oof_score(self):
        evidence, development = fixture()
        refit = copy.deepcopy(development)
        refit.update(phase="full_data_refit", roc_auc=None, fold_scores=[], fold_std=None,
                     selection={"development_roc_auc": development["roc_auc"]})
        with self.assertRaisesRegex(ValueError, "development source"):
            validate_evidence(evidence, [refit], draft=False)
        holdout = {"status": "complete", "kind": "independent_holdout", "roc_auc": 0.72,
                   "data_hashes": development["data_hashes"]}
        slides = create_slides(evidence, [refit, development], holdout=holdout)
        result = next(slide for slide in slides if slide.title.startswith("Refit final"))
        self.assertIn("Este refit não produz uma nova AUC OOF ou independente", result.points)

    def test_external_gpt2_has_credit_and_cannot_reuse_old_holdout(self):
        evidence, run = fixture()
        component = {
            "kind": "gpt2_target_first", "name": "GPT-2 target-first", "trained_by": "External fixture author",
            "training_code_url": "https://example.test/code", "training_code_commit": "fixture-only",
            "code_license": "MIT", "predictions_source_url": "https://example.test/predictions",
            "predictions_license": "CC BY 4.0", "validation": {"n_splits": 5, "seed": 42},
            "inference": {"target_first": True, "likelihood_ratio": True, "orders": 10,
                          "order_aggregation": "log-sum-exp by label", "test_aggregation": "mean of fold log-odds"},
        }
        run["model_details"]["external_components"] = [component]
        holdout = {"status": "complete", "kind": "independent_holdout", "roc_auc": 0.72,
                   "data_hashes": run["data_hashes"], "rows": 15, "positive_rows": 3}
        with self.assertRaisesRegex(ValueError, "must be historical"):
            validate_evidence(evidence, [run], False, holdout)
        slides = create_slides(evidence, [run], holdout=holdout, historical_holdout=True)
        ensure_pairs(slides)
        credits = next(slide for slide in slides if slide.title.startswith("GPT-2 externo"))
        self.assertIn("External fixture author", " ".join(credits.points))
        self.assertIn("não executou esse treinamento", " ".join(credits.points))
        historical = next(slide for slide in slides if slide.title.startswith("Reserva histórica"))
        self.assertIn("Não é independente para o GPT-2 nem para a nova composição", historical.points)
        for spec in gpt2_slides(component):
            if spec["formula"] and importlib.util.find_spec("matplotlib"):
                with self.subTest(formula=spec["title"]):
                    self.assertGreater(formula_image(Slide(**spec)).width, 0)

    def test_selected_external_weight_survives_standalone_comparison(self):
        selected = {"model_details": {"external_components": [{"training_code_url": "source",
                    "training_code_commit": "commit", "weight": 0.25}]}}
        standalone = copy.deepcopy(selected)
        standalone["model_details"]["external_components"][0].pop("weight")
        self.assertEqual(components_from([selected, standalone])[0]["weight"], 0.25)

    def test_html_escapes_content_and_preserves_code(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            slide = Slide("test", "<script>unsafe</script>", ["a < b"], "Roteiro", code="loss.backward()")
            write_html([slide], path, draft=True)
            text = (path / "slides.html").read_text(encoding="utf-8")
            self.assertIn("&lt;script&gt;unsafe&lt;/script&gt;", text)
            self.assertIn("loss.backward()", text)
            self.assertIn("RASCUNHO", text)

    def test_source_snippets_are_literal_and_have_line_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src/fraudlab").mkdir(parents=True)
            (root / "src/fraudlab/models.py").write_text("    loss.backward()\n    optimizer.step()\n", encoding="utf-8")
            (root / "src/fraudlab/experiment.py").write_text('oof_frame.to_csv(run / "oof.csv", index=False)\n', encoding="utf-8")
            slides = [Slide("09 / Ranking", "Evidência", [], "")]
            used = insert_code_slides(slides, root, required=True)
            self.assertEqual(len(used), 2)
            self.assertEqual(slides[0].code, "loss.backward()\noptimizer.step()")
            self.assertEqual(slides[0].code_reference, "src/fraudlab/models.py:1")

    def test_caption_chunks_preserve_text_and_time_format(self):
        text = "Uma explicação clara. " + "A palavra importa " * 30
        chunks = speech_chunks(text, 80)
        self.assertTrue(all(len(chunk) <= 80 for chunk in chunks))
        self.assertEqual(" ".join(chunks), text.strip())
        self.assertEqual(srt_time(3661.007), "01:01:01,007")

    def test_edge_cues_keep_measured_word_boundaries(self):
        words = [{"offset": i * 10_000_000, "duration": 5_000_000, "text": word}
                 for i, word in enumerate(("Esta", "frase", "tem", "tempos", "reais"))]
        cues = word_captions(words, max_chars=12)
        self.assertEqual(" ".join(cue["text"] for cue in cues), "Esta frase tem tempos reais")
        self.assertEqual(cues[0]["start"], 0)
        self.assertEqual(cues[0]["end"], 1.5)
        self.assertEqual(cues[-1]["end"], 4.5)
        punctuated = word_captions(words, text="Esta frase tem tempos reais.")
        self.assertEqual(punctuated[-1]["text"], "Esta frase tem tempos reais.")

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "presentation dependencies are optional")
    def test_all_conditional_formulas_can_be_rendered(self):
        evidence, run = fixture()
        run["model_details"]["architecture"]["categorical_cardinalities"] = [4]
        run["model_details"]["preprocessing"] = "Fold-local standard scaling"
        run["validation"]["holdout_fraction"] = 0.15
        run["rows"].update({"development": 85, "holdout": 15})
        blend, logistic, catboost = (copy.deepcopy(run) for _ in range(3))
        blend["model"], logistic["model"], catboost["model"] = "blend", "logistic", "catboost"
        lightgbm, glm = (copy.deepcopy(run) for _ in range(2))
        lightgbm["model"], glm["model"] = "lightgbm", "glm"
        lightgbm["model_details"].update(num_leaves=4, feature_set="digits",
            target_encoder={"internal_folds": 5, "smooth": "auto", "unknown": "training prior"})
        lightgbm["features"]["names"] = ["amount_usd__log1p", "time_of_day_hour__sin",
                                          "amount_balance_ratio", "amount_usd__cents_key"]
        raw_tree = copy.deepcopy(lightgbm)
        raw_tree["model_details"]["feature_set"] = "raw"
        glm["model_details"]["target_encoder"] = {"internal_folds": 5, "smooth": 20, "log_odds": True}
        glm["model_details"].update(solver="newton-cholesky", C=0.3)
        run["model_details"]["numeric_transform"] = "quantile"
        slides = create_slides(evidence, [blend, run, logistic, catboost, lightgbm, raw_tree, glm])
        for slide in slides:
            if slide.formula:
                with self.subTest(slide=slide.title):
                    image = formula_image(slide)
                    self.assertGreater(image.width, 0, slide.title)
                    self.assertEqual(image.mode, "RGBA")


if __name__ == "__main__":
    unittest.main()
