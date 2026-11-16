"""Evidence and teaching requirements; all metrics below are explicit test fixtures."""

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

from presentation.build import formula_image, insert_code_slides, speech_chunks, srt_time, word_captions, write_html
from presentation.content import Slide, create_slides, ensure_pairs, validate_evidence


def fixture():
    evidence = {
        "competition": {"url": "https://example.test/competition", "title": "TEST FIXTURE"},
        "checks": {"data_verified": True, "training_verified": True, "submission_verified": True},
        "links": {"github": "https://example.test/repo"},
        "results": {"public_auc": 0.7, "ranking": 8, "total_teams": 10,
                    "submission_id": "fixture-only", "observed_at": "2026-11-16T00:00:00Z"},
    }
    run = {
        "status": "complete", "model": "mlp", "roc_auc": 0.7, "data_hashes": {"train.csv": "fixture-only"},
        "features": {"count": 3}, "rows": {"train": 100, "test": 20},
        "validation": {"method": "stratified", "n_splits": 3},
        "model_details": {"architecture": {"hidden_dims": [8, 4], "numeric_features": 3,
            "dropout": 0.1, "hidden_activation": "ReLU", "output_activation": "sigmoid",
            "categorical_cardinalities": []}, "optimizer": "AdamW", "batch_size": 16,
            "early_stopping_patience": 3},
    }
    return evidence, run


class EvidenceTests(unittest.TestCase):
    def test_final_rejects_unverified_submission(self):
        evidence, run = fixture()
        evidence["checks"]["submission_verified"] = False
        with self.assertRaisesRegex(ValueError, "submission_verified"):
            validate_evidence(evidence, [run], draft=False)

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
        slides = create_slides(evidence, [blend, run, logistic, catboost])
        for slide in slides:
            if slide.formula:
                image = formula_image(slide)
                self.assertGreater(image.width, 0, slide.title)
                self.assertEqual(image.mode, "RGBA")


if __name__ == "__main__":
    unittest.main()
