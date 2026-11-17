"""Presentation continuation contracts; all small metrics below are fixtures."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from matplotlib.mathtext import MathTextParser

from .audio_cache import edge_paths
from .build import insert_code_slides, main
from .content import Slide, create_slides, ensure_pairs, number
from .continuation import (append_feature_pairs, append_offset_pair, append_tabm_pair,
                           append_tabm_embedding_pair,
                           append_tabm_result, append_selection_note, has_frequency,
                           missing_selection_context, append_historical_deviations)


def collector():
    slides = []
    def add(chapter, title, points, narration, **kwargs):
        slides.append(Slide(chapter, title, points, narration, **kwargs))
    def pair(chapter, title, points, narration, formula, formula_text,
             simple_title, simple_points, simple_narration, **kwargs):
        add(chapter, title, points, narration, kind="technical", formula=formula, formula_text=formula_text)
        add(chapter, simple_title, simple_points, simple_narration, kind="analogy")
    return slides, add, pair


def own(model="glm"):
    return {"run_name": "own", "status": "complete", "model": model,
            "feature_encoding": {"encoding": "gpt2"},
            "model_details": {"category_frequency": "log1p counts"}}


class ContinuationTests(unittest.TestCase):
    def test_new_complex_concepts_have_immediate_analogies_and_valid_formulas(self):
        slides, add, pair = collector()
        append_feature_pairs(pair, [own()])
        neural = {"status": "complete", "model": "tabm", "model_details": {
            "architecture": {"k": 16, "n_blocks": 2, "d_block": 256, "num_embeddings": {
                "name": "PiecewiseLinearEmbeddings", "version": "B", "activation": False,
                "d_embedding": 12, "n_bins": 48}}}}
        append_tabm_embedding_pair(pair, [neural])
        append_tabm_pair(pair, [neural])
        ensure_pairs(slides)
        self.assertEqual(len(slides), 8)
        text = " ".join(point for slide in slides for point in slide.points)
        self.assertIn("3388:35", text)
        self.assertIn("IDs não indicam ordem", text)
        for slide in slides:
            if slide.formula:
                MathTextParser("agg").parse("$" + slide.formula + "$", dpi=100)

    def test_neural_math_precedes_external_model_experiments_code_and_ranking(self):
        from .test_presentation import fixture
        evidence, mlp = fixture()
        tabm = copy.deepcopy(mlp)
        tabm.update(model="tabm", run_name="round8-tabm")
        tabm["model_details"] = {"architecture": {"k": 16, "n_blocks": 2, "d_block": 256,
            "num_embeddings": {"name": "PiecewiseLinearEmbeddings", "version": "B",
                "activation": False, "d_embedding": 12, "n_bins": 48}}}
        mlp["model_details"]["external_components"] = [{"kind": "gpt2_target_first",
            "trained_by": "Fixture author", "training_code_url": "https://example.test/code",
            "training_code_commit": "fixture-only", "code_license": "MIT",
            "predictions_source_url": "https://example.test/predictions", "predictions_license": "CC BY 4.0"}]
        slides = create_slides(evidence, [mlp, tabm])
        root = next(parent for parent in Path(__file__).resolve().parents
                    if (parent / "src/fraudlab").is_dir())
        insert_code_slides(slides, root, required=True)
        ensure_pairs(slides)
        titles = [slide.title for slide in slides]
        embedding = titles.index("TabM: representar posições entre cortes do treino")
        ensemble = titles.index("TabM: várias previsões com pesos compartilhados")
        external = titles.index("GPT-2 externo: autoria e responsabilidade")
        experiments = next(index for index, slide in enumerate(slides) if slide.chapter.startswith("07 /"))
        code = next(index for index, slide in enumerate(slides) if slide.code)
        ranking = titles.index("Submissão e posição observadas")
        self.assertLess(embedding, ensemble)
        self.assertLess(ensemble, external)
        self.assertLess(external, experiments)
        self.assertLess(experiments, code)
        self.assertLess(code, ranking)
        self.assertTrue(all(index < code for index, slide in enumerate(slides)
                            if slide.kind == "technical" and not slide.code))

    def test_unexecuted_or_comparison_only_method_is_not_selected(self):
        slides, _, pair = collector()
        selected = {"run_name": "selected", "status": "complete", "model": "glm", "model_details": {}}
        append_feature_pairs(pair, [selected, own()])
        append_offset_pair(pair, [selected, own("offset")])
        self.assertEqual(slides, [])
        incomplete = own()
        incomplete["status"] = "running"
        append_feature_pairs(pair, [incomplete])
        self.assertEqual(slides, [])
        selected["model"] = "blend"
        selected["feature_encoding"] = {"encoding": "gpt2"}  # A copied parent field is insufficient.
        selected["model_details"] = {"sources": ["runs/own"], "weights": [0]}
        append_feature_pairs(pair, [selected, own()])
        self.assertEqual(slides, [])

    def test_offset_frequency_requires_actual_verified_config_or_metadata(self):
        selected = {"run_name": "blend", "status": "complete", "model": "blend",
                    "model_details": {"sources": ["C:\\runs\\own"]}}
        offset = own("offset")
        offset["model_details"] = {"residual_parameters": {"n_estimators": 1000, "num_leaves": 4}}
        self.assertFalse(has_frequency([selected, offset]))
        offset["run_config"] = {"glm_frequency": True}
        self.assertTrue(has_frequency([selected, offset]))
        slides, _, pair = collector()
        append_offset_pair(pair, [selected, offset])
        ensure_pairs(slides)
        self.assertEqual(len(slides), 2)
        self.assertIn("1000", " ".join(slides[0].points))

    def test_exploratory_deviation_is_not_claimed_as_original_gate_pass(self):
        evidence = {"run": {"path": "runs/selected", "selection_reason_pt": "Exceção exploratória registrada."},
                    "public_evaluation": {"run": "selected", "protocol_deviation": True, "consistency_gate_passed": False,
                       "improved_folds": 3, "required_folds": 4, "total_folds": 5,
                       "declared_before_upload": True}}
        self.assertEqual(missing_selection_context(evidence), [])
        slides, add, _ = collector()
        self.assertTrue(append_selection_note(add, evidence))
        self.assertIn("3/5", " ".join(slides[0].points))
        self.assertIn("4/5", " ".join(slides[0].points))
        self.assertIn("não foi cumprido", slides[0].title)
        broken = copy.deepcopy(evidence)
        broken["public_evaluation"]["consistency_gate_passed"] = True
        self.assertTrue(missing_selection_context(broken))

    def test_code_sections_use_real_sources_and_keep_immediate_analogies(self):
        root = next(parent for parent in Path(__file__).resolve().parents
                    if (parent / "src/fraudlab").is_dir())
        slides = [Slide("test", "Submissão e posição observadas", [], "test")]
        used = insert_code_slides(slides, root, True, runs=[own()])
        ensure_pairs(slides)
        self.assertIn(root / "src/fraudlab/bpe.py", used)
        self.assertIn(root / "src/fraudlab/frequency.py", used)

    def test_historical_exception_is_not_assigned_to_current_selection(self):
        evidence = {"public_evaluation": {"protocol_deviation": False},
                    "historical_submissions": [{"submission_file_verified": True,
                        "submission_id": "fixture-old", "protocol_deviation": True,
                        "declared_before_upload": True,
                        "improved_folds": 3, "required_folds": 4, "total_folds": 5}]}
        self.assertEqual(missing_selection_context(evidence), [])
        slides, add, _ = collector()
        self.assertFalse(append_selection_note(add, evidence))
        append_historical_deviations(add, evidence)
        self.assertEqual(len(slides), 1)
        self.assertIn("histórica", slides[0].title)
        self.assertIn("3/5", " ".join(slides[0].points))

    def test_changed_run_config_is_rejected_before_any_render(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "evidence.json").write_text("{}", encoding="utf-8")
            (root / "metrics.json").write_text(json.dumps({"config_sha256": "a" * 64}), encoding="utf-8")
            (root / "config.json").write_text('{"glm_frequency":true}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "config does not match"):
                main(["--evidence", str(root / "evidence.json"), "--metrics", str(root / "metrics.json"),
                      "--output", str(root / "unused"), "--draft"])
            self.assertFalse((root / "unused").exists())

    def test_rejected_tabm_uses_recorded_comparison_and_not_selected_language(self):
        candidate = {"run_name": "round8-tabm", "status": "complete", "model": "tabm", "roc_auc": .68}
        baseline = {"run_name": "explore5-public25", "status": "complete", "model": "blend", "roc_auc": .7, "fold_scores": [.7, .7]}
        mixture = {"run_name": "round8-tabm10", "status": "complete", "model": "blend", "roc_auc": .69, "fold_scores": [.68, .71]}
        slides, add, _ = collector()
        append_tabm_result(add, [baseline, candidate, mixture], number)
        self.assertIn("descartado", slides[0].title)
        self.assertIn("1/2", " ".join(slides[0].points))


class AudioCacheTests(unittest.TestCase):
    def test_audio_is_reused_across_slide_index_changes_without_modifying_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous, current = root / "old", root / "new"
            previous.mkdir()
            request = {"text": "Texto estável.", "voice": "fixture-voice", "rate": 0}
            original = previous / "edge-09.request.json"
            original.write_text(json.dumps(request), encoding="utf-8")
            (previous / "edge-09.mp3").write_bytes(b"fixture audio only")
            (previous / "edge-09.jsonl").write_text("{}\n", encoding="utf-8")
            before = {path.name: path.read_bytes() for path in previous.iterdir()}
            mp3, metadata, cached_request, hit = edge_paths(current, request, previous)
            self.assertTrue(hit)
            self.assertNotEqual(mp3.stem, "edge-09")
            self.assertEqual(mp3.read_bytes(), b"fixture audio only")
            self.assertEqual(before, {path.name: path.read_bytes() for path in previous.iterdir()})
            self.assertTrue(edge_paths(current, request)[-1])

    def test_voice_rate_or_text_change_cannot_reuse_other_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = {"text": "A", "voice": "fixture", "rate": 0}
            mp3, metadata, description, hit = edge_paths(root, request)
            mp3.write_bytes(b"fixture")
            metadata.write_text("{}\n", encoding="utf-8")
            description.write_text(json.dumps(request), encoding="utf-8")
            for field, value in [("text", "B"), ("voice", "other"), ("rate", 1)]:
                changed = {**request, field: value}
                self.assertFalse(edge_paths(root, changed)[-1])


if __name__ == "__main__":
    unittest.main()
