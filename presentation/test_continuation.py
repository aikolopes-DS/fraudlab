"""Presentation continuation contracts; all small metrics below are fixtures."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from matplotlib.mathtext import MathTextParser

from .audio_cache import edge_paths
from .external import components_from
from .build import insert_code_slides, main
from .content import Slide, create_slides, ensure_pairs, number, spoken_year_2026, development_source, validate_evidence
from .competitive import (append_pooling_pairs, append_joint_pair, append_neural_offset_pair,
                          append_transductive_pair, append_storage_pair, append_nested_stack_pair,
                          append_later_results, append_nested_result, append_context)
from .continuation import (append_feature_pairs, append_offset_pair, append_tabm_pair,
                           append_tabm_embedding_pair, append_tabm_backbone_pair, append_proposal,
                           append_tabm_result, append_selection_note, has_frequency,
                           missing_selection_context, append_historical_deviations, selected_runs)


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
    def advanced_run(self):
        run = own("offset")
        run["features"] = {"count": 92}
        run["source_hashes"] = {"strings.py": "a" * 64}
        run["feature_encoding"] = {"bpe": {"encoding": "gpt2"}, "joint_encoding": {"output_count": 8}}
        run["model_details"].update(feature_set="joint_bpe", encoded_features=223, pooled_tokens={
            "algorithm": "hierarchical_pooled_decimal_token_rates",
            "smoothing": {"pooled_to_prior": 50, "field_to_pool": 2000, "markov_to_field_decimal": 200},
            "crossfit": {"n_splits": 5}})
        return run

    def test_current_math_pairs_are_immediate_and_renderable_without_scores(self):
        slides, _, pair = collector()
        selected = self.advanced_run()
        neural = {"status": "complete", "model": "neural_offset", "model_details": {"architecture": {
            "input_features": 207, "hidden_dims": [256, 256], "dropout": 0.1}}}
        transductive = {"status": "complete", "model_details": {"transductive_frequency": {"label_usage": "none"}}}
        stack = {"status": "complete", "model_details": {"algorithm": "nested_owned_logit_holdout_stack",
            "inner_fraction": 0.25, "inner_seed": 2026, "base_runs": ["a", "b", "c"],
            "architecture": {"external_teacher_in_meta": False}}}
        append_pooling_pairs(pair, [selected])
        append_joint_pair(pair, [selected])
        append_neural_offset_pair(pair, [neural])
        append_transductive_pair(pair, [transductive])
        append_storage_pair(pair, [selected], {})
        append_nested_stack_pair(pair, [stack])
        ensure_pairs(slides)
        self.assertEqual(len(slides), 14)
        parser = MathTextParser("path")
        for slide in slides:
            if slide.formula:
                with self.subTest(title=slide.title):
                    parser.parse("$" + slide.formula + "$")
        self.assertIn(r"\beta q_G(v)", slides[0].formula)
        self.assertIn(r"\gamma r_j(v)", slides[2].formula)
        self.assertIn("não significa que treinamos uma cadeia temporal", slides[2].narration)
        self.assertIn("BCEWithLogits", slides[6].formula)
        self.assertIn("não precisam ser positivos nem somar um", slides[-2].narration)
        self.assertIn("Ele não participou do meta-treino", slides[-2].narration)

    def test_stack_inputs_are_not_filtered_by_learned_coefficient_sign(self):
        stack = {"run_name": "stack", "model": "nested_stack", "status": "complete",
                 "model_details": {"base_runs": ["runs/a", "runs/b", "runs/c"], "weights": [-2, 0, 3]}}
        bases = [{"run_name": key, "model": "offset", "status": "complete"} for key in ("a", "b", "c")]
        self.assertEqual({run['run_name'] for run in selected_runs([stack, *bases])}, {"stack", "a", "b", "c"})

    def test_stack_explains_fold_oof_and_mean_of_probabilities_separately(self):
        slides, _, pair = collector()
        stack = {"status": "complete", "model_details": {"algorithm": "nested_owned_logit_holdout_stack",
                 "inner_fraction": 0.25, "inner_seed": 2026, "base_runs": ["a", "b", "c"],
                 "architecture": {"external_teacher_in_meta": False}}}
        append_nested_stack_pair(pair, [stack])
        ensure_pairs(slides)
        MathTextParser("path").parse("$" + slides[0].formula + "$")
        self.assertIn(r"\bar p_{\mathrm{teste}}=\frac{1}{5}\sum_{k=1}^{5}p_k", slides[0].formula)
        self.assertIn("cada linha recebe somente a previsão do seu fold externo", slides[0].narration)
        self.assertIn("não aplicamos sigmoid à média das margens", slides[0].narration)
        self.assertIn("sessenta por cento", slides[0].narration)
        self.assertIn("oitenta por cento", slides[0].narration)
        self.assertIn("não três colunas originais", slides[0].narration)
        self.assertIn("cinco equipes", slides[1].narration)

    def test_stack_meta_width_is_not_reported_as_original_feature_count(self):
        from .test_presentation import fixture
        evidence, neural = fixture()
        neural.update(run_name="neural", features={"count": 24})
        neural["model_details"]["feature_set"] = "raw"
        base = copy.deepcopy(neural)
        base.update(run_name="base", model="logistic", features={"count": 92, "names": ["amount"],
                    "numeric": 24, "categorical": 68}, target={"positive_rate": 0.02})
        base["model_details"]["feature_set"] = "joint_bpe"
        stack = copy.deepcopy(neural)
        stack.update(run_name="stack", model="nested_stack", features={"count": 3},
            model_details={"algorithm": "nested_owned_logit_holdout_stack", "base_runs": ["base", "base", "base"],
            "inner_fraction": 0.25, "inner_seed": 2026, "architecture": {"external_teacher_in_meta": False}})
        slides = create_slides(evidence, [stack, base, neural])
        data = next(slide for slide in slides if slide.title == "O que foi observado nos arquivos")
        self.assertIn("Colunas originais: 24; exemplo de base: 92 colunas após engenharia", data.points)
        self.assertIn("não colunas originais", data.narration)
        self.assertIn("Taxa positiva no treino: 2,000%", data.points)

    def test_stack_keeps_both_frequency_protocols_when_both_bases_are_used(self):
        slides, _, pair = collector()
        stack = {"run_name": "stack", "model": "nested_stack", "status": "complete",
                 "model_details": {"base_runs": ["local", "transductive", "third"]}}
        local, transductive = own(), own()
        local["run_name"] = "local"
        transductive["run_name"] = "transductive"
        transductive["model_details"]["transductive_frequency"] = {"label_usage": "none"}
        append_feature_pairs(pair, [stack, local, transductive])
        append_transductive_pair(pair, [stack, local, transductive])
        self.assertTrue(any(slide.title == "Frequência: quantas vezes uma chave apareceu no treino" for slide in slides))
        self.assertTrue(any(slide.title == "Experimento: contar entradas conhecidas do teste" for slide in slides))
        ensure_pairs(slides)

    def test_stack_trial_public_score_requires_matching_submission_bytes(self):
        slides, add, _ = collector()
        own_run = {"run_name": "stack", "status": "complete", "model": "nested_stack", "roc_auc": 0.8,
                   "model_details": {"algorithm": "nested_owned_logit_holdout_stack"}}
        hybrid = {"run_name": "hybrid", "status": "complete", "model": "blend", "roc_auc": 0.81,
                  "fold_scores": [0.81] * 5, "submission_sha256": "a" * 64, "model_details": {"sources": ["stack"]}}
        reference = {"run_name": "round18-pooled-share100-public25", "status": "complete",
                     "roc_auc": 0.80999, "fold_scores": [0.80999] * 5}
        record = {"run": "hybrid", "submission_file_verified": True, "submission_sha256": "b" * 64,
                  "submission_id": "fixture-trial", "public_auc": 0.79, "note": "Não promovida."}
        evidence = {"historical_submissions": [record]}
        append_nested_result(add, [own_run, hybrid, reference], evidence, number)
        self.assertFalse(any("AUC pública" in point for point in slides[0].points))
        record["submission_sha256"] = "a" * 64
        slides.clear()
        append_nested_result(add, [own_run, hybrid, reference], evidence, number)
        self.assertIn("AUC pública 0,79000", slides[0].points[-1])
        self.assertIn("Não promovida", slides[0].points[-1])
        self.assertIn("Não atribuímos", slides[0].narration)

    def test_unexecuted_and_comparison_only_pooling_are_not_selected_facts(self):
        slides, _, pair = collector()
        candidate = self.advanced_run()
        candidate["status"] = "running"
        append_pooling_pairs(pair, [candidate])
        append_joint_pair(pair, [candidate])
        append_storage_pair(pair, [candidate], {})
        self.assertFalse(slides)
        selected = own()
        candidate["status"] = "complete"
        append_pooling_pairs(pair, [selected, candidate])
        self.assertFalse(slides)

    def test_storage_probe_is_scoped_to_buffers_and_never_claimed_as_rss(self):
        slides, _, pair = collector()
        probe = {"rows": 1000000, "unique_values": 100, "values_exact": True,
                 "legacy_bytes": 65000000, "shared_bytes": 8005700, "process_rss_measured": False}
        append_storage_pair(pair, [self.advanced_run()], {"presentation_context": {"string_storage_probe": probe}})
        text = slides[0].narration
        self.assertIn("8,0057", text)
        self.assertIn("Não mede RSS", text)
        self.assertIn("não é um novo atributo", text.lower())
        slides.clear()
        probe["process_rss_measured"] = True
        append_storage_pair(pair, [self.advanced_run()], {"presentation_context": {"string_storage_probe": probe}})
        self.assertNotIn("65 para", slides[0].narration)

    def test_refit_resolves_explicit_hybrid_cv_instead_of_first_same_family(self):
        from .test_presentation import fixture
        evidence, neural = fixture()
        historical = copy.deepcopy(neural)
        historical.update(run_name="older-hybrid", model="blend", roc_auc=0.72)
        reference = copy.deepcopy(historical)
        reference.update(run_name="selected-hybrid", roc_auc=0.75,
                         target={"positive_rate": 0.02}, validation={"method": "stratified", "n_splits": 5})
        refit = copy.deepcopy(historical)
        refit.update(run_name="refit", model="offset", phase="full_data_refit", roc_auc=None, target=None,
                     validation={"independent_holdout": False}, selection={"reference_cv_run": "selected-hybrid",
                     "development_roc_auc": 0.75})
        runs = [refit, historical, reference, neural]
        self.assertIs(development_source(refit, runs), reference)
        self.assertEqual(validate_evidence(evidence, runs, False), [])
        slides = create_slides(evidence, runs)
        data = next(slide for slide in slides if slide.title == "O que foi observado nos arquivos")
        self.assertIn("Taxa positiva no treino: 2,000%", data.points)
        with self.assertRaisesRegex(ValueError, "development source"):
            validate_evidence(evidence, [refit, historical, neural], False)

    def test_refit_storage_uses_recorded_config_when_family_metadata_omits_features(self):
        slides, _, pair = collector()
        selected = self.advanced_run()
        selected["model_details"].pop("feature_set")
        selected["run_config"] = {"feature_set": "joint_bpe"}
        append_storage_pair(pair, [selected], {})
        self.assertEqual(len(slides), 2)

    def test_refit_external_record_keeps_selected_weight_and_code_coverage(self):
        component = {"training_code_url": "fixture", "training_code_commit": "fixed", "weight": 0.25}
        refit = {"external_components": [component]}
        comparison = {"model_details": {"external_components": [dict(component, weight=1)]}}
        self.assertEqual(components_from([refit, comparison]), [component])
        slides, add, _ = collector()
        add("test", "Submissão e posição observadas", [], "fixture")
        insert_code_slides(slides, Path(__file__).resolve().parents[1], True,
                           bool(components_from([refit])), {}, [])
        self.assertTrue(any(slide.code_reference.startswith("src/fraudlab/external.py:") for slide in slides))
        self.assertTrue(any("prediction_result[target] = sum(" in slide.code for slide in slides))

    def test_later_negative_results_do_not_promote_five_folds_with_small_delta(self):
        slides, add, _ = collector()
        baseline = {"run_name": "round10-offset-bpe-frequency-public25", "model": "blend", "status": "complete",
                    "roc_auc": 0.8, "fold_scores": [0.8] * 5}
        candidate = {"run_name": "round17-pooled-share050-public25", "model": "blend", "status": "complete",
                     "roc_auc": 0.80005, "fold_scores": [0.80005] * 5}
        append_later_results(add, [baseline, candidate], number)
        self.assertIn("5/5 folds; não passou gate R10", slides[0].points[0])
        self.assertNotIn("vinte mil", slides[0].narration)

    def test_snapshot_status_and_test_count_do_not_create_submission_claims(self):
        slides, add, _ = collector()
        append_context(add, {"presentation_context": {"tests": {"passed": 315, "subtests": 53, "scope": "fixture only"},
            "pending_rounds": [{"round": 20, "status": "planejada", "note_pt": "Sem novo score."}]}})
        self.assertIn("R20: planejada", slides[0].points[0])
        self.assertIn("não atribuímos score", slides[0].narration.lower())
        self.assertIn("315", slides[1].points[0])
        self.assertIn("53", slides[1].points[0])

    def test_new_complex_concepts_have_immediate_analogies_and_valid_formulas(self):
        slides, add, pair = collector()
        append_feature_pairs(pair, [own()])
        neural = {"status": "complete", "model": "tabm", "versions": {"tabm": "0.0.3"}, "model_details": {
            "architecture": {"k": 16, "n_blocks": 2, "d_block": 256, "dropout": 0.1, "num_embeddings": {
                "name": "PiecewiseLinearEmbeddings", "version": "B", "activation": False,
                "d_embedding": 12, "n_bins": 48}}}}
        append_tabm_embedding_pair(pair, [neural])
        append_tabm_backbone_pair(pair, [neural])
        append_tabm_pair(pair, [neural])
        ensure_pairs(slides)
        self.assertEqual(len(slides), 10)
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
        tabm.update(model="tabm", run_name="round8-tabm", versions={"tabm": "0.0.3"})
        tabm["model_details"] = {"architecture": {"k": 16, "n_blocks": 2, "d_block": 256, "dropout": 0.1,
            "num_embeddings": {"name": "PiecewiseLinearEmbeddings", "version": "B",
                "activation": False, "d_embedding": 12, "n_bins": 48}}}
        mlp["model_details"]["external_components"] = [{"kind": "gpt2_target_first",
            "trained_by": "Fixture author", "training_code_url": "https://example.test/code",
            "training_code_commit": "fixture-only", "code_license": "MIT",
            "predictions_source_url": "https://example.test/predictions", "predictions_license": "CC BY 4.0"}]
        glm = copy.deepcopy(mlp)
        glm.update(model="glm", run_name="glm")
        glm["model_details"] = {"C": 0.3}
        slides = create_slides(evidence, [mlp, tabm, glm])
        root = next(parent for parent in Path(__file__).resolve().parents
                    if (parent / "src/fraudlab").is_dir())
        insert_code_slides(slides, root, required=True)
        ensure_pairs(slides)
        titles = [slide.title for slide in slides]
        proposal = titles.index("A proposta: comparar, combinar e comprovar")
        formal = titles.index("Formalização: aprender uma função de risco")
        one_hot = titles.index("One-hot: uma posição para cada categoria")
        glm_formula = titles.index("GLM: combinar logits e categorias originais")
        embedding = titles.index("TabM: representar posições entre cortes do treino")
        backbone = titles.index("TabM: uma matriz comum, ajustes de cada membro")
        ensemble = titles.index("TabM: várias previsões com pesos compartilhados")
        external = titles.index("GPT-2 externo: autoria e responsabilidade")
        experiments = next(index for index, slide in enumerate(slides) if slide.chapter.startswith("07 /"))
        code = next(index for index, slide in enumerate(slides) if slide.code)
        ranking = titles.index("Submissão e posição observadas")
        self.assertEqual(proposal, 2)
        self.assertEqual(slides[proposal].kind, "explain")
        self.assertEqual(formal, proposal + 1)
        self.assertLess(one_hot, glm_formula)
        self.assertLess(embedding, backbone)
        self.assertLess(backbone, ensemble)
        self.assertLess(ensemble, external)
        self.assertLess(external, experiments)
        self.assertLess(experiments, code)
        self.assertLess(code, ranking)
        self.assertTrue(all(index < code for index, slide in enumerate(slides)
                            if slide.kind == "technical" and not slide.code))

    def test_proposal_names_only_positive_selected_contributions_and_credits(self):
        slides, add, _ = collector()
        selected = {"model": "blend", "run_name": "selected", "status": "complete",
                    "model_details": {"sources": ["runs/offset", "runs/lgb", "runs/external", "runs/unused"],
                                      "weights": [0.5625, 0.1875, 0.25, 0]}}
        members = [{"model": model, "run_name": key, "status": "complete"}
                   for key, model in [("offset", "offset"), ("lgb", "lightgbm"),
                                      ("external", "external_gpt2"), ("unused", "must_not_appear")]]
        members += [{"model": "mlp", "status": "complete"}, {"model": "tabm", "status": "complete"}]
        append_proposal(add, [selected, *members], [{"trained_by": "Fixture author"}], number)
        point = slides[0].points[0]
        for contribution in ["56,25% GLM com correções de árvores", "18,75% LightGBM próprio", "25,00% GPT-2 externo"]:
            self.assertIn(contribution, point)
        self.assertNotIn("unused", point)
        self.assertNotIn("must_not_appear", point)
        self.assertIn("treinadas e comparadas", slides[0].narration)
        self.assertIn("Fixture author", slides[0].narration)

    def test_tabm_backbone_keeps_member_bias_outside_scaling_and_head_separate(self):
        slides, _, pair = collector()
        run = {"model": "tabm", "status": "complete", "versions": {"tabm": "0.0.3"},
               "model_details": {"architecture": {"dropout": 0.1}}}
        append_tabm_backbone_pair(pair, [run])
        ensure_pairs(slides)
        self.assertIn(r"W^{\top})\odot s_m+b_m", slides[0].formula)
        self.assertIn("um viés próprio b", slides[0].narration)
        self.assertIn("pesos e viés separados para cada membro", slides[0].narration)
        self.assertIn("ela não compartilha a matriz W", slides[0].narration)
        run["versions"]["tabm"] = "unverified"
        before = len(slides)
        append_tabm_backbone_pair(pair, [run])
        self.assertEqual(len(slides), before)

    def test_mlp_te_maps_numeric_quantiles_and_target_logit_standard_scaling(self):
        from .test_presentation import fixture
        evidence, mlp = fixture()
        mlp["model_details"].update(numeric_transform="quantile")
        mlp["model_details"]["architecture"].update(categorical_transform="target",
            raw_numeric_features=2, numeric_features=7)
        slides = create_slides(evidence, [mlp])
        by_title = {slide.title: slide for slide in slides}
        architecture = by_title["MLP-TE: a arquitetura treinada"]
        self.assertIn("2 entradas numéricas usam quantis", architecture.narration)
        self.assertIn("5 logits do target encoding recebem escalonamento padrão", architecture.narration)
        standard = by_title["Escalonamento padrão: onde foi usado"]
        self.assertIn("atua nos logits de categorias", " ".join(standard.points))
        quantile = by_title["Quantis: preparação numérica da MLP-TE"]
        self.assertIn("ausências preenchidas pela mediana do treino", quantile.narration)
        mlp["model_details"]["numeric_transform"] = "standard"
        changed = create_slides(evidence, [mlp])
        self.assertNotIn("Quantis: preparação numérica da MLP-TE", [slide.title for slide in changed])

    def test_final_spoken_score_and_year_preserve_visible_values(self):
        from .test_presentation import fixture
        evidence, mlp = fixture()
        evidence["results"]["public_auc"] = 0.95134
        slides = create_slides(evidence, [mlp])
        final = next(slide for slide in slides if slide.title == "Submissão e posição observadas")
        self.assertIn("AUC pública: 0,95134", final.points)
        self.assertIn("zero vírgula nove cinco um três quatro", final.narration)
        self.assertIn("dois mil e vinte e seis", final.narration)
        self.assertNotIn("2026", final.narration)
        self.assertTrue(any("2026" in point for point in final.points))

    def test_year_normalization_also_handles_dated_context_without_changing_points(self):
        from .test_presentation import fixture
        evidence, mlp = fixture()
        evidence["limitations"] = ["Snapshot de 17/11/2026; relatório publicado em 2026."]
        slide = next(item for item in create_slides(evidence, [mlp]) if item.title == "O que a evidência permite afirmar")
        self.assertIn("17/11/2026", slide.points[0])
        self.assertIn("17 de novembro de dois mil e vinte e seis", slide.narration)
        self.assertNotIn("2026", slide.narration)
        self.assertEqual(spoken_year_2026(slide.narration), slide.narration)

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
