"""Teaching contracts using explicit fixtures, never competition results."""

import copy
from pathlib import Path
import unittest

from matplotlib.mathtext import MathTextParser

from .content import create_slides, ensure_pairs
from .build import insert_code_slides
from .realmlp import append_realmlp_pairs, realmlp_sources, resolved_realmlp
from .test_continuation import collector


def fixture():
    parameters = {
        "num_emb_type": "pbld",
        "plr_act_name": "linear",
        "plr_use_densenet": True,
        "plr_use_cos_bias": True,
        "use_parametric_act": True,
        "act": "silu",
        "ens_av_before_softmax": False,
        "opt": "adam",
        "use_ls": True,
        "use_ls_prior": False,
        "use_plr_embeddings": True,
        "lr_sched": "flat_anneal",
        "wd_sched": "cos_log_15",
        "p_drop_sched": "pow5",
        "ls_eps_sched": "sqrt_cos",
        "plr_hidden_1": 16,
        "plr_hidden_2": 8,
        "n_ens": 16,
        "hidden_sizes": [512, 256, 128],
        "n_epochs": 3,
        "embedding_size": 5,
        "max_one_hot_cat_size": 16,
        "lr": 0.073,
        "wd": 0.038,
        "p_drop": 0.05,
        "ls_eps": 0.01,
        "sq_mom": 0.988,
        "smooth_clip_max_abs_value": 3.0,
        "robust_scale_eps": 1e-30,
        "l2_normalize_eps": 1e-8,
        "tfms": [
            "one_hot",
            "median_center",
            "robust_scale",
            "smooth_clip",
            "embedding",
            "l2_normalize",
        ],
    }
    details = {
        "architecture": {
            "library": "pytabkit==1.8.0",
            "n_ens": 16,
            "hidden_dims": [512, 256, 128],
        },
        "preprocessing": {
            "columns": [f"fixture_{i}" for i in range(53)],
            "numeric_features": 20,
            "categorical_features": 33,
        },
        "refit": {"resolved_parameters": parameters, "completed_epochs": 2},
        "selected_epoch": 2,
        "config": {"inner_fraction": 0.15, "inner_seed": 42},
    }
    return {
        "model": "realmlp",
        "status": "complete",
        "roc_auc": 0.7,
        "model_details": details,
    }


class RealMLPContentTests(unittest.TestCase):
    def test_unfinished_or_absent_runs_never_become_executed_methods(self):
        run = fixture()
        run["status"] = "running"
        slides, _, pair = collector()
        append_realmlp_pairs(pair, [run])
        self.assertEqual(slides, [])
        self.assertEqual(realmlp_sources([run]), [])
        self.assertIsNone(resolved_realmlp([]))

    def test_resolved_architecture_is_required_instead_of_inferring_defaults(self):
        for key, value in (
            ("plr_act_name", "gelu"),
            ("ens_av_before_softmax", True),
            ("use_parametric_act", False),
            ("use_ls_prior", True),
        ):
            run = fixture()
            run["model_details"]["refit"]["resolved_parameters"][key] = value
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(ValueError, "configuração resolvida"),
            ):
                resolved_realmlp([run])
        run = fixture()
        del run["model_details"]["refit"]["resolved_parameters"]
        with self.assertRaisesRegex(ValueError, "configuração resolvida"):
            resolved_realmlp([run])

    def test_actual_epochs_are_required_for_every_completed_fold(self):
        run = fixture()
        second = copy.deepcopy(run["model_details"])
        second["refit"]["completed_epochs"] = 1
        run["model_details"]["folds"] = [copy.deepcopy(run["model_details"]), second]
        with self.assertRaisesRegex(ValueError, "épocas selecionadas"):
            resolved_realmlp([run])

    def test_realmlp_math_and_literal_code_stay_before_the_final_ranking(self):
        from .test_presentation import fixture as basic_fixture

        evidence, mlp = basic_fixture()
        real = fixture()
        real.update(
            data_hashes=mlp["data_hashes"],
            rows=mlp["rows"],
            validation=mlp["validation"],
        )
        slides = create_slides(evidence, [mlp, real])
        insert_code_slides(
            slides, Path(__file__).resolve().parents[1], True, False, {}, [mlp, real]
        )
        ensure_pairs(slides)
        fresh = next(
            i
            for i, slide in enumerate(slides)
            if slide.title.startswith("RealMLP: escolher E")
        )
        code = next(
            i
            for i, slide in enumerate(slides)
            if slide.code_reference.startswith("src/fraudlab/realmlp.py:")
        )
        ranking = next(
            i
            for i, slide in enumerate(slides)
            if slide.title == "Submissão e posição observadas"
        )
        self.assertLess(fresh + 1, code)
        self.assertLess(code + 1, ranking)
        self.assertIn("torch.softmax(model.predict(numbers)", slides[code].code)

    def test_new_formulas_have_immediate_daily_analogies_and_match_official_semantics(
        self,
    ):
        slides, _, pair = collector()
        append_realmlp_pairs(pair, [fixture()])
        ensure_pairs(slides)
        self.assertEqual(len(slides), 16)
        for slide in slides:
            if slide.formula:
                with self.subTest(title=slide.title):
                    MathTextParser("path").parse("$" + slide.formula + "$")
        periodic = slides[2]
        self.assertIn("7 coordenadas + valor original = 8", periodic.points[1])
        self.assertIn(r"2\pi", periodic.formula)
        self.assertIn(r"\mathbb{R}^{8}", periodic.formula)
        self.assertIn("não ficam", slides[12].points[2])
        self.assertIn("não é normalizar cada transação", slides[0].narration)
        self.assertIn("não afirmamos que ficará limitado", slides[4].narration)
        self.assertIn("pesos não são compartilhados", slides[4].points[-1])
        self.assertIn("depois da softmax", slides[6].points[1])
        self.assertNotIn(r"\frac{1}{M}", slides[8].formula)
        self.assertIn("soma essas médias", slides[8].narration)
        self.assertIn("não é comprimido", slides[10].narration)
        self.assertIn("épocas registradas por fold: [2]", slides[14].points[2])
        self.assertFalse(
            any("AUC OOF 0" in point for slide in slides for point in slide.points)
        )


if __name__ == "__main__":
    unittest.main()
