"""Score fusion on manufactured vectors and atomic toy run artifacts only."""

import json
import sys

import numpy as np
import pandas as pd
import pytest

from fraudlab.__main__ import main
from fraudlab.experiment import blend
from fraudlab.scores import SPACES, blend_scores


def test_probability_preserves_legacy_arithmetic_and_does_not_mutate_weights():
    columns = [
        np.array([0.0, 0.13, 0.2, 0.99999, 1.0]),
        np.array([0.8, 0.4, 0.6, 0.1, 0.0]),
        np.array([0.2, 0.3, 0.7, 1.0, 0.9]),
    ]
    weights = np.array([9.0, 3.0, 4.0])
    normalized = weights / weights.sum()
    frozen = sum(weight * column for weight, column in zip(normalized, columns))
    np.testing.assert_array_equal(blend_scores(columns, weights), frozen)
    np.testing.assert_array_equal(weights, [9, 3, 4])
    equal = sum(column / 3 for column in columns)
    np.testing.assert_allclose(blend_scores(columns), equal, rtol=0, atol=1e-16)


def test_logit_saturation_is_finite_symmetric_and_probability_identity_is_unclipped():
    extreme = np.array([0.0, 1.0, 1e-300, 1 - 1e-15])
    assert np.isfinite(blend_scores([extreme], space="logit")).all()
    np.testing.assert_allclose(
        blend_scores([extreme], space="logit"),
        [1e-6, 1 - 1e-6, 1e-6, 1 - 1e-6],
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        blend_scores([extreme, 1 - extreme], space="logit"), 0.5, atol=1e-11
    )
    np.testing.assert_array_equal(blend_scores([extreme]), extreme)
    ordinary = np.array([0.02, 0.1, 0.7, 0.99])
    np.testing.assert_allclose(
        blend_scores([ordinary, ordinary], [9, 7], space="logit"),
        ordinary,
        rtol=0,
        atol=1e-15,
    )


@pytest.mark.parametrize("space", ["rank", "rank_logit"])
def test_average_ties_rank_normalization_and_monotonic_invariance(space):
    values = np.array([0.2, 0.2, 0.9, 0.1])
    expected = np.array([2.5, 2.5, 4, 1]) / 5
    observed = blend_scores([values], space=space)
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(
        blend_scores([values], space=space),
        blend_scores([values**3], space=space),
    )
    np.testing.assert_array_equal(
        blend_scores([np.ones(8) * 0.1], space=space), np.ones(8) * 0.5
    )


@pytest.mark.parametrize(
    "columns,weights",
    [
        ([], None),
        ([[]], None),
        ([[0.1], [0.2, 0.3]], None),
        ([[np.nan]], None),
        ([[1.1]], None),
        ([[-0.1]], None),
        ([[0.2]], [0]),
        ([[0.2]], [-1]),
        ([[0.2]], [np.inf]),
        ([[0.2]], [1, 2]),
        ([[[0.2]]], None),
    ],
)
def test_invalid_score_shapes_ranges_or_weights_are_rejected(columns, weights):
    with pytest.raises(ValueError):
        blend_scores(columns, weights)


@pytest.fixture
def atomic_runs(tmp_path):
    runs = [tmp_path / name for name in ("offset", "lgbm", "external")]
    oof_values = [
        [0.1, 0.2, 0.8, 0.6, 0.3, 0.7, 0.3, 0.9],
        [0.2, 0.3, 0.4, 0.9, 0.1, 0.5, 0.5, 0.8],
        [0.4, 0.2, 0.7, 0.9, 0.1, 0.6, 0.5, 0.8],
    ]
    test_values = [[0.01, 0.2, 0.9], [0.2, 0.2, 0.7], [0.3, 0.4, 0.8]]
    for index, run in enumerate(runs):
        run.mkdir()
        components = (
            []
            if index < 2
            else [
                {
                    "predictions_source_url": "https://example.test/public-predictions",
                    "predictions_source_version": 2,
                    "predictions_license": "CC BY 4.0",
                    "trained_by_us": False,
                    "inference": {"import_transform": "sigmoid(log_odds)"},
                }
            ]
        )
        metadata = {
            "status": "complete",
            "phase": "exploratory_full_data_cv",
            "model": ["offset", "lightgbm", "external_gpt2"][index],
            "data_hashes": {"train.csv": "same", "test.csv": "same"},
            "validation": {"method": "stratified", "n_splits": 2},
            "model_details": {"external_components": components},
        }
        (run / "metrics.json").write_text(json.dumps(metadata))
        pd.DataFrame(
            {
                "id": np.arange(8),
                "fold": [0] * 4 + [1] * 4,
                "target": [0, 1, 0, 1] * 2,
                "prediction": oof_values[index],
            }
        ).to_csv(run / "oof.csv", index=False)
        pd.DataFrame({"id": [10, 11, 12], "is_fraud": test_values[index]}).to_csv(
            run / "predictions.csv", index=False
        )
    return runs, oof_values, test_values


@pytest.mark.parametrize("space", SPACES)
def test_real_artifact_blend_ranks_partitions_separately_and_records_correct_space(
    tmp_path, atomic_runs, space
):
    runs, oofs, tests = atomic_runs
    weights = [0.5625, 0.1875, 0.25]
    result = blend(runs, weights, tmp_path / "runs", space, space=space)
    output = tmp_path / "runs" / space
    np.testing.assert_allclose(
        pd.read_csv(output / "oof.csv").prediction,
        blend_scores(oofs, weights, space=space),
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        pd.read_csv(output / "predictions.csv").is_fraud,
        blend_scores(tests, weights, space=space),
        rtol=0,
        atol=1e-15,
    )
    component = result["external_components"][0]
    assert component == result["model_details"]["external_components"][0]
    assert component["inference"]["import_transform"] == "sigmoid(log_odds)"
    assert component["predictions_license"] == "CC BY 4.0"
    assert component["mixture_space"] == space
    if space == "probability":
        assert component["weight"] == 0.25 and "coefficient" not in component
    else:
        assert "weight" not in component
        assert (
            component["coefficient"] == 0.25 and component["coefficient_space"] == space
        )
        assert (
            result["model_details"]["score_transform"]["input_space"] == "probability"
        )
        assert json.loads((output / "config.json").read_text())["space"] == space


def test_default_probability_artifacts_remain_byte_identical_to_explicit_probability(
    tmp_path, atomic_runs
):
    runs, _, _ = atomic_runs
    output = tmp_path / "runs"
    implicit = blend(runs, [9, 3, 4], output, "implicit")
    explicit = blend(runs, [9, 3, 4], output, "explicit", space="probability")
    for name in ("oof.csv", "predictions.csv", "submission.csv", "config.json"):
        assert (output / "implicit" / name).read_bytes() == (
            output / "explicit" / name
        ).read_bytes()
    assert implicit["model_details"] == explicit["model_details"]


def test_nested_external_nonlinear_mixtures_are_rejected_in_both_directions(
    tmp_path, atomic_runs
):
    runs, _, _ = atomic_runs
    output = tmp_path / "runs"
    blend(runs, [9, 3, 4], output, "probability")
    with pytest.raises(ValueError, match="nested external"):
        blend(
            [output / "probability", runs[0]],
            [1, 1],
            output,
            "bad-logit",
            space="logit",
        )
    blend(runs, [9, 3, 4], output, "logit", space="logit")
    with pytest.raises(ValueError, match="nested external"):
        blend([output / "logit", runs[0]], [1, 1], output, "bad-probability")
    assert (
        not (output / "bad-logit").exists()
        and not (output / "bad-probability").exists()
    )
    # Existing nested probability composition remains supported.
    result = blend([output / "probability", runs[0]], [1, 1], output, "valid")
    assert result["external_components"][0]["weight"] == 0.125


def test_nonlinear_blend_keeps_alignment_and_phase_guards(tmp_path, atomic_runs):
    runs, _, _ = atomic_runs
    oof = pd.read_csv(runs[1] / "oof.csv")
    oof["fold"] = 1 - oof["fold"]
    oof.to_csv(runs[1] / "oof.csv", index=False)
    with pytest.raises(ValueError, match="OOF rows"):
        blend(runs, None, tmp_path / "runs", "bad", space="rank")
    for run in runs:
        path = run / "metrics.json"
        metric = json.loads(path.read_text())
        metric["phase"] = "full_data_refit"
        path.write_text(json.dumps(metric))
    with pytest.raises(ValueError, match="refits support probability"):
        blend(runs, None, tmp_path / "runs", "bad-refit", space="logit")


def test_nonlinear_external_source_cannot_hide_an_internal_mixture_weight(
    tmp_path, atomic_runs
):
    runs, _, _ = atomic_runs
    path = runs[-1] / "metrics.json"
    metric = json.loads(path.read_text())
    metric["model_details"]["external_components"][0]["weight"] = 0.5
    path.write_text(json.dumps(metric))
    with pytest.raises(ValueError, match="atomic unit-weight source"):
        blend(runs, [9, 3, 4], tmp_path / "runs", "bad", space="rank_logit")
    assert not (tmp_path / "runs/bad").exists()


def test_cli_exposes_rank_space_without_changing_default(
    tmp_path, atomic_runs, monkeypatch
):
    runs, _, tests = atomic_runs
    output = tmp_path / "runs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fraudlab",
            "blend",
            "--runs",
            *map(str, runs),
            "--weights",
            "9",
            "3",
            "4",
            "--space",
            "rank",
            "--name",
            "cli",
            "--output",
            str(output),
        ],
    )
    main()
    np.testing.assert_allclose(
        pd.read_csv(output / "cli/submission.csv").is_fraud,
        blend_scores(tests, [9, 3, 4], space="rank"),
    )
