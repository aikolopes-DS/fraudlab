"""Inner encoding folds affect training maps, never the outer validation contract."""

import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from fraudlab.__main__ import main
from fraudlab.models import fit_glm, make_glm_pipeline
from fraudlab.offset import fit_offset


@pytest.mark.parametrize("inner_folds", [None, 10])
def test_inner_encoding_excludes_unique_keys_own_labels(inner_folds):
    training = pd.DataFrame({"key": [f"unique-{index}" for index in range(100)]})
    y = np.tile([0, 1], 50)
    config = {"logistic_c": 0.3, "target_smoothing": 0}
    if inner_folds is not None:
        config["glm_inner_folds"] = inner_folds
    preprocessing = make_glm_pipeline(training, config, 42).named_steps[
        "preprocessing"
    ]
    encoded = preprocessing.fit_transform(training, y)
    encoder = preprocessing.named_transformers_["target_logits"].named_steps["target"]
    assert encoder.cv == (inner_folds or 5)
    # With unique keys, honest inner maps can only return the balanced prior.
    np.testing.assert_allclose(
        encoded[:, preprocessing.output_indices_["target_logits"]], 0
    )
    # Reusing the full map would expose each example's label instead.
    np.testing.assert_array_equal(encoder.transform(training)[:, 0], y)
    unknown = pd.DataFrame({"key": ["unseen"]})
    np.testing.assert_allclose(encoder.transform(unknown), 0.5)


@pytest.mark.parametrize("model_name", ["glm", "offset"])
def test_inner10_is_recorded_and_preserved_after_reload(tmp_path, model_name):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "value": rng.normal(size=140),
            "category": np.resize(["a", "b", "c"], 140),
        }
    )
    y = np.tile([0, 1], 50)
    training = frame.iloc[:100]
    validation = frame.iloc[100:120]
    test = frame.iloc[120:].copy()
    test.loc[test.index[0], "category"] = "unseen"
    config = {
        "logistic_c": 0.3,
        "target_smoothing": 50,
        "glm_inner_folds": 10,
        "threads": 1,
        "iterations": 8,
        "num_leaves": 4,
        "learning_rate": 0.03,
        "l2_leaf_reg": 5,
        "min_child_samples": 5,
    }
    fit = fit_glm if model_name == "glm" else fit_offset
    path = tmp_path / model_name
    p_validation, p_test, details = fit(training, y, validation, test, config, 42, path)
    saved = joblib.load(path.with_suffix(".joblib"))
    glm = saved if model_name == "glm" else saved.glm
    encoder = (
        glm.named_steps["preprocessing"]
        .named_transformers_["target_logits"]
        .named_steps["target"]
    )
    assert encoder.cv == 10
    assert details["target_encoder"]["internal_folds"] == 10
    np.testing.assert_allclose(saved.predict_proba(validation)[:, 1], p_validation)
    np.testing.assert_allclose(saved.predict_proba(test)[:, 1], p_test)
    assert np.isfinite(p_test).all() and ((p_test >= 0) & (p_test <= 1)).all()


def test_inner10_cannot_be_silently_ignored_by_another_model(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fraudlab",
            "train",
            "--model",
            "lightgbm",
            "--name",
            "simulation",
            "--glm-inner-folds",
            "10",
            "--validation-note",
            "Synthetic contract test.",
        ],
    )
    with pytest.raises(SystemExit, match="10 requires GLM/offset"):
        main()
