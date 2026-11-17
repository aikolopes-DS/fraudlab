"""A real fixed CatBoost fit must save usable predictions without an eval set."""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from fraudlab.models import fit_catboost


def test_fixed_catboost_without_validation_selection_can_be_reloaded(tmp_path):
    training = pd.DataFrame(
        {"value": np.linspace(-2, 2, 100), "category": ["a", "b"] * 50}
    )
    y = (training.value.to_numpy() > 0).astype(int)
    validation = pd.DataFrame({"value": [-1.0, 1.0], "category": ["a", "new"]})
    config = {
        "device": "cpu", "iterations": 5, "depth": 2, "learning_rate": 0.1,
        "l2_leaf_reg": 5, "threads": 1, "fixed_training": True,
    }
    path = tmp_path / "fixed"
    prediction, test_prediction, details = fit_catboost(
        training, y, validation, np.array([0, 1]), validation, config, 42, path
    )
    restored = CatBoostClassifier()
    restored.load_model(str(path.with_suffix(".cbm")))
    np.testing.assert_allclose(restored.predict_proba(validation)[:, 1], prediction)
    np.testing.assert_array_equal(prediction, test_prediction)
    assert np.isfinite(prediction).all()
    assert prediction[1] > prediction[0]
    assert details["best_iteration"] in (None, -1)
    assert details["trees_used"] == config["iterations"]
    assert details["early_stopping_patience"] is None
    assert details["resolved_parameters"]["iterations"] == config["iterations"]
