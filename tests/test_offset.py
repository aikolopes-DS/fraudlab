"""Simulations verify offset arithmetic, persistence and training TE separation."""

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

from fraudlab import models
from fraudlab.offset import fit_offset


class TrainingTransformGuard(BaseEstimator, TransformerMixin):
    """Unique keys reveal a full-map retransform; reject it if called during fit."""

    def fit_transform(self, frame, y):
        self.training_index_ = frame.index.copy()
        self.transform_calls_ = 0
        self.encoder_ = TargetEncoder(
            target_type="binary", smooth=0, cv=5, shuffle=True, random_state=42
        )
        self.encoded_training_ = self.encoder_.fit_transform(frame, y)
        return self.encoded_training_

    def transform(self, frame):
        if frame.index.equals(self.training_index_):
            raise AssertionError(
                "Training data was transformed using the full target map."
            )
        self.transform_calls_ += 1
        return self.encoder_.transform(frame)


def offset_config():
    return {
        "threads": 1,
        "iterations": 20,
        "num_leaves": 4,
        "learning_rate": 0.03,
        "l2_leaf_reg": 5.0,
        "min_child_samples": 5,
        "logistic_c": 0.3,
        "target_smoothing": 50.0,
        "glm_spline_knots": 0,
    }


def test_offset_reload_preserves_sum_of_glm_and_tree_margins(tmp_path):
    rng = np.random.default_rng(42)
    size = 400
    frame = pd.DataFrame(
        {"value": rng.normal(size=size), "category": rng.choice(["a", "b", "c"], size)}
    )
    y = (frame.value.to_numpy() ** 2 + rng.normal(scale=0.5, size=size) > 1).astype(int)
    validation = frame.iloc[300:350].copy()
    test = frame.iloc[350:].copy()
    test.loc[test.index[0], "category"] = "unseen"
    path = tmp_path / "fold-0"
    p_validation, p_test, details = fit_offset(
        frame.iloc[:300], y[:300], validation, test, offset_config(), 42, path
    )
    saved = joblib.load(path.with_suffix(".joblib"))
    np.testing.assert_allclose(saved.predict_proba(validation)[:, 1], p_validation)
    np.testing.assert_allclose(saved.predict_proba(test)[:, 1], p_test)
    encoded = saved.glm.named_steps["preprocessing"].transform(test)
    base_margin = saved.glm.named_steps["classifier"].decision_function(encoded)
    residual_margin = saved.residual.predict(encoded, raw_score=True)
    expected = np.exp(-np.logaddexp(0.0, -(base_margin + residual_margin)))
    np.testing.assert_allclose(p_test, expected)
    assert np.ptp(residual_margin) > 1e-5
    assert not np.allclose(p_test, np.exp(-np.logaddexp(0.0, -residual_margin)))
    assert np.isfinite(p_test).all() and ((p_test >= 0) & (p_test <= 1)).all()
    assert details["fixed_training"] and details["early_stopping_patience"] is None
    assert details["best_iteration"] == offset_config()["iterations"]


def test_offset_never_retransforms_training_with_full_target_encoding(
    tmp_path, monkeypatch
):
    def guarded_pipeline(training, config, seed):
        return Pipeline(
            [
                ("preprocessing", TrainingTransformGuard()),
                (
                    "classifier",
                    LogisticRegression(
                        C=config["logistic_c"], solver="newton-cholesky"
                    ),
                ),
            ]
        )

    monkeypatch.setattr(models, "make_glm_pipeline", guarded_pipeline, raising=False)
    training = pd.DataFrame({"key": [f"unique-{index}" for index in range(100)]})
    y = np.tile([0, 1], 50)
    validation = pd.DataFrame({"key": ["validation-unique"]}, index=[101])
    test = pd.DataFrame({"key": ["test-unique"]}, index=[102])
    path = tmp_path / "guarded"
    fit_offset(training, y, validation, test, offset_config(), 42, path)
    saved = joblib.load(path.with_suffix(".joblib"))
    guard = saved.glm.named_steps["preprocessing"]
    np.testing.assert_allclose(guard.encoded_training_, 0.5)
    full_map = guard.encoder_.transform(training)
    np.testing.assert_array_equal(full_map[:, 0], y)
    assert not np.array_equal(guard.encoded_training_, full_map)
    assert (
        guard.transform_calls_ == 0
    )  # Persistence precedes validation/test prediction.
