"""Fixed small trees correct a GLM margin using one cross-fitted training matrix."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


class OffsetClassifier:
    """Persist both terms: LightGBM does not add its training init_score at inference."""

    def __init__(self, glm, residual):
        self.glm = glm
        self.residual = residual

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        encoded = self.glm.named_steps["preprocessing"].transform(frame)
        margin = self.glm.named_steps["classifier"].decision_function(encoded)
        margin = margin + self.residual.predict(encoded, raw_score=True)
        probability = np.exp(-np.logaddexp(0.0, -margin))
        return np.column_stack((1 - probability, probability))


def fit_offset(
    training: pd.DataFrame,
    y: np.ndarray,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    config: dict,
    seed: int,
    path: Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    from lightgbm import LGBMClassifier

    # Local import keeps model dispatch free to import this module lazily.
    from .models import make_glm_pipeline

    glm = make_glm_pipeline(training, config, seed)
    preprocessing = glm.named_steps["preprocessing"]
    classifier = glm.named_steps["classifier"]
    with threadpool_limits(limits=config["threads"]):
        encoded_training = preprocessing.fit_transform(training, y)
        classifier.fit(encoded_training, y)
        initial_score = classifier.decision_function(encoded_training)
    parameters = {
        "n_estimators": config["iterations"],
        "num_leaves": config["num_leaves"],
        "learning_rate": config["learning_rate"],
        "reg_lambda": config["l2_leaf_reg"],
        "min_child_samples": config["min_child_samples"],
        "objective": "binary",
        "verbosity": -1,
        "random_state": seed,
        "n_jobs": config["threads"],
        "deterministic": True,
        "force_col_wise": True,
    }
    residual = LGBMClassifier(**parameters)
    residual.fit(encoded_training, y, init_score=initial_score)
    model = OffsetClassifier(glm, residual)
    joblib.dump(model, path.with_suffix(".joblib"))
    details = {
        "algorithm": "glm_logit_offset_with_lightgbm_residual",
        "prediction": "sigmoid(GLM decision_function + LightGBM raw_score)",
        "fixed_training": True,
        "early_stopping_patience": None,
        "best_iteration": int(residual.n_estimators_),
        "encoded_features": int(encoded_training.shape[1]),
        "glm": {
            "C": config["logistic_c"],
            "solver": classifier.solver,
            "iterations_used": int(classifier.n_iter_[0]),
            "spline_knots": config.get("glm_spline_knots", 0),
        },
        "target_encoder": {
            "internal_folds": 5,
            "smooth": config["target_smoothing"],
            "unknown": "training prior",
            "log_odds": True,
        },
        "residual_parameters": parameters,
        "preprocessing": (
            "Fold-local GLM preprocessing, including internally cross-fitted target "
            "logits, is fitted once. The identical encoded training matrix fits the "
            "GLM, supplies its init_score and fits residual trees. Training rows are "
            "never re-encoded with the full target map. Validation and test use "
            "training-only mappings. Both fitted models are saved together."
        ),
    }
    return (
        model.predict_proba(validation)[:, 1],
        model.predict_proba(test)[:, 1],
        details,
    )
