"""Smooth numeric effects remain training-local and survive checkpoint reload."""

import joblib
import numpy as np
import pandas as pd
import pytest

from fraudlab.__main__ import parser, validate_training_config
from fraudlab.models import fit_glm, predict_saved


def test_spline_knots_ignore_validation_extremes_and_reload_exactly(tmp_path):
    values = np.linspace(-2, 2, 240)
    frame = pd.DataFrame({"amount_usd": values, "category": np.tile(["a", "b"], 120)})
    y = (np.abs(values) > 1).astype(int)
    outside = pd.DataFrame({"amount_usd": [1000.0, -1000.0], "category": ["new", "a"]})
    config = {
        "logistic_c": 0.3,
        "target_smoothing": 50.0,
        "threads": 1,
        "glm_spline_knots": 8,
    }
    path = tmp_path / "fold-0"
    valid, test, details = fit_glm(frame, y, outside, outside, config, 42, path)
    assert np.isfinite(valid).all() and np.logical_and(valid >= 0, valid <= 1).all()
    assert details["numeric_splines"]["n_knots"] == 8
    saved = joblib.load(path.with_suffix(".joblib"))
    spline = (
        saved.named_steps["preprocessing"]
        .named_transformers_["numeric_splines"]
        .named_steps["spline"]
    )
    # Boundary knots come from training values; the cubic extension is internal.
    knots = spline.bsplines_[0].t
    assert knots[spline.degree] == values.min()
    assert knots[-spline.degree - 1] == values.max()
    np.testing.assert_allclose(
        predict_saved("glm", path, outside), test, rtol=0, atol=1e-14
    )


@pytest.mark.parametrize("knots,model", [(-1, "glm"), (1, "glm"), (8, "mlp")])
def test_invalid_spline_configuration_is_rejected(knots, model):
    config = vars(
        parser().parse_args(
            [
                "train",
                "--name",
                "test",
                "--model",
                model,
                "--validation-note",
                "synthetic contract",
                "--glm-spline-knots",
                str(knots),
            ]
        )
    )
    with pytest.raises(ValueError, match="spline"):
        validate_training_config(config)
