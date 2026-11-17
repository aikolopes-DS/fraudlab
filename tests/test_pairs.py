"""Fixed domain pairs preserve row identity and never use target values."""

import numpy as np
import pandas as pd

from fraudlab.features import transform_features
from fraudlab.models import fit_glm, predict_saved


def test_pair_keys_are_target_independent_and_respect_bin_boundaries():
    frame = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "is_fraud": [0, 1, 0],
            "cvv_retry_count": [1, 1, 2],
            "auth_method": ["OTP", "OTP", "PIN"],
            "velocity_score": [4.99, 4.99, 5.0],
            "distance_from_home_km": [9.9, 9.9, 10.0],
            "txn_count_last_24h": [2, 2, 3],
            "hours_since_last_txn": [1.9, 1.9, 2.0],
        }
    )
    result = transform_features(frame, "paired_digits")
    pairs = [name for name in result if name.endswith("__pair")]
    assert len(pairs) == 4
    assert result.loc[0, pairs].equals(result.loc[1, pairs])
    assert all(result.loc[0, name] != result.loc[2, name] for name in pairs)
    assert not any("id__" in name or "is_fraud" in name for name in pairs)
    excluded = transform_features(
        frame, "paired_digits", ("id", "is_fraud", "velocity_score")
    )
    assert len([name for name in excluded if name.endswith("__pair")]) == 1


def test_paired_frequency_glm_fits_and_reloads_finite_predictions(tmp_path):
    rng = np.random.default_rng(42)
    n = 300
    frame = pd.DataFrame(
        {
            "velocity_score": rng.uniform(0, 50, n),
            "cvv_retry_count": rng.integers(0, 4, n),
            "auth_method": rng.choice(["OTP", "PIN"], n),
            "distance_from_home_km": rng.uniform(0, 50, n),
            "txn_count_last_24h": rng.integers(0, 12, n),
            "hours_since_last_txn": rng.uniform(0, 24, n),
        }
    )
    y = (
        frame.velocity_score.to_numpy() + 10 * frame.cvv_retry_count.to_numpy() > 40
    ).astype(int)
    prepared = transform_features(frame, "paired_digits")
    config = {
        "logistic_c": 0.3,
        "target_smoothing": 50.0,
        "threads": 1,
        "glm_frequency": True,
    }
    path = tmp_path / "fold"
    valid, test, details = fit_glm(
        prepared.iloc[:240],
        y[:240],
        prepared.iloc[240:270],
        prepared.iloc[270:],
        config,
        42,
        path,
    )
    assert details["category_frequency"] is not None
    assert np.isfinite(valid).all() and np.isfinite(test).all()
    np.testing.assert_allclose(
        predict_saved("glm", path, prepared.iloc[270:]), test, atol=1e-14, rtol=0
    )
