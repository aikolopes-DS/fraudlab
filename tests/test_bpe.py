"""Numeric BPE contracts and a manufactured GLM train/holdout/refit flow."""

import json
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import TargetEncoder

pytest.importorskip("tiktoken", reason="Install the optional bpe extra.")

from fraudlab.__main__ import main
from fraudlab.bpe import COLUMNS, bpe_keys, manifest
from fraudlab.data import validate_submission
from fraudlab.features import transform_features
from fraudlab.holdout import evaluate
from fraudlab.refit import refit


def frame(values):
    return pd.DataFrame({name: values for name in COLUMNS})


def test_official_gpt2_numeric_pieces_are_categorical_keys():
    keys = bpe_keys(frame([61.16, 1269.35, 3233.0, 24810.07]))
    assert keys.iloc[:, 0].tolist() == ["8454", "1105", "513", "32996"]
    assert keys.iloc[:, 1].tolist() == ["-1", "3388", "25429", "940"]
    assert keys.iloc[:, 2].tolist() == ["-1:16", "3388:35", "25429:0", "940:7"]
    assert all(dtype == object for dtype in keys.dtypes)
    assert keys.shape == (4, 6)


def test_usd_rounding_half_even_carry_and_float_precision():
    observed = bpe_keys(
        frame([0, 61.1, 61.10, 61.16 + 1e-12, 9.982, 0.005, 0.015, 9.999])
    )
    canonical = bpe_keys(frame([0, 61.1, 61.1, 61.16, 9.98, 0, 0.02, 10]))
    pd.testing.assert_frame_equal(observed, canonical)
    assert observed.iloc[0, :3].tolist() == ["657", "-1", "-1:0"]


def test_fixed_hypothesis_intentionally_discards_third_integer_token():
    keys = bpe_keys(frame([99701.39]))
    assert keys.iloc[0, :3].tolist() == ["860", "5607", "5607:39"]


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, 1e15, 1e308, "61.16", True])
def test_invalid_numeric_inputs_are_rejected(value):
    with pytest.raises(ValueError):
        bpe_keys(frame([value]))


def test_keys_preserve_index_exclusions_and_do_not_read_labels():
    source = frame([1.01, 2.02]).set_axis([12, 99])
    source["id"], source["is_fraud"] = [1, 2], [0, 1]
    before = source.copy(deep=True)
    expected = bpe_keys(source)
    source["id"], source["is_fraud"] = [99, -1], [1, 0]
    pd.testing.assert_frame_equal(bpe_keys(source), expected)
    keys = bpe_keys(before, excluded=(COLUMNS[0],))
    assert keys.shape == (2, 3) and keys.index.tolist() == [12, 99]
    pd.testing.assert_series_equal(source[COLUMNS[0]], before[COLUMNS[0]])
    with pytest.raises(ValueError, match="Missing BPE input"):
        bpe_keys(source.drop(columns=COLUMNS[0]))


def test_unseen_bpe_categories_use_target_encoder_training_prior():
    y = np.asarray([0, 0, 0, 1] * 5)
    encoder = TargetEncoder(target_type="binary", smooth=50.0, cv=5)
    encoder.fit(bpe_keys(frame([61.16] * 20)), y)
    np.testing.assert_allclose(encoder.transform(bpe_keys(frame([24810.07]))), y.mean())


def test_bpe_feature_set_adds_only_six_keys_to_value_digits():
    source = frame([61.16, 1269.35])
    base = transform_features(source, "value_digits")
    extended = transform_features(source, "bpe_digits")
    pd.testing.assert_frame_equal(extended[base.columns], base)
    assert set(extended) - set(base) == set(bpe_keys(source))
    assert len(extended.columns) == len(base.columns) + 6


def test_cli_training_and_refit_record_encoding_and_valid_predictions(
    tmp_path, monkeypatch
):
    # Manufactured rows only: verifies the real pipeline, never competition AUC.
    rng = np.random.default_rng(42)
    data = tmp_path / "data"
    data.mkdir()
    source = pd.DataFrame(
        {
            "id": np.arange(180),
            "amount_usd": np.round(rng.uniform(1, 7000, 180), 2),
            "account_balance_usd": np.round(rng.uniform(60, 90000, 180), 2),
        }
    )
    training, test = source.iloc[:160].copy(), source.iloc[160:].copy()
    training["is_fraud"] = np.tile([0, 1], 80)
    test.loc[test.index[0], "amount_usd"] = 9.982
    training.to_csv(data / "train.csv", index=False)
    test.to_csv(data / "test.csv", index=False)
    sample = pd.DataFrame({"id": test["id"].iloc[::-1], "is_fraud": 0})
    sample.to_csv(data / "sample_submission.csv", index=False)
    output = tmp_path / "runs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fraudlab",
            "train",
            "--data",
            str(data),
            "--output",
            str(output),
            "--name",
            "bpe",
            "--model",
            "glm",
            "--feature-set",
            "bpe_digits",
            "--folds",
            "2",
            "--holdout-fraction",
            "0.2",
            "--threads",
            "1",
            "--logistic-c",
            "0.3",
            "--target-smoothing",
            "50",
            "--glm-inner-folds",
            "5",
            "--validation-note",
            "Manufactured rows; tests only, no competition score.",
        ],
    )
    main()
    run = output / "bpe"
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["feature_encoding"] == manifest()
    assert metrics["versions"]["tiktoken"] == "0.14.0"
    assert "bpe.py" in metrics["source_hashes"]
    report = tmp_path / "evaluation.json"
    evaluate([run], [1], report, "cpu")
    final = refit(run, report, output, "refit")
    assert final["feature_encoding"] == manifest()
    assert final["versions"]["tiktoken"] == "0.14.0"
    assert final["roc_auc"] is None
    validate_submission(pd.read_csv(output / "refit" / "submission.csv"), sample)
