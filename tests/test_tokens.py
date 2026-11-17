"""Complete-number keys: manufactured values and a synthetic train/refit contract."""

import json
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tiktoken", reason="Install the optional bpe extra.")

from fraudlab.__main__ import main
from fraudlab.bpe import manifest as old_manifest
from fraudlab.data import validate_submission
from fraudlab.experiment import source_provenance
from fraudlab.features import CONTINUOUS, transform_features
from fraudlab.holdout import evaluate
from fraudlab.pooled import canonical_decimal
from fraudlab.refit import refit
from fraudlab.tokens import COLUMNS, SUFFIXES, manifest, token_keys


def frame(values):
    return pd.DataFrame({name: values for name in COLUMNS})


def test_complete_float_uses_official_token_ids_and_retains_fractional_cents():
    keys = token_keys(frame([1269.35, 9.982, 9.98, 0.0]))
    assert keys.iloc[0, :4].tolist() == ["1105", "3388", "2327", "4"]
    assert keys.iloc[1, :4].tolist() == ["860", "13", "17", "4"]
    assert keys.iloc[2, :4].tolist() == ["860", "13", "4089", "3"]
    assert keys.iloc[3, :4].tolist() == ["657", "13", "15", "3"]
    assert keys.shape == (4, 42)
    assert COLUMNS == CONTINUOUS
    assert all(dtype == object for dtype in keys.dtypes)


def test_canonical_text_is_shared_with_pooling_and_does_not_round_cent_carry():
    assert canonical_decimal(1e-7) == "0.0000001"
    assert canonical_decimal(61.10) == "61.1"
    assert canonical_decimal(-0.0) == "0.0"
    keys = token_keys(frame([61.1, 61.10, -0.0, 0.0, 9.999, 10.0]))
    pd.testing.assert_series_equal(keys.iloc[0], keys.iloc[1], check_names=False)
    pd.testing.assert_series_equal(keys.iloc[2], keys.iloc[3], check_names=False)
    assert keys.iloc[4, 4:].tolist()[:3] == ["0", "1:9", "1:9"]
    assert keys.iloc[5, 4:].tolist()[:3] == ["1", "2:1", "2:10"]


def test_integer_views_distinguish_digit_length_and_preserve_input_index():
    source = frame([0.123, 9.1, 99.1, 1234.56]).set_axis([8, 8, 3, 101])
    keys = token_keys(source)
    assert keys.index.tolist() == source.index.tolist()
    assert keys.iloc[:, 4].tolist() == ["0", "0", "9", "123"]
    assert keys.iloc[:, 5].tolist() == ["1:0", "1:9", "2:9", "4:1"]
    assert keys.iloc[:, 6].tolist() == ["1:0", "1:9", "2:99", "4:12"]


@pytest.mark.parametrize(
    "value", [-1.0, np.nan, np.inf, -np.inf, 1e15, "9.982", True, 1 + 2j]
)
def test_invalid_or_inherited_cent_overflow_inputs_are_rejected(value):
    with pytest.raises(ValueError):
        token_keys(frame([value]))


def test_keys_ignore_labels_ids_and_other_columns_and_do_not_mutate_inputs():
    source = frame([1.1, 2.2]).set_axis([10, 99])
    source["id"], source["is_fraud"] = [0, 1], [1, 0]
    before = source.copy(deep=True)
    keys = token_keys(source)
    changed = source.assign(id=[99, 0], is_fraud=[0, 1], ignored=["a", "b"])
    pd.testing.assert_frame_equal(token_keys(changed), keys)
    pd.testing.assert_frame_equal(source, before)
    excluded = token_keys(source, excluded=(COLUMNS[0],))
    assert excluded.shape == (2, 35)
    assert not any(name.startswith(COLUMNS[0]) for name in excluded)
    with pytest.raises(ValueError, match="Missing full-token"):
        token_keys(source.drop(columns=COLUMNS[0]))


def test_empty_frames_exclusions_and_disjoint_batches_keep_the_same_contract():
    empty = frame(np.asarray([], dtype=np.float64))
    empty.index = pd.Index([], name="original_row")
    result = token_keys(empty)
    assert result.shape == (0, 42)
    assert result.index.equals(empty.index)
    assert result.index.name == "original_row"
    excluded = token_keys(pd.DataFrame(index=[7, 12]), excluded=COLUMNS)
    assert excluded.shape == (2, 0) and excluded.index.tolist() == [7, 12]
    source = frame([9.98, 61.1, 9.982]).set_axis([17, 18, 19])
    pd.testing.assert_frame_equal(
        token_keys(source.iloc[[2, 0]]), token_keys(source).iloc[[2, 0]]
    )


def test_token_digits_adds_exactly_42_keys_and_preserves_value_digits_and_old_bpe():
    source = frame([9.982, 1269.35])
    source["id"], source["is_fraud"] = [1, 2], [0, 1]
    base = transform_features(source, "value_digits")
    old = transform_features(source, "bpe_digits")
    extended = transform_features(source, "token_digits")
    pd.testing.assert_frame_equal(extended[base.columns], base)
    pd.testing.assert_frame_equal(transform_features(source, "bpe_digits"), old)
    assert set(extended) - set(base) == set(token_keys(source))
    assert len(extended.columns) == len(base.columns) + 42


def test_provenance_pins_new_representation_without_replacing_old_bpe_manifest():
    observed = source_provenance("token_digits")["feature_encoding"]
    assert observed == manifest()
    assert observed["output_count"] == len(COLUMNS) * len(SUFFIXES) == 42
    assert observed["encoding_files"] == old_manifest()["encoding_files"]
    assert source_provenance("bpe_digits")["feature_encoding"] == old_manifest()
    assert "fractional_cents" in observed and observed["csv_lexemes"].startswith(
        "not preserved"
    )


def test_synthetic_cli_train_holdout_refit_records_token_manifest(
    tmp_path, monkeypatch
):
    rng = np.random.default_rng(73)
    data = tmp_path / "data"
    data.mkdir()
    source = pd.DataFrame({name: rng.uniform(0, 300, 180).round(3) for name in COLUMNS})
    source["id"] = np.arange(180)
    training, test = source.iloc[:160].copy(), source.iloc[160:].copy()
    training["is_fraud"] = np.tile([0, 1], 80)
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
            "token",
            "--model",
            "glm",
            "--feature-set",
            "token_digits",
            "--folds",
            "2",
            "--holdout-fraction",
            "0.2",
            "--threads",
            "1",
            "--validation-note",
            "Manufactured rows, contract test only.",
        ],
    )
    main()
    run = output / "token"
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["feature_encoding"] == manifest()
    assert metrics["versions"]["tiktoken"] == "0.14.0"
    assert "tokens.py" in metrics["source_hashes"]
    report = tmp_path / "evaluation.json"
    evaluate([run], [1], report, "cpu")
    final = refit(run, report, output, "refit")
    assert final["feature_encoding"] == manifest()
    assert final["roc_auc"] is None
    validate_submission(pd.read_csv(output / "refit/submission.csv"), sample)
