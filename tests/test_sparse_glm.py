"""Toy fits verify sparse GLM contracts, never competition performance."""

import json
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning

from fraudlab.__main__ import main, parser, validate_training_config
from fraudlab.models import fit_glm, make_glm_pipeline, predict_saved


@pytest.fixture
def config():
    return {
        "logistic_c": 0.3,
        "target_smoothing": 50.0,
        "glm_inner_folds": 5,
        "glm_value_one_hot": True,
        "threads": 1,
    }


@pytest.fixture
def toy():
    rng = np.random.default_rng(121)
    frame = pd.DataFrame(
        {
            "number": rng.normal(size=120),
            "category": np.resize(["a", "b", "c", "d"], 120),
            "number__value_key": ["common-a"] * 50
            + ["common-b"] * 50
            + [f"rare-{index}" for index in range(20)],
        }
    )
    y = np.tile([0, 1], 60)
    outside = pd.DataFrame(
        {
            "number": [0.2, -0.3],
            "category": ["unknown", "a"],
            "number__value_key": ["unseen-value", "rare-0"],
        }
    )
    return frame, y, outside


def test_all_54_categorical_inputs_are_sparse_and_fold_local(config):
    frame = pd.DataFrame(
        {
            "numeric": np.linspace(-1, 1, 150),
            **{
                f"key_{index}__value_key": np.resize(["a", "b", "c"], 150)
                for index in range(54)
            },
        }
    )
    y = np.tile([0, 1], 75)
    preprocessing = make_glm_pipeline(frame, config, 42).named_steps["preprocessing"]
    design = preprocessing.fit_transform(frame, y)
    one_hot = preprocessing.named_transformers_["one_hot"]
    assert sparse.isspmatrix_csr(design)
    assert design.dtype == np.float64
    assert preprocessing.sparse_threshold == 1.0
    assert len(one_hot.categories_) == 54
    assert one_hot.feature_names_in_.tolist() == frame.columns[1:].tolist()
    assert one_hot.min_frequency == 5
    assert one_hot.sparse_output is True
    unknown = frame.iloc[:1].copy()
    unknown.loc[:, frame.columns[1:]] = "validation-only"
    transformed = preprocessing.transform(unknown)
    assert sparse.isspmatrix_csr(transformed)
    assert transformed.shape[1] == design.shape[1]
    assert transformed[:, preprocessing.output_indices_["one_hot"]].nnz == 0
    assert all("validation-only" not in values for values in one_hot.categories_)


def test_rare_values_share_training_pool_unknown_stays_zero(toy, config):
    frame, y, outside = toy
    preprocessing = make_glm_pipeline(frame, config, 42).named_steps["preprocessing"]
    preprocessing.fit_transform(frame, y)
    one_hot = preprocessing.named_transformers_["one_hot"]
    categories = frame[["category", "number__value_key"]].copy()
    a = categories.iloc[[100]].copy()
    b = a.copy()
    b["number__value_key"] = "rare-19"
    np.testing.assert_array_equal(
        one_hot.transform(a).toarray(), one_hot.transform(b).toarray()
    )
    assert len(one_hot.infrequent_categories_[1]) == 20
    assert "unseen-value" not in one_hot.categories_[1]
    encoded = one_hot.transform(outside[["category", "number__value_key"]])
    assert encoded[0].nnz == 0
    assert encoded[1].nnz == 2


def test_inner_target_encoding_still_excludes_each_unique_keys_label(config):
    frame = pd.DataFrame(
        {
            "category": np.resize(["a", "b", "c", "d"], 100),
            "number__value_key": [f"unique-{index}" for index in range(100)],
        }
    )
    y = np.tile([0, 1], 50)
    config = {**config, "target_smoothing": 0}
    preprocessing = make_glm_pipeline(frame, config, 42).named_steps["preprocessing"]
    design = preprocessing.fit_transform(frame, y)
    target_slice = preprocessing.output_indices_["target_logits"]
    np.testing.assert_allclose(design[:, target_slice].toarray()[:, 1], 0)
    encoder = preprocessing.named_transformers_["target_logits"].named_steps["target"]
    np.testing.assert_array_equal(encoder.transform(frame)[:, 1], y)
    assert encoder.cv == 5


def test_sparse_glm_reload_and_metadata(toy, config, tmp_path):
    frame, y, outside = toy
    path = tmp_path / "fold-0"
    validation, test, details = fit_glm(frame, y, outside, outside, config, 42, path)
    saved = joblib.load(path.with_suffix(".joblib"))
    assert sparse.isspmatrix_csr(saved.named_steps["preprocessing"].transform(outside))
    np.testing.assert_allclose(validation, test, rtol=0, atol=0)
    np.testing.assert_allclose(
        predict_saved("glm", path, outside), test, rtol=0, atol=1e-14
    )
    assert details["solver"] == "lbfgs"
    assert details["max_iter"] == 500
    assert details["tol"] == 1e-7
    assert details["converged"] is True
    assert 0 < details["iterations_used"] < 500
    assert details["target_encoder"]["internal_folds"] == 5
    assert details["target_encoder"]["smooth"] == 50
    assert details["value_one_hot"] == {
        "features": ["category", "number__value_key"],
        "min_frequency": 5,
        "handle_unknown": "ignore",
        "sparse_output": True,
    }
    assert np.isfinite(test).all() and ((test >= 0) & (test <= 1)).all()


def test_nonconverged_real_toy_fit_warns_and_rejects_checkpoint(
    toy, config, tmp_path, monkeypatch
):
    import fraudlab.models as models

    frame, y, outside = toy
    original = models.make_glm_pipeline

    def one_iteration(training, config, seed):
        pipeline = original(training, config, seed)
        pipeline.set_params(classifier__max_iter=1)
        return pipeline

    monkeypatch.setattr(models, "make_glm_pipeline", one_iteration)
    path = tmp_path / "invalid"
    with pytest.warns(ConvergenceWarning):
        with pytest.raises(
            RuntimeError, match="did not converge; iterations_used=1, max_iter=1"
        ):
            fit_glm(frame, y, outside, outside, config, 42, path)
    assert not path.with_suffix(".joblib").exists()


def test_option_off_preserves_original_dense_nominal_glm(toy, config):
    frame, y, _ = toy
    baseline = make_glm_pipeline(frame, {**config, "glm_value_one_hot": False}, 42)
    classifier = baseline.named_steps["classifier"]
    preprocessing = baseline.named_steps["preprocessing"]
    design = preprocessing.fit_transform(frame, y)
    assert isinstance(design, np.ndarray)
    assert classifier.solver == "newton-cholesky"
    assert classifier.max_iter == 100
    assert classifier.tol == 1e-7
    assert preprocessing.named_transformers_["one_hot"].feature_names_in_.tolist() == [
        "category"
    ]
    assert preprocessing.named_transformers_["one_hot"].min_frequency is None


@pytest.mark.parametrize("model", ["logistic", "lightgbm", "offset"])
def test_cli_rejects_silently_ignored_option(model):
    config = vars(
        parser().parse_args(
            [
                "train",
                "--model",
                model,
                "--name",
                "toy",
                "--glm-value-one-hot",
                "--validation-note",
                "Toy only.",
            ]
        )
    )
    with pytest.raises(ValueError, match="glm_value_one_hot is supported only for GLM"):
        validate_training_config(config)


def test_cli_toy_cv_records_option_and_converged_folds(toy, tmp_path, monkeypatch):
    frame, y, outside = toy
    frame = frame.rename(columns={"number": "amount_usd"}).drop(
        columns="number__value_key"
    )
    outside = outside.rename(columns={"number": "amount_usd"}).drop(
        columns="number__value_key"
    )
    frame.insert(0, "id", np.arange(len(frame)))
    outside.insert(0, "id", np.arange(1000, 1000 + len(outside)))
    frame["is_fraud"] = y
    data = tmp_path / "data"
    data.mkdir()
    frame.to_csv(data / "train.csv", index=False)
    outside.to_csv(data / "test.csv", index=False)
    pd.DataFrame({"id": outside.id, "is_fraud": 0}).to_csv(
        data / "sample_submission.csv", index=False
    )
    output = tmp_path / "runs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fraudlab",
            "train",
            "--model",
            "glm",
            "--name",
            "toy-sparse",
            "--data",
            str(data),
            "--output",
            str(output),
            "--feature-set",
            "value_digits",
            "--glm-value-one-hot",
            "--glm-inner-folds",
            "5",
            "--logistic-c",
            "0.3",
            "--target-smoothing",
            "50",
            "--folds",
            "5",
            "--seed",
            "42",
            "--threads",
            "1",
            "--phase",
            "exploratory_full_data_cv",
            "--validation-note",
            "Manufactured toy rows; no competition inference.",
        ],
    )
    main()
    run = output / "toy-sparse"
    config = json.loads((run / "config.json").read_text())
    metrics = json.loads((run / "metrics.json").read_text())
    assert config["glm_value_one_hot"] is True
    assert config["glm_frequency"] is False and config["glm_spline_knots"] == 0
    assert metrics["rows"]["oof"] == 120
    assert metrics["validation"]["independent_holdout"] is False
    assert metrics["validation"]["early_stopping_uses_validation"] is False
    assert len(metrics["fold_scores"]) == 5
    assert all(
        fold["converged"]
        and fold["solver"] == "lbfgs"
        and fold["value_one_hot"] is not None
        for fold in metrics["model_details"]["folds"]
    )
    assert all(
        fold["target_encoder"]["internal_folds"] == 5
        and fold["target_encoder"]["smooth"] == 50
        for fold in metrics["model_details"]["folds"]
    )
    oof = pd.read_csv(run / "oof.csv")
    assert oof.id.is_unique and oof.prediction.between(0, 1).all()
    assert set(oof.fold) == set(range(5))
