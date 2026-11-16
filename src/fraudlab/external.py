"""Import one explicitly approved public prediction release; never execute its code."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .data import fingerprint, load_data, submission, write_json
from .experiment import create_run, source_provenance, versions
from .models import feature_types
from .splits import make_splits

MANIFEST_SHA256 = "0083d23b46f71311ceeed66874b093ec927d25d98b37f6d2e78b1de17f0f22d0"
DATASET = "https://www.kaggle.com/datasets/masayakawamata/s6e11-gpt2-target-first-oof"
CODE = "https://github.com/Matapanino/s6e11-gpt2-target-first"
CODE_REVISION = "8446fb73c11583df4f7006aa08fa34472249b318"


def import_public_gpt2(source: Path, data: Path, output: Path, name: str) -> dict:
    provenance = source_provenance()
    if fingerprint(source / "manifest.json") != MANIFEST_SHA256:
        raise ValueError(
            "Public manifest does not match the reviewed, approved release."
        )
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    hashes = {
        filename: fingerprint(data / filename)
        for filename in ("train.csv", "test.csv", "sample_submission.csv")
    }
    if (
        hashes["train.csv"] != manifest["inputs"]["train_csv"]["sha256"]
        or hashes["test.csv"] != manifest["inputs"]["test_csv"]["sha256"]
    ):
        raise ValueError(
            "Public arrays are not aligned with these exact train/test CSV bytes."
        )
    expected = {
        **manifest["sha256"],
        "fold_ids.npy": manifest["inputs"]["fold_ids"]["sha256"],
    }
    arrays = {}
    for filename in ("oof.npy", "test.npy", "fold_ids.npy"):
        if fingerprint(source / filename) != expected[filename]:
            raise ValueError(f"Public array hash mismatch: {filename}")
        arrays[filename] = np.load(source / filename, allow_pickle=False)
    train, test, sample = load_data(data)
    for filename, rows in (("oof.npy", len(train)), ("test.npy", len(test))):
        array = arrays[filename]
        if (
            array.shape != (rows,)
            or array.dtype != np.float64
            or not np.isfinite(array).all()
        ):
            raise ValueError(
                f"Expected finite float64 logits in original CSV row order: {filename}"
            )
    fold_ids = arrays["fold_ids.npy"]
    if fold_ids.shape != (len(train),) or fold_ids.dtype != np.int8:
        raise ValueError("Expected int8 fold IDs, one per original training row.")
    expected_folds = np.full(len(train), -1, dtype=np.int8)
    for fold, (_, validation) in enumerate(
        make_splits(train, "is_fraud", "stratified", 5, 42)
    ):
        expected_folds[validation] = fold
    if not np.array_equal(fold_ids, expected_folds):
        raise ValueError(
            "Public folds differ from exact StratifiedKFold(5, shuffle=True, seed42)."
        )
    # Stable sigmoid: the approved arrays contain log-odds, not probabilities.
    oof_probability = np.exp(-np.logaddexp(0.0, -arrays["oof.npy"]))
    test_probability = np.exp(-np.logaddexp(0.0, -arrays["test.npy"]))
    score = float(roc_auc_score(train["is_fraud"], oof_probability))
    if abs(score - manifest["auc"]["pooled"]) > 1e-10:
        raise ValueError(
            "Reproduced public OOF AUC differs from the approved manifest."
        )
    fold_scores = [
        float(
            roc_auc_score(
                train.loc[fold_ids == fold, "is_fraud"],
                oof_probability[fold_ids == fold],
            )
        )
        for fold in range(5)
    ]
    result = submission(test["id"], test_probability, sample, "is_fraud")
    run = create_run(output, name)
    pd.DataFrame(
        {
            "id": train["id"],
            "fold": fold_ids,
            "target": train["is_fraud"],
            "prediction": oof_probability,
        }
    ).to_csv(run / "oof.csv", index=False)
    result.to_csv(run / "predictions.csv", index=False)
    result.to_csv(run / "submission.csv", index=False)
    raw_features = train.drop(columns=["id", "is_fraud"])
    numeric, categorical = feature_types(raw_features)
    component = {
        "name": "GPT-2 target-first",
        "kind": "gpt2_target_first",
        "source_type": "external_predictions",
        "trained_by": "Masaya Kawamata",
        "model_architecture": "GPT-2 target-first classifier",
        "training_code_url": CODE,
        "training_code_commit": CODE_REVISION,
        "code_license": "MIT",
        "predictions_source_url": DATASET,
        "predictions_source_version": 2,
        "predictions_download_ref": "masayakawamata/s6e11-gpt2-target-first-oof/2",
        "predictions_license": "CC BY 4.0",
        "data_license": "CC BY 4.0 (prediction dataset)",
        "method_credits": [
            "Chris Deotte: target-first GPT-2 method",
            "Borisov et al.: GReaT / Language Models are Realistic Tabular Data Generators",
            "uditjain13: original Credit Card Fraud Detection 2026 dataset",
        ],
        "training_executed_locally": False,
        "trained_by_us": False,
        "oof_auc": score,
        "validation": {
            "method": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": True,
            "seed": 42,
        },
        "inference": {
            "method": "target-first label likelihood ratio",
            "target_first": True,
            "likelihood_ratio": True,
            "prior": manifest["settings"]["prior"],
            "orders": manifest["settings"]["orders"],
            "order_aggregation": manifest["settings"]["combine"],
            "test_aggregation": "mean of fold log-odds",
            "test_ensemble_folds": 5,
            "source_test_aggregation": manifest["settings"]["test"],
            "import_transform": "sigmoid(log_odds), float64",
        },
    }
    metrics = {
        "status": "complete",
        "model": "external_gpt2",
        "external_components": [component],
        "phase": "exploratory_full_data_cv",
        "roc_auc": score,
        "fold_scores": fold_scores,
        "fold_std": float(np.std(fold_scores)),
        "rows": {"train": len(train), "test": len(test), "oof": len(train)},
        "features": {
            "count": raw_features.shape[1],
            "numeric": len(numeric),
            "categorical": len(categorical),
            "names": raw_features.columns.tolist(),
            "excluded": ["id", "is_fraud"],
        },
        "target": {
            "name": "is_fraud",
            "positive_rate": float(train["is_fraud"].mean()),
        },
        "seed": 42,
        "validation": {
            "method": "stratified",
            "n_splits": 5,
            "note": "Exploratory full-data CV with externally trained predictions; all labels, including the historical holdout, were used in fold training. The old holdout is not independent for this stage.",
            "group_column": None,
            "time_column": None,
            "holdout_fraction": 0.0,
            "holdout_seed": None,
            "independent_holdout": False,
            "early_stopping_uses_validation": False,
        },
        "model_details": {
            "external_components": [component],
            "training_executed_locally": False,
        },
        "external_provenance": {
            "manifest_sha256": MANIFEST_SHA256,
            "array_sha256": expected,
            "fold_ids_reproduced": True,
            "row_alignment": "Exact raw train/test SHA256 and original row order; no sorting by ID",
        },
        "data_hashes": hashes,
        "versions": versions(),
        **provenance,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "submission_sha256": fingerprint(run / "submission.csv"),
        "limitations": [
            "This neural component was trained by an external author, not by FraudLab. Preserve CC BY 4.0 attribution.",
            "No independent holdout is available for this exploratory full-data stage; the earlier holdout report applies only to the original protocol.",
            "Public test predictions average fold log-odds before sigmoid; OOF rows use only their excluded-fold model.",
        ],
    }
    write_json(
        run / "config.json",
        {
            "model": "external_gpt2",
            "phase": "exploratory_full_data_cv",
            "source": DATASET,
            "manifest_sha256": MANIFEST_SHA256,
            "logits_to_probabilities": "sigmoid",
        },
    )
    write_json(run / "metrics.json", metrics)
    write_json(run / "status.json", {"status": "complete"})
    return metrics
