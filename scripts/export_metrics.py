"""Export aggregate experiment evidence; never export rows or checkpoints."""

import argparse
import json
from pathlib import Path

FIELDS = (
    "status",
    "model",
    "phase",
    "selection",
    "evaluation_scope",
    "external_components",
    "roc_auc",
    "fold_scores",
    "fold_std",
    "rows",
    "features",
    "feature_encoding",
    "target",
    "seed",
    "validation",
    "elapsed_seconds",
    "versions",
    "data_hashes",
    "source_hashes",
    "model_details",
    "git_revision",
    "git_dirty",
    "created_at",
    "submission_sha256",
    "limitations",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("artifacts/runs"))
    parser.add_argument(
        "--output", type=Path, default=Path("evidence/experiments.json")
    )
    args = parser.parse_args()
    experiments = []
    for path in sorted(args.runs.glob("*/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        if metrics.get("status") != "complete":
            continue
        experiments.append(
            {
                "name": path.parent.name,
                **{key: metrics[key] for key in FIELDS if key in metrics},
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "scope": "Observed local experiments; not Kaggle scores",
                "experiments": experiments,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Exported {len(experiments)} aggregate experiments to {args.output}")


if __name__ == "__main__":
    main()
