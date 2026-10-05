"""Build the conservative v4 blend: 90% fold-safe v1 + 10% TabICLv2.

This blend is intentionally low-dimensional and fixed. It improved OOF
Accuracy in exactly one fold while matching the other four folds, and improved
ROC-AUC in four of five folds. No model training or Kaggle submission occurs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


BASE_DIR = Path(__file__).resolve().parents[1]
V4_DIR = BASE_DIR / "exports" / "v4"
SUB_DIR = BASE_DIR / "submissions"
V1_WEIGHT = 0.90
TABICL_WEIGHT = 0.10


def main() -> None:
    oof = pd.read_csv(V4_DIR / "model_zoo_oof.csv")
    test = pd.read_csv(V4_DIR / "model_zoo_test.csv")

    y = oof["Survived"].astype(int)
    p = V1_WEIGHT * oof["v1__Ensemble"] + TABICL_WEIGHT * oof["TabICLv2"]
    test_p = V1_WEIGHT * test["v1__Ensemble"] + TABICL_WEIGHT * test["TabICLv2"]

    fold_rows = []
    for fold in sorted(oof["fold"].unique()):
        mask = oof["fold"] == fold
        fold_rows.append(
            {
                "fold": int(fold),
                "accuracy": accuracy_score(y[mask], (p[mask] > 0.5).astype(int)),
                "roc_auc": roc_auc_score(y[mask], p[mask]),
            }
        )
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(V4_DIR / "blend_90_10_fold_metrics.csv", index=False)

    submission = pd.DataFrame(
        {
            "PassengerId": test["PassengerId"].astype(int),
            "Survived": (test_p > 0.5).astype(int),
        }
    )
    path = SUB_DIR / "submission_v4_blend_90_10.csv"
    submission.to_csv(path, index=False)

    comparison_rows = []
    for label, ref_path in [
        ("v1", SUB_DIR / "submission_v1_score_0.79186.csv"),
        ("v4_tabicl", SUB_DIR / "submission_v4.csv"),
    ]:
        if not ref_path.exists():
            continue
        ref = pd.read_csv(ref_path)
        merged = submission.merge(ref, on="PassengerId", suffixes=("_blend", "_ref"))
        comparison_rows.append(
            {
                "reference": label,
                "different_predictions": int(
                    (merged["Survived_blend"] != merged["Survived_ref"]).sum()
                ),
                "blend_positive_predictions": int(submission["Survived"].sum()),
                "reference_positive_predictions": int(ref["Survived"].sum()),
            }
        )
    pd.DataFrame(comparison_rows).to_csv(
        V4_DIR / "blend_90_10_submission_comparison.csv", index=False
    )

    metadata = {
        "v1_weight": V1_WEIGHT,
        "tabicl_weight": TABICL_WEIGHT,
        "threshold": 0.5,
        "oof_accuracy": float(accuracy_score(y, p > 0.5)),
        "oof_roc_auc": float(roc_auc_score(y, p)),
        "test_positive_predictions": int(submission["Survived"].sum()),
        "fold_metrics": fold_rows,
        "note": "Conservative fixed blend built from saved fold-safe predictions.",
    }
    with (V4_DIR / "blend_90_10_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("=== v4 conservative blend ===")
    print(f"OOF Accuracy: {metadata['oof_accuracy']:.5f}")
    print(f"OOF ROC-AUC : {metadata['oof_roc_auc']:.5f}")
    print(f"Test positives: {metadata['test_positive_predictions']} / {len(submission)}")
    print(f"Submission: {path}")
    if comparison_rows:
        print(pd.DataFrame(comparison_rows).to_string(index=False))


if __name__ == "__main__":
    main()
