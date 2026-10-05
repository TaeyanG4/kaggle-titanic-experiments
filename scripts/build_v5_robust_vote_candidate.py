"""Build the seed-robust Titanic v5 voting candidate.

Members:
  - v4b champion
  - RuleFit
  - MLP-PLR seed ensemble (mean probability of seeds 42, 142, 242)

The final label is a 2-of-3 hard vote at threshold 0.5.
No Kaggle submission is performed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


BASE_DIR = Path(__file__).resolve().parents[1]
V5_DIR = BASE_DIR / "exports" / "v5"
SUB_DIR = BASE_DIR / "submissions"


def main() -> None:
    oof = pd.read_csv(V5_DIR / "model_zoo_oof.csv")
    test = pd.read_csv(V5_DIR / "model_zoo_test.csv")
    seed_oof = pd.read_csv(V5_DIR / "seed_probes" / "MLP_PLR_seed_mean__oof.csv")
    seed_test = pd.read_csv(V5_DIR / "seed_probes" / "MLP_PLR_seed_mean__test.csv")

    if not oof["PassengerId"].equals(seed_oof["PassengerId"]):
        raise ValueError("OOF PassengerId mismatch.")
    if not test["PassengerId"].equals(seed_test["PassengerId"]):
        raise ValueError("Test PassengerId mismatch.")

    y = oof["Survived"].astype(int).to_numpy()
    vote_matrix = np.column_stack(
        [
            (oof["v4b__Champion"].to_numpy() > 0.5).astype(int),
            (oof["RuleFit"].to_numpy() > 0.5).astype(int),
            (seed_oof["probability"].to_numpy() > 0.5).astype(int),
        ]
    )
    pred = (vote_matrix.sum(axis=1) >= 2).astype(int)
    vote_fraction = vote_matrix.mean(axis=1)

    test_vote_matrix = np.column_stack(
        [
            (test["v4b__Champion"].to_numpy() > 0.5).astype(int),
            (test["RuleFit"].to_numpy() > 0.5).astype(int),
            (seed_test["probability"].to_numpy() > 0.5).astype(int),
        ]
    )
    test_pred = (test_vote_matrix.sum(axis=1) >= 2).astype(int)

    rows = []
    for fold in sorted(oof["fold"].unique()):
        mask = oof["fold"] == fold
        rows.append(
            {
                "fold": int(fold),
                "accuracy": accuracy_score(y[mask], pred[mask]),
                "vote_fraction_auc": roc_auc_score(y[mask], vote_fraction[mask]),
            }
        )
    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(V5_DIR / "robust_vote_fold_metrics.csv", index=False)

    submission = pd.DataFrame(
        {
            "PassengerId": test["PassengerId"].astype(int),
            "Survived": test_pred,
        }
    )
    path = SUB_DIR / "submission_v5_robust_vote.csv"
    submission.to_csv(path, index=False)

    comparison_rows = []
    for label, ref_path in [
        ("v4b", SUB_DIR / "submission_v4_score_0.79425.csv"),
        ("v5_seed42_vote", SUB_DIR / "submission_v5_hard_vote.csv"),
        ("v5_mlp_plr", SUB_DIR / "submission_v5.csv"),
    ]:
        if not ref_path.exists():
            continue
        ref = pd.read_csv(ref_path)
        merged = submission.merge(ref, on="PassengerId", suffixes=("_robust", "_ref"))
        comparison_rows.append(
            {
                "reference": label,
                "different_predictions": int(
                    (merged["Survived_robust"] != merged["Survived_ref"]).sum()
                ),
                "robust_positive_predictions": int(test_pred.sum()),
                "reference_positive_predictions": int(ref["Survived"].sum()),
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(V5_DIR / "robust_vote_submission_comparison.csv", index=False)

    metadata = {
        "version": "v5_robust_vote",
        "members": [
            "v4b__Champion",
            "RuleFit",
            "MLP_PLR mean probability over seeds 42/142/242",
        ],
        "oof_accuracy": float(accuracy_score(y, pred)),
        "vote_fraction_roc_auc": float(roc_auc_score(y, vote_fraction)),
        "fold_accuracy_std": float(fold_df["accuracy"].std(ddof=0)),
        "test_positive_predictions": int(test_pred.sum()),
        "status": "local_candidate_not_submitted",
    }
    with (V5_DIR / "robust_vote_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("=== v5 robust hard vote ===")
    print(f"OOF Accuracy: {metadata['oof_accuracy']:.5f}")
    print(f"Vote-fraction ROC-AUC: {metadata['vote_fraction_roc_auc']:.5f}")
    print(f"Fold Accuracy std: {metadata['fold_accuracy_std']:.5f}")
    print(f"Test positives: {metadata['test_positive_predictions']} / {len(test_pred)}")
    print(fold_df.to_string(index=False, float_format=lambda x: f'{x:.5f}'))
    if len(comparison):
        print()
        print(comparison.to_string(index=False))
    print(f"Submission: {path}")
    print("Kaggle submission: NOT performed.")


if __name__ == "__main__":
    main()
