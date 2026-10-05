"""Build the v5 top-3 hard-vote candidate.

Members:
  1) v4b champion (90% v1 + 10% TabICLv2)
  2) MLP-PLR
  3) RuleFit

Each member first makes its own binary decision at threshold 0.5. The final
prediction is the majority vote (2 of 3). This avoids fitting blend weights on
the same OOF data.

No Kaggle submission is performed by this script.
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
MEMBERS = ["v4b__Champion", "MLP_PLR", "RuleFit"]


def majority_predictions(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    votes = np.column_stack([(frame[c].to_numpy() > 0.5).astype(int) for c in MEMBERS])
    vote_fraction = votes.mean(axis=1)
    pred = (votes.sum(axis=1) >= 2).astype(int)
    return pred, vote_fraction


def main() -> None:
    oof = pd.read_csv(V5_DIR / "model_zoo_oof.csv")
    test = pd.read_csv(V5_DIR / "model_zoo_test.csv")
    y = oof["Survived"].astype(int).to_numpy()

    pred, vote_fraction = majority_predictions(oof)
    test_pred, test_vote_fraction = majority_predictions(test)

    accuracy = accuracy_score(y, pred)
    vote_auc = roc_auc_score(y, vote_fraction)

    rows = []
    for fold in sorted(oof["fold"].unique()):
        mask = oof["fold"] == fold
        fold_y = y[mask]
        fold_pred = pred[mask]
        fold_score = vote_fraction[mask]
        rows.append(
            {
                "fold": int(fold),
                "hard_vote_accuracy": accuracy_score(fold_y, fold_pred),
                "vote_fraction_auc": roc_auc_score(fold_y, fold_score),
                "v4b_accuracy": accuracy_score(
                    fold_y, oof.loc[mask, "v4b__Champion"].to_numpy() > 0.5
                ),
                "mlp_plr_accuracy": accuracy_score(
                    fold_y, oof.loc[mask, "MLP_PLR"].to_numpy() > 0.5
                ),
                "rulefit_accuracy": accuracy_score(
                    fold_y, oof.loc[mask, "RuleFit"].to_numpy() > 0.5
                ),
            }
        )
    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(V5_DIR / "hard_vote_fold_metrics.csv", index=False)

    submission = pd.DataFrame(
        {
            "PassengerId": test["PassengerId"].astype(int),
            "Survived": test_pred.astype(int),
        }
    )
    path = SUB_DIR / "submission_v5_hard_vote.csv"
    submission.to_csv(path, index=False)

    probs = pd.DataFrame(
        {
            "PassengerId": test["PassengerId"].astype(int),
            "v4b_vote": (test["v4b__Champion"] > 0.5).astype(int),
            "mlp_plr_vote": (test["MLP_PLR"] > 0.5).astype(int),
            "rulefit_vote": (test["RuleFit"] > 0.5).astype(int),
            "vote_fraction": test_vote_fraction,
            "Survived": test_pred,
        }
    )
    probs.to_csv(V5_DIR / "hard_vote_test_votes.csv", index=False)

    comparisons = []
    for label, ref_path in [
        ("v4b", SUB_DIR / "submission_v4_score_0.79425.csv"),
        ("v5_mlp_plr", SUB_DIR / "submission_v5.csv"),
    ]:
        if not ref_path.exists():
            continue
        ref = pd.read_csv(ref_path)
        merged = submission.merge(ref, on="PassengerId", suffixes=("_vote", "_ref"))
        comparisons.append(
            {
                "reference": label,
                "different_predictions": int(
                    (merged["Survived_vote"] != merged["Survived_ref"]).sum()
                ),
                "same_predictions": int(
                    (merged["Survived_vote"] == merged["Survived_ref"]).sum()
                ),
                "hard_vote_positive_predictions": int(test_pred.sum()),
                "reference_positive_predictions": int(ref["Survived"].sum()),
            }
        )
    comparison_df = pd.DataFrame(comparisons)
    comparison_df.to_csv(V5_DIR / "hard_vote_submission_comparison.csv", index=False)

    metadata = {
        "version": "v5_hard_vote",
        "members": MEMBERS,
        "rule": "majority of three binary predictions at threshold 0.5",
        "oof_accuracy": float(accuracy),
        "vote_fraction_roc_auc": float(vote_auc),
        "fold_accuracy_mean": float(fold_df["hard_vote_accuracy"].mean()),
        "fold_accuracy_std": float(fold_df["hard_vote_accuracy"].std(ddof=0)),
        "test_positive_predictions": int(test_pred.sum()),
        "status": "local_candidate_not_submitted",
        "note": "No learned blend weights; low-degree-of-freedom hard vote.",
    }
    with (V5_DIR / "hard_vote_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("=== v5 top-3 hard vote ===")
    print("Members:", MEMBERS)
    print(f"OOF Accuracy: {accuracy:.5f}")
    print(f"Vote-fraction ROC-AUC: {vote_auc:.5f}")
    print(f"Fold accuracy std: {metadata['fold_accuracy_std']:.5f}")
    print(f"Test positives: {int(test_pred.sum())} / {len(test_pred)}")
    print()
    print(fold_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    if len(comparison_df):
        print()
        print(comparison_df.to_string(index=False))
    print(f"Submission: {path}")
    print("Kaggle submission: NOT performed.")


if __name__ == "__main__":
    main()
