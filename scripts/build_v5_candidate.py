"""Build and audit the strongest Titanic v5 candidate from saved OOF predictions.

Current primary candidate: MLP_PLR.
No model training and no Kaggle submission are performed here.
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
MODEL = "MLP_PLR"
GRID = np.round(np.arange(0.35, 0.651, 0.01), 2)


def choose_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    scored = []
    for threshold in GRID:
        acc = accuracy_score(y, p > threshold)
        scored.append((float(threshold), float(acc)))
    best_acc = max(acc for _, acc in scored)
    tied = [t for t, acc in scored if acc == best_acc]
    selected = min(tied, key=lambda t: (abs(t - 0.5), t))
    return selected, best_acc


def main() -> None:
    oof = pd.read_csv(V5_DIR / "model_zoo_oof.csv")
    test = pd.read_csv(V5_DIR / "model_zoo_test.csv")
    y = oof["Survived"].astype(int).to_numpy()
    p = oof[MODEL].astype(float).to_numpy()

    fixed_acc = accuracy_score(y, p > 0.5)
    fixed_auc = roc_auc_score(y, p)

    crossfit_pred = np.zeros(len(oof), dtype=int)
    rows = []
    folds = oof["fold"].astype(int).to_numpy()
    for fold in sorted(np.unique(folds)):
        val_mask = folds == fold
        fit_mask = ~val_mask
        threshold, fit_acc = choose_threshold(y[fit_mask], p[fit_mask])
        crossfit_pred[val_mask] = (p[val_mask] > threshold).astype(int)
        rows.append(
            {
                "fold": int(fold),
                "threshold_selected_on_other_folds": threshold,
                "fit_accuracy": fit_acc,
                "heldout_accuracy": accuracy_score(y[val_mask], crossfit_pred[val_mask]),
            }
        )

    threshold_df = pd.DataFrame(rows)
    crossfit_acc = accuracy_score(y, crossfit_pred)
    median_threshold = float(np.median(threshold_df["threshold_selected_on_other_folds"]))

    # Conservative: threshold adjustment is promoted only if cross-fitted accuracy improves.
    if crossfit_acc > fixed_acc:
        final_threshold = median_threshold
        threshold_policy = "crossfit_median"
    else:
        final_threshold = 0.5
        threshold_policy = "fixed_0.5"

    test_prob = test[MODEL].astype(float).to_numpy()
    test_pred = (test_prob > final_threshold).astype(int)
    submission = pd.DataFrame(
        {
            "PassengerId": test["PassengerId"].astype(int),
            "Survived": test_pred,
        }
    )
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = SUB_DIR / "submission_v5.csv"
    submission.to_csv(submission_path, index=False)

    probability_df = pd.DataFrame(
        {
            "PassengerId": test["PassengerId"].astype(int),
            f"{MODEL}_probability": test_prob,
            "threshold": final_threshold,
            "Survived": test_pred,
        }
    )
    probability_df.to_csv(V5_DIR / "submission_v5_probabilities.csv", index=False)
    threshold_df.to_csv(V5_DIR / "threshold_crossfit.csv", index=False)

    comparisons = []
    refs = [
        ("v1", SUB_DIR / "submission_v1_score_0.79186.csv"),
        ("v4b", SUB_DIR / "submission_v4_score_0.79425.csv"),
    ]
    for label, path in refs:
        if not path.exists():
            continue
        ref = pd.read_csv(path)
        merged = submission.merge(ref, on="PassengerId", suffixes=("_v5", "_ref"))
        comparisons.append(
            {
                "reference": label,
                "different_predictions": int(
                    (merged["Survived_v5"] != merged["Survived_ref"]).sum()
                ),
                "same_predictions": int(
                    (merged["Survived_v5"] == merged["Survived_ref"]).sum()
                ),
                "v5_positive_predictions": int(submission["Survived"].sum()),
                "reference_positive_predictions": int(ref["Survived"].sum()),
            }
        )
    comparison_df = pd.DataFrame(comparisons)
    comparison_df.to_csv(V5_DIR / "submission_v5_comparison.csv", index=False)

    fold_metrics = []
    for fold in sorted(np.unique(folds)):
        mask = folds == fold
        fold_metrics.append(
            {
                "fold": int(fold),
                "accuracy_at_0_5": accuracy_score(y[mask], p[mask] > 0.5),
                "roc_auc": roc_auc_score(y[mask], p[mask]),
            }
        )
    pd.DataFrame(fold_metrics).to_csv(V5_DIR / "candidate_fold_metrics.csv", index=False)

    metadata = {
        "version": "v5",
        "model": MODEL,
        "oof_accuracy_at_0_5": float(fixed_acc),
        "oof_roc_auc": float(fixed_auc),
        "crossfit_threshold_accuracy": float(crossfit_acc),
        "fold_selected_thresholds": threshold_df[
            "threshold_selected_on_other_folds"
        ].tolist(),
        "median_crossfit_threshold": median_threshold,
        "final_threshold": final_threshold,
        "threshold_policy": threshold_policy,
        "test_positive_predictions": int(test_pred.sum()),
        "test_rows": int(len(test_pred)),
        "status": "local_candidate_not_submitted",
    }
    with (V5_DIR / "submission_v5_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("=== Titanic v5 candidate ===")
    print(f"Model: {MODEL}")
    print(f"OOF Accuracy @0.5: {fixed_acc:.5f}")
    print(f"OOF ROC-AUC      : {fixed_auc:.5f}")
    print(f"Cross-fit threshold Accuracy: {crossfit_acc:.5f}")
    print("Fold-selected thresholds:", threshold_df["threshold_selected_on_other_folds"].tolist())
    print(f"Final threshold policy: {threshold_policy} ({final_threshold:.2f})")
    print(f"Test positives: {int(test_pred.sum())} / {len(test_pred)}")
    if len(comparison_df):
        print(comparison_df.to_string(index=False))
    print(f"Submission: {submission_path}")
    print("Kaggle submission: NOT performed.")


if __name__ == "__main__":
    main()
