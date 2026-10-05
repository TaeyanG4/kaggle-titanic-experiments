"""Build the Titanic v4 submission candidate from saved Model Zoo OOF/test probabilities.

No model is trained here. The default candidate is TabICLv2 because it won the
fold-safe v4 screen. A small cross-fitted threshold audit is used only as a
robustness check. The final submission uses the median threshold selected from
the five outer folds only if its cross-fitted accuracy strictly improves over
the fixed 0.5 rule; otherwise it keeps 0.5.

No Kaggle submission is performed by this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


BASE_DIR = Path(__file__).resolve().parents[1]
V4_DIR = BASE_DIR / "exports" / "v4"
SUB_DIR = BASE_DIR / "submissions"
MODEL = "TabICLv2"
GRID = np.round(np.arange(0.35, 0.651, 0.01), 2)


def best_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    rows = []
    for threshold in GRID:
        score = accuracy_score(y, (p > threshold).astype(int))
        rows.append((float(threshold), float(score)))
    best_score = max(score for _, score in rows)
    tied = [threshold for threshold, score in rows if score == best_score]
    # Regularize toward the conventional 0.5 threshold when several values tie.
    selected = min(tied, key=lambda t: (abs(t - 0.5), t))
    return selected, best_score


def main() -> None:
    oof_path = V4_DIR / "model_zoo_oof.csv"
    test_path = V4_DIR / "model_zoo_test.csv"
    if not oof_path.exists() or not test_path.exists():
        raise FileNotFoundError("Run scripts/model_zoo_screen_v4.py first.")

    oof = pd.read_csv(oof_path)
    test = pd.read_csv(test_path)
    required_oof = {"PassengerId", "fold", "Survived", MODEL}
    required_test = {"PassengerId", MODEL}
    if not required_oof.issubset(oof.columns) or not required_test.issubset(test.columns):
        raise ValueError(f"Missing {MODEL} predictions in v4 artifacts.")

    y = oof["Survived"].astype(int).to_numpy()
    p = oof[MODEL].astype(float).to_numpy()
    fixed_pred = (p > 0.5).astype(int)
    fixed_acc = accuracy_score(y, fixed_pred)
    auc = roc_auc_score(y, p)

    crossfit_pred = np.zeros(len(oof), dtype=int)
    threshold_rows = []
    for fold in sorted(oof["fold"].astype(int).unique()):
        val_mask = oof["fold"].astype(int).to_numpy() == fold
        fit_mask = ~val_mask
        threshold, fit_acc = best_threshold(y[fit_mask], p[fit_mask])
        crossfit_pred[val_mask] = (p[val_mask] > threshold).astype(int)
        threshold_rows.append(
            {
                "fold": int(fold),
                "selected_threshold_from_other_folds": threshold,
                "threshold_fit_accuracy": fit_acc,
                "heldout_accuracy": accuracy_score(y[val_mask], crossfit_pred[val_mask]),
                "n_heldout": int(val_mask.sum()),
            }
        )

    threshold_df = pd.DataFrame(threshold_rows)
    crossfit_acc = accuracy_score(y, crossfit_pred)
    median_threshold = float(np.median(threshold_df["selected_threshold_from_other_folds"]))

    # Conservative promotion: threshold tuning must improve genuinely cross-fitted
    # accuracy before it is allowed to alter the test decision rule.
    if crossfit_acc > fixed_acc:
        selected_threshold = median_threshold
        threshold_policy = "crossfit_median"
    else:
        selected_threshold = 0.5
        threshold_policy = "fixed_0.5"

    test_prob = test[MODEL].astype(float).to_numpy()
    test_pred = (test_prob > selected_threshold).astype(int)

    SUB_DIR.mkdir(parents=True, exist_ok=True)
    V4_DIR.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame(
        {
            "PassengerId": test["PassengerId"].astype(int),
            "Survived": test_pred.astype(int),
        }
    )
    submission.to_csv(SUB_DIR / "submission_v4.csv", index=False)

    probabilities = pd.DataFrame(
        {
            "PassengerId": test["PassengerId"].astype(int),
            f"{MODEL}_probability": test_prob,
            "selected_threshold": selected_threshold,
            "Survived": test_pred,
        }
    )
    probabilities.to_csv(V4_DIR / "submission_v4_probabilities.csv", index=False)
    threshold_df.to_csv(V4_DIR / "threshold_crossfit.csv", index=False)

    comparisons = []
    for label, path in [
        ("v1", SUB_DIR / "submission_v1_score_0.79186.csv"),
        ("v2", SUB_DIR / "submission_v2.csv"),
        ("v3", SUB_DIR / "submission_v3.csv"),
    ]:
        if not path.exists():
            continue
        ref = pd.read_csv(path)
        merged = submission.merge(
            ref[["PassengerId", "Survived"]],
            on="PassengerId",
            suffixes=("_v4", f"_{label}"),
        )
        comparisons.append(
            {
                "reference": label,
                "different_predictions": int(
                    (merged["Survived_v4"] != merged[f"Survived_{label}"]).sum()
                ),
                "same_predictions": int(
                    (merged["Survived_v4"] == merged[f"Survived_{label}"]).sum()
                ),
                "v4_positive_predictions": int(submission["Survived"].sum()),
                "reference_positive_predictions": int(ref["Survived"].sum()),
            }
        )
    pd.DataFrame(comparisons).to_csv(V4_DIR / "submission_v4_comparison.csv", index=False)

    metadata = {
        "model": MODEL,
        "fixed_threshold": 0.5,
        "fixed_oof_accuracy": float(fixed_acc),
        "oof_roc_auc": float(auc),
        "crossfit_threshold_accuracy": float(crossfit_acc),
        "fold_selected_thresholds": threshold_df[
            "selected_threshold_from_other_folds"
        ].tolist(),
        "median_crossfit_threshold": median_threshold,
        "selected_threshold": selected_threshold,
        "threshold_policy": threshold_policy,
        "test_positive_predictions": int(test_pred.sum()),
        "test_rows": int(len(test_pred)),
        "note": "Candidate built from saved fold-safe v4 probabilities; no training and no Kaggle submission.",
    }
    with (V4_DIR / "submission_v4_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("=== Titanic v4 candidate ===")
    print(f"Model: {MODEL}")
    print(f"OOF Accuracy @0.5: {fixed_acc:.5f}")
    print(f"OOF ROC-AUC      : {auc:.5f}")
    print(f"Cross-fit threshold Accuracy: {crossfit_acc:.5f}")
    print("Fold-selected thresholds:", threshold_df["selected_threshold_from_other_folds"].tolist())
    print(f"Median threshold: {median_threshold:.2f}")
    print(f"Final policy: {threshold_policy}, threshold={selected_threshold:.2f}")
    print(f"Test positives: {int(test_pred.sum())} / {len(test_pred)}")
    print(f"Submission: {SUB_DIR / 'submission_v4.csv'}")
    print("No Kaggle submission was performed.")


if __name__ == "__main__":
    main()
