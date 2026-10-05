"""Compare fold-safe v1/v2 OOF predictions and build a fixed 50:50 hybrid.

Prerequisite: both audit profiles have been completed:
    python scripts/audit_group_survival.py --profile v1
    python scripts/audit_group_survival.py --profile v2

This script performs no model training and never submits to Kaggle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


BASE_DIR = Path(__file__).resolve().parents[1]
V1_DIR = BASE_DIR / "exports" / "wcg_audit_v1"
V2_DIR = BASE_DIR / "exports" / "wcg_audit_v2"
OUT_DIR = BASE_DIR / "exports" / "v3"
SUBMISSION_DIR = BASE_DIR / "submissions"

WEIGHTS_V1 = [0.00, 0.25, 0.50, 0.75, 1.00]


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing: {path}\n"
            "Run the missing audit first, e.g. "
            "python scripts/audit_group_survival.py --profile v1"
        )
    return path


def load_profile(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    oof = pd.read_csv(require(directory / "oof_fold_safe.csv"))
    test = pd.read_csv(require(directory / "test_group_survival_audit.csv"))
    for frame, label in ((oof, "OOF"), (test, "test")):
        if "fold_safe__Ensemble" not in frame.columns:
            raise KeyError(f"{label} missing fold_safe__Ensemble in {directory}")
    return oof, test


def main() -> None:
    v1_oof, v1_test = load_profile(V1_DIR)
    v2_oof, v2_test = load_profile(V2_DIR)

    oof_keys = ["PassengerId", "fold", "Survived"]
    if not v1_oof[oof_keys].equals(v2_oof[oof_keys]):
        raise ValueError("v1/v2 OOF PassengerId/fold/target manifests do not match exactly")
    if not v1_test["PassengerId"].equals(v2_test["PassengerId"]):
        raise ValueError("v1/v2 test PassengerId order does not match")

    y = v1_oof["Survived"].astype(int).to_numpy()
    folds = v1_oof["fold"].astype(int).to_numpy()
    p1 = v1_oof["fold_safe__Ensemble"].to_numpy(float)
    p2 = v2_oof["fold_safe__Ensemble"].to_numpy(float)
    t1 = v1_test["fold_safe__Ensemble"].to_numpy(float)
    t2 = v2_test["fold_safe__Ensemble"].to_numpy(float)

    scan_rows = []
    fold_rows = []
    for w1 in WEIGHTS_V1:
        w2 = 1.0 - w1
        prob = w1 * p1 + w2 * p2
        pred = (prob > 0.5).astype(int)
        scan_rows.append(
            {
                "v1_weight": w1,
                "v2_weight": w2,
                "accuracy": accuracy_score(y, pred),
                "roc_auc": roc_auc_score(y, prob),
                "positive_oof_predictions": int(pred.sum()),
            }
        )
        for fold in sorted(np.unique(folds)):
            mask = folds == fold
            fold_rows.append(
                {
                    "fold": int(fold),
                    "v1_weight": w1,
                    "v2_weight": w2,
                    "accuracy": accuracy_score(y[mask], pred[mask]),
                    "roc_auc": roc_auc_score(y[mask], prob[mask]),
                }
            )

    scan = pd.DataFrame(scan_rows)
    fold_metrics = pd.DataFrame(fold_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    scan.to_csv(OUT_DIR / "hybrid_weight_scan.csv", index=False)
    fold_metrics.to_csv(OUT_DIR / "hybrid_fold_metrics.csv", index=False)

    # Fixed hypothesis from plan.md: 50:50 blend. Keep it separate from any
    # exploratory weight scan so the grid does not silently become the answer.
    hybrid_oof = 0.5 * p1 + 0.5 * p2
    hybrid_test = 0.5 * t1 + 0.5 * t2
    hybrid_pred = (hybrid_test > 0.5).astype(int)

    pd.DataFrame(
        {
            "PassengerId": v1_oof["PassengerId"],
            "fold": v1_oof["fold"],
            "Survived": y,
            "v1_fold_safe": p1,
            "v2_fold_safe": p2,
            "hybrid_50_50": hybrid_oof,
        }
    ).to_csv(OUT_DIR / "hybrid_oof_50_50.csv", index=False)

    pd.DataFrame(
        {
            "PassengerId": v1_test["PassengerId"].astype(int),
            "v1_fold_safe": t1,
            "v2_fold_safe": t2,
            "hybrid_50_50": hybrid_test,
            "Prediction": hybrid_pred,
        }
    ).to_csv(OUT_DIR / "hybrid_test_50_50.csv", index=False)

    submission = pd.DataFrame(
        {
            "PassengerId": v1_test["PassengerId"].astype(int),
            "Survived": hybrid_pred,
        }
    )
    submission_path = SUBMISSION_DIR / "submission_v3_hybrid_50_50.csv"
    submission.to_csv(submission_path, index=False)

    print("=== Fold-safe v1/v2 hybrid scan ===")
    print(scan.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    row50 = scan[np.isclose(scan["v1_weight"], 0.5)].iloc[0]
    print("\n=== Fixed 50:50 hypothesis ===")
    print(f"OOF Accuracy: {row50['accuracy']:.5f}")
    print(f"OOF ROC-AUC : {row50['roc_auc']:.5f}")
    print(f"Test positives: {int(hybrid_pred.sum())} / {len(hybrid_pred)}")
    print(f"Candidate: {submission_path}")
    print("\nNo Kaggle submission was performed.")


if __name__ == "__main__":
    main()
