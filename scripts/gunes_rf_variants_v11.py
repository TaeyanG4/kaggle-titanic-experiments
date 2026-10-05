"""Exploit Gunes' exact 26-feature representation with documented RF variants.

Candidates:
  - single_best_full: original single_best_model fit on all training rows
  - single_best_cv:   same model, 5-fold test probability average
  - leaderboard_full: original leaderboard_model fit on all training rows
  - leaderboard_cv:   reproduction of v10, included as a reference

No Kaggle submission is performed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from reproduce_gunes_original_v10 import preprocess_original


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "exports" / "v11"
SUBMISSION_DIR = BASE_DIR / "submissions"


CONFIGS = {
    "single_best": dict(
        n_estimators=1100,
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=5,
    ),
    "leaderboard": dict(
        n_estimators=1750,
        max_depth=7,
        min_samples_split=6,
        min_samples_leaf=6,
    ),
}


def make_model(cfg: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        criterion="gini",
        **cfg,
        max_features="sqrt",
        oob_score=True,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )


def cv_predict(
    cfg_name: str,
    cfg: dict,
    X_train: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
):
    skf = StratifiedKFold(n_splits=5, random_state=5, shuffle=True)
    oof = np.zeros(len(y), dtype=float)
    test_probs = []
    fold_rows = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), 1):
        model = make_model(cfg)
        model.fit(X_train[tr_idx], y[tr_idx])
        va_prob = model.predict_proba(X_train[va_idx])[:, 1]
        te_prob = model.predict_proba(X_test)[:, 1]
        oof[va_idx] = va_prob
        test_probs.append(te_prob)
        fold_rows.append(
            {
                "config": cfg_name,
                "fold": fold,
                "accuracy": accuracy_score(y[va_idx], va_prob >= 0.5),
                "roc_auc": roc_auc_score(y[va_idx], va_prob),
                "oob_score": model.oob_score_,
            }
        )
    return oof, np.mean(test_probs, axis=0), pd.DataFrame(fold_rows)


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    X_train, y, X_test, passenger_ids, feature_names = preprocess_original(
        DATA_DIR / "train.csv", DATA_DIR / "test.csv"
    )

    train_ids = pd.read_csv(DATA_DIR / "train.csv")["PassengerId"].astype(int)
    oof_export = pd.DataFrame({"PassengerId": train_ids, "Survived": y})
    test_export = pd.DataFrame({"PassengerId": passenger_ids.to_numpy()})
    summary_rows = []
    fold_rows_all = []

    for name, cfg in CONFIGS.items():
        # Full-train original "single model" interpretation.
        full = make_model(cfg)
        full.fit(X_train, y)
        full_train_prob = full.predict_proba(X_train)[:, 1]
        full_test_prob = full.predict_proba(X_test)[:, 1]
        test_export[f"{name}_full"] = full_test_prob
        summary_rows.append(
            {
                "candidate": f"{name}_full",
                "evaluation": "in_sample_train_diagnostic",
                "accuracy": accuracy_score(y, full_train_prob >= 0.5),
                "roc_auc": roc_auc_score(y, full_train_prob),
                "oob_score": full.oob_score_,
                "test_positive_count": int((full_test_prob >= 0.5).sum()),
            }
        )

        # Historical 5-fold test averaging.
        oof, cv_test_prob, fold_df = cv_predict(
            name, cfg, X_train, y, X_test
        )
        fold_rows_all.append(fold_df)
        oof_export[f"{name}_cv"] = oof
        test_export[f"{name}_cv"] = cv_test_prob
        summary_rows.append(
            {
                "candidate": f"{name}_cv",
                "evaluation": "historical_leaky_cv",
                "accuracy": accuracy_score(y, oof >= 0.5),
                "roc_auc": roc_auc_score(y, oof),
                "oob_score": fold_df["oob_score"].mean(),
                "test_positive_count": int((cv_test_prob >= 0.5).sum()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    # Create all submission candidates.
    for col in [c for c in test_export.columns if c != "PassengerId"]:
        pd.DataFrame(
            {
                "PassengerId": test_export["PassengerId"].astype(int),
                "Survived": (test_export[col] >= 0.5).astype(int),
            }
        ).to_csv(SUBMISSION_DIR / f"submission_v11_{col}.csv", index=False)

    v10 = pd.read_csv(
        SUBMISSION_DIR / "submission_v10_gunes_original_reproduction.csv"
    )
    v5 = pd.read_csv(SUBMISSION_DIR / "submission_v5_score_0.79665.csv")
    compare_rows = []
    for col in [c for c in test_export.columns if c != "PassengerId"]:
        pred = (test_export[col] >= 0.5).astype(int)
        compare_rows.append(
            {
                "candidate": col,
                "positives": int(pred.sum()),
                "differences_vs_v10": int((pred.to_numpy() != v10["Survived"].to_numpy()).sum()),
                "differences_vs_v5": int((pred.to_numpy() != v5["Survived"].to_numpy()).sum()),
            }
        )

    summary.to_csv(EXPORT_DIR / "rf_variant_summary.csv", index=False)
    pd.concat(fold_rows_all, ignore_index=True).to_csv(
        EXPORT_DIR / "rf_variant_fold_metrics.csv", index=False
    )
    oof_export.to_csv(EXPORT_DIR / "rf_variant_oof.csv", index=False)
    test_export.to_csv(EXPORT_DIR / "rf_variant_test.csv", index=False)
    pd.DataFrame(compare_rows).to_csv(
        EXPORT_DIR / "rf_variant_submission_comparison.csv", index=False
    )
    with (EXPORT_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_count": len(feature_names),
                "feature_names": feature_names,
                "configs": CONFIGS,
                "source_reported_public": {
                    "single_best_model": 0.82775,
                    "leaderboard_model": 0.83732,
                },
                "no_submission": True,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=== RF variants ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n=== Test prediction differences ===")
    print(pd.DataFrame(compare_rows).to_string(index=False))
    print("\nNo Kaggle submission was performed.")


if __name__ == "__main__":
    main()
