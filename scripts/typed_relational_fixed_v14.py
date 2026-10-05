"""Trusted fixed-fold confirmation of the v14 typed relational feature block.

Uses the same exact Gunes-style 24 target-independent base features and
compares legacy2 vs smooth8 vs typed22 on the preserved seed-42 fold manifest.
All target-derived group features are fold-safe.

Test probabilities are also produced with each fold using the *actual test*
entity membership for transductive eligibility, but no submission is made.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from gunes_exact_foldsafe_v11 import structural_frames
from pseudo_test_relational_v14 import (
    VARIANTS,
    eligible_groups,
    make_rf,
    relation_features,
    role_flags,
)


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
V1_AUDIT = BASE_DIR / "exports" / "wcg_audit_v1"
EXPORT_DIR = BASE_DIR / "exports" / "v14"


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    train_df, test_df, base_cols, _, _ = structural_frames(train_raw, test_raw)
    train_df["IsWomanChild"] = role_flags(train_raw)
    test_df["IsWomanChild"] = role_flags(test_raw)

    manifest = pd.read_csv(V1_AUDIT / "fold_manifest_seed42.csv")
    if not train_df["PassengerId"].astype(int).equals(
        manifest["PassengerId"].astype(int)
    ):
        raise ValueError("PassengerId order mismatch vs trusted fold manifest.")
    folds = manifest["fold"].astype(int).to_numpy()
    y = train_df["Survived"].astype(int).to_numpy()

    rows = []
    fold_rows = []
    oof_export = pd.DataFrame(
        {
            "PassengerId": train_df["PassengerId"].astype(int),
            "fold": folds,
            "Survived": y,
        }
    )
    test_export = pd.DataFrame(
        {"PassengerId": test_df["PassengerId"].astype(int)}
    )

    for variant, rel_cols in VARIANTS.items():
        oof = np.zeros(len(train_df), dtype=float)
        test_probs = []
        for fold in sorted(np.unique(folds)):
            tr_idx = np.flatnonzero(folds != fold)
            va_idx = np.flatnonzero(folds == fold)
            tr = train_df.iloc[tr_idx].copy()
            va = train_df.iloc[va_idx].copy()

            # Validation-aware eligibility for OOF.
            allowed_fam_val, allowed_tic_val = eligible_groups(tr, va)
            tr_rel_val = relation_features(
                tr,
                tr,
                exclude_self=True,
                alpha=2.0,
                allowed_fam=allowed_fam_val,
                allowed_tic=allowed_tic_val,
            )
            va_rel = relation_features(
                tr,
                va,
                exclude_self=False,
                alpha=2.0,
                allowed_fam=allowed_fam_val,
                allowed_tic=allowed_tic_val,
            )
            xtr = np.column_stack(
                [tr[base_cols].to_numpy(dtype=float), tr_rel_val[rel_cols].to_numpy()]
            )
            xva = np.column_stack(
                [va[base_cols].to_numpy(dtype=float), va_rel[rel_cols].to_numpy()]
            )
            xtr = StandardScaler().fit_transform(xtr)
            xva = StandardScaler().fit_transform(xva)
            model = make_rf()
            model.fit(xtr, y[tr_idx])
            va_prob = model.predict_proba(xva)[:, 1]
            oof[va_idx] = va_prob

            # Actual-test-aware eligibility for fold test prediction.
            allowed_fam_test, allowed_tic_test = eligible_groups(tr, test_df)
            tr_rel_test = relation_features(
                tr,
                tr,
                exclude_self=True,
                alpha=2.0,
                allowed_fam=allowed_fam_test,
                allowed_tic=allowed_tic_test,
            )
            te_rel = relation_features(
                tr,
                test_df,
                exclude_self=False,
                alpha=2.0,
                allowed_fam=allowed_fam_test,
                allowed_tic=allowed_tic_test,
            )
            xtr_t = np.column_stack(
                [tr[base_cols].to_numpy(dtype=float), tr_rel_test[rel_cols].to_numpy()]
            )
            xte = np.column_stack(
                [test_df[base_cols].to_numpy(dtype=float), te_rel[rel_cols].to_numpy()]
            )
            xtr_t = StandardScaler().fit_transform(xtr_t)
            xte = StandardScaler().fit_transform(xte)
            test_model = make_rf()
            test_model.fit(xtr_t, y[tr_idx])
            test_probs.append(test_model.predict_proba(xte)[:, 1])

            fold_rows.append(
                {
                    "variant": variant,
                    "fold": int(fold),
                    "accuracy": accuracy_score(y[va_idx], va_prob >= 0.5),
                    "roc_auc": roc_auc_score(y[va_idx], va_prob),
                    "feature_count": len(base_cols) + len(rel_cols),
                }
            )
            print(
                f"{variant} fold={fold}: "
                f"acc={fold_rows[-1]['accuracy']:.5f} "
                f"auc={fold_rows[-1]['roc_auc']:.5f}",
                flush=True,
            )

        test_prob = np.mean(test_probs, axis=0)
        oof_export[variant] = oof
        test_export[variant] = test_prob
        vf = pd.DataFrame([x for x in fold_rows if x["variant"] == variant])
        rows.append(
            {
                "variant": variant,
                "accuracy": accuracy_score(y, oof >= 0.5),
                "roc_auc": roc_auc_score(y, oof),
                "fold_accuracy_std": vf["accuracy"].std(ddof=0),
                "fold_auc_std": vf["roc_auc"].std(ddof=0),
                "test_positive_count": int((test_prob >= 0.5).sum()),
                "feature_count": len(base_cols) + len(rel_cols),
            }
        )

    summary = pd.DataFrame(rows)
    baseline = summary.loc[summary["variant"] == "legacy2"].iloc[0]
    summary["delta_accuracy_vs_legacy"] = summary["accuracy"] - baseline["accuracy"]
    summary["delta_auc_vs_legacy"] = summary["roc_auc"] - baseline["roc_auc"]
    summary = summary.sort_values(["accuracy", "roc_auc"], ascending=False)

    pd.DataFrame(fold_rows).to_csv(
        EXPORT_DIR / "typed_relational_fixed_fold_metrics.csv", index=False
    )
    summary.to_csv(
        EXPORT_DIR / "typed_relational_fixed_summary.csv", index=False
    )
    oof_export.to_csv(
        EXPORT_DIR / "typed_relational_fixed_oof.csv", index=False
    )
    test_export.to_csv(
        EXPORT_DIR / "typed_relational_fixed_test.csv", index=False
    )

    print("\n=== trusted fixed-fold relational ranking ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nNo Kaggle submission was performed.")


if __name__ == "__main__":
    main()
