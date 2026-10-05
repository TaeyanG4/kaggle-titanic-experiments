"""Small mechanistic smoothing screen for the typed22 relation block.

alpha controls shrinkage toward the global survival prior:
  smooth = (survivors + alpha * prior) / (peer_count + alpha)

Screened on BOTH:
  - real-test-matched pseudo-test holdouts
  - trusted fixed 5-fold validation

No test predictions and no Kaggle submission are produced.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from gunes_exact_foldsafe_v11 import structural_frames
from pseudo_test_relational_v14 import (
    VARIANTS,
    eligible_groups,
    relation_features,
    role_flags,
)


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
V1_AUDIT = BASE_DIR / "exports" / "wcg_audit_v1"
V14_DIR = BASE_DIR / "exports" / "v14"
EXPORT_DIR = BASE_DIR / "exports" / "v15"

ALPHAS = [0.5, 1.0, 2.0, 4.0, 8.0]
TYPED = VARIANTS["typed22"]


def rf(seed: int):
    return RandomForestClassifier(
        n_estimators=800,
        max_depth=7,
        min_samples_split=6,
        min_samples_leaf=6,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )


def eval_split(
    train_df,
    y,
    base_cols,
    tr_idx,
    va_idx,
    alpha,
    seed,
):
    tr = train_df.iloc[tr_idx].copy()
    va = train_df.iloc[va_idx].copy()
    af, at = eligible_groups(tr, va)
    tr_rel = relation_features(
        tr, tr, exclude_self=True, alpha=alpha,
        allowed_fam=af, allowed_tic=at
    )
    va_rel = relation_features(
        tr, va, exclude_self=False, alpha=alpha,
        allowed_fam=af, allowed_tic=at
    )
    xtr = np.column_stack(
        [tr[base_cols].to_numpy(float), tr_rel[TYPED].to_numpy()]
    )
    xva = np.column_stack(
        [va[base_cols].to_numpy(float), va_rel[TYPED].to_numpy()]
    )
    xtr = StandardScaler().fit_transform(xtr)
    xva = StandardScaler().fit_transform(xva)
    m = rf(seed)
    m.fit(xtr, y[tr_idx])
    prob = m.predict_proba(xva)[:, 1]
    return (
        accuracy_score(y[va_idx], prob >= 0.5),
        roc_auc_score(y[va_idx], prob),
    )


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    train_df, _, base_cols, _, _ = structural_frames(train_raw, test_raw)
    train_df["IsWomanChild"] = role_flags(train_raw)
    y = train_df["Survived"].astype(int).to_numpy()

    manifest = pd.read_csv(V1_AUDIT / "fold_manifest_seed42.csv")
    folds = manifest["fold"].astype(int).to_numpy()
    pseudo = pd.read_csv(V14_DIR / "pseudo_test_manifest.csv")

    rows = []
    for alpha in ALPHAS:
        # trusted folds
        for f in sorted(np.unique(folds)):
            tr_idx = np.flatnonzero(folds != f)
            va_idx = np.flatnonzero(folds == f)
            acc, auc = eval_split(
                train_df, y, base_cols, tr_idx, va_idx, alpha, 100 + int(f)
            )
            rows.append({
                "surface": "trusted",
                "split": int(f),
                "alpha": alpha,
                "accuracy": acc,
                "roc_auc": auc,
            })

        # pseudo-test repeated holdouts
        for s in sorted(pseudo["split"].unique()):
            va_ids = set(
                pseudo.loc[pseudo["split"] == s, "PassengerId"].astype(int)
            )
            va_mask = train_df["PassengerId"].astype(int).isin(va_ids).to_numpy()
            va_idx = np.flatnonzero(va_mask)
            tr_idx = np.flatnonzero(~va_mask)
            acc, auc = eval_split(
                train_df, y, base_cols, tr_idx, va_idx, alpha, 200 + int(s)
            )
            rows.append({
                "surface": "pseudo",
                "split": int(s),
                "alpha": alpha,
                "accuracy": acc,
                "roc_auc": auc,
            })
        print(f"alpha={alpha} done", flush=True)

    metrics = pd.DataFrame(rows)
    summary = (
        metrics.groupby(["surface", "alpha"])
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            min_accuracy=("accuracy", "min"),
            mean_auc=("roc_auc", "mean"),
            std_auc=("roc_auc", "std"),
        )
        .reset_index()
    )
    combined = (
        summary.groupby("alpha")
        .agg(
            mean_of_surface_accuracy=("mean_accuracy", "mean"),
            worst_surface_accuracy=("mean_accuracy", "min"),
            mean_of_surface_auc=("mean_auc", "mean"),
            worst_surface_auc=("mean_auc", "min"),
        )
        .reset_index()
        .sort_values(
            ["worst_surface_accuracy", "mean_of_surface_auc"],
            ascending=False,
        )
    )
    metrics.to_csv(EXPORT_DIR / "alpha_screen_metrics.csv", index=False)
    summary.to_csv(EXPORT_DIR / "alpha_screen_summary.csv", index=False)
    combined.to_csv(EXPORT_DIR / "alpha_screen_combined.csv", index=False)

    print("\n=== alpha by surface ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n=== combined robustness ranking ===")
    print(combined.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nNo Kaggle submission was performed.")


if __name__ == "__main__":
    main()
