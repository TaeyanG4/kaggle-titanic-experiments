"""Small mechanism screen for WomanChild role definitions used by typed22.

Definitions are fixed a priori:
  current16: female OR Age < 16 OR Master
  classic15: female OR Age < 15 OR Master
  teen18:    female OR Age < 18 OR Master
  titleonly: female OR Master

Evaluated on pseudo-test and trusted surfaces with the same typed22 RF proxy.
No Kaggle submission is performed.
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
)


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
V1_AUDIT = BASE_DIR / "exports" / "wcg_audit_v1"
V14_DIR = BASE_DIR / "exports" / "v14"
EXPORT_DIR = BASE_DIR / "exports" / "v17"
TYPED = VARIANTS["typed22"]


def title(raw):
    return raw["Name"].str.extract(r",\s*([^.]*)\.", expand=False).fillna("")


def role(raw, name):
    t = title(raw)
    female = raw["Sex"].eq("female")
    master = t.eq("Master")
    if name == "current16":
        child = raw["Age"].fillna(99) < 16
    elif name == "classic15":
        child = raw["Age"].fillna(99) < 15
    elif name == "teen18":
        child = raw["Age"].fillna(99) < 18
    elif name == "titleonly":
        child = pd.Series(False, index=raw.index)
    else:
        raise ValueError(name)
    return (female | master | child).astype(int).to_numpy()


ROLES = ["current16", "classic15", "teen18", "titleonly"]


def rf(seed):
    return RandomForestClassifier(
        n_estimators=800,
        max_depth=7,
        min_samples_split=6,
        min_samples_leaf=6,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )


def eval_split(df, y, base_cols, tr_idx, va_idx, seed):
    tr = df.iloc[tr_idx].copy()
    va = df.iloc[va_idx].copy()
    af, at = eligible_groups(tr, va)
    tr_rel = relation_features(
        tr, tr, exclude_self=True, alpha=2.0,
        allowed_fam=af, allowed_tic=at
    )
    va_rel = relation_features(
        tr, va, exclude_self=False, alpha=2.0,
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
    p = m.predict_proba(xva)[:, 1]
    return accuracy_score(y[va_idx], p >= 0.5), roc_auc_score(y[va_idx], p)


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    base_df, _, base_cols, _, _ = structural_frames(train_raw, test_raw)
    y = base_df["Survived"].astype(int).to_numpy()
    manifest = pd.read_csv(V1_AUDIT / "fold_manifest_seed42.csv")
    folds = manifest["fold"].astype(int).to_numpy()
    pseudo = pd.read_csv(V14_DIR / "pseudo_test_manifest.csv")
    rows = []

    for role_name in ROLES:
        df = base_df.copy()
        df["IsWomanChild"] = role(train_raw, role_name)

        for f in sorted(np.unique(folds)):
            tr_idx = np.flatnonzero(folds != f)
            va_idx = np.flatnonzero(folds == f)
            acc, auc = eval_split(df, y, base_cols, tr_idx, va_idx, 100 + int(f))
            rows.append({
                "surface": "trusted",
                "role": role_name,
                "split": int(f),
                "accuracy": acc,
                "roc_auc": auc,
            })

        for s in sorted(pseudo["split"].unique()):
            ids = set(
                pseudo.loc[pseudo["split"] == s, "PassengerId"].astype(int)
            )
            mask = df["PassengerId"].astype(int).isin(ids).to_numpy()
            va_idx = np.flatnonzero(mask)
            tr_idx = np.flatnonzero(~mask)
            acc, auc = eval_split(df, y, base_cols, tr_idx, va_idx, 200 + int(s))
            rows.append({
                "surface": "pseudo",
                "role": role_name,
                "split": int(s),
                "accuracy": acc,
                "roc_auc": auc,
            })
        print(role_name, "done", flush=True)

    metrics = pd.DataFrame(rows)
    summary = (
        metrics.groupby(["surface", "role"])
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
        summary.groupby("role")
        .agg(
            mean_surface_accuracy=("mean_accuracy", "mean"),
            worst_surface_accuracy=("mean_accuracy", "min"),
            mean_surface_auc=("mean_auc", "mean"),
            worst_surface_auc=("mean_auc", "min"),
        )
        .reset_index()
        .sort_values(
            ["worst_surface_accuracy", "mean_surface_auc"],
            ascending=False,
        )
    )
    metrics.to_csv(EXPORT_DIR / "role_screen_metrics.csv", index=False)
    summary.to_csv(EXPORT_DIR / "role_screen_summary.csv", index=False)
    combined.to_csv(EXPORT_DIR / "role_screen_combined.csv", index=False)

    print("\n=== role screen by surface ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n=== combined ranking ===")
    print(combined.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nNo Kaggle submission was performed.")


if __name__ == "__main__":
    main()
