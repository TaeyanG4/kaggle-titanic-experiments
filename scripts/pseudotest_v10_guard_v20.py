"""Pseudo-test stress validation for the v18 v10 typed guard rules.

Reuses the exact five matched pseudo-test splits and the typed model predictions
saved by ``pseudotest_consensus_v19.py``.  Only the historical v10 analogue
(Gunes 24 structural features + LegacyRate/LegacyNA + RF) is retrained here.

No Kaggle submission is performed by this script.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from gunes_exact_foldsafe_v11 import structural_frames
from pseudo_test_relational_v14 import (
    build_pseudo_splits,
    eligible_groups,
    make_rf,
    relation_features,
    role_flags,
)
from v18_v10_typed_guard import RULES, apply


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
V19_DIR = BASE_DIR / "exports" / "v19"
EXPORT_DIR = BASE_DIR / "exports" / "v20"


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    train_g, _, base_cols, _, _ = structural_frames(train_raw, test_raw)
    train_g["IsWomanChild"] = role_flags(train_raw)
    y = train_g["Survived"].astype(int).to_numpy()

    typed = pd.read_csv(V19_DIR / "pseudotest_consensus_predictions.csv")
    _, pseudo = build_pseudo_splits(train_raw, test_raw)

    rows = []
    detail = []
    for split, (_, va_idx, _) in enumerate(pseudo):
        tr_idx = np.setdiff1d(np.arange(len(train_g)), va_idx)
        tr = train_g.iloc[tr_idx].copy()
        va = train_g.iloc[va_idx].copy()

        af, at = eligible_groups(tr, va)
        tr_rel = relation_features(
            tr, tr, exclude_self=True, alpha=2.0, allowed_fam=af, allowed_tic=at
        )
        va_rel = relation_features(
            tr, va, exclude_self=False, alpha=2.0, allowed_fam=af, allowed_tic=at
        )
        rel_cols = ["LegacyRate", "LegacyNA"]
        xtr = np.column_stack([tr[base_cols].to_numpy(dtype=float), tr_rel[rel_cols].to_numpy()])
        xva = np.column_stack([va[base_cols].to_numpy(dtype=float), va_rel[rel_cols].to_numpy()])
        # Match historical v10 preprocessing behavior.
        xtr = StandardScaler().fit_transform(xtr)
        xva = StandardScaler().fit_transform(xva)
        model = make_rf()
        model.fit(xtr, y[tr_idx])
        base_prob = model.predict_proba(xva)[:, 1]
        base = (base_prob >= 0.5).astype(int)

        tp = typed[typed["split"] == split].copy()
        expected_ids = va["PassengerId"].astype(int).to_numpy()
        if not np.array_equal(tp["PassengerId"].astype(int).to_numpy(), expected_ids):
            raise ValueError(f"split {split}: v19 prediction order mismatch")
        t3 = tp["t3"].astype(int).to_numpy()
        t25 = tp["t25"].astype(int).to_numpy()
        cat = tp["cat"].astype(int).to_numpy()
        na = va_rel["LegacyNA"].to_numpy()
        yv = y[va_idx]

        base_acc = accuracy_score(yv, base)
        rows.append(
            {
                "split": split,
                "rule": "base",
                "accuracy": base_acc,
                "roc_auc": roc_auc_score(yv, base_prob),
                "delta_vs_base": 0.0,
                "changed": 0,
            }
        )

        for rule in RULES:
            pred = apply(base, t3, t25, cat, na, rule)
            acc = accuracy_score(yv, pred)
            rows.append(
                {
                    "split": split,
                    "rule": rule,
                    "accuracy": acc,
                    "roc_auc": np.nan,
                    "delta_vs_base": acc - base_acc,
                    "changed": int(np.sum(pred != base)),
                }
            )
            for i, pid in enumerate(expected_ids):
                if pred[i] != base[i]:
                    detail.append(
                        {
                            "split": split,
                            "rule": rule,
                            "PassengerId": int(pid),
                            "Survived": int(yv[i]),
                            "base": int(base[i]),
                            "guard": int(pred[i]),
                            "t3": int(t3[i]),
                            "t25": int(t25[i]),
                            "cat": int(cat[i]),
                            "LegacyNA": float(na[i]),
                        }
                    )

        best = max(
            [r for r in rows if r["split"] == split],
            key=lambda z: (z["accuracy"], -z["changed"]),
        )
        print(
            f"split={split} base={base_acc:.5f} best={best['rule']} "
            f"acc={best['accuracy']:.5f} delta={best['delta_vs_base']:+.5f}",
            flush=True,
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(EXPORT_DIR / "pseudotest_v10_guard_metrics.csv", index=False)
    pd.DataFrame(detail).to_csv(EXPORT_DIR / "pseudotest_v10_guard_changes.csv", index=False)

    summary_rows = []
    base_by_split = metrics[metrics["rule"] == "base"].set_index("split")["accuracy"]
    for rule in RULES:
        r = metrics[metrics["rule"] == rule].set_index("split")
        d = r["accuracy"] - base_by_split
        summary_rows.append(
            {
                "rule": rule,
                "mean_accuracy": float(r["accuracy"].mean()),
                "mean_delta_vs_base": float(d.mean()),
                "positive_splits": int((d > 0).sum()),
                "nonnegative_splits": int((d >= 0).sum()),
                "negative_splits": int((d < 0).sum()),
                "mean_changed": float(r["changed"].mean()),
                "deltas": str([round(float(x), 6) for x in d.tolist()]),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["mean_delta_vs_base", "negative_splits", "mean_changed"],
        ascending=[False, True, True],
    )
    summary.to_csv(EXPORT_DIR / "pseudotest_v10_guard_summary.csv", index=False)
    print("\n=== pseudo-test v10 guard summary ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nNo Kaggle submission was performed.")


if __name__ == "__main__":
    main()
