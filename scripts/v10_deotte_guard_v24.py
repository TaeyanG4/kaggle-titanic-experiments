"""Final low-DOF guard audit: preserve v10, borrow only Deotte female-death rules.

The project has one Kaggle submission slot left today.  This script is the final
promotion gate before using it.  It avoids row-by-row leaderboard tuning and
tests a tiny set of domain-declared rules on:
  1) the existing fixed-fold v10 transductive analogue; and
  2) two independent Ticket/Family connected-component group-aware splits.

Rules are deliberately asymmetric because v23 showed that Deotte's group-aware
advantage came from female overrides, while male overrides degraded performance.

No Kaggle submission is performed by this script.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from deotte_wcg_xgb_v21 import (
    DEOTTE_EXACT_FEMALE_PERISH,
    predict_split as deotte_predict_split,
)
from gunes_exact_foldsafe_v11 import structural_frames
from groupaware_validation_v23 import connected_groups
from v10_trusted_conflict_selector_v13 import (
    eligible_groups,
    group_meta,
    historical_partition_scale,
    make_rf,
)


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
V13_DIR = BASE_DIR / "exports" / "v13"
V21_DIR = BASE_DIR / "exports" / "v21"
V10_DIR = BASE_DIR / "exports" / "v10"
EXPORT_DIR = BASE_DIR / "exports" / "v24"
SUB_DIR = BASE_DIR / "submissions"


RULES = [
    "wcg_female_death",
    "all_deotte_female_death",
    "wcg_core",
    "all_female_death_plus_wcg_boys",
]


def apply_rule(base: np.ndarray, raw: pd.DataFrame, d, rule: str) -> np.ndarray:
    out = base.copy()
    female = raw["Sex"].eq("female").to_numpy()
    boy = raw["Name"].str.contains("Master", regex=False, na=False).to_numpy()
    wcg_female_death = female & (d.wcg == 0)
    all_female_death = female & (d.wcg_both == 0)
    wcg_boy_live = boy & (d.wcg == 1)

    if rule == "wcg_female_death":
        out[wcg_female_death] = 0
    elif rule == "all_deotte_female_death":
        out[all_female_death] = 0
    elif rule == "wcg_core":
        out[wcg_female_death] = 0
        out[wcg_boy_live] = 1
    elif rule == "all_female_death_plus_wcg_boys":
        out[all_female_death] = 0
        out[wcg_boy_live] = 1
    else:
        raise KeyError(rule)
    return out


def transductive_fold_predict(train_df, base_cols, tr_idx, va_idx):
    y = train_df["Survived"].astype(int).to_numpy()
    tr = train_df.iloc[tr_idx].copy()
    va = train_df.iloc[va_idx].copy()
    fam, tic = eligible_groups(tr, va)
    tr_meta = group_meta(tr, tr, fam, tic, exclude_self=True)
    va_meta = group_meta(tr, va, fam, tic, exclude_self=False)
    xtr = np.column_stack([
        tr[base_cols].to_numpy(float),
        tr_meta[["SurvivalRate", "SurvivalRateNA"]].to_numpy(),
    ])
    xva = np.column_stack([
        va[base_cols].to_numpy(float),
        va_meta[["SurvivalRate", "SurvivalRateNA"]].to_numpy(),
    ])
    xtr = historical_partition_scale(xtr)
    xva = historical_partition_scale(xva)
    model = make_rf()
    model.fit(xtr, y[tr_idx])
    prob = model.predict_proba(xva)[:, 1]
    return (prob >= 0.5).astype(int), prob


def fixed_audit(train_raw: pd.DataFrame) -> pd.DataFrame:
    base_df = pd.read_csv(V13_DIR / "transductive_rf_oof.csv")
    de = pd.read_csv(V21_DIR / "deotte_fixed_oof.csv")
    if not base_df["PassengerId"].astype(int).equals(de["PassengerId"].astype(int)):
        raise ValueError("fixed OOF PassengerId mismatch")
    y = de["Survived"].astype(int).to_numpy()
    base = base_df["transductive_rf_pred"].astype(int).to_numpy()
    raw = train_raw.reset_index(drop=True)

    # Build a light object with the arrays apply_rule needs.
    class D: pass
    d = D()
    d.wcg = de["wcg"].astype(int).to_numpy()
    d.wcg_both = de["wcg_both"].astype(int).to_numpy()

    rows = [{
        "surface": "fixed",
        "rule": "base",
        "accuracy": accuracy_score(y, base),
        "delta": 0.0,
        "changed": 0,
    }]
    for rule in RULES:
        p = apply_rule(base, raw, d, rule)
        rows.append({
            "surface": "fixed",
            "rule": rule,
            "accuracy": accuracy_score(y, p),
            "delta": accuracy_score(y, p) - accuracy_score(y, base),
            "changed": int(np.sum(p != base)),
        })
    return pd.DataFrame(rows)


def groupaware_audit(train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> pd.DataFrame:
    train_df, _, base_cols, _, _ = structural_frames(train_raw, test_raw)
    y = train_df["Survived"].astype(int).to_numpy()
    groups = connected_groups(train_raw)
    rows = []

    for seed in [42, 777]:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        base_oof = np.zeros(len(train_raw), dtype=int)
        prob_oof = np.zeros(len(train_raw), dtype=float)
        rule_oof = {r: np.zeros(len(train_raw), dtype=int) for r in RULES}
        for fold, (tr_idx, va_idx) in enumerate(sgkf.split(train_raw, y, groups)):
            bp, bprob = transductive_fold_predict(train_df, base_cols, tr_idx, va_idx)
            base_oof[va_idx] = bp
            prob_oof[va_idx] = bprob
            d = deotte_predict_split(
                train_raw,
                tr_idx,
                va_idx,
                seed=24000 + seed + fold,
                imputer_fit_indices=tr_idx,
                special_link=True,
            )
            raw_va = train_raw.iloc[va_idx].reset_index(drop=True)
            for rule in RULES:
                rule_oof[rule][va_idx] = apply_rule(bp, raw_va, d, rule)

        base_acc = accuracy_score(y, base_oof)
        rows.append({
            "surface": f"group_seed{seed}",
            "rule": "base",
            "accuracy": base_acc,
            "delta": 0.0,
            "changed": 0,
            "base_auc": roc_auc_score(y, prob_oof),
        })
        for rule in RULES:
            p = rule_oof[rule]
            rows.append({
                "surface": f"group_seed{seed}",
                "rule": rule,
                "accuracy": accuracy_score(y, p),
                "delta": accuracy_score(y, p) - base_acc,
                "changed": int(np.sum(p != base_oof)),
                "base_auc": np.nan,
            })
    return pd.DataFrame(rows)


def build_test_candidates(test_raw: pd.DataFrame) -> pd.DataFrame:
    v10 = pd.read_csv(V10_DIR / "gunes_original_test_predictions.csv").sort_values("PassengerId").reset_index(drop=True)
    d = pd.read_csv(V21_DIR / "deotte_test_predictions.csv").sort_values("PassengerId").reset_index(drop=True)
    raw = test_raw.sort_values("PassengerId").reset_index(drop=True)
    if not v10["PassengerId"].astype(int).equals(raw["PassengerId"].astype(int)):
        raise ValueError("v10/test order mismatch")
    base = v10["Survived"].astype(int).to_numpy()

    # Historical exact female XGB outputs replace Python-port female XGB rows.
    wcg = d["wcg"].astype(int).to_numpy()
    exact_both = wcg.copy()
    pid_pos = {int(pid): i for i, pid in enumerate(raw["PassengerId"].astype(int))}
    for pid in DEOTTE_EXACT_FEMALE_PERISH:
        exact_both[pid_pos[pid]] = 0
    class D: pass
    dd = D(); dd.wcg = wcg; dd.wcg_both = exact_both

    out = pd.DataFrame({"PassengerId": raw["PassengerId"].astype(int), "v10": base})
    for rule in RULES:
        p = apply_rule(base, raw, dd, rule)
        out[rule] = p
        pd.DataFrame({"PassengerId": out["PassengerId"], "Survived": p}).to_csv(
            SUB_DIR / f"submission_v24_v10_deotte_{rule}.csv", index=False
        )
    return out


def main():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    fixed = fixed_audit(train_raw)
    group = groupaware_audit(train_raw, test_raw)
    metrics = pd.concat([fixed, group], ignore_index=True)
    metrics.to_csv(EXPORT_DIR / "v10_deotte_guard_metrics.csv", index=False)

    summary = []
    for rule in RULES:
        r = metrics[metrics["rule"] == rule]
        summary.append({
            "rule": rule,
            "mean_delta": float(r["delta"].mean()),
            "min_delta": float(r["delta"].min()),
            "positive_surfaces": int((r["delta"] > 0).sum()),
            "nonnegative_surfaces": int((r["delta"] >= 0).sum()),
            "mean_changed": float(r["changed"].mean()),
            "deltas": str([round(float(x), 6) for x in r["delta"]]),
        })
    summary = pd.DataFrame(summary).sort_values(
        ["min_delta", "mean_delta", "mean_changed"], ascending=[False, False, True]
    )
    summary.to_csv(EXPORT_DIR / "v10_deotte_guard_summary.csv", index=False)

    test = build_test_candidates(test_raw)
    comp = []
    for rule in RULES:
        comp.append({
            "rule": rule,
            "changed_test_vs_v10": int(np.sum(test[rule].to_numpy() != test["v10"].to_numpy())),
            "test_positives": int(test[rule].sum()),
        })
    pd.DataFrame(comp).to_csv(EXPORT_DIR / "v10_deotte_guard_test_comparison.csv", index=False)
    test.to_csv(EXPORT_DIR / "v10_deotte_guard_test.csv", index=False)

    print("=== validation surfaces ===")
    print(metrics.to_string(index=False, float_format=lambda x:f"{x:.5f}"))
    print("\n=== rule summary ===")
    print(summary.to_string(index=False, float_format=lambda x:f"{x:.5f}"))
    print("\n=== test changes vs v10 ===")
    print(pd.DataFrame(comp).to_string(index=False))
    print("\nNo Kaggle submission was performed.")


if __name__ == "__main__":
    main()
