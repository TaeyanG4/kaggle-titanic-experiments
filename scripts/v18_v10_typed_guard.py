"""Guard the public v10 champion with the strongest typed-relation consensus.

OOF analogue:
  v13 fold-safe transductive RF stands in for v10.

Typed trusted models:
  - TabPFN v3 + FamilyFare + typed22(alpha=2)
  - TabPFN v2.5 + FamilyFare + typed22(alpha=2)
  - CatBoost + FamilyFare + typed22(alpha=2)

Rules are low-degree switches from the transductive base when independent typed
models agree on the opposite class, optionally restricted by relation strength.

On real test, the same rule is applied to the actual v10 prediction.
No Kaggle submission is performed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


BASE_DIR = Path(__file__).resolve().parents[1]
V5_DIR = BASE_DIR / "exports" / "v5"
V10_DIR = BASE_DIR / "exports" / "v10"
V13_DIR = BASE_DIR / "exports" / "v13"
V14_TAB = BASE_DIR / "exports" / "v14" / "tabpfn"
V15_DIR = BASE_DIR / "exports" / "v15"
SUBMISSION_DIR = BASE_DIR / "submissions"


def apply(base, t3, t25, cat, na, rule):
    out = base.copy()
    all3 = (t3 == t25) & (t3 == cat) & (t3 != base)
    t3cat = (t3 == cat) & (t3 != base)
    if rule == "all3":
        mask = all3
    elif rule == "all3_weak":
        mask = all3 & (na < 1.0)
    elif rule == "all3_none":
        mask = all3 & (na == 0.0)
    elif rule == "t3cat_weak":
        mask = t3cat & (na < 1.0)
    elif rule == "t3cat_none":
        mask = t3cat & (na == 0.0)
    else:
        raise ValueError(rule)
    out[mask] = t3[mask]
    return out


RULES = ["all3", "all3_weak", "all3_none", "t3cat_weak", "t3cat_none"]


def main() -> None:
    trans = pd.read_csv(V13_DIR / "transductive_rf_oof.csv")
    trans_test = pd.read_csv(V13_DIR / "transductive_rf_test.csv")
    oof_meta = pd.read_csv(V13_DIR / "transductive_group_meta_oof.csv")
    test_meta = pd.read_csv(V13_DIR / "transductive_group_meta_test.csv")
    tab = pd.read_csv(V14_TAB / "tabpfn_typed_oof.csv")
    tabt = pd.read_csv(V14_TAB / "tabpfn_typed_test.csv")
    tree = pd.read_csv(V15_DIR / "typed_tree_screen_oof.csv")
    treet = pd.read_csv(V15_DIR / "typed_tree_screen_test.csv")
    zoo = pd.read_csv(V5_DIR / "model_zoo_oof.csv")
    zoot = pd.read_csv(V5_DIR / "model_zoo_test.csv")
    mlp = pd.read_csv(V5_DIR / "seed_probes" / "MLP_PLR_seed_mean__oof.csv")
    mlpt = pd.read_csv(V5_DIR / "seed_probes" / "MLP_PLR_seed_mean__test.csv")
    v10 = pd.read_csv(V10_DIR / "gunes_original_test_predictions.csv")

    y = trans["Survived"].astype(int).to_numpy()
    folds = trans["fold"].astype(int).to_numpy()
    base = trans["transductive_rf_pred"].astype(int).to_numpy()
    base_test = v10["Survived"].astype(int).to_numpy()
    na = oof_meta["SurvivalRateNA"].to_numpy()
    nat = test_meta["SurvivalRateNA"].to_numpy()

    t3 = (
        tab["TabPFN_v3__v2base_familyfare_typed22"].to_numpy() > 0.5
    ).astype(int)
    t25 = (
        tab["TabPFN_v2_5__v2base_familyfare_typed22"].to_numpy() > 0.5
    ).astype(int)
    cat = (
        tree["v2base_familyfare_typed22__CatBoost"].to_numpy() > 0.5
    ).astype(int)
    t3t = (
        tabt["TabPFN_v3__v2base_familyfare_typed22"].to_numpy() > 0.5
    ).astype(int)
    t25t = (
        tabt["TabPFN_v2_5__v2base_familyfare_typed22"].to_numpy() > 0.5
    ).astype(int)
    catt = (
        treet["v2base_familyfare_typed22__CatBoost"].to_numpy() > 0.5
    ).astype(int)

    v5 = (
        (
            (zoo["v4b__Champion"].to_numpy() > 0.5).astype(int)
            + (zoo["RuleFit"].to_numpy() > 0.5).astype(int)
            + (mlp["probability"].to_numpy() > 0.5).astype(int)
        )
        >= 2
    ).astype(int)
    v5t = (
        (
            (zoot["v4b__Champion"].to_numpy() > 0.5).astype(int)
            + (zoot["RuleFit"].to_numpy() > 0.5).astype(int)
            + (mlpt["probability"].to_numpy() > 0.5).astype(int)
        )
        >= 2
    ).astype(int)

    rows = [{
        "rule": "v10_analogue_base",
        "analogue_accuracy": accuracy_score(y, base),
        "delta_vs_analogue": 0.0,
        "delta_vs_v5": accuracy_score(y, base) - accuracy_score(y, v5),
        "changed_oof_vs_analogue": 0,
        "changed_test_vs_v10": 0,
        "changed_test_vs_v5": int(np.sum(base_test != v5t)),
        "fold_deltas_vs_analogue": "[0, 0, 0, 0, 0]",
    }]
    out = pd.DataFrame({"PassengerId": v10["PassengerId"].astype(int)})
    out["v10"] = base_test

    for rule in RULES:
        pred = apply(base, t3, t25, cat, na, rule)
        predt = apply(base_test, t3t, t25t, catt, nat, rule)
        fd = [
            int(
                np.sum(pred[folds == f] == y[folds == f])
                - np.sum(base[folds == f] == y[folds == f])
            )
            for f in sorted(np.unique(folds))
        ]
        rows.append({
            "rule": rule,
            "analogue_accuracy": accuracy_score(y, pred),
            "delta_vs_analogue": accuracy_score(y, pred) - accuracy_score(y, base),
            "delta_vs_v5": accuracy_score(y, pred) - accuracy_score(y, v5),
            "changed_oof_vs_analogue": int(np.sum(pred != base)),
            "changed_test_vs_v10": int(np.sum(predt != base_test)),
            "changed_test_vs_v5": int(np.sum(predt != v5t)),
            "fold_deltas_vs_analogue": str(fd),
        })
        out[rule] = predt
        pd.DataFrame({
            "PassengerId": v10["PassengerId"].astype(int),
            "Survived": predt,
        }).to_csv(
            SUBMISSION_DIR / f"submission_v18_v10_guard_{rule}.csv",
            index=False,
        )

    summary = pd.DataFrame(rows).sort_values(
        ["analogue_accuracy", "changed_test_vs_v10"],
        ascending=[False, True],
    )
    summary.to_csv(V15_DIR / "v18_v10_guard_summary.csv", index=False)
    out.to_csv(V15_DIR / "v18_v10_guard_test.csv", index=False)

    print("=== v18 v10 typed guard ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nChanged real-test rows for best rule:")
    best = summary.iloc[0]["rule"]
    if best != "v10_analogue_base":
        m = out[best].to_numpy() != base_test
        print(
            pd.DataFrame({
                "PassengerId": out.loc[m, "PassengerId"],
                "v10": base_test[m],
                best: out.loc[m, best],
                "v5": v5t[m],
                "relation_NA": nat[m],
            }).to_string(index=False)
        )
    print("\nNo Kaggle submission was performed.")


if __name__ == "__main__":
    main()
