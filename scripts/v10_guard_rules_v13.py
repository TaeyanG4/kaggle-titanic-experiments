"""Conservative guard rules for the v10 public-score champion.

v10 remains the default prediction. We only revert a v10/v5 disagreement when
trusted models provide unusually strong contrary evidence. Rules are validated
against the fold-safe transductive RF analogue from v13 before being transferred
to real test predictions.

No Kaggle submission is performed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
V5_DIR = BASE_DIR / "exports" / "v5"
V8_DIR = BASE_DIR / "exports" / "v8"
V10_DIR = BASE_DIR / "exports" / "v10"
V13_DIR = BASE_DIR / "exports" / "v13"
SUBMISSION_DIR = BASE_DIR / "submissions"


def title(name: str) -> str:
    m = re.search(r",\s*([^.]*)\.", str(name))
    return m.group(1).strip() if m else ""


def load_trusted():
    zoo = pd.read_csv(V5_DIR / "model_zoo_oof.csv")
    zoo_test = pd.read_csv(V5_DIR / "model_zoo_test.csv")
    mlp = pd.read_csv(V5_DIR / "seed_probes" / "MLP_PLR_seed_mean__oof.csv")
    mlp_test = pd.read_csv(V5_DIR / "seed_probes" / "MLP_PLR_seed_mean__test.csv")
    t8 = pd.read_csv(V8_DIR / "tabpfn_finalist_oof.csv")
    t8_test = pd.read_csv(V8_DIR / "tabpfn_finalist_test.csv")

    oof_votes = np.column_stack(
        [
            (zoo["v4b__Champion"].to_numpy() > 0.5).astype(int),
            (zoo["RuleFit"].to_numpy() > 0.5).astype(int),
            (mlp["probability"].to_numpy() > 0.5).astype(int),
            (t8["TabPFN_v3__familyfare"].to_numpy() > 0.5).astype(int),
        ]
    )
    test_votes = np.column_stack(
        [
            (zoo_test["v4b__Champion"].to_numpy() > 0.5).astype(int),
            (zoo_test["RuleFit"].to_numpy() > 0.5).astype(int),
            (mlp_test["probability"].to_numpy() > 0.5).astype(int),
            (t8_test["TabPFN_v3__familyfare"].to_numpy() > 0.5).astype(int),
        ]
    )
    v5_oof = ((oof_votes[:, :3].sum(axis=1)) >= 2).astype(int)
    v5_test = ((test_votes[:, :3].sum(axis=1)) >= 2).astype(int)
    tab_oof = oof_votes[:, 3]
    tab_test = test_votes[:, 3]
    return zoo, oof_votes, test_votes, v5_oof, v5_test, tab_oof, tab_test


def apply_guard(
    rule: str,
    base: np.ndarray,
    v5: np.ndarray,
    tab: np.ndarray,
    votes4: np.ndarray,
    group_na: np.ndarray,
    adult_male: np.ndarray,
) -> np.ndarray:
    out = base.copy()
    disagree = base != v5
    v5_tab_agree = (v5 == tab) & disagree
    unanimous4 = ((votes4.sum(axis=1) == 0) | (votes4.sum(axis=1) == 4)) & disagree
    no_group = group_na == 0

    if rule == "guard_unanimous4":
        mask = unanimous4
    elif rule == "guard_unanimous4_no_group":
        mask = unanimous4 & no_group
    elif rule == "guard_unanimous4_adult_male":
        mask = unanimous4 & adult_male
    elif rule == "guard_unanimous4_no_group_or_adult_male":
        mask = unanimous4 & (no_group | adult_male)
    elif rule == "guard_v5_tab_no_group":
        mask = v5_tab_agree & no_group
    elif rule == "guard_v5_tab_adult_male":
        mask = v5_tab_agree & adult_male
    elif rule == "guard_v5_tab_no_group_or_adult_male":
        mask = v5_tab_agree & (no_group | adult_male)
    else:
        raise ValueError(rule)
    out[mask] = v5[mask]
    return out


RULES = [
    "guard_unanimous4",
    "guard_unanimous4_no_group",
    "guard_unanimous4_adult_male",
    "guard_unanimous4_no_group_or_adult_male",
    "guard_v5_tab_no_group",
    "guard_v5_tab_adult_male",
    "guard_v5_tab_no_group_or_adult_male",
]


def main() -> None:
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    trans_oof = pd.read_csv(V13_DIR / "transductive_rf_oof.csv")
    oof_meta = pd.read_csv(V13_DIR / "transductive_group_meta_oof.csv")
    test_meta = pd.read_csv(V13_DIR / "transductive_group_meta_test.csv")
    v10 = pd.read_csv(V10_DIR / "gunes_original_test_predictions.csv")

    zoo, oof_votes4, test_votes4, v5_oof, v5_test, tab_oof, tab_test = load_trusted()
    y = zoo["Survived"].astype(int).to_numpy()
    folds = zoo["fold"].astype(int).to_numpy()
    trans_pred = trans_oof["transductive_rf_pred"].astype(int).to_numpy()
    v10_pred = v10["Survived"].astype(int).to_numpy()

    train_title = train_raw["Name"].map(title)
    test_title = test_raw["Name"].map(title)
    train_adult_male = (
        (train_raw["Sex"] == "male")
        & (~train_title.eq("Master"))
    ).to_numpy()
    test_adult_male = (
        (test_raw["Sex"] == "male")
        & (~test_title.eq("Master"))
    ).to_numpy()

    rows = []
    test_matrix = pd.DataFrame({"PassengerId": test_raw["PassengerId"].astype(int)})
    for rule in RULES:
        oof_pred = apply_guard(
            rule,
            trans_pred,
            v5_oof,
            tab_oof,
            oof_votes4,
            oof_meta["SurvivalRateNA"].to_numpy(),
            train_adult_male,
        )
        test_pred = apply_guard(
            rule,
            v10_pred,
            v5_test,
            tab_test,
            test_votes4,
            test_meta["SurvivalRateNA"].to_numpy(),
            test_adult_male,
        )
        fold_deltas = [
            int(
                np.sum(oof_pred[folds == f] == y[folds == f])
                - np.sum(trans_pred[folds == f] == y[folds == f])
            )
            for f in sorted(np.unique(folds))
        ]
        rows.append(
            {
                "rule": rule,
                "analogue_oof_accuracy": accuracy_score(y, oof_pred),
                "delta_vs_transductive_analogue": accuracy_score(y, oof_pred)
                - accuracy_score(y, trans_pred),
                "delta_vs_v5": accuracy_score(y, oof_pred)
                - accuracy_score(y, v5_oof),
                "oof_changes_vs_analogue": int(np.sum(oof_pred != trans_pred)),
                "test_changes_vs_v10": int(np.sum(test_pred != v10_pred)),
                "test_changes_vs_v5": int(np.sum(test_pred != v5_test)),
                "fold_deltas_vs_analogue": json.dumps(fold_deltas),
            }
        )
        test_matrix[rule] = test_pred
        pd.DataFrame(
            {
                "PassengerId": test_raw["PassengerId"].astype(int),
                "Survived": test_pred,
            }
        ).to_csv(
            SUBMISSION_DIR / f"submission_v13_{rule}.csv",
            index=False,
        )

    summary = pd.DataFrame(rows).sort_values(
        ["analogue_oof_accuracy", "test_changes_vs_v10"],
        ascending=[False, True],
    )
    summary.to_csv(V13_DIR / "v10_guard_rule_summary.csv", index=False)
    test_matrix.to_csv(V13_DIR / "v10_guard_rule_test.csv", index=False)

    # Show exactly which real test rows each guard changes from v10.
    detail_rows = []
    for rule in RULES:
        pred = test_matrix[rule].to_numpy()
        mask = pred != v10_pred
        for idx in np.flatnonzero(mask):
            detail_rows.append(
                {
                    "rule": rule,
                    "PassengerId": int(test_raw.iloc[idx]["PassengerId"]),
                    "Name": test_raw.iloc[idx]["Name"],
                    "Sex": test_raw.iloc[idx]["Sex"],
                    "Age": test_raw.iloc[idx]["Age"],
                    "Pclass": int(test_raw.iloc[idx]["Pclass"]),
                    "Title": test_title.iloc[idx],
                    "v10": int(v10_pred[idx]),
                    "guard": int(pred[idx]),
                    "v5": int(v5_test[idx]),
                    "TabPFN3_FF": int(tab_test[idx]),
                    "trusted4_sum": int(test_votes4[idx].sum()),
                    "SurvivalRateNA": float(test_meta.iloc[idx]["SurvivalRateNA"]),
                    "SurvivalRate": float(test_meta.iloc[idx]["SurvivalRate"]),
                }
            )
    detail = pd.DataFrame(detail_rows)
    detail.to_csv(V13_DIR / "v10_guard_changed_rows.csv", index=False)

    print("=== v10 conservative guard rules ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n=== Changed test rows ===")
    if len(detail):
        print(detail.to_string(index=False))
    else:
        print("(none)")
    print("\nNo Kaggle submission was performed.")


if __name__ == "__main__":
    main()
