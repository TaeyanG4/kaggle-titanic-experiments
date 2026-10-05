"""Ensemble probe for v14 typed-relation TabPFN finalists.

Tests only low-degree combinations:
  - replace one member of the trusted v5 3-vote ensemble
  - 5-member majority with v2.5/v3 typed finalists
  - consensus switches from v5 when both new TabPFNs agree
  - stricter consensus including the typed-relation RF
  - simple cross-fitted logistic meta model

No Kaggle submission is performed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


BASE_DIR = Path(__file__).resolve().parents[1]
V5_DIR = BASE_DIR / "exports" / "v5"
V14_DIR = BASE_DIR / "exports" / "v14"
TAB_DIR = V14_DIR / "tabpfn"
SUBMISSION_DIR = BASE_DIR / "submissions"


def majority(arrays: list[np.ndarray]) -> np.ndarray:
    mat = np.column_stack(arrays)
    return (mat.sum(axis=1) >= (mat.shape[1] // 2 + 1)).astype(int)


def main() -> None:
    zoo = pd.read_csv(V5_DIR / "model_zoo_oof.csv")
    zoo_test = pd.read_csv(V5_DIR / "model_zoo_test.csv")
    mlp = pd.read_csv(V5_DIR / "seed_probes" / "MLP_PLR_seed_mean__oof.csv")
    mlp_test = pd.read_csv(V5_DIR / "seed_probes" / "MLP_PLR_seed_mean__test.csv")
    tab = pd.read_csv(TAB_DIR / "tabpfn_typed_oof.csv")
    tab_test = pd.read_csv(TAB_DIR / "tabpfn_typed_test.csv")
    rf = pd.read_csv(V14_DIR / "typed_relational_fixed_oof.csv")
    rf_test = pd.read_csv(V14_DIR / "typed_relational_fixed_test.csv")

    ids = zoo["PassengerId"]
    test_ids = zoo_test["PassengerId"]
    for df in [mlp, tab, rf]:
        if not ids.equals(df["PassengerId"]):
            raise ValueError("OOF PassengerId mismatch.")
    for df in [mlp_test, tab_test, rf_test]:
        if not test_ids.equals(df["PassengerId"]):
            raise ValueError("Test PassengerId mismatch.")

    y = zoo["Survived"].astype(int).to_numpy()
    folds = zoo["fold"].astype(int).to_numpy()

    probs = {
        "v4b": zoo["v4b__Champion"].to_numpy(),
        "rule": zoo["RuleFit"].to_numpy(),
        "mlp": mlp["probability"].to_numpy(),
        "t3": tab[
            "TabPFN_v3__v2base_familyfare_typed22"
        ].to_numpy(),
        "t25": tab[
            "TabPFN_v2_5__v2base_familyfare_typed22"
        ].to_numpy(),
        "typedrf": rf["typed22"].to_numpy(),
    }
    test_probs = {
        "v4b": zoo_test["v4b__Champion"].to_numpy(),
        "rule": zoo_test["RuleFit"].to_numpy(),
        "mlp": mlp_test["probability"].to_numpy(),
        "t3": tab_test[
            "TabPFN_v3__v2base_familyfare_typed22"
        ].to_numpy(),
        "t25": tab_test[
            "TabPFN_v2_5__v2base_familyfare_typed22"
        ].to_numpy(),
        "typedrf": rf_test["typed22"].to_numpy(),
    }
    votes = {k: (v > 0.5).astype(int) for k, v in probs.items()}
    test_votes = {
        k: (v > 0.5).astype(int) for k, v in test_probs.items()
    }

    v5 = majority([votes["v4b"], votes["rule"], votes["mlp"]])
    v5_test = majority(
        [test_votes["v4b"], test_votes["rule"], test_votes["mlp"]]
    )
    base_acc = accuracy_score(y, v5)

    candidates: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}

    # 3-member replacements.
    candidates["v4b_rule_t3"] = (
        majority([votes["v4b"], votes["rule"], votes["t3"]]),
        majority(
            [test_votes["v4b"], test_votes["rule"], test_votes["t3"]]
        ),
        "hard_vote",
    )
    candidates["v4b_mlp_t3"] = (
        majority([votes["v4b"], votes["mlp"], votes["t3"]]),
        majority(
            [test_votes["v4b"], test_votes["mlp"], test_votes["t3"]]
        ),
        "hard_vote",
    )
    candidates["rule_mlp_t3"] = (
        majority([votes["rule"], votes["mlp"], votes["t3"]]),
        majority(
            [test_votes["rule"], test_votes["mlp"], test_votes["t3"]]
        ),
        "hard_vote",
    )

    # 5-member majority.
    candidates["vote5_v5_t3_t25"] = (
        majority(
            [
                votes["v4b"],
                votes["rule"],
                votes["mlp"],
                votes["t3"],
                votes["t25"],
            ]
        ),
        majority(
            [
                test_votes["v4b"],
                test_votes["rule"],
                test_votes["mlp"],
                test_votes["t3"],
                test_votes["t25"],
            ]
        ),
        "hard_vote",
    )

    # Consensus switches: keep v5 unless independent new models agree opposite.
    both_tab = v5.copy()
    both_tab_test = v5_test.copy()
    m = (votes["t3"] == votes["t25"]) & (votes["t3"] != v5)
    mt = (
        (test_votes["t3"] == test_votes["t25"])
        & (test_votes["t3"] != v5_test)
    )
    both_tab[m] = votes["t3"][m]
    both_tab_test[mt] = test_votes["t3"][mt]
    candidates["switch_if_t3_t25_agree"] = (
        both_tab,
        both_tab_test,
        "consensus_switch",
    )

    strict = v5.copy()
    strict_test = v5_test.copy()
    m = (
        (votes["t3"] == votes["t25"])
        & (votes["t3"] == votes["typedrf"])
        & (votes["t3"] != v5)
    )
    mt = (
        (test_votes["t3"] == test_votes["t25"])
        & (test_votes["t3"] == test_votes["typedrf"])
        & (test_votes["t3"] != v5_test)
    )
    strict[m] = votes["t3"][m]
    strict_test[mt] = test_votes["t3"][mt]
    candidates["switch_if_t3_t25_rf_agree"] = (
        strict,
        strict_test,
        "consensus_switch",
    )

    # Cross-fitted logistic on probabilities.
    meta_cols = ["v4b", "rule", "mlp", "t3", "t25", "typedrf"]
    X = np.column_stack([probs[c] for c in meta_cols])
    Xtest = np.column_stack([test_probs[c] for c in meta_cols])
    meta_oof = np.zeros(len(y), dtype=float)
    fold_test = []
    for fold in sorted(np.unique(folds)):
        tr = folds != fold
        va = folds == fold
        meta = LogisticRegression(
            C=0.15,
            solver="liblinear",
            max_iter=1000,
            random_state=42,
        )
        meta.fit(X[tr], y[tr])
        meta_oof[va] = meta.predict_proba(X[va])[:, 1]
        fold_test.append(meta.predict_proba(Xtest)[:, 1])
    meta_test = np.mean(fold_test, axis=0)
    candidates["stack6_crossfit"] = (
        (meta_oof > 0.5).astype(int),
        (meta_test > 0.5).astype(int),
        "crossfit_logistic",
    )

    rows = [
        {
            "candidate": "v5_robust",
            "kind": "baseline",
            "accuracy": base_acc,
            "delta_vs_v5": 0.0,
            "changed_oof_vs_v5": 0,
            "changed_test_vs_v5": 0,
            "fold_deltas_vs_v5": "[0, 0, 0, 0, 0]",
        }
    ]

    out_test = pd.DataFrame({"PassengerId": test_ids})
    out_oof = pd.DataFrame(
        {"PassengerId": ids, "fold": folds, "Survived": y}
    )
    out_test["v5_robust"] = v5_test
    out_oof["v5_robust"] = v5

    for name, (pred, test_pred, kind) in candidates.items():
        fold_deltas = [
            int(
                np.sum(pred[folds == f] == y[folds == f])
                - np.sum(v5[folds == f] == y[folds == f])
            )
            for f in sorted(np.unique(folds))
        ]
        rows.append(
            {
                "candidate": name,
                "kind": kind,
                "accuracy": accuracy_score(y, pred),
                "delta_vs_v5": accuracy_score(y, pred) - base_acc,
                "changed_oof_vs_v5": int(np.sum(pred != v5)),
                "changed_test_vs_v5": int(
                    np.sum(test_pred != v5_test)
                ),
                "fold_deltas_vs_v5": str(fold_deltas),
            }
        )
        out_oof[name] = pred
        out_test[name] = test_pred
        pd.DataFrame(
            {
                "PassengerId": test_ids.astype(int),
                "Survived": test_pred.astype(int),
            }
        ).to_csv(
            SUBMISSION_DIR / f"submission_v14_{name}.csv",
            index=False,
        )

    # Add meta probability diagnostics separately.
    out_oof["stack6_probability"] = meta_oof
    out_test["stack6_probability"] = meta_test

    summary = pd.DataFrame(rows).sort_values(
        ["accuracy", "changed_oof_vs_v5"], ascending=[False, True]
    )
    summary.to_csv(V14_DIR / "v14_ensemble_summary.csv", index=False)
    out_oof.to_csv(V14_DIR / "v14_ensemble_oof.csv", index=False)
    out_test.to_csv(V14_DIR / "v14_ensemble_test.csv", index=False)

    print("=== v14 ensemble probe ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(
        "\nstack6 AUC:",
        f"{roc_auc_score(y, meta_oof):.5f}",
    )
    print("\nNo Kaggle submission was performed.")


if __name__ == "__main__":
    main()
