"""Ensemble probe using newly unlocked TabPFN v2.5/v3 candidates.

This is intentionally small and predeclared. It tests:
  * 3-member hard-vote replacements
  * 5-member majority votes
  * simple equal-probability averages
  * cross-fitted logistic stacking on the same preserved outer folds

No Kaggle submission is performed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


BASE_DIR = Path(__file__).resolve().parents[1]
V5_DIR = BASE_DIR / "exports" / "v5"
V8_DIR = BASE_DIR / "exports" / "v8"
V12_DIR = BASE_DIR / "exports" / "v12"
SUBMISSION_DIR = BASE_DIR / "submissions"


def load_inputs():
    zoo = pd.read_csv(V5_DIR / "model_zoo_oof.csv")
    zoo_test = pd.read_csv(V5_DIR / "model_zoo_test.csv")
    mlp = pd.read_csv(V5_DIR / "seed_probes" / "MLP_PLR_seed_mean__oof.csv")
    mlp_test = pd.read_csv(V5_DIR / "seed_probes" / "MLP_PLR_seed_mean__test.csv")
    t8 = pd.read_csv(V8_DIR / "tabpfn_finalist_oof.csv")
    t8_test = pd.read_csv(V8_DIR / "tabpfn_finalist_test.csv")
    t12 = pd.read_csv(V12_DIR / "tabpfn_fe_oof.csv")
    t12_test = pd.read_csv(V12_DIR / "tabpfn_fe_test.csv")

    ids = zoo["PassengerId"]
    for df, name in [(mlp, "mlp"), (t8, "v8"), (t12, "v12")]:
        if not ids.equals(df["PassengerId"]):
            raise ValueError(f"OOF PassengerId mismatch: {name}")
    test_ids = zoo_test["PassengerId"]
    for df, name in [(mlp_test, "mlp_test"), (t8_test, "v8_test"), (t12_test, "v12_test")]:
        if not test_ids.equals(df["PassengerId"]):
            raise ValueError(f"Test PassengerId mismatch: {name}")

    oof = pd.DataFrame(
        {
            "PassengerId": ids,
            "fold": zoo["fold"].astype(int),
            "Survived": zoo["Survived"].astype(int),
            "v4b": zoo["v4b__Champion"],
            "RuleFit": zoo["RuleFit"],
            "MLPMean": mlp["probability"],
            "TabPFN25_FF": t8["TabPFN_v2_5__familyfare"],
            "TabPFN3_FF": t8["TabPFN_v3__familyfare"],
            "TabPFN3_FF_GunesAll": t12["TabPFN_v3__familyfare_gunes_all"],
            "TabPFN25_GunesRates": t12["TabPFN_v2_5__gunes_rates"],
            "TabPFN3_GunesRates": t12["TabPFN_v3__gunes_rates"],
        }
    )
    test = pd.DataFrame(
        {
            "PassengerId": test_ids,
            "v4b": zoo_test["v4b__Champion"],
            "RuleFit": zoo_test["RuleFit"],
            "MLPMean": mlp_test["probability"],
            "TabPFN25_FF": t8_test["TabPFN_v2_5__familyfare"],
            "TabPFN3_FF": t8_test["TabPFN_v3__familyfare"],
            "TabPFN3_FF_GunesAll": t12_test["TabPFN_v3__familyfare_gunes_all"],
            "TabPFN25_GunesRates": t12_test["TabPFN_v2_5__gunes_rates"],
            "TabPFN3_GunesRates": t12_test["TabPFN_v3__gunes_rates"],
        }
    )
    return oof, test


HARD_VOTES = {
    "v5_robust": ["v4b", "RuleFit", "MLPMean"],
    "v4b_rule_t3ff": ["v4b", "RuleFit", "TabPFN3_FF"],
    "v4b_mlp_t3ff": ["v4b", "MLPMean", "TabPFN3_FF"],
    "rule_mlp_t3ff": ["RuleFit", "MLPMean", "TabPFN3_FF"],
    "v4b_rule_t3ffga": ["v4b", "RuleFit", "TabPFN3_FF_GunesAll"],
    "v4b_mlp_t3ffga": ["v4b", "MLPMean", "TabPFN3_FF_GunesAll"],
    "rule_mlp_t3ffga": ["RuleFit", "MLPMean", "TabPFN3_FF_GunesAll"],
    "vote5_v5_t25ff_t3ff": [
        "v4b", "RuleFit", "MLPMean", "TabPFN25_FF", "TabPFN3_FF"
    ],
    "vote5_v5_t3ff_t3ffga": [
        "v4b", "RuleFit", "MLPMean", "TabPFN3_FF", "TabPFN3_FF_GunesAll"
    ],
    "vote5_v5_t25gr_t3ffga": [
        "v4b", "RuleFit", "MLPMean", "TabPFN25_GunesRates", "TabPFN3_FF_GunesAll"
    ],
}

SOFT_AVERAGES = {
    "soft4_v5_t3ff": ["v4b", "RuleFit", "MLPMean", "TabPFN3_FF"],
    "soft4_v5_t3ffga": ["v4b", "RuleFit", "MLPMean", "TabPFN3_FF_GunesAll"],
    "soft5_v5_t25ff_t3ff": [
        "v4b", "RuleFit", "MLPMean", "TabPFN25_FF", "TabPFN3_FF"
    ],
}

STACKS = {
    "stack_v5_t25ff_t3ff": [
        "v4b", "RuleFit", "MLPMean", "TabPFN25_FF", "TabPFN3_FF"
    ],
    "stack_v5_t3ff_t3ffga": [
        "v4b", "RuleFit", "MLPMean", "TabPFN3_FF", "TabPFN3_FF_GunesAll"
    ],
    "stack_v5_t25ff_t3ff_t3ffga": [
        "v4b", "RuleFit", "MLPMean",
        "TabPFN25_FF", "TabPFN3_FF", "TabPFN3_FF_GunesAll"
    ],
}


def majority(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    votes = np.column_stack([(df[c].to_numpy() > 0.5).astype(int) for c in cols])
    return (votes.sum(axis=1) >= (len(cols) // 2 + 1)).astype(int)


def main() -> None:
    oof, test = load_inputs()
    y = oof["Survived"].to_numpy()
    folds = oof["fold"].to_numpy()
    results = []
    pred_export = pd.DataFrame(
        {"PassengerId": oof["PassengerId"], "fold": folds, "Survived": y}
    )
    test_export = pd.DataFrame({"PassengerId": test["PassengerId"]})

    base_pred = majority(oof, HARD_VOTES["v5_robust"])
    base_acc = accuracy_score(y, base_pred)

    for name, cols in HARD_VOTES.items():
        pred = majority(oof, cols)
        te_pred = majority(test, cols)
        results.append(
            {
                "candidate": name,
                "kind": "hard_vote",
                "accuracy": accuracy_score(y, pred),
                "roc_auc": np.nan,
                "delta_vs_v5": accuracy_score(y, pred) - base_acc,
                "oof_disagreement_vs_v5": int(np.sum(pred != base_pred)),
                "test_disagreement_vs_v5": int(
                    np.sum(te_pred != majority(test, HARD_VOTES["v5_robust"]))
                ),
            }
        )
        pred_export[name] = pred
        test_export[name] = te_pred

    for name, cols in SOFT_AVERAGES.items():
        prob = oof[cols].mean(axis=1).to_numpy()
        te_prob = test[cols].mean(axis=1).to_numpy()
        pred = (prob > 0.5).astype(int)
        te_pred = (te_prob > 0.5).astype(int)
        results.append(
            {
                "candidate": name,
                "kind": "soft_average",
                "accuracy": accuracy_score(y, pred),
                "roc_auc": roc_auc_score(y, prob),
                "delta_vs_v5": accuracy_score(y, pred) - base_acc,
                "oof_disagreement_vs_v5": int(np.sum(pred != base_pred)),
                "test_disagreement_vs_v5": int(
                    np.sum(te_pred != majority(test, HARD_VOTES["v5_robust"]))
                ),
            }
        )
        pred_export[name] = prob
        test_export[name] = te_prob

    # Cross-fitted meta model: fixed outer folds, no same-row fitting/evaluation.
    for name, cols in STACKS.items():
        X = oof[cols].to_numpy()
        Xtest = test[cols].to_numpy()
        meta_oof = np.zeros(len(oof), dtype=float)
        fold_test_probs = []
        for fold in sorted(np.unique(folds)):
            tr = folds != fold
            va = folds == fold
            meta = LogisticRegression(
                C=0.25,
                penalty="l2",
                solver="liblinear",
                max_iter=1000,
                random_state=42,
            )
            meta.fit(X[tr], y[tr])
            meta_oof[va] = meta.predict_proba(X[va])[:, 1]
            fold_test_probs.append(meta.predict_proba(Xtest)[:, 1])
        meta_test = np.mean(fold_test_probs, axis=0)
        pred = (meta_oof > 0.5).astype(int)
        te_pred = (meta_test > 0.5).astype(int)
        results.append(
            {
                "candidate": name,
                "kind": "cross_fitted_logistic",
                "accuracy": accuracy_score(y, pred),
                "roc_auc": roc_auc_score(y, meta_oof),
                "delta_vs_v5": accuracy_score(y, pred) - base_acc,
                "oof_disagreement_vs_v5": int(np.sum(pred != base_pred)),
                "test_disagreement_vs_v5": int(
                    np.sum(te_pred != majority(test, HARD_VOTES["v5_robust"]))
                ),
            }
        )
        pred_export[name] = meta_oof
        test_export[name] = meta_test

    result_df = pd.DataFrame(results).sort_values(
        ["accuracy", "roc_auc"], ascending=[False, False], na_position="last"
    )
    result_df.to_csv(V12_DIR / "tabpfn_ensemble_probe_summary.csv", index=False)
    pred_export.to_csv(V12_DIR / "tabpfn_ensemble_probe_oof.csv", index=False)
    test_export.to_csv(V12_DIR / "tabpfn_ensemble_probe_test.csv", index=False)
    with (V12_DIR / "tabpfn_ensemble_probe_metadata.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "hard_votes": HARD_VOTES,
                "soft_averages": SOFT_AVERAGES,
                "stacks": STACKS,
                "stack_model": "LogisticRegression(C=0.25, L2, liblinear)",
                "stack_validation": "cross-fitted on preserved outer folds",
                "no_submission": True,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=== TabPFN ensemble probe ===")
    print(result_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nNo Kaggle submission was performed.")


if __name__ == "__main__":
    main()
