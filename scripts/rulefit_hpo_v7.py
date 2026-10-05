"""Limited RuleFit HPO for Titanic v7 with ensemble-aware diagnostics.

The current public champion uses:
  v4b vote + RuleFit vote + MLP-PLR 3-seed-mean vote

This script keeps the trusted v2 representation and fold-safe GroupSurvival,
screens a small predeclared RuleFit grid on the exact saved five folds, then
re-runs the top configurations with multiple RuleFit random seeds.

The primary evidence is:
  1) RuleFit OOF Accuracy/AUC
  2) resulting 2-of-3 hard-vote Accuracy with fixed v4b + MLP seed-mean OOF
  3) seed stability

No Kaggle submission is performed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from imodels import RuleFitClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

try:
    from audit_group_survival import add_group_survival, build_base_frame, model_feature_columns
except ModuleNotFoundError:
    from scripts.audit_group_survival import add_group_survival, build_base_frame, model_feature_columns


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
V1_AUDIT = BASE_DIR / "exports" / "wcg_audit_v1"
V5_DIR = BASE_DIR / "exports" / "v5"
EXPORT_DIR = BASE_DIR / "exports" / "v7"


CONFIGS = {
    "baseline": dict(n_estimators=160, tree_size=4, max_rules=40, include_linear=True),
    "rules20": dict(n_estimators=160, tree_size=4, max_rules=20, include_linear=True),
    "rules60": dict(n_estimators=160, tree_size=4, max_rules=60, include_linear=True),
    "tree3": dict(n_estimators=160, tree_size=3, max_rules=40, include_linear=True),
    "tree5": dict(n_estimators=160, tree_size=5, max_rules=40, include_linear=True),
    "no_linear": dict(n_estimators=160, tree_size=4, max_rules=40, include_linear=False),
    "fixed_tree_size": dict(
        n_estimators=160,
        tree_size=4,
        max_rules=40,
        include_linear=True,
        exp_rand_tree_size=False,
    ),
    "trim05": dict(
        n_estimators=160,
        tree_size=4,
        max_rules=40,
        include_linear=True,
        lin_trim_quantile=0.05,
    ),
    "trees240": dict(n_estimators=240, tree_size=4, max_rules=40, include_linear=True),
}


def make_model(params: dict, seed: int):
    return RuleFitClassifier(
        **params,
        random_state=seed,
        verbose=0,
    )


def hard_vote_accuracy(
    rule_prob: np.ndarray,
    v4b_prob: np.ndarray,
    mlp_prob: np.ndarray,
    y: np.ndarray,
) -> tuple[float, np.ndarray]:
    votes = (
        (rule_prob > 0.5).astype(int)
        + (v4b_prob > 0.5).astype(int)
        + (mlp_prob > 0.5).astype(int)
    )
    pred = (votes >= 2).astype(int)
    return accuracy_score(y, pred), pred


def evaluate_config(
    config_name: str,
    params: dict,
    seed: int,
    train_base: pd.DataFrame,
    test_base: pd.DataFrame,
    folds: np.ndarray,
    v4b_prob: np.ndarray,
    mlp_prob: np.ndarray,
):
    y = train_base["Survived"].astype(int).to_numpy()
    features = model_feature_columns(train_base)
    oof = np.zeros(len(train_base), dtype=float)
    test_probs = []
    fold_rows = []

    for fold in sorted(np.unique(folds)):
        tr_idx = np.flatnonzero(folds != fold)
        va_idx = np.flatnonzero(folds == fold)
        tr = train_base.iloc[tr_idx].copy()
        va = train_base.iloc[va_idx].copy()
        te = test_base.copy()

        tr["GroupSurvival"] = add_group_survival(tr, tr, exclude_self=True).to_numpy()
        va["GroupSurvival"] = add_group_survival(tr, va, exclude_self=False).to_numpy()
        te["GroupSurvival"] = add_group_survival(tr, te, exclude_self=False).to_numpy()

        model = make_model(params, seed=seed + int(fold))
        model.fit(tr[features], y[tr_idx])
        va_prob = np.asarray(model.predict_proba(va[features]))[:, 1]
        te_prob = np.asarray(model.predict_proba(te[features]))[:, 1]
        oof[va_idx] = va_prob
        test_probs.append(te_prob)

        ensemble_acc, ensemble_pred = hard_vote_accuracy(
            oof[va_idx],
            v4b_prob[va_idx],
            mlp_prob[va_idx],
            y[va_idx],
        )
        fold_rows.append(
            {
                "config": config_name,
                "seed": seed,
                "fold": int(fold),
                "rulefit_accuracy": accuracy_score(y[va_idx], va_prob > 0.5),
                "rulefit_auc": roc_auc_score(y[va_idx], va_prob),
                "hard_vote_accuracy": ensemble_acc,
            }
        )

    test_prob = np.mean(test_probs, axis=0)
    vote_acc, vote_pred = hard_vote_accuracy(oof, v4b_prob, mlp_prob, y)
    return {
        "config": config_name,
        "seed": seed,
        "rulefit_accuracy": accuracy_score(y, oof > 0.5),
        "rulefit_auc": roc_auc_score(y, oof),
        "hard_vote_accuracy": vote_acc,
        "hard_vote_changed_vs_baseline_v5": None,
        "fold_rulefit_std": pd.DataFrame(fold_rows)["rulefit_accuracy"].std(ddof=0),
        "fold_hard_vote_std": pd.DataFrame(fold_rows)["hard_vote_accuracy"].std(ddof=0),
        "oof": oof,
        "test": test_prob,
        "hard_vote_pred": vote_pred,
        "fold_rows": fold_rows,
    }


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    train_base, test_base = build_base_frame(train_raw, test_raw, profile="v2")
    manifest = pd.read_csv(V1_AUDIT / "fold_manifest_seed42.csv")
    folds = manifest["fold"].astype(int).to_numpy()
    y = train_base["Survived"].astype(int).to_numpy()

    zoo = pd.read_csv(V5_DIR / "model_zoo_oof.csv")
    mlp = pd.read_csv(V5_DIR / "seed_probes" / "MLP_PLR_seed_mean__oof.csv")
    if not train_base["PassengerId"].equals(zoo["PassengerId"]):
        raise ValueError("v5 OOF PassengerId mismatch.")
    if not train_base["PassengerId"].equals(mlp["PassengerId"]):
        raise ValueError("MLP seed-mean PassengerId mismatch.")
    v4b_prob = zoo["v4b__Champion"].to_numpy()
    mlp_prob = mlp["probability"].to_numpy()
    baseline_rule = zoo["RuleFit"].to_numpy()
    baseline_vote_acc, baseline_vote_pred = hard_vote_accuracy(
        baseline_rule, v4b_prob, mlp_prob, y
    )

    screen_rows = []
    fold_rows = []
    oof_export = pd.DataFrame(
        {"PassengerId": train_base["PassengerId"], "fold": folds, "Survived": y}
    )
    test_export = pd.DataFrame({"PassengerId": test_base["PassengerId"]})

    print(f"Existing v5 robust hard-vote Accuracy: {baseline_vote_acc:.5f}", flush=True)
    print("\n=== RuleFit config screen, seed 42 ===", flush=True)
    seed42_results = {}
    for name, params in CONFIGS.items():
        res = evaluate_config(
            name,
            params,
            42,
            train_base,
            test_base,
            folds,
            v4b_prob,
            mlp_prob,
        )
        res["hard_vote_changed_vs_baseline_v5"] = int(
            np.sum(res["hard_vote_pred"] != baseline_vote_pred)
        )
        seed42_results[name] = res
        screen_rows.append(
            {k: v for k, v in res.items() if k not in {"oof", "test", "hard_vote_pred", "fold_rows"}}
        )
        fold_rows.extend(res["fold_rows"])
        oof_export[f"{name}__seed42"] = res["oof"]
        test_export[f"{name}__seed42"] = res["test"]
        print(
            f"{name:16s} rule={res['rulefit_accuracy']:.5f}/{res['rulefit_auc']:.5f} "
            f"vote={res['hard_vote_accuracy']:.5f} "
            f"changed={res['hard_vote_changed_vs_baseline_v5']}",
            flush=True,
        )

    seed42_df = pd.DataFrame(screen_rows).sort_values(
        ["hard_vote_accuracy", "rulefit_accuracy", "rulefit_auc"],
        ascending=False,
    )
    top_configs = seed42_df.head(3)["config"].tolist()

    print("\n=== seed stability for top configs ===", flush=True)
    stability_rows = []
    all_seed_rows = screen_rows.copy()
    for name in top_configs:
        for seed in [142, 242]:
            res = evaluate_config(
                name,
                CONFIGS[name],
                seed,
                train_base,
                test_base,
                folds,
                v4b_prob,
                mlp_prob,
            )
            res["hard_vote_changed_vs_baseline_v5"] = int(
                np.sum(res["hard_vote_pred"] != baseline_vote_pred)
            )
            all_seed_rows.append(
                {k: v for k, v in res.items() if k not in {"oof", "test", "hard_vote_pred", "fold_rows"}}
            )
            fold_rows.extend(res["fold_rows"])
            oof_export[f"{name}__seed{seed}"] = res["oof"]
            test_export[f"{name}__seed{seed}"] = res["test"]
            print(
                f"{name:16s} seed={seed} rule={res['rulefit_accuracy']:.5f} "
                f"vote={res['hard_vote_accuracy']:.5f}",
                flush=True,
            )

    all_seed_df = pd.DataFrame(all_seed_rows)
    for name in top_configs:
        g = all_seed_df[all_seed_df["config"] == name]
        stability_rows.append(
            {
                "config": name,
                "mean_rulefit_accuracy": g["rulefit_accuracy"].mean(),
                "min_rulefit_accuracy": g["rulefit_accuracy"].min(),
                "std_rulefit_accuracy": g["rulefit_accuracy"].std(ddof=0),
                "mean_hard_vote_accuracy": g["hard_vote_accuracy"].mean(),
                "min_hard_vote_accuracy": g["hard_vote_accuracy"].min(),
                "std_hard_vote_accuracy": g["hard_vote_accuracy"].std(ddof=0),
                "mean_rulefit_auc": g["rulefit_auc"].mean(),
            }
        )
    stability = pd.DataFrame(stability_rows).sort_values(
        ["mean_hard_vote_accuracy", "min_hard_vote_accuracy", "mean_rulefit_accuracy"],
        ascending=False,
    )

    seed42_df.to_csv(EXPORT_DIR / "rulefit_hpo_screen.csv", index=False)
    all_seed_df.to_csv(EXPORT_DIR / "rulefit_hpo_all_seeds.csv", index=False)
    stability.to_csv(EXPORT_DIR / "rulefit_hpo_stability.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(EXPORT_DIR / "rulefit_hpo_fold_metrics.csv", index=False)
    oof_export.to_csv(EXPORT_DIR / "rulefit_hpo_oof.csv", index=False)
    test_export.to_csv(EXPORT_DIR / "rulefit_hpo_test.csv", index=False)
    with (EXPORT_DIR / "rulefit_hpo_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "configs": CONFIGS,
                "trusted_fold_manifest": str(V1_AUDIT / "fold_manifest_seed42.csv"),
                "v5_robust_hard_vote_accuracy_reference": baseline_vote_acc,
                "screen_seed": 42,
                "stability_seeds": [42, 142, 242],
                "notes": [
                    "Grid is deliberately small and predeclared.",
                    "All RuleFit predictions are OOF on the same trusted folds.",
                    "Hard-vote diagnostics use fixed v4b and MLP 3-seed-mean OOF predictions.",
                    "No Kaggle submission is performed.",
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== RuleFit seed42 ranking ===")
    print(seed42_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n=== top config seed stability ===")
    print(stability.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
