"""Limited v7 CatBoost HPO on the strongest v6 representation.

Representation is intentionally fixed:
  v2 one-hot features + fold-safe GroupSurvival + AdultMale_v6

This script screens a small predeclared grid on the trusted seed-42 folds, then
stress-tests the top three configurations on two alternate StratifiedKFold
manifests (seeds 123 and 777). This is not a large Optuna search and does not
touch Kaggle.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from audit_group_survival import add_group_survival, build_base_frame, model_feature_columns
    from feature_ablation_v6 import build_target_independent_blocks
except ModuleNotFoundError:
    from scripts.audit_group_survival import add_group_survival, build_base_frame, model_feature_columns
    from scripts.feature_ablation_v6 import build_target_independent_blocks


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
V1_AUDIT = BASE_DIR / "exports" / "wcg_audit_v1"
EXPORT_DIR = BASE_DIR / "exports" / "v7"


CONFIGS = {
    "baseline": dict(iterations=130, depth=4, learning_rate=0.035, l2_leaf_reg=3.0),
    "d3_l2_3": dict(iterations=180, depth=3, learning_rate=0.035, l2_leaf_reg=3.0),
    "d4_l2_1": dict(iterations=180, depth=4, learning_rate=0.030, l2_leaf_reg=1.0),
    "d4_l2_6": dict(iterations=180, depth=4, learning_rate=0.030, l2_leaf_reg=6.0),
    "d5_l2_3": dict(iterations=180, depth=5, learning_rate=0.030, l2_leaf_reg=3.0),
    "d5_l2_6": dict(iterations=180, depth=5, learning_rate=0.030, l2_leaf_reg=6.0),
    "slow_d3_l2_6": dict(iterations=260, depth=3, learning_rate=0.020, l2_leaf_reg=6.0),
    "slow_d4_l2_10": dict(iterations=260, depth=4, learning_rate=0.020, l2_leaf_reg=10.0),
}


def make_model(params: dict, seed: int):
    return CatBoostClassifier(
        **params,
        random_seed=seed,
        verbose=0,
        thread_count=-1,
        loss_function="Logloss",
        allow_writing_files=False,
    )


def evaluate_config(
    config_name: str,
    params: dict,
    train_base: pd.DataFrame,
    test_base: pd.DataFrame,
    adult_train: pd.DataFrame,
    adult_test: pd.DataFrame,
    splits,
    *,
    split_seed: int,
):
    y = train_base["Survived"].astype(int).reset_index(drop=True)
    base_cols = model_feature_columns(train_base)
    features = base_cols + ["AdultMale_v6"]
    oof = np.zeros(len(train_base), dtype=float)
    test_probs = []
    fold_rows = []

    for fold, (tr_idx, va_idx) in enumerate(splits):
        tr = train_base.iloc[tr_idx].copy()
        va = train_base.iloc[va_idx].copy()
        te = test_base.copy()
        tr["GroupSurvival"] = add_group_survival(tr, tr, exclude_self=True).to_numpy()
        va["GroupSurvival"] = add_group_survival(tr, va, exclude_self=False).to_numpy()
        te["GroupSurvival"] = add_group_survival(tr, te, exclude_self=False).to_numpy()

        tr = tr.merge(
            adult_train.iloc[tr_idx][["PassengerId", "AdultMale_v6"]],
            on="PassengerId",
            how="left",
        )
        va = va.merge(
            adult_train.iloc[va_idx][["PassengerId", "AdultMale_v6"]],
            on="PassengerId",
            how="left",
        )
        te = te.merge(
            adult_test[["PassengerId", "AdultMale_v6"]],
            on="PassengerId",
            how="left",
        )

        model = make_model(params, seed=1000 * split_seed + fold)
        model.fit(tr[features], y.iloc[tr_idx])
        va_prob = model.predict_proba(va[features])[:, 1]
        te_prob = model.predict_proba(te[features])[:, 1]
        oof[va_idx] = va_prob
        test_probs.append(te_prob)
        fold_rows.append(
            {
                "config": config_name,
                "split_seed": split_seed,
                "fold": fold,
                "accuracy": accuracy_score(y.iloc[va_idx], va_prob > 0.5),
                "roc_auc": roc_auc_score(y.iloc[va_idx], va_prob),
            }
        )

    test_prob = np.mean(test_probs, axis=0)
    return {
        "config": config_name,
        "split_seed": split_seed,
        "accuracy": accuracy_score(y, oof > 0.5),
        "roc_auc": roc_auc_score(y, oof),
        "fold_accuracy_std": pd.DataFrame(fold_rows)["accuracy"].std(ddof=0),
        "fold_auc_std": pd.DataFrame(fold_rows)["roc_auc"].std(ddof=0),
        "oof": oof,
        "test": test_prob,
        "fold_rows": fold_rows,
    }


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    train_base, test_base = build_base_frame(train_raw, test_raw, profile="v2")
    y = train_base["Survived"].astype(int).reset_index(drop=True)
    blocks = build_target_independent_blocks(train_raw, test_raw)
    inter_train, inter_test = blocks["interactions"]
    adult_train = inter_train[["PassengerId", "AdultMale_v6"]].copy()
    adult_test = inter_test[["PassengerId", "AdultMale_v6"]].copy()

    manifest = pd.read_csv(V1_AUDIT / "fold_manifest_seed42.csv")
    fold_values = manifest["fold"].astype(int).to_numpy()
    trusted_splits = [
        (np.flatnonzero(fold_values != f), np.flatnonzero(fold_values == f))
        for f in sorted(np.unique(fold_values))
    ]

    screen_results = []
    all_fold_rows = []
    oof_export = pd.DataFrame(
        {
            "PassengerId": train_base["PassengerId"],
            "fold": fold_values,
            "Survived": y,
        }
    )
    test_export = pd.DataFrame({"PassengerId": test_base["PassengerId"]})

    print("=== v7 CatBoost limited HPO: trusted seed-42 folds ===", flush=True)
    for name, params in CONFIGS.items():
        res = evaluate_config(
            name,
            params,
            train_base,
            test_base,
            adult_train,
            adult_test,
            trusted_splits,
            split_seed=42,
        )
        screen_results.append({k: v for k, v in res.items() if k not in {"oof", "test", "fold_rows"}})
        all_fold_rows.extend(res["fold_rows"])
        oof_export[name] = res["oof"]
        test_export[name] = res["test"]
        print(
            f"{name:18s} acc={res['accuracy']:.5f} auc={res['roc_auc']:.5f} "
            f"fold_std={res['fold_accuracy_std']:.5f}",
            flush=True,
        )

    screen = pd.DataFrame(screen_results).sort_values(
        ["accuracy", "roc_auc", "fold_accuracy_std"],
        ascending=[False, False, True],
    )
    top_configs = screen.head(3)["config"].tolist()

    stress_results = []
    print("\n=== stress validation on alternate fold seeds ===", flush=True)
    for split_seed in [123, 777]:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=split_seed)
        splits = list(skf.split(train_base, y))
        for name in top_configs:
            res = evaluate_config(
                name,
                CONFIGS[name],
                train_base,
                test_base,
                adult_train,
                adult_test,
                splits,
                split_seed=split_seed,
            )
            stress_results.append(
                {k: v for k, v in res.items() if k not in {"oof", "test", "fold_rows"}}
            )
            all_fold_rows.extend(res["fold_rows"])
            print(
                f"seed={split_seed} {name:18s} acc={res['accuracy']:.5f} "
                f"auc={res['roc_auc']:.5f}",
                flush=True,
            )

    stress = pd.DataFrame(stress_results)
    stability_rows = []
    for name in top_configs:
        rows = pd.concat(
            [
                screen[screen["config"] == name],
                stress[stress["config"] == name],
            ],
            ignore_index=True,
        )
        stability_rows.append(
            {
                "config": name,
                "mean_accuracy_across_manifests": rows["accuracy"].mean(),
                "min_accuracy_across_manifests": rows["accuracy"].min(),
                "std_accuracy_across_manifests": rows["accuracy"].std(ddof=0),
                "mean_auc_across_manifests": rows["roc_auc"].mean(),
            }
        )
    stability = pd.DataFrame(stability_rows).sort_values(
        ["mean_accuracy_across_manifests", "min_accuracy_across_manifests"],
        ascending=False,
    )

    screen.to_csv(EXPORT_DIR / "catboost_hpo_screen.csv", index=False)
    stress.to_csv(EXPORT_DIR / "catboost_hpo_stress.csv", index=False)
    stability.to_csv(EXPORT_DIR / "catboost_hpo_stability.csv", index=False)
    pd.DataFrame(all_fold_rows).to_csv(EXPORT_DIR / "catboost_hpo_fold_metrics.csv", index=False)
    oof_export.to_csv(EXPORT_DIR / "catboost_hpo_oof_seed42.csv", index=False)
    test_export.to_csv(EXPORT_DIR / "catboost_hpo_test_seed42.csv", index=False)
    with (EXPORT_DIR / "catboost_hpo_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "configs": CONFIGS,
                "representation": "v2 one-hot + fold-safe GroupSurvival + AdultMale_v6",
                "trusted_manifest_seed": 42,
                "stress_manifest_seeds": [123, 777],
                "notes": [
                    "Small predeclared grid only; not an exhaustive HPO.",
                    "Top configs are stress-tested on alternate fold manifests.",
                    "No Kaggle submission is performed.",
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== trusted-fold ranking ===")
    print(screen.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n=== cross-manifest stability ===")
    print(stability.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
