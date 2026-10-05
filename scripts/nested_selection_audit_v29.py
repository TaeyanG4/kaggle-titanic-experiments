"""v29: nested selection-bias audit for Titanic candidate selection.

Outer 5-fold is never used for model selection. Inside each outer-train split,
3-fold inner CV selects exactly one of five previously motivated pipelines:

  - Deotte WCG+XGB Python port
  - FamilyFare + typed22 CatBoost
  - FamilyFare + typed22 RF6
  - FamilyFare + EB25 CatBoost
  - FamilyFare + typed22 + EB25 CatBoost

The selected pipeline is then retrained on all outer-train rows and evaluated
once on the untouched outer validation fold. A separately retrained v5 robust
vote is the fixed benchmark; v5 is not eligible for inner selection.

This estimates generalization of the *selection process*, not merely one model.
No Kaggle submission is performed.
"""

from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from audit_group_survival import add_group_survival, build_base_frame, build_models as build_v1_models, model_feature_columns
from deotte_wcg_xgb_v21 import predict_split as deotte_predict_split
from feature_ablation_v6 import build_models as build_tree_models
from model_zoo_screen_v4 import build_optional_model
from model_zoo_screen_v5 import make_model
from partial_pooling_transfer_v28 import fold_features, prepare_rel
from tabpfn_finalist_v8 import prepare_data


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTER_SEED = int(os.environ.get("V29_OUTER_SEED", "31415"))
EXPORT_DIR = BASE_DIR / "exports" / ("v29" if OUTER_SEED == 31415 else f"v29_seed{OUTER_SEED}")

INNER_SEED_BASE = 29000
N_OUTER = 5
N_INNER = 3

CANDIDATES = [
    "deotte",
    "typed_cat",
    "typed_rf6",
    "eb_cat",
    "typed_eb_cat",
]


def majority3(a, b, c):
    return ((a.astype(int) + b.astype(int) + c.astype(int)) >= 2).astype(int)


def candidate_predict(
    name: str,
    train_raw: pd.DataFrame,
    train_v2: pd.DataFrame,
    rel_train: pd.DataFrame,
    base_cols: list[str],
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = train_v2["Survived"].astype(int).to_numpy()
    if name == "deotte":
        b = deotte_predict_split(
            train_raw,
            tr_idx,
            va_idx,
            seed=seed,
            imputer_fit_indices=tr_idx,
            special_link=True,
        )
        return b.wcg_both.astype(int), b.score.astype(float)

    if name == "typed_cat":
        variant = "familyfare_typed22"
        tr, va, cols = fold_features(train_v2, rel_train, tr_idx, va_idx, base_cols, variant)
        model = build_tree_models(seed)["CatBoost"]
    elif name == "typed_rf6":
        variant = "familyfare_typed22"
        tr, va, cols = fold_features(train_v2, rel_train, tr_idx, va_idx, base_cols, variant)
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=7,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features=.7,
            random_state=seed,
            n_jobs=-1,
        )
    elif name == "eb_cat":
        variant = "familyfare_eb25"
        tr, va, cols = fold_features(train_v2, rel_train, tr_idx, va_idx, base_cols, variant)
        model = build_tree_models(seed)["CatBoost"]
    elif name == "typed_eb_cat":
        variant = "familyfare_typed22_eb25"
        tr, va, cols = fold_features(train_v2, rel_train, tr_idx, va_idx, base_cols, variant)
        model = build_tree_models(seed)["CatBoost"]
    else:
        raise KeyError(name)

    model.fit(tr[cols], y[tr_idx])
    prob = np.asarray(model.predict_proba(va[cols]))[:, 1].astype(float)
    return (prob > .5).astype(int), prob


def v5_predict(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    train_v1: pd.DataFrame,
    train_v2: pd.DataFrame,
    v1_cols: list[str],
    v2_cols: list[str],
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    y = train_v2["Survived"].astype(int).to_numpy()

    tr1 = train_v1.iloc[tr_idx].copy(); va1 = train_v1.iloc[va_idx].copy()
    tr1["GroupSurvival"] = add_group_survival(tr1, tr1, exclude_self=True).to_numpy()
    va1["GroupSurvival"] = add_group_survival(tr1, va1, exclude_self=False).to_numpy()
    v1_probs = []
    for model in build_v1_models(seed, profile="v1").values():
        model.fit(tr1[v1_cols], y[tr_idx])
        v1_probs.append(np.asarray(model.predict_proba(va1[v1_cols]))[:, 1])
    v1_prob = np.mean(v1_probs, axis=0)

    tr2 = train_v2.iloc[tr_idx].copy(); va2 = train_v2.iloc[va_idx].copy()
    tr2["GroupSurvival"] = add_group_survival(tr2, tr2, exclude_self=True).to_numpy()
    va2["GroupSurvival"] = add_group_survival(tr2, va2, exclude_self=False).to_numpy()

    tabicl = build_optional_model("TabICLv2")
    tabicl.fit(tr2[v2_cols], y[tr_idx])
    tab_p = np.asarray(tabicl.predict_proba(va2[v2_cols]))[:, 1]
    v4b = 0.90 * v1_prob + 0.10 * tab_p

    rule = make_model("RuleFit", seed + 1000)
    rule.fit(tr2[v2_cols], y[tr_idx])
    rule_p = np.asarray(rule.predict_proba(va2[v2_cols]))[:, 1]

    mlp_ps = []
    for s in [42, 142, 242]:
        mlp = make_model("MLP_PLR", s + seed)
        mlp.fit(tr2[v2_cols], y[tr_idx])
        mlp_ps.append(np.asarray(mlp.predict_proba(va2[v2_cols]))[:, 1])
    mlp_p = np.mean(mlp_ps, axis=0)

    pred = majority3(v4b > .5, rule_p > .5, mlp_p > .5)
    score = (v4b + rule_p + mlp_p) / 3.0
    return pred, score


def main():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    y = train_raw["Survived"].astype(int).to_numpy()

    train_v1, _ = build_base_frame(train_raw, test_raw, profile="v1")
    v1_cols = model_feature_columns(train_v1)
    train_v2, _, _, v2_cols = prepare_data()
    rel_train, _ = prepare_rel(train_raw, test_raw)

    outer = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=OUTER_SEED)
    nested_pred = np.zeros(len(train_raw), dtype=int)
    nested_score = np.zeros(len(train_raw), dtype=float)
    v5_pred = np.zeros(len(train_raw), dtype=int)
    v5_score = np.zeros(len(train_raw), dtype=float)
    oracle_pred = np.zeros(len(train_raw), dtype=int)
    fold_id = np.full(len(train_raw), -1, dtype=int)
    outer_rows = []
    inner_rows = []

    for outer_fold, (outer_tr, outer_va) in enumerate(outer.split(train_raw, y)):
        print(f"\n=== OUTER FOLD {outer_fold}: train={len(outer_tr)} val={len(outer_va)} ===", flush=True)
        fold_id[outer_va] = outer_fold

        # Inner selection only sees outer-train rows.
        y_outer = y[outer_tr]
        inner = StratifiedKFold(
            n_splits=N_INNER,
            shuffle=True,
            random_state=INNER_SEED_BASE + outer_fold,
        )
        candidate_inner = {name: [] for name in CANDIDATES}
        candidate_inner_auc = {name: [] for name in CANDIDATES}

        for inner_fold, (itr_local, iva_local) in enumerate(inner.split(outer_tr, y_outer)):
            itr = outer_tr[itr_local]
            iva = outer_tr[iva_local]
            for ci, name in enumerate(CANDIDATES):
                pred, prob = candidate_predict(
                    name,
                    train_raw,
                    train_v2,
                    rel_train,
                    v2_cols,
                    itr,
                    iva,
                    seed=INNER_SEED_BASE + outer_fold * 100 + inner_fold * 10 + ci,
                )
                acc = accuracy_score(y[iva], pred)
                auc = roc_auc_score(y[iva], prob)
                candidate_inner[name].append(acc)
                candidate_inner_auc[name].append(auc)
                inner_rows.append({
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "candidate": name,
                    "accuracy": acc,
                    "roc_auc": auc,
                })

        rank = []
        for name in CANDIDATES:
            rank.append({
                "candidate": name,
                "mean_accuracy": float(np.mean(candidate_inner[name])),
                "min_accuracy": float(np.min(candidate_inner[name])),
                "mean_auc": float(np.mean(candidate_inner_auc[name])),
            })
        rank_df = pd.DataFrame(rank).sort_values(
            ["mean_accuracy", "min_accuracy", "mean_auc"],
            ascending=False,
        )
        selected = str(rank_df.iloc[0]["candidate"])
        print("inner ranking:")
        print(rank_df.to_string(index=False, float_format=lambda v: f"{v:.5f}"), flush=True)
        print(f"selected: {selected}", flush=True)

        # Train selected candidate once on all outer-train rows.
        sp, ss = candidate_predict(
            selected,
            train_raw,
            train_v2,
            rel_train,
            v2_cols,
            outer_tr,
            outer_va,
            seed=30000 + outer_fold,
        )
        nested_pred[outer_va] = sp
        nested_score[outer_va] = ss

        # Fixed v5 benchmark, completely independent of inner selection.
        vp, vs = v5_predict(
            train_raw,
            test_raw,
            train_v1,
            train_v2,
            v1_cols,
            v2_cols,
            outer_tr,
            outer_va,
            seed=31000 + outer_fold,
        )
        v5_pred[outer_va] = vp
        v5_score[outer_va] = vs

        # Oracle diagnostic: evaluate every candidate on outer fold but NEVER use
        # this to choose the nested candidate. It measures the selection regret.
        oracle_results = []
        for ci, name in enumerate(CANDIDATES):
            op, os = candidate_predict(
                name,
                train_raw,
                train_v2,
                rel_train,
                v2_cols,
                outer_tr,
                outer_va,
                seed=32000 + outer_fold * 10 + ci,
            )
            oracle_results.append((name, accuracy_score(y[outer_va], op), op, os))
        oracle_results.sort(key=lambda z: z[1], reverse=True)
        oracle_name, oracle_acc, oracle_p, _ = oracle_results[0]
        oracle_pred[outer_va] = oracle_p

        nested_acc = accuracy_score(y[outer_va], sp)
        v5_acc = accuracy_score(y[outer_va], vp)
        outer_rows.append({
            "outer_fold": outer_fold,
            "selected": selected,
            "selected_inner_mean_accuracy": float(rank_df.iloc[0]["mean_accuracy"]),
            "selected_inner_min_accuracy": float(rank_df.iloc[0]["min_accuracy"]),
            "nested_outer_accuracy": nested_acc,
            "v5_outer_accuracy": v5_acc,
            "delta_vs_v5": nested_acc - v5_acc,
            "oracle_candidate": oracle_name,
            "oracle_outer_accuracy": oracle_acc,
            "selection_regret": oracle_acc - nested_acc,
            "n_val": len(outer_va),
        })
        print(
            f"outer result: selected={selected} nested={nested_acc:.5f} "
            f"v5={v5_acc:.5f} oracle={oracle_name}:{oracle_acc:.5f}",
            flush=True,
        )

    outer_df = pd.DataFrame(outer_rows)
    inner_df = pd.DataFrame(inner_rows)
    outer_df.to_csv(EXPORT_DIR / "nested_outer_folds.csv", index=False)
    inner_df.to_csv(EXPORT_DIR / "nested_inner_metrics.csv", index=False)

    oof = pd.DataFrame({
        "PassengerId": train_raw["PassengerId"].astype(int),
        "fold": fold_id,
        "Survived": y,
        "nested_pred": nested_pred,
        "nested_score": nested_score,
        "v5_pred": v5_pred,
        "v5_score": v5_score,
        "oracle_pred": oracle_pred,
    })
    oof.to_csv(EXPORT_DIR / "nested_oof.csv", index=False)

    summary = pd.DataFrame([
        {
            "candidate": "nested_selector",
            "accuracy": accuracy_score(y, nested_pred),
            "roc_auc": roc_auc_score(y, nested_score),
            "fold_accuracy_std": float(outer_df["nested_outer_accuracy"].std(ddof=0)),
        },
        {
            "candidate": "v5_fixed_benchmark",
            "accuracy": accuracy_score(y, v5_pred),
            "roc_auc": roc_auc_score(y, v5_score),
            "fold_accuracy_std": float(outer_df["v5_outer_accuracy"].std(ddof=0)),
        },
        {
            "candidate": "outer_oracle_upper_bound",
            "accuracy": accuracy_score(y, oracle_pred),
            "roc_auc": np.nan,
            "fold_accuracy_std": float(outer_df["oracle_outer_accuracy"].std(ddof=0)),
        },
    ])
    summary["delta_vs_v5"] = summary["accuracy"] - float(summary.loc[summary["candidate"] == "v5_fixed_benchmark", "accuracy"].iloc[0])
    summary.to_csv(EXPORT_DIR / "nested_summary.csv", index=False)

    selection_counts = outer_df["selected"].value_counts().rename_axis("candidate").reset_index(name="selected_outer_folds")
    selection_counts.to_csv(EXPORT_DIR / "selection_counts.csv", index=False)

    print("\n=== NESTED SUMMARY ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\n=== OUTER FOLDS ===")
    print(outer_df.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\n=== SELECTION COUNTS ===")
    print(selection_counts.to_string(index=False))
    print("\nNo Kaggle submission was performed.")


if __name__ == "__main__":
    main()
