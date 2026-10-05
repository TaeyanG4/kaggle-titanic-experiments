"""Stress-test the v15 typed consensus on the five matched pseudo-test splits.

This script retrains every member needed for the trusted v5 robust vote and the
new v15 strict typed consensus on each pseudo-test split.  The pseudo splits are
selected without Survived and were matched to the real Kaggle test relation
geometry in ``pseudo_test_relational_v14.py``.

No Kaggle submission is performed here.  The output is intended to be the
promotion gate for the already-built v15 candidate submission.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from audit_group_survival import (
    add_group_survival,
    build_base_frame,
    build_models as build_v1_models,
    model_feature_columns,
)
from feature_ablation_v6 import build_models as build_typed_models
from group_key_audit_v6 import peer_feature
from gunes_exact_foldsafe_v11 import structural_frames
from model_zoo_screen_v4 import build_optional_model
from model_zoo_screen_v5 import make_model
from pseudo_test_relational_v14 import (
    VARIANTS as REL_VARIANTS,
    build_pseudo_splits,
    eligible_groups,
    relation_features,
    role_flags,
)
from tabpfn_finalist_v8 import make_tabpfn, prepare_data


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "exports" / "v19"

TYPED_COLS = REL_VARIANTS["typed22"]
FAMILYFARE_COLS = [
    "FamilyFareAny",
    "FamilyFareMean",
    "FamilyFareSmooth",
    "FamilyFareCount",
]
MLP_SEEDS = [42, 142, 242]


def majority3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return ((a.astype(int) + b.astype(int) + c.astype(int)) >= 2).astype(int)


def add_familyfare(ref: pd.DataFrame, app: pd.DataFrame, exclude_self: bool) -> pd.DataFrame:
    stats = peer_feature(
        ref,
        app,
        group_col="FamilyFareGroup_v6",
        wcg_only=False,
        exclude_self=exclude_self,
    )
    out = app.copy()
    for source, dest in zip(["Any", "Mean", "Smooth", "Count"], FAMILYFARE_COLS):
        out[dest] = stats[source].to_numpy()
    return out


def add_rel(df: pd.DataFrame, rel: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in TYPED_COLS:
        out[c] = rel[c].to_numpy()
    return out


def metric_row(split: int, name: str, y: np.ndarray, prob: np.ndarray | None, pred: np.ndarray):
    return {
        "split": split,
        "candidate": name,
        "accuracy": accuracy_score(y, pred),
        "roc_auc": (roc_auc_score(y, prob) if prob is not None else np.nan),
        "n": len(y),
    }


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")

    # Target-independent global representations intentionally match the project
    # pipelines.  Supervised relational features are rebuilt inside each split.
    train_v1, _ = build_base_frame(train_raw, test_raw, profile="v1")
    v1_cols = model_feature_columns(train_v1)
    train_v2, _, _, v2_cols = prepare_data()

    rel_train, _, _, _, _ = structural_frames(train_raw, test_raw)
    rel_train["IsWomanChild"] = role_flags(train_raw)

    if not train_v1["PassengerId"].astype(int).equals(train_v2["PassengerId"].astype(int)):
        raise ValueError("v1/v2 PassengerId mismatch")
    if not train_v2["PassengerId"].astype(int).equals(rel_train["PassengerId"].astype(int)):
        raise ValueError("v2/relational PassengerId mismatch")

    y_all = train_v2["Survived"].astype(int).to_numpy()
    _, pseudo = build_pseudo_splits(train_raw, test_raw)

    rows = []
    pred_rows = []

    for split, (_, va_idx, _) in enumerate(pseudo):
        tr_idx = np.setdiff1d(np.arange(len(train_v2)), va_idx)
        y_tr = y_all[tr_idx]
        y_va = y_all[va_idx]
        print(f"\n=== pseudo split {split}: train={len(tr_idx)} val={len(va_idx)} ===", flush=True)

        # ----- v1 ensemble -----
        tr1 = train_v1.iloc[tr_idx].copy()
        va1 = train_v1.iloc[va_idx].copy()
        tr1["GroupSurvival"] = add_group_survival(tr1, tr1, exclude_self=True).to_numpy()
        va1["GroupSurvival"] = add_group_survival(tr1, va1, exclude_self=False).to_numpy()
        v1_probs = []
        for model in build_v1_models(4200 + split, profile="v1").values():
            model.fit(tr1[v1_cols], y_tr)
            v1_probs.append(np.asarray(model.predict_proba(va1[v1_cols]))[:, 1])
        v1_prob = np.mean(v1_probs, axis=0)

        # ----- trusted v2 matrix / v5 members -----
        tr2 = train_v2.iloc[tr_idx].copy()
        va2 = train_v2.iloc[va_idx].copy()
        tr2["GroupSurvival"] = add_group_survival(tr2, tr2, exclude_self=True).to_numpy()
        va2["GroupSurvival"] = add_group_survival(tr2, va2, exclude_self=False).to_numpy()

        tabicl = build_optional_model("TabICLv2")
        tabicl.fit(tr2[v2_cols], y_tr)
        tabicl_prob = np.asarray(tabicl.predict_proba(va2[v2_cols]))[:, 1]
        v4b_prob = 0.90 * v1_prob + 0.10 * tabicl_prob

        rule = make_model("RuleFit", 5200 + split)
        rule.fit(tr2[v2_cols], y_tr)
        rule_prob = np.asarray(rule.predict_proba(va2[v2_cols]))[:, 1]

        mlp_probs = []
        for seed in MLP_SEEDS:
            mlp = make_model("MLP_PLR", seed + 100 * split)
            mlp.fit(tr2[v2_cols], y_tr)
            mlp_probs.append(np.asarray(mlp.predict_proba(va2[v2_cols]))[:, 1])
        mlp_prob = np.mean(mlp_probs, axis=0)

        v5_pred = majority3(v4b_prob > 0.5, rule_prob > 0.5, mlp_prob > 0.5)
        v5_score = (v4b_prob + rule_prob + mlp_prob) / 3.0

        # ----- FamilyFare + typed22 representation -----
        trt = add_familyfare(tr2, tr2, True)
        vat = add_familyfare(tr2, va2, False)

        rr = rel_train.iloc[tr_idx].copy()
        rv = rel_train.iloc[va_idx].copy()
        af, at = eligible_groups(rr, rv)
        rr_rel = relation_features(
            rr, rr, exclude_self=True, alpha=2.0, allowed_fam=af, allowed_tic=at
        )
        rv_rel = relation_features(
            rr, rv, exclude_self=False, alpha=2.0, allowed_fam=af, allowed_tic=at
        )
        trt = add_rel(trt, rr_rel)
        vat = add_rel(vat, rv_rel)
        typed_cols = list(dict.fromkeys(v2_cols + FAMILYFARE_COLS + TYPED_COLS))

        cat = build_typed_models(6200 + split)["CatBoost"]
        cat.fit(trt[typed_cols], y_tr)
        cat_prob = np.asarray(cat.predict_proba(vat[typed_cols]))[:, 1]

        t25 = make_tabpfn("TabPFN_v2_5", seed=7200 + split)
        t25.fit(trt[typed_cols], y_tr)
        t25_prob = np.asarray(t25.predict_proba(vat[typed_cols]))[:, 1].astype(float)

        t3 = make_tabpfn("TabPFN_v3", seed=8200 + split)
        t3.fit(trt[typed_cols], y_tr)
        t3_prob = np.asarray(t3.predict_proba(vat[typed_cols]))[:, 1].astype(float)

        cat_pred = (cat_prob > 0.5).astype(int)
        t25_pred = (t25_prob > 0.5).astype(int)
        t3_pred = (t3_prob > 0.5).astype(int)

        strict = v5_pred.copy()
        switch_mask = (t3_pred == t25_pred) & (t3_pred == cat_pred) & (t3_pred != v5_pred)
        strict[switch_mask] = t3_pred[switch_mask]

        pair = v5_pred.copy()
        pair_mask = (t3_pred == cat_pred) & (t3_pred != v5_pred)
        pair[pair_mask] = t3_pred[pair_mask]

        rows.extend(
            [
                metric_row(split, "v5_robust_retrained", y_va, v5_score, v5_pred),
                metric_row(split, "typed_cat", y_va, cat_prob, cat_pred),
                metric_row(split, "typed_t25", y_va, t25_prob, t25_pred),
                metric_row(split, "typed_t3", y_va, t3_prob, t3_pred),
                metric_row(split, "switch_t3_t25_cat", y_va, None, strict),
                metric_row(split, "switch_t3_cat", y_va, None, pair),
            ]
        )
        rows[-2]["switch_count"] = int(switch_mask.sum())
        rows[-1]["switch_count"] = int(pair_mask.sum())

        for local_pos, row_idx in enumerate(va_idx):
            pred_rows.append(
                {
                    "split": split,
                    "PassengerId": int(train_v2.iloc[row_idx]["PassengerId"]),
                    "Survived": int(y_va[local_pos]),
                    "v5": int(v5_pred[local_pos]),
                    "cat": int(cat_pred[local_pos]),
                    "t25": int(t25_pred[local_pos]),
                    "t3": int(t3_pred[local_pos]),
                    "switch_t3_t25_cat": int(strict[local_pos]),
                    "switch_t3_cat": int(pair[local_pos]),
                }
            )

        print(
            f"v5={accuracy_score(y_va, v5_pred):.5f} "
            f"strict={accuracy_score(y_va, strict):.5f} "
            f"pair={accuracy_score(y_va, pair):.5f} "
            f"strict_switches={int(switch_mask.sum())}",
            flush=True,
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(EXPORT_DIR / "pseudotest_consensus_metrics.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(EXPORT_DIR / "pseudotest_consensus_predictions.csv", index=False)

    summary = (
        metrics.groupby("candidate", as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            min_accuracy=("accuracy", "min"),
            mean_auc=("roc_auc", "mean"),
        )
        .sort_values("mean_accuracy", ascending=False)
    )
    base = metrics[metrics["candidate"] == "v5_robust_retrained"].set_index("split")
    delta_rows = []
    for candidate in ["switch_t3_t25_cat", "switch_t3_cat"]:
        c = metrics[metrics["candidate"] == candidate].set_index("split")
        d = c["accuracy"] - base["accuracy"]
        delta_rows.append(
            {
                "candidate": candidate,
                "mean_delta_vs_v5": float(d.mean()),
                "positive_splits": int((d > 0).sum()),
                "nonnegative_splits": int((d >= 0).sum()),
                "negative_splits": int((d < 0).sum()),
                "deltas": str([round(float(x), 6) for x in d.tolist()]),
            }
        )
    deltas = pd.DataFrame(delta_rows)
    summary.to_csv(EXPORT_DIR / "pseudotest_consensus_summary.csv", index=False)
    deltas.to_csv(EXPORT_DIR / "pseudotest_consensus_deltas.csv", index=False)

    print("\n=== pseudo-test consensus summary ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n=== paired deltas vs retrained v5 ===")
    print(deltas.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nNo Kaggle submission was performed.")


if __name__ == "__main__":
    main()
