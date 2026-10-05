"""Transductive relation-graph feature screen.

Extends typed22 with two target-independent / leakage-safe structures:

1. inference counts
   How many validation/test passengers share this Family/Ticket?

2. Family-Ticket connected components
   Build a graph on reference + inference passengers, connecting rows that share
   an eligible Family or Ticket. For each row expose:
     - reference peer count
     - inference component count
     - component total size
     - smoothed component survival from reference labels only
     - same-role (WomanChild / AdultMale) component survival/count

Reference-row target features exclude the row itself.

Evaluated on BOTH real-test-matched pseudo holdouts and trusted fixed folds.
No Kaggle submission is performed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from gunes_exact_foldsafe_v11 import structural_frames
from pseudo_test_relational_v14 import (
    VARIANTS as BASE_VARIANTS,
    eligible_groups,
    relation_features,
    role_flags,
)


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
V1_AUDIT = BASE_DIR / "exports" / "wcg_audit_v1"
V14_DIR = BASE_DIR / "exports" / "v14"
EXPORT_DIR = BASE_DIR / "exports" / "v17"
ALPHA = 2.0
TYPED = BASE_VARIANTS["typed22"]


def union_find_components(ref: pd.DataFrame, inf: pd.DataFrame):
    n_ref = len(ref)
    combo = pd.concat(
        [
            ref.assign(_side="ref"),
            inf.assign(_side="inf"),
        ],
        ignore_index=True,
        sort=False,
    )
    n = len(combo)
    parent = np.arange(n)
    rank = np.zeros(n, dtype=int)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    # Family edges only for meaningful family groups.
    fam_med = ref.groupby("Family")["Family_Size"].median()
    for fam, idxs in combo.groupby("Family", sort=False).groups.items():
        if float(fam_med.get(fam, 0.0)) <= 1 or len(idxs) <= 1:
            continue
        idxs = list(idxs)
        root = idxs[0]
        for j in idxs[1:]:
            union(root, j)

    # Ticket edges.
    for _, idxs in combo.groupby("Ticket", sort=False).groups.items():
        if len(idxs) <= 1:
            continue
        idxs = list(idxs)
        root = idxs[0]
        for j in idxs[1:]:
            union(root, j)

    comp = np.array([find(i) for i in range(n)], dtype=int)
    return combo, comp, n_ref


def inference_count_features(
    ref: pd.DataFrame,
    target: pd.DataFrame,
    inf: pd.DataFrame,
) -> pd.DataFrame:
    fam_inf = inf["Family"].value_counts()
    tic_inf = inf["Ticket"].value_counts()
    fam_ref = ref["Family"].value_counts()
    tic_ref = ref["Ticket"].value_counts()
    fam_med = ref.groupby("Family")["Family_Size"].median()

    fam_ok = target["Family"].map(fam_med).fillna(0).to_numpy() > 1
    inf_fam = target["Family"].map(fam_inf).fillna(0).to_numpy(float)
    ref_fam = target["Family"].map(fam_ref).fillna(0).to_numpy(float)
    inf_fam = np.where(fam_ok, inf_fam, 0.0)
    ref_fam = np.where(fam_ok, ref_fam, 0.0)
    inf_tic = target["Ticket"].map(tic_inf).fillna(0).to_numpy(float)
    ref_tic = target["Ticket"].map(tic_ref).fillna(0).to_numpy(float)

    return pd.DataFrame(
        {
            "InferenceFamilyCount": inf_fam,
            "InferenceTicketCount": inf_tic,
            "TotalFamilyCount": inf_fam + ref_fam,
            "TotalTicketCount": inf_tic + ref_tic,
            "InferenceRelationCount": inf_fam + inf_tic,
        },
        index=target.index,
    )


def component_features(
    ref: pd.DataFrame,
    target: pd.DataFrame,
    inf: pd.DataFrame,
    *,
    target_side: str,
    exclude_self: bool,
) -> pd.DataFrame:
    combo, comp, n_ref = union_find_components(ref, inf)
    combo = combo.copy()
    combo["_comp"] = comp
    prior = float(ref["Survived"].mean())

    # Reference aggregates per component.
    ref_combo = combo.iloc[:n_ref].copy()
    inf_combo = combo.iloc[n_ref:].copy()
    agg = ref_combo.groupby("_comp").agg(
        RefCount=("PassengerId", "size"),
        SurvivedSum=("Survived", "sum"),
    )
    inf_count = inf_combo.groupby("_comp").size()

    wc = ref_combo[ref_combo["IsWomanChild"] == 1].groupby("_comp").agg(
        WCCount=("PassengerId", "size"),
        WCSum=("Survived", "sum"),
    )
    am = ref_combo[ref_combo["IsWomanChild"] == 0].groupby("_comp").agg(
        AMCount=("PassengerId", "size"),
        AMSum=("Survived", "sum"),
    )

    # Map target rows to combo component IDs.
    if target_side == "ref":
        target_comp = comp[:n_ref]
    elif target_side == "inf":
        target_comp = comp[n_ref:]
    else:
        raise ValueError(target_side)
    if len(target_comp) != len(target):
        raise ValueError("Target/component length mismatch.")

    rows = []
    for pos, (_, row) in enumerate(target.iterrows()):
        cid = target_comp[pos]
        ref_count = float(agg.loc[cid, "RefCount"]) if cid in agg.index else 0.0
        surv_sum = float(agg.loc[cid, "SurvivedSum"]) if cid in agg.index else 0.0
        wc_count = float(wc.loc[cid, "WCCount"]) if cid in wc.index else 0.0
        wc_sum = float(wc.loc[cid, "WCSum"]) if cid in wc.index else 0.0
        am_count = float(am.loc[cid, "AMCount"]) if cid in am.index else 0.0
        am_sum = float(am.loc[cid, "AMSum"]) if cid in am.index else 0.0

        if exclude_self and target_side == "ref":
            ref_count -= 1.0
            surv_sum -= float(row["Survived"])
            if int(row["IsWomanChild"]) == 1:
                wc_count -= 1.0
                wc_sum -= float(row["Survived"])
            else:
                am_count -= 1.0
                am_sum -= float(row["Survived"])

        ref_count = max(ref_count, 0.0)
        wc_count = max(wc_count, 0.0)
        am_count = max(am_count, 0.0)
        inf_n = float(inf_count.get(cid, 0.0))
        comp_rate = (
            (surv_sum + ALPHA * prior) / (ref_count + ALPHA)
            if ref_count > 0
            else prior
        )
        same_count = wc_count if int(row["IsWomanChild"]) == 1 else am_count
        same_sum = wc_sum if int(row["IsWomanChild"]) == 1 else am_sum
        same_rate = (
            (same_sum + ALPHA * prior) / (same_count + ALPHA)
            if same_count > 0
            else prior
        )
        rows.append(
            {
                "ComponentRefCount": ref_count,
                "ComponentInferenceCount": inf_n,
                "ComponentTotalSize": ref_count + inf_n,
                "ComponentSmooth": comp_rate,
                "ComponentSameRoleCount": same_count,
                "ComponentSameRoleSmooth": same_rate,
            }
        )
    return pd.DataFrame(rows, index=target.index)


COUNT_COLS = [
    "InferenceFamilyCount",
    "InferenceTicketCount",
    "TotalFamilyCount",
    "TotalTicketCount",
    "InferenceRelationCount",
]
GRAPH_COLS = [
    "ComponentRefCount",
    "ComponentInferenceCount",
    "ComponentTotalSize",
    "ComponentSmooth",
    "ComponentSameRoleCount",
    "ComponentSameRoleSmooth",
]
VARIANTS = {
    "typed22": TYPED,
    "typed_counts": TYPED + COUNT_COLS,
    "typed_graph": TYPED + GRAPH_COLS,
    "typed_graph_counts": TYPED + COUNT_COLS + GRAPH_COLS,
}


def rf(seed):
    return RandomForestClassifier(
        n_estimators=1000,
        max_depth=7,
        min_samples_split=6,
        min_samples_leaf=6,
        max_features="sqrt",
        random_state=seed,
        n_jobs=-1,
    )


def build_extra(ref, target, inf, target_side, exclude_self):
    af, at = eligible_groups(ref, inf)
    typed = relation_features(
        ref,
        target,
        exclude_self=exclude_self,
        alpha=ALPHA,
        allowed_fam=af,
        allowed_tic=at,
    )
    counts = inference_count_features(ref, target, inf)
    graph = component_features(
        ref,
        target,
        inf,
        target_side=target_side,
        exclude_self=exclude_self,
    )
    return pd.concat(
        [
            typed.reset_index(drop=True),
            counts.reset_index(drop=True),
            graph.reset_index(drop=True),
        ],
        axis=1,
    )


def eval_one(train_df, y, base_cols, tr_idx, va_idx, variant, seed):
    tr = train_df.iloc[tr_idx].copy().reset_index(drop=True)
    va = train_df.iloc[va_idx].copy().reset_index(drop=True)
    tr_ex = build_extra(tr, tr, va, "ref", True)
    va_ex = build_extra(tr, va, va, "inf", False)
    cols = VARIANTS[variant]
    xtr = np.column_stack([tr[base_cols].to_numpy(float), tr_ex[cols].to_numpy()])
    xva = np.column_stack([va[base_cols].to_numpy(float), va_ex[cols].to_numpy()])
    xtr = StandardScaler().fit_transform(xtr)
    xva = StandardScaler().fit_transform(xva)
    m = rf(seed)
    m.fit(xtr, y[tr_idx])
    p = m.predict_proba(xva)[:, 1]
    return accuracy_score(y[va_idx], p >= 0.5), roc_auc_score(y[va_idx], p)


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    train_df, _, base_cols, _, _ = structural_frames(train_raw, test_raw)
    train_df["IsWomanChild"] = role_flags(train_raw)
    y = train_df["Survived"].astype(int).to_numpy()

    manifest = pd.read_csv(V1_AUDIT / "fold_manifest_seed42.csv")
    folds = manifest["fold"].astype(int).to_numpy()
    pseudo = pd.read_csv(V14_DIR / "pseudo_test_manifest.csv")
    rows = []

    for variant in VARIANTS:
        for f in sorted(np.unique(folds)):
            tr_idx = np.flatnonzero(folds != f)
            va_idx = np.flatnonzero(folds == f)
            acc, auc = eval_one(
                train_df, y, base_cols, tr_idx, va_idx, variant, 100 + int(f)
            )
            rows.append({
                "surface": "trusted",
                "split": int(f),
                "variant": variant,
                "accuracy": acc,
                "roc_auc": auc,
            })
        for s in sorted(pseudo["split"].unique()):
            va_ids = set(
                pseudo.loc[pseudo["split"] == s, "PassengerId"].astype(int)
            )
            va_mask = train_df["PassengerId"].astype(int).isin(va_ids).to_numpy()
            tr_idx = np.flatnonzero(~va_mask)
            va_idx = np.flatnonzero(va_mask)
            acc, auc = eval_one(
                train_df, y, base_cols, tr_idx, va_idx, variant, 200 + int(s)
            )
            rows.append({
                "surface": "pseudo",
                "split": int(s),
                "variant": variant,
                "accuracy": acc,
                "roc_auc": auc,
            })
        print(f"{variant} done", flush=True)

    metrics = pd.DataFrame(rows)
    summary = (
        metrics.groupby(["surface", "variant"])
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            min_accuracy=("accuracy", "min"),
            mean_auc=("roc_auc", "mean"),
            std_auc=("roc_auc", "std"),
        )
        .reset_index()
    )
    combined = (
        summary.groupby("variant")
        .agg(
            mean_surface_accuracy=("mean_accuracy", "mean"),
            worst_surface_accuracy=("mean_accuracy", "min"),
            mean_surface_auc=("mean_auc", "mean"),
            worst_surface_auc=("mean_auc", "min"),
        )
        .reset_index()
        .sort_values(
            ["worst_surface_accuracy", "mean_surface_auc"],
            ascending=False,
        )
    )
    metrics.to_csv(EXPORT_DIR / "graph_screen_metrics.csv", index=False)
    summary.to_csv(EXPORT_DIR / "graph_screen_summary.csv", index=False)
    combined.to_csv(EXPORT_DIR / "graph_screen_combined.csv", index=False)

    print("\n=== graph/count screen by surface ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n=== combined ranking ===")
    print(combined.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nNo Kaggle submission was performed.")


if __name__ == "__main__":
    main()
