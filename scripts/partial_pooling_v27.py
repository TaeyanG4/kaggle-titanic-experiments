"""v27: empirical-Bayes partial pooling for Titanic relational survival features.

This is intentionally different from the existing typed22 block. typed22 uses
a fixed global prior and fixed smoothing strength. Here we:

1. estimate an empirical shrinkage strength from fold-train groups;
2. use Role(WomanChild/AdultMale) x Pclass priors for each application row;
3. shrink Family, Ticket, and FamilyFare peer evidence toward those priors;
4. compare EB-only and typed22+EB representations on fixed and alternate folds;
5. report shift-weighted Accuracy using the v26 cross-fitted train/test weights.

All target-derived statistics are created strictly from fold-train labels.
No Kaggle submission is performed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from gunes_exact_foldsafe_v11 import structural_frames
from group_key_audit_v6 import helper_keys
from pseudo_test_relational_v14 import VARIANTS as REL_VARIANTS, make_rf, relation_features, role_flags


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "exports" / "v27"
V1_AUDIT = BASE_DIR / "exports" / "wcg_audit_v1"
V26_DIR = BASE_DIR / "exports" / "v26"

SEEDS = [42, 123, 777, 2026]
TYPED_COLS = REL_VARIANTS["typed22"]


def context_priors(ref: pd.DataFrame):
    global_prior = float(ref["Survived"].mean())
    role_prior = ref.groupby("IsWomanChild")["Survived"].mean().to_dict()
    role_class = ref.groupby(["IsWomanChild", "Pclass"])["Survived"].mean().to_dict()
    return global_prior, role_prior, role_class


def row_prior(row: pd.Series, global_prior: float, role_prior: dict, role_class: dict) -> float:
    key = (int(row["IsWomanChild"]), int(row["Pclass"]))
    if key in role_class:
        return float(role_class[key])
    role = int(row["IsWomanChild"])
    return float(role_prior.get(role, global_prior))


def estimate_alpha(ref: pd.DataFrame, group_col: str, *, same_role: bool) -> float:
    """Method-of-moments empirical Beta prior strength, clipped for stability."""
    d = ref.copy()
    keys = [group_col, "IsWomanChild"] if same_role else [group_col]
    agg = d.groupby(keys, sort=False)["Survived"].agg(["sum", "count"])
    agg = agg[agg["count"] >= 2].copy()
    if len(agg) < 5:
        return 4.0
    p = (agg["sum"] / agg["count"]).to_numpy(float)
    n = agg["count"].to_numpy(float)
    m = float(d["Survived"].mean())
    between = float(np.var(p, ddof=1))
    bin_noise = float(np.mean(max(m * (1 - m), 1e-6) / n))
    tau2 = max(between - bin_noise, 1e-4)
    alpha = m * (1 - m) / tau2 - 1.0
    return float(np.clip(alpha, 1.0, 20.0))


def family_allowed(ref: pd.DataFrame) -> set[str]:
    # Match the project's protection against unrelated singleton surnames.
    med = ref.groupby("Family")["Family_Size"].median()
    return {g for g, size in med.items() if float(size) > 1}


def eb_group(
    ref: pd.DataFrame,
    app: pd.DataFrame,
    *,
    group_col: str,
    exclude_self: bool,
    same_role: bool,
    alpha: float,
    allowed: set | None = None,
):
    global_prior, role_prior, role_class = context_priors(ref)
    grouped = ref.groupby(group_col, sort=False)
    posterior = np.zeros(len(app), dtype=float)
    count = np.zeros(len(app), dtype=float)
    deviation = np.zeros(len(app), dtype=float)
    prior_arr = np.zeros(len(app), dtype=float)
    for pos, (_, row) in enumerate(app.iterrows()):
        prior = row_prior(row, global_prior, role_prior, role_class)
        prior_arr[pos] = prior
        key = row[group_col]
        if allowed is not None and key not in allowed:
            posterior[pos] = prior
            continue
        if key not in grouped.groups:
            posterior[pos] = prior
            continue
        peers = ref.loc[grouped.groups[key]]
        if exclude_self:
            peers = peers[peers["PassengerId"] != row["PassengerId"]]
        if same_role:
            peers = peers[peers["IsWomanChild"] == int(row["IsWomanChild"])]
        n = float(len(peers))
        if n == 0:
            posterior[pos] = prior
            continue
        s = float(peers["Survived"].sum())
        post = (s + alpha * prior) / (n + alpha)
        posterior[pos] = post
        count[pos] = n
        deviation[pos] = post - prior
    return posterior, count, deviation, prior_arr


def eb_features(ref: pd.DataFrame, app: pd.DataFrame, *, exclude_self: bool) -> tuple[pd.DataFrame, dict[str, float]]:
    allowed_family = family_allowed(ref)
    specs = [
        ("Family", "Family", allowed_family),
        ("Ticket", "Ticket", None),
        ("FamilyFare", "FamilyFareGroup_v6", None),
    ]
    out = pd.DataFrame(index=app.index)
    alpha_meta = {}
    context_prior = None
    all_posts = []; all_counts = []
    role_posts = []; role_counts = []

    for prefix, col, allowed in specs:
        a_all = estimate_alpha(ref, col, same_role=False)
        a_role = estimate_alpha(ref, col, same_role=True)
        alpha_meta[f"{prefix}_alpha"] = a_all
        alpha_meta[f"{prefix}_role_alpha"] = a_role

        p, n, d, prior = eb_group(
            ref, app, group_col=col, exclude_self=exclude_self,
            same_role=False, alpha=a_all, allowed=allowed,
        )
        rp, rn, rd, _ = eb_group(
            ref, app, group_col=col, exclude_self=exclude_self,
            same_role=True, alpha=a_role, allowed=allowed,
        )
        if context_prior is None:
            context_prior = prior
        out[f"EB_{prefix}"] = p
        out[f"EB_{prefix}Count"] = n
        out[f"EB_{prefix}Delta"] = d
        out[f"EB_{prefix}SameRole"] = rp
        out[f"EB_{prefix}SameRoleCount"] = rn
        out[f"EB_{prefix}SameRoleDelta"] = rd
        all_posts.append(p); all_counts.append(n)
        role_posts.append(rp); role_counts.append(rn)

    all_posts = np.vstack(all_posts); all_counts = np.vstack(all_counts)
    role_posts = np.vstack(role_posts); role_counts = np.vstack(role_counts)
    prior = np.asarray(context_prior)
    # Reliability-weighted combination; no-evidence rows remain at context prior.
    rel = all_counts / (all_counts + 4.0)
    rrel = role_counts / (role_counts + 4.0)
    denom = rel.sum(axis=0)
    rdenom = rrel.sum(axis=0)
    out["EB_ContextPrior"] = prior
    out["EB_Combined"] = np.where(denom > 0, (rel * all_posts).sum(axis=0) / np.maximum(denom, 1e-9), prior)
    out["EB_SameRoleCombined"] = np.where(rdenom > 0, (rrel * role_posts).sum(axis=0) / np.maximum(rdenom, 1e-9), prior)
    out["EB_EvidenceCount"] = all_counts.sum(axis=0)
    out["EB_SameRoleEvidenceCount"] = role_counts.sum(axis=0)
    out["EB_FamilyTicketGap"] = np.abs(out["EB_Family"] - out["EB_Ticket"])
    out["EB_RoleFamilyTicketGap"] = np.abs(out["EB_FamilySameRole"] - out["EB_TicketSameRole"])
    return out, alpha_meta


EB_COLS = [
    "EB_ContextPrior",
    "EB_Family", "EB_FamilyCount", "EB_FamilyDelta",
    "EB_FamilySameRole", "EB_FamilySameRoleCount", "EB_FamilySameRoleDelta",
    "EB_Ticket", "EB_TicketCount", "EB_TicketDelta",
    "EB_TicketSameRole", "EB_TicketSameRoleCount", "EB_TicketSameRoleDelta",
    "EB_FamilyFare", "EB_FamilyFareCount", "EB_FamilyFareDelta",
    "EB_FamilyFareSameRole", "EB_FamilyFareSameRoleCount", "EB_FamilyFareSameRoleDelta",
    "EB_Combined", "EB_SameRoleCombined", "EB_EvidenceCount", "EB_SameRoleEvidenceCount",
    "EB_FamilyTicketGap", "EB_RoleFamilyTicketGap",
]


VARIANTS = {
    "typed22": TYPED_COLS,
    "eb25": EB_COLS,
    "typed22_plus_eb25": TYPED_COLS + EB_COLS,
}


def attach_helpers(train_df, test_df, train_raw, test_raw):
    tr_help, te_help = helper_keys(train_raw, test_raw)
    tr = train_df.merge(tr_help[["PassengerId", "FamilyFareGroup_v6"]], on="PassengerId", how="left")
    te = test_df.merge(te_help[["PassengerId", "FamilyFareGroup_v6"]], on="PassengerId", how="left")
    return tr, te


def eval_seed(train_df: pd.DataFrame, base_cols: list[str], folds: np.ndarray, seed_label: int):
    y = train_df["Survived"].astype(int).to_numpy()
    oof = {k: np.zeros(len(train_df), dtype=float) for k in VARIANTS}
    fold_rows = []
    alpha_rows = []
    for fold in sorted(np.unique(folds)):
        tr_idx = np.flatnonzero(folds != fold); va_idx = np.flatnonzero(folds == fold)
        tr = train_df.iloc[tr_idx].copy(); va = train_df.iloc[va_idx].copy()
        # typed22 baseline using the existing fold-safe implementation.
        tr_typed = relation_features(tr, tr, exclude_self=True, alpha=2.0)
        va_typed = relation_features(tr, va, exclude_self=False, alpha=2.0)
        tr_eb, alphas = eb_features(tr, tr, exclude_self=True)
        va_eb, _ = eb_features(tr, va, exclude_self=False)
        alpha_rows.append({"seed": seed_label, "fold": int(fold), **alphas})

        blocks_tr = {"typed22": tr_typed, "eb25": tr_eb, "typed22_plus_eb25": pd.concat([tr_typed.reset_index(drop=True), tr_eb.reset_index(drop=True)], axis=1)}
        blocks_va = {"typed22": va_typed, "eb25": va_eb, "typed22_plus_eb25": pd.concat([va_typed.reset_index(drop=True), va_eb.reset_index(drop=True)], axis=1)}
        for variant, rel_cols in VARIANTS.items():
            bt = blocks_tr[variant]; bv = blocks_va[variant]
            xtr = np.column_stack([tr[base_cols].to_numpy(float), bt[rel_cols].to_numpy(float)])
            xva = np.column_stack([va[base_cols].to_numpy(float), bv[rel_cols].to_numpy(float)])
            # Preserve historical project comparison semantics.
            xtr = StandardScaler().fit_transform(xtr)
            xva = StandardScaler().fit_transform(xva)
            model = make_rf()
            model.fit(xtr, y[tr_idx])
            p = model.predict_proba(xva)[:, 1]
            oof[variant][va_idx] = p
            fold_rows.append({
                "seed": seed_label, "fold": int(fold), "variant": variant,
                "accuracy": accuracy_score(y[va_idx], p >= .5),
                "roc_auc": roc_auc_score(y[va_idx], p),
            })
    summary = []
    for variant, p in oof.items():
        f = pd.DataFrame([r for r in fold_rows if r["variant"] == variant])
        summary.append({
            "seed": seed_label, "variant": variant,
            "accuracy": accuracy_score(y, p >= .5),
            "roc_auc": roc_auc_score(y, p),
            "fold_acc_std": float(f["accuracy"].std(ddof=0)),
        })
    return pd.DataFrame(summary), pd.DataFrame(fold_rows), pd.DataFrame(alpha_rows), oof


def weighted_accuracy(pred: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(weights * (pred == y)) / np.sum(weights))


def main():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_raw = pd.read_csv(DATA_DIR / "train.csv"); test_raw = pd.read_csv(DATA_DIR / "test.csv")
    train_df, test_df, base_cols, _, _ = structural_frames(train_raw, test_raw)
    train_df["IsWomanChild"] = role_flags(train_raw); test_df["IsWomanChild"] = role_flags(test_raw)
    train_df, test_df = attach_helpers(train_df, test_df, train_raw, test_raw)
    y = train_df["Survived"].astype(int).to_numpy()

    manifest = pd.read_csv(V1_AUDIT / "fold_manifest_seed42.csv")
    fixed = manifest["fold"].astype(int).to_numpy()
    summaries = []; folds_all = []; alphas_all = []; fixed_oof = None

    for seed in SEEDS:
        if seed == 42:
            folds = fixed
        else:
            folds = np.full(len(train_df), -1, dtype=int)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for fold, (_, va_idx) in enumerate(skf.split(train_df, y)):
                folds[va_idx] = fold
        print(f"=== partial-pooling seed {seed} ===", flush=True)
        s, f, a, oof = eval_seed(train_df, base_cols, folds, seed)
        summaries.append(s); folds_all.append(f); alphas_all.append(a)
        if seed == 42:
            fixed_oof = oof
        print(s.to_string(index=False, float_format=lambda v:f"{v:.5f}"), flush=True)

    summary = pd.concat(summaries, ignore_index=True)
    fold_df = pd.concat(folds_all, ignore_index=True)
    alpha_df = pd.concat(alphas_all, ignore_index=True)
    summary.to_csv(EXPORT_DIR / "partial_pooling_seed_summary.csv", index=False)
    fold_df.to_csv(EXPORT_DIR / "partial_pooling_fold_metrics.csv", index=False)
    alpha_df.to_csv(EXPORT_DIR / "partial_pooling_alphas.csv", index=False)

    overall = summary.groupby("variant", as_index=False).agg(
        mean_accuracy=("accuracy", "mean"), min_accuracy=("accuracy", "min"),
        std_accuracy=("accuracy", "std"), mean_auc=("roc_auc", "mean"), min_auc=("roc_auc", "min")
    ).sort_values(["mean_accuracy", "mean_auc"], ascending=False)
    overall.to_csv(EXPORT_DIR / "partial_pooling_overall.csv", index=False)

    oof_export = pd.DataFrame({"PassengerId": train_df["PassengerId"].astype(int), "Survived": y})
    for variant, p in fixed_oof.items(): oof_export[variant] = p
    oof_export.to_csv(EXPORT_DIR / "partial_pooling_fixed_oof.csv", index=False)

    # Shift-weighted fixed-OOF check from v26. Use both independent propensity estimators.
    wdf = pd.read_csv(V26_DIR / "train_shift_weights.csv")
    weighted_rows = []
    for wm in ["logistic", "lightgbm"]:
        w = wdf[wdf["weight_model"] == wm].sort_values("train_row")["weight"].to_numpy(float)
        for variant, p in fixed_oof.items():
            pred = (p >= .5).astype(int)
            weighted_rows.append({
                "weight_model": wm, "variant": variant,
                "weighted_accuracy": weighted_accuracy(pred, y, w),
                "unweighted_accuracy": accuracy_score(y, pred),
            })
    weighted = pd.DataFrame(weighted_rows)
    weighted.to_csv(EXPORT_DIR / "partial_pooling_shift_weighted.csv", index=False)

    print("\n=== partial-pooling overall ===")
    print(overall.to_string(index=False, float_format=lambda v:f"{v:.5f}"))
    print("\n=== shift-weighted fixed OOF ===")
    print(weighted.to_string(index=False, float_format=lambda v:f"{v:.5f}"))
    print("\nNo Kaggle submission was performed.")


if __name__ == "__main__":
    main()
