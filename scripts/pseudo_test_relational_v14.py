"""Pseudo-test validation matched to the real Titanic test relation geometry.

Goals
-----
1. Build repeated validation holdouts whose *covariate / relation structure*
   resembles the real Kaggle test:
     - Family overlap with reference train
     - Ticket overlap with reference train
     - Family+Ticket overlap
     - mean peer counts
     - Sex / Pclass / child / Age-missing proportions
2. Compare relation feature blocks under those matched holdouts:
     - legacy: Gunes-style median SurvivalRate + availability
     - smooth: Bayesian-smoothed Family/Ticket rates + peer counts
     - typed: smooth features plus WomanChild / AdultMale peer rates

The split search never uses Survived. Target-derived features use reference
labels only, and reference rows are self-excluded.

No Kaggle submission is performed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from gunes_exact_foldsafe_v11 import structural_frames


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "exports" / "v14"

N_PSEUDO = 5
VALID_SIZE = 178
SEARCH_CANDIDATES = 12000
RNG_SEED = 1405


def surname(name: str) -> str:
    s = str(name)
    if "(" in s:
        s = s.split("(")[0]
    return re.sub(r"[^A-Za-z0-9 ]", "", s.split(",")[0]).strip()


def role_flags(df: pd.DataFrame) -> np.ndarray:
    title = (
        df["Name"].str.extract(r",\s*([^.]*)\.", expand=False).fillna("")
    )
    return (
        (df["Sex"] == "female")
        | (df["Age"].fillna(99) < 16)
        | title.eq("Master")
    ).astype(int).to_numpy()


def relation_arrays(train_raw: pd.DataFrame):
    d = train_raw.copy()
    d["Family"] = d["Name"].map(surname)
    d["FamilySize"] = d["SibSp"] + d["Parch"] + 1

    fam_codes, fam_uniques = pd.factorize(d["Family"], sort=False)
    tic_codes, tic_uniques = pd.factorize(d["Ticket"].astype(str), sort=False)
    fam_total = np.bincount(fam_codes, minlength=len(fam_uniques))
    tic_total = np.bincount(tic_codes, minlength=len(tic_uniques))
    fam_med = d.groupby("Family")["FamilySize"].median()
    fam_eligible = np.array(
        [float(fam_med.loc[g]) > 1 for g in fam_uniques],
        dtype=bool,
    )
    return d, fam_codes, tic_codes, fam_total, tic_total, fam_eligible


def real_test_target(train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> dict[str, float]:
    tr = train_raw.copy()
    te = test_raw.copy()
    tr["Family"] = tr["Name"].map(surname)
    te["Family"] = te["Name"].map(surname)
    tr["FamilySize"] = tr["SibSp"] + tr["Parch"] + 1
    te["FamilySize"] = te["SibSp"] + te["Parch"] + 1

    fam_count = tr["Family"].value_counts()
    fam_med = tr.groupby("Family")["FamilySize"].median()
    tic_count = tr["Ticket"].astype(str).value_counts()
    fam_peer = te["Family"].map(fam_count).fillna(0).astype(int).to_numpy()
    fam_ok = te["Family"].map(fam_med).fillna(0).to_numpy() > 1
    fam_peer = np.where(fam_ok, fam_peer, 0)
    tic_peer = (
        te["Ticket"].astype(str).map(tic_count).fillna(0).astype(int).to_numpy()
    )
    fam_avail = fam_peer > 0
    tic_avail = tic_peer > 0

    return {
        "family": float(fam_avail.mean()),
        "ticket": float(tic_avail.mean()),
        "both": float((fam_avail & tic_avail).mean()),
        "family_peer_mean": float(fam_peer.mean()),
        "ticket_peer_mean": float(tic_peer.mean()),
        "female": float((te["Sex"] == "female").mean()),
        "p1": float((te["Pclass"] == 1).mean()),
        "p2": float((te["Pclass"] == 2).mean()),
        "child": float((te["Age"].fillna(99) < 16).mean()),
        "age_missing": float(te["Age"].isna().mean()),
    }


def candidate_stats(
    train_raw: pd.DataFrame,
    idx: np.ndarray,
    fam_codes: np.ndarray,
    tic_codes: np.ndarray,
    fam_total: np.ndarray,
    tic_total: np.ndarray,
    fam_eligible: np.ndarray,
) -> dict[str, float]:
    fam_val_counts = np.bincount(
        fam_codes[idx], minlength=len(fam_total)
    )
    tic_val_counts = np.bincount(
        tic_codes[idx], minlength=len(tic_total)
    )
    fam_peer = fam_total[fam_codes[idx]] - fam_val_counts[fam_codes[idx]]
    tic_peer = tic_total[tic_codes[idx]] - tic_val_counts[tic_codes[idx]]
    fam_peer = np.where(fam_eligible[fam_codes[idx]], fam_peer, 0)
    fam_avail = fam_peer > 0
    tic_avail = tic_peer > 0
    v = train_raw.iloc[idx]
    return {
        "family": float(fam_avail.mean()),
        "ticket": float(tic_avail.mean()),
        "both": float((fam_avail & tic_avail).mean()),
        "family_peer_mean": float(fam_peer.mean()),
        "ticket_peer_mean": float(tic_peer.mean()),
        "female": float((v["Sex"] == "female").mean()),
        "p1": float((v["Pclass"] == 1).mean()),
        "p2": float((v["Pclass"] == 2).mean()),
        "child": float((v["Age"].fillna(99) < 16).mean()),
        "age_missing": float(v["Age"].isna().mean()),
    }


TOL = {
    "family": 0.025,
    "ticket": 0.025,
    "both": 0.025,
    "family_peer_mean": 0.12,
    "ticket_peer_mean": 0.12,
    "female": 0.025,
    "p1": 0.025,
    "p2": 0.025,
    "child": 0.020,
    "age_missing": 0.025,
}


def objective(stats: dict[str, float], target: dict[str, float]) -> float:
    return float(
        sum(
            ((stats[k] - target[k]) / TOL[k]) ** 2
            for k in TOL
        )
    )


def build_pseudo_splits(train_raw: pd.DataFrame, test_raw: pd.DataFrame):
    (
        _,
        fam_codes,
        tic_codes,
        fam_total,
        tic_total,
        fam_eligible,
    ) = relation_arrays(train_raw)
    target = real_test_target(train_raw, test_raw)
    rng = np.random.default_rng(RNG_SEED)

    candidates: list[tuple[float, np.ndarray, dict[str, float]]] = []
    n = len(train_raw)
    # Sample without using Survived. Pclass/Sex proportions are part of the
    # objective, so simple random sampling is sufficient and keeps the search
    # target-independent.
    for _ in range(SEARCH_CANDIDATES):
        idx = np.sort(rng.choice(n, size=VALID_SIZE, replace=False))
        s = candidate_stats(
            train_raw, idx, fam_codes, tic_codes, fam_total, tic_total, fam_eligible
        )
        candidates.append((objective(s, target), idx, s))

    candidates.sort(key=lambda x: x[0])
    # Greedily keep good but not near-identical holdouts.
    chosen = []
    for cand in candidates[:1000]:
        _, idx, _ = cand
        idx_set = set(idx.tolist())
        if not chosen:
            chosen.append(cand)
        else:
            max_jaccard = max(
                len(idx_set & set(c[1].tolist()))
                / len(idx_set | set(c[1].tolist()))
                for c in chosen
            )
            if max_jaccard <= 0.18:
                chosen.append(cand)
        if len(chosen) >= N_PSEUDO:
            break
    if len(chosen) < N_PSEUDO:
        # Relax diversity if needed; objective quality matters more.
        used = {tuple(c[1].tolist()) for c in chosen}
        for cand in candidates[:1000]:
            key = tuple(cand[1].tolist())
            if key not in used:
                chosen.append(cand)
                used.add(key)
            if len(chosen) >= N_PSEUDO:
                break

    return target, chosen


def eligible_groups(ref: pd.DataFrame, inf: pd.DataFrame):
    fam_stats = ref.groupby("Family")["Family_Size"].median()
    inf_fam = set(inf["Family"])
    inf_tic = set(inf["Ticket"])
    allowed_fam = {
        g for g, size in fam_stats.items()
        if g in inf_fam and float(size) > 1
    }
    allowed_tic = set(ref["Ticket"]) & inf_tic
    return allowed_fam, allowed_tic


def group_values(
    ref: pd.DataFrame,
    app: pd.DataFrame,
    *,
    group_col: str,
    allowed: set,
    exclude_self: bool,
    prior: float,
    alpha: float,
    role_specific: int | None = None,
):
    groups = ref.groupby(group_col, sort=False)
    rate = np.full(len(app), prior, dtype=float)
    count = np.zeros(len(app), dtype=float)
    median = np.full(len(app), prior, dtype=float)
    for pos, (_, row) in enumerate(app.iterrows()):
        key = row[group_col]
        if key not in allowed or key not in groups.groups:
            continue
        peers = ref.loc[groups.groups[key]]
        if exclude_self:
            peers = peers[peers["PassengerId"] != row["PassengerId"]]
        if role_specific is not None:
            peers = peers[peers["IsWomanChild"] == role_specific]
        if len(peers) == 0:
            continue
        n = float(len(peers))
        s = float(peers["Survived"].sum())
        count[pos] = n
        rate[pos] = (s + alpha * prior) / (n + alpha)
        median[pos] = float(peers["Survived"].median())
    return rate, count, median


def relation_features(
    ref: pd.DataFrame,
    app: pd.DataFrame,
    *,
    exclude_self: bool,
    alpha: float = 2.0,
    allowed_fam: set | None = None,
    allowed_tic: set | None = None,
) -> pd.DataFrame:
    prior = float(ref["Survived"].mean())
    if allowed_fam is None or allowed_tic is None:
        allowed_fam, allowed_tic = eligible_groups(ref, app)

    fam_s, fam_n, fam_med = group_values(
        ref, app, group_col="Family", allowed=allowed_fam,
        exclude_self=exclude_self, prior=prior, alpha=alpha
    )
    tic_s, tic_n, tic_med = group_values(
        ref, app, group_col="Ticket", allowed=allowed_tic,
        exclude_self=exclude_self, prior=prior, alpha=alpha
    )
    fam_wc, fam_wc_n, _ = group_values(
        ref, app, group_col="Family", allowed=allowed_fam,
        exclude_self=exclude_self, prior=prior, alpha=alpha, role_specific=1
    )
    fam_am, fam_am_n, _ = group_values(
        ref, app, group_col="Family", allowed=allowed_fam,
        exclude_self=exclude_self, prior=prior, alpha=alpha, role_specific=0
    )
    tic_wc, tic_wc_n, _ = group_values(
        ref, app, group_col="Ticket", allowed=allowed_tic,
        exclude_self=exclude_self, prior=prior, alpha=alpha, role_specific=1
    )
    tic_am, tic_am_n, _ = group_values(
        ref, app, group_col="Ticket", allowed=allowed_tic,
        exclude_self=exclude_self, prior=prior, alpha=alpha, role_specific=0
    )

    fam_av = (fam_n > 0).astype(float)
    tic_av = (tic_n > 0).astype(float)
    legacy_rate = (fam_med + tic_med) / 2.0
    legacy_na = (fam_av + tic_av) / 2.0
    total_n = fam_n + tic_n
    combined_smooth = np.where(
        total_n > 0,
        (
            (fam_s * (fam_n + alpha) - alpha * prior)
            + (tic_s * (tic_n + alpha) - alpha * prior)
            + alpha * prior
        )
        / (total_n + alpha),
        prior,
    )

    role = app["IsWomanChild"].to_numpy()
    same_fam = np.where(role == 1, fam_wc, fam_am)
    same_fam_n = np.where(role == 1, fam_wc_n, fam_am_n)
    same_tic = np.where(role == 1, tic_wc, tic_am)
    same_tic_n = np.where(role == 1, tic_wc_n, tic_am_n)
    same_total_n = same_fam_n + same_tic_n
    same_combined = np.where(
        same_total_n > 0,
        (
            (same_fam * (same_fam_n + alpha) - alpha * prior)
            + (same_tic * (same_tic_n + alpha) - alpha * prior)
            + alpha * prior
        )
        / (same_total_n + alpha),
        prior,
    )

    return pd.DataFrame(
        {
            "LegacyRate": legacy_rate,
            "LegacyNA": legacy_na,
            "FamilySmooth": fam_s,
            "FamilyCount": fam_n,
            "TicketSmooth": tic_s,
            "TicketCount": tic_n,
            "CombinedSmooth": combined_smooth,
            "EvidenceCount": total_n,
            "FamilyWC": fam_wc,
            "FamilyWCCount": fam_wc_n,
            "FamilyAM": fam_am,
            "FamilyAMCount": fam_am_n,
            "TicketWC": tic_wc,
            "TicketWCCount": tic_wc_n,
            "TicketAM": tic_am,
            "TicketAMCount": tic_am_n,
            "SameRoleFamily": same_fam,
            "SameRoleFamilyCount": same_fam_n,
            "SameRoleTicket": same_tic,
            "SameRoleTicketCount": same_tic_n,
            "SameRoleCombined": same_combined,
            "SameRoleEvidenceCount": same_total_n,
        },
        index=app.index,
    )


VARIANTS = {
    "legacy2": ["LegacyRate", "LegacyNA"],
    "smooth8": [
        "FamilySmooth", "FamilyCount", "TicketSmooth", "TicketCount",
        "CombinedSmooth", "EvidenceCount", "LegacyRate", "LegacyNA",
    ],
    "typed22": [
        "LegacyRate", "LegacyNA",
        "FamilySmooth", "FamilyCount", "TicketSmooth", "TicketCount",
        "CombinedSmooth", "EvidenceCount",
        "FamilyWC", "FamilyWCCount", "FamilyAM", "FamilyAMCount",
        "TicketWC", "TicketWCCount", "TicketAM", "TicketAMCount",
        "SameRoleFamily", "SameRoleFamilyCount",
        "SameRoleTicket", "SameRoleTicketCount",
        "SameRoleCombined", "SameRoleEvidenceCount",
    ],
}


def make_rf():
    return RandomForestClassifier(
        criterion="gini",
        n_estimators=1750,
        max_depth=7,
        min_samples_split=6,
        min_samples_leaf=6,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    target, chosen = build_pseudo_splits(train_raw, test_raw)

    # Structural features are the same exact Gunes-style 24 target-independent
    # columns used by the v11 reproduction code.
    train_df, test_df, base_cols, _, _ = structural_frames(train_raw, test_raw)
    role_train = role_flags(train_raw)
    role_test = role_flags(test_raw)
    train_df["IsWomanChild"] = role_train
    test_df["IsWomanChild"] = role_test
    y = train_df["Survived"].astype(int).to_numpy()

    split_rows = []
    manifest_rows = []
    metric_rows = []

    for split_id, (obj, va_idx, stats) in enumerate(chosen):
        tr_idx = np.setdiff1d(np.arange(len(train_df)), va_idx)
        tr = train_df.iloc[tr_idx].copy()
        va = train_df.iloc[va_idx].copy()

        split_rows.append(
            {
                "split": split_id,
                "objective": obj,
                **{f"actual_{k}": stats[k] for k in target},
                **{f"target_{k}": target[k] for k in target},
            }
        )
        for idx in va_idx:
            manifest_rows.append(
                {
                    "split": split_id,
                    "PassengerId": int(train_df.iloc[idx]["PassengerId"]),
                    "is_validation": 1,
                }
            )

        allowed_fam, allowed_tic = eligible_groups(tr, va)
        tr_rel = relation_features(
            tr,
            tr,
            exclude_self=True,
            alpha=2.0,
            allowed_fam=allowed_fam,
            allowed_tic=allowed_tic,
        )
        va_rel = relation_features(
            tr,
            va,
            exclude_self=False,
            alpha=2.0,
            allowed_fam=allowed_fam,
            allowed_tic=allowed_tic,
        )

        for variant, rel_cols in VARIANTS.items():
            xtr = np.column_stack(
                [tr[base_cols].to_numpy(dtype=float), tr_rel[rel_cols].to_numpy()]
            )
            xva = np.column_stack(
                [va[base_cols].to_numpy(dtype=float), va_rel[rel_cols].to_numpy()]
            )
            # Historical v10 behavior: fit scaler independently on partitions.
            xtr = StandardScaler().fit_transform(xtr)
            xva = StandardScaler().fit_transform(xva)
            model = make_rf()
            model.fit(xtr, y[tr_idx])
            prob = model.predict_proba(xva)[:, 1]
            metric_rows.append(
                {
                    "split": split_id,
                    "variant": variant,
                    "accuracy": accuracy_score(y[va_idx], prob >= 0.5),
                    "roc_auc": roc_auc_score(y[va_idx], prob),
                    "feature_count": len(base_cols) + len(rel_cols),
                    "pseudo_objective": obj,
                }
            )
            print(
                f"split={split_id} {variant}: "
                f"acc={metric_rows[-1]['accuracy']:.5f} "
                f"auc={metric_rows[-1]['roc_auc']:.5f}",
                flush=True,
            )

    split_df = pd.DataFrame(split_rows)
    metrics = pd.DataFrame(metric_rows)
    summary = (
        metrics.groupby("variant")
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            min_accuracy=("accuracy", "min"),
            mean_auc=("roc_auc", "mean"),
            std_auc=("roc_auc", "std"),
            min_auc=("roc_auc", "min"),
            feature_count=("feature_count", "first"),
        )
        .reset_index()
        .sort_values(["mean_accuracy", "mean_auc"], ascending=False)
    )

    # Paired deltas against the historical legacy relation block.
    wide_acc = metrics.pivot(index="split", columns="variant", values="accuracy")
    wide_auc = metrics.pivot(index="split", columns="variant", values="roc_auc")
    delta_rows = []
    for variant in VARIANTS:
        if variant == "legacy2":
            continue
        delta_rows.append(
            {
                "variant": variant,
                "mean_delta_accuracy_vs_legacy": float(
                    (wide_acc[variant] - wide_acc["legacy2"]).mean()
                ),
                "positive_accuracy_splits": int(
                    ((wide_acc[variant] - wide_acc["legacy2"]) > 0).sum()
                ),
                "mean_delta_auc_vs_legacy": float(
                    (wide_auc[variant] - wide_auc["legacy2"]).mean()
                ),
                "positive_auc_splits": int(
                    ((wide_auc[variant] - wide_auc["legacy2"]) > 0).sum()
                ),
            }
        )
    deltas = pd.DataFrame(delta_rows)

    split_df.to_csv(EXPORT_DIR / "pseudo_test_split_stats.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(
        EXPORT_DIR / "pseudo_test_manifest.csv", index=False
    )
    metrics.to_csv(EXPORT_DIR / "pseudo_test_relational_metrics.csv", index=False)
    summary.to_csv(EXPORT_DIR / "pseudo_test_relational_summary.csv", index=False)
    deltas.to_csv(EXPORT_DIR / "pseudo_test_relational_deltas.csv", index=False)
    with (EXPORT_DIR / "pseudo_test_target.json").open("w", encoding="utf-8") as f:
        json.dump(target, f, ensure_ascii=False, indent=2)

    print("\n=== Real-test matched pseudo split stats ===")
    print(split_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== Relational feature ranking ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n=== Paired deltas vs legacy2 ===")
    print(deltas.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nNo Kaggle submission was performed.")


if __name__ == "__main__":
    main()
