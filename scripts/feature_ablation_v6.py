"""Titanic v6 feature-engineering ablation on the trusted fold manifest.

The experiment is intentionally feature-first. It reuses the v2 preprocessing
and fold-safe WCG baseline, then adds four independently testable blocks:

  1) cabin_missing : missingness + cabin/name structure
  2) relational    : surname/ticket/family/travel-party structure
  3) interactions  : sex/class/title/age/fare/family interactions
  4) wcg_detail    : fold-safe target-derived peer counts and smoothed outcomes

Public Titanic notebooks/discussions repeatedly use title, family size, cabin,
ticket, sex/class interactions, and group survival. This script tests only the
parts that are missing or materially richer than the current v2 pipeline.

No Kaggle submission is performed.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imodels import RuleFitClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

try:
    from audit_group_survival import add_group_survival, build_base_frame, model_feature_columns
except ModuleNotFoundError:
    from scripts.audit_group_survival import add_group_survival, build_base_frame, model_feature_columns


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
V1_AUDIT = BASE_DIR / "exports" / "wcg_audit_v1"
EXPORT_DIR = BASE_DIR / "exports" / "v6"

BLOCKS = ("cabin_missing", "relational", "interactions", "wcg_detail")
VARIANTS = {
    "baseline": [],
    "cabin_missing": ["cabin_missing"],
    "relational": ["relational"],
    "interactions": ["interactions"],
    "wcg_detail": ["wcg_detail"],
    "structural_all": ["cabin_missing", "relational", "interactions"],
    "all": ["cabin_missing", "relational", "interactions", "wcg_detail"],
}


def title_series(name: pd.Series) -> pd.Series:
    title = name.str.extract(r" ([A-Za-z]+)\.", expand=False)
    mapping = {
        "Mr": "Mr", "Miss": "Miss", "Mlle": "Miss", "Ms": "Miss",
        "Mrs": "Mrs", "Mme": "Mrs", "Master": "Master",
        "Dr": "Officer", "Rev": "Officer", "Col": "Officer",
        "Major": "Officer", "Capt": "Officer",
        "Sir": "Royalty", "Don": "Royalty", "Countess": "Royalty",
        "Lady": "Royalty", "Dona": "Royalty", "Jonkheer": "Royalty",
    }
    return title.map(mapping).fillna("Other")


def union_find_components(keys_a: pd.Series, keys_b: pd.Series) -> np.ndarray:
    n = len(keys_a)
    parent = np.arange(n)
    size = np.ones(n, dtype=int)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    for keys in (keys_a, keys_b):
        groups: dict[object, list[int]] = {}
        for idx, key in enumerate(keys.tolist()):
            groups.setdefault(key, []).append(idx)
        for members in groups.values():
            if len(members) <= 1:
                continue
            first = members[0]
            for other in members[1:]:
                union(first, other)
    return np.array([find(i) for i in range(n)], dtype=int)


def build_target_independent_blocks(train_raw: pd.DataFrame, test_raw: pd.DataFrame):
    train = train_raw.copy()
    test = test_raw.copy()
    train["_source"] = "train"
    test["_source"] = "test"
    df = pd.concat([train, test], ignore_index=True, sort=False)

    df["TitleRaw"] = title_series(df["Name"])
    df["FamilySizeRaw"] = df["SibSp"] + df["Parch"] + 1
    df["Surname"] = df["Name"].str.split(",").str[0].str.strip()
    df["FamilyKey"] = df["Surname"] + "_" + df["FamilySizeRaw"].astype(str)
    df["AgeMissing_v6"] = df["Age"].isna().astype(int)
    df["CabinKnown_v6"] = df["Cabin"].notna().astype(int)
    df["EmbarkedMissing_v6"] = df["Embarked"].isna().astype(int)
    df["FareMissing_v6"] = df["Fare"].isna().astype(int)
    df["NameLength_v6"] = df["Name"].str.len().astype(float)
    df["CabinCount_v6"] = (
        df["Cabin"].fillna("").str.split().str.len().where(df["Cabin"].notna(), 0).astype(float)
    )
    cabin_number = df["Cabin"].fillna("").str.extract(r"(\d+)", expand=False)
    df["CabinNumber_v6"] = pd.to_numeric(cabin_number, errors="coerce").fillna(-1).astype(float)

    embarked_fill = df["Embarked"].fillna("S")
    fare_fill = df["Fare"].copy()
    med_fare = (
        pd.DataFrame({"Pclass": df["Pclass"], "Embarked": embarked_fill, "Fare": fare_fill})
        .groupby(["Pclass", "Embarked"])["Fare"]
        .transform("median")
    )
    fare_fill = fare_fill.fillna(med_fare).fillna(fare_fill.median())
    age_fill = df["Age"].copy()
    med_age = (
        pd.DataFrame({"Title": df["TitleRaw"], "Pclass": df["Pclass"], "Age": age_fill})
        .groupby(["Title", "Pclass"])["Age"]
        .transform("median")
    )
    age_fill = age_fill.fillna(med_age).fillna(age_fill.median())

    # Block 1: missingness / cabin / name.
    cabin_cols = [
        "PassengerId",
        "AgeMissing_v6",
        "CabinKnown_v6",
        "EmbarkedMissing_v6",
        "FareMissing_v6",
        "NameLength_v6",
        "CabinCount_v6",
        "CabinNumber_v6",
    ]
    cabin = df[cabin_cols].copy()

    # Block 2: target-independent relational structure.
    df["SurnameFreq_v6"] = df["Surname"].map(df["Surname"].value_counts()).astype(float)
    df["FamilyGroupFreq_v6"] = df["FamilyKey"].map(df["FamilyKey"].value_counts()).astype(float)
    df["SharedFamily_v6"] = (df["FamilyGroupFreq_v6"] > 1).astype(int)
    ticket_freq = df["Ticket"].map(df["Ticket"].value_counts()).astype(float)
    df["SharedTicket_v6"] = (ticket_freq > 1).astype(int)

    is_child = (age_fill <= 12).astype(int)
    is_female = (df["Sex"] == "female").astype(int)
    is_adult_male = ((df["Sex"] == "male") & (age_fill > 12)).astype(int)

    for key_col, prefix in [("Ticket", "Ticket"), ("FamilyKey", "Family")]:
        df[f"{prefix}FemaleCount_v6"] = df.groupby(key_col)["Sex"].transform(
            lambda s: (s == "female").sum()
        ).astype(float)
        df[f"{prefix}ChildCount_v6"] = (
            pd.Series(is_child, index=df.index).groupby(df[key_col]).transform("sum").astype(float)
        )
        df[f"{prefix}AdultMaleCount_v6"] = (
            pd.Series(is_adult_male, index=df.index).groupby(df[key_col]).transform("sum").astype(float)
        )

    roots = union_find_components(df["Ticket"], df["FamilyKey"])
    df["TravelRoot_v6"] = roots
    df["TravelGroupSize_v6"] = df.groupby("TravelRoot_v6")["PassengerId"].transform("size").astype(float)
    df["TravelFemaleCount_v6"] = (
        pd.Series(is_female, index=df.index).groupby(df["TravelRoot_v6"]).transform("sum").astype(float)
    )
    df["TravelChildCount_v6"] = (
        pd.Series(is_child, index=df.index).groupby(df["TravelRoot_v6"]).transform("sum").astype(float)
    )
    df["TravelAdultMaleCount_v6"] = (
        pd.Series(is_adult_male, index=df.index).groupby(df["TravelRoot_v6"]).transform("sum").astype(float)
    )
    df["TravelCabinKnownRate_v6"] = (
        df.groupby("TravelRoot_v6")["CabinKnown_v6"].transform("mean").astype(float)
    )
    relational_cols = [
        "PassengerId",
        "SurnameFreq_v6",
        "FamilyGroupFreq_v6",
        "SharedFamily_v6",
        "SharedTicket_v6",
        "TicketFemaleCount_v6",
        "TicketChildCount_v6",
        "TicketAdultMaleCount_v6",
        "FamilyFemaleCount_v6",
        "FamilyChildCount_v6",
        "FamilyAdultMaleCount_v6",
        "TravelGroupSize_v6",
        "TravelFemaleCount_v6",
        "TravelChildCount_v6",
        "TravelAdultMaleCount_v6",
        "TravelCabinKnownRate_v6",
    ]
    relational = df[relational_cols].copy()

    # Block 3: explicit interactions/bands for linear/rule/neural models.
    df["IsMother_v6"] = (
        (df["Sex"] == "female") & (age_fill > 18) & (df["Parch"] > 0) & (df["TitleRaw"] != "Miss")
    ).astype(int)
    df["AdultMale_v6"] = ((df["Sex"] == "male") & (age_fill >= 18)).astype(int)
    df["YoungChild_v6"] = (age_fill <= 6).astype(int)
    df["Teen_v6"] = ((age_fill > 12) & (age_fill < 18)).astype(int)
    df["Senior_v6"] = (age_fill >= 60).astype(int)
    df["FamilyClass_v6"] = df["FamilySizeRaw"] * df["Pclass"]
    df["AgeFamilyRatio_v6"] = age_fill / df["FamilySizeRaw"].clip(lower=1)
    df["FarePerFamily_v6"] = fare_fill / df["FamilySizeRaw"].clip(lower=1)
    df["SexPclass_v6"] = df["Sex"].astype(str) + "_P" + df["Pclass"].astype(str)
    df["TitlePclass_v6"] = df["TitleRaw"].astype(str) + "_P" + df["Pclass"].astype(str)
    df["AgeBand_v6"] = pd.cut(
        age_fill,
        bins=[-np.inf, 6, 12, 17, 25, 40, 60, np.inf],
        labels=["0_6", "7_12", "13_17", "18_25", "26_40", "41_60", "61_plus"],
    )
    try:
        df["FareBand_v6"] = pd.qcut(
            np.log1p(fare_fill), q=6, labels=False, duplicates="drop"
        ).astype(str)
    except ValueError:
        df["FareBand_v6"] = "0"
    inter_base = df[
        [
            "PassengerId",
            "IsMother_v6",
            "AdultMale_v6",
            "YoungChild_v6",
            "Teen_v6",
            "Senior_v6",
            "FamilyClass_v6",
            "AgeFamilyRatio_v6",
            "FarePerFamily_v6",
            "SexPclass_v6",
            "TitlePclass_v6",
            "AgeBand_v6",
            "FareBand_v6",
        ]
    ].copy()
    interactions = pd.get_dummies(
        inter_base,
        columns=["SexPclass_v6", "TitlePclass_v6", "AgeBand_v6", "FareBand_v6"],
        drop_first=False,
        dtype=int,
    )

    def split_block(block: pd.DataFrame):
        merged = df[["PassengerId", "_source"]].merge(block, on="PassengerId", how="left")
        tr = merged[merged["_source"] == "train"].drop(columns="_source").reset_index(drop=True)
        te = merged[merged["_source"] == "test"].drop(columns="_source").reset_index(drop=True)
        return tr, te

    return {
        "cabin_missing": split_block(cabin),
        "relational": split_block(relational),
        "interactions": split_block(interactions),
    }


def peer_target_features(
    reference_labeled: pd.DataFrame,
    apply_df: pd.DataFrame,
    *,
    exclude_self: bool,
) -> pd.DataFrame:
    """Detailed leakage-safe WCG target statistics for Ticket and FamilyGroup."""
    ref = reference_labeled[reference_labeled["IsWomanOrChild"] == 1].copy()
    out = pd.DataFrame(index=apply_df.index)
    for group_col, prefix in [("Ticket", "TicketWCG"), ("FamilyGroup", "FamilyWCG")]:
        groups = ref.groupby(group_col, sort=False)
        mean_vals = []
        smooth_vals = []
        counts = []
        has_peers = []
        unanimous = []
        for _, row in apply_df.iterrows():
            key = row[group_col]
            if key not in groups.groups:
                peers = ref.iloc[0:0]
            else:
                peers = ref.loc[groups.groups[key]]
                if exclude_self:
                    peers = peers[peers["PassengerId"] != row["PassengerId"]]
            count = len(peers)
            counts.append(float(count))
            has_peers.append(int(count > 0))
            if count == 0:
                mean_vals.append(0.5)
                smooth_vals.append(0.5)
                unanimous.append(0)
            else:
                m = float(peers["Survived"].mean())
                mean_vals.append(m)
                smooth_vals.append(float((peers["Survived"].sum() + 1.0) / (count + 2.0)))
                unanimous.append(int(m in (0.0, 1.0)))
        out[f"{prefix}Mean_v6"] = mean_vals
        out[f"{prefix}Smooth_v6"] = smooth_vals
        out[f"{prefix}PeerCount_v6"] = counts
        out[f"{prefix}HasPeers_v6"] = has_peers
        out[f"{prefix}Unanimous_v6"] = unanimous
    return out


def build_models(seed: int):
    return {
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=120, max_depth=3, learning_rate=0.035, subsample=0.85,
            random_state=seed,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=140, max_depth=3, learning_rate=0.035,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=2,
            reg_lambda=1.2, random_state=seed, eval_metric="logloss", n_jobs=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=180, depth=4, learning_rate=0.035, l2_leaf_reg=4,
            random_seed=seed, verbose=0, thread_count=-1,
        ),
        "RuleFit": RuleFitClassifier(
            n_estimators=160, tree_size=4, max_rules=40,
            include_linear=True, random_state=seed, verbose=0,
        ),
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.7, max_iter=3000, random_state=seed),
        ),
    }


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    train_base, test_base = build_base_frame(train_raw, test_raw, profile="v2")
    base_cols = model_feature_columns(train_base)
    blocks = build_target_independent_blocks(train_raw, test_raw)

    manifest = pd.read_csv(V1_AUDIT / "fold_manifest_seed42.csv")
    if not train_base["PassengerId"].equals(manifest["PassengerId"]):
        raise ValueError("PassengerId does not match trusted fold manifest.")
    folds = manifest["fold"].astype(int).to_numpy()
    splits = [
        (fold, np.flatnonzero(folds != fold), np.flatnonzero(folds == fold))
        for fold in sorted(np.unique(folds))
    ]
    y = train_base["Survived"].astype(int).reset_index(drop=True)
    model_names = list(build_models(42))

    all_oof = pd.DataFrame({
        "PassengerId": train_base["PassengerId"],
        "fold": folds,
        "Survived": y,
    })
    all_test = pd.DataFrame({"PassengerId": test_base["PassengerId"]})
    fold_rows = []
    summary_rows = []
    feature_rows = []

    for variant, enabled_blocks in VARIANTS.items():
        print(f"\n=== v6 variant: {variant} / blocks={enabled_blocks} ===", flush=True)
        variant_oof = {m: np.zeros(len(train_base), dtype=float) for m in model_names}
        variant_test = {m: [] for m in model_names}
        variant_feature_names: list[str] | None = None

        for fold, tr_idx, va_idx in splits:
            tr = train_base.iloc[tr_idx].copy()
            va = train_base.iloc[va_idx].copy()
            te = test_base.copy()
            tr["GroupSurvival"] = add_group_survival(tr, tr, exclude_self=True).to_numpy()
            va["GroupSurvival"] = add_group_survival(tr, va, exclude_self=False).to_numpy()
            te["GroupSurvival"] = add_group_survival(tr, te, exclude_self=False).to_numpy()

            extra_cols = []
            for block in enabled_blocks:
                if block == "wcg_detail":
                    tr_w = peer_target_features(tr, tr, exclude_self=True)
                    va_w = peer_target_features(tr, va, exclude_self=False)
                    te_w = peer_target_features(tr, te, exclude_self=False)
                    for col in tr_w.columns:
                        tr[col] = tr_w[col].to_numpy()
                        va[col] = va_w[col].to_numpy()
                        te[col] = te_w[col].to_numpy()
                        extra_cols.append(col)
                else:
                    block_tr, block_te = blocks[block]
                    cols = [c for c in block_tr.columns if c != "PassengerId"]
                    tr = tr.merge(block_tr[["PassengerId"] + cols], on="PassengerId", how="left")
                    va = va.merge(block_tr[["PassengerId"] + cols], on="PassengerId", how="left")
                    te = te.merge(block_te[["PassengerId"] + cols], on="PassengerId", how="left")
                    extra_cols.extend(cols)

            feature_cols = base_cols + list(dict.fromkeys(extra_cols))
            if variant_feature_names is None:
                variant_feature_names = feature_cols
            if tr[feature_cols].isna().any().any() or va[feature_cols].isna().any().any():
                bad = tr[feature_cols].columns[tr[feature_cols].isna().any()].tolist()
                raise ValueError(f"NaN in {variant} features: {bad}")

            for model_name, model in build_models(42 + fold).items():
                started = time.time()
                model.fit(tr[feature_cols], y.iloc[tr_idx])
                val_prob = np.asarray(model.predict_proba(va[feature_cols]))[:, 1]
                test_prob = np.asarray(model.predict_proba(te[feature_cols]))[:, 1]
                variant_oof[model_name][va_idx] = val_prob
                variant_test[model_name].append(test_prob)
                fold_rows.append({
                    "variant": variant,
                    "fold": int(fold),
                    "model": model_name,
                    "accuracy": accuracy_score(y.iloc[va_idx], val_prob > 0.5),
                    "roc_auc": roc_auc_score(y.iloc[va_idx], val_prob),
                    "feature_count": len(feature_cols),
                    "seconds": time.time() - started,
                })

        # Equal-probability panel is diagnostic, not an optimized blend.
        panel_oof = np.mean([variant_oof[m] for m in model_names], axis=0)
        panel_test = np.mean(
            [np.mean(variant_test[m], axis=0) for m in model_names], axis=0
        )
        all_oof[f"{variant}__Panel"] = panel_oof
        all_test[f"{variant}__Panel"] = panel_test
        summary_rows.append({
            "variant": variant,
            "model": "Panel",
            "accuracy": accuracy_score(y, panel_oof > 0.5),
            "roc_auc": roc_auc_score(y, panel_oof),
            "feature_count": len(variant_feature_names or []),
        })
        for model_name in model_names:
            prob = variant_oof[model_name]
            test_prob = np.mean(variant_test[model_name], axis=0)
            all_oof[f"{variant}__{model_name}"] = prob
            all_test[f"{variant}__{model_name}"] = test_prob
            summary_rows.append({
                "variant": variant,
                "model": model_name,
                "accuracy": accuracy_score(y, prob > 0.5),
                "roc_auc": roc_auc_score(y, prob),
                "feature_count": len(variant_feature_names or []),
            })
        feature_rows.append({
            "variant": variant,
            "blocks": ",".join(enabled_blocks) if enabled_blocks else "none",
            "feature_count": len(variant_feature_names or []),
            "features": "|".join(variant_feature_names or []),
        })

    summary = pd.DataFrame(summary_rows)
    baseline = summary[summary["variant"] == "baseline"][
        ["model", "accuracy", "roc_auc"]
    ].rename(columns={"accuracy": "baseline_accuracy", "roc_auc": "baseline_roc_auc"})
    summary = summary.merge(baseline, on="model", how="left")
    summary["delta_accuracy_vs_baseline"] = summary["accuracy"] - summary["baseline_accuracy"]
    summary["delta_auc_vs_baseline"] = summary["roc_auc"] - summary["baseline_roc_auc"]
    summary = summary.sort_values(["model", "accuracy", "roc_auc"], ascending=[True, False, False])

    pd.DataFrame(fold_rows).to_csv(EXPORT_DIR / "feature_ablation_fold_metrics.csv", index=False)
    summary.to_csv(EXPORT_DIR / "feature_ablation_summary.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(EXPORT_DIR / "feature_sets.csv", index=False)
    all_oof.to_csv(EXPORT_DIR / "feature_ablation_oof.csv", index=False)
    all_test.to_csv(EXPORT_DIR / "feature_ablation_test.csv", index=False)

    best_by_model = (
        summary.sort_values(["model", "accuracy", "roc_auc"], ascending=[True, False, False])
        .groupby("model", as_index=False)
        .first()
    )
    best_by_model.to_csv(EXPORT_DIR / "best_variant_by_model.csv", index=False)

    with (EXPORT_DIR / "feature_ablation_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "version": "v6",
                "fold_manifest": str(V1_AUDIT / "fold_manifest_seed42.csv"),
                "variants": VARIANTS,
                "models": model_names,
                "note": "All target-derived WCG-detail features are constructed inside each fold.",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== v6 best variant by model ===")
    print(best_by_model.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n=== v6 panel variants ===")
    print(
        summary[summary["model"] == "Panel"]
        .sort_values(["accuracy", "roc_auc"], ascending=False)
        .to_string(index=False, float_format=lambda x: f"{x:.5f}")
    )
    print(f"\nArtifacts: {EXPORT_DIR}")


if __name__ == "__main__":
    main()
