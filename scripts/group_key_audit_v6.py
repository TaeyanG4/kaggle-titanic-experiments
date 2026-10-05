"""Audit alternative Titanic family/ticket group-survival representations.

Motivation from public Titanic feature-engineering practice:
- families are sometimes keyed by LastName + Fare rather than LastName + FamilySize;
- a peer-survival feature often uses "any peer survived" instead of requiring
  unanimous peer outcomes.

All target-derived features here are constructed strictly inside each fold.
No Kaggle submission is performed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

try:
    from audit_group_survival import add_group_survival, build_base_frame, model_feature_columns
    from feature_ablation_v6 import DATA_DIR, V1_AUDIT, EXPORT_DIR, build_models, title_series
except ModuleNotFoundError:
    from scripts.audit_group_survival import add_group_survival, build_base_frame, model_feature_columns
    from scripts.feature_ablation_v6 import DATA_DIR, V1_AUDIT, EXPORT_DIR, build_models, title_series


MODEL_NAMES = ["GradientBoosting", "XGBoost", "CatBoost"]


def helper_keys(train_raw: pd.DataFrame, test_raw: pd.DataFrame):
    train = train_raw.copy()
    test = test_raw.copy()
    train["_source"] = "train"
    test["_source"] = "test"
    df = pd.concat([train, test], ignore_index=True, sort=False)
    df["Surname_v6"] = df["Name"].str.split(",").str[0].str.strip()
    df["Title_v6"] = title_series(df["Name"])
    embarked = df["Embarked"].fillna("S")
    fare = df["Fare"].copy()
    med_fare = (
        pd.DataFrame({"Pclass": df["Pclass"], "Embarked": embarked, "Fare": fare})
        .groupby(["Pclass", "Embarked"])["Fare"]
        .transform("median")
    )
    fare = fare.fillna(med_fare).fillna(fare.median())
    # Round only for stable string formatting. Titanic family-ticket fares are
    # already represented at low decimal precision.
    df["FamilyFareGroup_v6"] = df["Surname_v6"] + "_" + fare.round(4).astype(str)
    df["SurnameGroup_v6"] = df["Surname_v6"]
    # Missing cabins must not become one giant shared group.
    df["CabinGroup_v6"] = df["Cabin"].fillna(
        "__MISSING_" + df["PassengerId"].astype(str)
    )
    cols = ["PassengerId", "FamilyFareGroup_v6", "SurnameGroup_v6", "CabinGroup_v6"]
    tr = df[df["_source"] == "train"][cols].reset_index(drop=True)
    te = df[df["_source"] == "test"][cols].reset_index(drop=True)
    return tr, te


def add_helpers(base: pd.DataFrame, helpers: pd.DataFrame) -> pd.DataFrame:
    return base.merge(helpers, on="PassengerId", how="left")


def peer_feature(
    reference: pd.DataFrame,
    apply_df: pd.DataFrame,
    *,
    group_col: str,
    wcg_only: bool,
    exclude_self: bool,
) -> pd.DataFrame:
    ref = reference.copy()
    if wcg_only:
        ref = ref[ref["IsWomanOrChild"] == 1].copy()
    groups = ref.groupby(group_col, sort=False)
    any_survive = []
    mean_survive = []
    smooth_survive = []
    peer_count = []
    for _, row in apply_df.iterrows():
        key = row[group_col]
        if key not in groups.groups:
            peers = ref.iloc[0:0]
        else:
            peers = ref.loc[groups.groups[key]]
            if exclude_self:
                peers = peers[peers["PassengerId"] != row["PassengerId"]]
        n = len(peers)
        peer_count.append(float(n))
        if n == 0:
            any_survive.append(0.5)
            mean_survive.append(0.5)
            smooth_survive.append(0.5)
        else:
            s = float(peers["Survived"].sum())
            m = float(peers["Survived"].mean())
            any_survive.append(float(s > 0))
            mean_survive.append(m)
            smooth_survive.append((s + 1.0) / (n + 2.0))
    return pd.DataFrame(
        {
            "Any": any_survive,
            "Mean": mean_survive,
            "Smooth": smooth_survive,
            "Count": peer_count,
        },
        index=apply_df.index,
    )


VARIANTS = {
    "baseline": [],
    "familyfare_all": [("FamilyFareGroup_v6", False, "FamilyFareAll")],
    "familyfare_wcg": [("FamilyFareGroup_v6", True, "FamilyFareWCG")],
    "ticket_all": [("Ticket", False, "TicketAll")],
    "ticket_wcg_any": [("Ticket", True, "TicketWCGAny")],
    "surname_all": [("SurnameGroup_v6", False, "SurnameAll")],
    "cabin_all": [("CabinGroup_v6", False, "CabinAll")],
    "familyfare_ticket_all": [
        ("FamilyFareGroup_v6", False, "FamilyFareAll"),
        ("Ticket", False, "TicketAll"),
    ],
    "familyfare_ticket_wcg": [
        ("FamilyFareGroup_v6", True, "FamilyFareWCG"),
        ("Ticket", True, "TicketWCGAny"),
    ],
}


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    train_base, test_base = build_base_frame(train_raw, test_raw, profile="v2")
    tr_help, te_help = helper_keys(train_raw, test_raw)
    train_base = add_helpers(train_base, tr_help)
    test_base = add_helpers(test_base, te_help)
    base_cols = model_feature_columns(
        train_base.drop(columns=["FamilyFareGroup_v6", "SurnameGroup_v6", "CabinGroup_v6"])
    )

    manifest = pd.read_csv(V1_AUDIT / "fold_manifest_seed42.csv")
    folds = manifest["fold"].astype(int).to_numpy()
    y = train_base["Survived"].astype(int).reset_index(drop=True)
    splits = [
        (fold, np.flatnonzero(folds != fold), np.flatnonzero(folds == fold))
        for fold in sorted(np.unique(folds))
    ]

    rows = []
    oof_export = pd.DataFrame({
        "PassengerId": train_base["PassengerId"],
        "fold": folds,
        "Survived": y,
    })
    test_export = pd.DataFrame({"PassengerId": test_base["PassengerId"]})

    for variant, specs in VARIANTS.items():
        print(f"\n=== group-key variant: {variant} ===", flush=True)
        oof = {m: np.zeros(len(train_base), dtype=float) for m in MODEL_NAMES}
        test_probs = {m: [] for m in MODEL_NAMES}
        extra_names: list[str] = []

        for fold, tr_idx, va_idx in splits:
            tr = train_base.iloc[tr_idx].copy()
            va = train_base.iloc[va_idx].copy()
            te = test_base.copy()
            tr["GroupSurvival"] = add_group_survival(tr, tr, exclude_self=True).to_numpy()
            va["GroupSurvival"] = add_group_survival(tr, va, exclude_self=False).to_numpy()
            te["GroupSurvival"] = add_group_survival(tr, te, exclude_self=False).to_numpy()

            fold_extra = []
            for group_col, wcg_only, prefix in specs:
                tr_stats = peer_feature(
                    tr, tr, group_col=group_col, wcg_only=wcg_only, exclude_self=True
                )
                va_stats = peer_feature(
                    tr, va, group_col=group_col, wcg_only=wcg_only, exclude_self=False
                )
                te_stats = peer_feature(
                    tr, te, group_col=group_col, wcg_only=wcg_only, exclude_self=False
                )
                for stat in ("Any", "Mean", "Smooth", "Count"):
                    col = f"{prefix}_{stat}_v6"
                    tr[col] = tr_stats[stat].to_numpy()
                    va[col] = va_stats[stat].to_numpy()
                    te[col] = te_stats[stat].to_numpy()
                    fold_extra.append(col)
            extra_names = fold_extra
            features = base_cols + fold_extra

            for model_name in MODEL_NAMES:
                model = build_models(42 + int(fold))[model_name]
                model.fit(tr[features], y.iloc[tr_idx])
                va_prob = np.asarray(model.predict_proba(va[features]))[:, 1]
                te_prob = np.asarray(model.predict_proba(te[features]))[:, 1]
                oof[model_name][va_idx] = va_prob
                test_probs[model_name].append(te_prob)

        panel = np.mean([oof[m] for m in MODEL_NAMES], axis=0)
        panel_test = np.mean(
            [np.mean(test_probs[m], axis=0) for m in MODEL_NAMES], axis=0
        )
        oof_export[f"{variant}__Panel"] = panel
        test_export[f"{variant}__Panel"] = panel_test
        rows.append({
            "variant": variant,
            "model": "Panel",
            "extra_features": len(extra_names),
            "accuracy": accuracy_score(y, panel > 0.5),
            "roc_auc": roc_auc_score(y, panel),
        })
        for model_name in MODEL_NAMES:
            prob = oof[model_name]
            tprob = np.mean(test_probs[model_name], axis=0)
            oof_export[f"{variant}__{model_name}"] = prob
            test_export[f"{variant}__{model_name}"] = tprob
            rows.append({
                "variant": variant,
                "model": model_name,
                "extra_features": len(extra_names),
                "accuracy": accuracy_score(y, prob > 0.5),
                "roc_auc": roc_auc_score(y, prob),
            })

    result = pd.DataFrame(rows)
    baseline = result[result["variant"] == "baseline"][
        ["model", "accuracy", "roc_auc"]
    ].rename(columns={"accuracy": "baseline_accuracy", "roc_auc": "baseline_roc_auc"})
    result = result.merge(baseline, on="model", how="left")
    result["delta_accuracy"] = result["accuracy"] - result["baseline_accuracy"]
    result["delta_auc"] = result["roc_auc"] - result["baseline_roc_auc"]
    result.to_csv(EXPORT_DIR / "group_key_audit_summary.csv", index=False)
    oof_export.to_csv(EXPORT_DIR / "group_key_audit_oof.csv", index=False)
    test_export.to_csv(EXPORT_DIR / "group_key_audit_test.csv", index=False)

    print("\n=== group-key panel ranking ===")
    print(
        result[result["model"] == "Panel"]
        .sort_values(["accuracy", "roc_auc"], ascending=False)
        .to_string(index=False, float_format=lambda x: f"{x:.5f}")
    )
    print("\n=== best rows overall ===")
    print(
        result.sort_values(["accuracy", "roc_auc"], ascending=False)
        .head(20)
        .to_string(index=False, float_format=lambda x: f"{x:.5f}")
    )


if __name__ == "__main__":
    main()
