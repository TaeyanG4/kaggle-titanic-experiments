"""Targeted Titanic v6 feature-group screen.

This follows the broad v6 block ablation. Instead of keeping a large feature
block that may contain both useful and noisy columns, each logical feature
group is added to the same v2 + fold-safe WCG baseline and evaluated on the
trusted five folds with three stable tree models.

No Kaggle submission is performed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

try:
    from audit_group_survival import add_group_survival, build_base_frame, model_feature_columns
    from feature_ablation_v6 import (
        DATA_DIR,
        V1_AUDIT,
        EXPORT_DIR,
        build_models,
        build_target_independent_blocks,
        peer_target_features,
    )
except ModuleNotFoundError:
    from scripts.audit_group_survival import add_group_survival, build_base_frame, model_feature_columns
    from scripts.feature_ablation_v6 import (
        DATA_DIR,
        V1_AUDIT,
        EXPORT_DIR,
        build_models,
        build_target_independent_blocks,
        peer_target_features,
    )


MODEL_NAMES = ["GradientBoosting", "XGBoost", "CatBoost"]


def column_groups(blocks):
    cabin_cols = [c for c in blocks["cabin_missing"][0].columns if c != "PassengerId"]
    rel_cols = [c for c in blocks["relational"][0].columns if c != "PassengerId"]
    int_cols = [c for c in blocks["interactions"][0].columns if c != "PassengerId"]
    groups = {
        "AgeMissing": ["AgeMissing_v6"],
        "CabinKnown": ["CabinKnown_v6"],
        "CabinCount": ["CabinCount_v6"],
        "CabinNumber": ["CabinNumber_v6"],
        "NameLength": ["NameLength_v6"],
        "CabinCore": ["CabinKnown_v6", "CabinCount_v6", "CabinNumber_v6"],
        "SurnameFreq": ["SurnameFreq_v6"],
        "FamilyGroupFreq": ["FamilyGroupFreq_v6"],
        "SharedGroupFlags": ["SharedFamily_v6", "SharedTicket_v6"],
        "TicketComposition": [c for c in rel_cols if c.startswith("Ticket") and "Count" in c],
        "FamilyComposition": [c for c in rel_cols if c.startswith("Family") and "Count" in c],
        "TravelGroup": [c for c in rel_cols if c.startswith("Travel")],
        "IsMother": ["IsMother_v6"],
        "AdultMale": ["AdultMale_v6"],
        "YoungChild": ["YoungChild_v6"],
        "FamilyClass": ["FamilyClass_v6"],
        "AgeFamilyRatio": ["AgeFamilyRatio_v6"],
        "FarePerFamily": ["FarePerFamily_v6"],
        "SexPclass": [c for c in int_cols if c.startswith("SexPclass_v6_")],
        "TitlePclass": [c for c in int_cols if c.startswith("TitlePclass_v6_")],
        "AgeBand": [c for c in int_cols if c.startswith("AgeBand_v6_")],
        "FareBand": [c for c in int_cols if c.startswith("FareBand_v6_")],
    }
    # Sanity: all named columns must exist in their source blocks.
    available = set(cabin_cols + rel_cols + int_cols)
    for name, cols in groups.items():
        missing = [c for c in cols if c not in available]
        if missing:
            raise KeyError(f"{name} missing columns: {missing}")
    return groups


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    train_base, test_base = build_base_frame(train_raw, test_raw, profile="v2")
    base_cols = model_feature_columns(train_base)
    blocks = build_target_independent_blocks(train_raw, test_raw)
    groups = column_groups(blocks)

    # Build one wide target-independent table so each group can be selected cheaply.
    train_extra = pd.DataFrame({"PassengerId": train_base["PassengerId"]})
    test_extra = pd.DataFrame({"PassengerId": test_base["PassengerId"]})
    for block in ("cabin_missing", "relational", "interactions"):
        btr, bte = blocks[block]
        cols = [c for c in btr.columns if c != "PassengerId"]
        train_extra = train_extra.merge(btr[["PassengerId"] + cols], on="PassengerId", how="left")
        test_extra = test_extra.merge(bte[["PassengerId"] + cols], on="PassengerId", how="left")

    manifest = pd.read_csv(V1_AUDIT / "fold_manifest_seed42.csv")
    folds = manifest["fold"].astype(int).to_numpy()
    y = train_base["Survived"].astype(int).reset_index(drop=True)
    splits = [
        (fold, np.flatnonzero(folds != fold), np.flatnonzero(folds == fold))
        for fold in sorted(np.unique(folds))
    ]

    group_defs = {"baseline": []}
    group_defs.update(groups)
    group_defs.update({
        "TicketWCGDetail": [
            "TicketWCGMean_v6", "TicketWCGSmooth_v6", "TicketWCGPeerCount_v6",
            "TicketWCGHasPeers_v6", "TicketWCGUnanimous_v6",
        ],
        "FamilyWCGDetail": [
            "FamilyWCGMean_v6", "FamilyWCGSmooth_v6", "FamilyWCGPeerCount_v6",
            "FamilyWCGHasPeers_v6", "FamilyWCGUnanimous_v6",
        ],
        "BothWCGDetail": [
            "TicketWCGMean_v6", "TicketWCGSmooth_v6", "TicketWCGPeerCount_v6",
            "TicketWCGHasPeers_v6", "TicketWCGUnanimous_v6",
            "FamilyWCGMean_v6", "FamilyWCGSmooth_v6", "FamilyWCGPeerCount_v6",
            "FamilyWCGHasPeers_v6", "FamilyWCGUnanimous_v6",
        ],
    })

    rows = []
    panel_oof_by_group = {}
    model_oof_by_group = {}

    for group_name, extra_cols in group_defs.items():
        print(f"\n=== targeted group: {group_name} ({len(extra_cols)} cols) ===", flush=True)
        oof = {m: np.zeros(len(train_base), dtype=float) for m in MODEL_NAMES}
        for fold, tr_idx, va_idx in splits:
            tr = train_base.iloc[tr_idx].copy()
            va = train_base.iloc[va_idx].copy()
            tr["GroupSurvival"] = add_group_survival(tr, tr, exclude_self=True).to_numpy()
            va["GroupSurvival"] = add_group_survival(tr, va, exclude_self=False).to_numpy()

            if any(c.endswith("WCGDetail") for c in []):  # pragma: no cover
                pass

            wcg_cols = [c for c in extra_cols if "WCG" in c]
            structural_cols = [c for c in extra_cols if c not in wcg_cols]
            if structural_cols:
                tr = tr.merge(
                    train_extra.iloc[tr_idx][["PassengerId"] + structural_cols],
                    on="PassengerId", how="left",
                )
                va = va.merge(
                    train_extra.iloc[va_idx][["PassengerId"] + structural_cols],
                    on="PassengerId", how="left",
                )
            if wcg_cols:
                tr_w = peer_target_features(tr, tr, exclude_self=True)
                va_w = peer_target_features(tr, va, exclude_self=False)
                for col in wcg_cols:
                    tr[col] = tr_w[col].to_numpy()
                    va[col] = va_w[col].to_numpy()

            features = base_cols + extra_cols
            for model_name in MODEL_NAMES:
                model = build_models(42 + fold)[model_name]
                model.fit(tr[features], y.iloc[tr_idx])
                prob = np.asarray(model.predict_proba(va[features]))[:, 1]
                oof[model_name][va_idx] = prob

        model_oof_by_group[group_name] = oof
        panel = np.mean([oof[m] for m in MODEL_NAMES], axis=0)
        panel_oof_by_group[group_name] = panel
        for model_name in MODEL_NAMES:
            rows.append({
                "feature_group": group_name,
                "model": model_name,
                "n_extra_features": len(extra_cols),
                "accuracy": accuracy_score(y, oof[model_name] > 0.5),
                "roc_auc": roc_auc_score(y, oof[model_name]),
            })
        rows.append({
            "feature_group": group_name,
            "model": "Panel",
            "n_extra_features": len(extra_cols),
            "accuracy": accuracy_score(y, panel > 0.5),
            "roc_auc": roc_auc_score(y, panel),
        })

    result = pd.DataFrame(rows)
    baseline = result[result["feature_group"] == "baseline"][
        ["model", "accuracy", "roc_auc"]
    ].rename(columns={"accuracy": "baseline_accuracy", "roc_auc": "baseline_roc_auc"})
    result = result.merge(baseline, on="model", how="left")
    result["delta_accuracy"] = result["accuracy"] - result["baseline_accuracy"]
    result["delta_auc"] = result["roc_auc"] - result["baseline_roc_auc"]
    result.to_csv(EXPORT_DIR / "targeted_feature_screen.csv", index=False)

    panel_result = result[result["model"] == "Panel"].sort_values(
        ["accuracy", "roc_auc"], ascending=False
    )
    panel_result.to_csv(EXPORT_DIR / "targeted_feature_panel_rank.csv", index=False)

    export_oof = pd.DataFrame({
        "PassengerId": train_base["PassengerId"],
        "fold": folds,
        "Survived": y,
    })
    for group_name, prob in panel_oof_by_group.items():
        export_oof[group_name] = prob
    export_oof.to_csv(EXPORT_DIR / "targeted_feature_panel_oof.csv", index=False)

    print("\n=== targeted feature groups: panel ranking ===")
    print(panel_result.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\nSaved: {EXPORT_DIR / 'targeted_feature_screen.csv'}")


if __name__ == "__main__":
    main()
