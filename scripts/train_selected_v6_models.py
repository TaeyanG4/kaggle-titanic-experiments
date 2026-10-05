"""Train selected v6 feature-engineered models on the trusted five folds.

Feature choices are predeclared from the v6 one-group screen:

- CatBoost: AdultMale + FarePerFamily + TicketWCGDetail
- XGBoost: NameLength + FarePerFamily + TicketWCGDetail
- GradientBoosting: AgeMissing + FarePerFamily + FamilyWCGDetail
- RuleFit: FarePerFamily + AdultMale + IsMother + AgeMissing
- MLP_PLR: FarePerFamily + AdultMale + IsMother + AgeMissing + CabinCount

The point is not to maximize a giant feature set, but to use the small subset
that showed evidence for each model family.

No Kaggle submission is performed.
"""

from __future__ import annotations

import json
import time
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

from pytabkit.models.sklearn.sklearn_interfaces import MLP_PLR_D_Classifier


TICKET_WCG = [
    "TicketWCGMean_v6",
    "TicketWCGSmooth_v6",
    "TicketWCGPeerCount_v6",
    "TicketWCGHasPeers_v6",
    "TicketWCGUnanimous_v6",
]
FAMILY_WCG = [
    "FamilyWCGMean_v6",
    "FamilyWCGSmooth_v6",
    "FamilyWCGPeerCount_v6",
    "FamilyWCGHasPeers_v6",
    "FamilyWCGUnanimous_v6",
]

MODEL_SPECS = {
    "CatBoost_v6": {
        "base_model": "CatBoost",
        "structural": ["AdultMale_v6", "FarePerFamily_v6"],
        "wcg": TICKET_WCG,
    },
    "XGBoost_v6": {
        "base_model": "XGBoost",
        "structural": ["NameLength_v6", "FarePerFamily_v6"],
        "wcg": TICKET_WCG,
    },
    "GradientBoosting_v6": {
        "base_model": "GradientBoosting",
        "structural": ["AgeMissing_v6", "FarePerFamily_v6"],
        "wcg": FAMILY_WCG,
    },
    "RuleFit_v6": {
        "base_model": "RuleFit",
        "structural": ["FarePerFamily_v6", "AdultMale_v6", "IsMother_v6", "AgeMissing_v6"],
        "wcg": [],
    },
    "MLP_PLR_v6": {
        "base_model": "MLP_PLR",
        "structural": [
            "FarePerFamily_v6",
            "AdultMale_v6",
            "IsMother_v6",
            "AgeMissing_v6",
            "CabinCount_v6",
        ],
        "wcg": [],
    },
}


def make_mlp(seed: int):
    return MLP_PLR_D_Classifier(
        device="cuda",
        random_state=seed,
        n_cv=1,
        n_refit=0,
        n_repeats=1,
        val_fraction=0.18,
        verbosity=0,
        max_epochs=100,
        es_patience=15,
        batch_size=128,
        use_checkpoints=False,
    )


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    train_base, test_base = build_base_frame(train_raw, test_raw, profile="v2")
    base_cols = model_feature_columns(train_base)
    blocks = build_target_independent_blocks(train_raw, test_raw)

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

    out_oof = pd.DataFrame({
        "PassengerId": train_base["PassengerId"],
        "fold": folds,
        "Survived": y,
    })
    out_test = pd.DataFrame({"PassengerId": test_base["PassengerId"]})
    fold_rows = []
    summary_rows = []

    for model_name, spec in MODEL_SPECS.items():
        print(f"\n=== {model_name} ===", flush=True)
        oof = np.zeros(len(train_base), dtype=float)
        test_probs = []
        started_model = time.time()

        for fold, tr_idx, va_idx in splits:
            tr = train_base.iloc[tr_idx].copy()
            va = train_base.iloc[va_idx].copy()
            te = test_base.copy()
            tr["GroupSurvival"] = add_group_survival(tr, tr, exclude_self=True).to_numpy()
            va["GroupSurvival"] = add_group_survival(tr, va, exclude_self=False).to_numpy()
            te["GroupSurvival"] = add_group_survival(tr, te, exclude_self=False).to_numpy()

            structural = spec["structural"]
            if structural:
                tr = tr.merge(
                    train_extra.iloc[tr_idx][["PassengerId"] + structural],
                    on="PassengerId", how="left",
                )
                va = va.merge(
                    train_extra.iloc[va_idx][["PassengerId"] + structural],
                    on="PassengerId", how="left",
                )
                te = te.merge(
                    test_extra[["PassengerId"] + structural],
                    on="PassengerId", how="left",
                )

            if spec["wcg"]:
                tr_w = peer_target_features(tr, tr, exclude_self=True)
                va_w = peer_target_features(tr, va, exclude_self=False)
                te_w = peer_target_features(tr, te, exclude_self=False)
                for col in spec["wcg"]:
                    tr[col] = tr_w[col].to_numpy()
                    va[col] = va_w[col].to_numpy()
                    te[col] = te_w[col].to_numpy()

            features = base_cols + structural + spec["wcg"]
            started = time.time()
            if spec["base_model"] == "MLP_PLR":
                model = make_mlp(42 + int(fold))
            else:
                model = build_models(42 + int(fold))[spec["base_model"]]
            model.fit(tr[features], y.iloc[tr_idx])
            va_prob = np.asarray(model.predict_proba(va[features]))[:, 1]
            te_prob = np.asarray(model.predict_proba(te[features]))[:, 1]
            oof[va_idx] = va_prob
            test_probs.append(te_prob)
            fold_rows.append({
                "model": model_name,
                "fold": int(fold),
                "accuracy": accuracy_score(y.iloc[va_idx], va_prob > 0.5),
                "roc_auc": roc_auc_score(y.iloc[va_idx], va_prob),
                "seconds": time.time() - started,
                "feature_count": len(features),
            })
            print(
                f"fold {fold}: acc={fold_rows[-1]['accuracy']:.5f} "
                f"auc={fold_rows[-1]['roc_auc']:.5f}",
                flush=True,
            )

        test_prob = np.mean(test_probs, axis=0)
        out_oof[model_name] = oof
        out_test[model_name] = test_prob
        model_folds = pd.DataFrame([r for r in fold_rows if r["model"] == model_name])
        summary_rows.append({
            "model": model_name,
            "accuracy": accuracy_score(y, oof > 0.5),
            "roc_auc": roc_auc_score(y, oof),
            "fold_accuracy_mean": model_folds["accuracy"].mean(),
            "fold_accuracy_std": model_folds["accuracy"].std(ddof=0),
            "fold_auc_mean": model_folds["roc_auc"].mean(),
            "fold_auc_std": model_folds["roc_auc"].std(ddof=0),
            "elapsed_seconds": time.time() - started_model,
            "feature_count": int(model_folds["feature_count"].iloc[0]),
        })

    summary = pd.DataFrame(summary_rows).sort_values(
        ["accuracy", "roc_auc"], ascending=False
    )
    pd.DataFrame(fold_rows).to_csv(EXPORT_DIR / "selected_models_fold_metrics.csv", index=False)
    summary.to_csv(EXPORT_DIR / "selected_models_summary.csv", index=False)
    out_oof.to_csv(EXPORT_DIR / "selected_models_oof.csv", index=False)
    out_test.to_csv(EXPORT_DIR / "selected_models_test.csv", index=False)
    with (EXPORT_DIR / "selected_models_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(MODEL_SPECS, f, ensure_ascii=False, indent=2)

    print("\n=== v6 selected feature models ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
