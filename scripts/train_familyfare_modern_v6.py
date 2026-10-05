"""Test fold-safe LastName+Fare peer-survival features in RuleFit and MLP-PLR.

No Kaggle submission is performed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

try:
    from audit_group_survival import add_group_survival, build_base_frame, model_feature_columns
    from feature_ablation_v6 import DATA_DIR, V1_AUDIT, EXPORT_DIR, build_models
    from group_key_audit_v6 import helper_keys, add_helpers, peer_feature
except ModuleNotFoundError:
    from scripts.audit_group_survival import add_group_survival, build_base_frame, model_feature_columns
    from scripts.feature_ablation_v6 import DATA_DIR, V1_AUDIT, EXPORT_DIR, build_models
    from scripts.group_key_audit_v6 import helper_keys, add_helpers, peer_feature

from pytabkit.models.sklearn.sklearn_interfaces import MLP_PLR_D_Classifier


def make_mlp(seed: int):
    return MLP_PLR_D_Classifier(
        device="cuda",
        random_state=int(seed),
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

    variants = {
        "baseline": False,
        "familyfare_all": True,
    }
    model_names = ["RuleFit", "MLP_PLR"]
    rows = []
    out_oof = pd.DataFrame({
        "PassengerId": train_base["PassengerId"],
        "fold": folds,
        "Survived": y,
    })
    out_test = pd.DataFrame({"PassengerId": test_base["PassengerId"]})

    for variant, use_familyfare in variants.items():
        for model_name in model_names:
            print(f"\n=== {variant} / {model_name} ===", flush=True)
            oof = np.zeros(len(train_base))
            test_probs = []
            fold_acc = []
            fold_auc = []
            for fold, tr_idx, va_idx in splits:
                tr = train_base.iloc[tr_idx].copy()
                va = train_base.iloc[va_idx].copy()
                te = test_base.copy()
                tr["GroupSurvival"] = add_group_survival(tr, tr, exclude_self=True).to_numpy()
                va["GroupSurvival"] = add_group_survival(tr, va, exclude_self=False).to_numpy()
                te["GroupSurvival"] = add_group_survival(tr, te, exclude_self=False).to_numpy()

                extras = []
                if use_familyfare:
                    tr_stats = peer_feature(
                        tr, tr, group_col="FamilyFareGroup_v6",
                        wcg_only=False, exclude_self=True,
                    )
                    va_stats = peer_feature(
                        tr, va, group_col="FamilyFareGroup_v6",
                        wcg_only=False, exclude_self=False,
                    )
                    te_stats = peer_feature(
                        tr, te, group_col="FamilyFareGroup_v6",
                        wcg_only=False, exclude_self=False,
                    )
                    for stat in ("Any", "Mean", "Smooth", "Count"):
                        col = f"FamilyFareAll_{stat}_v6"
                        tr[col] = tr_stats[stat].to_numpy()
                        va[col] = va_stats[stat].to_numpy()
                        te[col] = te_stats[stat].to_numpy()
                        extras.append(col)

                features = base_cols + extras
                if model_name == "RuleFit":
                    model = build_models(42 + int(fold))["RuleFit"]
                else:
                    model = make_mlp(42 + int(fold))
                model.fit(tr[features], y.iloc[tr_idx])
                vp = np.asarray(model.predict_proba(va[features]))[:, 1]
                tp = np.asarray(model.predict_proba(te[features]))[:, 1]
                oof[va_idx] = vp
                test_probs.append(tp)
                fold_acc.append(accuracy_score(y.iloc[va_idx], vp > 0.5))
                fold_auc.append(roc_auc_score(y.iloc[va_idx], vp))

            test_prob = np.mean(test_probs, axis=0)
            col = f"{variant}__{model_name}"
            out_oof[col] = oof
            out_test[col] = test_prob
            rows.append({
                "variant": variant,
                "model": model_name,
                "accuracy": accuracy_score(y, oof > 0.5),
                "roc_auc": roc_auc_score(y, oof),
                "fold_accuracy_std": float(np.std(fold_acc)),
                "fold_auc_std": float(np.std(fold_auc)),
            })

    result = pd.DataFrame(rows)
    base = result[result["variant"] == "baseline"][
        ["model", "accuracy", "roc_auc"]
    ].rename(columns={"accuracy":"baseline_accuracy","roc_auc":"baseline_roc_auc"})
    result = result.merge(base, on="model", how="left")
    result["delta_accuracy"] = result["accuracy"] - result["baseline_accuracy"]
    result["delta_auc"] = result["roc_auc"] - result["baseline_roc_auc"]
    result.to_csv(EXPORT_DIR / "familyfare_modern_summary.csv", index=False)
    out_oof.to_csv(EXPORT_DIR / "familyfare_modern_oof.csv", index=False)
    out_test.to_csv(EXPORT_DIR / "familyfare_modern_test.csv", index=False)
    print("\n=== familyfare modern models ===")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
