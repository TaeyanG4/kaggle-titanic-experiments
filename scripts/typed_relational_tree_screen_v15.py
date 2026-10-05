"""Screen typed22 relational features across strong tree families.

Variants:
  - v2base_typed22
  - v2base_familyfare_typed22

Models:
  - GradientBoosting
  - XGBoost
  - CatBoost

All group-target features are fold-safe and validation-aware. Test inference
uses actual-test-aware eligibility with separately fitted fold models.

No Kaggle submission is performed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from audit_group_survival import add_group_survival
from feature_ablation_v6 import build_models
from group_key_audit_v6 import peer_feature
from gunes_exact_foldsafe_v11 import structural_frames
from pseudo_test_relational_v14 import (
    VARIANTS as REL_VARIANTS,
    eligible_groups,
    relation_features,
    role_flags,
)
from tabpfn_finalist_v8 import load_champion_context, prepare_data


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "exports" / "v15"

MODEL_NAMES = ["GradientBoosting", "XGBoost", "CatBoost"]
TYPED_COLS = REL_VARIANTS["typed22"]
FAMILYFARE_COLS = [
    "FamilyFareAny",
    "FamilyFareMean",
    "FamilyFareSmooth",
    "FamilyFareCount",
]
VARIANTS = ["v2base_typed22", "v2base_familyfare_typed22"]


def add_familyfare(ref, app, exclude_self):
    stats = peer_feature(
        ref,
        app,
        group_col="FamilyFareGroup_v6",
        wcg_only=False,
        exclude_self=exclude_self,
    )
    out = app.copy()
    for source, dest in zip(
        ["Any", "Mean", "Smooth", "Count"], FAMILYFARE_COLS
    ):
        out[dest] = stats[source].to_numpy()
    return out


def add_rel(df, rel):
    out = df.copy()
    for c in TYPED_COLS:
        out[c] = rel[c].to_numpy()
    return out


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    train, test, folds, base_cols = prepare_data()
    rel_train, rel_test, _, _, _ = structural_frames(train_raw, test_raw)
    rel_train["IsWomanChild"] = role_flags(train_raw)
    rel_test["IsWomanChild"] = role_flags(test_raw)

    if not train["PassengerId"].astype(int).equals(
        rel_train["PassengerId"].astype(int)
    ):
        raise ValueError("Train PassengerId mismatch.")

    y = train["Survived"].astype(int).to_numpy()
    zoo, _, champ, _, champ_test = load_champion_context()
    champ_correct = champ == y

    summary_rows = []
    diversity_rows = []
    fold_rows = []
    oof_export = pd.DataFrame(
        {
            "PassengerId": train["PassengerId"].astype(int),
            "fold": folds,
            "Survived": y,
        }
    )
    test_export = pd.DataFrame(
        {"PassengerId": test["PassengerId"].astype(int)}
    )

    for variant in VARIANTS:
        for model_name in MODEL_NAMES:
            oof = np.zeros(len(train), dtype=float)
            test_probs = []
            print(f"\n=== {variant} / {model_name} ===", flush=True)

            for fold in sorted(np.unique(folds)):
                tr_idx = np.flatnonzero(folds != fold)
                va_idx = np.flatnonzero(folds == fold)
                tr = train.iloc[tr_idx].copy()
                va = train.iloc[va_idx].copy()
                te = test.copy()
                rr = rel_train.iloc[tr_idx].copy()
                rv = rel_train.iloc[va_idx].copy()

                gs_tr = add_group_survival(
                    tr, tr, exclude_self=True
                ).to_numpy()
                gs_va = add_group_survival(
                    tr, va, exclude_self=False
                ).to_numpy()
                gs_te = add_group_survival(
                    tr, te, exclude_self=False
                ).to_numpy()
                tr["GroupSurvival"] = gs_tr
                va["GroupSurvival"] = gs_va
                te["GroupSurvival"] = gs_te

                if variant == "v2base_familyfare_typed22":
                    tr = add_familyfare(tr, tr, True)
                    va = add_familyfare(
                        train.iloc[tr_idx].assign(
                            GroupSurvival=gs_tr
                        ),
                        va,
                        False,
                    )
                    te = add_familyfare(
                        train.iloc[tr_idx].assign(
                            GroupSurvival=gs_tr
                        ),
                        te,
                        False,
                    )
                    extra = FAMILYFARE_COLS.copy()
                else:
                    extra = []

                # OOF validation-aware relation features.
                afv, atv = eligible_groups(rr, rv)
                rr_val = relation_features(
                    rr,
                    rr,
                    exclude_self=True,
                    alpha=2.0,
                    allowed_fam=afv,
                    allowed_tic=atv,
                )
                rv_rel = relation_features(
                    rr,
                    rv,
                    exclude_self=False,
                    alpha=2.0,
                    allowed_fam=afv,
                    allowed_tic=atv,
                )
                tr_val = add_rel(tr, rr_val)
                va_val = add_rel(va, rv_rel)

                features = list(
                    dict.fromkeys(base_cols + extra + TYPED_COLS)
                )
                model = build_models(42 + int(fold))[model_name]
                model.fit(tr_val[features], y[tr_idx])
                va_prob = model.predict_proba(
                    va_val[features]
                )[:, 1]
                oof[va_idx] = va_prob

                # Test-aware relation features require a separate fit because
                # train-side transductive activation differs.
                aft, att = eligible_groups(rr, rel_test)
                rr_test = relation_features(
                    rr,
                    rr,
                    exclude_self=True,
                    alpha=2.0,
                    allowed_fam=aft,
                    allowed_tic=att,
                )
                rt_rel = relation_features(
                    rr,
                    rel_test,
                    exclude_self=False,
                    alpha=2.0,
                    allowed_fam=aft,
                    allowed_tic=att,
                )
                tr_test = add_rel(tr, rr_test)
                te_test = add_rel(te, rt_rel)
                test_model = build_models(
                    1042 + int(fold)
                )[model_name]
                test_model.fit(tr_test[features], y[tr_idx])
                test_probs.append(
                    test_model.predict_proba(te_test[features])[:, 1]
                )

                fold_rows.append(
                    {
                        "variant": variant,
                        "model": model_name,
                        "fold": int(fold),
                        "accuracy": accuracy_score(
                            y[va_idx], va_prob > 0.5
                        ),
                        "roc_auc": roc_auc_score(
                            y[va_idx], va_prob
                        ),
                        "feature_count": len(features),
                    }
                )
                print(
                    f"fold={fold} "
                    f"acc={fold_rows[-1]['accuracy']:.5f} "
                    f"auc={fold_rows[-1]['roc_auc']:.5f}",
                    flush=True,
                )

            test_prob = np.mean(test_probs, axis=0)
            pred = (oof > 0.5).astype(int)
            correct = pred == y
            stem = f"{variant}__{model_name}"
            oof_export[stem] = oof
            test_export[stem] = test_prob
            vf = pd.DataFrame(
                [
                    x
                    for x in fold_rows
                    if x["variant"] == variant
                    and x["model"] == model_name
                ]
            )
            summary_rows.append(
                {
                    "variant": variant,
                    "model": model_name,
                    "accuracy": accuracy_score(y, pred),
                    "roc_auc": roc_auc_score(y, oof),
                    "fold_accuracy_std": vf["accuracy"].std(ddof=0),
                    "fold_auc_std": vf["roc_auc"].std(ddof=0),
                    "test_positive_count": int(
                        (test_prob > 0.5).sum()
                    ),
                    "feature_count": int(
                        vf["feature_count"].iloc[0]
                    ),
                }
            )
            diversity_rows.append(
                {
                    "variant": variant,
                    "model": model_name,
                    "disagreement_vs_v5": int(
                        np.sum(pred != champ)
                    ),
                    "v5_wrong_candidate_right": int(
                        np.sum((~champ_correct) & correct)
                    ),
                    "v5_right_candidate_wrong": int(
                        np.sum(champ_correct & (~correct))
                    ),
                    "net_unique_correct_vs_v5": int(
                        np.sum((~champ_correct) & correct)
                        - np.sum(champ_correct & (~correct))
                    ),
                    "test_disagreement_vs_v5": int(
                        np.sum(
                            (test_prob > 0.5).astype(int)
                            != champ_test
                        )
                    ),
                }
            )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["accuracy", "roc_auc"], ascending=False
    )
    diversity = pd.DataFrame(diversity_rows).sort_values(
        ["v5_wrong_candidate_right", "net_unique_correct_vs_v5"],
        ascending=False,
    )
    summary.to_csv(
        EXPORT_DIR / "typed_tree_screen_summary.csv", index=False
    )
    diversity.to_csv(
        EXPORT_DIR / "typed_tree_screen_diversity.csv", index=False
    )
    pd.DataFrame(fold_rows).to_csv(
        EXPORT_DIR / "typed_tree_screen_fold_metrics.csv", index=False
    )
    oof_export.to_csv(
        EXPORT_DIR / "typed_tree_screen_oof.csv", index=False
    )
    test_export.to_csv(
        EXPORT_DIR / "typed_tree_screen_test.csv", index=False
    )

    print("\n=== typed relational tree screen ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n=== diversity vs v5 ===")
    print(diversity.to_string(index=False))
    print("\nNo Kaggle submission was performed.")


if __name__ == "__main__":
    main()
