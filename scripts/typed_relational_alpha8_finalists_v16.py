"""Confirm alpha=8 typed22 finalists on trusted fixed folds.

Finalists:
  - TabPFN v2.5
  - TabPFN v3
  - CatBoost

Representation:
  v2 base + FamilyFare + typed22(alpha=8)

Validation-aware and actual-test-aware transductive relation matrices are fit
with separate models per fold. No Kaggle submission is performed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from audit_group_survival import add_group_survival
from feature_ablation_v6 import build_models
from gunes_exact_foldsafe_v11 import structural_frames
from pseudo_test_relational_v14 import (
    eligible_groups,
    relation_features,
    role_flags,
)
from tabpfn_finalist_v8 import load_champion_context, make_tabpfn, prepare_data
from tabpfn_typed_relational_v14 import (
    FAMILYFARE_COLS,
    TYPED_COLS,
    add_familyfare,
    add_rel_columns,
)


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "exports" / "v16"
ALPHA = 8.0
FINALISTS = ["TabPFN_v2_5", "TabPFN_v3", "CatBoost"]


def build_model(name: str, seed: int):
    if name.startswith("TabPFN"):
        return make_tabpfn(name, seed=seed)
    return build_models(seed)["CatBoost"]


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    train, test, folds, base_cols = prepare_data()
    rel_train, rel_test, _, _, _ = structural_frames(train_raw, test_raw)
    rel_train["IsWomanChild"] = role_flags(train_raw)
    rel_test["IsWomanChild"] = role_flags(test_raw)

    y = train["Survived"].astype(int).to_numpy()
    zoo, _, champ, _, champ_test = load_champion_context()
    champ_correct = champ == y
    features = list(dict.fromkeys(base_cols + FAMILYFARE_COLS + TYPED_COLS))

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

    for name in FINALISTS:
        print(f"\n=== {name} alpha8 ===", flush=True)
        oof = np.zeros(len(train), dtype=float)
        test_probs = []

        for fold in sorted(np.unique(folds)):
            tr_idx = np.flatnonzero(folds != fold)
            va_idx = np.flatnonzero(folds == fold)
            tr0 = train.iloc[tr_idx].copy()
            va0 = train.iloc[va_idx].copy()
            te0 = test.copy()
            rr = rel_train.iloc[tr_idx].copy()
            rv = rel_train.iloc[va_idx].copy()

            gs_tr = add_group_survival(
                tr0, tr0, exclude_self=True
            ).to_numpy()
            gs_va = add_group_survival(
                tr0, va0, exclude_self=False
            ).to_numpy()
            gs_te = add_group_survival(
                tr0, te0, exclude_self=False
            ).to_numpy()
            tr0["GroupSurvival"] = gs_tr
            va0["GroupSurvival"] = gs_va
            te0["GroupSurvival"] = gs_te

            tr_ff = add_familyfare(tr0, tr0, exclude_self=True)
            va_ff = add_familyfare(
                train.iloc[tr_idx].assign(GroupSurvival=gs_tr),
                va0,
                exclude_self=False,
            )
            te_ff = add_familyfare(
                train.iloc[tr_idx].assign(GroupSurvival=gs_tr),
                te0,
                exclude_self=False,
            )

            # validation-aware relation activation
            afv, atv = eligible_groups(rr, rv)
            tr_rel_v = relation_features(
                rr,
                rr,
                exclude_self=True,
                alpha=ALPHA,
                allowed_fam=afv,
                allowed_tic=atv,
            )
            va_rel = relation_features(
                rr,
                rv,
                exclude_self=False,
                alpha=ALPHA,
                allowed_fam=afv,
                allowed_tic=atv,
            )
            tr_val = add_rel_columns(tr_ff, tr_rel_v)
            va = add_rel_columns(va_ff, va_rel)

            val_model = build_model(name, 42 + int(fold))
            val_model.fit(tr_val[features], y[tr_idx])
            va_prob = np.asarray(
                val_model.predict_proba(va[features])
            )[:, 1].astype(float)
            oof[va_idx] = va_prob

            # actual-test-aware relation activation
            aft, att = eligible_groups(rr, rel_test)
            tr_rel_t = relation_features(
                rr,
                rr,
                exclude_self=True,
                alpha=ALPHA,
                allowed_fam=aft,
                allowed_tic=att,
            )
            te_rel = relation_features(
                rr,
                rel_test,
                exclude_self=False,
                alpha=ALPHA,
                allowed_fam=aft,
                allowed_tic=att,
            )
            tr_test = add_rel_columns(tr_ff, tr_rel_t)
            te = add_rel_columns(te_ff, te_rel)
            test_model = build_model(name, 1042 + int(fold))
            test_model.fit(tr_test[features], y[tr_idx])
            test_probs.append(
                np.asarray(
                    test_model.predict_proba(te[features])
                )[:, 1].astype(float)
            )

            fold_rows.append(
                {
                    "model": name,
                    "fold": int(fold),
                    "accuracy": accuracy_score(y[va_idx], va_prob > 0.5),
                    "roc_auc": roc_auc_score(y[va_idx], va_prob),
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
        fdf = pd.DataFrame([x for x in fold_rows if x["model"] == name])
        summary_rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y, pred),
                "roc_auc": roc_auc_score(y, oof),
                "fold_accuracy_std": fdf["accuracy"].std(ddof=0),
                "fold_auc_std": fdf["roc_auc"].std(ddof=0),
                "test_positive_count": int((test_prob > 0.5).sum()),
                "feature_count": len(features),
            }
        )
        diversity_rows.append(
            {
                "model": name,
                "disagreement_vs_v5": int(np.sum(pred != champ)),
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
                    np.sum((test_prob > 0.5).astype(int) != champ_test)
                ),
            }
        )
        oof_export[name] = oof
        test_export[name] = test_prob

    summary = pd.DataFrame(summary_rows).sort_values(
        ["accuracy", "roc_auc"], ascending=False
    )
    diversity = pd.DataFrame(diversity_rows).sort_values(
        ["v5_wrong_candidate_right", "net_unique_correct_vs_v5"],
        ascending=False,
    )
    summary.to_csv(EXPORT_DIR / "alpha8_finalist_summary.csv", index=False)
    diversity.to_csv(EXPORT_DIR / "alpha8_finalist_diversity.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(
        EXPORT_DIR / "alpha8_finalist_fold_metrics.csv", index=False
    )
    oof_export.to_csv(EXPORT_DIR / "alpha8_finalist_oof.csv", index=False)
    test_export.to_csv(EXPORT_DIR / "alpha8_finalist_test.csv", index=False)

    print("\n=== alpha8 finalists ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n=== diversity vs v5 ===")
    print(diversity.to_string(index=False))
    print("\nNo Kaggle submission was performed.")


if __name__ == "__main__":
    main()
