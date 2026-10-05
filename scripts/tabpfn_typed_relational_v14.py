"""TabPFN v2.5/v3 confirmation of the v14 typed relational feature block.

Representations
---------------
1. v2base_typed22
   Trusted v2 feature matrix + fold-safe GroupSurvival + typed22 relation block.

2. v2base_familyfare_typed22
   Above + fold-safe FamilyFare peer statistics.

3. gunes24_typed22
   Exact Gunes-style 24 target-independent features + typed22 relation block.

For transductive correctness, each outer fold fits:
  * one validation-aware model whose train relation features are activated only
    for groups also appearing in that fold's validation partition;
  * one test-aware model whose train relation features are activated only for
    groups also appearing in the actual Kaggle test.

No Kaggle submission is performed.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from audit_group_survival import add_group_survival
from group_key_audit_v6 import peer_feature
from gunes_exact_foldsafe_v11 import structural_frames
from pseudo_test_relational_v14 import (
    VARIANTS as REL_VARIANTS,
    eligible_groups,
    relation_features,
    role_flags,
)
from tabpfn_finalist_v8 import (
    load_champion_context,
    make_tabpfn,
    prepare_data,
)


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "exports" / "v14" / "tabpfn"
MODEL_DIR = EXPORT_DIR / "models"

MODELS = ["TabPFN_v2_5", "TabPFN_v3"]
TYPED_COLS = REL_VARIANTS["typed22"]
FAMILYFARE_COLS = [
    "FamilyFareAny",
    "FamilyFareMean",
    "FamilyFareSmooth",
    "FamilyFareCount",
]
VARIANTS = [
    "v2base_typed22",
    "v2base_familyfare_typed22",
    "gunes24_typed22",
]


def add_familyfare(
    tr: pd.DataFrame,
    app: pd.DataFrame,
    *,
    exclude_self: bool,
) -> pd.DataFrame:
    stats = peer_feature(
        tr,
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


def add_rel_columns(df: pd.DataFrame, rel: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in TYPED_COLS:
        out[c] = rel[c].to_numpy()
    return out


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")

    train_v2, test_v2, folds, v2_cols = prepare_data()
    train_g, test_g, gunes_cols, _, _ = structural_frames(train_raw, test_raw)
    train_g["IsWomanChild"] = role_flags(train_raw)
    test_g["IsWomanChild"] = role_flags(test_raw)

    if not train_v2["PassengerId"].astype(int).equals(
        train_g["PassengerId"].astype(int)
    ):
        raise ValueError("Train PassengerId mismatch across representations.")
    if not test_v2["PassengerId"].astype(int).equals(
        test_g["PassengerId"].astype(int)
    ):
        raise ValueError("Test PassengerId mismatch across representations.")

    y = train_v2["Survived"].astype(int).to_numpy()
    zoo, _, champion_pred, _, champion_test_pred = load_champion_context()
    champion_correct = champion_pred == y

    summary_rows = []
    diversity_rows = []
    fold_frames = []
    oof_export = pd.DataFrame(
        {
            "PassengerId": train_v2["PassengerId"].astype(int),
            "fold": folds,
            "Survived": y,
            "champion_pred": champion_pred,
        }
    )
    test_export = pd.DataFrame(
        {
            "PassengerId": test_v2["PassengerId"].astype(int),
            "champion_pred": champion_test_pred,
        }
    )
    failures = []

    for model_name in MODELS:
        for variant in VARIANTS:
            stem = f"{model_name}__{variant}"
            print(f"\n=== {stem} ===", flush=True)
            try:
                oof = np.zeros(len(train_v2), dtype=float)
                test_probs = []
                fold_rows = []

                for fold in sorted(np.unique(folds)):
                    tr_idx = np.flatnonzero(folds != fold)
                    va_idx = np.flatnonzero(folds == fold)

                    rel_tr = train_g.iloc[tr_idx].copy()
                    rel_va = train_g.iloc[va_idx].copy()

                    # ----- validation-aware typed relation block -----
                    afv, atv = eligible_groups(rel_tr, rel_va)
                    tr_rel_val = relation_features(
                        rel_tr,
                        rel_tr,
                        exclude_self=True,
                        alpha=2.0,
                        allowed_fam=afv,
                        allowed_tic=atv,
                    )
                    va_rel = relation_features(
                        rel_tr,
                        rel_va,
                        exclude_self=False,
                        alpha=2.0,
                        allowed_fam=afv,
                        allowed_tic=atv,
                    )

                    # ----- actual-test-aware typed relation block -----
                    aft, att = eligible_groups(rel_tr, test_g)
                    tr_rel_test = relation_features(
                        rel_tr,
                        rel_tr,
                        exclude_self=True,
                        alpha=2.0,
                        allowed_fam=aft,
                        allowed_tic=att,
                    )
                    te_rel = relation_features(
                        rel_tr,
                        test_g,
                        exclude_self=False,
                        alpha=2.0,
                        allowed_fam=aft,
                        allowed_tic=att,
                    )

                    if variant.startswith("v2base"):
                        tr_val = train_v2.iloc[tr_idx].copy()
                        va = train_v2.iloc[va_idx].copy()
                        tr_test = train_v2.iloc[tr_idx].copy()
                        te = test_v2.copy()

                        # Same fold-safe WCG signal for both fit contexts.
                        gs_tr = add_group_survival(
                            tr_val, tr_val, exclude_self=True
                        ).to_numpy()
                        gs_va = add_group_survival(
                            tr_val, va, exclude_self=False
                        ).to_numpy()
                        gs_te = add_group_survival(
                            tr_val, te, exclude_self=False
                        ).to_numpy()
                        tr_val["GroupSurvival"] = gs_tr
                        tr_test["GroupSurvival"] = gs_tr
                        va["GroupSurvival"] = gs_va
                        te["GroupSurvival"] = gs_te

                        extra_cols = []
                        if variant == "v2base_familyfare_typed22":
                            tr_val = add_familyfare(
                                tr_val, tr_val, exclude_self=True
                            )
                            tr_test = add_familyfare(
                                tr_test, tr_test, exclude_self=True
                            )
                            va = add_familyfare(
                                train_v2.iloc[tr_idx].assign(
                                    GroupSurvival=gs_tr
                                ),
                                va,
                                exclude_self=False,
                            )
                            te = add_familyfare(
                                train_v2.iloc[tr_idx].assign(
                                    GroupSurvival=gs_tr
                                ),
                                te,
                                exclude_self=False,
                            )
                            extra_cols.extend(FAMILYFARE_COLS)

                        tr_val = add_rel_columns(tr_val, tr_rel_val)
                        va = add_rel_columns(va, va_rel)
                        tr_test = add_rel_columns(tr_test, tr_rel_test)
                        te = add_rel_columns(te, te_rel)
                        features = list(
                            dict.fromkeys(v2_cols + extra_cols + TYPED_COLS)
                        )
                    else:
                        tr_val = add_rel_columns(
                            train_g.iloc[tr_idx].copy(), tr_rel_val
                        )
                        va = add_rel_columns(
                            train_g.iloc[va_idx].copy(), va_rel
                        )
                        tr_test = add_rel_columns(
                            train_g.iloc[tr_idx].copy(), tr_rel_test
                        )
                        te = add_rel_columns(test_g.copy(), te_rel)
                        features = gunes_cols + TYPED_COLS

                    # Separate models are intentional: validation-aware and
                    # actual-test-aware train feature matrices differ.
                    val_model = make_tabpfn(
                        model_name, seed=42 + int(fold)
                    )
                    val_model.fit(tr_val[features], y[tr_idx])
                    va_prob = np.asarray(
                        val_model.predict_proba(va[features])
                    )[:, 1].astype(float)
                    oof[va_idx] = va_prob

                    test_model = make_tabpfn(
                        model_name, seed=1042 + int(fold)
                    )
                    test_model.fit(tr_test[features], y[tr_idx])
                    te_prob = np.asarray(
                        test_model.predict_proba(te[features])
                    )[:, 1].astype(float)
                    test_probs.append(te_prob)

                    fold_rows.append(
                        {
                            "model": model_name,
                            "variant": variant,
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
                        f"fold {fold}: "
                        f"acc={fold_rows[-1]['accuracy']:.5f} "
                        f"auc={fold_rows[-1]['roc_auc']:.5f}",
                        flush=True,
                    )

                test_prob = np.mean(test_probs, axis=0)
                pred = (oof > 0.5).astype(int)
                correct = pred == y
                fold_df = pd.DataFrame(fold_rows)

                summary_rows.append(
                    {
                        "model": model_name,
                        "variant": variant,
                        "accuracy": accuracy_score(y, pred),
                        "roc_auc": roc_auc_score(y, oof),
                        "fold_accuracy_std": fold_df["accuracy"].std(ddof=0),
                        "fold_auc_std": fold_df["roc_auc"].std(ddof=0),
                        "feature_count": int(
                            fold_df["feature_count"].iloc[0]
                        ),
                        "test_positive_count": int(
                            (test_prob > 0.5).sum()
                        ),
                    }
                )
                diversity_rows.append(
                    {
                        "model": model_name,
                        "variant": variant,
                        "binary_disagreement_vs_v5": int(
                            np.sum(pred != champion_pred)
                        ),
                        "v5_wrong_candidate_right": int(
                            np.sum((~champion_correct) & correct)
                        ),
                        "v5_right_candidate_wrong": int(
                            np.sum(champion_correct & (~correct))
                        ),
                        "net_unique_correct_vs_v5": int(
                            np.sum((~champion_correct) & correct)
                            - np.sum(champion_correct & (~correct))
                        ),
                        "correlation_vs_v4b": float(
                            np.corrcoef(
                                oof,
                                zoo["v4b__Champion"].to_numpy(),
                            )[0, 1]
                        ),
                        "correlation_vs_rulefit": float(
                            np.corrcoef(
                                oof,
                                zoo["RuleFit"].to_numpy(),
                            )[0, 1]
                        ),
                    }
                )
                oof_export[stem] = oof
                test_export[stem] = test_prob
                fold_frames.append(fold_df)

                pd.DataFrame(
                    {
                        "PassengerId": train_v2["PassengerId"].astype(int),
                        "fold": folds,
                        "Survived": y,
                        "probability": oof,
                    }
                ).to_csv(MODEL_DIR / f"{stem}__oof.csv", index=False)
                pd.DataFrame(
                    {
                        "PassengerId": test_v2["PassengerId"].astype(int),
                        "probability": test_prob,
                    }
                ).to_csv(MODEL_DIR / f"{stem}__test.csv", index=False)
                fold_df.to_csv(
                    MODEL_DIR / f"{stem}__fold_metrics.csv", index=False
                )
            except Exception as exc:
                failures.append(
                    {
                        "model": model_name,
                        "variant": variant,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
                print(
                    f"[FAILED] {stem}: {type(exc).__name__}: {exc}",
                    flush=True,
                )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["accuracy", "roc_auc"], ascending=False
    )
    diversity = pd.DataFrame(diversity_rows).sort_values(
        ["v5_wrong_candidate_right", "net_unique_correct_vs_v5"],
        ascending=False,
    )
    summary.to_csv(EXPORT_DIR / "tabpfn_typed_summary.csv", index=False)
    diversity.to_csv(EXPORT_DIR / "tabpfn_typed_diversity.csv", index=False)
    oof_export.to_csv(EXPORT_DIR / "tabpfn_typed_oof.csv", index=False)
    test_export.to_csv(EXPORT_DIR / "tabpfn_typed_test.csv", index=False)
    if fold_frames:
        pd.concat(fold_frames, ignore_index=True).to_csv(
            EXPORT_DIR / "tabpfn_typed_fold_metrics.csv", index=False
        )
    pd.DataFrame(
        [
            {
                "model": x["model"],
                "variant": x["variant"],
                "error": x["error"],
            }
            for x in failures
        ]
    ).to_csv(EXPORT_DIR / "failures.csv", index=False)
    with (EXPORT_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "models": MODELS,
                "variants": VARIANTS,
                "typed_columns": TYPED_COLS,
                "alpha": 2.0,
                "validation": "trusted fixed folds, validation-aware relation eligibility",
                "test_inference": "actual-test-aware relation eligibility, separate fold model",
                "no_submission": True,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== TabPFN typed relational ranking ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n=== Diversity vs v5 robust ===")
    print(diversity.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    if failures:
        print(
            "\nFailures:",
            [(x["model"], x["variant"]) for x in failures],
        )
    print("\nNo Kaggle submission was performed.")


if __name__ == "__main__":
    main()
