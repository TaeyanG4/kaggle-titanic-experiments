"""TabPFN v2.5/v3 retraining on the strongest new Titanic feature blocks.

This script extends the already-working v8 TabPFN harness with feature blocks
that survived v6-v9 audits:

  structural_gunes
      Sex x Pclass alternate age, Age/Fare quantile bins, grouped Deck,
      coarse Title one-hot features.

  gunes_rates
      fold-safe surname/ticket peer survival-rate components.

  familyfare
      fold-safe LastName+Fare peer survival statistics.

Variants are deliberately block-level rather than arbitrary feature soup:
  - gunes_structural
  - gunes_rates
  - gunes_all
  - familyfare_gunes_rates
  - familyfare_gunes_all

All target-derived columns are computed inside each trusted fold. No Kaggle
submission is performed.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from tabpfn_finalist_v8 import make_tabpfn, prepare_data, load_champion_context
from group_key_audit_v6 import peer_feature
from gunes_feature_audit_v9 import build_gunes_structural, add_gunes_target_features


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "exports" / "v12"
MODEL_DIR = EXPORT_DIR / "models"
MODELS = ["TabPFN_v2_5", "TabPFN_v3"]

VARIANTS = {
    "gunes_structural": {
        "gunes_structural": True,
        "gunes_rates": False,
        "familyfare": False,
    },
    "gunes_rates": {
        "gunes_structural": False,
        "gunes_rates": True,
        "familyfare": False,
    },
    "gunes_all": {
        "gunes_structural": True,
        "gunes_rates": True,
        "familyfare": False,
    },
    "familyfare_gunes_rates": {
        "gunes_structural": False,
        "gunes_rates": True,
        "familyfare": True,
    },
    "familyfare_gunes_all": {
        "gunes_structural": True,
        "gunes_rates": True,
        "familyfare": True,
    },
}


def prepare_augmented_data():
    train_base, test_base, folds, base_cols = prepare_data()
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    train_s, test_s = build_gunes_structural(train_raw, test_raw)

    structural_cols = [
        c
        for c in train_s.columns
        if c not in {"PassengerId", "Family_Gunes"}
    ]
    train_base = train_base.merge(train_s, on="PassengerId", how="left")
    test_base = test_base.merge(test_s, on="PassengerId", how="left")

    return train_base, test_base, folds, base_cols, structural_cols


def run_one(
    model_name: str,
    variant_name: str,
    spec: dict,
    train_base: pd.DataFrame,
    test_base: pd.DataFrame,
    folds: np.ndarray,
    base_cols: list[str],
    structural_cols: list[str],
):
    y = train_base["Survived"].astype(int).to_numpy()
    oof = np.zeros(len(train_base), dtype=float)
    test_probs = []
    fold_rows = []

    familyfare_cols = [
        "FamilyFareAny",
        "FamilyFareMean",
        "FamilyFareSmooth",
        "FamilyFareCount",
    ]
    gunes_rate_cols = [
        "GunesFamilyRate",
        "GunesFamilyRateAvailable",
        "GunesTicketRate",
        "GunesTicketRateAvailable",
        "GunesSurvivalRate",
        "GunesSurvivalRateAvailable",
    ]

    for fold in sorted(np.unique(folds)):
        tr_idx = np.flatnonzero(folds != fold)
        va_idx = np.flatnonzero(folds == fold)
        tr = train_base.iloc[tr_idx].copy()
        va = train_base.iloc[va_idx].copy()
        te = test_base.copy()

        # Trusted fold-safe WCG already forms part of the v2 base representation.
        from audit_group_survival import add_group_survival
        tr["GroupSurvival"] = add_group_survival(tr, tr, exclude_self=True).to_numpy()
        va["GroupSurvival"] = add_group_survival(tr, va, exclude_self=False).to_numpy()
        te["GroupSurvival"] = add_group_survival(tr, te, exclude_self=False).to_numpy()

        extra_cols: list[str] = []

        if spec["familyfare"]:
            tr_stats = peer_feature(
                tr, tr, group_col="FamilyFareGroup_v6", wcg_only=False, exclude_self=True
            )
            va_stats = peer_feature(
                tr, va, group_col="FamilyFareGroup_v6", wcg_only=False, exclude_self=False
            )
            te_stats = peer_feature(
                tr, te, group_col="FamilyFareGroup_v6", wcg_only=False, exclude_self=False
            )
            for source_col, out_col in zip(
                ["Any", "Mean", "Smooth", "Count"], familyfare_cols
            ):
                tr[out_col] = tr_stats[source_col].to_numpy()
                va[out_col] = va_stats[source_col].to_numpy()
                te[out_col] = te_stats[source_col].to_numpy()
            extra_cols.extend(familyfare_cols)

        if spec["gunes_rates"]:
            tr_rates = add_gunes_target_features(tr, tr, exclude_self=True)
            va_rates = add_gunes_target_features(tr, va, exclude_self=False)
            te_rates = add_gunes_target_features(tr, te, exclude_self=False)
            for c in gunes_rate_cols:
                tr[c] = tr_rates[c].to_numpy()
                va[c] = va_rates[c].to_numpy()
                te[c] = te_rates[c].to_numpy()
            extra_cols.extend(gunes_rate_cols)

        if spec["gunes_structural"]:
            extra_cols.extend(structural_cols)

        # De-duplicate while preserving order.
        features = list(dict.fromkeys(base_cols + extra_cols))
        model = make_tabpfn(model_name, seed=42 + int(fold))
        model.fit(tr[features], y[tr_idx])
        va_prob = np.asarray(model.predict_proba(va[features]))[:, 1].astype(float)
        te_prob = np.asarray(model.predict_proba(te[features]))[:, 1].astype(float)
        oof[va_idx] = va_prob
        test_probs.append(te_prob)
        fold_rows.append(
            {
                "model": model_name,
                "variant": variant_name,
                "fold": int(fold),
                "accuracy": accuracy_score(y[va_idx], va_prob > 0.5),
                "roc_auc": roc_auc_score(y[va_idx], va_prob),
                "feature_count": len(features),
            }
        )
        print(
            f"{model_name}/{variant_name} fold {fold}: "
            f"acc={fold_rows[-1]['accuracy']:.5f} "
            f"auc={fold_rows[-1]['roc_auc']:.5f}",
            flush=True,
        )

    return oof, np.mean(test_probs, axis=0), pd.DataFrame(fold_rows)


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (
        train_base,
        test_base,
        folds,
        base_cols,
        structural_cols,
    ) = prepare_augmented_data()
    y = train_base["Survived"].astype(int).to_numpy()

    zoo, zoo_test, champion_pred, _, champion_test_pred = load_champion_context()
    champ_correct = champion_pred == y

    summary_rows = []
    diversity_rows = []
    fold_frames = []
    oof_export = pd.DataFrame(
        {
            "PassengerId": train_base["PassengerId"].astype(int),
            "fold": folds,
            "Survived": y,
            "champion_pred": champion_pred,
        }
    )
    test_export = pd.DataFrame(
        {
            "PassengerId": test_base["PassengerId"].astype(int),
            "champion_pred": champion_test_pred,
        }
    )
    failures = []

    for model_name in MODELS:
        for variant_name, spec in VARIANTS.items():
            stem = f"{model_name}__{variant_name}"
            oof_path = MODEL_DIR / f"{stem}__oof.csv"
            test_path = MODEL_DIR / f"{stem}__test.csv"
            fold_path = MODEL_DIR / f"{stem}__fold_metrics.csv"
            try:
                if oof_path.exists() and test_path.exists() and fold_path.exists():
                    print(f"[resume] {stem}", flush=True)
                    oof_df = pd.read_csv(oof_path)
                    te_df = pd.read_csv(test_path)
                    fold_df = pd.read_csv(fold_path)
                    oof = oof_df["probability"].to_numpy()
                    test_prob = te_df["probability"].to_numpy()
                else:
                    print(f"\n=== {stem} ===", flush=True)
                    oof, test_prob, fold_df = run_one(
                        model_name,
                        variant_name,
                        spec,
                        train_base,
                        test_base,
                        folds,
                        base_cols,
                        structural_cols,
                    )
                    pd.DataFrame(
                        {
                            "PassengerId": train_base["PassengerId"].astype(int),
                            "fold": folds,
                            "Survived": y,
                            "probability": oof,
                        }
                    ).to_csv(oof_path, index=False)
                    pd.DataFrame(
                        {
                            "PassengerId": test_base["PassengerId"].astype(int),
                            "probability": test_prob,
                        }
                    ).to_csv(test_path, index=False)
                    fold_df.to_csv(fold_path, index=False)

                pred = (oof > 0.5).astype(int)
                correct = pred == y
                summary_rows.append(
                    {
                        "model": model_name,
                        "variant": variant_name,
                        "accuracy": accuracy_score(y, pred),
                        "roc_auc": roc_auc_score(y, oof),
                        "fold_accuracy_std": fold_df["accuracy"].std(ddof=0),
                        "fold_auc_std": fold_df["roc_auc"].std(ddof=0),
                        "feature_count": int(fold_df["feature_count"].iloc[0]),
                    }
                )
                diversity_rows.append(
                    {
                        "model": model_name,
                        "variant": variant_name,
                        "binary_disagreement_vs_v5": int(np.sum(pred != champion_pred)),
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
                        "correlation_vs_v4b": float(
                            np.corrcoef(
                                oof, zoo["v4b__Champion"].to_numpy()
                            )[0, 1]
                        ),
                        "correlation_vs_rulefit": float(
                            np.corrcoef(oof, zoo["RuleFit"].to_numpy())[0, 1]
                        ),
                    }
                )
                oof_export[stem] = oof
                test_export[stem] = test_prob
                fold_frames.append(fold_df)
            except Exception as exc:
                failures.append(
                    {
                        "model": model_name,
                        "variant": variant_name,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
                print(f"[FAILED] {stem}: {type(exc).__name__}: {exc}", flush=True)

    summary = pd.DataFrame(summary_rows).sort_values(
        ["accuracy", "roc_auc"], ascending=False
    )
    diversity = pd.DataFrame(diversity_rows).sort_values(
        ["v5_wrong_candidate_right", "net_unique_correct_vs_v5"],
        ascending=False,
    )
    summary.to_csv(EXPORT_DIR / "tabpfn_fe_summary.csv", index=False)
    diversity.to_csv(EXPORT_DIR / "tabpfn_fe_diversity.csv", index=False)
    oof_export.to_csv(EXPORT_DIR / "tabpfn_fe_oof.csv", index=False)
    test_export.to_csv(EXPORT_DIR / "tabpfn_fe_test.csv", index=False)
    if fold_frames:
        pd.concat(fold_frames, ignore_index=True).to_csv(
            EXPORT_DIR / "tabpfn_fe_fold_metrics.csv", index=False
        )
    pd.DataFrame(
        [
            {"model": x["model"], "variant": x["variant"], "error": x["error"]}
            for x in failures
        ]
    ).to_csv(EXPORT_DIR / "failures.csv", index=False)

    with (EXPORT_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "models": MODELS,
                "variants": VARIANTS,
                "base_feature_count": len(base_cols),
                "gunes_structural_columns": structural_cols,
                "validation": "trusted fixed five folds; all target features fold-safe",
                "no_submission": True,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== TabPFN feature-engineering ranking ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n=== Diversity vs v5 robust ===")
    print(diversity.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    if failures:
        print("\nFailures:", [(x["model"], x["variant"]) for x in failures])
    print("\nNo Kaggle submission was performed.")


if __name__ == "__main__":
    main()
