"""Leakage audit for Titanic GroupSurvival (WCG) feature.

This script compares three otherwise-identical validation variants on the same
StratifiedKFold manifest:

A. no_wcg       : GroupSurvival fixed to 0.5
B. global_loo   : legacy behavior; all training labels are visible except self
C. fold_safe    : validation/test GroupSurvival uses labels from fold-train only

The purpose is diagnostic. Variant B is intentionally leakage-prone and must
never be promoted as the validation baseline.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DEFAULT_EXPORT_DIR = BASE_DIR / "exports"

VARIANTS = ("no_wcg", "global_loo", "fold_safe")


def normalize_ticket_prefix(ticket: str) -> str:
    """Normalize ticket prefix for stable grouping/debug output only."""
    value = str(ticket).upper().strip()
    prefix = re.sub(r"\d", "", value)
    prefix = re.sub(r"[\s./]+", "", prefix)
    return prefix or "NONE"


def build_base_frame(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    profile: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build target-independent features while retaining WCG helper columns.

    This intentionally mirrors the current champion's non-target preprocessing
    so the audit isolates GroupSurvival construction rather than changing many
    factors at once.
    """
    train = train_df.copy()
    test = test_df.copy()
    train["_source"] = "train"
    test["_source"] = "test"

    df = pd.concat([train, test], ignore_index=True, sort=False)

    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title_mapping = {
        "Mr": "Mr",
        "Miss": "Miss",
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mrs": "Mrs",
        "Mme": "Mrs",
        "Master": "Master",
        "Dr": "Officer",
        "Rev": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Capt": "Officer",
        "Sir": "Royalty",
        "Don": "Royalty",
        "Countess": "Royalty",
        "Lady": "Royalty",
        "Dona": "Royalty",
        "Jonkheer": "Royalty",
    }
    df["Title"] = df["Title"].map(title_mapping).fillna("Other")
    if profile == "v2":
        df["IsMarriedWoman"] = (df["Title"] == "Mrs").astype(int)

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    if profile == "v2":
        df["FamilyType"] = pd.cut(
            df["FamilySize"],
            bins=[0, 1, 4, 6, 20],
            labels=["Alone", "Small", "Medium", "Large"],
        )
    df["Deck"] = df["Cabin"].apply(lambda x: x[0] if pd.notnull(x) else "U")

    df["Embarked"] = df["Embarked"].fillna("S")

    med_fare = df.groupby(["Pclass", "Embarked"])["Fare"].transform("median")
    df["Fare"] = df["Fare"].fillna(med_fare).fillna(df["Fare"].median())
    df["LogFare"] = np.log1p(df["Fare"])

    med_age = df.groupby(["Title", "Pclass"])["Age"].transform("median")
    df["Age"] = df["Age"].fillna(med_age).fillna(df["Age"].median())

    ticket_counts = df["Ticket"].value_counts()
    df["TicketFreq"] = df["Ticket"].map(ticket_counts)
    df["TicketPrefix_debug"] = df["Ticket"].map(normalize_ticket_prefix)
    if profile == "v2":
        df["FarePerPerson"] = df["Fare"] / df["TicketFreq"]
        df["LogFarePerPerson"] = np.log1p(df["FarePerPerson"])

    df["LastName"] = df["Name"].apply(lambda x: x.split(",")[0].strip())
    df["IsWomanOrChild"] = (
        (df["Sex"] == "female") | (df["Title"] == "Master") | (df["Age"] <= 12)
    ).astype(int)
    df["FamilyGroup"] = df["LastName"] + "_" + df["FamilySize"].astype(str)

    if profile == "v2":
        df["Age_Pclass"] = df["Age"] * df["Pclass"]
        df["IsChild"] = (df["Age"] <= 12).astype(int)

        def clean_ticket_prefix(ticket: str) -> str:
            parts = str(ticket).replace("/", "").replace(".", "").strip().split(" ")
            return parts[0].upper() if len(parts) > 1 else "NONE"

        df["TicketPrefix"] = df["Ticket"].apply(clean_ticket_prefix)
        top_prefixes = df["TicketPrefix"].value_counts()
        top_prefixes = top_prefixes[top_prefixes > 10].index
        df["TicketPrefix"] = df["TicketPrefix"].apply(
            lambda x: x if x in top_prefixes else "OTHER"
        )

    # GroupSurvival is populated later according to each validation variant.
    df["GroupSurvival"] = 0.5

    cat_cols = ["Sex", "Embarked", "Title", "Deck"]
    if profile == "v2":
        cat_cols += ["FamilyType", "TicketPrefix"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # TicketPrefix_debug is retained only to make future diagnostics easier;
    # it is not a model feature in this Phase-0 audit.
    drop_cols = ["Name", "Cabin", "LastName", "Fare"]
    if profile == "v2":
        drop_cols.append("FarePerPerson")
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    train_base = df[df["_source"] == "train"].copy().reset_index(drop=True)
    test_base = df[df["_source"] == "test"].copy().reset_index(drop=True)

    if train_base["Survived"].isna().any():
        raise ValueError("Train contains missing Survived labels after preprocessing.")
    if not test_base["Survived"].isna().all():
        raise ValueError("Test unexpectedly contains Survived labels.")

    train_base["Survived"] = train_base["Survived"].astype(int)
    return train_base, test_base


def add_group_survival(
    reference_labeled: pd.DataFrame,
    apply_df: pd.DataFrame,
    *,
    exclude_self: bool,
) -> pd.Series:
    """Create GroupSurvival using labels from reference_labeled only.

    The current project semantics are preserved: Ticket signal is computed
    first, then FamilyGroup may overwrite it only when peer outcomes are
    unanimously 0 or unanimously 1. Mixed/absent peer outcomes leave the
    current value unchanged (default 0.5).
    """
    if reference_labeled["Survived"].isna().any():
        raise ValueError("reference_labeled must contain known Survived labels only.")

    result = pd.Series(0.5, index=apply_df.index, dtype=float)
    ref_wc = reference_labeled[reference_labeled["IsWomanOrChild"] == 1].copy()

    for group_col in ("Ticket", "FamilyGroup"):
        grouped = ref_wc.groupby(group_col, sort=False)
        for idx, row in apply_df.iterrows():
            key = row[group_col]
            if key not in grouped.groups:
                continue

            peers = ref_wc.loc[grouped.groups[key]]
            if exclude_self:
                peers = peers[peers["PassengerId"] != row["PassengerId"]]
            if peers.empty:
                continue

            mean_survival = peers["Survived"].mean()
            if mean_survival == 1.0:
                result.loc[idx] = 1.0
            elif mean_survival == 0.0:
                result.loc[idx] = 0.0

    return result


def model_feature_columns(df: pd.DataFrame) -> list[str]:
    helper_cols = {
        "Survived",
        "PassengerId",
        "Ticket",
        "FamilyGroup",
        "TicketPrefix_debug",
        "_source",
    }
    cols = [c for c in df.columns if c not in helper_cols]
    object_cols = [c for c in cols if df[c].dtype == "object"]
    if object_cols:
        raise TypeError(f"Unexpected object model columns: {object_cols}")
    return cols


def build_models(random_state: int, profile: str) -> dict[str, object]:
    if profile == "v1":
        return {
            "RandomForest": RandomForestClassifier(
                n_estimators=150,
                max_depth=5,
                min_samples_split=4,
                random_state=random_state,
                n_jobs=-1,
            ),
            "ExtraTrees": ExtraTreesClassifier(
                n_estimators=150,
                max_depth=5,
                min_samples_split=4,
                random_state=random_state,
                n_jobs=-1,
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.04,
                random_state=random_state,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.04,
                random_state=random_state,
                eval_metric="logloss",
                n_jobs=-1,
            ),
            "LightGBM": LGBMClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.04,
                random_state=random_state,
                verbose=-1,
                n_jobs=-1,
            ),
            "CatBoost": CatBoostClassifier(
                iterations=120,
                depth=4,
                learning_rate=0.04,
                random_seed=random_state,
                verbose=0,
                thread_count=-1,
            ),
        }

    if profile == "v2":
        return {
            "RandomForest": RandomForestClassifier(
                n_estimators=180,
                max_depth=5,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
            ),
            "ExtraTrees": ExtraTreesClassifier(
                n_estimators=180,
                max_depth=5,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=110,
                max_depth=3,
                learning_rate=0.035,
                subsample=0.85,
                random_state=random_state,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=110,
                max_depth=3,
                learning_rate=0.035,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=random_state,
                eval_metric="logloss",
                n_jobs=-1,
            ),
            "LightGBM": LGBMClassifier(
                n_estimators=110,
                max_depth=3,
                learning_rate=0.035,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=random_state,
                verbose=-1,
                n_jobs=-1,
            ),
            "CatBoost": CatBoostClassifier(
                iterations=130,
                depth=4,
                learning_rate=0.035,
                l2_leaf_reg=3,
                random_seed=random_state,
                verbose=0,
                thread_count=-1,
            ),
        }

    raise ValueError(f"Unknown profile: {profile}")


def ensemble_weights(profile: str, selected_models: list[str]) -> dict[str, float]:
    if profile == "v1":
        raw = {name: 1.0 for name in selected_models}
    elif profile == "v2":
        v2 = {
            "CatBoost": 0.25,
            "GradientBoosting": 0.20,
            "XGBoost": 0.20,
            "RandomForest": 0.15,
            "ExtraTrees": 0.10,
            "LightGBM": 0.10,
        }
        raw = {name: v2[name] for name in selected_models}
    else:
        raise ValueError(f"Unknown profile: {profile}")

    total = sum(raw.values())
    return {name: weight / total for name, weight in raw.items()}


def parse_models(value: str, available: list[str]) -> list[str]:
    if value.lower() == "all":
        return available
    requested = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [name for name in requested if name not in available]
    if unknown:
        raise ValueError(f"Unknown models {unknown}. Available: {available}")
    if not requested:
        raise ValueError("At least one model must be selected.")
    return requested


def run_audit(
    train_base: pd.DataFrame,
    test_base: pd.DataFrame,
    export_dir: Path,
    *,
    n_splits: int,
    random_state: int,
    selected_models: list[str],
    profile: str,
) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)

    y = train_base["Survived"].astype(int).reset_index(drop=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(skf.split(train_base, y))

    fold_manifest = pd.DataFrame({"PassengerId": train_base["PassengerId"], "fold": -1})
    for fold, (_, val_idx) in enumerate(splits):
        fold_manifest.loc[val_idx, "fold"] = fold
    fold_manifest.to_csv(export_dir / f"fold_manifest_seed{random_state}.csv", index=False)

    global_loo_train = add_group_survival(train_base, train_base, exclude_self=True)
    global_loo_test = add_group_survival(train_base, test_base, exclude_self=False)

    models_template = build_models(random_state, profile)
    weights = ensemble_weights(profile, selected_models)
    model_cols = model_feature_columns(train_base)

    oof = {
        variant: {name: np.zeros(len(train_base), dtype=float) for name in selected_models}
        for variant in VARIANTS
    }
    test_fold_probs = {
        variant: {name: [] for name in selected_models}
        for variant in VARIANTS
    }
    fold_rows: list[dict[str, object]] = []

    start = time.time()
    for fold, (train_idx, val_idx) in enumerate(splits):
        fold_train = train_base.iloc[train_idx].copy()
        fold_val = train_base.iloc[val_idx].copy()

        fold_safe_train = add_group_survival(fold_train, fold_train, exclude_self=True)
        fold_safe_val = add_group_survival(fold_train, fold_val, exclude_self=False)
        fold_safe_test = add_group_survival(fold_train, test_base, exclude_self=False)

        variant_frames = {}
        for variant in VARIANTS:
            tr = fold_train.copy()
            va = fold_val.copy()
            te = test_base.copy()

            if variant == "no_wcg":
                tr["GroupSurvival"] = 0.5
                va["GroupSurvival"] = 0.5
                te["GroupSurvival"] = 0.5
            elif variant == "global_loo":
                tr["GroupSurvival"] = global_loo_train.iloc[train_idx].to_numpy()
                va["GroupSurvival"] = global_loo_train.iloc[val_idx].to_numpy()
                te["GroupSurvival"] = global_loo_test.to_numpy()
            elif variant == "fold_safe":
                tr["GroupSurvival"] = fold_safe_train.to_numpy()
                va["GroupSurvival"] = fold_safe_val.to_numpy()
                te["GroupSurvival"] = fold_safe_test.to_numpy()
            else:  # pragma: no cover
                raise AssertionError(variant)

            variant_frames[variant] = (tr, va, te)

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        for variant in VARIANTS:
            tr, va, te = variant_frames[variant]
            X_train = tr[model_cols]
            X_val = va[model_cols]
            X_test = te[model_cols]

            fold_model_probs = []
            for model_name in selected_models:
                model = copy.deepcopy(models_template[model_name])
                model.fit(X_train, y_train)
                val_prob = model.predict_proba(X_val)[:, 1]
                test_prob = model.predict_proba(X_test)[:, 1]

                oof[variant][model_name][val_idx] = val_prob
                test_fold_probs[variant][model_name].append(test_prob)
                fold_model_probs.append(val_prob)

                fold_rows.append(
                    {
                        "fold": fold,
                        "variant": variant,
                        "model": model_name,
                        "accuracy": accuracy_score(y_val, (val_prob > 0.5).astype(int)),
                        "roc_auc": roc_auc_score(y_val, val_prob),
                        "n_train": len(train_idx),
                        "n_val": len(val_idx),
                    }
                )

            ensemble_prob = np.zeros(len(val_idx), dtype=float)
            for model_name, val_prob in zip(selected_models, fold_model_probs):
                ensemble_prob += val_prob * weights[model_name]
            fold_rows.append(
                {
                    "fold": fold,
                    "variant": variant,
                    "model": "Ensemble",
                    "accuracy": accuracy_score(y_val, (ensemble_prob > 0.5).astype(int)),
                    "roc_auc": roc_auc_score(y_val, ensemble_prob),
                    "n_train": len(train_idx),
                    "n_val": len(val_idx),
                }
            )

        elapsed = time.time() - start
        print(f"[fold {fold + 1}/{n_splits}] complete - elapsed {elapsed:.1f}s", flush=True)

    cv_rows = pd.DataFrame(fold_rows)
    cv_rows.to_csv(export_dir / "cv_group_survival_audit.csv", index=False)

    summary_rows = []
    oof_export = pd.DataFrame(
        {
            "PassengerId": train_base["PassengerId"],
            "fold": fold_manifest["fold"],
            "Survived": y,
        }
    )
    test_export = pd.DataFrame({"PassengerId": test_base["PassengerId"]})

    for variant in VARIANTS:
        model_oof_probs = []
        model_test_probs = []
        for model_name in selected_models:
            prob = oof[variant][model_name]
            test_prob = np.mean(test_fold_probs[variant][model_name], axis=0)
            model_oof_probs.append(prob)
            model_test_probs.append(test_prob)
            oof_export[f"{variant}__{model_name}"] = prob
            test_export[f"{variant}__{model_name}"] = test_prob

            summary_rows.append(
                {
                    "variant": variant,
                    "model": model_name,
                    "accuracy": accuracy_score(y, (prob > 0.5).astype(int)),
                    "roc_auc": roc_auc_score(y, prob),
                }
            )

        ensemble_oof = np.zeros(len(train_base), dtype=float)
        ensemble_test = np.zeros(len(test_base), dtype=float)
        for model_name, model_oof, model_test in zip(selected_models, model_oof_probs, model_test_probs):
            ensemble_oof += model_oof * weights[model_name]
            ensemble_test += model_test * weights[model_name]
        oof_export[f"{variant}__Ensemble"] = ensemble_oof
        test_export[f"{variant}__Ensemble"] = ensemble_test
        summary_rows.append(
            {
                "variant": variant,
                "model": "Ensemble",
                "accuracy": accuracy_score(y, (ensemble_oof > 0.5).astype(int)),
                "roc_auc": roc_auc_score(y, ensemble_oof),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(["model", "variant"]).reset_index(drop=True)
    summary.to_csv(export_dir / "cv_group_survival_summary.csv", index=False)
    oof_export.to_csv(export_dir / "oof_group_survival_audit.csv", index=False)
    test_export.to_csv(export_dir / "test_group_survival_audit.csv", index=False)

    fold_safe_cols = ["PassengerId", "fold", "Survived"] + [
        c for c in oof_export.columns if c.startswith("fold_safe__")
    ]
    oof_export[fold_safe_cols].to_csv(export_dir / "oof_fold_safe.csv", index=False)

    ensemble_folds = cv_rows[cv_rows["model"] == "Ensemble"].pivot(
        index="fold", columns="variant", values=["accuracy", "roc_auc"]
    )
    paired_rows = []
    for fold in sorted(cv_rows["fold"].unique()):
        row = {"fold": int(fold)}
        for metric in ("accuracy", "roc_auc"):
            a = float(ensemble_folds.loc[fold, (metric, "no_wcg")])
            b = float(ensemble_folds.loc[fold, (metric, "global_loo")])
            c = float(ensemble_folds.loc[fold, (metric, "fold_safe")])
            row[f"{metric}__fold_safe_minus_no_wcg"] = c - a
            row[f"{metric}__global_loo_minus_fold_safe"] = b - c
        paired_rows.append(row)
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(export_dir / "cv_group_survival_paired_deltas.csv", index=False)

    metadata = {
        "n_splits": n_splits,
        "random_state": random_state,
        "models": selected_models,
        "profile": profile,
        "ensemble_weights": weights,
        "variants": list(VARIANTS),
        "model_feature_count": len(model_cols),
        "model_features": model_cols,
        "elapsed_seconds": round(time.time() - start, 3),
        "note": "global_loo is diagnostic only and leakage-prone across CV folds",
    }
    with open(export_dir / "group_survival_audit_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n=== Global OOF summary ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    ensemble_summary = summary[summary["model"] == "Ensemble"].set_index("variant")
    a_acc = ensemble_summary.loc["no_wcg", "accuracy"]
    b_acc = ensemble_summary.loc["global_loo", "accuracy"]
    c_acc = ensemble_summary.loc["fold_safe", "accuracy"]
    a_auc = ensemble_summary.loc["no_wcg", "roc_auc"]
    b_auc = ensemble_summary.loc["global_loo", "roc_auc"]
    c_auc = ensemble_summary.loc["fold_safe", "roc_auc"]
    print("\n=== Audit deltas (Ensemble) ===")
    print(f"fold_safe - no_wcg : Accuracy {c_acc - a_acc:+.5f}, ROC-AUC {c_auc - a_auc:+.5f}")
    print(f"global_loo - fold_safe: Accuracy {b_acc - c_acc:+.5f}, ROC-AUC {b_auc - c_auc:+.5f}")
    print(f"\nArtifacts written to: {export_dir}")


def run_check_only(train_base: pd.DataFrame, test_base: pd.DataFrame, profile: str) -> None:
    """Fast structural checks without fitting any model."""
    model_cols = model_feature_columns(train_base)
    safe = add_group_survival(train_base.iloc[:700].copy(), train_base.iloc[700:].copy(), exclude_self=False)
    if safe.isna().any() or not set(safe.unique()).issubset({0.0, 0.5, 1.0}):
        raise AssertionError("GroupSurvival structural check failed.")
    if len(train_base) != 891 or len(test_base) != 418:
        raise AssertionError(f"Unexpected Titanic shapes: train={len(train_base)}, test={len(test_base)}")
    if train_base[model_cols].isna().any().any() or test_base[model_cols].isna().any().any():
        raise AssertionError("Model matrix contains NaN after base preprocessing.")

    print("CHECK_ONLY PASS")
    print(f"profile={profile}, train rows={len(train_base)}, test rows={len(test_base)}, model features={len(model_cols)}")
    print(f"fold-safe probe values={sorted(safe.unique().tolist())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Titanic GroupSurvival CV leakage.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--profile", choices=["v1", "v2"], default="v2")
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated model names or 'all'.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate preprocessing and WCG construction without model training.",
    )
    args = parser.parse_args()

    train_path = args.data_dir / "train.csv"
    test_path = args.data_dir / "test.csv"
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    train_base, test_base = build_base_frame(train_raw, test_raw, profile=args.profile)

    if args.check_only:
        run_check_only(train_base, test_base, args.profile)
        return

    available = list(build_models(args.random_state, args.profile).keys())
    selected = parse_models(args.models, available)
    export_dir = args.export_dir or (DEFAULT_EXPORT_DIR / f"wcg_audit_{args.profile}")
    print(f"Profile: {args.profile}")
    print(f"Models: {selected}")
    print(f"Variants: {list(VARIANTS)}")
    print(f"CV: StratifiedKFold(n_splits={args.n_splits}, shuffle=True, random_state={args.random_state})")
    run_audit(
        train_base,
        test_base,
        export_dir,
        n_splits=args.n_splits,
        random_state=args.random_state,
        selected_models=selected,
        profile=args.profile,
    )


if __name__ == "__main__":
    main()
