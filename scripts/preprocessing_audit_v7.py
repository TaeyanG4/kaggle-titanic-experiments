"""Titanic v7 preprocessing ablation: relational Age and Deck imputation.

Public Titanic practice often suggests using family/ticket relationships to
recover missing attributes. This script isolates that preprocessing question:

  baseline
    Age: Title x Pclass median
    Deck: U when Cabin missing

  age_family_ticket
    Age: FamilyGroup median -> Ticket median -> Title x Pclass median

  age_familyfare_ticket
    Age: LastName+Fare median -> Ticket median -> Title x Pclass median

  deck_family_ticket
    Deck: FamilyGroup known deck -> Ticket known deck -> U

  deck_familyfare_ticket
    Deck: LastName+Fare known deck -> Ticket known deck -> U

  relational_family_ticket
    both Age and Deck relationship-based preprocessing

  relational_familyfare_ticket
    both using LastName+Fare as the family key

The screen uses GradientBoosting, XGBoost and CatBoost on the exact trusted
five-fold manifest. GroupSurvival remains fold-safe. No Kaggle submission.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

try:
    from audit_group_survival import (
        add_group_survival,
        build_models,
        model_feature_columns,
        normalize_ticket_prefix,
    )
except ModuleNotFoundError:
    from scripts.audit_group_survival import (
        add_group_survival,
        build_models,
        model_feature_columns,
        normalize_ticket_prefix,
    )


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
V1_AUDIT = BASE_DIR / "exports" / "wcg_audit_v1"
EXPORT_DIR = BASE_DIR / "exports" / "v7"
MODEL_NAMES = ["GradientBoosting", "XGBoost", "CatBoost"]


TITLE_MAP = {
    "Mr": "Mr", "Miss": "Miss", "Mlle": "Miss", "Ms": "Miss",
    "Mrs": "Mrs", "Mme": "Mrs", "Master": "Master",
    "Dr": "Officer", "Rev": "Officer", "Col": "Officer",
    "Major": "Officer", "Capt": "Officer",
    "Sir": "Royalty", "Don": "Royalty", "Countess": "Royalty",
    "Lady": "Royalty", "Dona": "Royalty", "Jonkheer": "Royalty",
}

VARIANTS = {
    "baseline": {"age_key": None, "deck_key": None},
    "age_family_ticket": {"age_key": "FamilyGroup", "deck_key": None},
    "age_familyfare_ticket": {"age_key": "FamilyFareKey", "deck_key": None},
    "deck_family_ticket": {"age_key": None, "deck_key": "FamilyGroup"},
    "deck_familyfare_ticket": {"age_key": None, "deck_key": "FamilyFareKey"},
    "relational_family_ticket": {"age_key": "FamilyGroup", "deck_key": "FamilyGroup"},
    "relational_familyfare_ticket": {
        "age_key": "FamilyFareKey",
        "deck_key": "FamilyFareKey",
    },
}


def first_known_deck(series: pd.Series):
    known = series.dropna().astype(str)
    if known.empty:
        return np.nan
    return known.iloc[0][0]


def build_variant(train_raw: pd.DataFrame, test_raw: pd.DataFrame, spec: dict):
    train = train_raw.copy()
    test = test_raw.copy()
    train["_source"] = "train"
    test["_source"] = "test"
    df = pd.concat([train, test], ignore_index=True, sort=False)

    df["Title"] = (
        df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False).map(TITLE_MAP).fillna("Other")
    )
    df["LastName"] = df["Name"].str.split(",").str[0].str.strip()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["FamilyGroup"] = df["LastName"] + "_" + df["FamilySize"].astype(str)
    df["Embarked"] = df["Embarked"].fillna("S")

    fare_med = df.groupby(["Pclass", "Embarked"])["Fare"].transform("median")
    df["Fare"] = df["Fare"].fillna(fare_med).fillna(df["Fare"].median())
    df["FamilyFareKey"] = df["LastName"] + "_" + df["Fare"].round(4).astype(str)

    # Age: relational median first when requested, ticket median second, then
    # the trusted Title x Pclass fallback.
    age = df["Age"].copy()
    if spec["age_key"]:
        rel_med = df.groupby(spec["age_key"])["Age"].transform("median")
        age = age.fillna(rel_med)
        ticket_med = df.groupby("Ticket")["Age"].transform("median")
        age = age.fillna(ticket_med)
    title_class_med = (
        pd.DataFrame({"Title": df["Title"], "Pclass": df["Pclass"], "Age": df["Age"]})
        .groupby(["Title", "Pclass"])["Age"]
        .transform("median")
    )
    df["Age"] = age.fillna(title_class_med).fillna(df["Age"].median())

    # Deck: propagate a known deck inside the selected relationship group, then
    # ticket, otherwise unknown U.
    original_deck = df["Cabin"].apply(lambda x: x[0] if pd.notnull(x) else np.nan)
    deck = original_deck.copy()
    if spec["deck_key"]:
        rel_map = df.groupby(spec["deck_key"])["Cabin"].agg(first_known_deck)
        ticket_map = df.groupby("Ticket")["Cabin"].agg(first_known_deck)
        deck = deck.fillna(df[spec["deck_key"]].map(rel_map))
        deck = deck.fillna(df["Ticket"].map(ticket_map))
    df["Deck"] = deck.fillna("U").astype(str)

    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["FamilyType"] = pd.cut(
        df["FamilySize"],
        bins=[0, 1, 4, 6, 20],
        labels=["Alone", "Small", "Medium", "Large"],
    )
    df["IsMarriedWoman"] = (df["Title"] == "Mrs").astype(int)
    df["LogFare"] = np.log1p(df["Fare"])

    ticket_counts = df["Ticket"].value_counts()
    df["TicketFreq"] = df["Ticket"].map(ticket_counts)
    df["FarePerPerson"] = df["Fare"] / df["TicketFreq"]
    df["LogFarePerPerson"] = np.log1p(df["FarePerPerson"])
    df["TicketPrefix_debug"] = df["Ticket"].map(normalize_ticket_prefix)
    df["IsWomanOrChild"] = (
        (df["Sex"] == "female") | (df["Title"] == "Master") | (df["Age"] <= 12)
    ).astype(int)
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
    df["GroupSurvival"] = 0.5

    cat_cols = ["Sex", "Embarked", "Title", "Deck", "FamilyType", "TicketPrefix"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    drop_cols = [
        "Name",
        "Cabin",
        "LastName",
        "Fare",
        "FarePerPerson",
        "FamilyFareKey",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    train_df = df[df["_source"] == "train"].copy().reset_index(drop=True)
    test_df = df[df["_source"] == "test"].copy().reset_index(drop=True)
    train_df["Survived"] = train_df["Survived"].astype(int)
    return train_df, test_df


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    manifest = pd.read_csv(V1_AUDIT / "fold_manifest_seed42.csv")
    folds = manifest["fold"].astype(int).to_numpy()
    splits = [
        (np.flatnonzero(folds != f), np.flatnonzero(folds == f))
        for f in sorted(np.unique(folds))
    ]

    summary_rows = []
    fold_rows = []
    oof_export = pd.DataFrame(
        {
            "PassengerId": train_raw["PassengerId"],
            "fold": folds,
            "Survived": train_raw["Survived"].astype(int),
        }
    )
    test_export = pd.DataFrame({"PassengerId": test_raw["PassengerId"]})

    for variant, spec in VARIANTS.items():
        print(f"\n=== preprocessing variant: {variant} ===", flush=True)
        train_df, test_df = build_variant(train_raw, test_raw, spec)
        if not train_df["PassengerId"].equals(manifest["PassengerId"]):
            raise ValueError("PassengerId order mismatch.")
        y = train_df["Survived"].astype(int).reset_index(drop=True)
        features = model_feature_columns(train_df)
        model_oof = {m: np.zeros(len(train_df), dtype=float) for m in MODEL_NAMES}
        model_test = {m: [] for m in MODEL_NAMES}

        for fold, (tr_idx, va_idx) in enumerate(splits):
            tr = train_df.iloc[tr_idx].copy()
            va = train_df.iloc[va_idx].copy()
            te = test_df.copy()
            tr["GroupSurvival"] = add_group_survival(tr, tr, exclude_self=True).to_numpy()
            va["GroupSurvival"] = add_group_survival(tr, va, exclude_self=False).to_numpy()
            te["GroupSurvival"] = add_group_survival(tr, te, exclude_self=False).to_numpy()

            for model_name in MODEL_NAMES:
                model = build_models(42 + fold, profile="v2")[model_name]
                model.fit(tr[features], y.iloc[tr_idx])
                va_prob = np.asarray(model.predict_proba(va[features]))[:, 1]
                te_prob = np.asarray(model.predict_proba(te[features]))[:, 1]
                model_oof[model_name][va_idx] = va_prob
                model_test[model_name].append(te_prob)
                fold_rows.append(
                    {
                        "variant": variant,
                        "model": model_name,
                        "fold": fold,
                        "accuracy": accuracy_score(y.iloc[va_idx], va_prob > 0.5),
                        "roc_auc": roc_auc_score(y.iloc[va_idx], va_prob),
                    }
                )

        panel = np.mean([model_oof[m] for m in MODEL_NAMES], axis=0)
        panel_test = np.mean(
            [np.mean(model_test[m], axis=0) for m in MODEL_NAMES], axis=0
        )
        oof_export[f"{variant}__Panel"] = panel
        test_export[f"{variant}__Panel"] = panel_test
        summary_rows.append(
            {
                "variant": variant,
                "model": "Panel",
                "accuracy": accuracy_score(y, panel > 0.5),
                "roc_auc": roc_auc_score(y, panel),
                "feature_count": len(features),
            }
        )
        for model_name in MODEL_NAMES:
            prob = model_oof[model_name]
            tprob = np.mean(model_test[model_name], axis=0)
            oof_export[f"{variant}__{model_name}"] = prob
            test_export[f"{variant}__{model_name}"] = tprob
            summary_rows.append(
                {
                    "variant": variant,
                    "model": model_name,
                    "accuracy": accuracy_score(y, prob > 0.5),
                    "roc_auc": roc_auc_score(y, prob),
                    "feature_count": len(features),
                }
            )

        print(
            f"panel acc={summary_rows[-4]['accuracy']:.5f} "
            f"auc={summary_rows[-4]['roc_auc']:.5f}",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    baseline = summary[summary["variant"] == "baseline"][
        ["model", "accuracy", "roc_auc"]
    ].rename(
        columns={"accuracy": "baseline_accuracy", "roc_auc": "baseline_roc_auc"}
    )
    summary = summary.merge(baseline, on="model", how="left")
    summary["delta_accuracy"] = summary["accuracy"] - summary["baseline_accuracy"]
    summary["delta_auc"] = summary["roc_auc"] - summary["baseline_roc_auc"]
    summary = summary.sort_values(["accuracy", "roc_auc"], ascending=False)

    summary.to_csv(EXPORT_DIR / "preprocessing_audit_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(
        EXPORT_DIR / "preprocessing_audit_fold_metrics.csv", index=False
    )
    oof_export.to_csv(EXPORT_DIR / "preprocessing_audit_oof.csv", index=False)
    test_export.to_csv(EXPORT_DIR / "preprocessing_audit_test.csv", index=False)
    with (EXPORT_DIR / "preprocessing_audit_metadata.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "variants": VARIANTS,
                "models": MODEL_NAMES,
                "fold_manifest": str(V1_AUDIT / "fold_manifest_seed42.csv"),
                "notes": [
                    "Relational Age/Deck preprocessing uses only target-independent columns.",
                    "Combined train+test is used for target-independent group medians, consistent with existing project preprocessing.",
                    "GroupSurvival remains fold-safe.",
                    "No Kaggle submission is performed.",
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("\n=== preprocessing ranking ===")
    print(summary.head(30).to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
