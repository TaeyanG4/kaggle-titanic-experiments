"""Reproduce Gunes Evitan's Titanic leaderboard pipeline as closely as possible.

This is intentionally a *historical reproduction* of the public notebook, not
the project's preferred leakage-safe validation pipeline.

Source:
  Gunes Evitan - Titanic: Advanced Feature Engineering Tutorial
  https://www.kaggle.com/code/gunesevitan/titanic-advanced-feature-engineering-tutorial

Compatibility differences only:
  * sklearn no longer accepts max_features='auto' for RandomForestClassifier;
    'sqrt' is the documented equivalent of the historical classifier default.
  * deprecated pandas groupby/indexing syntax is rewritten without changing the
    intended computed values.

The original notebook:
  * combines train+test for target-independent preprocessing,
  * fits LabelEncoder / OneHotEncoder separately to train and test,
  * fits StandardScaler separately to train and test,
  * constructs Family/Ticket survival rates from all training labels and uses
    only groups that also occur in test,
  * then performs 5-fold RF fitting and averages test probabilities.

No Kaggle submission is performed by this script itself.
"""

from __future__ import annotations

import json
import string
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "exports" / "v10"
SUBMISSION_DIR = BASE_DIR / "submissions"
SEED = 42
N = 5


def concat_df(train_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)


def divide_df(all_data: pd.DataFrame):
    return all_data.loc[:890].copy(), all_data.loc[891:].drop(["Survived"], axis=1).copy()


def extract_surname(data: pd.Series) -> list[str]:
    families: list[str] = []
    for i in range(len(data)):
        name = data.iloc[i]
        name_no_bracket = name.split("(")[0] if "(" in name else name
        family = name_no_bracket.split(",")[0]
        for c in string.punctuation:
            family = family.replace(c, "").strip()
        families.append(family)
    return families


def preprocess_original(train_path: Path, test_path: Path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    passenger_ids = df_test["PassengerId"].astype(int).copy()
    df_all = concat_df(df_train, df_test)

    # Original Age imputation: combined train+test, Sex x Pclass medians.
    age_medians = df_all.groupby(["Sex", "Pclass"])["Age"].transform("median")
    df_all["Age"] = df_all["Age"].fillna(age_medians)

    # Original explicit Embarked correction based on external Titanic research.
    df_all["Embarked"] = df_all["Embarked"].fillna("S")

    # Original Fare imputation for third-class solo passenger.
    med_fare = df_all.loc[
        (df_all["Pclass"] == 3) & (df_all["Parch"] == 0) & (df_all["SibSp"] == 0),
        "Fare",
    ].median()
    df_all["Fare"] = df_all["Fare"].fillna(med_fare)

    # Cabin -> Deck, then tutorial grouping.
    df_all["Deck"] = df_all["Cabin"].apply(lambda s: s[0] if pd.notnull(s) else "M")
    df_all.loc[df_all["Deck"] == "T", "Deck"] = "A"
    df_all["Deck"] = df_all["Deck"].replace(["A", "B", "C"], "ABC")
    df_all["Deck"] = df_all["Deck"].replace(["D", "E"], "DE")
    df_all["Deck"] = df_all["Deck"].replace(["F", "G"], "FG")
    df_all.drop(["Cabin"], inplace=True, axis=1)

    df_train, df_test = divide_df(df_all)
    df_all = concat_df(df_train, df_test)

    # Quantile bins on combined train+test.
    df_all["Fare"] = pd.qcut(df_all["Fare"], 13)
    df_all["Age"] = pd.qcut(df_all["Age"], 10)

    # Frequency features.
    df_all["Family_Size"] = df_all["SibSp"] + df_all["Parch"] + 1
    family_map = {
        1: "Alone",
        2: "Small",
        3: "Small",
        4: "Small",
        5: "Medium",
        6: "Medium",
        7: "Large",
        8: "Large",
        11: "Large",
    }
    df_all["Family_Size_Grouped"] = df_all["Family_Size"].map(family_map)
    df_all["Ticket_Frequency"] = df_all.groupby("Ticket")["Ticket"].transform("count")

    # Title and Is_Married.
    df_all["Title"] = (
        df_all["Name"].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    )
    df_all["Is_Married"] = 0
    df_all.loc[df_all["Title"] == "Mrs", "Is_Married"] = 1
    df_all["Title"] = df_all["Title"].replace(
        ["Miss", "Mrs", "Ms", "Mlle", "Lady", "Mme", "the Countess", "Dona"],
        "Miss/Mrs/Ms",
    )
    df_all["Title"] = df_all["Title"].replace(
        ["Dr", "Col", "Major", "Jonkheer", "Capt", "Sir", "Don", "Rev"],
        "Dr/Military/Noble/Clergy",
    )

    # Surname family target encoding.
    df_all["Family"] = extract_surname(df_all["Name"])
    df_train = df_all.loc[:890].copy()
    df_test = df_all.loc[891:].copy()
    dfs = [df_train, df_test]

    non_unique_families = [
        x for x in df_train["Family"].unique() if x in df_test["Family"].unique()
    ]
    non_unique_tickets = [
        x for x in df_train["Ticket"].unique() if x in df_test["Ticket"].unique()
    ]

    family_stats = (
        df_train.groupby("Family")
        .agg(Survived=("Survived", "median"), Family_Size=("Family_Size", "median"))
    )
    ticket_stats = (
        df_train.groupby("Ticket")
        .agg(
            Survived=("Survived", "median"),
            Ticket_Frequency=("Ticket_Frequency", "median"),
        )
    )

    family_rates = {
        family: float(row["Survived"])
        for family, row in family_stats.iterrows()
        if family in non_unique_families and row["Family_Size"] > 1
    }
    ticket_rates = {
        ticket: float(row["Survived"])
        for ticket, row in ticket_stats.iterrows()
        if ticket in non_unique_tickets and row["Ticket_Frequency"] > 1
    }

    mean_survival_rate = float(np.mean(df_train["Survived"]))
    for df in [df_train, df_test]:
        df["Family_Survival_Rate"] = df["Family"].map(family_rates).fillna(mean_survival_rate)
        df["Family_Survival_Rate_NA"] = df["Family"].isin(family_rates).astype(int)
        df["Ticket_Survival_Rate"] = df["Ticket"].map(ticket_rates).fillna(mean_survival_rate)
        df["Ticket_Survival_Rate_NA"] = df["Ticket"].isin(ticket_rates).astype(int)
        df["Survival_Rate"] = (
            df["Ticket_Survival_Rate"] + df["Family_Survival_Rate"]
        ) / 2
        df["Survival_Rate_NA"] = (
            df["Ticket_Survival_Rate_NA"] + df["Family_Survival_Rate_NA"]
        ) / 2

    # Original notebook fits encoders separately on train/test.
    non_numeric_features = [
        "Embarked",
        "Sex",
        "Deck",
        "Title",
        "Family_Size_Grouped",
        "Age",
        "Fare",
    ]
    dfs = [df_train, df_test]
    for df in dfs:
        for feature in non_numeric_features:
            df[feature] = LabelEncoder().fit_transform(df[feature])

    cat_features = [
        "Pclass",
        "Sex",
        "Deck",
        "Embarked",
        "Title",
        "Family_Size_Grouped",
    ]
    encoded_features: list[pd.DataFrame] = []
    for df in dfs:
        for feature in cat_features:
            # Current sklearn name; behavior equivalent to historical default.
            encoded_feat = OneHotEncoder(sparse_output=False).fit_transform(
                df[feature].values.reshape(-1, 1)
            )
            n = df[feature].nunique()
            cols = [f"{feature}_{j}" for j in range(1, n + 1)]
            encoded_df = pd.DataFrame(encoded_feat, columns=cols)
            encoded_df.index = df.index
            encoded_features.append(encoded_df)

    df_train = pd.concat([df_train, *encoded_features[:6]], axis=1)
    df_test = pd.concat([df_test, *encoded_features[6:]], axis=1)

    drop_cols = [
        "Deck",
        "Embarked",
        "Family",
        "Family_Size",
        "Family_Size_Grouped",
        "Survived",
        "Name",
        "Parch",
        "PassengerId",
        "Pclass",
        "Sex",
        "SibSp",
        "Ticket",
        "Title",
        "Ticket_Survival_Rate",
        "Family_Survival_Rate",
        "Ticket_Survival_Rate_NA",
        "Family_Survival_Rate_NA",
    ]

    feature_train = df_train.drop(columns=drop_cols)
    feature_test = df_test.drop(columns=drop_cols)
    if list(feature_train.columns) != list(feature_test.columns):
        raise ValueError(
            "Train/test feature columns do not match. "
            f"train={list(feature_train.columns)}, test={list(feature_test.columns)}"
        )
    if feature_train.shape[1] != 26:
        raise ValueError(f"Expected original 26 features, got {feature_train.shape[1]}")

    # The original notebook fits scalers independently on train and test.
    X_train = StandardScaler().fit_transform(feature_train)
    y_train = df_train["Survived"].astype(int).to_numpy()
    X_test = StandardScaler().fit_transform(feature_test)

    return X_train, y_train, X_test, passenger_ids, feature_train.columns.tolist()


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, passenger_ids, feature_names = preprocess_original(
        DATA_DIR / "train.csv", DATA_DIR / "test.csv"
    )
    print("X_train:", X_train.shape, "X_test:", X_test.shape)
    print("features:", feature_names)

    leaderboard_model = RandomForestClassifier(
        criterion="gini",
        n_estimators=1750,
        max_depth=7,
        min_samples_split=6,
        min_samples_leaf=6,
        max_features="sqrt",  # historical classifier max_features='auto'
        oob_score=True,
        random_state=SEED,
        n_jobs=-1,
        verbose=0,
    )

    probs = np.zeros((len(X_test), N), dtype=float)
    oof = np.zeros(len(X_train), dtype=float)
    fold_rows = []
    oob_mean = 0.0
    skf = StratifiedKFold(n_splits=N, random_state=N, shuffle=True)

    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        leaderboard_model.fit(X_train[trn_idx], y_train[trn_idx])
        val_prob = leaderboard_model.predict_proba(X_train[val_idx])[:, 1]
        test_prob = leaderboard_model.predict_proba(X_test)[:, 1]
        oof[val_idx] = val_prob
        probs[:, fold - 1] = test_prob
        oob_mean += leaderboard_model.oob_score_ / N
        fold_rows.append(
            {
                "fold": fold,
                "accuracy": accuracy_score(y_train[val_idx], val_prob >= 0.5),
                "roc_auc": roc_auc_score(y_train[val_idx], val_prob),
                "oob_score": leaderboard_model.oob_score_,
            }
        )
        print(
            f"fold={fold} acc={fold_rows[-1]['accuracy']:.5f} "
            f"auc={fold_rows[-1]['roc_auc']:.5f} "
            f"oob={fold_rows[-1]['oob_score']:.5f}",
            flush=True,
        )

    test_probability = probs.mean(axis=1)
    test_pred = (test_probability >= 0.5).astype(int)

    submission = pd.DataFrame(
        {"PassengerId": passenger_ids.to_numpy(), "Survived": test_pred}
    )
    submission_path = SUBMISSION_DIR / "submission_v10_gunes_original_reproduction.csv"
    submission.to_csv(submission_path, index=False)

    pd.DataFrame(
        {
            "PassengerId": passenger_ids.to_numpy(),
            "probability": test_probability,
            "Survived": test_pred,
        }
    ).to_csv(EXPORT_DIR / "gunes_original_test_predictions.csv", index=False)
    pd.DataFrame(
        {"PassengerId": pd.read_csv(DATA_DIR / "train.csv")["PassengerId"], "Survived": y_train, "oof_probability": oof}
    ).to_csv(EXPORT_DIR / "gunes_original_oof.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(EXPORT_DIR / "gunes_original_fold_metrics.csv", index=False)

    metadata = {
        "source": "Gunes Evitan - Titanic Advanced Feature Engineering Tutorial",
        "source_url": "https://www.kaggle.com/code/gunesevitan/titanic-advanced-feature-engineering-tutorial",
        "historical_reported_public_score": 0.83732,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "model": {
            "n_estimators": 1750,
            "max_depth": 7,
            "min_samples_split": 6,
            "min_samples_leaf": 6,
            "max_features_original": "auto",
            "max_features_compatibility": "sqrt",
            "random_state": 42,
        },
        "cv": {"n_splits": 5, "shuffle": True, "random_state": 5},
        "local_oof_accuracy": float(accuracy_score(y_train, oof >= 0.5)),
        "local_oof_auc": float(roc_auc_score(y_train, oof)),
        "average_oob_score": float(oob_mean),
        "test_positive_count": int(test_pred.sum()),
        "compatibility_notes": [
            "Deprecated pandas syntax updated without intended semantic changes.",
            "RandomForest max_features='auto' replaced by equivalent 'sqrt' in current sklearn.",
        ],
    }
    with (EXPORT_DIR / "gunes_original_reproduction_metadata.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nOOF Accuracy:", metadata["local_oof_accuracy"])
    print("OOF AUC:", metadata["local_oof_auc"])
    print("Average OOB:", metadata["average_oob_score"])
    print("Test positives:", metadata["test_positive_count"])
    print("\nFirst 10 submission rows:")
    print(submission.head(10).to_string(index=False))
    print("\nSaved:", submission_path)


if __name__ == "__main__":
    main()
