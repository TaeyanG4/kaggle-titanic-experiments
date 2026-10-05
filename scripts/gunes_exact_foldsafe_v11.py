"""Fold-safe version of Gunes' exact 26-feature RandomForest representation.

This isolates one question:
How much of the public-strength Gunes pipeline comes from the representation/RF
versus the historical global target encoding?

Target-independent preprocessing may use train+test jointly, matching the
historical tutorial. Target-derived Family/Ticket survival rates are strictly:
  - fold-train row: peer labels in fold-train, self excluded
  - validation/test: fold-train labels only

Two documented RF configs are evaluated:
  - single_best
  - leaderboard

The script also compares simple 2-of-3 hard votes with the trusted v5 members.
No Kaggle submission is performed.
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
V5_DIR = BASE_DIR / "exports" / "v5"
EXPORT_DIR = BASE_DIR / "exports" / "v11"


RF_CONFIGS = {
    "single_best": dict(
        n_estimators=1100, max_depth=5, min_samples_split=4, min_samples_leaf=5
    ),
    "leaderboard": dict(
        n_estimators=1750, max_depth=7, min_samples_split=6, min_samples_leaf=6
    ),
}


def extract_surname(series: pd.Series) -> pd.Series:
    def one(name: str) -> str:
        name_no_bracket = name.split("(")[0] if "(" in name else name
        family = name_no_bracket.split(",")[0]
        for c in string.punctuation:
            family = family.replace(c, "").strip()
        return family
    return series.astype(str).map(one)


def structural_frames(train_raw: pd.DataFrame, test_raw: pd.DataFrame):
    tr = train_raw.copy()
    te = test_raw.copy()
    tr["_source"] = "train"
    te["_source"] = "test"
    df = pd.concat([tr, te], ignore_index=True, sort=True)

    # Historical target-independent preprocessing.
    age_med = df.groupby(["Sex", "Pclass"])["Age"].transform("median")
    df["Age"] = df["Age"].fillna(age_med)
    df["Embarked"] = df["Embarked"].fillna("S")
    med_fare = df.loc[
        (df["Pclass"] == 3) & (df["Parch"] == 0) & (df["SibSp"] == 0), "Fare"
    ].median()
    df["Fare"] = df["Fare"].fillna(med_fare)
    df["Deck"] = df["Cabin"].apply(lambda s: s[0] if pd.notna(s) else "M")
    df.loc[df["Deck"] == "T", "Deck"] = "A"
    df["Deck"] = df["Deck"].replace(["A", "B", "C"], "ABC")
    df["Deck"] = df["Deck"].replace(["D", "E"], "DE")
    df["Deck"] = df["Deck"].replace(["F", "G"], "FG")

    df["FareBin"] = pd.qcut(df["Fare"], 13)
    df["AgeBin"] = pd.qcut(df["Age"], 10)
    df["Family_Size"] = df["SibSp"] + df["Parch"] + 1
    family_map = {
        1: "Alone", 2: "Small", 3: "Small", 4: "Small",
        5: "Medium", 6: "Medium", 7: "Large", 8: "Large", 11: "Large",
    }
    df["Family_Size_Grouped"] = df["Family_Size"].map(family_map)
    df["Ticket_Frequency"] = df.groupby("Ticket")["Ticket"].transform("count")
    df["Title"] = (
        df["Name"].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    )
    df["Is_Married"] = (df["Title"] == "Mrs").astype(int)
    df["Title"] = df["Title"].replace(
        ["Miss", "Mrs", "Ms", "Mlle", "Lady", "Mme", "the Countess", "Dona"],
        "Miss/Mrs/Ms",
    )
    df["Title"] = df["Title"].replace(
        ["Dr", "Col", "Major", "Jonkheer", "Capt", "Sir", "Don", "Rev"],
        "Dr/Military/Noble/Clergy",
    )
    df["Family"] = extract_surname(df["Name"])

    # Encode Age/Fare bins ordinally on combined train+test.
    df["Age"] = LabelEncoder().fit_transform(df["AgeBin"])
    df["Fare"] = LabelEncoder().fit_transform(df["FareBin"])

    # One-hot categoricals on combined train+test for stable fold columns.
    cat_features = [
        "Pclass", "Sex", "Deck", "Embarked", "Title", "Family_Size_Grouped"
    ]
    encoded_parts = []
    encoded_names = []
    for feature in cat_features:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        arr = enc.fit_transform(df[[feature]])
        cols = [f"{feature}_{i+1}" for i in range(arr.shape[1])]
        encoded_parts.append(pd.DataFrame(arr, columns=cols, index=df.index))
        encoded_names.extend(cols)
    df = pd.concat([df, *encoded_parts], axis=1)

    base_feature_names = ["Age", "Fare", "Ticket_Frequency", "Is_Married"]
    base_feature_names += encoded_names
    if len(base_feature_names) != 24:
        raise ValueError(f"Expected 24 target-independent features, got {len(base_feature_names)}")

    train_df = df[df["_source"] == "train"].copy().reset_index(drop=True)
    test_df = df[df["_source"] == "test"].copy().reset_index(drop=True)
    train_df["Survived"] = train_df["Survived"].astype(int)

    # Historical eligibility: group occurs in both train and test and size > 1.
    test_families = set(test_df["Family"])
    test_tickets = set(test_df["Ticket"])
    family_size_med = train_df.groupby("Family")["Family_Size"].median()
    ticket_freq_med = train_df.groupby("Ticket")["Ticket_Frequency"].median()
    allowed_families = {
        g for g, size in family_size_med.items() if g in test_families and size > 1
    }
    allowed_tickets = {
        g for g, freq in ticket_freq_med.items() if g in test_tickets and freq > 1
    }
    return train_df, test_df, base_feature_names, allowed_families, allowed_tickets


def median_peer_rate(
    ref: pd.DataFrame,
    app: pd.DataFrame,
    group_col: str,
    allowed_groups: set,
    exclude_self: bool,
    fallback: float,
):
    grouped = ref.groupby(group_col, sort=False)
    rates = np.full(len(app), fallback, dtype=float)
    avail = np.zeros(len(app), dtype=float)
    for j, (_, row) in enumerate(app.iterrows()):
        key = row[group_col]
        if key not in allowed_groups or key not in grouped.groups:
            continue
        peers = ref.loc[grouped.groups[key]]
        if exclude_self:
            peers = peers[peers["PassengerId"] != row["PassengerId"]]
        if len(peers) == 0:
            continue
        rates[j] = float(peers["Survived"].median())
        avail[j] = 1.0
    return rates, avail


def attach_target_features(
    ref: pd.DataFrame,
    app: pd.DataFrame,
    allowed_families: set,
    allowed_tickets: set,
    *,
    exclude_self: bool,
):
    fallback = float(ref["Survived"].mean())
    fam_rate, fam_na = median_peer_rate(
        ref, app, "Family", allowed_families, exclude_self, fallback
    )
    tic_rate, tic_na = median_peer_rate(
        ref, app, "Ticket", allowed_tickets, exclude_self, fallback
    )
    return np.column_stack(
        [
            (fam_rate + tic_rate) / 2.0,
            (fam_na + tic_na) / 2.0,
        ]
    )


def make_rf(cfg: dict):
    return RandomForestClassifier(
        criterion="gini",
        **cfg,
        max_features="sqrt",
        oob_score=True,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    train_df, test_df, base_cols, allowed_families, allowed_tickets = structural_frames(
        train_raw, test_raw
    )
    y = train_df["Survived"].to_numpy()
    skf = StratifiedKFold(n_splits=5, random_state=5, shuffle=True)
    splits = list(skf.split(train_df, y))

    oof_export = pd.DataFrame(
        {"PassengerId": train_df["PassengerId"].astype(int), "Survived": y}
    )
    test_export = pd.DataFrame({"PassengerId": test_df["PassengerId"].astype(int)})
    summary_rows = []
    fold_rows = []

    for name, cfg in RF_CONFIGS.items():
        oof = np.zeros(len(train_df), dtype=float)
        test_probs = []
        for fold, (tr_idx, va_idx) in enumerate(splits, 1):
            tr = train_df.iloc[tr_idx].copy()
            va = train_df.iloc[va_idx].copy()

            tr_target = attach_target_features(
                tr, tr, allowed_families, allowed_tickets, exclude_self=True
            )
            va_target = attach_target_features(
                tr, va, allowed_families, allowed_tickets, exclude_self=False
            )
            te_target = attach_target_features(
                tr, test_df, allowed_families, allowed_tickets, exclude_self=False
            )

            Xtr = np.column_stack([tr[base_cols].to_numpy(dtype=float), tr_target])
            Xva = np.column_stack([va[base_cols].to_numpy(dtype=float), va_target])
            Xte = np.column_stack([test_df[base_cols].to_numpy(dtype=float), te_target])

            scaler = StandardScaler()
            Xtr = scaler.fit_transform(Xtr)
            Xva = scaler.transform(Xva)
            Xte = scaler.transform(Xte)

            model = make_rf(cfg)
            model.fit(Xtr, y[tr_idx])
            va_prob = model.predict_proba(Xva)[:, 1]
            te_prob = model.predict_proba(Xte)[:, 1]
            oof[va_idx] = va_prob
            test_probs.append(te_prob)
            fold_rows.append(
                {
                    "candidate": name,
                    "fold": fold,
                    "accuracy": accuracy_score(y[va_idx], va_prob >= 0.5),
                    "roc_auc": roc_auc_score(y[va_idx], va_prob),
                    "oob_score": model.oob_score_,
                }
            )

        test_prob = np.mean(test_probs, axis=0)
        oof_export[name] = oof
        test_export[name] = test_prob
        candidate_folds = pd.DataFrame(
            [r for r in fold_rows if r["candidate"] == name]
        )
        summary_rows.append(
            {
                "candidate": name,
                "accuracy": accuracy_score(y, oof >= 0.5),
                "roc_auc": roc_auc_score(y, oof),
                "fold_accuracy_std": candidate_folds["accuracy"].std(ddof=0),
                "fold_auc_std": candidate_folds["roc_auc"].std(ddof=0),
                "test_positive_count": int((test_prob >= 0.5).sum()),
            }
        )

    # Trusted v5 robust reconstruction.
    zoo = pd.read_csv(V5_DIR / "model_zoo_oof.csv")
    zoo_test = pd.read_csv(V5_DIR / "model_zoo_test.csv")
    mlp = pd.read_csv(V5_DIR / "seed_probes" / "MLP_PLR_seed_mean__oof.csv")
    mlp_test = pd.read_csv(V5_DIR / "seed_probes" / "MLP_PLR_seed_mean__test.csv")
    v4b = (zoo["v4b__Champion"].to_numpy() > 0.5).astype(int)
    rule = (zoo["RuleFit"].to_numpy() > 0.5).astype(int)
    mlpv = (mlp["probability"].to_numpy() > 0.5).astype(int)
    v5_pred = ((v4b + rule + mlpv) >= 2).astype(int)
    v5t = (
        (zoo_test["v4b__Champion"].to_numpy() > 0.5).astype(int)
        + (zoo_test["RuleFit"].to_numpy() > 0.5).astype(int)
        + (mlp_test["probability"].to_numpy() > 0.5).astype(int)
    )
    v5_test_pred = (v5t >= 2).astype(int)

    ensemble_rows = []
    # Swap each of the three v5 members with fold-safe Gunes RF vote.
    for rf_name in RF_CONFIGS:
        rf_vote = (oof_export[rf_name].to_numpy() >= 0.5).astype(int)
        rf_test_vote = (test_export[rf_name].to_numpy() >= 0.5).astype(int)
        candidates = {
            f"v4b_rule_{rf_name}": ((v4b + rule + rf_vote) >= 2).astype(int),
            f"v4b_mlp_{rf_name}": ((v4b + mlpv + rf_vote) >= 2).astype(int),
            f"rule_mlp_{rf_name}": ((rule + mlpv + rf_vote) >= 2).astype(int),
        }
        test_candidates = {
            f"v4b_rule_{rf_name}": (
                (
                    (zoo_test["v4b__Champion"].to_numpy() > 0.5).astype(int)
                    + (zoo_test["RuleFit"].to_numpy() > 0.5).astype(int)
                    + rf_test_vote
                )
                >= 2
            ).astype(int),
            f"v4b_mlp_{rf_name}": (
                (
                    (zoo_test["v4b__Champion"].to_numpy() > 0.5).astype(int)
                    + (mlp_test["probability"].to_numpy() > 0.5).astype(int)
                    + rf_test_vote
                )
                >= 2
            ).astype(int),
            f"rule_mlp_{rf_name}": (
                (
                    (zoo_test["RuleFit"].to_numpy() > 0.5).astype(int)
                    + (mlp_test["probability"].to_numpy() > 0.5).astype(int)
                    + rf_test_vote
                )
                >= 2
            ).astype(int),
        }
        for ens_name, pred in candidates.items():
            ensemble_rows.append(
                {
                    "ensemble": ens_name,
                    "accuracy": accuracy_score(y, pred),
                    "delta_vs_v5": accuracy_score(y, pred) - accuracy_score(y, v5_pred),
                    "binary_disagreement_vs_v5": int(np.sum(pred != v5_pred)),
                    "test_disagreement_vs_v5": int(
                        np.sum(test_candidates[ens_name] != v5_test_pred)
                    ),
                }
            )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["accuracy", "roc_auc"], ascending=False
    )
    ensemble = pd.DataFrame(ensemble_rows).sort_values(
        ["accuracy", "binary_disagreement_vs_v5"], ascending=[False, True]
    )
    summary.to_csv(EXPORT_DIR / "gunes_exact_foldsafe_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(
        EXPORT_DIR / "gunes_exact_foldsafe_fold_metrics.csv", index=False
    )
    oof_export.to_csv(EXPORT_DIR / "gunes_exact_foldsafe_oof.csv", index=False)
    test_export.to_csv(EXPORT_DIR / "gunes_exact_foldsafe_test.csv", index=False)
    ensemble.to_csv(EXPORT_DIR / "gunes_exact_foldsafe_ensemble.csv", index=False)
    with (EXPORT_DIR / "gunes_exact_foldsafe_metadata.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "base_feature_count": len(base_cols),
                "final_feature_count": len(base_cols) + 2,
                "rf_configs": RF_CONFIGS,
                "target_encoding": "fold-safe peer median; train self-excluded",
                "historical_test_group_eligibility_retained": True,
                "no_submission": True,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=== Exact Gunes 26-feature fold-safe RF ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n=== Hard-vote replacements vs v5 robust ===")
    print(ensemble.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nNo Kaggle submission was performed.")


if __name__ == "__main__":
    main()
