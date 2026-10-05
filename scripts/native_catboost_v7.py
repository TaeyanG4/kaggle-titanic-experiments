"""Titanic v7 native-categorical CatBoost representation audit.

Goal
----
The current v1-v6 pipeline one-hot encodes low/medium-cardinality categoricals
before CatBoost sees them. v7 asks whether CatBoost performs better when it gets
the categorical structure directly, including selected relational keys.

All target-derived group features are fold-safe. The exact saved 5-fold
manifest from the trusted v1 audit is reused.

No Kaggle submission is performed.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

try:
    from audit_group_survival import add_group_survival
except ModuleNotFoundError:
    from scripts.audit_group_survival import add_group_survival


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
V1_AUDIT = BASE_DIR / "exports" / "wcg_audit_v1"
EXPORT_DIR = BASE_DIR / "exports" / "v7"


TITLE_MAP = {
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


def normalize_ticket_prefix(ticket: str) -> str:
    value = str(ticket).upper().strip()
    prefix = re.sub(r"\d", "", value)
    prefix = re.sub(r"[\s./]+", "", prefix)
    return prefix or "NONE"


def build_raw_frame(train_raw: pd.DataFrame, test_raw: pd.DataFrame):
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
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["FamilyType"] = pd.cut(
        df["FamilySize"],
        bins=[0, 1, 4, 6, 20],
        labels=["Alone", "Small", "Medium", "Large"],
    ).astype(str)
    df["Deck"] = df["Cabin"].fillna("U").astype(str).str[0]
    df["Embarked"] = df["Embarked"].fillna("S").astype(str)
    df["AgeMissing"] = df["Age"].isna().astype(int)
    df["CabinKnown"] = df["Cabin"].notna().astype(int)
    df["CabinCount"] = (
        df["Cabin"].fillna("").astype(str).str.split().str.len().where(df["Cabin"].notna(), 0)
    ).astype(float)
    df["NameLength"] = df["Name"].astype(str).str.len().astype(float)

    fare_median = df.groupby(["Pclass", "Embarked"])["Fare"].transform("median")
    df["Fare"] = df["Fare"].fillna(fare_median).fillna(df["Fare"].median())
    df["LogFare"] = np.log1p(df["Fare"])

    age_median = df.groupby(["Title", "Pclass"])["Age"].transform("median")
    df["Age"] = df["Age"].fillna(age_median).fillna(df["Age"].median())

    ticket_counts = df["Ticket"].astype(str).value_counts()
    df["TicketFreq"] = df["Ticket"].astype(str).map(ticket_counts).astype(float)
    df["FarePerPerson"] = df["Fare"] / df["TicketFreq"].clip(lower=1)
    df["LogFarePerPerson"] = np.log1p(df["FarePerPerson"])
    df["FarePerFamily"] = df["Fare"] / df["FamilySize"].clip(lower=1)

    df["Ticket"] = df["Ticket"].fillna("UNKNOWN").astype(str)
    df["TicketPrefix"] = df["Ticket"].map(normalize_ticket_prefix)
    df["FamilyGroup"] = df["LastName"] + "_" + df["FamilySize"].astype(str)
    df["FamilyFareKey"] = df["LastName"] + "_" + df["Fare"].round(4).astype(str)
    # Missing cabins are not treated as one shared cabin group.
    df["CabinKey"] = df["Cabin"].fillna(
        "__MISSING_" + df["PassengerId"].astype(str)
    ).astype(str)

    df["IsMarriedWoman"] = (df["Title"] == "Mrs").astype(int)
    df["IsChild"] = (df["Age"] <= 12).astype(int)
    df["AdultMale"] = ((df["Sex"] == "male") & (df["Age"] >= 18)).astype(int)
    df["IsWomanOrChild"] = (
        (df["Sex"] == "female") | (df["Title"] == "Master") | (df["Age"] <= 12)
    ).astype(int)
    df["Age_Pclass"] = df["Age"] * df["Pclass"]
    df["GroupSurvival"] = 0.5

    categorical = [
        "Sex",
        "Embarked",
        "Title",
        "Deck",
        "FamilyType",
        "TicketPrefix",
        "LastName",
        "Ticket",
        "FamilyGroup",
        "FamilyFareKey",
        "CabinKey",
    ]
    for c in categorical:
        df[c] = df[c].fillna("UNKNOWN").astype(str)

    train_df = df[df["_source"] == "train"].copy().reset_index(drop=True)
    test_df = df[df["_source"] == "test"].copy().reset_index(drop=True)
    train_df["Survived"] = train_df["Survived"].astype(int)
    return train_df, test_df


def peer_stats(
    reference: pd.DataFrame,
    apply_df: pd.DataFrame,
    *,
    group_col: str,
    exclude_self: bool,
) -> pd.DataFrame:
    groups = reference.groupby(group_col, sort=False)
    rows = []
    for _, row in apply_df.iterrows():
        key = row[group_col]
        if key not in groups.groups:
            peers = reference.iloc[0:0]
        else:
            peers = reference.loc[groups.groups[key]]
            if exclude_self:
                peers = peers[peers["PassengerId"] != row["PassengerId"]]
        n = len(peers)
        if n == 0:
            rows.append((0.5, 0.5, 0.5, 0.0))
        else:
            s = float(peers["Survived"].sum())
            m = float(peers["Survived"].mean())
            rows.append((float(s > 0), m, (s + 1.0) / (n + 2.0), float(n)))
    return pd.DataFrame(rows, columns=["Any", "Mean", "Smooth", "Count"], index=apply_df.index)


NUMERIC_BASE = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "LogFare",
    "FamilySize",
    "IsAlone",
    "TicketFreq",
    "FarePerPerson",
    "LogFarePerPerson",
    "FarePerFamily",
    "Age_Pclass",
    "IsChild",
    "IsMarriedWoman",
    "AdultMale",
    "AgeMissing",
    "CabinKnown",
    "CabinCount",
    "NameLength",
    "GroupSurvival",
]

CORE_CATS = ["Sex", "Embarked", "Title", "Deck", "FamilyType", "TicketPrefix"]

VARIANTS = {
    "native_core": {
        "cats": CORE_CATS,
        "peer_group": None,
    },
    "native_lastname": {
        "cats": CORE_CATS + ["LastName"],
        "peer_group": None,
    },
    "native_ticket": {
        "cats": CORE_CATS + ["Ticket"],
        "peer_group": None,
    },
    "native_familyfare_key": {
        "cats": CORE_CATS + ["FamilyFareKey"],
        "peer_group": None,
    },
    "native_relational": {
        "cats": CORE_CATS + ["LastName", "Ticket", "FamilyGroup", "FamilyFareKey"],
        "peer_group": None,
    },
    "native_relational_fftarget": {
        "cats": CORE_CATS + ["LastName", "Ticket", "FamilyGroup", "FamilyFareKey"],
        "peer_group": "FamilyFareKey",
    },
    "native_relational_cabin": {
        "cats": CORE_CATS + ["LastName", "Ticket", "FamilyGroup", "FamilyFareKey", "CabinKey"],
        "peer_group": None,
    },
}


def make_model(seed: int, params: dict | None = None):
    cfg = dict(
        iterations=260,
        depth=4,
        learning_rate=0.03,
        l2_leaf_reg=3.0,
        random_seed=seed,
        verbose=0,
        thread_count=-1,
        loss_function="Logloss",
        allow_writing_files=False,
    )
    if params:
        cfg.update(params)
    return CatBoostClassifier(**cfg)


def run_variant(
    name: str,
    spec: dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    folds: np.ndarray,
    *,
    seed: int = 42,
    params: dict | None = None,
):
    y = train_df["Survived"].astype(int).to_numpy()
    oof = np.zeros(len(train_df), dtype=float)
    test_probs = []
    fold_rows = []
    peer_cols = []

    for fold in sorted(np.unique(folds)):
        tr_idx = np.flatnonzero(folds != fold)
        va_idx = np.flatnonzero(folds == fold)
        tr = train_df.iloc[tr_idx].copy()
        va = train_df.iloc[va_idx].copy()
        te = test_df.copy()

        tr["GroupSurvival"] = add_group_survival(tr, tr, exclude_self=True).to_numpy()
        va["GroupSurvival"] = add_group_survival(tr, va, exclude_self=False).to_numpy()
        te["GroupSurvival"] = add_group_survival(tr, te, exclude_self=False).to_numpy()

        peer_cols = []
        if spec["peer_group"]:
            for target_df, exclude_self, prefix in [
                (tr, True, "train"),
                (va, False, "val"),
                (te, False, "test"),
            ]:
                stats = peer_stats(
                    tr,
                    target_df,
                    group_col=spec["peer_group"],
                    exclude_self=exclude_self,
                )
                for stat in ["Any", "Mean", "Smooth", "Count"]:
                    col = f"FamilyFarePeer{stat}"
                    target_df[col] = stats[stat].to_numpy()
                    if col not in peer_cols:
                        peer_cols.append(col)

        features = NUMERIC_BASE + spec["cats"] + peer_cols
        cat_features = spec["cats"]
        model = make_model(seed + int(fold), params=params)
        started = time.time()
        model.fit(
            tr[features],
            y[tr_idx],
            cat_features=cat_features,
        )
        val_prob = model.predict_proba(va[features])[:, 1]
        test_prob = model.predict_proba(te[features])[:, 1]
        elapsed = time.time() - started
        oof[va_idx] = val_prob
        test_probs.append(test_prob)
        fold_rows.append(
            {
                "variant": name,
                "fold": int(fold),
                "accuracy": accuracy_score(y[va_idx], val_prob > 0.5),
                "roc_auc": roc_auc_score(y[va_idx], val_prob),
                "seconds": elapsed,
                "n_features": len(features),
                "n_cat_features": len(cat_features),
            }
        )

    test_prob = np.mean(test_probs, axis=0)
    return oof, test_prob, pd.DataFrame(fold_rows)


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    train_df, test_df = build_raw_frame(train_raw, test_raw)
    manifest = pd.read_csv(V1_AUDIT / "fold_manifest_seed42.csv")
    if not train_df["PassengerId"].equals(manifest["PassengerId"]):
        raise ValueError("Fold manifest PassengerId order mismatch.")
    folds = manifest["fold"].astype(int).to_numpy()
    y = train_df["Survived"].astype(int).to_numpy()

    oof_export = pd.DataFrame(
        {"PassengerId": train_df["PassengerId"], "fold": folds, "Survived": y}
    )
    test_export = pd.DataFrame({"PassengerId": test_df["PassengerId"]})
    summary_rows = []
    all_folds = []

    for name, spec in VARIANTS.items():
        print(f"\n=== {name} ===", flush=True)
        started = time.time()
        oof, test_prob, fold_df = run_variant(
            name, spec, train_df, test_df, folds, seed=42
        )
        all_folds.append(fold_df)
        oof_export[name] = oof
        test_export[name] = test_prob
        summary_rows.append(
            {
                "variant": name,
                "accuracy": accuracy_score(y, oof > 0.5),
                "roc_auc": roc_auc_score(y, oof),
                "fold_accuracy_std": fold_df["accuracy"].std(ddof=0),
                "fold_auc_std": fold_df["roc_auc"].std(ddof=0),
                "elapsed_seconds": time.time() - started,
                "n_features": int(fold_df["n_features"].iloc[0]),
                "n_cat_features": int(fold_df["n_cat_features"].iloc[0]),
            }
        )
        print(
            f"accuracy={summary_rows[-1]['accuracy']:.5f} "
            f"auc={summary_rows[-1]['roc_auc']:.5f} "
            f"fold_std={summary_rows[-1]['fold_accuracy_std']:.5f}",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["accuracy", "roc_auc"], ascending=False
    )
    summary.to_csv(EXPORT_DIR / "native_catboost_summary.csv", index=False)
    pd.concat(all_folds, ignore_index=True).to_csv(
        EXPORT_DIR / "native_catboost_fold_metrics.csv", index=False
    )
    oof_export.to_csv(EXPORT_DIR / "native_catboost_oof.csv", index=False)
    test_export.to_csv(EXPORT_DIR / "native_catboost_test.csv", index=False)
    with (EXPORT_DIR / "native_catboost_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "variants": VARIANTS,
                "numeric_features": NUMERIC_BASE,
                "seed": 42,
                "fold_manifest": str(V1_AUDIT / "fold_manifest_seed42.csv"),
                "notes": [
                    "All target-derived GroupSurvival and FamilyFare peer stats are fold-safe.",
                    "No Kaggle submission is performed.",
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== v7 native CatBoost ranking ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
