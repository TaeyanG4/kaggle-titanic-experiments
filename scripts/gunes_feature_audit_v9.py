"""Fold-safe audit of ideas from Gunes Evitan's Titanic FE tutorial.

The public tutorial is used as a hypothesis source, not copied as a validation
recipe. In particular, its target-derived family/ticket survival rates are
rebuilt fold-safely here:

- fold-train rows: peer labels only, self excluded
- fold-validation/test rows: labels from fold-train only

Target-independent ideas tested:
- Sex x Pclass Age imputation as an alternate Age representation
- 10-quantile Age bin and 13-quantile Fare bin
- grouped Deck: ABC / DE / FG / M
- coarser title grouping used by the tutorial

Target-derived ideas tested:
- surname peer survival rate
- ticket peer survival rate
- averaged SurvivalRate plus availability indicator

All experiments reuse the trusted seed-42 fold manifest. No Kaggle submission.
"""

from __future__ import annotations

import json
import string
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

try:
    from audit_group_survival import add_group_survival, build_base_frame, model_feature_columns
    from feature_ablation_v6 import build_models
except ModuleNotFoundError:
    from scripts.audit_group_survival import add_group_survival, build_base_frame, model_feature_columns
    from scripts.feature_ablation_v6 import build_models


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
V1_AUDIT = BASE_DIR / "exports" / "wcg_audit_v1"
V5_DIR = BASE_DIR / "exports" / "v5"
EXPORT_DIR = BASE_DIR / "exports" / "v9"
MODEL_NAMES = ["GradientBoosting", "XGBoost", "CatBoost"]


def extract_family(name: pd.Series) -> pd.Series:
    def one(value: str) -> str:
        s = str(value)
        if "(" in s:
            s = s.split("(")[0]
        family = s.split(",")[0]
        for c in string.punctuation:
            family = family.replace(c, "")
        return family.strip()
    return name.map(one)


def build_gunes_structural(train_raw: pd.DataFrame, test_raw: pd.DataFrame):
    train = train_raw.copy()
    test = test_raw.copy()
    train["_source"] = "train"
    test["_source"] = "test"
    df = pd.concat([train, test], ignore_index=True, sort=False)

    # Alternate Age representation: Sex x Pclass median, as in the tutorial.
    age_gunes = df["Age"].copy()
    age_med = df.groupby(["Sex", "Pclass"])["Age"].transform("median")
    age_gunes = age_gunes.fillna(age_med).fillna(df["Age"].median())
    df["AgeSexPclass_Gunes"] = age_gunes

    # Fare handling follows the tutorial's logic as closely as possible.
    fare_gunes = df["Fare"].copy()
    fare_med = df.groupby(["Pclass", "Parch", "SibSp"])["Fare"].transform("median")
    fare_gunes = fare_gunes.fillna(fare_med).fillna(df["Fare"].median())

    # qcut is target-independent. Codes preserve the tutorial's ordinal-bin idea.
    df["AgeBinQ10_Gunes"] = pd.qcut(
        age_gunes, q=10, labels=False, duplicates="drop"
    ).astype(float)
    df["FareBinQ13_Gunes"] = pd.qcut(
        fare_gunes, q=13, labels=False, duplicates="drop"
    ).astype(float)

    # Deck grouping: T -> A, ABC / DE / FG / M.
    deck = df["Cabin"].apply(lambda x: str(x)[0] if pd.notna(x) else "M")
    deck = deck.replace({"T": "A"})
    deck = deck.replace({"A": "ABC", "B": "ABC", "C": "ABC",
                         "D": "DE", "E": "DE", "F": "FG", "G": "FG"})
    df["DeckGroup_Gunes"] = deck

    # Coarser title grouping from the tutorial.
    title = df["Name"].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
    title = title.replace(
        ["Miss", "Mrs", "Ms", "Mlle", "Lady", "Mme", "the Countess", "Dona"],
        "Miss/Mrs/Ms",
    )
    title = title.replace(
        ["Dr", "Col", "Major", "Jonkheer", "Capt", "Sir", "Don", "Rev"],
        "Dr/Military/Noble/Clergy",
    )
    df["TitleGroup_Gunes"] = title.fillna("Other")

    df["Family_Gunes"] = extract_family(df["Name"])

    structural = pd.get_dummies(
        df[[
            "PassengerId",
            "_source",
            "AgeSexPclass_Gunes",
            "AgeBinQ10_Gunes",
            "FareBinQ13_Gunes",
            "DeckGroup_Gunes",
            "TitleGroup_Gunes",
            "Family_Gunes",
        ]],
        columns=["DeckGroup_Gunes", "TitleGroup_Gunes"],
        drop_first=False,
        dtype=int,
    )

    train_s = structural[structural["_source"] == "train"].drop(columns="_source").reset_index(drop=True)
    test_s = structural[structural["_source"] == "test"].drop(columns="_source").reset_index(drop=True)
    return train_s, test_s


def peer_rate(
    reference: pd.DataFrame,
    apply_df: pd.DataFrame,
    *,
    group_col: str,
    exclude_self: bool,
    fallback: float,
) -> tuple[np.ndarray, np.ndarray]:
    groups = reference.groupby(group_col, sort=False)
    rates = []
    available = []
    for _, row in apply_df.iterrows():
        key = row[group_col]
        if key not in groups.groups:
            peers = reference.iloc[0:0]
        else:
            peers = reference.loc[groups.groups[key]]
            if exclude_self:
                peers = peers[peers["PassengerId"] != row["PassengerId"]]
        if len(peers) == 0:
            rates.append(float(fallback))
            available.append(0.0)
        else:
            # Original tutorial used median of binary Survived per group.
            rates.append(float(peers["Survived"].median()))
            available.append(1.0)
    return np.asarray(rates), np.asarray(available)


def add_gunes_target_features(
    reference: pd.DataFrame,
    apply_df: pd.DataFrame,
    *,
    exclude_self: bool,
) -> pd.DataFrame:
    fallback = float(reference["Survived"].mean())
    fam_rate, fam_avail = peer_rate(
        reference,
        apply_df,
        group_col="Family_Gunes",
        exclude_self=exclude_self,
        fallback=fallback,
    )
    tic_rate, tic_avail = peer_rate(
        reference,
        apply_df,
        group_col="Ticket",
        exclude_self=exclude_self,
        fallback=fallback,
    )
    out = pd.DataFrame(index=apply_df.index)
    out["GunesFamilyRate"] = fam_rate
    out["GunesFamilyRateAvailable"] = fam_avail
    out["GunesTicketRate"] = tic_rate
    out["GunesTicketRateAvailable"] = tic_avail
    out["GunesSurvivalRate"] = (fam_rate + tic_rate) / 2.0
    out["GunesSurvivalRateAvailable"] = (fam_avail + tic_avail) / 2.0
    return out


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    train_base, test_base = build_base_frame(train_raw, test_raw, profile="v2")
    train_s, test_s = build_gunes_structural(train_raw, test_raw)

    # Family_Gunes is a target-encoding helper only, not a raw model feature.
    train_base = train_base.merge(
        train_s[["PassengerId", "Family_Gunes"]], on="PassengerId", how="left"
    )
    test_base = test_base.merge(
        test_s[["PassengerId", "Family_Gunes"]], on="PassengerId", how="left"
    )

    base_cols = model_feature_columns(train_base.drop(columns=["Family_Gunes"]))
    manifest = pd.read_csv(V1_AUDIT / "fold_manifest_seed42.csv")
    if not train_base["PassengerId"].equals(manifest["PassengerId"]):
        raise ValueError("PassengerId order mismatch vs trusted folds.")
    folds = manifest["fold"].astype(int).to_numpy()
    y = train_base["Survived"].astype(int).to_numpy()

    structural_cols_all = [
        c for c in train_s.columns
        if c not in {"PassengerId", "Family_Gunes"}
    ]
    agefare_cols = [
        "AgeSexPclass_Gunes", "AgeBinQ10_Gunes", "FareBinQ13_Gunes"
    ]
    deck_cols = [c for c in structural_cols_all if c.startswith("DeckGroup_Gunes_")]
    title_cols = [c for c in structural_cols_all if c.startswith("TitleGroup_Gunes_")]
    structural_cols = agefare_cols + deck_cols + title_cols
    exact_rate_cols = ["GunesSurvivalRate", "GunesSurvivalRateAvailable"]
    component_rate_cols = [
        "GunesFamilyRate", "GunesFamilyRateAvailable",
        "GunesTicketRate", "GunesTicketRateAvailable",
        "GunesSurvivalRate", "GunesSurvivalRateAvailable",
    ]

    variants = {
        "baseline": ([], []),
        "gunes_agefare_bins": (agefare_cols, []),
        "gunes_deck_group": (deck_cols, []),
        "gunes_title_group": (title_cols, []),
        "gunes_structural": (structural_cols, []),
        "gunes_survival_rate_exact": ([], exact_rate_cols),
        "gunes_survival_rate_components": ([], component_rate_cols),
        "gunes_all": (structural_cols, component_rate_cols),
    }

    train_struct = train_s.drop(columns=["Family_Gunes"])
    test_struct = test_s.drop(columns=["Family_Gunes"])
    train_base = train_base.merge(train_struct, on="PassengerId", how="left")
    test_base = test_base.merge(test_struct, on="PassengerId", how="left")

    result_rows = []
    fold_rows = []
    oof_export = pd.DataFrame(
        {"PassengerId": train_base["PassengerId"], "fold": folds, "Survived": y}
    )
    test_export = pd.DataFrame({"PassengerId": test_base["PassengerId"]})

    for variant, (struct_cols, target_cols) in variants.items():
        print(f"\n=== {variant} ===", flush=True)
        oof = {m: np.zeros(len(train_base), dtype=float) for m in MODEL_NAMES}
        test_probs = {m: [] for m in MODEL_NAMES}

        for fold in sorted(np.unique(folds)):
            tr_idx = np.flatnonzero(folds != fold)
            va_idx = np.flatnonzero(folds == fold)
            tr = train_base.iloc[tr_idx].copy()
            va = train_base.iloc[va_idx].copy()
            te = test_base.copy()

            tr["GroupSurvival"] = add_group_survival(tr, tr, exclude_self=True).to_numpy()
            va["GroupSurvival"] = add_group_survival(tr, va, exclude_self=False).to_numpy()
            te["GroupSurvival"] = add_group_survival(tr, te, exclude_self=False).to_numpy()

            if target_cols:
                tr_t = add_gunes_target_features(tr, tr, exclude_self=True)
                va_t = add_gunes_target_features(tr, va, exclude_self=False)
                te_t = add_gunes_target_features(tr, te, exclude_self=False)
                for c in target_cols:
                    tr[c] = tr_t[c].to_numpy()
                    va[c] = va_t[c].to_numpy()
                    te[c] = te_t[c].to_numpy()

            features = base_cols + struct_cols + target_cols
            for model_name in MODEL_NAMES:
                model = build_models(42 + int(fold))[model_name]
                model.fit(tr[features], y[tr_idx])
                va_prob = np.asarray(model.predict_proba(va[features]))[:, 1]
                te_prob = np.asarray(model.predict_proba(te[features]))[:, 1]
                oof[model_name][va_idx] = va_prob
                test_probs[model_name].append(te_prob)
                fold_rows.append(
                    {
                        "variant": variant,
                        "model": model_name,
                        "fold": int(fold),
                        "accuracy": accuracy_score(y[va_idx], va_prob > 0.5),
                        "roc_auc": roc_auc_score(y[va_idx], va_prob),
                        "feature_count": len(features),
                    }
                )

        panel = np.mean([oof[m] for m in MODEL_NAMES], axis=0)
        panel_test = np.mean(
            [np.mean(test_probs[m], axis=0) for m in MODEL_NAMES], axis=0
        )
        oof_export[f"{variant}__Panel"] = panel
        test_export[f"{variant}__Panel"] = panel_test
        result_rows.append(
            {
                "variant": variant,
                "model": "Panel",
                "accuracy": accuracy_score(y, panel > 0.5),
                "roc_auc": roc_auc_score(y, panel),
                "feature_count": len(base_cols + struct_cols + target_cols),
            }
        )
        for model_name in MODEL_NAMES:
            prob = oof[model_name]
            tprob = np.mean(test_probs[model_name], axis=0)
            oof_export[f"{variant}__{model_name}"] = prob
            test_export[f"{variant}__{model_name}"] = tprob
            result_rows.append(
                {
                    "variant": variant,
                    "model": model_name,
                    "accuracy": accuracy_score(y, prob > 0.5),
                    "roc_auc": roc_auc_score(y, prob),
                    "feature_count": len(base_cols + struct_cols + target_cols),
                }
            )
        print(
            f"panel acc={result_rows[-4]['accuracy']:.5f} "
            f"auc={result_rows[-4]['roc_auc']:.5f}",
            flush=True,
        )

    results = pd.DataFrame(result_rows)
    baseline = results[results["variant"] == "baseline"][
        ["model", "accuracy", "roc_auc"]
    ].rename(
        columns={"accuracy": "baseline_accuracy", "roc_auc": "baseline_roc_auc"}
    )
    results = results.merge(baseline, on="model", how="left")
    results["delta_accuracy"] = results["accuracy"] - results["baseline_accuracy"]
    results["delta_auc"] = results["roc_auc"] - results["baseline_roc_auc"]
    results = results.sort_values(["accuracy", "roc_auc"], ascending=False)

    # Diversity against current v5 robust champion for panel variants.
    zoo = pd.read_csv(V5_DIR / "model_zoo_oof.csv")
    mlp = pd.read_csv(V5_DIR / "seed_probes" / "MLP_PLR_seed_mean__oof.csv")
    champ_votes = (
        (zoo["v4b__Champion"].to_numpy() > 0.5).astype(int)
        + (zoo["RuleFit"].to_numpy() > 0.5).astype(int)
        + (mlp["probability"].to_numpy() > 0.5).astype(int)
    )
    champ_pred = (champ_votes >= 2).astype(int)
    champ_correct = champ_pred == y
    diversity_rows = []
    for variant in variants:
        prob = oof_export[f"{variant}__Panel"].to_numpy()
        pred = (prob > 0.5).astype(int)
        correct = pred == y
        diversity_rows.append(
            {
                "variant": variant,
                "panel_accuracy": accuracy_score(y, pred),
                "binary_disagreement_vs_v5": int(np.sum(pred != champ_pred)),
                "v5_wrong_variant_right": int(np.sum((~champ_correct) & correct)),
                "v5_right_variant_wrong": int(np.sum(champ_correct & (~correct))),
                "net_unique_correct_vs_v5": int(
                    np.sum((~champ_correct) & correct)
                    - np.sum(champ_correct & (~correct))
                ),
            }
        )

    results.to_csv(EXPORT_DIR / "gunes_feature_audit_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(
        EXPORT_DIR / "gunes_feature_audit_fold_metrics.csv", index=False
    )
    oof_export.to_csv(EXPORT_DIR / "gunes_feature_audit_oof.csv", index=False)
    test_export.to_csv(EXPORT_DIR / "gunes_feature_audit_test.csv", index=False)
    diversity = pd.DataFrame(diversity_rows).sort_values(
        ["panel_accuracy", "net_unique_correct_vs_v5"], ascending=False
    )
    diversity.to_csv(EXPORT_DIR / "gunes_feature_audit_diversity.csv", index=False)

    with (EXPORT_DIR / "research_notes.md").open("w", encoding="utf-8") as f:
        f.write(
            "# Gunes Evitan Titanic FE transfer audit\n\n"
            "Source: Titanic - Advanced Feature Engineering Tutorial.\n\n"
            "Ideas were reimplemented from scratch under the project's fixed "
            "fold-safe validation. Target-derived family/ticket rates never "
            "see validation labels.\n"
        )
    with (EXPORT_DIR / "gunes_feature_audit_metadata.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "source": "Gunes Evitan - Titanic Advanced Feature Engineering Tutorial",
                "source_url": "https://www.kaggle.com/code/gunesevitan/titanic-advanced-feature-engineering-tutorial",
                "variants": list(variants),
                "models": MODEL_NAMES,
                "fold_manifest": str(V1_AUDIT / "fold_manifest_seed42.csv"),
                "target_encoding_policy": "fold-safe peer labels only; self-excluded for fold-train",
                "no_submission": True,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== Gunes transfer audit: panel ranking ===")
    print(
        results[results["model"] == "Panel"]
        .sort_values(["accuracy", "roc_auc"], ascending=False)
        .to_string(index=False, float_format=lambda x: f"{x:.5f}")
    )
    print("\n=== Best individual rows ===")
    print(results.head(20).to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n=== Diversity vs v5 robust ===")
    print(diversity.to_string(index=False))
    print("\nNo Kaggle submission was performed.")


if __name__ == "__main__":
    main()
