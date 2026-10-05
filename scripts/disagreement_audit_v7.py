"""Describe where the current v5 robust champion is wrong and alternatives are right.

This is an analysis-only residual audit. It does not create rules or retrain a
model. The goal is to identify compact, mechanistic feature hypotheses for the
next preprocessing/feature-engineering cycle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
V5_DIR = BASE_DIR / "exports" / "v5"
V6_DIR = BASE_DIR / "exports" / "v6"
V7_DIR = BASE_DIR / "exports" / "v7"


def title_from_name(name: pd.Series) -> pd.Series:
    raw = name.str.extract(r" ([A-Za-z]+)\.", expand=False)
    mapping = {
        "Mr": "Mr", "Miss": "Miss", "Mlle": "Miss", "Ms": "Miss",
        "Mrs": "Mrs", "Mme": "Mrs", "Master": "Master",
        "Dr": "Officer", "Rev": "Officer", "Col": "Officer",
        "Major": "Officer", "Capt": "Officer",
        "Sir": "Royalty", "Don": "Royalty", "Countess": "Royalty",
        "Lady": "Royalty", "Dona": "Royalty", "Jonkheer": "Royalty",
    }
    return raw.map(mapping).fillna("Other")


def build_context(train_raw: pd.DataFrame) -> pd.DataFrame:
    d = train_raw.copy()
    d["Title"] = title_from_name(d["Name"])
    d["LastName"] = d["Name"].str.split(",").str[0].str.strip()
    d["FamilySize"] = d["SibSp"] + d["Parch"] + 1
    d["IsAlone"] = (d["FamilySize"] == 1).astype(int)
    d["Deck"] = d["Cabin"].fillna("U").astype(str).str[0]
    d["CabinKnown"] = d["Cabin"].notna().astype(int)
    d["AgeMissing"] = d["Age"].isna().astype(int)
    d["AdultMaleRaw"] = ((d["Sex"] == "male") & (d["Age"].fillna(99) >= 18)).astype(int)
    d["WomanOrMaster"] = ((d["Sex"] == "female") | (d["Title"] == "Master")).astype(int)
    d["FamilyBand"] = pd.cut(
        d["FamilySize"],
        bins=[0, 1, 4, 6, 20],
        labels=["Alone", "Small", "Medium", "Large"],
    ).astype(str)
    d["FareBandAudit"] = pd.qcut(d["Fare"], 5, duplicates="drop").astype(str)
    return d


def main() -> None:
    V7_DIR.mkdir(parents=True, exist_ok=True)
    raw = build_context(pd.read_csv(DATA_DIR / "train.csv"))
    zoo = pd.read_csv(V5_DIR / "model_zoo_oof.csv")
    mlpmean = pd.read_csv(V5_DIR / "seed_probes" / "MLP_PLR_seed_mean__oof.csv")
    group = pd.read_csv(V6_DIR / "group_key_audit_oof.csv")
    cat = pd.read_csv(V7_DIR / "catboost_hpo_oof_seed42.csv")

    base = zoo[["PassengerId", "fold", "Survived", "v4b__Champion", "RuleFit", "TabPFN_v2"]].copy()
    base = base.merge(
        mlpmean[["PassengerId", "probability"]].rename(columns={"probability": "MLPMean"}),
        on="PassengerId",
        how="left",
    )
    keep_group = [
        "PassengerId",
        "familyfare_all__Panel",
        "familyfare_all__GradientBoosting",
        "familyfare_all__CatBoost",
        "ticket_wcg_any__CatBoost",
    ]
    base = base.merge(group[keep_group], on="PassengerId", how="left")
    base = base.merge(
        cat[["PassengerId", "slow_d3_l2_6"]].rename(
            columns={"slow_d3_l2_6": "CatAdultMaleTuned"}
        ),
        on="PassengerId",
        how="left",
    )
    base = base.merge(raw, on=["PassengerId", "Survived"], how="left")

    v4b_vote = (base["v4b__Champion"] > 0.5).astype(int)
    rule_vote = (base["RuleFit"] > 0.5).astype(int)
    mlp_vote = (base["MLPMean"] > 0.5).astype(int)
    base["ChampionPred"] = ((v4b_vote + rule_vote + mlp_vote) >= 2).astype(int)
    base["ChampionCorrect"] = (base["ChampionPred"] == base["Survived"]).astype(int)

    candidate_cols = {
        "FamilyFarePanel": "familyfare_all__Panel",
        "FamilyFareGB": "familyfare_all__GradientBoosting",
        "FamilyFareCat": "familyfare_all__CatBoost",
        "TicketWCGCat": "ticket_wcg_any__CatBoost",
        "TabPFNv2": "TabPFN_v2",
        "CatAdultMaleTuned": "CatAdultMaleTuned",
    }

    summary_rows = []
    for label, col in candidate_cols.items():
        pred = (base[col] > 0.5).astype(int)
        cand_correct = pred == base["Survived"]
        champ_wrong = base["ChampionCorrect"] == 0
        champ_right = ~champ_wrong
        rescue = champ_wrong & cand_correct
        harm = champ_right & (~cand_correct)
        base[f"{label}Pred"] = pred
        base[f"{label}Rescue"] = rescue.astype(int)
        summary_rows.append(
            {
                "candidate": label,
                "candidate_accuracy": float(cand_correct.mean()),
                "champion_wrong_candidate_right": int(rescue.sum()),
                "champion_right_candidate_wrong": int(harm.sum()),
                "net_vs_champion": int(rescue.sum() - harm.sum()),
                "binary_disagreement": int((pred != base["ChampionPred"]).sum()),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["champion_wrong_candidate_right", "net_vs_champion"],
        ascending=False,
    )
    summary.to_csv(V7_DIR / "disagreement_candidate_summary.csv", index=False)

    # Export every champion error with human-readable context and alternate votes.
    context_cols = [
        "PassengerId", "fold", "Survived", "ChampionPred", "Sex", "Pclass", "Age",
        "AgeMissing", "Title", "FamilySize", "FamilyBand", "IsAlone", "SibSp", "Parch",
        "Fare", "FareBandAudit", "Embarked", "Deck", "CabinKnown", "Ticket", "LastName",
    ]
    pred_cols = [f"{x}Pred" for x in candidate_cols]
    rescue_cols = [f"{x}Rescue" for x in candidate_cols]
    errors = base.loc[base["ChampionCorrect"] == 0, context_cols + pred_cols + rescue_cols].copy()
    errors["RescueCount"] = errors[rescue_cols].sum(axis=1)
    errors = errors.sort_values(["RescueCount", "PassengerId"], ascending=[False, True])
    errors.to_csv(V7_DIR / "champion_error_rows.csv", index=False)

    # Slice analysis: only report slices with enough support to reduce noise.
    slice_cols = ["Sex", "Pclass", "Title", "FamilyBand", "IsAlone", "Deck", "AgeMissing"]
    slice_rows = []
    for slice_col in slice_cols:
        for value, g in base.groupby(slice_col, dropna=False):
            if len(g) < 12:
                continue
            champ_acc = g["ChampionCorrect"].mean()
            row = {
                "slice_feature": slice_col,
                "slice_value": str(value),
                "n": len(g),
                "champion_accuracy": champ_acc,
                "champion_errors": int((g["ChampionCorrect"] == 0).sum()),
            }
            for label, col in candidate_cols.items():
                cand_acc = ((g[col] > 0.5).astype(int) == g["Survived"]).mean()
                row[f"{label}_accuracy"] = cand_acc
                row[f"{label}_delta_vs_champion"] = cand_acc - champ_acc
            slice_rows.append(row)
    slices = pd.DataFrame(slice_rows)
    delta_cols = [c for c in slices.columns if c.endswith("_delta_vs_champion")]
    slices["best_alt_delta"] = slices[delta_cols].max(axis=1)
    slices = slices.sort_values(
        ["best_alt_delta", "champion_errors", "n"], ascending=[False, False, False]
    )
    slices.to_csv(V7_DIR / "disagreement_slice_summary.csv", index=False)

    # Pairwise high-consensus rescue: at least two alternates agree on the correct label.
    alt_pred_cols = [f"{x}Pred" for x in candidate_cols]
    pred_matrix = base[alt_pred_cols].to_numpy()
    y = base["Survived"].to_numpy()
    correct_matrix = pred_matrix == y[:, None]
    base["AltCorrectCount"] = correct_matrix.sum(axis=1)
    consensus_rescue = base[
        (base["ChampionCorrect"] == 0) & (base["AltCorrectCount"] >= 2)
    ][context_cols + ["AltCorrectCount"] + alt_pred_cols].copy()
    consensus_rescue.to_csv(V7_DIR / "consensus_rescue_rows.csv", index=False)

    print("=== candidate rescue summary ===")
    print(summary.to_string(index=False))
    print("\n=== top slices where an alternate beats champion ===")
    print(
        slices.head(20).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x),
        )
    )
    print(f"\nChampion errors: {len(errors)}")
    print(f"Consensus rescue rows (>=2 alternatives correct): {len(consensus_rescue)}")


if __name__ == "__main__":
    main()
