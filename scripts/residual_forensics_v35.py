"""v35: passenger-level residual forensics across repeated OOF surfaces.

Purpose
-------
Stop inventing more tabular features blindly. Instead, identify passengers that
remain hard across seeds and model families, then ask whether their errors are
associated with information that current engineered features may have discarded
(raw Name/Ticket/Cabin strings, rare titles, unusual family/ticket structure).

Inputs are immutable saved OOF predictions from v33 plus raw train.csv.
No model training and no Kaggle submission.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
V33_DIR = BASE_DIR / "exports" / "v33"
EXPORT_DIR = BASE_DIR / "exports" / "v35"


def title_from_name(name: str) -> str:
    m = re.search(r",\s*([^.]*)\.", str(name))
    return m.group(1).strip() if m else "Unknown"


def ticket_prefix(ticket: str) -> str:
    s = str(ticket).upper().strip()
    p = re.sub(r"\d", "", s)
    p = re.sub(r"[\s./]+", "", p)
    return p or "NUMERIC"


def add_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["TitleRaw"] = out["Name"].map(title_from_name)
    out["Surname"] = out["Name"].str.split(",").str[0].str.strip()
    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    out["TicketPrefixRaw"] = out["Ticket"].map(ticket_prefix)
    out["TicketFreqTrain"] = out["Ticket"].astype(str).map(out["Ticket"].astype(str).value_counts())
    out["SurnameFreqTrain"] = out["Surname"].map(out["Surname"].value_counts())
    out["FamilyKey"] = out["Surname"] + "_" + out["FamilySize"].astype(str)
    out["FamilyKeyFreqTrain"] = out["FamilyKey"].map(out["FamilyKey"].value_counts())
    out["CabinKnown"] = out["Cabin"].notna().astype(int)
    out["DeckRaw"] = out["Cabin"].fillna("U").astype(str).str[0]
    out["CabinCountRaw"] = out["Cabin"].fillna("").astype(str).map(lambda s: len([x for x in s.split() if x]))
    out["NameLength"] = out["Name"].astype(str).str.len()
    out["TicketLength"] = out["Ticket"].astype(str).str.len()
    out["NameHasParen"] = out["Name"].astype(str).str.contains("\\(", regex=True).astype(int)
    out["AgeMissing"] = out["Age"].isna().astype(int)
    out["FareMissing"] = out["Fare"].isna().astype(int)
    out["EmbarkedMissing"] = out["Embarked"].isna().astype(int)
    return out


def aggregate_repeated(pred: pd.DataFrame) -> pd.DataFrame:
    r = pred[pred["surface"].str.startswith("repeated_")].copy()
    r["seed"] = r["surface"].str.replace("repeated_", "", regex=False).astype(int)
    models = ["v5", "deotte", "eb", "consensus"]
    for m in models:
        r[m + "_correct"] = (r[m].astype(int) == r["Survived"].astype(int)).astype(int)

    agg = r.groupby(["PassengerId", "Survived"], as_index=False).agg(
        n_surfaces=("seed", "count"),
        v5_correct_rate=("v5_correct", "mean"),
        deotte_correct_rate=("deotte_correct", "mean"),
        eb_correct_rate=("eb_correct", "mean"),
        consensus_correct_rate=("consensus_correct", "mean"),
        v5_pred_rate=("v5", "mean"),
        deotte_pred_rate=("deotte", "mean"),
        eb_pred_rate=("eb", "mean"),
        consensus_pred_rate=("consensus", "mean"),
    )

    # Seed instability and family disagreement.
    instability = []
    for pid, g in r.groupby("PassengerId"):
        arr = g[["v5", "deotte", "eb"]].to_numpy(int)
        row_disagree = (arr.max(axis=1) != arr.min(axis=1)).mean()
        model_seed_var = np.mean([g[c].astype(float).var(ddof=0) for c in ["v5", "deotte", "eb"]])
        instability.append({
            "PassengerId": int(pid),
            "family_disagreement_rate": float(row_disagree),
            "mean_model_seed_variance": float(model_seed_var),
        })
    agg = agg.merge(pd.DataFrame(instability), on="PassengerId", how="left")

    agg["mean_family_correct_rate"] = agg[["v5_correct_rate", "deotte_correct_rate", "eb_correct_rate"]].mean(axis=1)
    agg["all3_family_hard"] = (agg["mean_family_correct_rate"] <= 0.20).astype(int)
    agg["consensus_stable_hard"] = (agg["consensus_correct_rate"] <= 0.20).astype(int)
    agg["consensus_stable_easy"] = (agg["consensus_correct_rate"] >= 1.00).astype(int)
    agg["ambiguous"] = ((agg["consensus_correct_rate"] > 0.20) & (agg["consensus_correct_rate"] < 1.00)).astype(int)
    return agg


def subgroup_table(full: pd.DataFrame, col: str, min_n: int = 5) -> pd.DataFrame:
    rows = []
    global_hard = float(full["consensus_stable_hard"].mean())
    for key, g in full.groupby(col, dropna=False):
        if len(g) < min_n:
            continue
        hard = float(g["consensus_stable_hard"].mean())
        rows.append({
            "feature": col,
            "value": str(key),
            "n": len(g),
            "hard_n": int(g["consensus_stable_hard"].sum()),
            "hard_rate": hard,
            "lift_vs_global": hard / global_hard if global_hard > 0 else np.nan,
            "mean_consensus_correct": float(g["consensus_correct_rate"].mean()),
            "mean_family_disagreement": float(g["family_disagreement_rate"].mean()),
        })
    return pd.DataFrame(rows)


def numeric_bin_table(full: pd.DataFrame, col: str, bins: int = 5) -> pd.DataFrame:
    x = full[col]
    if x.notna().sum() < 20 or x.nunique(dropna=True) < 3:
        return pd.DataFrame()
    try:
        b = pd.qcut(x, q=min(bins, x.nunique(dropna=True)), duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    tmp = full.copy(); tmp["_bin"] = b.astype(str)
    out = subgroup_table(tmp, "_bin", min_n=5)
    if not out.empty:
        out["feature"] = col + "_quantile"
    return out


def main():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    train = add_raw_features(pd.read_csv(DATA_DIR / "train.csv"))
    pred = pd.read_csv(V33_DIR / "predictions.csv")
    hardness = aggregate_repeated(pred)
    full = train.merge(hardness, on=["PassengerId", "Survived"], how="left")

    full["SexPclass"] = full["Sex"].astype(str) + "_P" + full["Pclass"].astype(str)
    full["TitlePclassRaw"] = full["TitleRaw"].astype(str) + "_P" + full["Pclass"].astype(str)
    full["FamilySizeBand"] = pd.cut(full["FamilySize"], [0,1,4,6,20], labels=["alone","small","medium","large"]).astype(str)
    full["TicketFreqBand"] = pd.cut(full["TicketFreqTrain"], [0,1,2,4,100], labels=["single","pair","small_group","large_group"]).astype(str)
    full["AgeBandRaw"] = pd.cut(full["Age"], [0,5,12,18,30,45,60,100], right=False).astype(str)
    full["FareBandRaw"] = pd.qcut(full["Fare"], 5, duplicates="drop").astype(str)

    full.to_csv(EXPORT_DIR / "passenger_hardness.csv", index=False)

    # Stable hard/easy exports for manual forensic review.
    stable_hard = full[full["consensus_stable_hard"] == 1].copy()
    stable_hard = stable_hard.sort_values(
        ["mean_family_correct_rate", "family_disagreement_rate", "PassengerId"],
        ascending=[True, False, True],
    )
    cols = [
        "PassengerId","Survived","Pclass","Sex","Age","Fare","Name","TitleRaw","Surname",
        "SibSp","Parch","FamilySize","Ticket","TicketPrefixRaw","TicketFreqTrain",
        "Cabin","DeckRaw","Embarked","v5_correct_rate","deotte_correct_rate","eb_correct_rate",
        "consensus_correct_rate","family_disagreement_rate","mean_model_seed_variance",
    ]
    stable_hard[cols].to_csv(EXPORT_DIR / "stable_hard_cases.csv", index=False)
    full[full["consensus_stable_easy"] == 1][cols].to_csv(EXPORT_DIR / "stable_easy_cases.csv", index=False)

    # Categorical and binned numerical slice lifts.
    slices = []
    for col in [
        "Sex","Pclass","SexPclass","TitleRaw","TitlePclassRaw","Embarked","DeckRaw",
        "CabinKnown","FamilySizeBand","TicketFreqBand","TicketPrefixRaw","AgeMissing",
        "NameHasParen","IsAlone",
    ]:
        z = subgroup_table(full, col, min_n=5)
        if not z.empty: slices.append(z)
    for col in ["Age","Fare","NameLength","TicketLength","SurnameFreqTrain","FamilyKeyFreqTrain"]:
        z = numeric_bin_table(full, col)
        if not z.empty: slices.append(z)
    slice_df = pd.concat(slices, ignore_index=True) if slices else pd.DataFrame()
    if not slice_df.empty:
        slice_df = slice_df.sort_values(["lift_vs_global","n"], ascending=[False,False])
    slice_df.to_csv(EXPORT_DIR / "hard_case_slice_lifts.csv", index=False)

    # Which model family uniquely rescues stable-v5 failures?
    repeated = pred[pred["surface"].str.startswith("repeated_")].copy()
    repeated["v5_wrong"] = repeated["v5"].astype(int) != repeated["Survived"].astype(int)
    repeated["deotte_rescue"] = repeated["v5_wrong"] & (repeated["deotte"].astype(int) == repeated["Survived"].astype(int))
    repeated["eb_rescue"] = repeated["v5_wrong"] & (repeated["eb"].astype(int) == repeated["Survived"].astype(int))
    rescue = repeated.groupby("PassengerId", as_index=False).agg(
        v5_wrong_surfaces=("v5_wrong","sum"),
        deotte_rescue_surfaces=("deotte_rescue","sum"),
        eb_rescue_surfaces=("eb_rescue","sum"),
    )
    rescue = rescue.merge(train[["PassengerId","Name","Sex","Pclass","Age","Fare","Ticket","Cabin","TitleRaw","TicketPrefixRaw"]], on="PassengerId", how="left")
    rescue = rescue.sort_values(["v5_wrong_surfaces","deotte_rescue_surfaces","eb_rescue_surfaces"], ascending=False)
    rescue.to_csv(EXPORT_DIR / "v5_rescue_profile.csv", index=False)

    global_hard_rate = float(full["consensus_stable_hard"].mean())
    summary = pd.DataFrame([{
        "n_train": len(full),
        "n_repeated_surfaces": int(pred[pred["surface"].str.startswith("repeated_")]["surface"].nunique()),
        "stable_hard_n": int(full["consensus_stable_hard"].sum()),
        "stable_hard_rate": global_hard_rate,
        "stable_easy_n": int(full["consensus_stable_easy"].sum()),
        "ambiguous_n": int(full["ambiguous"].sum()),
        "all3_family_hard_n": int(full["all3_family_hard"].sum()),
        "mean_family_disagreement_rate": float(full["family_disagreement_rate"].mean()),
        "mean_model_seed_variance": float(full["mean_model_seed_variance"].mean()),
    }])
    summary.to_csv(EXPORT_DIR / "summary.csv", index=False)

    print("=== v35 residual summary ===")
    print(summary.to_string(index=False, float_format=lambda x:f"{x:.5f}"))
    print("\n=== top hard-case slice lifts ===")
    if not slice_df.empty:
        print(slice_df.head(25).to_string(index=False, float_format=lambda x:f"{x:.3f}"))
    print("\n=== first stable hard cases ===")
    print(stable_hard[cols].head(30).to_string(index=False))
    print("\nNo Kaggle submission was performed.")


if __name__ == "__main__":
    main()
