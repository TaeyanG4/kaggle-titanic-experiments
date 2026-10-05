"""v26: train-vs-test adversarial validation and shift-weighted OOF audit.

Goal
----
Diagnose whether the 891 train rows and 418 test rows differ in observable X
space enough to explain repeated CV -> Public leaderboard disagreement.

Important safeguards
--------------------
* PassengerId is NEVER used: it trivially separates train/test by construction.
* Survived is NEVER used by the adversarial classifier.
* All engineered features are target-independent X-only features.
* Propensity probabilities are generated out-of-fold over the combined
  train+test domain-classification dataset.
* The resulting importance weights are diagnostic. They do not use Public LB.

Outputs
-------
exports/v26/adversarial_summary.csv
exports/v26/adversarial_seed_metrics.csv
exports/v26/adversarial_oof.csv
exports/v26/feature_shift_summary.csv
exports/v26/weighted_candidate_audit.csv
exports/v26/train_shift_weights.csv

No Kaggle submission is performed.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "exports" / "v26"


SEEDS = [42, 123, 777, 2026]
N_SPLITS = 5


def normalize_ticket_prefix(ticket: str) -> str:
    value = str(ticket).upper().strip()
    prefix = re.sub(r"\d", "", value)
    prefix = re.sub(r"[\s./]+", "", prefix)
    return prefix or "NONE"


def collapse_title(name: str) -> str:
    m = re.search(r" ([A-Za-z]+)\.", str(name))
    title = m.group(1) if m else "Other"
    mapping = {
        "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
        "Dr": "Officer", "Rev": "Officer", "Col": "Officer",
        "Major": "Officer", "Capt": "Officer",
        "Sir": "Royalty", "Don": "Royalty", "Countess": "Royalty",
        "Lady": "Royalty", "Dona": "Royalty", "Jonkheer": "Royalty",
    }
    return mapping.get(title, title if title in {"Mr", "Miss", "Mrs", "Master"} else "Other")


def build_features(train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    tr = train_raw.copy(); te = test_raw.copy()
    tr["Domain"] = 0; te["Domain"] = 1
    all_df = pd.concat([tr, te], ignore_index=True, sort=False)

    # Missingness is itself a potentially shifted signal, so preserve flags.
    all_df["AgeMissing"] = all_df["Age"].isna().astype(int)
    all_df["FareMissing"] = all_df["Fare"].isna().astype(int)
    all_df["CabinMissing"] = all_df["Cabin"].isna().astype(int)
    all_df["EmbarkedMissing"] = all_df["Embarked"].isna().astype(int)

    all_df["Title"] = all_df["Name"].map(collapse_title)
    all_df["FamilySize"] = all_df["SibSp"] + all_df["Parch"] + 1
    all_df["IsAlone"] = (all_df["FamilySize"] == 1).astype(int)
    all_df["FamilySizeBin"] = pd.cut(
        all_df["FamilySize"], bins=[0, 1, 4, 6, 20], labels=["Alone", "Small", "Medium", "Large"]
    ).astype(str)
    all_df["Deck"] = all_df["Cabin"].astype(str).str[0].where(all_df["Cabin"].notna(), "U")
    all_df["TicketPrefix"] = all_df["Ticket"].map(normalize_ticket_prefix)
    # Collapse rare prefixes so a few repeated tickets do not become row-source identifiers.
    prefix_counts = all_df["TicketPrefix"].value_counts()
    all_df["TicketPrefix"] = all_df["TicketPrefix"].where(all_df["TicketPrefix"].map(prefix_counts) >= 10, "OTHER")

    all_df["Surname"] = all_df["Name"].str.split(",").str[0].str.strip()
    all_df["FamilyGroup"] = all_df["Surname"] + "_" + all_df["FamilySize"].astype(str)
    all_df["TicketFreq"] = all_df["Ticket"].astype(str).map(all_df["Ticket"].astype(str).value_counts()).astype(float)
    all_df["SurnameFreq"] = all_df["Surname"].map(all_df["Surname"].value_counts()).astype(float)
    all_df["FamilyGroupFreq"] = all_df["FamilyGroup"].map(all_df["FamilyGroup"].value_counts()).astype(float)

    all_df["Embarked"] = all_df["Embarked"].fillna("S")
    fare_med = all_df.groupby(["Pclass", "Embarked"])["Fare"].transform("median")
    all_df["FareFilled"] = all_df["Fare"].fillna(fare_med).fillna(all_df["Fare"].median())
    age_med = all_df.groupby(["Title", "Pclass"])["Age"].transform("median")
    all_df["AgeFilled"] = all_df["Age"].fillna(age_med).fillna(all_df["Age"].median())
    all_df["LogFare"] = np.log1p(all_df["FareFilled"])
    all_df["FarePerPerson"] = all_df["FareFilled"] / all_df["TicketFreq"].clip(lower=1)
    all_df["LogFarePerPerson"] = np.log1p(all_df["FarePerPerson"])
    all_df["AgePclass"] = all_df["AgeFilled"] * all_df["Pclass"]
    all_df["SexPclass"] = all_df["Sex"].astype(str) + "_" + all_df["Pclass"].astype(str)
    all_df["TitlePclass"] = all_df["Title"].astype(str) + "_" + all_df["Pclass"].astype(str)

    numeric = [
        "Pclass", "AgeFilled", "SibSp", "Parch", "FareFilled", "LogFare",
        "FamilySize", "IsAlone", "TicketFreq", "SurnameFreq", "FamilyGroupFreq",
        "FarePerPerson", "LogFarePerPerson", "AgePclass",
        "AgeMissing", "FareMissing", "CabinMissing", "EmbarkedMissing",
    ]
    categorical = [
        "Sex", "Embarked", "Title", "Deck", "TicketPrefix", "FamilySizeBin",
        "SexPclass", "TitlePclass",
    ]

    feat = all_df[numeric + categorical].copy()
    feat = pd.get_dummies(feat, columns=categorical, drop_first=False, dtype=float)
    feat = feat.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_domain = all_df["Domain"].astype(int).to_numpy()
    meta = all_df[["Domain"]].copy()
    meta["PassengerId"] = all_df["PassengerId"].astype(int)
    return feat.astype(float), y_domain, meta


def feature_groups(columns: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    def take(name: str, prefixes: list[str]):
        groups[name] = [c for c in columns if any(c == p or c.startswith(p + "_") for p in prefixes)]
    take("raw_core", ["Pclass", "AgeFilled", "SibSp", "Parch", "FareFilled", "Sex", "Embarked"])
    take("missingness", ["AgeMissing", "FareMissing", "CabinMissing", "EmbarkedMissing"])
    take("family_ticket_counts", ["FamilySize", "IsAlone", "TicketFreq", "SurnameFreq", "FamilyGroupFreq", "FamilySizeBin"])
    take("title_deck", ["Title", "Deck", "TitlePclass"])
    take("fare_engineering", ["LogFare", "FarePerPerson", "LogFarePerPerson"])
    take("interactions", ["AgePclass", "SexPclass", "TitlePclass"])
    take("ticket_prefix", ["TicketPrefix"])
    return groups


def reps_from_full(x: pd.DataFrame) -> dict[str, list[str]]:
    groups = feature_groups(list(x.columns))
    all_cols = list(x.columns)
    raw = sorted(set(groups["raw_core"] + groups["missingness"]))
    no_counts = [c for c in all_cols if c not in groups["family_ticket_counts"]]
    no_missing = [c for c in all_cols if c not in groups["missingness"]]
    no_title_deck = [c for c in all_cols if c not in groups["title_deck"]]
    no_fare_eng = [c for c in all_cols if c not in groups["fare_engineering"]]
    no_ticket_prefix = [c for c in all_cols if c not in groups["ticket_prefix"]]
    return {
        "raw_core": raw,
        "full": all_cols,
        "no_counts": no_counts,
        "no_missingness": no_missing,
        "no_title_deck": no_title_deck,
        "no_fare_engineering": no_fare_eng,
        "no_ticket_prefix": no_ticket_prefix,
    }


def make_model(name: str, seed: int):
    if name == "logistic":
        return LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", random_state=seed)
    if name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=600, max_depth=7, min_samples_leaf=4,
            max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=seed,
        )
    if name == "lightgbm":
        return LGBMClassifier(
            n_estimators=240, num_leaves=12, max_depth=4, learning_rate=0.03,
            min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=5.0, reg_alpha=0.1, class_weight="balanced",
            random_state=seed, n_jobs=-1, verbosity=-1,
        )
    raise KeyError(name)


def cv_predict(x: pd.DataFrame, y: np.ndarray, cols: list[str], model_name: str):
    seed_metrics = []
    probs_by_seed = []
    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        oof = np.zeros(len(x), dtype=float)
        fold_metrics = []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(x, y)):
            xtr = x.iloc[tr_idx][cols].to_numpy(float)
            xva = x.iloc[va_idx][cols].to_numpy(float)
            if model_name == "logistic":
                scaler = StandardScaler()
                xtr = scaler.fit_transform(xtr); xva = scaler.transform(xva)
            model = make_model(model_name, seed * 10 + fold)
            model.fit(xtr, y[tr_idx])
            p = model.predict_proba(xva)[:, 1]
            oof[va_idx] = p
            fold_metrics.append(roc_auc_score(y[va_idx], p))
        seed_metrics.append({
            "seed": seed,
            "auc": roc_auc_score(y, oof),
            "accuracy": accuracy_score(y, oof >= 0.5),
            "fold_auc_std": float(np.std(fold_metrics)),
        })
        probs_by_seed.append(oof)
    return pd.DataFrame(seed_metrics), np.mean(np.vstack(probs_by_seed), axis=0)


def distribution_summary(train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["Pclass", "Age", "SibSp", "Parch", "Fare"]:
        tr = pd.to_numeric(train_raw[col], errors="coerce")
        te = pd.to_numeric(test_raw[col], errors="coerce")
        pooled = pd.concat([tr, te], ignore_index=True)
        sd = float(pooled.std())
        rows.append({
            "feature": col,
            "kind": "numeric",
            "train_mean": float(tr.mean()), "test_mean": float(te.mean()),
            "std_mean_diff": float((te.mean() - tr.mean()) / sd) if sd > 0 else 0.0,
            "train_missing": float(tr.isna().mean()), "test_missing": float(te.isna().mean()),
        })
    for col in ["Sex", "Embarked"]:
        tv = train_raw[col].fillna("<NA>").value_counts(normalize=True)
        ev = test_raw[col].fillna("<NA>").value_counts(normalize=True)
        cats = sorted(set(tv.index) | set(ev.index))
        l1 = 0.5 * sum(abs(float(tv.get(c, 0)) - float(ev.get(c, 0))) for c in cats)
        rows.append({"feature": col, "kind": "categorical", "total_variation": l1})
    for col in ["Age", "Fare", "Cabin", "Embarked"]:
        rows.append({
            "feature": col + "Missing",
            "kind": "missingness",
            "train_rate": float(train_raw[col].isna().mean()),
            "test_rate": float(test_raw[col].isna().mean()),
            "abs_diff": float(abs(train_raw[col].isna().mean() - test_raw[col].isna().mean())),
        })
    return pd.DataFrame(rows)


def density_ratio_weights(domain_prob: np.ndarray, domain_y: np.ndarray, model_name: str) -> pd.DataFrame:
    p_test = float(np.mean(domain_y == 1)); p_train = 1.0 - p_test
    idx = np.flatnonzero(domain_y == 0)
    p = np.clip(domain_prob[idx], 1e-3, 1 - 1e-3)
    ratio = (p / (1 - p)) * (p_train / p_test)
    lo, hi = np.quantile(ratio, [0.01, 0.99])
    clipped = np.clip(ratio, max(0.05, lo), min(10.0, hi))
    clipped = clipped / np.mean(clipped)
    ess = (clipped.sum() ** 2) / np.sum(clipped ** 2)
    return pd.DataFrame({
        "train_row": idx,
        "weight": clipped,
        "raw_ratio": ratio,
        "domain_prob": p,
        "weight_model": model_name,
        "effective_sample_size": ess,
    })


def load_candidate_predictions(train_raw: pd.DataFrame) -> dict[str, np.ndarray]:
    y = train_raw["Survived"].astype(int).to_numpy()
    zoo = pd.read_csv(BASE_DIR / "exports" / "v5" / "model_zoo_oof.csv")
    mlp = pd.read_csv(BASE_DIR / "exports" / "v5" / "seed_probes" / "MLP_PLR_seed_mean__oof.csv")
    v5 = (((zoo["v4b__Champion"].to_numpy() > .5).astype(int)
          + (zoo["RuleFit"].to_numpy() > .5).astype(int)
          + (mlp["probability"].to_numpy() > .5).astype(int)) >= 2).astype(int)
    de = pd.read_csv(BASE_DIR / "exports" / "v21" / "deotte_fixed_oof.csv")["wcg_both"].astype(int).to_numpy()
    rf = (pd.read_csv(BASE_DIR / "exports" / "v22" / "hpo_screen_oof.csv")["RandomForest_6"].to_numpy() > .5).astype(int)
    trans = pd.read_csv(BASE_DIR / "exports" / "v13" / "transductive_rf_oof.csv")["transductive_rf_pred"].astype(int).to_numpy()
    v25 = ((v5 + de + rf) >= 2).astype(int)
    return {"v5_robust": v5, "deotte": de, "typed_rf6": rf, "v10_analogue": trans, "v25_majority": v25, "_y": y}


def weighted_candidate_audit(weights_map: dict[str, pd.DataFrame], train_raw: pd.DataFrame) -> pd.DataFrame:
    preds = load_candidate_predictions(train_raw)
    y = preds.pop("_y")
    rows = []
    for wname, wdf in weights_map.items():
        w = wdf.sort_values("train_row")["weight"].to_numpy(float)
        ess = float(wdf["effective_sample_size"].iloc[0])
        for name, pred in preds.items():
            correct = (pred == y).astype(float)
            weighted_acc = float(np.sum(w * correct) / np.sum(w))
            rows.append({
                "weight_model": wname,
                "candidate": name,
                "unweighted_accuracy": accuracy_score(y, pred),
                "weighted_accuracy": weighted_acc,
                "weighted_delta_vs_v5": np.nan,
                "effective_sample_size": ess,
            })
    out = pd.DataFrame(rows)
    for wname in out["weight_model"].unique():
        mask = out["weight_model"] == wname
        base = float(out.loc[mask & (out["candidate"] == "v5_robust"), "weighted_accuracy"].iloc[0])
        out.loc[mask, "weighted_delta_vs_v5"] = out.loc[mask, "weighted_accuracy"] - base
    return out.sort_values(["weight_model", "weighted_accuracy"], ascending=[True, False])


def main():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    x, y_domain, meta = build_features(train_raw, test_raw)
    reps = reps_from_full(x)

    all_seed_metrics = []
    summary_rows = []
    oof_out = meta.copy()

    # Full multi-model audit + ablation with the strongest nonlinear detector.
    jobs = []
    for model in ["logistic", "extra_trees", "lightgbm"]:
        jobs.append(("full", model))
    for rep in ["raw_core", "no_counts", "no_missingness", "no_title_deck", "no_fare_engineering", "no_ticket_prefix"]:
        jobs.append((rep, "lightgbm"))

    for rep, model in jobs:
        print(f"=== adversarial {rep} / {model} ===", flush=True)
        seed_df, prob = cv_predict(x, y_domain, reps[rep], model)
        seed_df.insert(0, "representation", rep); seed_df.insert(1, "model", model)
        all_seed_metrics.append(seed_df)
        oof_out[f"{rep}__{model}"] = prob
        summary_rows.append({
            "representation": rep,
            "model": model,
            "mean_seed_auc": float(seed_df["auc"].mean()),
            "std_seed_auc": float(seed_df["auc"].std()),
            "min_seed_auc": float(seed_df["auc"].min()),
            "max_seed_auc": float(seed_df["auc"].max()),
            "auc_of_mean_oof": roc_auc_score(y_domain, prob),
            "accuracy_of_mean_oof": accuracy_score(y_domain, prob >= .5),
            "n_features": len(reps[rep]),
        })
        print(seed_df[["seed", "auc", "accuracy"]].to_string(index=False, float_format=lambda v:f"{v:.5f}"), flush=True)

    seed_metrics = pd.concat(all_seed_metrics, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values("auc_of_mean_oof", ascending=False)
    seed_metrics.to_csv(EXPORT_DIR / "adversarial_seed_metrics.csv", index=False)
    summary.to_csv(EXPORT_DIR / "adversarial_summary.csv", index=False)
    oof_out.to_csv(EXPORT_DIR / "adversarial_oof.csv", index=False)

    dist = distribution_summary(train_raw, test_raw)
    dist.to_csv(EXPORT_DIR / "feature_shift_summary.csv", index=False)

    weights_map = {}
    for model in ["logistic", "lightgbm"]:
        col = f"full__{model}"
        wdf = density_ratio_weights(oof_out[col].to_numpy(float), y_domain, model)
        wdf["PassengerId"] = train_raw.iloc[wdf["train_row"].astype(int)]["PassengerId"].astype(int).to_numpy()
        weights_map[model] = wdf
    weights_all = pd.concat(weights_map.values(), ignore_index=True)
    weights_all.to_csv(EXPORT_DIR / "train_shift_weights.csv", index=False)

    audit = weighted_candidate_audit(weights_map, train_raw)
    audit.to_csv(EXPORT_DIR / "weighted_candidate_audit.csv", index=False)

    print("\n=== adversarial summary ===")
    print(summary.to_string(index=False, float_format=lambda v:f"{v:.5f}"))
    print("\n=== simple train/test shift summary ===")
    print(dist.to_string(index=False, float_format=lambda v:f"{v:.5f}"))
    print("\n=== shift-weighted candidate audit ===")
    print(audit.to_string(index=False, float_format=lambda v:f"{v:.5f}"))
    print("\nNo Kaggle submission was performed.")


if __name__ == "__main__":
    main()
