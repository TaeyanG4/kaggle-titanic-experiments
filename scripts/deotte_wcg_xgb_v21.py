"""Python port and leakage-aware audit of Chris Deotte's Titanic WCG+XGBoost.

Public source:
  cdeotte/titanic-wcg-xgboost-0-84688

The original notebook is R.  This port preserves its key mechanisms:
  * person type: man / boy(Master) / woman
  * precise GroupId = Surname + Pclass + masked-ticket + Fare + Embarked
  * singleton WCG collapse + same-ticket nanny/relative attachment
  * historical Wilkes/Hocking structural link (PassengerId 893 -> 775)
  * WCG deterministic overrides
  * 2-feature XGBoost for adult males and solo females with the original
    thresholds (male survive p>=0.90, female perish p<=0.08)

R's rpart imputation is approximated with sklearn regression trees because R is
not installed in the local environment.  The exact WCG mechanism itself does
not depend on Age; Age only affects the two XGBoost refinements.

Outputs include the project's trusted fixed-fold OOF audit, repeated stratified
stress results, and current-test predictions.  No Kaggle submission is made.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "exports" / "v21"
SUB_DIR = BASE_DIR / "submissions"
FOLD_MANIFEST = BASE_DIR / "exports" / "wcg_audit_v1" / "fold_manifest_seed42.csv"

# Exact R notebook execution outputs recovered from a public notebook mirror.
# These are predictions/model outputs only; no Titanic test labels are used.
DEOTTE_EXACT_MALE_LIVE = [926, 942, 1094, 1215]
DEOTTE_EXACT_FEMALE_PERISH = [928, 990, 1030, 1061, 1091, 1098, 1160, 1172, 1205, 1304]


def fmt_num(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{float(x):.12g}"


def person_type(df: pd.DataFrame) -> pd.Series:
    out = pd.Series("man", index=df.index, dtype=object)
    out[df["Name"].str.contains("Master", regex=False, na=False)] = "boy"
    out[df["Sex"].eq("female")] = "woman"
    return out


def _tree_impute(
    fit_df: pd.DataFrame,
    apply_df: pd.DataFrame,
    target: str,
    categorical: list[str],
    numeric: list[str],
) -> np.ndarray:
    observed = fit_df[target].notna()
    if observed.sum() < 20:
        return np.full(len(apply_df), float(fit_df[target].median()))
    prep = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("num", "passthrough", numeric),
        ],
        remainder="drop",
    )
    tree = DecisionTreeRegressor(
        random_state=42,
        min_samples_split=20,
        min_samples_leaf=7,
        max_depth=30,
    )
    model = make_pipeline(prep, tree)
    model.fit(fit_df.loc[observed, categorical + numeric], fit_df.loc[observed, target])
    return model.predict(apply_df[categorical + numeric])


def preprocess_scope(scope: pd.DataFrame, fit_indices: np.ndarray | None = None) -> pd.DataFrame:
    """Approximate original rpart Age/Fare imputation and engineer XGB inputs.

    If fit_indices is None, fit target-independent imputers on the full visible
    scope (historical/transductive mode).  Otherwise fit them on fit_indices
    only and apply to all rows in scope (strict preprocessing stress mode).
    """
    out = scope.copy().reset_index(drop=True)
    out["TitleDeotte"] = person_type(out)
    fit = out if fit_indices is None else out.iloc[fit_indices].copy()

    age_missing = out["Age"].isna()
    if age_missing.any():
        pred = _tree_impute(
            fit,
            out.loc[age_missing],
            "Age",
            ["TitleDeotte"],
            ["Pclass", "SibSp", "Parch"],
        )
        out.loc[age_missing, "Age"] = pred
    out["Age"] = out["Age"].fillna(float(fit["Age"].median()))

    # Fare model uses imputed Age, mirroring the R notebook's ordering.
    fit2 = out if fit_indices is None else out.iloc[fit_indices].copy()
    fare_missing = out["Fare"].isna()
    if fare_missing.any():
        pred = _tree_impute(
            fit2,
            out.loc[fare_missing],
            "Fare",
            ["TitleDeotte", "Embarked", "Sex"],
            ["Pclass", "Age"],
        )
        out.loc[fare_missing, "Fare"] = pred
    out["Fare"] = out["Fare"].fillna(float(fit2["Fare"].median()))

    counts = out["Ticket"].astype(str).value_counts()
    out["TicketFreq"] = out["Ticket"].astype(str).map(counts).astype(float)
    out["FareAdj"] = out["Fare"] / out["TicketFreq"]
    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["SurnameDeotte"] = out["Name"].str.split(",").str[0].str.strip()
    out["MaskedTicket"] = out["Ticket"].astype(str).map(lambda s: (s[:-1] + "X") if s else "X")
    out["FareKey"] = out["Fare"].map(fmt_num)
    out["EmbarkedKey"] = out["Embarked"].fillna("").astype(str)
    out["RawGroupId"] = (
        out["SurnameDeotte"] + "-" + out["Pclass"].astype(str) + "-" + out["MaskedTicket"]
        + "-" + out["FareKey"] + "-" + out["EmbarkedKey"]
    )
    out["TicketIdDeotte"] = (
        out["Pclass"].astype(str) + "-" + out["MaskedTicket"] + "-" + out["FareKey"]
        + "-" + out["EmbarkedKey"]
    )
    return out


def assign_group_ids(scope: pd.DataFrame, *, special_link: bool = True) -> pd.DataFrame:
    out = scope.copy()
    out["GroupIdDeotte"] = out["RawGroupId"].astype(object)
    out.loc[out["TitleDeotte"].eq("man"), "GroupIdDeotte"] = "noGroup"

    if special_link:
        src = out.index[out["PassengerId"].eq(775)].tolist()
        dst = out.index[out["PassengerId"].eq(893)].tolist()
        if src and dst:
            out.loc[dst[0], "GroupIdDeotte"] = out.loc[src[0], "GroupIdDeotte"]

    freq = out["GroupIdDeotte"].value_counts(dropna=False)
    out["GroupFreqDeotte"] = out["GroupIdDeotte"].map(freq).astype(int)
    singleton = out["GroupFreqDeotte"].le(1)
    out.loc[singleton, "GroupIdDeotte"] = "noGroup"

    # Match the R loop: for a non-man singleton, assign the first GroupId among
    # passengers with the same TicketId.  If that first id is noGroup, it stays.
    first_group_by_ticket = out.groupby("TicketIdDeotte", sort=False)["GroupIdDeotte"].first()
    candidates = out["TitleDeotte"].ne("man") & out["GroupIdDeotte"].eq("noGroup")
    out.loc[candidates, "GroupIdDeotte"] = out.loc[candidates, "TicketIdDeotte"].map(first_group_by_ticket)
    out["GroupIdDeotte"] = out["GroupIdDeotte"].fillna("noGroup")
    return out


def group_survival(
    scope: pd.DataFrame,
    labeled_indices: np.ndarray,
    apply_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ref = scope.iloc[labeled_indices].copy()
    if ref["Survived"].isna().any():
        raise ValueError("Reference labels must be known")
    means = ref.groupby("GroupIdDeotte", sort=False)["Survived"].mean()
    app = scope.iloc[apply_indices]
    gs = app["GroupIdDeotte"].map(means).to_numpy(dtype=float)
    missing = np.isnan(gs)
    gs[missing & app["Pclass"].to_numpy().astype(int).eq(3) if False else np.zeros(len(gs), dtype=bool)] = 0.0
    # Vectorized fallback, preserving original rule: unknown Pclass 3 -> 0, else 1.
    pclass = app["Pclass"].to_numpy(dtype=int)
    gs[missing] = np.where(pclass[missing] == 3, 0.0, 1.0)
    exact = np.isclose(gs, 0.0) | np.isclose(gs, 1.0)
    return gs, exact


def make_xgb(seed: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.1,
        min_child_weight=1,
        gamma=0.0,  # source spells 'gammma', so xgboost used its default gamma=0
        colsample_bytree=1.0,
        subsample=1.0,
        objective="binary:logistic",
        eval_metric="error",
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
    )


@dataclass
class PredBundle:
    gender: np.ndarray
    wcg: np.ndarray
    wcg_male: np.ndarray
    wcg_female: np.ndarray
    wcg_both: np.ndarray
    score: np.ndarray
    group_survival: np.ndarray
    wcg_exact: np.ndarray


def predict_split(
    scope_raw: pd.DataFrame,
    labeled_indices: np.ndarray,
    apply_indices: np.ndarray,
    *,
    seed: int,
    imputer_fit_indices: np.ndarray | None,
    special_link: bool = True,
) -> PredBundle:
    scope = preprocess_scope(scope_raw, fit_indices=imputer_fit_indices)
    scope = assign_group_ids(scope, special_link=special_link)
    y_ref = scope.iloc[labeled_indices]["Survived"].astype(int).to_numpy()
    app = scope.iloc[apply_indices]

    gender = app["Sex"].eq("female").astype(int).to_numpy()
    gs, exact = group_survival(scope, labeled_indices, apply_indices)
    wcg = gender.copy()
    title = app["TitleDeotte"].to_numpy()
    wcg[(title == "woman") & np.isclose(gs, 0.0)] = 0
    wcg[(title == "boy") & np.isclose(gs, 1.0)] = 1

    male_pred = wcg.copy()
    female_pred = wcg.copy()
    both_pred = wcg.copy()
    soft = wcg.astype(float)

    # Adult-male XGB.
    ref = scope.iloc[labeled_indices]
    man_ref_mask = ref["TitleDeotte"].eq("man").to_numpy()
    man_app_mask = app["TitleDeotte"].eq("man").to_numpy()
    if man_ref_mask.sum() >= 20 and man_app_mask.any():
        xm_tr = np.column_stack(
            [
                ref.loc[man_ref_mask, "FareAdj"].to_numpy() / 10.0,
                ref.loc[man_ref_mask, "FamilySize"].to_numpy()
                + ref.loc[man_ref_mask, "Age"].to_numpy() / 70.0,
            ]
        )
        xm_ap = np.column_stack(
            [
                app.loc[man_app_mask, "FareAdj"].to_numpy() / 10.0,
                app.loc[man_app_mask, "FamilySize"].to_numpy()
                + app.loc[man_app_mask, "Age"].to_numpy() / 70.0,
            ]
        )
        m = make_xgb(seed)
        m.fit(xm_tr, y_ref[man_ref_mask])
        pm = m.predict_proba(xm_ap)[:, 1]
        live = pm >= 0.90
        male_positions = np.flatnonzero(man_app_mask)
        male_pred[male_positions[live]] = 1
        both_pred[male_positions[live]] = 1
        soft[male_positions] = pm

    # Solo-female XGB. The original only applies it to non-WCG test women.
    woman_ref_mask = ref["TitleDeotte"].eq("woman").to_numpy() & ref["FamilySize"].eq(1).to_numpy()
    woman_app_mask = (
        app["TitleDeotte"].eq("woman").to_numpy()
        & app["FamilySize"].eq(1).to_numpy()
        & (~exact)
    )
    if woman_ref_mask.sum() >= 20 and woman_app_mask.any():
        xf_tr = np.column_stack(
            [
                ref.loc[woman_ref_mask, "FareAdj"].to_numpy() / 10.0,
                ref.loc[woman_ref_mask, "Age"].to_numpy() / 15.0,
            ]
        )
        xf_ap = np.column_stack(
            [
                app.loc[woman_app_mask, "FareAdj"].to_numpy() / 10.0,
                app.loc[woman_app_mask, "Age"].to_numpy() / 15.0,
            ]
        )
        f = make_xgb(seed + 10000)
        f.fit(xf_tr, y_ref[woman_ref_mask])
        pf = f.predict_proba(xf_ap)[:, 1]
        perish = pf <= 0.08
        female_positions = np.flatnonzero(woman_app_mask)
        female_pred[female_positions[perish]] = 0
        both_pred[female_positions[perish]] = 0
        soft[female_positions] = pf

    return PredBundle(
        gender=gender,
        wcg=wcg,
        wcg_male=male_pred,
        wcg_female=female_pred,
        wcg_both=both_pred,
        score=soft,
        group_survival=gs,
        wcg_exact=exact,
    )


def fixed_fold_audit(train_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(FOLD_MANIFEST)
    if not train_raw["PassengerId"].astype(int).reset_index(drop=True).equals(
        manifest["PassengerId"].astype(int).reset_index(drop=True)
    ):
        raise ValueError("Fold manifest mismatch")
    folds = manifest["fold"].astype(int).to_numpy()
    y = train_raw["Survived"].astype(int).to_numpy()
    variants = ["gender", "wcg", "wcg_male", "wcg_female", "wcg_both"]
    oof = {v: np.zeros(len(train_raw), dtype=int) for v in variants}
    score = np.zeros(len(train_raw), dtype=float)
    rows = []

    for fold in sorted(np.unique(folds)):
        tr_idx = np.flatnonzero(folds != fold)
        va_idx = np.flatnonzero(folds == fold)
        b = predict_split(
            train_raw,
            tr_idx,
            va_idx,
            seed=2100 + int(fold),
            imputer_fit_indices=tr_idx,
            special_link=True,
        )
        for v in variants:
            oof[v][va_idx] = getattr(b, v)
            rows.append(
                {
                    "fold": int(fold),
                    "variant": v,
                    "accuracy": accuracy_score(y[va_idx], getattr(b, v)),
                    "changed_vs_gender": int(np.sum(getattr(b, v) != b.gender)),
                }
            )
        score[va_idx] = b.score

    summary = []
    for v in variants:
        fd = [r["accuracy"] for r in rows if r["variant"] == v]
        summary.append(
            {
                "variant": v,
                "accuracy": accuracy_score(y, oof[v]),
                "fold_std": float(np.std(fd)),
                "errors": int(np.sum(oof[v] != y)),
            }
        )
    out = pd.DataFrame(
        {
            "PassengerId": train_raw["PassengerId"].astype(int),
            "fold": folds,
            "Survived": y,
            **oof,
            "deotte_score": score,
        }
    )
    return pd.DataFrame(summary).sort_values("accuracy", ascending=False), out


def repeated_audit(train_raw: pd.DataFrame) -> pd.DataFrame:
    y = train_raw["Survived"].astype(int).to_numpy()
    rows = []
    for split_seed in [42, 123, 777, 2026]:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=split_seed)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(train_raw, y)):
            b = predict_split(
                train_raw,
                tr_idx,
                va_idx,
                seed=split_seed * 10 + fold,
                imputer_fit_indices=tr_idx,
                special_link=True,
            )
            for v in ["gender", "wcg", "wcg_male", "wcg_female", "wcg_both"]:
                rows.append(
                    {
                        "split_seed": split_seed,
                        "fold": fold,
                        "variant": v,
                        "accuracy": accuracy_score(y[va_idx], getattr(b, v)),
                    }
                )
    return pd.DataFrame(rows)


def test_predictions(train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> pd.DataFrame:
    train = train_raw.copy()
    test = test_raw.copy()
    test["Survived"] = np.nan
    scope = pd.concat([train, test], ignore_index=True, sort=False)
    tr_idx = np.arange(len(train))
    te_idx = np.arange(len(train), len(scope))
    b = predict_split(
        scope,
        tr_idx,
        te_idx,
        seed=21,
        imputer_fit_indices=None,  # historical notebook sees train+test covariates
        special_link=True,
    )
    return pd.DataFrame(
        {
            "PassengerId": test_raw["PassengerId"].astype(int),
            "gender": b.gender,
            "wcg": b.wcg,
            "wcg_male": b.wcg_male,
            "wcg_female": b.wcg_female,
            "wcg_both": b.wcg_both,
            "deotte_score": b.score,
            "GroupSurvival": b.group_survival,
            "WCGExact": b.wcg_exact.astype(int),
        }
    )


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    fixed_summary, fixed_oof = fixed_fold_audit(train)
    fixed_summary.to_csv(EXPORT_DIR / "deotte_fixed_summary.csv", index=False)
    fixed_oof.to_csv(EXPORT_DIR / "deotte_fixed_oof.csv", index=False)

    repeated = repeated_audit(train)
    repeated.to_csv(EXPORT_DIR / "deotte_repeated_folds.csv", index=False)
    repeated_summary = (
        repeated.groupby("variant", as_index=False)
        .agg(mean_accuracy=("accuracy", "mean"), std_accuracy=("accuracy", "std"), min_accuracy=("accuracy", "min"), max_accuracy=("accuracy", "max"))
        .sort_values("mean_accuracy", ascending=False)
    )
    repeated_summary.to_csv(EXPORT_DIR / "deotte_repeated_summary.csv", index=False)

    tp = test_predictions(train, test)
    tp.to_csv(EXPORT_DIR / "deotte_test_predictions.csv", index=False)
    for v in ["gender", "wcg", "wcg_male", "wcg_female", "wcg_both"]:
        pd.DataFrame({"PassengerId": tp["PassengerId"], "Survived": tp[v].astype(int)}).to_csv(
            SUB_DIR / f"submission_v21_deotte_{v}.csv", index=False
        )

    # Historical exact public WCG+XGBoost candidate. Start from the deterministic
    # WCG output, then apply the exact R notebook's four male-live and ten
    # solo-female-perish XGBoost decisions.
    exact = tp["wcg"].astype(int).to_numpy().copy()
    pid_to_pos = {int(pid): i for i, pid in enumerate(tp["PassengerId"].astype(int))}
    for pid in DEOTTE_EXACT_MALE_LIVE:
        exact[pid_to_pos[pid]] = 1
    for pid in DEOTTE_EXACT_FEMALE_PERISH:
        exact[pid_to_pos[pid]] = 0
    tp["wcg_xgb_exact_public"] = exact
    pd.DataFrame({"PassengerId": tp["PassengerId"].astype(int), "Survived": exact}).to_csv(
        SUB_DIR / "submission_v21_deotte_wcg_xgb_exact_public.csv", index=False
    )
    tp.to_csv(EXPORT_DIR / "deotte_test_predictions.csv", index=False)

    v10_path = SUB_DIR / "submission_v10_score_0.81578.csv"
    comparisons = []
    if v10_path.exists():
        v10 = pd.read_csv(v10_path).sort_values("PassengerId")
        for v in ["gender", "wcg", "wcg_male", "wcg_female", "wcg_both"]:
            p = tp.sort_values("PassengerId")[v].to_numpy(dtype=int)
            comparisons.append(
                {
                    "variant": v,
                    "test_positives": int(p.sum()),
                    "changed_vs_v10": int(np.sum(p != v10["Survived"].to_numpy(dtype=int))),
                }
            )
        p = exact
        comparisons.append(
            {
                "variant": "wcg_xgb_exact_public",
                "test_positives": int(p.sum()),
                "changed_vs_v10": int(np.sum(p != v10["Survived"].to_numpy(dtype=int))),
            }
        )
    pd.DataFrame(comparisons).to_csv(EXPORT_DIR / "deotte_vs_v10.csv", index=False)

    print("=== Deotte fixed-fold audit ===")
    print(fixed_summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n=== Deotte repeated stress ===")
    print(repeated_summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    if comparisons:
        print("\n=== Current-test comparison vs v10 ===")
        print(pd.DataFrame(comparisons).to_string(index=False))
    print("\nNo Kaggle submission was performed.")


if __name__ == "__main__":
    main()
