"""Audit and selectively combine v10's transductive group signal with trusted models.

Why this exists
---------------
v10 is the public-score champion (0.81578) and uses a strong transductive
Family/Ticket mechanism:

  * identify surname/ticket groups that occur in both train and the inference set
  * compute their survival medians from training labels
  * expose Survival_Rate and Survival_Rate_NA to the RF

That is legal test-covariate awareness, but the historical notebook's local CV
was optimistic because it created target encodings before splitting.

This script builds a closer *fold-safe transductive analogue*:

  * for each validation fold, only groups appearing in BOTH fold-train and
    fold-validation are eligible;
  * rates use fold-train labels only;
  * train rows use self-excluded peer labels;
  * target-independent scaling remains transductive (train and inference
    partition scaled separately), matching the historical inference recipe.

The OOF analogue is then used to validate simple, predeclared selectors between
the trusted v5 prediction and the Gunes/transductive prediction. The same rules
are transferred to test, where the high-public v10 prediction stands in for the
transductive candidate.

No Kaggle submission is performed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from gunes_exact_foldsafe_v11 import structural_frames


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
V1_AUDIT = BASE_DIR / "exports" / "wcg_audit_v1"
V5_DIR = BASE_DIR / "exports" / "v5"
V8_DIR = BASE_DIR / "exports" / "v8"
V10_DIR = BASE_DIR / "exports" / "v10"
V12_DIR = BASE_DIR / "exports" / "v12"
EXPORT_DIR = BASE_DIR / "exports" / "v13"
SUBMISSION_DIR = BASE_DIR / "submissions"


RF_CFG = dict(
    n_estimators=1750,
    max_depth=7,
    min_samples_split=6,
    min_samples_leaf=6,
)


def make_rf():
    return RandomForestClassifier(
        criterion="gini",
        **RF_CFG,
        max_features="sqrt",
        oob_score=True,
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )


def eligible_groups(
    reference: pd.DataFrame,
    inference: pd.DataFrame,
) -> tuple[set[str], set[str]]:
    """Historical eligibility, but relative to the current inference partition."""
    inference_families = set(inference["Family"])
    inference_tickets = set(inference["Ticket"])
    fam_stats = reference.groupby("Family")["Family_Size"].median()
    ticket_stats = reference.groupby("Ticket")["Ticket_Frequency"].median()
    fam = {
        g
        for g, size in fam_stats.items()
        if g in inference_families and float(size) > 1
    }
    ticket = {
        g
        for g, freq in ticket_stats.items()
        if g in inference_tickets and float(freq) > 1
    }
    return fam, ticket


def peer_rate(
    reference: pd.DataFrame,
    apply_df: pd.DataFrame,
    *,
    group_col: str,
    allowed_groups: set[str],
    exclude_self: bool,
    fallback: float,
) -> tuple[np.ndarray, np.ndarray]:
    groups = reference.groupby(group_col, sort=False)
    rate = np.full(len(apply_df), fallback, dtype=float)
    available = np.zeros(len(apply_df), dtype=float)
    for pos, (_, row) in enumerate(apply_df.iterrows()):
        key = row[group_col]
        if key not in allowed_groups or key not in groups.groups:
            continue
        peers = reference.loc[groups.groups[key]]
        if exclude_self:
            peers = peers[peers["PassengerId"] != row["PassengerId"]]
        if len(peers) == 0:
            continue
        rate[pos] = float(peers["Survived"].median())
        available[pos] = 1.0
    return rate, available


def group_meta(
    reference: pd.DataFrame,
    apply_df: pd.DataFrame,
    allowed_families: set[str],
    allowed_tickets: set[str],
    *,
    exclude_self: bool,
) -> pd.DataFrame:
    fallback = float(reference["Survived"].mean())
    fam_rate, fam_avail = peer_rate(
        reference,
        apply_df,
        group_col="Family",
        allowed_groups=allowed_families,
        exclude_self=exclude_self,
        fallback=fallback,
    )
    tic_rate, tic_avail = peer_rate(
        reference,
        apply_df,
        group_col="Ticket",
        allowed_groups=allowed_tickets,
        exclude_self=exclude_self,
        fallback=fallback,
    )
    return pd.DataFrame(
        {
            "FamilyRate": fam_rate,
            "FamilyAvailable": fam_avail,
            "TicketRate": tic_rate,
            "TicketAvailable": tic_avail,
            "SurvivalRate": (fam_rate + tic_rate) / 2.0,
            "SurvivalRateNA": (fam_avail + tic_avail) / 2.0,
        },
        index=apply_df.index,
    )


def historical_partition_scale(x: np.ndarray) -> np.ndarray:
    """Mirror original separate scaler fit on each partition."""
    return StandardScaler().fit_transform(x)


def build_transductive_oof_and_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_cols: list[str],
    folds: np.ndarray,
):
    y = train_df["Survived"].astype(int).to_numpy()
    oof = np.zeros(len(train_df), dtype=float)
    oof_meta = pd.DataFrame(
        index=train_df.index,
        columns=[
            "FamilyRate",
            "FamilyAvailable",
            "TicketRate",
            "TicketAvailable",
            "SurvivalRate",
            "SurvivalRateNA",
        ],
        dtype=float,
    )
    test_probs = []
    fold_rows = []

    # Full-train test metadata is what the real v10-style inference sees.
    full_allowed_fam, full_allowed_ticket = eligible_groups(train_df, test_df)
    full_test_meta = group_meta(
        train_df,
        test_df,
        full_allowed_fam,
        full_allowed_ticket,
        exclude_self=False,
    ).reset_index(drop=True)

    for fold in sorted(np.unique(folds)):
        tr_idx = np.flatnonzero(folds != fold)
        va_idx = np.flatnonzero(folds == fold)
        tr = train_df.iloc[tr_idx].copy()
        va = train_df.iloc[va_idx].copy()

        # ---- OOF model: validation-aware, label-safe ----
        allowed_fam, allowed_ticket = eligible_groups(tr, va)
        tr_meta = group_meta(
            tr, tr, allowed_fam, allowed_ticket, exclude_self=True
        )
        va_meta = group_meta(
            tr, va, allowed_fam, allowed_ticket, exclude_self=False
        )
        oof_meta.iloc[va_idx] = va_meta.to_numpy()

        xtr = np.column_stack(
            [
                tr[base_cols].to_numpy(dtype=float),
                tr_meta[["SurvivalRate", "SurvivalRateNA"]].to_numpy(),
            ]
        )
        xva = np.column_stack(
            [
                va[base_cols].to_numpy(dtype=float),
                va_meta[["SurvivalRate", "SurvivalRateNA"]].to_numpy(),
            ]
        )
        xtr = historical_partition_scale(xtr)
        xva = historical_partition_scale(xva)
        model = make_rf()
        model.fit(xtr, y[tr_idx])
        va_prob = model.predict_proba(xva)[:, 1]
        oof[va_idx] = va_prob

        fold_rows.append(
            {
                "fold": int(fold),
                "accuracy": accuracy_score(y[va_idx], va_prob >= 0.5),
                "roc_auc": roc_auc_score(y[va_idx], va_prob),
                "validation_family_eligible_rows": int(
                    (va_meta["FamilyAvailable"] > 0).sum()
                ),
                "validation_ticket_eligible_rows": int(
                    (va_meta["TicketAvailable"] > 0).sum()
                ),
            }
        )

        # ---- Test model: actual-test-aware, still fold-train-label-only ----
        test_allowed_fam, test_allowed_ticket = eligible_groups(tr, test_df)
        tr_test_meta = group_meta(
            tr, tr, test_allowed_fam, test_allowed_ticket, exclude_self=True
        )
        te_meta = group_meta(
            tr, test_df, test_allowed_fam, test_allowed_ticket, exclude_self=False
        )
        xtr_test = np.column_stack(
            [
                tr[base_cols].to_numpy(dtype=float),
                tr_test_meta[["SurvivalRate", "SurvivalRateNA"]].to_numpy(),
            ]
        )
        xte = np.column_stack(
            [
                test_df[base_cols].to_numpy(dtype=float),
                te_meta[["SurvivalRate", "SurvivalRateNA"]].to_numpy(),
            ]
        )
        xtr_test = historical_partition_scale(xtr_test)
        xte = historical_partition_scale(xte)
        test_model = make_rf()
        test_model.fit(xtr_test, y[tr_idx])
        test_probs.append(test_model.predict_proba(xte)[:, 1])

    return (
        oof,
        np.mean(test_probs, axis=0),
        oof_meta.reset_index(drop=True),
        full_test_meta,
        pd.DataFrame(fold_rows),
    )


def load_trusted_predictions(train_ids: pd.Series, test_ids: pd.Series):
    zoo = pd.read_csv(V5_DIR / "model_zoo_oof.csv")
    zoo_test = pd.read_csv(V5_DIR / "model_zoo_test.csv")
    mlp = pd.read_csv(V5_DIR / "seed_probes" / "MLP_PLR_seed_mean__oof.csv")
    mlp_test = pd.read_csv(V5_DIR / "seed_probes" / "MLP_PLR_seed_mean__test.csv")
    v8 = pd.read_csv(V8_DIR / "tabpfn_finalist_oof.csv")
    v8_test = pd.read_csv(V8_DIR / "tabpfn_finalist_test.csv")
    v12 = pd.read_csv(V12_DIR / "tabpfn_fe_oof.csv")
    v12_test = pd.read_csv(V12_DIR / "tabpfn_fe_test.csv")

    for df, label in [(zoo, "zoo"), (mlp, "mlp"), (v8, "v8"), (v12, "v12")]:
        if not train_ids.reset_index(drop=True).equals(df["PassengerId"].reset_index(drop=True)):
            raise ValueError(f"OOF PassengerId mismatch: {label}")
    for df, label in [
        (zoo_test, "zoo_test"),
        (mlp_test, "mlp_test"),
        (v8_test, "v8_test"),
        (v12_test, "v12_test"),
    ]:
        if not test_ids.reset_index(drop=True).equals(df["PassengerId"].reset_index(drop=True)):
            raise ValueError(f"Test PassengerId mismatch: {label}")

    v5_vote_matrix = np.column_stack(
        [
            (zoo["v4b__Champion"].to_numpy() > 0.5).astype(int),
            (zoo["RuleFit"].to_numpy() > 0.5).astype(int),
            (mlp["probability"].to_numpy() > 0.5).astype(int),
        ]
    )
    v5 = (v5_vote_matrix.sum(axis=1) >= 2).astype(int)
    v5_vote_fraction = v5_vote_matrix.mean(axis=1)

    v5_test_vote_matrix = np.column_stack(
        [
            (zoo_test["v4b__Champion"].to_numpy() > 0.5).astype(int),
            (zoo_test["RuleFit"].to_numpy() > 0.5).astype(int),
            (mlp_test["probability"].to_numpy() > 0.5).astype(int),
        ]
    )
    v5_test = (v5_test_vote_matrix.sum(axis=1) >= 2).astype(int)
    v5_test_vote_fraction = v5_test_vote_matrix.mean(axis=1)

    return {
        "v5": v5,
        "v5_vote_fraction": v5_vote_fraction,
        "v5_test": v5_test,
        "v5_test_vote_fraction": v5_test_vote_fraction,
        "v4b": zoo["v4b__Champion"].to_numpy(),
        "v4b_test": zoo_test["v4b__Champion"].to_numpy(),
        "rule": zoo["RuleFit"].to_numpy(),
        "rule_test": zoo_test["RuleFit"].to_numpy(),
        "mlp": mlp["probability"].to_numpy(),
        "mlp_test": mlp_test["probability"].to_numpy(),
        "t3ff": v8["TabPFN_v3__familyfare"].to_numpy(),
        "t3ff_test": v8_test["TabPFN_v3__familyfare"].to_numpy(),
        "t3ffga": v12["TabPFN_v3__familyfare_gunes_all"].to_numpy(),
        "t3ffga_test": v12_test["TabPFN_v3__familyfare_gunes_all"].to_numpy(),
    }


def apply_selector(
    name: str,
    trusted: np.ndarray,
    trans: np.ndarray,
    tab: np.ndarray,
    meta: pd.DataFrame,
) -> np.ndarray:
    trans_pred = (trans >= 0.5).astype(int)
    tab_pred = (tab >= 0.5).astype(int)
    out = trusted.copy()
    disagree = trans_pred != trusted
    any_group = meta["SurvivalRateNA"].to_numpy() > 0
    both_group = meta["SurvivalRateNA"].to_numpy() >= 1.0
    extreme = np.isin(meta["SurvivalRate"].to_numpy(), [0.0, 1.0])
    tab_agrees_trans = tab_pred == trans_pred

    if name == "majority_v5_trans_tab":
        votes = trusted + trans_pred + tab_pred
        return (votes >= 2).astype(int)
    if name == "prefer_trans_any_group":
        out[disagree & any_group] = trans_pred[disagree & any_group]
    elif name == "prefer_trans_both_groups":
        out[disagree & both_group] = trans_pred[disagree & both_group]
    elif name == "prefer_trans_extreme_group":
        mask = disagree & any_group & extreme
        out[mask] = trans_pred[mask]
    elif name == "prefer_trans_group_tab_agree":
        mask = disagree & any_group & tab_agrees_trans
        out[mask] = trans_pred[mask]
    elif name == "prefer_trans_both_group_tab_agree":
        mask = disagree & both_group & tab_agrees_trans
        out[mask] = trans_pred[mask]
    elif name == "prefer_trans_extreme_tab_agree":
        mask = disagree & any_group & extreme & tab_agrees_trans
        out[mask] = trans_pred[mask]
    else:
        raise ValueError(name)
    return out


SELECTOR_NAMES = [
    "majority_v5_trans_tab",
    "prefer_trans_any_group",
    "prefer_trans_both_groups",
    "prefer_trans_extreme_group",
    "prefer_trans_group_tab_agree",
    "prefer_trans_both_group_tab_agree",
    "prefer_trans_extreme_tab_agree",
]


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    train_df, test_df, base_cols, _, _ = structural_frames(train_raw, test_raw)
    manifest = pd.read_csv(V1_AUDIT / "fold_manifest_seed42.csv")
    if not train_df["PassengerId"].astype(int).equals(manifest["PassengerId"].astype(int)):
        raise ValueError("Trusted fold manifest mismatch.")
    folds = manifest["fold"].astype(int).to_numpy()
    y = train_df["Survived"].astype(int).to_numpy()

    (
        trans_oof,
        trans_test,
        oof_meta,
        test_meta,
        fold_df,
    ) = build_transductive_oof_and_test(train_df, test_df, base_cols, folds)

    trusted = load_trusted_predictions(
        train_df["PassengerId"].astype(int),
        test_df["PassengerId"].astype(int),
    )

    v10_test_df = pd.read_csv(V10_DIR / "gunes_original_test_predictions.csv")
    v10_pred = v10_test_df["Survived"].astype(int).to_numpy()
    v10_prob = v10_test_df["probability"].to_numpy()

    trans_pred = (trans_oof >= 0.5).astype(int)
    trans_test_pred = (trans_test >= 0.5).astype(int)
    t3ff_pred = (trusted["t3ff"] >= 0.5).astype(int)
    t3ff_test_pred = (trusted["t3ff_test"] >= 0.5).astype(int)

    print(
        "Transductive fold-safe RF:",
        f"acc={accuracy_score(y, trans_pred):.5f}",
        f"auc={roc_auc_score(y, trans_oof):.5f}",
        flush=True,
    )

    # OOF disagreement slice audit.
    audit = pd.DataFrame(
        {
            "PassengerId": train_df["PassengerId"].astype(int),
            "fold": folds,
            "Survived": y,
            "v5": trusted["v5"],
            "trans_rf": trans_pred,
            "trans_prob": trans_oof,
            "tabpfn3_ff": t3ff_pred,
            "tabpfn3_ff_prob": trusted["t3ff"],
        }
    )
    audit = pd.concat([audit, oof_meta], axis=1)
    audit["v5_correct"] = (audit["v5"] == audit["Survived"]).astype(int)
    audit["trans_correct"] = (audit["trans_rf"] == audit["Survived"]).astype(int)
    audit["disagree"] = (audit["v5"] != audit["trans_rf"]).astype(int)

    slice_rows = []
    disagreement = audit[audit["disagree"] == 1].copy()
    conditions = {
        "all_disagreements": np.ones(len(disagreement), dtype=bool),
        "any_group": disagreement["SurvivalRateNA"].to_numpy() > 0,
        "both_groups": disagreement["SurvivalRateNA"].to_numpy() >= 1.0,
        "extreme_rate": np.isin(disagreement["SurvivalRate"].to_numpy(), [0.0, 1.0]),
        "tab_agrees_trans": disagreement["tabpfn3_ff"].to_numpy()
        == disagreement["trans_rf"].to_numpy(),
        "group_and_tab_agree": (
            (disagreement["SurvivalRateNA"].to_numpy() > 0)
            & (
                disagreement["tabpfn3_ff"].to_numpy()
                == disagreement["trans_rf"].to_numpy()
            )
        ),
    }
    for name, mask in conditions.items():
        g = disagreement.loc[mask]
        if len(g) == 0:
            continue
        slice_rows.append(
            {
                "slice": name,
                "n": len(g),
                "v5_accuracy": g["v5_correct"].mean(),
                "trans_accuracy": g["trans_correct"].mean(),
                "trans_minus_v5_correct": int(
                    g["trans_correct"].sum() - g["v5_correct"].sum()
                ),
            }
        )

    # Predeclared selector rules.
    selector_rows = []
    selector_oof = pd.DataFrame(
        {
            "PassengerId": train_df["PassengerId"].astype(int),
            "fold": folds,
            "Survived": y,
            "v5": trusted["v5"],
            "trans_rf": trans_pred,
        }
    )
    selector_test = pd.DataFrame(
        {
            "PassengerId": test_df["PassengerId"].astype(int),
            "v5": trusted["v5_test"],
            "v10": v10_pred,
        }
    )

    for name in SELECTOR_NAMES:
        oof_pred = apply_selector(
            name,
            trusted["v5"],
            trans_oof,
            trusted["t3ff"],
            oof_meta,
        )
        # Transfer the same rule to real test, replacing the analogue RF
        # candidate with the actual high-public v10 prediction.
        test_pred = apply_selector(
            name,
            trusted["v5_test"],
            v10_prob,
            trusted["t3ff_test"],
            test_meta,
        )
        fold_accs = [
            accuracy_score(y[folds == f], oof_pred[folds == f])
            for f in sorted(np.unique(folds))
        ]
        selector_rows.append(
            {
                "selector": name,
                "accuracy": accuracy_score(y, oof_pred),
                "delta_vs_v5": accuracy_score(y, oof_pred)
                - accuracy_score(y, trusted["v5"]),
                "changed_oof_vs_v5": int(np.sum(oof_pred != trusted["v5"])),
                "changed_test_vs_v5": int(
                    np.sum(test_pred != trusted["v5_test"])
                ),
                "changed_test_vs_v10": int(np.sum(test_pred != v10_pred)),
                "fold_accuracy_std": float(np.std(fold_accs)),
                "fold_deltas_vs_v5": json.dumps(
                    [
                        int(
                            np.sum(oof_pred[folds == f] == y[folds == f])
                            - np.sum(trusted["v5"][folds == f] == y[folds == f])
                        )
                        for f in sorted(np.unique(folds))
                    ]
                ),
            }
        )
        selector_oof[name] = oof_pred
        selector_test[name] = test_pred
        pd.DataFrame(
            {
                "PassengerId": test_df["PassengerId"].astype(int),
                "Survived": test_pred.astype(int),
            }
        ).to_csv(
            SUBMISSION_DIR / f"submission_v13_{name}.csv",
            index=False,
        )

    # Cross-fitted conservative logistic meta model with the group mechanism
    # explicitly exposed. This is a secondary check, not the primary selector.
    meta_features = np.column_stack(
        [
            trusted["v4b"],
            trusted["rule"],
            trusted["mlp"],
            trusted["t3ff"],
            trans_oof,
            oof_meta[
                [
                    "FamilyRate",
                    "FamilyAvailable",
                    "TicketRate",
                    "TicketAvailable",
                    "SurvivalRate",
                    "SurvivalRateNA",
                ]
            ].to_numpy(),
        ]
    )
    meta_test_features = np.column_stack(
        [
            trusted["v4b_test"],
            trusted["rule_test"],
            trusted["mlp_test"],
            trusted["t3ff_test"],
            v10_prob,
            test_meta[
                [
                    "FamilyRate",
                    "FamilyAvailable",
                    "TicketRate",
                    "TicketAvailable",
                    "SurvivalRate",
                    "SurvivalRateNA",
                ]
            ].to_numpy(),
        ]
    )
    meta_oof = np.zeros(len(y), dtype=float)
    meta_test_probs = []
    for fold in sorted(np.unique(folds)):
        fit = folds != fold
        val = folds == fold
        model = LogisticRegression(
            C=0.1,
            solver="liblinear",
            max_iter=1000,
            random_state=42,
        )
        model.fit(meta_features[fit], y[fit])
        meta_oof[val] = model.predict_proba(meta_features[val])[:, 1]
        meta_test_probs.append(model.predict_proba(meta_test_features)[:, 1])
    meta_test = np.mean(meta_test_probs, axis=0)
    meta_pred = (meta_oof >= 0.5).astype(int)
    meta_test_pred = (meta_test >= 0.5).astype(int)
    selector_rows.append(
        {
            "selector": "crossfit_logistic_group_meta",
            "accuracy": accuracy_score(y, meta_pred),
            "delta_vs_v5": accuracy_score(y, meta_pred)
            - accuracy_score(y, trusted["v5"]),
            "changed_oof_vs_v5": int(np.sum(meta_pred != trusted["v5"])),
            "changed_test_vs_v5": int(
                np.sum(meta_test_pred != trusted["v5_test"])
            ),
            "changed_test_vs_v10": int(np.sum(meta_test_pred != v10_pred)),
            "fold_accuracy_std": float(
                np.std(
                    [
                        accuracy_score(y[folds == f], meta_pred[folds == f])
                        for f in sorted(np.unique(folds))
                    ]
                )
            ),
            "fold_deltas_vs_v5": json.dumps(
                [
                    int(
                        np.sum(meta_pred[folds == f] == y[folds == f])
                        - np.sum(trusted["v5"][folds == f] == y[folds == f])
                    )
                    for f in sorted(np.unique(folds))
                ]
            ),
        }
    )
    selector_oof["crossfit_logistic_group_meta"] = meta_oof
    selector_test["crossfit_logistic_group_meta"] = meta_test
    pd.DataFrame(
        {
            "PassengerId": test_df["PassengerId"].astype(int),
            "Survived": meta_test_pred,
        }
    ).to_csv(
        SUBMISSION_DIR / "submission_v13_crossfit_logistic_group_meta.csv",
        index=False,
    )

    # Detailed actual-test conflict table: v10 vs trusted v5.
    conflict_mask = v10_pred != trusted["v5_test"]
    conflict = test_raw.loc[
        conflict_mask,
        [
            "PassengerId",
            "Pclass",
            "Name",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Ticket",
            "Fare",
            "Cabin",
            "Embarked",
        ],
    ].copy()
    conflict["Family"] = test_df.loc[conflict_mask, "Family"].to_numpy()
    conflict["FamilySize"] = test_df.loc[conflict_mask, "Family_Size"].to_numpy()
    conflict["v10"] = v10_pred[conflict_mask]
    conflict["v10_prob"] = v10_prob[conflict_mask]
    conflict["v5"] = trusted["v5_test"][conflict_mask]
    conflict["v5_vote_fraction"] = trusted["v5_test_vote_fraction"][
        conflict_mask
    ]
    conflict["TabPFN3_FF"] = t3ff_test_pred[conflict_mask]
    conflict["TabPFN3_FF_prob"] = trusted["t3ff_test"][conflict_mask]
    conflict["TabPFN3_FF_GunesAll"] = (
        trusted["t3ffga_test"][conflict_mask] >= 0.5
    ).astype(int)
    conflict["RuleFit_prob"] = trusted["rule_test"][conflict_mask]
    conflict["MLPMean_prob"] = trusted["mlp_test"][conflict_mask]
    conflict["FamilyRate"] = test_meta.loc[conflict_mask, "FamilyRate"].to_numpy()
    conflict["FamilyAvailable"] = test_meta.loc[
        conflict_mask, "FamilyAvailable"
    ].to_numpy()
    conflict["TicketRate"] = test_meta.loc[conflict_mask, "TicketRate"].to_numpy()
    conflict["TicketAvailable"] = test_meta.loc[
        conflict_mask, "TicketAvailable"
    ].to_numpy()
    conflict["SurvivalRate"] = test_meta.loc[
        conflict_mask, "SurvivalRate"
    ].to_numpy()
    conflict["SurvivalRateNA"] = test_meta.loc[
        conflict_mask, "SurvivalRateNA"
    ].to_numpy()
    conflict["TabPFN_agrees_v10"] = (
        conflict["TabPFN3_FF"] == conflict["v10"]
    ).astype(int)
    conflict["has_group_signal"] = (conflict["SurvivalRateNA"] > 0).astype(int)

    # Artifacts.
    pd.DataFrame(
        {
            "PassengerId": train_df["PassengerId"].astype(int),
            "fold": folds,
            "Survived": y,
            "transductive_rf_probability": trans_oof,
            "transductive_rf_pred": trans_pred,
        }
    ).to_csv(EXPORT_DIR / "transductive_rf_oof.csv", index=False)
    pd.DataFrame(
        {
            "PassengerId": test_df["PassengerId"].astype(int),
            "transductive_rf_probability": trans_test,
            "transductive_rf_pred": trans_test_pred,
            "v10_probability": v10_prob,
            "v10_pred": v10_pred,
        }
    ).to_csv(EXPORT_DIR / "transductive_rf_test.csv", index=False)
    oof_meta.to_csv(EXPORT_DIR / "transductive_group_meta_oof.csv", index=False)
    test_meta.to_csv(EXPORT_DIR / "transductive_group_meta_test.csv", index=False)
    fold_df.to_csv(EXPORT_DIR / "transductive_rf_fold_metrics.csv", index=False)
    audit.to_csv(EXPORT_DIR / "oof_disagreement_audit.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(
        EXPORT_DIR / "oof_disagreement_slices.csv", index=False
    )
    selector_summary = pd.DataFrame(selector_rows).sort_values(
        ["accuracy", "fold_accuracy_std"], ascending=[False, True]
    )
    selector_summary.to_csv(EXPORT_DIR / "selector_summary.csv", index=False)
    selector_oof.to_csv(EXPORT_DIR / "selector_oof.csv", index=False)
    selector_test.to_csv(EXPORT_DIR / "selector_test.csv", index=False)
    conflict.to_csv(EXPORT_DIR / "test_v10_v5_conflicts.csv", index=False)

    with (EXPORT_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "mechanism": "fold-safe transductive Family/Ticket group eligibility",
                "trusted_folds": str(V1_AUDIT / "fold_manifest_seed42.csv"),
                "rf_config": RF_CFG,
                "selector_rules": SELECTOR_NAMES,
                "test_candidate_mapping": "OOF transductive RF analogue -> actual v10 test prediction",
                "no_submission": True,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== OOF disagreement slices ===")
    print(pd.DataFrame(slice_rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== Selector ranking ===")
    print(selector_summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n=== Actual test conflicts: v10 vs v5 ===")
    print(
        conflict[
            [
                "PassengerId",
                "Sex",
                "Pclass",
                "Age",
                "Family",
                "Ticket",
                "v10",
                "v5",
                "TabPFN3_FF",
                "SurvivalRate",
                "SurvivalRateNA",
                "TabPFN_agrees_v10",
            ]
        ].to_string(index=False)
    )
    print(f"\nConflict rows: {len(conflict)}")
    print("No Kaggle submission was performed.")


if __name__ == "__main__":
    main()
