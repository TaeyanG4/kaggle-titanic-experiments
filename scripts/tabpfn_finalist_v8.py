"""Titanic TabPFN finalist screen for v2 / v2.5 / v3.

Purpose
-------
Run all TabPFN generations on the exact same leakage-safe five-fold protocol
and compare two representations:

1) baseline
   trusted v2 numeric/one-hot feature matrix + fold-safe GroupSurvival

2) familyfare
   baseline + fold-safe LastName+Fare peer-survival statistics

TabPFN v2 can run immediately. v2.5/v3 may require one-time Prior Labs
license/authentication before weights can be downloaded. A blocked model is
recorded and does not destroy completed artifacts.

No Kaggle submission is performed.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

try:
    from audit_group_survival import add_group_survival, build_base_frame, model_feature_columns
    from group_key_audit_v6 import helper_keys, peer_feature
except ModuleNotFoundError:
    from scripts.audit_group_survival import add_group_survival, build_base_frame, model_feature_columns
    from scripts.group_key_audit_v6 import helper_keys, peer_feature


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
V1_AUDIT = BASE_DIR / "exports" / "wcg_audit_v1"
V5_DIR = BASE_DIR / "exports" / "v5"
EXPORT_DIR = BASE_DIR / "exports" / "v8"
MODEL_DIR = EXPORT_DIR / "models"

SUPPORTED_MODELS = ["TabPFN_v2", "TabPFN_v2_5", "TabPFN_v3"]
SUPPORTED_VARIANTS = ["baseline", "familyfare"]


def parse_csv_arg(value: str, allowed: list[str]) -> list[str]:
    if value.strip().lower() == "all":
        return allowed.copy()
    out = [x.strip() for x in value.split(",") if x.strip()]
    unknown = [x for x in out if x not in allowed]
    if unknown:
        raise ValueError(f"Unknown values: {unknown}. Allowed: {allowed}")
    return out


def make_tabpfn(name: str, seed: int):
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    version_map = {
        "TabPFN_v2": ModelVersion.V2,
        "TabPFN_v2_5": ModelVersion.V2_5,
        "TabPFN_v3": ModelVersion.V3,
    }
    return TabPFNClassifier.create_default_for_version(
        version_map[name],
        device="cuda",
        random_state=seed,
        show_progress_bar=False,
        fit_mode="fit_preprocessors",
    )


def load_champion_context():
    zoo = pd.read_csv(V5_DIR / "model_zoo_oof.csv")
    zoo_test = pd.read_csv(V5_DIR / "model_zoo_test.csv")
    mlp = pd.read_csv(V5_DIR / "seed_probes" / "MLP_PLR_seed_mean__oof.csv")
    mlp_test = pd.read_csv(V5_DIR / "seed_probes" / "MLP_PLR_seed_mean__test.csv")
    if not zoo["PassengerId"].equals(mlp["PassengerId"]):
        raise ValueError("Champion OOF PassengerId mismatch.")
    if not zoo_test["PassengerId"].equals(mlp_test["PassengerId"]):
        raise ValueError("Champion test PassengerId mismatch.")

    vote_matrix = np.column_stack(
        [
            (zoo["v4b__Champion"].to_numpy() > 0.5).astype(int),
            (zoo["RuleFit"].to_numpy() > 0.5).astype(int),
            (mlp["probability"].to_numpy() > 0.5).astype(int),
        ]
    )
    champion_pred = (vote_matrix.sum(axis=1) >= 2).astype(int)
    champion_vote_fraction = vote_matrix.mean(axis=1)

    test_vote_matrix = np.column_stack(
        [
            (zoo_test["v4b__Champion"].to_numpy() > 0.5).astype(int),
            (zoo_test["RuleFit"].to_numpy() > 0.5).astype(int),
            (mlp_test["probability"].to_numpy() > 0.5).astype(int),
        ]
    )
    champion_test_pred = (test_vote_matrix.sum(axis=1) >= 2).astype(int)
    return zoo, zoo_test, champion_pred, champion_vote_fraction, champion_test_pred


def prepare_data():
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    train_base, test_base = build_base_frame(train_raw, test_raw, profile="v2")
    tr_help, te_help = helper_keys(train_raw, test_raw)
    train_base = train_base.merge(tr_help[["PassengerId", "FamilyFareGroup_v6"]], on="PassengerId", how="left")
    test_base = test_base.merge(te_help[["PassengerId", "FamilyFareGroup_v6"]], on="PassengerId", how="left")
    manifest = pd.read_csv(V1_AUDIT / "fold_manifest_seed42.csv")
    if not train_base["PassengerId"].equals(manifest["PassengerId"]):
        raise ValueError("Fold manifest PassengerId order mismatch.")
    folds = manifest["fold"].astype(int).to_numpy()
    base_cols = model_feature_columns(train_base.drop(columns=["FamilyFareGroup_v6"]))
    return train_base, test_base, folds, base_cols


def run_one(
    model_name: str,
    variant: str,
    train_base: pd.DataFrame,
    test_base: pd.DataFrame,
    folds: np.ndarray,
    base_cols: list[str],
):
    y = train_base["Survived"].astype(int).to_numpy()
    oof = np.zeros(len(train_base), dtype=float)
    test_probs = []
    fold_rows = []
    familyfare_cols = [
        "FamilyFareAny",
        "FamilyFareMean",
        "FamilyFareSmooth",
        "FamilyFareCount",
    ]

    for fold in sorted(np.unique(folds)):
        tr_idx = np.flatnonzero(folds != fold)
        va_idx = np.flatnonzero(folds == fold)
        tr = train_base.iloc[tr_idx].copy()
        va = train_base.iloc[va_idx].copy()
        te = test_base.copy()

        tr["GroupSurvival"] = add_group_survival(tr, tr, exclude_self=True).to_numpy()
        va["GroupSurvival"] = add_group_survival(tr, va, exclude_self=False).to_numpy()
        te["GroupSurvival"] = add_group_survival(tr, te, exclude_self=False).to_numpy()

        extra_cols = []
        if variant == "familyfare":
            tr_stats = peer_feature(
                tr,
                tr,
                group_col="FamilyFareGroup_v6",
                wcg_only=False,
                exclude_self=True,
            )
            va_stats = peer_feature(
                tr,
                va,
                group_col="FamilyFareGroup_v6",
                wcg_only=False,
                exclude_self=False,
            )
            te_stats = peer_feature(
                tr,
                te,
                group_col="FamilyFareGroup_v6",
                wcg_only=False,
                exclude_self=False,
            )
            for source_col, out_col in zip(
                ["Any", "Mean", "Smooth", "Count"], familyfare_cols
            ):
                tr[out_col] = tr_stats[source_col].to_numpy()
                va[out_col] = va_stats[source_col].to_numpy()
                te[out_col] = te_stats[source_col].to_numpy()
            extra_cols = familyfare_cols

        features = base_cols + extra_cols
        model = make_tabpfn(model_name, seed=42 + int(fold))
        model.fit(tr[features], y[tr_idx])
        va_prob = np.asarray(model.predict_proba(va[features]))[:, 1].astype(float)
        te_prob = np.asarray(model.predict_proba(te[features]))[:, 1].astype(float)
        oof[va_idx] = va_prob
        test_probs.append(te_prob)
        fold_rows.append(
            {
                "model": model_name,
                "variant": variant,
                "fold": int(fold),
                "accuracy": accuracy_score(y[va_idx], va_prob > 0.5),
                "roc_auc": roc_auc_score(y[va_idx], va_prob),
                "feature_count": len(features),
            }
        )
        print(
            f"{model_name}/{variant} fold {fold}: "
            f"acc={fold_rows[-1]['accuracy']:.5f} "
            f"auc={fold_rows[-1]['roc_auc']:.5f}",
            flush=True,
        )

    return oof, np.mean(test_probs, axis=0), pd.DataFrame(fold_rows)


def save_model_run(
    model_name: str,
    variant: str,
    train_base: pd.DataFrame,
    test_base: pd.DataFrame,
    folds: np.ndarray,
    oof: np.ndarray,
    test_prob: np.ndarray,
    fold_df: pd.DataFrame,
):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{model_name}__{variant}"
    pd.DataFrame(
        {
            "PassengerId": train_base["PassengerId"],
            "fold": folds,
            "Survived": train_base["Survived"].astype(int),
            "probability": oof,
        }
    ).to_csv(MODEL_DIR / f"{stem}__oof.csv", index=False)
    pd.DataFrame(
        {
            "PassengerId": test_base["PassengerId"],
            "probability": test_prob,
        }
    ).to_csv(MODEL_DIR / f"{stem}__test.csv", index=False)
    fold_df.to_csv(MODEL_DIR / f"{stem}__fold_metrics.csv", index=False)


def load_existing(model_name: str, variant: str, train_base, test_base):
    stem = f"{model_name}__{variant}"
    oof_path = MODEL_DIR / f"{stem}__oof.csv"
    test_path = MODEL_DIR / f"{stem}__test.csv"
    fold_path = MODEL_DIR / f"{stem}__fold_metrics.csv"
    if not (oof_path.exists() and test_path.exists() and fold_path.exists()):
        return None
    oof_df = pd.read_csv(oof_path)
    test_df = pd.read_csv(test_path)
    if not oof_df["PassengerId"].equals(train_base["PassengerId"]):
        raise ValueError(f"{stem} saved OOF PassengerId mismatch.")
    if not test_df["PassengerId"].equals(test_base["PassengerId"]):
        raise ValueError(f"{stem} saved test PassengerId mismatch.")
    return (
        oof_df["probability"].to_numpy(),
        test_df["probability"].to_numpy(),
        pd.read_csv(fold_path),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="all")
    parser.add_argument("--variants", default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    models = parse_csv_arg(args.models, SUPPORTED_MODELS)
    variants = parse_csv_arg(args.variants, SUPPORTED_VARIANTS)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_base, test_base, folds, base_cols = prepare_data()

    if args.check_only:
        import torch
        from tabpfn.constants import ModelVersion

        print("CHECK_ONLY PASS")
        print("models:", models)
        print("variants:", variants)
        print("rows:", len(train_base), len(test_base))
        print("base feature count:", len(base_cols))
        print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
        print("ModelVersion:", [x for x in dir(ModelVersion) if x.startswith("V")])
        return

    zoo, zoo_test, champion_pred, champion_vote_fraction, champion_test_pred = (
        load_champion_context()
    )
    y = train_base["Survived"].astype(int).to_numpy()
    completed = {}
    failures = []

    for model_name in models:
        for variant in variants:
            key = f"{model_name}__{variant}"
            try:
                existing = None if args.force else load_existing(
                    model_name, variant, train_base, test_base
                )
                if existing is None:
                    print(f"\n=== {key} ===", flush=True)
                    result = run_one(
                        model_name,
                        variant,
                        train_base,
                        test_base,
                        folds,
                        base_cols,
                    )
                    save_model_run(
                        model_name,
                        variant,
                        train_base,
                        test_base,
                        folds,
                        *result,
                    )
                else:
                    print(f"[resume] {key}", flush=True)
                    result = existing
                completed[key] = result
            except Exception as exc:
                failures.append(
                    {
                        "model": model_name,
                        "variant": variant,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    }
                )
                print(f"[BLOCKED/FAILED] {key}: {type(exc).__name__}: {exc}", flush=True)

    # Rehydrate every previously completed run before rebuilding aggregate
    # artifacts. This keeps the summary resume-safe when a later invocation
    # requests only blocked models (for example v2.5/v3 before license auth).
    for model_name in SUPPORTED_MODELS:
        for variant in SUPPORTED_VARIANTS:
            key = f"{model_name}__{variant}"
            if key in completed:
                continue
            existing = load_existing(model_name, variant, train_base, test_base)
            if existing is not None:
                completed[key] = existing

    summary_rows = []
    diversity_rows = []
    combined_oof = pd.DataFrame(
        {
            "PassengerId": train_base["PassengerId"],
            "fold": folds,
            "Survived": y,
            "champion_pred": champion_pred,
            "champion_vote_fraction": champion_vote_fraction,
        }
    )
    combined_test = pd.DataFrame(
        {
            "PassengerId": test_base["PassengerId"],
            "champion_pred": champion_test_pred,
        }
    )
    champ_correct = champion_pred == y

    for key, (oof, test_prob, fold_df) in completed.items():
        pred = (oof > 0.5).astype(int)
        cand_correct = pred == y
        model_name, variant = key.split("__", 1)
        summary_rows.append(
            {
                "model": model_name,
                "variant": variant,
                "accuracy": accuracy_score(y, pred),
                "roc_auc": roc_auc_score(y, oof),
                "fold_accuracy_std": fold_df["accuracy"].std(ddof=0),
                "fold_auc_std": fold_df["roc_auc"].std(ddof=0),
                "feature_count": int(fold_df["feature_count"].iloc[0]),
            }
        )
        diversity_rows.append(
            {
                "model": model_name,
                "variant": variant,
                "binary_disagreement_vs_champion": int(np.sum(pred != champion_pred)),
                "champion_wrong_candidate_right": int(
                    np.sum((~champ_correct) & cand_correct)
                ),
                "champion_right_candidate_wrong": int(
                    np.sum(champ_correct & (~cand_correct))
                ),
                "net_unique_correct_vs_champion": int(
                    np.sum((~champ_correct) & cand_correct)
                    - np.sum(champ_correct & (~cand_correct))
                ),
                "correlation_vs_v4b_probability": float(
                    np.corrcoef(oof, zoo["v4b__Champion"].to_numpy())[0, 1]
                ),
                "correlation_vs_rulefit": float(
                    np.corrcoef(oof, zoo["RuleFit"].to_numpy())[0, 1]
                ),
                "correlation_vs_mlp_seed_mean": float(
                    np.corrcoef(
                        oof,
                        pd.read_csv(
                            V5_DIR / "seed_probes" / "MLP_PLR_seed_mean__oof.csv"
                        )["probability"].to_numpy(),
                    )[0, 1]
                ),
            }
        )
        combined_oof[key] = oof
        combined_test[key] = test_prob

    summary = pd.DataFrame(summary_rows)
    if len(summary):
        summary = summary.sort_values(["accuracy", "roc_auc"], ascending=False)
    diversity = pd.DataFrame(diversity_rows)
    if len(diversity):
        diversity = diversity.sort_values(
            ["champion_wrong_candidate_right", "net_unique_correct_vs_champion"],
            ascending=False,
        )
    summary.to_csv(EXPORT_DIR / "tabpfn_finalist_summary.csv", index=False)
    diversity.to_csv(EXPORT_DIR / "tabpfn_finalist_diversity.csv", index=False)
    combined_oof.to_csv(EXPORT_DIR / "tabpfn_finalist_oof.csv", index=False)
    combined_test.to_csv(EXPORT_DIR / "tabpfn_finalist_test.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": f["model"],
                "variant": f["variant"],
                "error": f["error"],
            }
            for f in failures
        ]
    ).to_csv(EXPORT_DIR / "tabpfn_finalist_failures.csv", index=False)
    with (EXPORT_DIR / "tabpfn_finalist_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "requested_models": models,
                "requested_variants": variants,
                "completed": list(completed),
                "blocked_or_failed": [
                    {
                        "model": x["model"],
                        "variant": x["variant"],
                        "error": x["error"],
                    }
                    for x in failures
                ],
                "fold_manifest": str(V1_AUDIT / "fold_manifest_seed42.csv"),
                "champion_reference": "v5 robust hard vote, Public 0.79665, OOF Accuracy 0.85410",
                "no_submission": True,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== TabPFN finalist summary ===")
    print(summary.to_string(index=False) if len(summary) else "(no completed models)")
    print("\n=== Diversity vs v5 robust champion ===")
    print(diversity.to_string(index=False) if len(diversity) else "(none)")
    if failures:
        print("\nBlocked/failed:", [(x["model"], x["variant"]) for x in failures])
    print("No Kaggle submission was performed.")


if __name__ == "__main__":
    main()
