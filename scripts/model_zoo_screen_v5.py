"""Titanic v5 aggressive multi-family Model Zoo screening.

This expands the trusted v4 pipeline with a much broader set of model families:

Foundation:
  - TabPFN v3 / v2.5 / v2

Rules / interpretable:
  - RuleFit / FIGS

Modern tabular DL via pytabkit:
  - TabM / RealMLP / TabR / FT-Transformer / MLP-PLR / RTDL-MLP / ResNet

Other:
  - sklearn MLP / xRFM

All models use the exact saved five-fold manifest and fold-safe WCG construction.
Each model is persisted immediately, so the run is resumable and one model
failure does not discard completed work.

No Kaggle submission is performed by this script.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from audit_group_survival import add_group_survival, build_base_frame, model_feature_columns


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
V1_AUDIT = BASE_DIR / "exports" / "wcg_audit_v1"
V4_DIR = BASE_DIR / "exports" / "v4"
V5_DIR = BASE_DIR / "exports" / "v5"
MODEL_DIR = V5_DIR / "models"

DEFAULT_MODELS = [
    "TabPFN_v3",
    "TabPFN_v2_5",
    "TabPFN_v2",
    "RuleFit",
    "FIGS",
    "TabM",
    "RealMLP",
    "TabR",
    "FTTransformer",
    "MLP_PLR",
    "MLP_RTDL",
    "ResNet_RTDL",
    "MLP_SKL",
    "xRFM",
]


def parse_models(value: str) -> list[str]:
    if value.strip().lower() in {"all", "default"}:
        return DEFAULT_MODELS.copy()
    requested = [x.strip() for x in value.split(",") if x.strip()]
    unknown = [x for x in requested if x not in DEFAULT_MODELS]
    if unknown:
        raise ValueError(f"Unknown v5 models: {unknown}. Available: {DEFAULT_MODELS}")
    return requested


def load_reference_predictions():
    v1_oof = pd.read_csv(V1_AUDIT / "oof_fold_safe.csv")
    v1_test = pd.read_csv(V1_AUDIT / "test_group_survival_audit.csv")
    v4_oof = pd.read_csv(V4_DIR / "model_zoo_oof.csv")
    v4_test = pd.read_csv(V4_DIR / "model_zoo_test.csv")

    required_manifest = ["PassengerId", "fold", "Survived"]
    if not v1_oof[required_manifest].equals(v4_oof[required_manifest]):
        raise ValueError("v1 and v4 OOF manifests do not match.")
    if not v1_test["PassengerId"].equals(v4_test["PassengerId"]):
        raise ValueError("v1 and v4 test PassengerId order does not match.")

    out_oof = v1_oof[required_manifest].copy()
    out_oof["v1__Ensemble"] = v1_oof["fold_safe__Ensemble"]
    out_oof["TabICLv2"] = v4_oof["TabICLv2"]
    out_oof["EBM"] = v4_oof["EBM"]
    out_oof["v4b__Champion"] = 0.90 * out_oof["v1__Ensemble"] + 0.10 * out_oof["TabICLv2"]

    out_test = pd.DataFrame({"PassengerId": v1_test["PassengerId"]})
    out_test["v1__Ensemble"] = v1_test["fold_safe__Ensemble"]
    out_test["TabICLv2"] = v4_test["TabICLv2"]
    out_test["EBM"] = v4_test["EBM"]
    out_test["v4b__Champion"] = 0.90 * out_test["v1__Ensemble"] + 0.10 * out_test["TabICLv2"]
    return out_oof, out_test


def splits_from_manifest(train_base: pd.DataFrame, reference_oof: pd.DataFrame):
    if not train_base["PassengerId"].reset_index(drop=True).equals(
        reference_oof["PassengerId"].reset_index(drop=True)
    ):
        raise ValueError("Current train rows do not match saved fold manifest.")
    fold_values = reference_oof["fold"].astype(int).to_numpy()
    splits = []
    for fold in sorted(np.unique(fold_values)):
        val_idx = np.flatnonzero(fold_values == fold)
        train_idx = np.flatnonzero(fold_values != fold)
        splits.append((int(fold), train_idx, val_idx))
    return splits


def make_model(name: str, random_state: int):
    if name.startswith("TabPFN_"):
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        versions = {
            "TabPFN_v3": ModelVersion.V3,
            "TabPFN_v2_5": ModelVersion.V2_5,
            "TabPFN_v2": ModelVersion.V2,
        }
        return TabPFNClassifier.create_default_for_version(
            versions[name],
            device="cuda",
            random_state=random_state,
            show_progress_bar=False,
            fit_mode="fit_preprocessors",
        )

    if name == "RuleFit":
        from imodels import RuleFitClassifier

        return RuleFitClassifier(
            n_estimators=160,
            tree_size=4,
            max_rules=40,
            include_linear=True,
            random_state=random_state,
            verbose=0,
        )

    if name == "FIGS":
        from imodels import FIGSClassifier

        return FIGSClassifier(
            max_rules=16,
            max_trees=6,
            max_depth=5,
            random_state=random_state,
            n_jobs=-1,
        )

    from pytabkit.models.sklearn.sklearn_interfaces import (
        FTT_D_Classifier,
        MLP_PLR_D_Classifier,
        MLP_RTDL_D_Classifier,
        MLP_SKL_D_Classifier,
        RealMLP_TD_Classifier,
        RealTabR_D_Classifier,
        Resnet_RTDL_D_Classifier,
        TabM_D_Classifier,
        XRFM_D_Classifier,
    )

    common = dict(
        device="cuda",
        random_state=random_state,
        n_cv=1,
        n_refit=0,
        n_repeats=1,
        val_fraction=0.18,
        verbosity=0,
    )

    if name == "TabM":
        return TabM_D_Classifier(
            **common,
            n_epochs=100,
            patience=15,
            batch_size=128,
            compile_model=False,
            allow_amp=True,
        )
    if name == "RealMLP":
        return RealMLP_TD_Classifier(
            **common,
            n_epochs=128,
            batch_size=128,
            use_early_stopping=True,
        )
    if name == "TabR":
        cpu_common = dict(common)
        cpu_common["device"] = "cpu"
        return RealTabR_D_Classifier(
            **cpu_common,
            n_epochs=80,
            patience=12,
            batch_size=128,
            context_size=64,
        )
    if name == "FTTransformer":
        return FTT_D_Classifier(
            **common,
            max_epochs=80,
            es_patience=12,
            batch_size=128,
            use_checkpoints=False,
        )
    if name == "MLP_PLR":
        return MLP_PLR_D_Classifier(
            **common,
            max_epochs=100,
            es_patience=15,
            batch_size=128,
            use_checkpoints=False,
        )
    if name == "MLP_RTDL":
        return MLP_RTDL_D_Classifier(
            **common,
            max_epochs=100,
            es_patience=15,
            batch_size=128,
            use_checkpoints=False,
        )
    if name == "ResNet_RTDL":
        return Resnet_RTDL_D_Classifier(
            **common,
            max_epochs=100,
            es_patience=15,
            batch_size=128,
            use_checkpoints=False,
        )
    if name == "MLP_SKL":
        return MLP_SKL_D_Classifier(**common)
    if name == "xRFM":
        cpu_common = dict(common)
        cpu_common["device"] = "cpu"
        return XRFM_D_Classifier(
            **cpu_common,
            iters=5,
            max_leaf_samples=256,
            time_limit_s=120.0,
        )
    raise KeyError(name)


def save_model_artifacts(
    name: str,
    passenger_ids: pd.Series,
    folds: pd.Series,
    y: pd.Series,
    test_ids: pd.Series,
    oof_prob: np.ndarray,
    test_prob: np.ndarray,
    fold_rows: list[dict],
    elapsed: float,
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "PassengerId": passenger_ids,
            "fold": folds,
            "Survived": y,
            "probability": oof_prob,
        }
    ).to_csv(MODEL_DIR / f"{name}__oof.csv", index=False)
    pd.DataFrame(
        {
            "PassengerId": test_ids,
            "probability": test_prob,
        }
    ).to_csv(MODEL_DIR / f"{name}__test.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(MODEL_DIR / f"{name}__fold_metrics.csv", index=False)
    with (MODEL_DIR / f"{name}__metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": name,
                "elapsed_seconds": elapsed,
                "status": "complete",
            },
            f,
            indent=2,
        )


def load_saved_model(name: str, reference_oof: pd.DataFrame, reference_test: pd.DataFrame):
    oof_path = MODEL_DIR / f"{name}__oof.csv"
    test_path = MODEL_DIR / f"{name}__test.csv"
    metrics_path = MODEL_DIR / f"{name}__fold_metrics.csv"
    if not (oof_path.exists() and test_path.exists() and metrics_path.exists()):
        return None
    oof = pd.read_csv(oof_path)
    test = pd.read_csv(test_path)
    if not oof["PassengerId"].equals(reference_oof["PassengerId"]):
        raise ValueError(f"Saved {name} OOF PassengerId mismatch.")
    if not test["PassengerId"].equals(reference_test["PassengerId"]):
        raise ValueError(f"Saved {name} test PassengerId mismatch.")
    return oof["probability"].to_numpy(), test["probability"].to_numpy(), pd.read_csv(metrics_path)


def train_one_model(
    name: str,
    train_base: pd.DataFrame,
    test_base: pd.DataFrame,
    reference_oof: pd.DataFrame,
    reference_test: pd.DataFrame,
    splits,
    model_cols: list[str],
    random_state: int,
    force: bool,
):
    if not force:
        saved = load_saved_model(name, reference_oof, reference_test)
        if saved is not None:
            print(f"[resume] {name}: using saved artifacts", flush=True)
            return saved

    y = train_base["Survived"].astype(int).reset_index(drop=True)
    oof_prob = np.zeros(len(train_base), dtype=float)
    test_fold_probs = []
    fold_rows = []
    started = time.time()

    print(f"\n=== {name} ===", flush=True)
    for fold, train_idx, val_idx in splits:
        fold_train = train_base.iloc[train_idx].copy()
        fold_val = train_base.iloc[val_idx].copy()
        fold_test = test_base.copy()

        fold_train["GroupSurvival"] = add_group_survival(
            fold_train, fold_train, exclude_self=True
        ).to_numpy()
        fold_val["GroupSurvival"] = add_group_survival(
            fold_train, fold_val, exclude_self=False
        ).to_numpy()
        fold_test["GroupSurvival"] = add_group_survival(
            fold_train, fold_test, exclude_self=False
        ).to_numpy()

        X_train = fold_train[model_cols]
        X_val = fold_val[model_cols]
        X_test = fold_test[model_cols]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        fold_started = time.time()
        model = make_model(name, random_state + fold)
        model.fit(X_train, y_train)
        val_prob = np.asarray(model.predict_proba(X_val))[:, 1].astype(float)
        test_prob = np.asarray(model.predict_proba(X_test))[:, 1].astype(float)
        elapsed = time.time() - fold_started

        if not np.isfinite(val_prob).all() or not np.isfinite(test_prob).all():
            raise ValueError(f"{name} produced non-finite probabilities.")

        oof_prob[val_idx] = val_prob
        test_fold_probs.append(test_prob)
        row = {
            "model": name,
            "fold": fold,
            "accuracy": accuracy_score(y_val, val_prob > 0.5),
            "roc_auc": roc_auc_score(y_val, val_prob),
            "seconds": elapsed,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
        }
        fold_rows.append(row)
        print(
            f"fold {fold}: acc={row['accuracy']:.5f} auc={row['roc_auc']:.5f} "
            f"time={elapsed:.1f}s",
            flush=True,
        )

    test_prob = np.mean(test_fold_probs, axis=0)
    total_elapsed = time.time() - started
    save_model_artifacts(
        name,
        train_base["PassengerId"],
        reference_oof["fold"],
        y,
        test_base["PassengerId"],
        oof_prob,
        test_prob,
        fold_rows,
        total_elapsed,
    )
    return oof_prob, test_prob, pd.DataFrame(fold_rows)


def summarize(
    reference_oof: pd.DataFrame,
    reference_test: pd.DataFrame,
    model_results: dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]],
    failures: list[dict],
) -> None:
    V5_DIR.mkdir(parents=True, exist_ok=True)
    y = reference_oof["Survived"].astype(int).to_numpy()
    champ = reference_oof["v4b__Champion"].to_numpy()
    champ_pred = champ > 0.5
    champ_ok = champ_pred == y

    combined_oof = reference_oof.copy()
    combined_test = reference_test.copy()
    summary_rows = [
        {
            "model": "v4b__Champion",
            "family": "Champion blend",
            "accuracy": accuracy_score(y, champ_pred),
            "roc_auc": roc_auc_score(y, champ),
            "fold_accuracy_std": np.nan,
            "fold_auc_std": np.nan,
        }
    ]
    diversity_rows = []
    blend_rows = []

    family_map = {
        "TabPFN_v3": "Foundation",
        "TabPFN_v2_5": "Foundation",
        "TabPFN_v2": "Foundation",
        "RuleFit": "Rule/Interpretable",
        "FIGS": "Rule/Interpretable",
        "TabM": "Modern DL",
        "RealMLP": "Modern DL",
        "TabR": "Retrieval DL",
        "FTTransformer": "Transformer",
        "MLP_PLR": "Modern DL",
        "MLP_RTDL": "Modern DL",
        "ResNet_RTDL": "Modern DL",
        "MLP_SKL": "Classical NN",
        "xRFM": "Kernel/Feature",
    }

    for name, (prob, test_prob, fold_df) in model_results.items():
        combined_oof[name] = prob
        combined_test[name] = test_prob
        pred = prob > 0.5
        cand_ok = pred == y
        summary_rows.append(
            {
                "model": name,
                "family": family_map.get(name, "Other"),
                "accuracy": accuracy_score(y, pred),
                "roc_auc": roc_auc_score(y, prob),
                "fold_accuracy_std": fold_df["accuracy"].std(ddof=0),
                "fold_auc_std": fold_df["roc_auc"].std(ddof=0),
            }
        )
        diversity_rows.append(
            {
                "model": name,
                "probability_correlation_vs_v4b": float(np.corrcoef(champ, prob)[0, 1]),
                "binary_disagreement_count": int(np.sum(champ_pred != pred)),
                "v4b_wrong_candidate_right": int(np.sum((~champ_ok) & cand_ok)),
                "candidate_wrong_v4b_right": int(np.sum(champ_ok & (~cand_ok))),
                "net_unique_correct_vs_v4b": int(
                    np.sum((~champ_ok) & cand_ok) - np.sum(champ_ok & (~cand_ok))
                ),
            }
        )
        for w in (0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
            blend = (1.0 - w) * champ + w * prob
            blend_rows.append(
                {
                    "candidate": name,
                    "candidate_weight": w,
                    "champion_weight": 1.0 - w,
                    "accuracy": accuracy_score(y, blend > 0.5),
                    "roc_auc": roc_auc_score(y, blend),
                    "note": "diagnostic fixed-weight scan; same OOF, not unbiased weight optimization",
                }
            )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["accuracy", "roc_auc"], ascending=False
    )
    diversity = pd.DataFrame(diversity_rows).sort_values(
        ["net_unique_correct_vs_v4b", "v4b_wrong_candidate_right"],
        ascending=False,
    )
    blends = pd.DataFrame(blend_rows).sort_values(
        ["accuracy", "roc_auc"], ascending=False
    )
    corr_cols = ["v4b__Champion", "v1__Ensemble", "TabICLv2", "EBM"] + list(model_results)
    corr = combined_oof[corr_cols].corr()

    combined_oof.to_csv(V5_DIR / "model_zoo_oof.csv", index=False)
    combined_test.to_csv(V5_DIR / "model_zoo_test.csv", index=False)
    summary.to_csv(V5_DIR / "model_zoo_summary.csv", index=False)
    diversity.to_csv(V5_DIR / "diversity_vs_v4b.csv", index=False)
    blends.to_csv(V5_DIR / "fixed_blend_diagnostics.csv", index=False)
    corr.to_csv(V5_DIR / "pairwise_probability_correlation.csv")
    pd.DataFrame(failures).to_csv(V5_DIR / "failures.csv", index=False)

    print("\n=== v5 Model Zoo summary ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n=== Best fixed blend diagnostics ===")
    print(blends.head(20).to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n=== Diversity vs v4b ===")
    print(diversity.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    if failures:
        print("\n=== Failures ===")
        print(pd.DataFrame(failures)[["model", "error"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Titanic v5 aggressive model-zoo screening.")
    parser.add_argument("--models", default="all")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    selected = parse_models(args.models)
    V5_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    train_base, test_base = build_base_frame(train_raw, test_raw, profile="v2")
    reference_oof, reference_test = load_reference_predictions()
    splits = splits_from_manifest(train_base, reference_oof)
    model_cols = model_feature_columns(train_base)

    if args.check_only:
        print("CHECK_ONLY PASS")
        print("Models:", selected)
        print("Rows:", len(train_base), len(test_base))
        print("Features:", len(model_cols))
        print("Folds:", [(f, len(tr), len(va)) for f, tr, va in splits])
        return

    results = {}
    failures = []
    for name in selected:
        try:
            results[name] = train_one_model(
                name,
                train_base,
                test_base,
                reference_oof,
                reference_test,
                splits,
                model_cols,
                args.random_state,
                args.force,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            failures.append(
                {
                    "model": name,
                    "error": error,
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"[FAILED] {name}: {error}", flush=True)
            with (MODEL_DIR / f"{name}__failure.txt").open("w", encoding="utf-8") as f:
                f.write(traceback.format_exc())

    summarize(reference_oof, reference_test, results, failures)
    with (V5_DIR / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_models": selected,
                "completed_models": list(results),
                "failed_models": [x["model"] for x in failures],
                "random_state": args.random_state,
                "folds": 5,
                "feature_profile": "v2 + fold-safe WCG",
                "champion_reference": "v4b = 90% v1 + 10% TabICLv2",
                "notes": [
                    "Per-model artifacts make this run resumable.",
                    "Fixed blend scans are diagnostic only.",
                    "No Kaggle submission is performed.",
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
