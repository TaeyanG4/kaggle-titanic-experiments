"""Fold-safe Titanic v4 Diverse Model Zoo screening.

The script trains only NEW model families. Existing v1/v2 tree predictions are
reused from their already-completed fold-safe audits so we do not spend compute
retraining the same models.

Default Stage-1 models intentionally favor diversity:
HistGradientBoosting, AdaBoost, LogisticRegression, RBF-SVM, KNN, LDA, QDA,
GaussianNB, plus EBM when installed.

Foundation models are opt-in because first use may download checkpoints:
TabPFN_v3, TabPFN_v2_5, TabICLv2.

No Kaggle submission is performed by this script.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from audit_group_survival import add_group_survival, build_base_frame, model_feature_columns


warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "exports" / "v4"
V1_AUDIT = BASE_DIR / "exports" / "wcg_audit_v1"
V2_AUDIT = BASE_DIR / "exports" / "wcg_audit_v2"

STAGE1 = [
    "HistGradientBoosting",
    "AdaBoost",
    "LogisticRegression",
    "RBF_SVM",
    "KNN",
    "LDA",
    "QDA",
    "GaussianNB",
    "EBM",
]
FOUNDATION = ["TabPFN_v3", "TabPFN_v2_5", "TabICLv2"]


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def build_sklearn_model(name: str, random_state: int):
    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=12,
            l2_regularization=1.0,
            random_state=random_state,
        )
    if name == "AdaBoost":
        return AdaBoostClassifier(
            n_estimators=180,
            learning_rate=0.035,
            random_state=random_state,
        )
    if name == "LogisticRegression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=3000, random_state=random_state),
        )
    if name == "RBF_SVM":
        return make_pipeline(
            StandardScaler(),
            SVC(C=1.0, gamma="scale", probability=True, random_state=random_state),
        )
    if name == "KNN":
        return make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=15, weights="distance", p=2),
        )
    if name == "LDA":
        return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    if name == "QDA":
        return make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis(reg_param=0.10))
    if name == "GaussianNB":
        return make_pipeline(StandardScaler(), GaussianNB(var_smoothing=1e-9))
    raise KeyError(name)


def build_optional_model(name: str):
    if name == "EBM":
        if not module_available("interpret"):
            raise ImportError("EBM requires package 'interpret'.")
        from interpret.glassbox import ExplainableBoostingClassifier

        return ExplainableBoostingClassifier(
            interactions=5,
            max_bins=64,
            max_rounds=3000,
            learning_rate=0.03,
            random_state=42,
        )

    if name in {"TabPFN_v3", "TabPFN_v2_5"}:
        if not module_available("tabpfn"):
            raise ImportError(f"{name} requires package 'tabpfn'.")
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        version_attr = "V3" if name == "TabPFN_v3" else "V2_5"
        version = getattr(ModelVersion, version_attr)
        return TabPFNClassifier.create_default_for_version(version)

    if name == "TabICLv2":
        if not module_available("tabicl"):
            raise ImportError("TabICLv2 requires package 'tabicl'.")
        from tabicl import TabICLClassifier

        return TabICLClassifier()

    raise KeyError(name)


def builder_for(name: str, random_state: int) -> Callable[[], object]:
    if name in STAGE1 and name != "EBM":
        return lambda: clone(build_sklearn_model(name, random_state))
    return lambda: build_optional_model(name)


def package_for(name: str) -> str:
    if name == "EBM":
        return "interpret"
    if name.startswith("TabPFN"):
        return "tabpfn"
    if name == "TabICLv2":
        return "tabicl"
    return "sklearn"


def model_is_available(name: str) -> bool:
    pkg = package_for(name)
    return module_available(pkg)


def parse_models(value: str) -> list[str]:
    supported = STAGE1 + FOUNDATION
    key = value.strip().lower()
    if key == "stage1":
        requested = STAGE1
    elif key == "foundation":
        requested = FOUNDATION
    elif key in {"all", "all-supported"}:
        requested = supported
    else:
        requested = [x.strip() for x in value.split(",") if x.strip()]
    unknown = [x for x in requested if x not in supported]
    if unknown:
        raise ValueError(f"Unknown v4 model(s): {unknown}. Supported now: {supported}")
    return requested


def load_reference_predictions():
    required = [
        V1_AUDIT / "oof_fold_safe.csv",
        V1_AUDIT / "test_group_survival_audit.csv",
        V2_AUDIT / "oof_fold_safe.csv",
        V2_AUDIT / "test_group_survival_audit.csv",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Required fold-safe audit artifacts missing: {missing}")

    v1_oof = pd.read_csv(required[0])
    v1_test = pd.read_csv(required[1])
    v2_oof = pd.read_csv(required[2])
    v2_test = pd.read_csv(required[3])

    manifest_cols = ["PassengerId", "fold", "Survived"]
    if not v1_oof[manifest_cols].equals(v2_oof[manifest_cols]):
        raise ValueError("v1/v2 OOF manifests are not identical.")

    reference_oof = v1_oof[manifest_cols].copy()
    reference_oof["v1__Ensemble"] = v1_oof["fold_safe__Ensemble"]
    reference_oof["v2__Ensemble"] = v2_oof["fold_safe__Ensemble"]
    for col in [c for c in v1_oof.columns if c.startswith("fold_safe__") and c != "fold_safe__Ensemble"]:
        reference_oof["v1__" + col.replace("fold_safe__", "")] = v1_oof[col]
    for col in [c for c in v2_oof.columns if c.startswith("fold_safe__") and c != "fold_safe__Ensemble"]:
        reference_oof["v2__" + col.replace("fold_safe__", "")] = v2_oof[col]

    reference_test = pd.DataFrame({"PassengerId": v1_test["PassengerId"]})
    if not v1_test["PassengerId"].equals(v2_test["PassengerId"]):
        raise ValueError("v1/v2 test PassengerId order differs.")
    reference_test["v1__Ensemble"] = v1_test["fold_safe__Ensemble"]
    reference_test["v2__Ensemble"] = v2_test["fold_safe__Ensemble"]
    return reference_oof, reference_test


def split_from_manifest(train_base: pd.DataFrame, reference_oof: pd.DataFrame):
    if not train_base["PassengerId"].reset_index(drop=True).equals(reference_oof["PassengerId"].reset_index(drop=True)):
        raise ValueError("Current train rows do not match the saved fold manifest.")
    folds = reference_oof["fold"].astype(int).to_numpy()
    splits = []
    for fold in sorted(np.unique(folds)):
        val_idx = np.flatnonzero(folds == fold)
        train_idx = np.flatnonzero(folds != fold)
        splits.append((int(fold), train_idx, val_idx))
    return splits


def run_screen(
    train_base: pd.DataFrame,
    test_base: pd.DataFrame,
    reference_oof: pd.DataFrame,
    reference_test: pd.DataFrame,
    selected_models: list[str],
    random_state: int,
) -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    y = train_base["Survived"].astype(int).reset_index(drop=True)
    model_cols = model_feature_columns(train_base)
    splits = split_from_manifest(train_base, reference_oof)

    available = [m for m in selected_models if model_is_available(m)]
    skipped = [m for m in selected_models if m not in available]
    if not available:
        raise RuntimeError(f"None of the requested models are installed. Requested: {selected_models}")

    print("Requested:", selected_models)
    print("Available:", available)
    if skipped:
        print("Skipped (missing dependency):", skipped)
    print(f"Feature profile: v2 ({len(model_cols)} numeric features, fold-safe WCG)")
    print("Fold manifest: reused from completed v1/v2 audit")

    oof_out = reference_oof.copy()
    test_out = reference_test.copy()
    fold_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    total_start = time.time()

    for model_name in available:
        model_oof = np.zeros(len(train_base), dtype=float)
        fold_test_probs = []
        model_start = time.time()
        print(f"\n=== {model_name} ===", flush=True)

        for fold, train_idx, val_idx in splits:
            fold_train = train_base.iloc[train_idx].copy()
            fold_val = train_base.iloc[val_idx].copy()

            fold_train["GroupSurvival"] = add_group_survival(
                fold_train, fold_train, exclude_self=True
            ).to_numpy()
            fold_val["GroupSurvival"] = add_group_survival(
                fold_train, fold_val, exclude_self=False
            ).to_numpy()
            fold_test = test_base.copy()
            fold_test["GroupSurvival"] = add_group_survival(
                fold_train, fold_test, exclude_self=False
            ).to_numpy()

            X_train = fold_train[model_cols]
            X_val = fold_val[model_cols]
            X_test = fold_test[model_cols]
            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            fit_start = time.time()
            model = builder_for(model_name, random_state)()
            model.fit(X_train, y_train)
            val_prob = np.asarray(model.predict_proba(X_val))[:, 1]
            test_prob = np.asarray(model.predict_proba(X_test))[:, 1]
            elapsed = time.time() - fit_start

            model_oof[val_idx] = val_prob
            fold_test_probs.append(test_prob)
            fold_rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "accuracy": accuracy_score(y_val, (val_prob > 0.5).astype(int)),
                    "roc_auc": roc_auc_score(y_val, val_prob),
                    "fit_predict_seconds": elapsed,
                    "n_train": len(train_idx),
                    "n_val": len(val_idx),
                }
            )
            print(
                f"fold {fold}: acc={fold_rows[-1]['accuracy']:.5f} "
                f"auc={fold_rows[-1]['roc_auc']:.5f} time={elapsed:.1f}s",
                flush=True,
            )

        test_prob_avg = np.mean(fold_test_probs, axis=0)
        oof_out[model_name] = model_oof
        test_out[model_name] = test_prob_avg
        model_folds = pd.DataFrame([r for r in fold_rows if r["model"] == model_name])
        summary_rows.append(
            {
                "model": model_name,
                "family": (
                    "Foundation"
                    if model_name in FOUNDATION
                    else "Interpretable"
                    if model_name == "EBM"
                    else "Classical/Tree"
                ),
                "accuracy": accuracy_score(y, (model_oof > 0.5).astype(int)),
                "roc_auc": roc_auc_score(y, model_oof),
                "fold_accuracy_mean": model_folds["accuracy"].mean(),
                "fold_accuracy_std": model_folds["accuracy"].std(ddof=0),
                "fold_auc_mean": model_folds["roc_auc"].mean(),
                "fold_auc_std": model_folds["roc_auc"].std(ddof=0),
                "elapsed_seconds": time.time() - model_start,
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["accuracy", "roc_auc"], ascending=False
    )

    champion_prob = reference_oof["v1__Ensemble"].to_numpy()
    champion_pred = (champion_prob > 0.5).astype(int)
    y_np = y.to_numpy()
    diversity_rows = []
    blend_rows = []
    for model_name in available:
        prob = oof_out[model_name].to_numpy()
        pred = (prob > 0.5).astype(int)
        champ_ok = champion_pred == y_np
        cand_ok = pred == y_np
        diversity_rows.append(
            {
                "model": model_name,
                "probability_correlation_vs_v1": float(np.corrcoef(champion_prob, prob)[0, 1]),
                "binary_disagreement_count": int(np.sum(champion_pred != pred)),
                "v1_wrong_candidate_right": int(np.sum((~champ_ok) & cand_ok)),
                "candidate_wrong_v1_right": int(np.sum(champ_ok & (~cand_ok))),
                "both_wrong": int(np.sum((~champ_ok) & (~cand_ok))),
                "net_unique_correct_vs_v1": int(
                    np.sum((~champ_ok) & cand_ok) - np.sum(champ_ok & (~cand_ok))
                ),
            }
        )
        for candidate_weight in (0.10, 0.20, 0.30, 0.50):
            blend = (1.0 - candidate_weight) * champion_prob + candidate_weight * prob
            blend_rows.append(
                {
                    "base": "v1__Ensemble",
                    "candidate": model_name,
                    "candidate_weight": candidate_weight,
                    "v1_weight": 1.0 - candidate_weight,
                    "accuracy": accuracy_score(y_np, (blend > 0.5).astype(int)),
                    "roc_auc": roc_auc_score(y_np, blend),
                    "note": "diagnostic fixed-weight scan; do not treat best row as unbiased",
                }
            )

    diversity = pd.DataFrame(diversity_rows).sort_values(
        ["v1_wrong_candidate_right", "probability_correlation_vs_v1"],
        ascending=[False, True],
    )
    blends = pd.DataFrame(blend_rows)

    corr_cols = ["v1__Ensemble", "v2__Ensemble"] + available
    corr = oof_out[corr_cols].corr()

    fold_df.to_csv(EXPORT_DIR / "model_zoo_fold_metrics.csv", index=False)
    summary.to_csv(EXPORT_DIR / "model_zoo_summary.csv", index=False)
    oof_out.to_csv(EXPORT_DIR / "model_zoo_oof.csv", index=False)
    test_out.to_csv(EXPORT_DIR / "model_zoo_test.csv", index=False)
    diversity.to_csv(EXPORT_DIR / "diversity_vs_v1.csv", index=False)
    blends.to_csv(EXPORT_DIR / "fixed_blend_diagnostics.csv", index=False)
    corr.to_csv(EXPORT_DIR / "pairwise_probability_correlation.csv")

    metadata = {
        "version": "v4",
        "profile": "v2-features-with-fold-safe-wcg",
        "random_state": random_state,
        "fold_manifest_source": str(V1_AUDIT / "fold_manifest_seed42.csv"),
        "requested_models": selected_models,
        "trained_models": available,
        "skipped_models": skipped,
        "feature_count": len(model_cols),
        "feature_columns": model_cols,
        "elapsed_seconds": round(time.time() - total_start, 3),
        "notes": [
            "Existing v1/v2 tree OOF predictions are reused rather than retrained.",
            "All new supervised GroupSurvival features are constructed inside each fold.",
            "Fixed blend rows are diagnostic only; selecting the best row on the same OOF is optimistic.",
            "No Kaggle submission is performed.",
        ],
    }
    with (EXPORT_DIR / "model_zoo_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n=== v4 new-model OOF summary ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n=== Diversity vs v1 champion ===")
    print(diversity.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\nArtifacts written to: {EXPORT_DIR}")
    print("No Kaggle submission was performed.")


def run_check_only(
    train_base: pd.DataFrame,
    test_base: pd.DataFrame,
    reference_oof: pd.DataFrame,
    selected_models: list[str],
) -> None:
    model_cols = model_feature_columns(train_base)
    splits = split_from_manifest(train_base, reference_oof)
    fold, train_idx, val_idx = splits[0]
    fold_train = train_base.iloc[train_idx].copy()
    fold_val = train_base.iloc[val_idx].copy()
    fold_train["GroupSurvival"] = add_group_survival(
        fold_train, fold_train, exclude_self=True
    ).to_numpy()
    fold_val["GroupSurvival"] = add_group_survival(
        fold_train, fold_val, exclude_self=False
    ).to_numpy()
    if fold_train[model_cols].isna().any().any() or fold_val[model_cols].isna().any().any():
        raise AssertionError("v4 model matrix contains NaN.")
    print("CHECK_ONLY PASS")
    print(f"rows train/test: {len(train_base)}/{len(test_base)}")
    print(f"feature count: {len(model_cols)}")
    print(f"fold count: {len(splits)}, probe fold={fold}, n_train={len(train_idx)}, n_val={len(val_idx)}")
    for name in selected_models:
        print(f"{name:22s} available={model_is_available(name)} package={package_for(name)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Titanic v4 diverse model-zoo fold-safe screening.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--models", default="stage1", help="stage1, foundation, all-supported, or comma list")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    selected = parse_models(args.models)
    train_raw = pd.read_csv(args.data_dir / "train.csv")
    test_raw = pd.read_csv(args.data_dir / "test.csv")
    train_base, test_base = build_base_frame(train_raw, test_raw, profile="v2")
    reference_oof, reference_test = load_reference_predictions()

    if args.check_only:
        run_check_only(train_base, test_base, reference_oof, selected)
        return

    run_screen(
        train_base,
        test_base,
        reference_oof,
        reference_test,
        selected,
        args.random_state,
    )


if __name__ == "__main__":
    main()
