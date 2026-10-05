"""Probe MLP-PLR seed stability under the exact v5 fold-safe protocol."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from audit_group_survival import add_group_survival, build_base_frame, model_feature_columns
from model_zoo_screen_v5 import DATA_DIR, V5_DIR, load_reference_predictions, make_model, splits_from_manifest


SEEDS = [142, 242]
OUT_DIR = V5_DIR / "seed_probes"


def run_seed(seed: int, train_base, test_base, reference_oof, splits, model_cols):
    out_oof = OUT_DIR / f"MLP_PLR_seed{seed}__oof.csv"
    out_test = OUT_DIR / f"MLP_PLR_seed{seed}__test.csv"
    if out_oof.exists() and out_test.exists():
        oof = pd.read_csv(out_oof)["probability"].to_numpy()
        test = pd.read_csv(out_test)["probability"].to_numpy()
        return oof, test

    y = train_base["Survived"].astype(int).reset_index(drop=True)
    oof = np.zeros(len(train_base), dtype=float)
    test_probs = []
    for fold, tr_idx, va_idx in splits:
        tr = train_base.iloc[tr_idx].copy()
        va = train_base.iloc[va_idx].copy()
        te = test_base.copy()
        tr["GroupSurvival"] = add_group_survival(tr, tr, exclude_self=True).to_numpy()
        va["GroupSurvival"] = add_group_survival(tr, va, exclude_self=False).to_numpy()
        te["GroupSurvival"] = add_group_survival(tr, te, exclude_self=False).to_numpy()
        model = make_model("MLP_PLR", seed + fold)
        model.fit(tr[model_cols], y.iloc[tr_idx])
        oof[va_idx] = np.asarray(model.predict_proba(va[model_cols]))[:, 1]
        test_probs.append(np.asarray(model.predict_proba(te[model_cols]))[:, 1])

    test = np.mean(test_probs, axis=0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "PassengerId": train_base["PassengerId"],
            "fold": reference_oof["fold"],
            "Survived": y,
            "probability": oof,
        }
    ).to_csv(out_oof, index=False)
    pd.DataFrame(
        {
            "PassengerId": test_base["PassengerId"],
            "probability": test,
        }
    ).to_csv(out_test, index=False)
    return oof, test


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    train_base, test_base = build_base_frame(train_raw, test_raw, profile="v2")
    reference_oof, reference_test = load_reference_predictions()
    splits = splits_from_manifest(train_base, reference_oof)
    model_cols = model_feature_columns(train_base)
    y = train_base["Survived"].astype(int).to_numpy()

    base = pd.read_csv(V5_DIR / "models" / "MLP_PLR__oof.csv")["probability"].to_numpy()
    base_test = pd.read_csv(V5_DIR / "models" / "MLP_PLR__test.csv")["probability"].to_numpy()

    seed_probs = {42: base}
    seed_test = {42: base_test}
    rows = []
    for seed in SEEDS:
        prob, test_prob = run_seed(
            seed, train_base, test_base, reference_oof, splits, model_cols
        )
        seed_probs[seed] = prob
        seed_test[seed] = test_prob

    for seed, prob in seed_probs.items():
        rows.append(
            {
                "seed": seed,
                "accuracy": accuracy_score(y, prob > 0.5),
                "roc_auc": roc_auc_score(y, prob),
            }
        )

    mean_prob = np.mean(list(seed_probs.values()), axis=0)
    mean_test = np.mean(list(seed_test.values()), axis=0)
    rows.append(
        {
            "seed": "mean_3_seeds",
            "accuracy": accuracy_score(y, mean_prob > 0.5),
            "roc_auc": roc_auc_score(y, mean_prob),
        }
    )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "seed_summary.csv", index=False)

    pd.DataFrame(
        {
            "PassengerId": train_base["PassengerId"],
            "fold": reference_oof["fold"],
            "Survived": y,
            "probability": mean_prob,
        }
    ).to_csv(OUT_DIR / "MLP_PLR_seed_mean__oof.csv", index=False)
    pd.DataFrame(
        {
            "PassengerId": test_base["PassengerId"],
            "probability": mean_test,
        }
    ).to_csv(OUT_DIR / "MLP_PLR_seed_mean__test.csv", index=False)

    with (OUT_DIR / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "seeds": [42] + SEEDS,
                "protocol": "same fixed 5-fold manifest + fold-safe WCG",
            },
            f,
            indent=2,
        )

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
