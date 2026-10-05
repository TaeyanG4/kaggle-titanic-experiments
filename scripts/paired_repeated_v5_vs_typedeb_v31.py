"""v31: paired repeated Stratified CV, v5 robust vs fixed typed22+EB CatBoost.

Completes the missing v5 benchmarks for seeds 123/777/2026 and combines them
with existing paired evidence for seeds 42/31415/27182. This prevents promoting
the promising v30 result merely because two new split seeds favored it.

No Kaggle submission is performed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from audit_group_survival import build_base_frame, model_feature_columns
from nested_selection_audit_v29 import v5_predict
from tabpfn_finalist_v8 import prepare_data


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "exports" / "v31"
SEEDS_TO_RUN = [123, 777, 2026]


def main():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    y = train_raw["Survived"].astype(int).to_numpy()
    train_v1, _ = build_base_frame(train_raw, test_raw, profile="v1")
    v1_cols = model_feature_columns(train_v1)
    train_v2, _, _, v2_cols = prepare_data()

    rows = []
    for seed in SEEDS_TO_RUN:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        pred = np.zeros(len(train_raw), dtype=int)
        score = np.zeros(len(train_raw), dtype=float)
        fold_rows = []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(train_raw, y)):
            print(f"v5 seed={seed} fold={fold}", flush=True)
            p, s = v5_predict(
                train_raw, test_raw, train_v1, train_v2, v1_cols, v2_cols,
                tr_idx, va_idx, seed=seed * 10 + fold,
            )
            pred[va_idx] = p; score[va_idx] = s
            fold_rows.append({
                "seed": seed, "fold": fold,
                "accuracy": accuracy_score(y[va_idx], p),
                "roc_auc": roc_auc_score(y[va_idx], s),
            })
        rows.append({
            "seed": seed,
            "v5_accuracy": accuracy_score(y, pred),
            "v5_auc": roc_auc_score(y, score),
            "v5_fold_std": float(pd.DataFrame(fold_rows)["accuracy"].std(ddof=0)),
        })
        pd.DataFrame({
            "PassengerId": train_raw["PassengerId"].astype(int),
            "Survived": y, "v5_pred": pred, "v5_score": score,
        }).to_csv(EXPORT_DIR / f"v5_oof_seed{seed}.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(EXPORT_DIR / f"v5_fold_metrics_seed{seed}.csv", index=False)

    newly = pd.DataFrame(rows)
    newly.to_csv(EXPORT_DIR / "v5_new_seed_summary.csv", index=False)

    # Existing typed+EB CatBoost results from v28 and v30.
    v28_fixed = pd.read_csv(BASE_DIR / "exports" / "v28" / "fixed_screen.csv")
    typed42 = float(v28_fixed.loc[v28_fixed["candidate"] == "familyfare_typed22_eb25__CatBoost", "accuracy"].iloc[0])
    v28_alt = pd.read_csv(BASE_DIR / "exports" / "v28" / "alternate_confirmation.csv")
    typed_alt = v28_alt[v28_alt["candidate"] == "familyfare_typed22_eb25__CatBoost"].set_index("seed")["accuracy"].to_dict()
    v30_seed = pd.read_csv(BASE_DIR / "exports" / "v30" / "fixed_outer_seed_summary.csv")
    typed_new = v30_seed[v30_seed["candidate"] == "typed_eb_cat"].set_index("outer_seed")["accuracy"].to_dict()

    v5_existing = {
        42: 0.8540965207631874,
        31415: float(pd.read_csv(BASE_DIR / "exports" / "v29" / "nested_summary.csv").query("candidate == 'v5_fixed_benchmark'")["accuracy"].iloc[0]),
        27182: float(pd.read_csv(BASE_DIR / "exports" / "v29_seed27182" / "nested_summary.csv").query("candidate == 'v5_fixed_benchmark'")["accuracy"].iloc[0]),
    }
    for _, r in newly.iterrows():
        v5_existing[int(r["seed"])] = float(r["v5_accuracy"])

    typed = {42: typed42, **{int(k): float(v) for k, v in typed_alt.items()}, **{int(k): float(v) for k, v in typed_new.items()}}
    comparison = []
    for seed in [42, 123, 777, 2026, 31415, 27182]:
        comparison.append({
            "seed": seed,
            "v5_accuracy": v5_existing[seed],
            "typed_eb_accuracy": typed[seed],
            "delta_typedeb_vs_v5": typed[seed] - v5_existing[seed],
        })
    comp = pd.DataFrame(comparison)
    comp.to_csv(EXPORT_DIR / "paired_seed_comparison.csv", index=False)
    summary = pd.DataFrame([{
        "n_seeds": len(comp),
        "v5_mean": float(comp["v5_accuracy"].mean()),
        "typed_eb_mean": float(comp["typed_eb_accuracy"].mean()),
        "mean_delta": float(comp["delta_typedeb_vs_v5"].mean()),
        "median_delta": float(comp["delta_typedeb_vs_v5"].median()),
        "positive_seeds": int((comp["delta_typedeb_vs_v5"] > 0).sum()),
        "nonnegative_seeds": int((comp["delta_typedeb_vs_v5"] >= 0).sum()),
        "negative_seeds": int((comp["delta_typedeb_vs_v5"] < 0).sum()),
        "min_delta": float(comp["delta_typedeb_vs_v5"].min()),
        "max_delta": float(comp["delta_typedeb_vs_v5"].max()),
    }])
    summary.to_csv(EXPORT_DIR / "paired_summary.csv", index=False)
    print("\n=== paired seed comparison ===")
    print(comp.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\n=== paired summary ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\nNo Kaggle submission was performed.")


if __name__ == "__main__":
    main()
