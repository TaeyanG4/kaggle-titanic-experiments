"""v30: fixed-candidate outer-CV audit to isolate selection overhead.

Uses the same two outer seeds as v29 (31415, 27182), but does no inner model
selection. Every candidate is trained once per outer fold and evaluated on the
untouched outer validation. This answers whether the unstable inner selector is
worse than simply precommitting to one candidate.

No Kaggle submission is performed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from nested_selection_audit_v29 import CANDIDATES, candidate_predict
from partial_pooling_transfer_v28 import prepare_rel
from tabpfn_finalist_v8 import prepare_data


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "exports" / "v30"
OUTER_SEEDS = [31415, 27182]


def main():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    train_v2, _, _, base_cols = prepare_data()
    rel_train, _ = prepare_rel(train_raw, test_raw)
    y = train_raw["Survived"].astype(int).to_numpy()

    fold_rows = []
    seed_rows = []
    oof_store = {}
    for outer_seed in OUTER_SEEDS:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=outer_seed)
        fold_id = np.full(len(train_raw), -1, dtype=int)
        preds = {name: np.zeros(len(train_raw), dtype=int) for name in CANDIDATES}
        probs = {name: np.zeros(len(train_raw), dtype=float) for name in CANDIDATES}
        for fold, (tr_idx, va_idx) in enumerate(skf.split(train_raw, y)):
            fold_id[va_idx] = fold
            print(f"seed={outer_seed} fold={fold}", flush=True)
            for ci, name in enumerate(CANDIDATES):
                p, s = candidate_predict(
                    name,
                    train_raw,
                    train_v2,
                    rel_train,
                    base_cols,
                    tr_idx,
                    va_idx,
                    seed=outer_seed + fold * 100 + ci,
                )
                preds[name][va_idx] = p
                probs[name][va_idx] = s
                fold_rows.append({
                    "outer_seed": outer_seed,
                    "fold": fold,
                    "candidate": name,
                    "accuracy": accuracy_score(y[va_idx], p),
                    "roc_auc": roc_auc_score(y[va_idx], s),
                    "n_val": len(va_idx),
                })
        frame = pd.DataFrame({
            "PassengerId": train_raw["PassengerId"].astype(int),
            "Survived": y,
            "fold": fold_id,
        })
        for name in CANDIDATES:
            frame[name] = preds[name]
            frame[name + "_score"] = probs[name]
            seed_rows.append({
                "outer_seed": outer_seed,
                "candidate": name,
                "accuracy": accuracy_score(y, preds[name]),
                "roc_auc": roc_auc_score(y, probs[name]),
            })
        frame.to_csv(EXPORT_DIR / f"fixed_outer_oof_seed{outer_seed}.csv", index=False)
        oof_store[outer_seed] = frame

    fold_df = pd.DataFrame(fold_rows)
    seed_df = pd.DataFrame(seed_rows)
    fold_df.to_csv(EXPORT_DIR / "fixed_outer_fold_metrics.csv", index=False)
    seed_df.to_csv(EXPORT_DIR / "fixed_outer_seed_summary.csv", index=False)

    # Bring in the v5/nested selector benchmarks from v29 without retraining them.
    benchmark_rows = []
    for seed, path in [
        (31415, BASE_DIR / "exports" / "v29" / "nested_summary.csv"),
        (27182, BASE_DIR / "exports" / "v29_seed27182" / "nested_summary.csv"),
    ]:
        s = pd.read_csv(path)
        for _, row in s[s["candidate"].isin(["nested_selector", "v5_fixed_benchmark"])].iterrows():
            benchmark_rows.append({
                "outer_seed": seed,
                "candidate": row["candidate"],
                "accuracy": row["accuracy"],
                "roc_auc": row["roc_auc"],
            })
    all_seed = pd.concat([seed_df, pd.DataFrame(benchmark_rows)], ignore_index=True)
    all_seed.to_csv(EXPORT_DIR / "all_seed_summary.csv", index=False)

    overall = all_seed.groupby("candidate", as_index=False).agg(
        mean_accuracy=("accuracy", "mean"),
        min_accuracy=("accuracy", "min"),
        max_accuracy=("accuracy", "max"),
        std_accuracy=("accuracy", "std"),
        mean_auc=("roc_auc", "mean"),
    ).sort_values(["mean_accuracy", "min_accuracy"], ascending=False)
    overall.to_csv(EXPORT_DIR / "fixed_outer_overall.csv", index=False)

    print("\n=== fixed outer seed summary ===")
    print(all_seed.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\n=== overall ===")
    print(overall.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\nNo Kaggle submission was performed.")


if __name__ == "__main__":
    main()
