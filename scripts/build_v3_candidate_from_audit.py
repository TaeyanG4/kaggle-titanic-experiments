"""Build a local v3 submission candidate from completed fold-safe audit outputs.

No model training is performed here. The script consumes the already generated
v2 audit probabilities and creates a reproducible candidate plus comparison
artifacts against the preserved v1/v2 submissions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_DIR = BASE_DIR / "exports" / "wcg_audit_v2"
DEFAULT_EXPORT_DIR = BASE_DIR / "exports" / "v3"
DEFAULT_SUBMISSION = BASE_DIR / "submissions" / "submission_v3.csv"


def load_submission(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    expected = ["PassengerId", "Survived"]
    if list(df.columns) != expected:
        raise ValueError(f"{label}: unexpected columns {list(df.columns)}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v3 from fold-safe audit probabilities.")
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")

    test_path = args.audit_dir / "test_group_survival_audit.csv"
    summary_path = args.audit_dir / "cv_group_survival_summary.csv"
    paired_path = args.audit_dir / "cv_group_survival_paired_deltas.csv"
    if not test_path.exists() or not summary_path.exists() or not paired_path.exists():
        raise FileNotFoundError(
            "Required audit outputs are missing. Run: "
            "python scripts/audit_group_survival.py --profile v2"
        )

    test_probs = pd.read_csv(test_path)
    summary = pd.read_csv(summary_path)
    paired = pd.read_csv(paired_path)

    prob_col = "fold_safe__Ensemble"
    if prob_col not in test_probs.columns:
        raise KeyError(f"Missing required probability column: {prob_col}")

    candidate = pd.DataFrame(
        {
            "PassengerId": test_probs["PassengerId"].astype(int),
            "Survived": (test_probs[prob_col].to_numpy() > args.threshold).astype(int),
        }
    )

    if len(candidate) != 418:
        raise ValueError(f"Expected 418 rows, got {len(candidate)}")
    if candidate["PassengerId"].duplicated().any():
        raise ValueError("Duplicate PassengerId found")
    if not set(candidate["Survived"].unique()).issubset({0, 1}):
        raise ValueError("Survived must be binary")

    args.export_dir.mkdir(parents=True, exist_ok=True)
    args.submission.parent.mkdir(parents=True, exist_ok=True)
    candidate.to_csv(args.submission, index=False)

    probability_artifact = test_probs[["PassengerId", prob_col]].rename(
        columns={prob_col: "SurvivalProbability"}
    )
    probability_artifact["Prediction"] = candidate["Survived"]
    probability_artifact.to_csv(args.export_dir / "submission_v3_probabilities.csv", index=False)

    v1 = load_submission(BASE_DIR / "submissions" / "submission_v1_score_0.79186.csv", "v1")
    v2 = load_submission(BASE_DIR / "submissions" / "submission_v2.csv", "v2")

    comparisons = []
    diff_frame = candidate.rename(columns={"Survived": "v3_fold_safe"}).copy()
    for label, old in (("v1", v1), ("v2", v2)):
        if old is None:
            continue
        merged = candidate.merge(old, on="PassengerId", suffixes=("_v3", f"_{label}"))
        diff_count = int((merged["Survived_v3"] != merged[f"Survived_{label}"]).sum())
        comparisons.append(
            {
                "candidate": "v3_fold_safe",
                "reference": label,
                "different_predictions": diff_count,
                "same_predictions": int(len(merged) - diff_count),
                "v3_positive_count": int(merged["Survived_v3"].sum()),
                "reference_positive_count": int(merged[f"Survived_{label}"].sum()),
            }
        )
        diff_frame = diff_frame.merge(
            old.rename(columns={"Survived": label}), on="PassengerId", how="left"
        )

    if "v1" in diff_frame.columns:
        diff_frame["diff_vs_v1"] = diff_frame["v3_fold_safe"] != diff_frame["v1"]
    if "v2" in diff_frame.columns:
        diff_frame["diff_vs_v2"] = diff_frame["v3_fold_safe"] != diff_frame["v2"]

    pd.DataFrame(comparisons).to_csv(args.export_dir / "submission_v3_comparison.csv", index=False)
    diff_frame.to_csv(args.export_dir / "submission_v3_prediction_diff.csv", index=False)

    ens = summary[summary["model"] == "Ensemble"].set_index("variant")
    audit_metrics = pd.DataFrame(
        [
            {
                "metric": "accuracy",
                "no_wcg": ens.loc["no_wcg", "accuracy"],
                "fold_safe": ens.loc["fold_safe", "accuracy"],
                "global_loo": ens.loc["global_loo", "accuracy"],
            },
            {
                "metric": "roc_auc",
                "no_wcg": ens.loc["no_wcg", "roc_auc"],
                "fold_safe": ens.loc["fold_safe", "roc_auc"],
                "global_loo": ens.loc["global_loo", "roc_auc"],
            },
        ]
    )
    audit_metrics["fold_safe_minus_no_wcg"] = audit_metrics["fold_safe"] - audit_metrics["no_wcg"]
    audit_metrics["global_loo_minus_fold_safe"] = audit_metrics["global_loo"] - audit_metrics["fold_safe"]
    audit_metrics.to_csv(args.export_dir / "v3_audit_metrics.csv", index=False)

    print("=== v3 fold-safe local candidate ===")
    print(f"threshold: {args.threshold:.3f}")
    print(f"positive predictions: {int(candidate['Survived'].sum())} / {len(candidate)}")
    print(f"submission: {args.submission}")
    print("\n=== Audit metrics ===")
    print(audit_metrics.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    if comparisons:
        print("\n=== Prediction differences ===")
        print(pd.DataFrame(comparisons).to_string(index=False))
    print(f"\nArtifacts: {args.export_dir}")


if __name__ == "__main__":
    main()
