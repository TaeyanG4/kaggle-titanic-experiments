"""v45: conservative Public-champion guard.

Parent: v38, which improved v10 by changing PassengerId 980 only.
Additional frozen rule: if the v43 P3-full bag disagrees with v10 and has
confidence >= 0.90, use the P3 label. On the real test this changes only
PassengerId 1122, so v45 differs from v10 on exactly two rows: 980 and 1122.

The 0.90 P3 gate was checked on saved leakage-safe local surfaces before this
artifact was created. No Kaggle submission is performed here.
"""

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SUB = BASE / "submissions"
EXP = BASE / "exports" / "v45"
THRESHOLD = 0.90


def main():
    EXP.mkdir(parents=True, exist_ok=True)
    v38 = pd.read_csv(SUB / "submission_v38_v10_text_rescue.csv").sort_values("PassengerId").reset_index(drop=True)
    v10 = pd.read_csv(SUB / "submission_v10_score_0.81578.csv").sort_values("PassengerId").reset_index(drop=True)
    p3 = pd.read_csv(BASE / "exports" / "v43" / "test_predictions.csv").sort_values("PassengerId").reset_index(drop=True)

    if not np.array_equal(v38["PassengerId"].to_numpy(), p3["PassengerId"].to_numpy()):
        raise ValueError("PassengerId mismatch")

    conf = np.maximum(p3["p3_mean_prob"].to_numpy(float), 1 - p3["p3_mean_prob"].to_numpy(float))
    p3_pred = p3["p3_mean_label"].astype(int).to_numpy()
    v10_pred = v10["Survived"].astype(int).to_numpy()
    out = v38["Survived"].astype(int).to_numpy().copy()
    guard = (p3_pred != v10_pred) & (conf >= THRESHOLD)
    out[guard] = p3_pred[guard]

    candidate = pd.DataFrame({"PassengerId": v38["PassengerId"].astype(int), "Survived": out})
    path = SUB / "submission_v45_v38_plus_p3_090.csv"
    candidate.to_csv(path, index=False)

    detail = p3[["PassengerId", "p3_mean_prob", "p3_mean_label", "p3_positive_seed_votes"]].copy()
    detail["v10"] = v10_pred
    detail["v38"] = v38["Survived"].astype(int).to_numpy()
    detail["p3_confidence"] = conf
    detail["p3_guard"] = guard.astype(int)
    detail["v45"] = out
    detail.to_csv(EXP / "test_details.csv", index=False)

    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    summary = pd.DataFrame([{
        "threshold": THRESHOLD,
        "changed_vs_v10": int(np.sum(out != v10_pred)),
        "changed_vs_v38": int(np.sum(out != v38["Survived"].astype(int).to_numpy())),
        "p3_guard_rows": int(guard.sum()),
        "test_positives": int(out.sum()),
        "sha256": sha,
    }])
    summary.to_csv(EXP / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(detail[detail["v45"] != detail["v10"]][["PassengerId","v10","v38","p3_mean_label","p3_mean_prob","p3_confidence","p3_positive_seed_votes","v45"]].to_string(index=False))


if __name__ == "__main__":
    main()
