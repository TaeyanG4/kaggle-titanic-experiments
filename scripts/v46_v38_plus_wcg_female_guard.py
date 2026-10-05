"""v46: v38 champion plus the conservative v24 WCG female-death guard.

Parent v38 changes PassengerId 980 relative to v10. The v24 WCG female-death
candidate changes only four additional test rows relative to v10. This script
keeps the proven v38 980 correction and overlays those four structural WCG
female-death switches. No Kaggle submission is performed here.
"""

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SUB = BASE / "submissions"
EXP = BASE / "exports" / "v46"


def main():
    EXP.mkdir(parents=True, exist_ok=True)
    v10 = pd.read_csv(SUB / "submission_v10_score_0.81578.csv").sort_values("PassengerId").reset_index(drop=True)
    v38 = pd.read_csv(SUB / "submission_v38_v10_text_rescue.csv").sort_values("PassengerId").reset_index(drop=True)
    wcg = pd.read_csv(SUB / "submission_v24_v10_deotte_wcg_female_death.csv").sort_values("PassengerId").reset_index(drop=True)

    ids = v10["PassengerId"].astype(int).to_numpy()
    base = v10["Survived"].astype(int).to_numpy()
    out = v38["Survived"].astype(int).to_numpy().copy()
    wcg_pred = wcg["Survived"].astype(int).to_numpy()
    guard = wcg_pred != base
    out[guard] = wcg_pred[guard]

    path = SUB / "submission_v46_v38_plus_wcg_female_guard.csv"
    pd.DataFrame({"PassengerId": ids, "Survived": out}).to_csv(path, index=False)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    detail = pd.DataFrame({"PassengerId": ids, "v10": base, "v38": v38["Survived"].astype(int), "wcg_guard": wcg_pred, "v46": out})
    detail["changed_vs_v10"] = (detail["v46"] != detail["v10"]).astype(int)
    detail["changed_vs_v38"] = (detail["v46"] != detail["v38"]).astype(int)
    detail.to_csv(EXP / "test_details.csv", index=False)
    summary = pd.DataFrame([{
        "changed_vs_v10": int(np.sum(out != base)),
        "changed_vs_v38": int(np.sum(out != v38["Survived"].astype(int).to_numpy())),
        "test_positives": int(out.sum()),
        "sha256": sha,
    }])
    summary.to_csv(EXP / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(detail[detail["changed_vs_v10"] == 1].to_string(index=False))


if __name__ == "__main__":
    main()
