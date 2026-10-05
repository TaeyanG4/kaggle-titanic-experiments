"""v47: v38 champion plus the broader Deotte female-death guard.

The parent v38 contains the validated PassengerId 980 correction. The broader
v24 Deotte female-death artifact changes 13 rows relative to v10; four of those
are the conservative WCG female-death rows already used by v46, and nine are
additional Deotte female-death decisions.

This script only constructs and audits the artifact; submission is external.
"""

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parents[1]
SUB = BASE / "submissions"
EXP = BASE / "exports" / "v47"


def main():
    EXP.mkdir(parents=True, exist_ok=True)
    v10 = pd.read_csv(SUB / "submission_v10_score_0.81578.csv").sort_values("PassengerId").reset_index(drop=True)
    v38 = pd.read_csv(SUB / "submission_v38_v10_text_rescue.csv").sort_values("PassengerId").reset_index(drop=True)
    broad = pd.read_csv(SUB / "submission_v24_v10_deotte_all_deotte_female_death.csv").sort_values("PassengerId").reset_index(drop=True)

    ids = v10["PassengerId"].astype(int).to_numpy()
    base = v10["Survived"].astype(int).to_numpy()
    v38_pred = v38["Survived"].astype(int).to_numpy()
    broad_pred = broad["Survived"].astype(int).to_numpy()
    out = v38_pred.copy()
    guard = broad_pred != base
    out[guard] = broad_pred[guard]

    path = SUB / "submission_v47_v38_plus_all_deotte_female_guard.csv"
    pd.DataFrame({"PassengerId": ids, "Survived": out}).to_csv(path, index=False)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()

    detail = pd.DataFrame({
        "PassengerId": ids,
        "v10": base,
        "v38": v38_pred,
        "broad_deotte_guard": broad_pred,
        "v47": out,
    })
    detail["changed_vs_v10"] = (detail["v47"] != detail["v10"]).astype(int)
    detail["changed_vs_v38"] = (detail["v47"] != detail["v38"]).astype(int)
    detail.to_csv(EXP / "test_details.csv", index=False)

    summary = pd.DataFrame([{
        "changed_vs_v10": int(np.sum(out != base)),
        "changed_vs_v38": int(np.sum(out != v38_pred)),
        "test_positives": int(out.sum()),
        "sha256": sha,
    }])
    summary.to_csv(EXP / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(detail[detail["changed_vs_v10"] == 1].to_string(index=False))


if __name__ == "__main__":
    main()
