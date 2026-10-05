"""Cross-fitted logistic stacking probes using saved v5/v6 OOF predictions.

This is a diagnostic only. It does not train base models or create a Kaggle
submission.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
V5_DIR = BASE_DIR / "exports" / "v5"
V6_DIR = BASE_DIR / "exports" / "v6"


SETS = {
    "robust3": ["v4b", "rulefit", "mlpmean"],
    "plus_ffgb": ["v4b", "rulefit", "mlpmean", "ff_gb"],
    "plus_ff2": ["v4b", "rulefit", "mlpmean", "ff_gb", "ff_cat"],
    "plus_tabpfn_ff": ["v4b", "rulefit", "mlpmean", "tabpfn2", "ff_gb"],
    "all6": ["v4b", "rulefit", "mlpmean", "tabpfn2", "tabicl", "ff_gb"],
}


def main() -> None:
    v5 = pd.read_csv(V5_DIR / "model_zoo_oof.csv")
    ff = pd.read_csv(V6_DIR / "group_key_audit_oof.csv")
    sm = pd.read_csv(V5_DIR / "seed_probes" / "MLP_PLR_seed_mean__oof.csv")
    y = v5["Survived"].astype(int).to_numpy()
    folds = v5["fold"].astype(int).to_numpy()
    data = {
        "v4b": v5["v4b__Champion"].to_numpy(),
        "rulefit": v5["RuleFit"].to_numpy(),
        "mlpmean": sm["probability"].to_numpy(),
        "tabpfn2": v5["TabPFN_v2"].to_numpy(),
        "tabicl": v5["TabICLv2"].to_numpy(),
        "ff_gb": ff["familyfare_all__GradientBoosting"].to_numpy(),
        "ff_cat": ff["familyfare_ticket_all__CatBoost"].to_numpy(),
    }
    rows = []
    pred_export = pd.DataFrame({
        "PassengerId": v5["PassengerId"],
        "fold": folds,
        "Survived": y,
    })
    for name, cols in SETS.items():
        X = np.column_stack([data[c] for c in cols])
        pred = np.zeros(len(y), dtype=float)
        for fold in sorted(np.unique(folds)):
            tr = folds != fold
            va = folds == fold
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.0, max_iter=2000),
            )
            model.fit(X[tr], y[tr])
            pred[va] = model.predict_proba(X[va])[:, 1]
        pred_export[name] = pred
        rows.append({
            "stack": name,
            "members": "|".join(cols),
            "accuracy": accuracy_score(y, pred > 0.5),
            "roc_auc": roc_auc_score(y, pred),
            "fold0_accuracy": accuracy_score(y[folds == 0], pred[folds == 0] > 0.5),
            "fold1_accuracy": accuracy_score(y[folds == 1], pred[folds == 1] > 0.5),
            "fold2_accuracy": accuracy_score(y[folds == 2], pred[folds == 2] > 0.5),
            "fold3_accuracy": accuracy_score(y[folds == 3], pred[folds == 3] > 0.5),
            "fold4_accuracy": accuracy_score(y[folds == 4], pred[folds == 4] > 0.5),
        })
    result = pd.DataFrame(rows).sort_values(["accuracy", "roc_auc"], ascending=False)
    result.to_csv(V6_DIR / "stacking_probe.csv", index=False)
    pred_export.to_csv(V6_DIR / "stacking_probe_oof.csv", index=False)
    print(result.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
