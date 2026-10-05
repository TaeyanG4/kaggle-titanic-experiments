"""Consensus probe combining v5 with the strongest v14/v15 typed models.

Strong new members:
  - TabPFN v3 + v2base + FamilyFare + typed22
  - CatBoost + v2base + FamilyFare + typed22

The main hypothesis is conservative: keep v5 unless independent typed models
agree on the opposite class.

No Kaggle submission is performed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


BASE_DIR = Path(__file__).resolve().parents[1]
V5_DIR = BASE_DIR / "exports" / "v5"
V14_TAB = BASE_DIR / "exports" / "v14" / "tabpfn"
V15_DIR = BASE_DIR / "exports" / "v15"
SUBMISSION_DIR = BASE_DIR / "submissions"


def majority(*arrs):
    mat = np.column_stack(arrs)
    return (mat.sum(axis=1) >= (mat.shape[1] // 2 + 1)).astype(int)


def main() -> None:
    zoo = pd.read_csv(V5_DIR / "model_zoo_oof.csv")
    zoo_t = pd.read_csv(V5_DIR / "model_zoo_test.csv")
    mlp = pd.read_csv(V5_DIR / "seed_probes" / "MLP_PLR_seed_mean__oof.csv")
    mlp_t = pd.read_csv(V5_DIR / "seed_probes" / "MLP_PLR_seed_mean__test.csv")
    tab = pd.read_csv(V14_TAB / "tabpfn_typed_oof.csv")
    tab_t = pd.read_csv(V14_TAB / "tabpfn_typed_test.csv")
    tree = pd.read_csv(V15_DIR / "typed_tree_screen_oof.csv")
    tree_t = pd.read_csv(V15_DIR / "typed_tree_screen_test.csv")

    ids = zoo["PassengerId"]
    tids = zoo_t["PassengerId"]
    for d in [mlp, tab, tree]:
        if not ids.equals(d["PassengerId"]):
            raise ValueError("OOF PassengerId mismatch")
    for d in [mlp_t, tab_t, tree_t]:
        if not tids.equals(d["PassengerId"]):
            raise ValueError("Test PassengerId mismatch")

    y = zoo["Survived"].astype(int).to_numpy()
    folds = zoo["fold"].astype(int).to_numpy()

    p = {
        "v4b": zoo["v4b__Champion"].to_numpy(),
        "rule": zoo["RuleFit"].to_numpy(),
        "mlp": mlp["probability"].to_numpy(),
        "t3": tab["TabPFN_v3__v2base_familyfare_typed22"].to_numpy(),
        "t25": tab["TabPFN_v2_5__v2base_familyfare_typed22"].to_numpy(),
        "cat": tree["v2base_familyfare_typed22__CatBoost"].to_numpy(),
    }
    pt = {
        "v4b": zoo_t["v4b__Champion"].to_numpy(),
        "rule": zoo_t["RuleFit"].to_numpy(),
        "mlp": mlp_t["probability"].to_numpy(),
        "t3": tab_t["TabPFN_v3__v2base_familyfare_typed22"].to_numpy(),
        "t25": tab_t["TabPFN_v2_5__v2base_familyfare_typed22"].to_numpy(),
        "cat": tree_t["v2base_familyfare_typed22__CatBoost"].to_numpy(),
    }
    v = {k: (x > 0.5).astype(int) for k, x in p.items()}
    vt = {k: (x > 0.5).astype(int) for k, x in pt.items()}

    v5 = majority(v["v4b"], v["rule"], v["mlp"])
    v5t = majority(vt["v4b"], vt["rule"], vt["mlp"])
    base_acc = accuracy_score(y, v5)

    candidates = {}
    candidates["rule_mlp_cat"] = (
        majority(v["rule"], v["mlp"], v["cat"]),
        majority(vt["rule"], vt["mlp"], vt["cat"]),
        "hard_vote",
    )
    candidates["rule_t3_cat"] = (
        majority(v["rule"], v["t3"], v["cat"]),
        majority(vt["rule"], vt["t3"], vt["cat"]),
        "hard_vote",
    )
    candidates["mlp_t3_cat"] = (
        majority(v["mlp"], v["t3"], v["cat"]),
        majority(vt["mlp"], vt["t3"], vt["cat"]),
        "hard_vote",
    )
    candidates["vote5_v5_t3_cat"] = (
        majority(v["v4b"], v["rule"], v["mlp"], v["t3"], v["cat"]),
        majority(vt["v4b"], vt["rule"], vt["mlp"], vt["t3"], vt["cat"]),
        "hard_vote",
    )
    candidates["vote5_v5_t25_cat"] = (
        majority(v["v4b"], v["rule"], v["mlp"], v["t25"], v["cat"]),
        majority(vt["v4b"], vt["rule"], vt["mlp"], vt["t25"], vt["cat"]),
        "hard_vote",
    )

    # v5 -> typed consensus switches.
    def switch(base, a, b):
        out = base.copy()
        m = (a == b) & (a != base)
        out[m] = a[m]
        return out

    candidates["switch_t3_cat"] = (
        switch(v5, v["t3"], v["cat"]),
        switch(v5t, vt["t3"], vt["cat"]),
        "consensus_switch",
    )
    candidates["switch_t25_cat"] = (
        switch(v5, v["t25"], v["cat"]),
        switch(v5t, vt["t25"], vt["cat"]),
        "consensus_switch",
    )

    strict = v5.copy()
    strict_t = v5t.copy()
    m = (
        (v["t3"] == v["t25"])
        & (v["t3"] == v["cat"])
        & (v["t3"] != v5)
    )
    mt = (
        (vt["t3"] == vt["t25"])
        & (vt["t3"] == vt["cat"])
        & (vt["t3"] != v5t)
    )
    strict[m] = v["t3"][m]
    strict_t[mt] = vt["t3"][mt]
    candidates["switch_t3_t25_cat"] = (
        strict,
        strict_t,
        "consensus_switch",
    )

    # Cross-fitted logistic on the five strongest independent-ish probabilities.
    cols = ["v4b", "rule", "mlp", "t3", "cat"]
    X = np.column_stack([p[c] for c in cols])
    Xt = np.column_stack([pt[c] for c in cols])
    moof = np.zeros(len(y), dtype=float)
    mt_probs = []
    for f in sorted(np.unique(folds)):
        tr = folds != f
        va = folds == f
        model = LogisticRegression(
            C=0.1,
            solver="liblinear",
            max_iter=1000,
            random_state=42,
        )
        model.fit(X[tr], y[tr])
        moof[va] = model.predict_proba(X[va])[:, 1]
        mt_probs.append(model.predict_proba(Xt)[:, 1])
    mtest = np.mean(mt_probs, axis=0)
    candidates["stack5_crossfit"] = (
        (moof > 0.5).astype(int),
        (mtest > 0.5).astype(int),
        "crossfit_logistic",
    )

    rows = [{
        "candidate": "v5_robust",
        "kind": "baseline",
        "accuracy": base_acc,
        "delta_vs_v5": 0.0,
        "changed_oof_vs_v5": 0,
        "changed_test_vs_v5": 0,
        "fold_deltas_vs_v5": "[0, 0, 0, 0, 0]",
    }]
    oo = pd.DataFrame({"PassengerId": ids, "fold": folds, "Survived": y})
    tt = pd.DataFrame({"PassengerId": tids})
    oo["v5_robust"] = v5
    tt["v5_robust"] = v5t

    for name, (pred, predt, kind) in candidates.items():
        deltas = [
            int(
                np.sum(pred[folds == f] == y[folds == f])
                - np.sum(v5[folds == f] == y[folds == f])
            )
            for f in sorted(np.unique(folds))
        ]
        rows.append({
            "candidate": name,
            "kind": kind,
            "accuracy": accuracy_score(y, pred),
            "delta_vs_v5": accuracy_score(y, pred) - base_acc,
            "changed_oof_vs_v5": int(np.sum(pred != v5)),
            "changed_test_vs_v5": int(np.sum(predt != v5t)),
            "fold_deltas_vs_v5": str(deltas),
        })
        oo[name] = pred
        tt[name] = predt
        pd.DataFrame({
            "PassengerId": tids.astype(int),
            "Survived": predt.astype(int),
        }).to_csv(
            SUBMISSION_DIR / f"submission_v15_{name}.csv",
            index=False,
        )

    oo["stack5_probability"] = moof
    tt["stack5_probability"] = mtest
    summary = pd.DataFrame(rows).sort_values(
        ["accuracy", "changed_oof_vs_v5"],
        ascending=[False, True],
    )
    summary.to_csv(V15_DIR / "v15_consensus_summary.csv", index=False)
    oo.to_csv(V15_DIR / "v15_consensus_oof.csv", index=False)
    tt.to_csv(V15_DIR / "v15_consensus_test.csv", index=False)

    print("=== v15 consensus probe ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("stack5 AUC", f"{roc_auc_score(y, moof):.5f}")
    print("\nNo Kaggle submission was performed.")


if __name__ == "__main__":
    main()
