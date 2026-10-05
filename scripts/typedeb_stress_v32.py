"""v32: final stress audit for fixed typed22+EB25 CatBoost.

Checks the v31 finalist on:
  1) two Ticket+Family connected-component group-aware splits;
  2) the five existing matched pseudo-test splits;
  3) test prediction delta / headroom vs v5 and public champion v10.

No Kaggle submission is performed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

from feature_ablation_v6 import build_models as build_tree_models
from groupaware_validation_v23 import connected_groups
from partial_pooling_transfer_v28 import fold_features, prepare_rel
from pseudo_test_relational_v14 import build_pseudo_splits
from tabpfn_finalist_v8 import prepare_data


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "exports" / "v32"
SUB_DIR = BASE_DIR / "submissions"
VARIANT = "familyfare_typed22_eb25"


def fit_predict(train, rel_train, base_cols, y, tr_idx, va_idx, seed):
    tr, va, cols = fold_features(train, rel_train, tr_idx, va_idx, base_cols, VARIANT)
    model = build_tree_models(seed)["CatBoost"]
    model.fit(tr[cols], y[tr_idx])
    prob = np.asarray(model.predict_proba(va[cols]))[:, 1]
    return (prob > .5).astype(int), prob


def groupaware(train_raw, train, rel_train, base_cols, y):
    groups = connected_groups(train_raw)
    rows=[]; seed_oof={}
    for seed in [42,777]:
        sgkf=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
        pred=np.zeros(len(train),dtype=int)
        for fold,(tr_idx,va_idx) in enumerate(sgkf.split(train,y,groups)):
            p,_=fit_predict(train,rel_train,base_cols,y,tr_idx,va_idx,32000+seed+fold)
            pred[va_idx]=p
            rows.append({"seed":seed,"fold":fold,"accuracy":accuracy_score(y[va_idx],p)})
        seed_oof[seed]=pred
    summary=pd.DataFrame([
        {"seed":seed,"accuracy":accuracy_score(y,p)} for seed,p in seed_oof.items()
    ])
    return pd.DataFrame(rows),summary


def pseudo(train_raw,test_raw,train,rel_train,base_cols,y):
    _, splits=build_pseudo_splits(train_raw,test_raw)
    v19=pd.read_csv(BASE_DIR/"exports"/"v19"/"pseudotest_consensus_predictions.csv")
    rows=[]; detail=[]
    for split,(_,va_idx,_) in enumerate(splits):
        tr_idx=np.setdiff1d(np.arange(len(train)),va_idx)
        p,_=fit_predict(train,rel_train,base_cols,y,tr_idx,va_idx,33000+split)
        ref=v19[v19["split"]==split].reset_index(drop=True)
        if not np.array_equal(ref["PassengerId"].astype(int).to_numpy(),train.iloc[va_idx]["PassengerId"].astype(int).to_numpy()):
            raise ValueError("pseudo order mismatch")
        v5=ref["v5"].astype(int).to_numpy()
        yy=y[va_idx]
        rows.append({
            "split":split,
            "typed_eb_accuracy":accuracy_score(yy,p),
            "v5_accuracy":accuracy_score(yy,v5),
            "delta":accuracy_score(yy,p)-accuracy_score(yy,v5),
            "changed_vs_v5":int(np.sum(p!=v5)),
        })
        for i,pid in enumerate(train.iloc[va_idx]["PassengerId"].astype(int)):
            detail.append({"split":split,"PassengerId":int(pid),"Survived":int(yy[i]),"typed_eb":int(p[i]),"v5":int(v5[i])})
    return pd.DataFrame(rows),pd.DataFrame(detail)


def test_candidate(train_raw,test_raw,train,test,folds,rel_train,rel_test,base_cols,y):
    probs=[]
    for fold in sorted(np.unique(folds)):
        tr_idx=np.flatnonzero(folds!=fold)
        # fold_features assumes validation rows come from train, so build a combined
        # temporary frame where test rows are appended and select them as validation.
        combo=pd.concat([train,test],ignore_index=True,sort=False)
        rel_combo=pd.concat([rel_train,rel_test],ignore_index=True,sort=False)
        te_idx=np.arange(len(train),len(combo))
        tr,te,cols=fold_features(combo,rel_combo,tr_idx,te_idx,base_cols,VARIANT)
        model=build_tree_models(34000+int(fold))["CatBoost"]
        model.fit(tr[cols],y[tr_idx])
        probs.append(np.asarray(model.predict_proba(te[cols]))[:,1])
    prob=np.mean(probs,axis=0); pred=(prob>.5).astype(int)
    out=pd.DataFrame({"PassengerId":test_raw["PassengerId"].astype(int),"Survived":pred})
    out.to_csv(SUB_DIR/"submission_v32_typedeb_cat.csv",index=False)
    return out,prob


def main():
    EXPORT_DIR.mkdir(parents=True,exist_ok=True); SUB_DIR.mkdir(parents=True,exist_ok=True)
    train_raw=pd.read_csv(DATA_DIR/"train.csv"); test_raw=pd.read_csv(DATA_DIR/"test.csv")
    train,test,folds,base_cols=prepare_data(); rel_train,rel_test=prepare_rel(train_raw,test_raw)
    y=train["Survived"].astype(int).to_numpy()

    gf,gs=groupaware(train_raw,train,rel_train,base_cols,y)
    gf.to_csv(EXPORT_DIR/"groupaware_folds.csv",index=False); gs.to_csv(EXPORT_DIR/"groupaware_summary.csv",index=False)
    pm,pdeta=pseudo(train_raw,test_raw,train,rel_train,base_cols,y)
    pm.to_csv(EXPORT_DIR/"pseudo_summary.csv",index=False); pdeta.to_csv(EXPORT_DIR/"pseudo_predictions.csv",index=False)

    sub,prob=test_candidate(train_raw,test_raw,train,test,folds,rel_train,rel_test,base_cols,y)
    v5=pd.read_csv(SUB_DIR/"submission_v5_robust_vote.csv").sort_values("PassengerId").reset_index(drop=True)
    v10=pd.read_csv(SUB_DIR/"submission_v10_score_0.81578.csv").sort_values("PassengerId").reset_index(drop=True)
    cand=sub.sort_values("PassengerId").reset_index(drop=True)
    d5=int(np.sum(cand.Survived.to_numpy()!=v5.Survived.to_numpy()))
    d10=int(np.sum(cand.Survived.to_numpy()!=v10.Survived.to_numpy()))
    # Approx correct-count gap from known Public scores, using all 418 rows.
    v5_correct=round(0.79665*418); v10_correct=round(0.81578*418); gap=v10_correct-v5_correct
    # If candidate is viewed as a switch from v5, champion-beating requires 2w-d > gap.
    required_w=(d5+gap)/2.0
    headroom={
        "changed_vs_v5":d5,"changed_vs_v10":d10,"v10_minus_v5_correct_gap":gap,
        "required_correct_switches_to_beat_v10_strictly":float(np.floor(required_w)+1),
        "required_switch_precision":float((np.floor(required_w)+1)/d5) if d5 else None,
        "max_net_gain_vs_v5":d5,
        "test_positives":int(cand.Survived.sum()),
    }
    pd.DataFrame([headroom]).to_csv(EXPORT_DIR/"headroom.csv",index=False)
    pd.DataFrame({"PassengerId":cand.PassengerId,"probability":prob,"prediction":cand.Survived}).to_csv(EXPORT_DIR/"test_predictions.csv",index=False)

    print("=== group-aware ==="); print(gs.to_string(index=False,float_format=lambda x:f"{x:.5f}"))
    print("\n=== pseudo ==="); print(pm.to_string(index=False,float_format=lambda x:f"{x:.5f}"))
    print("\n=== headroom ==="); print(pd.DataFrame([headroom]).to_string(index=False))
    print("\nNo Kaggle submission was performed.")


if __name__=="__main__": main()
