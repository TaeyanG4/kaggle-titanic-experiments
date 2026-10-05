"""Final promotion gate for the remaining Titanic submission slot.

Candidate: majority(v5 robust, Deotte WCG+XGB, typed FamilyFare RF6).

Why this exact trio:
  * v5: strongest long-standing trusted mixed-family baseline;
  * Deotte: strongest group-aware relational/domain model;
  * RF6: best alternate-confirmed classical HPO configuration on typed22.

The candidate already improved fixed OOF and both group-aware surfaces.  This
script adds the five pre-existing matched pseudo-test splits, then builds the
actual test candidate using the exact public Deotte R-notebook decisions and a
fold-averaged actual-test-aware RF6. No Kaggle submission is performed here.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from audit_group_survival import add_group_survival
from deotte_wcg_xgb_v21 import predict_split as deotte_predict_split
from group_key_audit_v6 import peer_feature
from gunes_exact_foldsafe_v11 import structural_frames
from pseudo_test_relational_v14 import VARIANTS as REL_VARIANTS, build_pseudo_splits, eligible_groups, relation_features, role_flags
from tabpfn_finalist_v8 import prepare_data


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "exports" / "v25"
SUB_DIR = BASE_DIR / "submissions"
V5_DIR = BASE_DIR / "exports" / "v5"
V19_DIR = BASE_DIR / "exports" / "v19"
V21_DIR = BASE_DIR / "exports" / "v21"
V23_DIR = BASE_DIR / "exports" / "v23"

TYPED_COLS = REL_VARIANTS["typed22"]
FAMILYFARE_COLS = ["FamilyFareAny", "FamilyFareMean", "FamilyFareSmooth", "FamilyFareCount"]
RF6_CFG = dict(n_estimators=500,max_depth=7,min_samples_split=8,min_samples_leaf=4,max_features=.7)


def make_rf(seed):
    return RandomForestClassifier(**RF6_CFG, random_state=seed, n_jobs=-1)


def add_familyfare(ref, app, exclude_self):
    stats = peer_feature(ref, app, group_col="FamilyFareGroup_v6", wcg_only=False, exclude_self=exclude_self)
    out = app.copy()
    for source,dest in zip(["Any","Mean","Smooth","Count"],FAMILYFARE_COLS):
        out[dest] = stats[source].to_numpy()
    return out


def add_rel(df, rel):
    out=df.copy()
    for c in TYPED_COLS:
        out[c]=rel[c].to_numpy()
    return out


def majority(a,b,c):
    return ((a.astype(int)+b.astype(int)+c.astype(int))>=2).astype(int)


def build_rf_features(train, rel_train, tr_idx, va_idx, base_cols):
    tr=train.iloc[tr_idx].copy(); va=train.iloc[va_idx].copy()
    tr["GroupSurvival"] = add_group_survival(tr,tr,exclude_self=True).to_numpy()
    va["GroupSurvival"] = add_group_survival(tr,va,exclude_self=False).to_numpy()
    tr_ff=add_familyfare(tr,tr,True)
    va_ff=add_familyfare(train.iloc[tr_idx].assign(GroupSurvival=tr["GroupSurvival"].to_numpy()),va,False)
    rr=rel_train.iloc[tr_idx].copy(); rv=rel_train.iloc[va_idx].copy()
    af,at=eligible_groups(rr,rv)
    tr_rel=relation_features(rr,rr,exclude_self=True,alpha=2.0,allowed_fam=af,allowed_tic=at)
    va_rel=relation_features(rr,rv,exclude_self=False,alpha=2.0,allowed_fam=af,allowed_tic=at)
    tr_ff=add_rel(tr_ff,tr_rel); va_ff=add_rel(va_ff,va_rel)
    cols=list(dict.fromkeys(base_cols+FAMILYFARE_COLS+TYPED_COLS))
    return tr_ff,va_ff,cols


def pseudo_audit(train_raw,test_raw,train,rel_train,base_cols):
    y=train["Survived"].astype(int).to_numpy()
    saved=pd.read_csv(V19_DIR/"pseudotest_consensus_predictions.csv")
    _,pseudo=build_pseudo_splits(train_raw,test_raw)
    rows=[]; details=[]
    for split,(_,va_idx,_) in enumerate(pseudo):
        tr_idx=np.setdiff1d(np.arange(len(train)),va_idx)
        s=saved[saved["split"]==split].reset_index(drop=True)
        expected=train.iloc[va_idx]["PassengerId"].astype(int).to_numpy()
        if not np.array_equal(s["PassengerId"].astype(int).to_numpy(),expected):
            raise ValueError(f"pseudo split {split} order mismatch")
        v5=s["v5"].astype(int).to_numpy()
        db=deotte_predict_split(train_raw,tr_idx,va_idx,seed=25000+split,imputer_fit_indices=tr_idx,special_link=True)
        de=db.wcg_both.astype(int)
        trf,vaf,cols=build_rf_features(train,rel_train,tr_idx,va_idx,base_cols)
        rf=make_rf(25000+split)
        rf.fit(trf[cols],y[tr_idx]); rprob=rf.predict_proba(vaf[cols])[:,1]; rfp=(rprob>.5).astype(int)
        maj=majority(v5,de,rfp)
        yy=y[va_idx]
        for name,p in [("v5",v5),("deotte",de),("rf6",rfp),("majority",maj)]:
            rows.append({"split":split,"candidate":name,"accuracy":accuracy_score(yy,p),"correct":int(np.sum(p==yy))})
        details.extend({"split":split,"PassengerId":int(expected[i]),"Survived":int(yy[i]),"v5":int(v5[i]),"deotte":int(de[i]),"rf6":int(rfp[i]),"majority":int(maj[i])} for i in range(len(yy)))
        print(f"pseudo {split}: v5={accuracy_score(yy,v5):.5f} deotte={accuracy_score(yy,de):.5f} rf6={accuracy_score(yy,rfp):.5f} majority={accuracy_score(yy,maj):.5f}",flush=True)
    return pd.DataFrame(rows),pd.DataFrame(details)


def fixed_and_group_summary():
    # Fixed OOF.
    z=pd.read_csv(V5_DIR/"model_zoo_oof.csv")
    m=pd.read_csv(V5_DIR/"seed_probes"/"MLP_PLR_seed_mean__oof.csv")
    d=pd.read_csv(V21_DIR/"deotte_fixed_oof.csv")
    rf=pd.read_csv(BASE_DIR/"exports"/"v22"/"hpo_screen_oof.csv")
    y=d["Survived"].astype(int).to_numpy()
    v5=(((z["v4b__Champion"].to_numpy()>.5).astype(int)+(z["RuleFit"].to_numpy()>.5).astype(int)+(m["probability"].to_numpy()>.5).astype(int))>=2).astype(int)
    de=d["wcg_both"].astype(int).to_numpy(); rfp=(rf["RandomForest_6"].to_numpy()>.5).astype(int)
    maj=majority(v5,de,rfp)
    rows=[{"surface":"fixed","v5_accuracy":accuracy_score(y,v5),"deotte_accuracy":accuracy_score(y,de),"rf6_accuracy":accuracy_score(y,rfp),"majority_accuracy":accuracy_score(y,maj),"delta_vs_v5":accuracy_score(y,maj)-accuracy_score(y,v5)}]
    for seed in [42,777]:
        f=pd.read_csv(V23_DIR/f"groupaware_oof_seed{seed}.csv")
        yy=f["Survived"].astype(int).to_numpy(); vv=f["v5_robust"].astype(int).to_numpy(); dd=f["deotte"].astype(int).to_numpy(); rr=f["typed_rf6"].astype(int).to_numpy(); mm=majority(vv,dd,rr)
        rows.append({"surface":f"group_seed{seed}","v5_accuracy":accuracy_score(yy,vv),"deotte_accuracy":accuracy_score(yy,dd),"rf6_accuracy":accuracy_score(yy,rr),"majority_accuracy":accuracy_score(yy,mm),"delta_vs_v5":accuracy_score(yy,mm)-accuracy_score(yy,vv)})
    return pd.DataFrame(rows)


def build_rf6_test(train_raw,test_raw,train,test,folds,base_cols,rel_train,rel_test):
    y=train["Survived"].astype(int).to_numpy(); cols=list(dict.fromkeys(base_cols+FAMILYFARE_COLS+TYPED_COLS)); probs=[]
    for fold in sorted(np.unique(folds)):
        tr_idx=np.flatnonzero(folds!=fold)
        tr=train.iloc[tr_idx].copy(); te=test.copy(); rr=rel_train.iloc[tr_idx].copy()
        gs_tr=add_group_survival(tr,tr,exclude_self=True).to_numpy(); gs_te=add_group_survival(tr,te,exclude_self=False).to_numpy(); tr["GroupSurvival"]=gs_tr; te["GroupSurvival"]=gs_te
        tr=add_familyfare(tr,tr,True); te=add_familyfare(train.iloc[tr_idx].assign(GroupSurvival=gs_tr),te,False)
        af,at=eligible_groups(rr,rel_test)
        tr_rel=relation_features(rr,rr,exclude_self=True,alpha=2.0,allowed_fam=af,allowed_tic=at); te_rel=relation_features(rr,rel_test,exclude_self=False,alpha=2.0,allowed_fam=af,allowed_tic=at)
        tr=add_rel(tr,tr_rel); te=add_rel(te,te_rel)
        model=make_rf(26000+int(fold)); model.fit(tr[cols],y[tr_idx]); probs.append(model.predict_proba(te[cols])[:,1])
    return np.mean(probs,axis=0)


def build_test_candidate(train_raw,test_raw,train,test,folds,base_cols,rel_train,rel_test):
    z=pd.read_csv(V5_DIR/"model_zoo_test.csv")
    m=pd.read_csv(V5_DIR/"seed_probes"/"MLP_PLR_seed_mean__test.csv")
    v5=(((z["v4b__Champion"].to_numpy()>.5).astype(int)+(z["RuleFit"].to_numpy()>.5).astype(int)+(m["probability"].to_numpy()>.5).astype(int))>=2).astype(int)
    de=pd.read_csv(SUB_DIR/"submission_v21_deotte_wcg_xgb_exact_public.csv").sort_values("PassengerId")["Survived"].astype(int).to_numpy()
    rprob=build_rf6_test(train_raw,test_raw,train,test,folds,base_cols,rel_train,rel_test); rfp=(rprob>.5).astype(int)
    maj=majority(v5,de,rfp)
    out=pd.DataFrame({"PassengerId":test_raw["PassengerId"].astype(int),"Survived":maj})
    out.to_csv(SUB_DIR/"submission_v25_final_majority.csv",index=False)
    detail=pd.DataFrame({"PassengerId":test_raw["PassengerId"].astype(int),"v5":v5,"deotte_exact":de,"rf6":rfp,"rf6_prob":rprob,"majority":maj})
    detail.to_csv(EXPORT_DIR/"final_test_votes.csv",index=False)
    return detail


def main():
    EXPORT_DIR.mkdir(parents=True,exist_ok=True); SUB_DIR.mkdir(parents=True,exist_ok=True)
    train_raw=pd.read_csv(DATA_DIR/"train.csv"); test_raw=pd.read_csv(DATA_DIR/"test.csv")
    train,test,folds,base_cols=prepare_data(); rel_train,rel_test,_,_,_=structural_frames(train_raw,test_raw); rel_train["IsWomanChild"]=role_flags(train_raw); rel_test["IsWomanChild"]=role_flags(test_raw)
    ps,pdetail=pseudo_audit(train_raw,test_raw,train,rel_train,base_cols); ps.to_csv(EXPORT_DIR/"pseudo_metrics.csv",index=False); pdetail.to_csv(EXPORT_DIR/"pseudo_predictions.csv",index=False)
    base=ps[ps.candidate=="v5"].set_index("split"); maj=ps[ps.candidate=="majority"].set_index("split"); deltas=maj.accuracy-base.accuracy
    pseudo_summary={"mean_v5":float(base.accuracy.mean()),"mean_majority":float(maj.accuracy.mean()),"mean_delta":float(deltas.mean()),"positive_splits":int((deltas>0).sum()),"nonnegative_splits":int((deltas>=0).sum()),"negative_splits":int((deltas<0).sum()),"deltas":[float(x) for x in deltas]}
    with (EXPORT_DIR/"pseudo_summary.json").open("w",encoding="utf-8") as f: json.dump(pseudo_summary,f,indent=2)
    surf=fixed_and_group_summary(); surf.to_csv(EXPORT_DIR/"validation_surfaces.csv",index=False)
    detail=build_test_candidate(train_raw,test_raw,train,test,folds,base_cols,rel_train,rel_test)
    v10=pd.read_csv(SUB_DIR/"submission_v10_score_0.81578.csv").sort_values("PassengerId")["Survived"].astype(int).to_numpy(); comparison={"changed_vs_v10":int(np.sum(detail.majority.to_numpy()!=v10)),"changed_vs_v5":int(np.sum(detail.majority.to_numpy()!=detail.v5.to_numpy())),"changed_vs_deotte":int(np.sum(detail.majority.to_numpy()!=detail.deotte_exact.to_numpy())),"test_positives":int(detail.majority.sum())}
    with (EXPORT_DIR/"test_comparison.json").open("w",encoding="utf-8") as f: json.dump(comparison,f,indent=2)
    print("\n=== validation surfaces ==="); print(surf.to_string(index=False,float_format=lambda x:f"{x:.5f}")); print("\n=== pseudo summary ==="); print(json.dumps(pseudo_summary,indent=2)); print("\n=== test comparison ==="); print(json.dumps(comparison,indent=2)); print("\nNo Kaggle submission was performed.")


if __name__=="__main__": main()
