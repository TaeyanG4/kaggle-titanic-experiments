"""v34: build robust EB test predictions and apply the mandatory headroom gate.

No Kaggle submission is performed.

EB test probability is bagged over five split seeds x five folds. Each fold
constructs all supervised relational features from fold-train labels only and
predicts the real test. We report both mean-probability and seed-majority EB
labels, then form majority(v5, Deotte exact, EB).
"""

from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from audit_group_survival import add_group_survival
from feature_ablation_v6 import build_models as build_typed_models
from group_key_audit_v6 import helper_keys, peer_feature
from gunes_exact_foldsafe_v11 import structural_frames
from partial_pooling_v27 import EB_COLS, eb_features
from pseudo_test_relational_v14 import role_flags
from tabpfn_finalist_v8 import prepare_data


BASE_DIR=Path(__file__).resolve().parents[1]
DATA_DIR=BASE_DIR/"data"
EXPORT_DIR=BASE_DIR/"exports"/"v34"
SUB_DIR=BASE_DIR/"submissions"
V33_DIR=BASE_DIR/"exports"/"v33"
V32_DIR=BASE_DIR/"exports"/"v32"
SEEDS=[42,123,777,2026,2901]
FAMILYFARE_COLS=["FamilyFareAny","FamilyFareMean","FamilyFareSmooth","FamilyFareCount"]

def majority(a,b,c): return ((a.astype(int)+b.astype(int)+c.astype(int))>=2).astype(int)
def add_familyfare(ref,app,exclude_self):
    s=peer_feature(ref,app,group_col="FamilyFareGroup_v6",wcg_only=False,exclude_self=exclude_self); out=app.copy()
    for a,b in zip(["Any","Mean","Smooth","Count"],FAMILYFARE_COLS): out[b]=s[a].to_numpy()
    return out
def add_block(df,b):
    out=df.copy(); b=b.reset_index(drop=True)
    for c in EB_COLS: out[c]=b[c].to_numpy()
    return out
def prepare_rel(train_raw,test_raw):
    tr,te,_,_,_=structural_frames(train_raw,test_raw); tr["IsWomanChild"]=role_flags(train_raw); te["IsWomanChild"]=role_flags(test_raw); h,ht=helper_keys(train_raw,test_raw)
    tr=tr.merge(h[["PassengerId","FamilyFareGroup_v6"]],on="PassengerId",how="left"); te=te.merge(ht[["PassengerId","FamilyFareGroup_v6"]],on="PassengerId",how="left")
    return tr,te

def fold_test_prob(train,test,rel_train,rel_test,y,tr_idx,base_cols,seed):
    tr=train.iloc[tr_idx].copy(); te=test.copy(); rr=rel_train.iloc[tr_idx].copy()
    gs_tr=add_group_survival(tr,tr,exclude_self=True).to_numpy(); gs_te=add_group_survival(tr,te,exclude_self=False).to_numpy(); tr["GroupSurvival"]=gs_tr; te["GroupSurvival"]=gs_te
    tr=add_familyfare(tr,tr,True); te=add_familyfare(train.iloc[tr_idx].assign(GroupSurvival=gs_tr),te,False)
    tre,_=eb_features(rr,rr,exclude_self=True); tee,_=eb_features(rr,rel_test,exclude_self=False); tr=add_block(tr,tre); te=add_block(te,tee)
    cols=list(dict.fromkeys(base_cols+FAMILYFARE_COLS+EB_COLS)); m=build_typed_models(seed)["CatBoost"]; m.fit(tr[cols],y[tr_idx]); return np.asarray(m.predict_proba(te[cols]))[:,1]

def required_switch_precision(d,g):
    # Need integer w satisfying 2w-d > g.
    w_min=math.floor((d+g)/2)+1
    return w_min, (w_min/d if d>0 else float('inf'))

def empirical_switch_stats():
    r=pd.read_csv(V33_DIR/"surface_metrics.csv"); q=r[r.surface.str.startswith("repeated_")]; p=r[r.surface.str.startswith("pseudo_")]; g=pd.read_csv(V32_DIR/"surface_metrics.csv"); g=g[g.surface.str.startswith("group_")]
    def one(name,z):
        rescue=float(z.rescue.sum()); harm=float(z.harm.sum()); total=rescue+harm
        return {"family":name,"rescue":int(rescue),"harm":int(harm),"switch_precision":rescue/total if total else np.nan,"net":int(rescue-harm)}
    return pd.DataFrame([one("repeated",q),one("pseudo",p),one("group",g)])

def main():
    EXPORT_DIR.mkdir(parents=True,exist_ok=True)
    traw=pd.read_csv(DATA_DIR/"train.csv"); teraw=pd.read_csv(DATA_DIR/"test.csv"); train,test,_,base_cols=prepare_data(); rel_train,rel_test=prepare_rel(traw,teraw); y=train.Survived.astype(int).to_numpy()
    seed_probs=[]; seed_labels=[]; per_seed=[]
    for split_seed in SEEDS:
        skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=split_seed); fps=[]
        for fold,(tr_idx,_) in enumerate(skf.split(train,y)):
            print(f"seed={split_seed} fold={fold}",flush=True); fps.append(fold_test_prob(train,test,rel_train,rel_test,y,tr_idx,base_cols,34000+split_seed+fold))
        prob=np.mean(fps,axis=0); lab=(prob>.5).astype(int); seed_probs.append(prob); seed_labels.append(lab); per_seed.append({"seed":split_seed,"test_positives":int(lab.sum()),"mean_probability":float(prob.mean())})
    seed_probs=np.vstack(seed_probs); seed_labels=np.vstack(seed_labels); mean_prob=seed_probs.mean(axis=0); eb_mean=(mean_prob>.5).astype(int); eb_vote=(seed_labels.sum(axis=0)>=3).astype(int)

    v5=pd.read_csv(SUB_DIR/"submission_v5_score_0.79665.csv").sort_values("PassengerId").Survived.astype(int).to_numpy(); de=pd.read_csv(SUB_DIR/"submission_v21_deotte_wcg_xgb_exact_public.csv").sort_values("PassengerId").Survived.astype(int).to_numpy(); v10=pd.read_csv(SUB_DIR/"submission_v10_score_0.81578.csv").sort_values("PassengerId").Survived.astype(int).to_numpy(); ids=teraw.sort_values("PassengerId").PassengerId.astype(int).to_numpy()
    cmean=majority(v5,de,eb_mean); cvote=majority(v5,de,eb_vote)
    details=pd.DataFrame({"PassengerId":ids,"v5":v5,"deotte":de,"v10":v10,"eb_mean_prob":mean_prob,"eb_mean_label":eb_mean,"eb_seed_vote":eb_vote,"eb_vote_count":seed_labels.sum(axis=0),"consensus_mean":cmean,"consensus_vote":cvote})
    for i,s in enumerate(SEEDS): details[f"eb_seed_{s}"]=seed_labels[i]; details[f"eb_prob_{s}"]=seed_probs[i]
    details.to_csv(EXPORT_DIR/"test_votes.csv",index=False); pd.DataFrame(per_seed).to_csv(EXPORT_DIR/"eb_seed_test_summary.csv",index=False)

    # Historical displayed scores correspond to 333/418 and 341/418 correct.
    v5_correct=333; v10_correct=341; gap=v10_correct-v5_correct
    rows=[]
    for name,c in [("consensus_mean",cmean),("consensus_vote",cvote)]:
        d=int(np.sum(c!=v5)); wmin,prec=required_switch_precision(d,gap)
        rows.append({"candidate":name,"changed_vs_v5":d,"changed_vs_v10":int(np.sum(c!=v10)),"test_positives":int(c.sum()),"champion_gap_correct":gap,"wins_needed_among_changed_to_beat_v10":wmin,"required_switch_precision":prec,"mathematically_can_beat_v10":int(wmin<=d)})
        pd.DataFrame({"PassengerId":ids,"Survived":c}).to_csv(SUB_DIR/f"submission_v34_{name}.csv",index=False)
    head=pd.DataFrame(rows); head.to_csv(EXPORT_DIR/"headroom.csv",index=False)
    emp=empirical_switch_stats(); emp.to_csv(EXPORT_DIR/"empirical_switch_precision.csv",index=False)
    summary={"eb_mean_vs_seed_vote_differences":int(np.sum(eb_mean!=eb_vote)),"consensus_variant_differences":int(np.sum(cmean!=cvote)),"eb_rows_unanimous_5of5":int(np.sum((seed_labels.sum(axis=0)==0)|(seed_labels.sum(axis=0)==5))),"eb_rows_split_vote":int(np.sum((seed_labels.sum(axis=0)>0)&(seed_labels.sum(axis=0)<5)))}
    with (EXPORT_DIR/"summary.json").open("w",encoding="utf-8") as f: json.dump(summary,f,indent=2)
    print("\n=== headroom ==="); print(head.to_string(index=False,float_format=lambda x:f"{x:.5f}")); print("\n=== empirical switch precision ==="); print(emp.to_string(index=False,float_format=lambda x:f"{x:.5f}")); print("\n=== stability ==="); print(json.dumps(summary,indent=2)); print("\nNo Kaggle submission was performed.")

if __name__=="__main__": main()
