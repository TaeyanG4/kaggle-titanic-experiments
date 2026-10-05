"""v43: real-test finalist for the v42 P3-full ticket-prefix hypothesis.

Build a leakage-safe bagged CatBoost test prediction using:
  FamilyFare + typed22 + EB25 + numeric ticket P3 EB/same-role features.

Each of six split seeds trains five fold models. All supervised relational and
numeric-prefix encodings for the real test use fold-train labels only. Test
covariates are visible only for target-independent feature construction.

This script creates candidate artifacts and headroom/stability diagnostics only.
It does not submit to Kaggle.
"""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from audit_group_survival import add_group_survival
from feature_ablation_v6 import build_models as build_tree_models
from partial_pooling_transfer_v28 import FAMILYFARE_COLS, TYPED_COLS, add_block, add_familyfare, prepare_rel
from partial_pooling_v27 import EB_COLS, eb_features
from pseudo_test_relational_v14 import relation_features
from tabpfn_finalist_v8 import prepare_data
from ticket_numeric_prefix_v41 import augment_rel, prefix_block


BASE_DIR=Path(__file__).resolve().parents[1]
DATA_DIR=BASE_DIR/"data"
EXPORT_DIR=BASE_DIR/"exports"/"v43"
SUB_DIR=BASE_DIR/"submissions"
SEEDS=[42,123,777,2026,31415,27182]
P3_FULL=["TicketP3EB","TicketP3Count","TicketP3Delta","TicketP3RoleEB","TicketP3RoleCount","TicketP3RoleDelta","TicketDigitLength"]


def fold_train_test(train,test,rel_train,rel_test,y,tr_idx,base_cols,seed):
    tr=train.iloc[tr_idx].copy(); te=test.copy(); rr=rel_train.iloc[tr_idx].copy(); rt=rel_test.copy()
    gs_tr=add_group_survival(tr,tr,exclude_self=True).to_numpy(); gs_te=add_group_survival(tr,te,exclude_self=False).to_numpy()
    tr["GroupSurvival"]=gs_tr; te["GroupSurvival"]=gs_te
    tr=add_familyfare(tr,tr,True); te=add_familyfare(train.iloc[tr_idx].assign(GroupSurvival=gs_tr),te,False)

    tr_t=relation_features(rr,rr,exclude_self=True,alpha=2.0); te_t=relation_features(rr,rt,exclude_self=False,alpha=2.0)
    tr=add_block(tr,tr_t,TYPED_COLS); te=add_block(te,te_t,TYPED_COLS)
    tr_e,_=eb_features(rr,rr,exclude_self=True); te_e,_=eb_features(rr,rt,exclude_self=False)
    tr=add_block(tr,tr_e,EB_COLS); te=add_block(te,te_e,EB_COLS)
    tr_p=prefix_block(rr,rr,True); te_p=prefix_block(rr,rt,False)
    for c in P3_FULL:
        tr[c]=tr_p[c].to_numpy(); te[c]=te_p[c].to_numpy()

    cols=list(dict.fromkeys(list(base_cols)+FAMILYFARE_COLS+TYPED_COLS+EB_COLS+P3_FULL))
    m=build_tree_models(seed)["CatBoost"]; m.fit(tr[cols],y[tr_idx]); return np.asarray(m.predict_proba(te[cols]))[:,1]


def required_switch_precision(d,gap):
    if d==0: return math.inf,math.inf
    w=math.floor((d+gap)/2)+1
    return w,w/d


def main():
    EXPORT_DIR.mkdir(parents=True,exist_ok=True); SUB_DIR.mkdir(parents=True,exist_ok=True)
    traw=pd.read_csv(DATA_DIR/"train.csv"); teraw=pd.read_csv(DATA_DIR/"test.csv"); train,test,_,base_cols=prepare_data(); rel_train,rel_test=prepare_rel(traw,teraw); rel_train=augment_rel(rel_train); rel_test=augment_rel(rel_test); y=train.Survived.astype(int).to_numpy()

    seed_probs=[]; seed_labels=[]; seed_rows=[]
    for seed in SEEDS:
        sk=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed); fps=[]
        for f,(tr_idx,_) in enumerate(sk.split(train,y)):
            print(f"seed={seed} fold={f}",flush=True); fps.append(fold_train_test(train,test,rel_train,rel_test,y,tr_idx,base_cols,49000+seed+f))
        prob=np.mean(fps,axis=0); lab=(prob>=.5).astype(int); seed_probs.append(prob); seed_labels.append(lab); seed_rows.append({"seed":seed,"positive_labels":int(lab.sum()),"mean_probability":float(prob.mean())})
    seed_probs=np.vstack(seed_probs); seed_labels=np.vstack(seed_labels); mean_prob=seed_probs.mean(axis=0); candidate=(mean_prob>=.5).astype(int); vote=(seed_labels.sum(axis=0)>=3).astype(int)

    ids=teraw.PassengerId.astype(int).to_numpy(); v5=pd.read_csv(SUB_DIR/"submission_v5_score_0.79665.csv").sort_values("PassengerId").Survived.astype(int).to_numpy(); v10=pd.read_csv(SUB_DIR/"submission_v10_score_0.81578.csv").sort_values("PassengerId").Survived.astype(int).to_numpy()
    details=pd.DataFrame({"PassengerId":ids,"v5":v5,"v10":v10,"p3_mean_prob":mean_prob,"p3_mean_label":candidate,"p3_seed_vote":vote,"p3_positive_seed_votes":seed_labels.sum(axis=0)})
    for i,s in enumerate(SEEDS): details[f"prob_seed_{s}"]=seed_probs[i]; details[f"label_seed_{s}"]=seed_labels[i]
    details.to_csv(EXPORT_DIR/"test_predictions.csv",index=False); pd.DataFrame(seed_rows).to_csv(EXPORT_DIR/"seed_stability.csv",index=False)
    pd.DataFrame({"PassengerId":ids,"Survived":candidate}).to_csv(SUB_DIR/"submission_v43_ticket_p3_full.csv",index=False)
    pd.DataFrame({"PassengerId":ids,"Survived":vote}).to_csv(SUB_DIR/"submission_v43_ticket_p3_seedvote.csv",index=False)

    rows=[]
    for name,c in [("mean_probability",candidate),("seed_vote",vote)]:
        d5=int(np.sum(c!=v5)); d10=int(np.sum(c!=v10)); w5,p5=required_switch_precision(d5,8)
        rows.append({"candidate":name,"changed_vs_v5":d5,"changed_vs_v10":d10,"test_positives":int(c.sum()),"v5_to_v10_gap":8,"wins_needed_among_v5_switches_to_beat_v10":w5,"required_precision_vs_v5":p5,"mathematically_can_beat_v10_from_v5":int(w5<=d5),"seed_unanimous_rows":int(np.sum((seed_labels.sum(axis=0)==0)|(seed_labels.sum(axis=0)==len(SEEDS)))),"seed_split_rows":int(np.sum((seed_labels.sum(axis=0)>0)&(seed_labels.sum(axis=0)<len(SEEDS))))})
    head=pd.DataFrame(rows); head.to_csv(EXPORT_DIR/"headroom.csv",index=False)
    print("\n=== seed stability ==="); print(pd.DataFrame(seed_rows).to_string(index=False,float_format=lambda x:f"{x:.5f}")); print("\n=== headroom ==="); print(head.to_string(index=False,float_format=lambda x:f"{x:.5f}")); print("\n=== disagreements with v10 ==="); print(details[details.p3_mean_label!=details.v10][["PassengerId","v5","v10","p3_mean_label","p3_mean_prob","p3_positive_seed_votes"]].to_string(index=False)); print("\nNo Kaggle submission was performed.")


if __name__=="__main__": main()
