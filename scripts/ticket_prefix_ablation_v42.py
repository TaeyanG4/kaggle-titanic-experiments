"""v42: mechanism ablation for numeric ticket-prefix EB features.

v41 showed strong pseudo/repeated gains but one repeated-seed collapse and a
mixed group-aware result. This script isolates whether instability comes from
the coarse P2 block, the finer P3 block, or the role-specific P3 statistics.

Predeclared variants:
  * p2_full      : all P2 EB/same-role features + digit length
  * p3_simple    : P3 EB/count/delta + digit length
  * p3_full      : all P3 EB/same-role features + digit length
  * p2_p3_full   : the exact v41 block

The parent representation is unchanged FamilyFare+typed22+EB25 CatBoost.
Evaluation uses the same six repeated, two group-aware, and five matched pseudo
surfaces. No test prediction and no Kaggle submission.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from feature_ablation_v6 import build_models as build_tree_models
from groupaware_validation_v23 import connected_groups
from partial_pooling_transfer_v28 import fold_features, prepare_rel
from pseudo_test_relational_v14 import build_pseudo_splits
from tabpfn_finalist_v8 import prepare_data
from ticket_numeric_prefix_v41 import augment_rel, prefix_block


BASE_DIR=Path(__file__).resolve().parents[1]
DATA_DIR=BASE_DIR/"data"
EXPORT_DIR=BASE_DIR/"exports"/"v42"
PARENT="familyfare_typed22_eb25"
SEEDS=[42,123,777,2026,31415,27182]

P2=["TicketP2EB","TicketP2Count","TicketP2Delta","TicketP2RoleEB","TicketP2RoleCount","TicketP2RoleDelta"]
P3=["TicketP3EB","TicketP3Count","TicketP3Delta","TicketP3RoleEB","TicketP3RoleCount","TicketP3RoleDelta"]
VARIANTS={
    "p2_full":P2+["TicketDigitLength"],
    "p3_simple":["TicketP3EB","TicketP3Count","TicketP3Delta","TicketDigitLength"],
    "p3_full":P3+["TicketDigitLength"],
    "p2_p3_full":P2+P3+["TicketDigitLength"],
}


def split_features(train,rel,base_cols,tr_idx,va_idx):
    tr,va,base=fold_features(train,rel,tr_idx,va_idx,base_cols,PARENT)
    rr=rel.iloc[tr_idx].copy(); rv=rel.iloc[va_idx].copy(); pt=prefix_block(rr,rr,True); pv=prefix_block(rr,rv,False)
    for c in sorted(set(sum(VARIANTS.values(),[]))): tr[c]=pt[c].to_numpy(); va[c]=pv[c].to_numpy()
    return tr,va,base


def evaluate_surface(train,rel,base_cols,y,tr_idx,va_idx,seed):
    tr,va,base_cols2=split_features(train,rel,base_cols,tr_idx,va_idx)
    probs={}
    parent=build_tree_models(seed)["CatBoost"]; parent.fit(tr[base_cols2],y[tr_idx]); probs["parent"]=np.asarray(parent.predict_proba(va[base_cols2]))[:,1]
    for j,(name,extra) in enumerate(VARIANTS.items()):
        cols=base_cols2+extra; m=build_tree_models(seed+1000*(j+1))["CatBoost"]; m.fit(tr[cols],y[tr_idx]); probs[name]=np.asarray(m.predict_proba(va[cols]))[:,1]
    return probs


def add_rows(rows,family,split,yv,probs):
    pb=probs["parent"]; bacc=accuracy_score(yv,pb>.5); bauc=roc_auc_score(yv,pb)
    for name in VARIANTS:
        p=probs[name]; rows.append({"family":family,"split":split,"variant":name,"parent_acc":bacc,"accuracy":accuracy_score(yv,p>.5),"delta":accuracy_score(yv,p>.5)-bacc,"parent_auc":bauc,"auc":roc_auc_score(yv,p),"auc_delta":roc_auc_score(yv,p)-bauc})


def main():
    EXPORT_DIR.mkdir(parents=True,exist_ok=True)
    traw=pd.read_csv(DATA_DIR/"train.csv"); teraw=pd.read_csv(DATA_DIR/"test.csv"); train,_,fixed,base_cols=prepare_data(); rel,_=prepare_rel(traw,teraw); rel=augment_rel(rel); y=train.Survived.astype(int).to_numpy(); rows=[]

    for seed in SEEDS:
        if seed==42: folds=fixed.copy()
        else:
            folds=np.full(len(train),-1,int); sk=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
            for f,(_,va) in enumerate(sk.split(train,y)): folds[va]=f
        full={k:np.zeros(len(train)) for k in ["parent",*VARIANTS]}
        for f in sorted(np.unique(folds)):
            tr=np.flatnonzero(folds!=f); va=np.flatnonzero(folds==f); p=evaluate_surface(train,rel,base_cols,y,tr,va,46000+seed+int(f))
            for k,v in p.items(): full[k][va]=v
        add_rows(rows,"repeated",seed,y,full); print("repeated",seed," ".join(f"{k}={accuracy_score(y,full[k]>.5):.5f}" for k in full),flush=True)

    groups=connected_groups(traw)
    for seed in [42,777]:
        sg=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed); full={k:np.zeros(len(train)) for k in ["parent",*VARIANTS]}
        for f,(tr,va) in enumerate(sg.split(train,y,groups)):
            p=evaluate_surface(train,rel,base_cols,y,tr,va,47000+seed+f)
            for k,v in p.items(): full[k][va]=v
        add_rows(rows,"group",seed,y,full); print("group",seed," ".join(f"{k}={accuracy_score(y,full[k]>.5):.5f}" for k in full),flush=True)

    _,pseudo=build_pseudo_splits(traw,teraw)
    for s,(_,va,_) in enumerate(pseudo):
        tr=np.setdiff1d(np.arange(len(train)),va); p=evaluate_surface(train,rel,base_cols,y,tr,va,48000+s); add_rows(rows,"pseudo",s,y[va],p); print("pseudo",s," ".join(f"{k}={accuracy_score(y[va],p[k]>.5):.5f}" for k in p),flush=True)

    m=pd.DataFrame(rows); m.to_csv(EXPORT_DIR/"metrics.csv",index=False)
    sm=m.groupby(["variant","family"],as_index=False).agg(mean_accuracy=("accuracy","mean"),mean_delta=("delta","mean"),min_delta=("delta","min"),max_delta=("delta","max"),positive=("delta",lambda s:int((s>0).sum())),nonnegative=("delta",lambda s:int((s>=0).sum())),negative=("delta",lambda s:int((s<0).sum())),mean_auc_delta=("auc_delta","mean"))
    sm.to_csv(EXPORT_DIR/"summary.csv",index=False)
    rank=sm.groupby("variant",as_index=False).agg(mean_delta_all=("mean_delta","mean"),worst_family_mean_delta=("mean_delta","min"),worst_single_delta=("min_delta","min"),negative_surfaces=("negative","sum"),positive_surfaces=("positive","sum"),mean_auc_delta=("mean_auc_delta","mean")).sort_values(["worst_family_mean_delta","mean_delta_all","negative_surfaces"],ascending=[False,False,True]); rank.to_csv(EXPORT_DIR/"ranking.csv",index=False)
    print("\n=== v42 by family ==="); print(sm.to_string(index=False,float_format=lambda x:f"{x:.5f}")); print("\n=== v42 ranking ==="); print(rank.to_string(index=False,float_format=lambda x:f"{x:.5f}")); print("\nNo Kaggle submission was performed.")


if __name__=="__main__": main()
