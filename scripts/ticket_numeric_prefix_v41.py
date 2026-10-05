"""v41: fold-safe ticket numeric-prefix empirical-Bayes features.

Residual/text analysis found signal in numeric ticket blocks (e.g. the generic
three-digit prefix representation), beyond existing non-numeric TicketPrefix,
TicketFreq, and exact Ticket relations. To avoid a row-specific rule, this test
adds the same generic prefix2/prefix3 representation for every passenger.

For each fold, prefix survival posteriors are computed from fold-train labels
only and shrunk toward Role x Pclass priors using empirical-Bayes strength.
The block is added to the strong FamilyFare+typed22+EB CatBoost pipeline and
compared on six repeated seeds, two group-aware seeds, and five matched pseudo
holdouts. No submission.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from feature_ablation_v6 import build_models as build_tree_models
from groupaware_validation_v23 import connected_groups
from partial_pooling_transfer_v28 import fold_features, prepare_rel
from partial_pooling_v27 import eb_group, estimate_alpha
from pseudo_test_relational_v14 import build_pseudo_splits
from tabpfn_finalist_v8 import prepare_data


BASE_DIR=Path(__file__).resolve().parents[1]
DATA_DIR=BASE_DIR/"data"
EXPORT_DIR=BASE_DIR/"exports"/"v41"
VARIANT="familyfare_typed22_eb25"
REPEATED_SEEDS=[42,123,777,2026,31415,27182]

PREFIX_COLS=[]
for p in ["P2","P3"]:
    PREFIX_COLS += [f"Ticket{p}EB",f"Ticket{p}Count",f"Ticket{p}Delta",f"Ticket{p}RoleEB",f"Ticket{p}RoleCount",f"Ticket{p}RoleDelta"]
PREFIX_COLS += ["TicketDigitLength"]


def digits(ticket): return "".join(re.findall(r"\d",str(ticket)))
def pfx(ticket,n):
    d=digits(ticket)
    return d[:n] if d else "NONE"


def augment_rel(rel: pd.DataFrame):
    out=rel.copy(); out["TicketNumP2"]=out.Ticket.map(lambda x:pfx(x,2)); out["TicketNumP3"]=out.Ticket.map(lambda x:pfx(x,3)); out["TicketDigitLength"]=out.Ticket.map(lambda x:len(digits(x))).astype(float); return out


def prefix_block(ref,app,exclude_self):
    out=pd.DataFrame(index=app.index)
    for name,col in [("P2","TicketNumP2"),("P3","TicketNumP3")]:
        a=estimate_alpha(ref,col,same_role=False); ar=estimate_alpha(ref,col,same_role=True)
        post,cnt,delta,_=eb_group(ref,app,group_col=col,exclude_self=exclude_self,same_role=False,alpha=a,allowed=None)
        rpost,rcnt,rdelta,_=eb_group(ref,app,group_col=col,exclude_self=exclude_self,same_role=True,alpha=ar,allowed=None)
        out[f"Ticket{name}EB"]=post; out[f"Ticket{name}Count"]=cnt; out[f"Ticket{name}Delta"]=delta; out[f"Ticket{name}RoleEB"]=rpost; out[f"Ticket{name}RoleCount"]=rcnt; out[f"Ticket{name}RoleDelta"]=rdelta
    out["TicketDigitLength"]=app["TicketDigitLength"].to_numpy(float)
    return out.reset_index(drop=True)


def eval_split(train,rel,base_cols,y,tr_idx,va_idx,seed):
    tr,va,cols=fold_features(train,rel,tr_idx,va_idx,base_cols,VARIANT); rr=rel.iloc[tr_idx].copy(); rv=rel.iloc[va_idx].copy(); pt=prefix_block(rr,rr,True); pv=prefix_block(rr,rv,False)
    for c in PREFIX_COLS: tr[c]=pt[c].to_numpy(); va[c]=pv[c].to_numpy()
    b=build_tree_models(seed)["CatBoost"]; n=build_tree_models(seed+1000)["CatBoost"]; b.fit(tr[cols],y[tr_idx]); pb=np.asarray(b.predict_proba(va[cols]))[:,1]; ncols=cols+PREFIX_COLS; n.fit(tr[ncols],y[tr_idx]); pn=np.asarray(n.predict_proba(va[ncols]))[:,1]; return pb,pn


def add_row(rows,family,split,yv,pb,pn):
    rows.append({"family":family,"split":split,"base_acc":accuracy_score(yv,pb>.5),"prefix_acc":accuracy_score(yv,pn>.5),"delta":accuracy_score(yv,pn>.5)-accuracy_score(yv,pb>.5),"base_auc":roc_auc_score(yv,pb),"prefix_auc":roc_auc_score(yv,pn)})


def main():
    EXPORT_DIR.mkdir(parents=True,exist_ok=True); traw=pd.read_csv(DATA_DIR/"train.csv"); teraw=pd.read_csv(DATA_DIR/"test.csv"); train,_,fixed_folds,base_cols=prepare_data(); rel,_=prepare_rel(traw,teraw); rel=augment_rel(rel); y=train.Survived.astype(int).to_numpy(); rows=[]
    # Six repeated seeds.
    for seed in REPEATED_SEEDS:
        if seed==42: folds=fixed_folds.copy()
        else:
            folds=np.full(len(train),-1,int); sk=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
            for f,(_,va) in enumerate(sk.split(train,y)): folds[va]=f
        base=np.zeros(len(train)); pref=np.zeros(len(train))
        for f in sorted(np.unique(folds)):
            tr=np.flatnonzero(folds!=f); va=np.flatnonzero(folds==f); pb,pn=eval_split(train,rel,base_cols,y,tr,va,43000+seed+int(f)); base[va]=pb; pref[va]=pn
        add_row(rows,"repeated",seed,y,base,pref); print(f"repeated {seed}: base={accuracy_score(y,base>.5):.5f} prefix={accuracy_score(y,pref>.5):.5f}",flush=True)
    # Group-aware.
    groups=connected_groups(traw)
    for seed in [42,777]:
        sg=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed); base=np.zeros(len(train)); pref=np.zeros(len(train))
        for f,(tr,va) in enumerate(sg.split(train,y,groups)):
            pb,pn=eval_split(train,rel,base_cols,y,tr,va,44000+seed+f); base[va]=pb; pref[va]=pn
        add_row(rows,"group",seed,y,base,pref); print(f"group {seed}: base={accuracy_score(y,base>.5):.5f} prefix={accuracy_score(y,pref>.5):.5f}",flush=True)
    # Matched pseudo.
    _,pseudo=build_pseudo_splits(traw,teraw)
    for s,(_,va,_) in enumerate(pseudo):
        tr=np.setdiff1d(np.arange(len(train)),va); pb,pn=eval_split(train,rel,base_cols,y,tr,va,45000+s); add_row(rows,"pseudo",s,y[va],pb,pn); print(f"pseudo {s}: base={accuracy_score(y[va],pb>.5):.5f} prefix={accuracy_score(y[va],pn>.5):.5f}",flush=True)
    m=pd.DataFrame(rows); m.to_csv(EXPORT_DIR/"metrics.csv",index=False); sm=m.groupby("family",as_index=False).agg(mean_base_acc=("base_acc","mean"),mean_prefix_acc=("prefix_acc","mean"),mean_delta=("delta","mean"),min_delta=("delta","min"),positive=("delta",lambda s:int((s>0).sum())),nonnegative=("delta",lambda s:int((s>=0).sum())),mean_base_auc=("base_auc","mean"),mean_prefix_auc=("prefix_auc","mean")); sm.to_csv(EXPORT_DIR/"summary.csv",index=False)
    print("\n=== ticket numeric prefix summary ==="); print(sm.to_string(index=False,float_format=lambda x:f"{x:.5f}")); print("\n=== all surfaces ==="); print(m.to_string(index=False,float_format=lambda x:f"{x:.5f}")); print("\nNo Kaggle submission was performed.")


if __name__=="__main__": main()
