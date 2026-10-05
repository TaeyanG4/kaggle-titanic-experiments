"""v33: repeated + matched-pseudo promotion audit for v5/Deotte/EB consensus.

Candidate: 2-of-3 majority(v5 robust, Deotte WCG/XGB, FamilyFare+EB CatBoost).

Repeated audit reuses v30 immutable v5/EB OOF predictions and trains only
Deotte on the exact same folds. Matched pseudo-test audit reuses v19 immutable
v5 predictions and trains Deotte + EB on the five predeclared pseudo splits.

No test prediction and no Kaggle submission.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from audit_group_survival import add_group_survival
from deotte_wcg_xgb_v21 import predict_split as deotte_predict_split
from feature_ablation_v6 import build_models as build_typed_models
from group_key_audit_v6 import helper_keys, peer_feature
from gunes_exact_foldsafe_v11 import structural_frames
from partial_pooling_v27 import EB_COLS, eb_features
from pseudo_test_relational_v14 import build_pseudo_splits, role_flags
from tabpfn_finalist_v8 import prepare_data


BASE_DIR=Path(__file__).resolve().parents[1]
DATA_DIR=BASE_DIR/"data"
EXPORT_DIR=BASE_DIR/"exports"/"v33"
V30_DIR=BASE_DIR/"exports"/"v30"
V19_DIR=BASE_DIR/"exports"/"v19"
FAMILYFARE_COLS=["FamilyFareAny","FamilyFareMean","FamilyFareSmooth","FamilyFareCount"]

def majority(a,b,c): return ((a.astype(int)+b.astype(int)+c.astype(int))>=2).astype(int)
def paired(y,new,base):
    rescue=int(np.sum((new==y)&(base!=y))); harm=int(np.sum((new!=y)&(base==y)))
    return rescue,harm,rescue-harm
def add_familyfare(ref,app,exclude_self):
    s=peer_feature(ref,app,group_col="FamilyFareGroup_v6",wcg_only=False,exclude_self=exclude_self); out=app.copy()
    for a,b in zip(["Any","Mean","Smooth","Count"],FAMILYFARE_COLS): out[b]=s[a].to_numpy()
    return out
def add_block(df,b):
    out=df.copy(); b=b.reset_index(drop=True)
    for c in EB_COLS: out[c]=b[c].to_numpy()
    return out
def prepare_rel(train_raw,test_raw):
    tr,_,_,_,_=structural_frames(train_raw,test_raw); tr["IsWomanChild"]=role_flags(train_raw); h,_=helper_keys(train_raw,test_raw)
    return tr.merge(h[["PassengerId","FamilyFareGroup_v6"]],on="PassengerId",how="left")

def train_eb(train_v2,rel,y,tr_idx,va_idx,base_cols,seed):
    tr=train_v2.iloc[tr_idx].copy(); va=train_v2.iloc[va_idx].copy(); rr=rel.iloc[tr_idx].copy(); rv=rel.iloc[va_idx].copy()
    gs_tr=add_group_survival(tr,tr,exclude_self=True).to_numpy(); gs_va=add_group_survival(tr,va,exclude_self=False).to_numpy(); tr["GroupSurvival"]=gs_tr; va["GroupSurvival"]=gs_va
    tr=add_familyfare(tr,tr,True); va=add_familyfare(train_v2.iloc[tr_idx].assign(GroupSurvival=gs_tr),va,False)
    tre,_=eb_features(rr,rr,exclude_self=True); vae,_=eb_features(rr,rv,exclude_self=False); tr=add_block(tr,tre); va=add_block(va,vae)
    cols=list(dict.fromkeys(base_cols+FAMILYFARE_COLS+EB_COLS)); m=build_typed_models(seed)["CatBoost"]; m.fit(tr[cols],y[tr_idx]); p=np.asarray(m.predict_proba(va[cols]))[:,1]
    return (p>.5).astype(int),p

def repeated_audit(train_raw):
    saved=pd.read_csv(V30_DIR/"repeated_oof.csv"); y=train_raw.Survived.astype(int).to_numpy(); rows=[]; details=[]
    for seed in sorted(saved.seed.unique()):
        s=saved[saved.seed==seed].sort_values("PassengerId").reset_index(drop=True); folds=s.fold.astype(int).to_numpy(); v5=s.v5.astype(int).to_numpy(); eb=s.eb_cat.astype(int).to_numpy(); de=np.zeros(len(train_raw),dtype=int)
        if int(seed)==2901:
            n=pd.read_csv(BASE_DIR/"exports"/"v29"/"nested_oof.csv").sort_values("PassengerId"); de=n.deotte.astype(int).to_numpy()
        else:
            for fold in sorted(np.unique(folds)):
                tr_idx=np.flatnonzero(folds!=fold); va_idx=np.flatnonzero(folds==fold)
                d=deotte_predict_split(train_raw,tr_idx,va_idx,seed=33000+int(seed)+int(fold),imputer_fit_indices=tr_idx,special_link=True); de[va_idx]=d.wcg_both.astype(int)
        con=majority(v5,de,eb); r,h,n=paired(y,con,v5)
        rows.append({"surface":f"repeated_{int(seed)}","v5_accuracy":accuracy_score(y,v5),"consensus_accuracy":accuracy_score(y,con),"delta":accuracy_score(y,con)-accuracy_score(y,v5),"rescue":r,"harm":h,"net":n})
        details.append(pd.DataFrame({"PassengerId":train_raw.PassengerId.astype(int),"Survived":y,"surface":f"repeated_{int(seed)}","v5":v5,"deotte":de,"eb":eb,"consensus":con}))
        print(f"repeated {int(seed)}: v5={accuracy_score(y,v5):.5f} consensus={accuracy_score(y,con):.5f} net={n:+d}",flush=True)
    return pd.DataFrame(rows),pd.concat(details,ignore_index=True)

def pseudo_audit(train_raw,test_raw,train_v2,rel,base_cols):
    y=train_raw.Survived.astype(int).to_numpy(); saved=pd.read_csv(V19_DIR/"pseudotest_consensus_predictions.csv"); _,splits=build_pseudo_splits(train_raw,test_raw); rows=[]; details=[]
    for split,(_,va_idx,_) in enumerate(splits):
        tr_idx=np.setdiff1d(np.arange(len(train_raw)),va_idx); s=saved[saved.split==split].reset_index(drop=True); ids=train_raw.iloc[va_idx].PassengerId.astype(int).to_numpy()
        if not np.array_equal(s.PassengerId.astype(int).to_numpy(),ids): raise ValueError("pseudo order mismatch")
        v5=s.v5.astype(int).to_numpy(); d=deotte_predict_split(train_raw,tr_idx,va_idx,seed=34000+split,imputer_fit_indices=tr_idx,special_link=True); de=d.wcg_both.astype(int); eb,_=train_eb(train_v2,rel,y,tr_idx,va_idx,base_cols,35000+split); con=majority(v5,de,eb); yy=y[va_idx]; r,h,n=paired(yy,con,v5)
        rows.append({"surface":f"pseudo_{split}","v5_accuracy":accuracy_score(yy,v5),"consensus_accuracy":accuracy_score(yy,con),"delta":accuracy_score(yy,con)-accuracy_score(yy,v5),"rescue":r,"harm":h,"net":n})
        details.append(pd.DataFrame({"PassengerId":ids,"Survived":yy,"surface":f"pseudo_{split}","v5":v5,"deotte":de,"eb":eb,"consensus":con}))
        print(f"pseudo {split}: v5={accuracy_score(yy,v5):.5f} consensus={accuracy_score(yy,con):.5f} net={n:+d}",flush=True)
    return pd.DataFrame(rows),pd.concat(details,ignore_index=True)

def main():
    EXPORT_DIR.mkdir(parents=True,exist_ok=True); traw=pd.read_csv(DATA_DIR/"train.csv"); teraw=pd.read_csv(DATA_DIR/"test.csv"); train,_,_,base_cols=prepare_data(); rel=prepare_rel(traw,teraw)
    rr,rd=repeated_audit(traw); pr,pdeta=pseudo_audit(traw,teraw,train,rel,base_cols); allr=pd.concat([rr,pr],ignore_index=True); allr.to_csv(EXPORT_DIR/"surface_metrics.csv",index=False); pd.concat([rd,pdeta],ignore_index=True).to_csv(EXPORT_DIR/"predictions.csv",index=False)
    summary=[]
    for prefix in ["repeated_","pseudo_"]:
        z=allr[allr.surface.str.startswith(prefix)]; summary.append({"surface_family":prefix.rstrip('_'),"mean_delta":float(z.delta.mean()),"min_delta":float(z.delta.min()),"positive":int((z.delta>0).sum()),"nonnegative":int((z.delta>=0).sum()),"negative":int((z.delta<0).sum()),"total_net_correct":int(z.net.sum())})
    sm=pd.DataFrame(summary); sm.to_csv(EXPORT_DIR/"summary.csv",index=False); print("\n=== summary ==="); print(sm.to_string(index=False,float_format=lambda x:f"{x:.5f}")); print("\nNo Kaggle submission was performed.")

if __name__=="__main__": main()
