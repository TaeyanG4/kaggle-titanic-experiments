"""v44: paired switch-precision audit for the v43 P3-full finalist.

The v43 test candidate differs from the submitted v5 on 24 rows and therefore
needs >=17/24 = 70.83% correct switches to beat the known v10 public score.
This script measures the analogous switch precision against v5 on clean local
surfaces, using exactly the v42 P3-full model seeds.

It also forms a six-seed bagged OOF P3 prediction and compares it to the original
fixed v5 robust OOF, which most closely mirrors v43-vs-submitted-v5 at test time.
No submission.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from feature_ablation_v6 import build_models as build_tree_models
from groupaware_validation_v23 import connected_groups
from pseudo_test_relational_v14 import build_pseudo_splits
from raw_string_text_v36 import load_v5
from tabpfn_finalist_v8 import prepare_data
from partial_pooling_transfer_v28 import prepare_rel
from ticket_numeric_prefix_v41 import augment_rel
from ticket_prefix_ablation_v42 import split_features


BASE_DIR=Path(__file__).resolve().parents[1]
DATA_DIR=BASE_DIR/"data"
EXPORT_DIR=BASE_DIR/"exports"/"v44"
SEEDS=[42,123,777,2026,31415,27182]
P3_FULL=["TicketP3EB","TicketP3Count","TicketP3Delta","TicketP3RoleEB","TicketP3RoleCount","TicketP3RoleDelta","TicketDigitLength"]


def fit_split(train,rel,base_cols,y,tr_idx,va_idx,model_seed):
    tr,va,parent_cols=split_features(train,rel,base_cols,tr_idx,va_idx)
    cols=parent_cols+P3_FULL; m=build_tree_models(model_seed)["CatBoost"]; m.fit(tr[cols],y[tr_idx]); return np.asarray(m.predict_proba(va[cols]))[:,1]


def paired(y,new,base):
    m=new!=base; r=int(np.sum(m&(new==y)&(base!=y))); h=int(np.sum(m&(new!=y)&(base==y))); d=int(m.sum()); return d,r,h,r-h,(r/d if d else np.nan)


def original_v5_fixed(y):
    z=pd.read_csv(BASE_DIR/"exports"/"v5"/"model_zoo_oof.csv"); mlp=pd.read_csv(BASE_DIR/"exports"/"v5"/"seed_probes"/"MLP_PLR_seed_mean__oof.csv")
    pred=(((z["v4b__Champion"].to_numpy()>.5).astype(int)+(z.RuleFit.to_numpy()>.5).astype(int)+(mlp.probability.to_numpy()>.5).astype(int))>=2).astype(int)
    if not np.array_equal(z.Survived.astype(int).to_numpy(),y): raise ValueError("v5 y mismatch")
    return pred


def main():
    EXPORT_DIR.mkdir(parents=True,exist_ok=True); traw=pd.read_csv(DATA_DIR/"train.csv"); teraw=pd.read_csv(DATA_DIR/"test.csv"); train,_,fixed,base_cols=prepare_data(); rel,_=prepare_rel(traw,teraw); rel=augment_rel(rel); y=train.Survived.astype(int).to_numpy(); rows=[]; details=[]; repeated_probs=[]

    # Repeated stratified, exact p3_full seed semantics from v42: 46000+seed+fold + 3000.
    for seed in SEEDS:
        if seed==42: folds=fixed.copy()
        else:
            folds=np.full(len(train),-1,int); sk=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
            for f,(_,va) in enumerate(sk.split(train,y)): folds[va]=f
        prob=np.zeros(len(train))
        for f in sorted(np.unique(folds)):
            tr=np.flatnonzero(folds!=f); va=np.flatnonzero(folds==f); prob[va]=fit_split(train,rel,base_cols,y,tr,va,49000+seed+int(f))
        pred=(prob>=.5).astype(int); v5df=load_v5(seed); v5=v5df.v5.astype(int).to_numpy(); d,r,h,n,prec=paired(y,pred,v5)
        rows.append({"family":"repeated","surface":f"repeated_{seed}","p3_accuracy":accuracy_score(y,pred),"v5_accuracy":accuracy_score(y,v5),"switches":d,"rescue":r,"harm":h,"net":n,"switch_precision":prec})
        details.append(pd.DataFrame({"PassengerId":train.PassengerId.astype(int),"Survived":y,"surface":f"repeated_{seed}","p3_prob":prob,"p3_pred":pred,"v5":v5})); repeated_probs.append(prob)
        print(f"repeated {seed}: p3={accuracy_score(y,pred):.5f} v5={accuracy_score(y,v5):.5f} switches={d} precision={prec:.3f} net={n:+d}",flush=True)

    # Bagged OOF across the six independently cross-fitted repeats vs original fixed v5.
    bag_prob=np.mean(np.vstack(repeated_probs),axis=0); bag=(bag_prob>=.5).astype(int); v5fixed=original_v5_fixed(y); d,r,h,n,prec=paired(y,bag,v5fixed)
    rows.append({"family":"bagged","surface":"bagged6_vs_original_v5","p3_accuracy":accuracy_score(y,bag),"v5_accuracy":accuracy_score(y,v5fixed),"switches":d,"rescue":r,"harm":h,"net":n,"switch_precision":prec})
    pd.DataFrame({"PassengerId":train.PassengerId.astype(int),"Survived":y,"p3_bag_prob":bag_prob,"p3_bag_pred":bag,"v5_fixed":v5fixed}).to_csv(EXPORT_DIR/"bagged_oof.csv",index=False)

    # Group-aware, exact v42 model seeds.
    groups=connected_groups(traw)
    for seed in [42,777]:
        sg=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed); prob=np.zeros(len(train))
        for f,(tr,va) in enumerate(sg.split(train,y,groups)): prob[va]=fit_split(train,rel,base_cols,y,tr,va,50000+seed+f)
        pred=(prob>=.5).astype(int); old=pd.read_csv(BASE_DIR/"exports"/"v23"/f"groupaware_oof_seed{seed}.csv").sort_values("PassengerId"); v5=old.v5_robust.astype(int).to_numpy(); d,r,h,n,prec=paired(y,pred,v5)
        rows.append({"family":"group","surface":f"group_{seed}","p3_accuracy":accuracy_score(y,pred),"v5_accuracy":accuracy_score(y,v5),"switches":d,"rescue":r,"harm":h,"net":n,"switch_precision":prec})
        details.append(pd.DataFrame({"PassengerId":train.PassengerId.astype(int),"Survived":y,"surface":f"group_{seed}","p3_prob":prob,"p3_pred":pred,"v5":v5}))
        print(f"group {seed}: p3={accuracy_score(y,pred):.5f} v5={accuracy_score(y,v5):.5f} switches={d} precision={prec:.3f} net={n:+d}",flush=True)

    # Matched pseudo, exact v42 model seeds.
    saved=pd.read_csv(BASE_DIR/"exports"/"v19"/"pseudotest_consensus_predictions.csv"); _,pseudo=build_pseudo_splits(traw,teraw)
    for s,(_,va,_) in enumerate(pseudo):
        tr=np.setdiff1d(np.arange(len(train)),va); prob=fit_split(train,rel,base_cols,y,tr,va,51000+s); pred=(prob>=.5).astype(int); ids=train.iloc[va].PassengerId.astype(int).to_numpy(); ref=saved[saved.split==s].set_index("PassengerId").loc[ids]; v5=ref.v5.astype(int).to_numpy(); yy=y[va]; d,r,h,n,prec=paired(yy,pred,v5)
        rows.append({"family":"pseudo","surface":f"pseudo_{s}","p3_accuracy":accuracy_score(yy,pred),"v5_accuracy":accuracy_score(yy,v5),"switches":d,"rescue":r,"harm":h,"net":n,"switch_precision":prec})
        details.append(pd.DataFrame({"PassengerId":ids,"Survived":yy,"surface":f"pseudo_{s}","p3_prob":prob,"p3_pred":pred,"v5":v5}))
        print(f"pseudo {s}: p3={accuracy_score(yy,pred):.5f} v5={accuracy_score(yy,v5):.5f} switches={d} precision={prec:.3f} net={n:+d}",flush=True)

    m=pd.DataFrame(rows); m.to_csv(EXPORT_DIR/"surface_metrics.csv",index=False); pd.concat(details,ignore_index=True).to_csv(EXPORT_DIR/"predictions.csv",index=False)
    sm=m.groupby("family",as_index=False).agg(mean_p3_accuracy=("p3_accuracy","mean"),mean_v5_accuracy=("v5_accuracy","mean"),switches=("switches","sum"),rescue=("rescue","sum"),harm=("harm","sum"),net=("net","sum"),mean_switch_precision=("switch_precision","mean"),min_switch_precision=("switch_precision","min"))
    sm["pooled_switch_precision"]=sm.rescue/(sm.rescue+sm.harm); sm.to_csv(EXPORT_DIR/"summary.csv",index=False)
    print("\n=== v44 summary ==="); print(sm.to_string(index=False,float_format=lambda x:f"{x:.5f}")); print("\n=== bagged OOF ==="); print(m[m.family=="bagged"].to_string(index=False,float_format=lambda x:f"{x:.5f}")); print("\nNo Kaggle submission was performed.")


if __name__=="__main__": main()
