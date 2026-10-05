"""v38: validate the high-confidence raw-text rescue directly on a v10 analogue.

Rule is frozen from v37:
  if text_only disagrees with the base model and text confidence >= 0.75,
  replace the base prediction with the text prediction.

Here the base is not v5; it is the leakage-safe transductive analogue of the
historical v10 Gunes RF. We validate the exact same rule on:
  * six repeated stratified 5-fold seeds;
  * two Ticket+Family group-aware 5-fold seeds;
  * five matched pseudo-test holdouts.

Only after these checks do we construct the direct v10 test candidate. The
real-test rule changes exactly the rows justified by the frozen text gate.
No Kaggle submission is performed.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from groupaware_validation_v23 import connected_groups
from gunes_exact_foldsafe_v11 import structural_frames
from pseudo_test_relational_v14 import build_pseudo_splits
from v10_trusted_conflict_selector_v13 import eligible_groups, group_meta, historical_partition_scale, make_rf


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "exports" / "v38"
SUB_DIR = BASE_DIR / "submissions"
V36 = BASE_DIR / "exports" / "v36" / "predictions.csv"
V37 = BASE_DIR / "exports" / "v37" / "test_details.csv"
THRESHOLD = 0.75
REPEATED_SEEDS = [42,123,777,2026,31415,27182]


def trans_predict(train_df, base_cols, tr_idx, va_idx):
    y=train_df.Survived.astype(int).to_numpy(); tr=train_df.iloc[tr_idx].copy(); va=train_df.iloc[va_idx].copy()
    fam,tic=eligible_groups(tr,va)
    trm=group_meta(tr,tr,fam,tic,exclude_self=True); vam=group_meta(tr,va,fam,tic,exclude_self=False)
    xtr=np.column_stack([tr[base_cols].to_numpy(float),trm[["SurvivalRate","SurvivalRateNA"]].to_numpy()])
    xva=np.column_stack([va[base_cols].to_numpy(float),vam[["SurvivalRate","SurvivalRateNA"]].to_numpy()])
    xtr=historical_partition_scale(xtr); xva=historical_partition_scale(xva)
    m=make_rf(); m.fit(xtr,y[tr_idx]); prob=m.predict_proba(xva)[:,1]
    return (prob>=.5).astype(int),prob


def gate(base,text_pred,text_prob):
    conf=np.maximum(text_prob,1-text_prob); mask=(text_pred!=base)&(conf>=THRESHOLD); out=base.copy(); out[mask]=text_pred[mask]; return out,mask,conf


def paired(y,new,base):
    rescue=int(np.sum((new==y)&(base!=y))); harm=int(np.sum((new!=y)&(base==y))); return rescue,harm,rescue-harm


def get_text(surface):
    p=pd.read_csv(V36); q=p[(p.surface==surface)&(p.variant=="text_only")].copy(); return q.sort_values("PassengerId").reset_index(drop=True)


def repeated(train_raw,train_df,base_cols):
    y=train_df.Survived.astype(int).to_numpy(); rows=[]; details=[]
    for seed in REPEATED_SEEDS:
        skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed); base=np.zeros(len(train_df),int); prob=np.zeros(len(train_df),float); fold_id=np.full(len(train_df),-1,int)
        for fold,(tr_idx,va_idx) in enumerate(skf.split(train_df,y)):
            fold_id[va_idx]=fold; p,s=trans_predict(train_df,base_cols,tr_idx,va_idx); base[va_idx]=p; prob[va_idx]=s
        tx=get_text(f"repeated_{seed}")
        if not np.array_equal(tx.PassengerId.astype(int).to_numpy(),train_raw.PassengerId.astype(int).to_numpy()): raise ValueError(f"text order {seed}")
        out,mask,conf=gate(base,tx.text_pred.astype(int).to_numpy(),tx.text_prob.to_numpy(float)); r,h,n=paired(y,out,base)
        rows.append({"surface":f"repeated_{seed}","family":"repeated","base_accuracy":accuracy_score(y,base),"gated_accuracy":accuracy_score(y,out),"delta":accuracy_score(y,out)-accuracy_score(y,base),"base_auc":roc_auc_score(y,prob),"switches":int(mask.sum()),"rescue":r,"harm":h,"net":n})
        details.append(pd.DataFrame({"PassengerId":train_raw.PassengerId.astype(int),"Survived":y,"surface":f"repeated_{seed}","fold":fold_id,"base":base,"base_prob":prob,"text_pred":tx.text_pred.astype(int),"text_prob":tx.text_prob,"switch":mask.astype(int),"gated":out}))
        print(f"repeated {seed}: base={accuracy_score(y,base):.5f} gated={accuracy_score(y,out):.5f} switches={mask.sum()} net={n:+d}",flush=True)
    return pd.DataFrame(rows),pd.concat(details,ignore_index=True)


def group(train_raw,train_df,base_cols):
    y=train_df.Survived.astype(int).to_numpy(); groups=connected_groups(train_raw); rows=[]; details=[]
    for seed in [42,777]:
        sg=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed); base=np.zeros(len(train_df),int); prob=np.zeros(len(train_df),float); fold_id=np.full(len(train_df),-1,int)
        for fold,(tr_idx,va_idx) in enumerate(sg.split(train_df,y,groups)):
            fold_id[va_idx]=fold; p,s=trans_predict(train_df,base_cols,tr_idx,va_idx); base[va_idx]=p; prob[va_idx]=s
        tx=get_text(f"group_{seed}"); out,mask,conf=gate(base,tx.text_pred.astype(int).to_numpy(),tx.text_prob.to_numpy(float)); r,h,n=paired(y,out,base)
        rows.append({"surface":f"group_{seed}","family":"group","base_accuracy":accuracy_score(y,base),"gated_accuracy":accuracy_score(y,out),"delta":accuracy_score(y,out)-accuracy_score(y,base),"base_auc":roc_auc_score(y,prob),"switches":int(mask.sum()),"rescue":r,"harm":h,"net":n})
        details.append(pd.DataFrame({"PassengerId":train_raw.PassengerId.astype(int),"Survived":y,"surface":f"group_{seed}","fold":fold_id,"base":base,"base_prob":prob,"text_pred":tx.text_pred.astype(int),"text_prob":tx.text_prob,"switch":mask.astype(int),"gated":out}))
        print(f"group {seed}: base={accuracy_score(y,base):.5f} gated={accuracy_score(y,out):.5f} switches={mask.sum()} net={n:+d}",flush=True)
    return pd.DataFrame(rows),pd.concat(details,ignore_index=True)


def pseudo(train_raw,test_raw,train_df,base_cols):
    y=train_df.Survived.astype(int).to_numpy(); _,splits=build_pseudo_splits(train_raw,test_raw); rows=[]; details=[]
    for split,(_,va_idx,_) in enumerate(splits):
        tr_idx=np.setdiff1d(np.arange(len(train_df)),va_idx); base,prob=trans_predict(train_df,base_cols,tr_idx,va_idx); tx=get_text(f"pseudo_{split}"); ids=train_raw.iloc[va_idx].PassengerId.astype(int).to_numpy();
        if not np.array_equal(tx.PassengerId.astype(int).to_numpy(),np.sort(ids)):
            # v36 pseudo detail is in validation-index order, not necessarily sorted; align by ID.
            tx=tx.set_index("PassengerId").loc[ids].reset_index()
        yy=y[va_idx]; out,mask,conf=gate(base,tx.text_pred.astype(int).to_numpy(),tx.text_prob.to_numpy(float)); r,h,n=paired(yy,out,base)
        rows.append({"surface":f"pseudo_{split}","family":"pseudo","base_accuracy":accuracy_score(yy,base),"gated_accuracy":accuracy_score(yy,out),"delta":accuracy_score(yy,out)-accuracy_score(yy,base),"base_auc":roc_auc_score(yy,prob),"switches":int(mask.sum()),"rescue":r,"harm":h,"net":n})
        details.append(pd.DataFrame({"PassengerId":ids,"Survived":yy,"surface":f"pseudo_{split}","base":base,"base_prob":prob,"text_pred":tx.text_pred.astype(int),"text_prob":tx.text_prob,"switch":mask.astype(int),"gated":out}))
        print(f"pseudo {split}: base={accuracy_score(yy,base):.5f} gated={accuracy_score(yy,out):.5f} switches={mask.sum()} net={n:+d}",flush=True)
    return pd.DataFrame(rows),pd.concat(details,ignore_index=True)


def main():
    EXPORT_DIR.mkdir(parents=True,exist_ok=True); SUB_DIR.mkdir(parents=True,exist_ok=True)
    traw=pd.read_csv(DATA_DIR/"train.csv"); teraw=pd.read_csv(DATA_DIR/"test.csv"); train_df,_,base_cols,_,_=structural_frames(traw,teraw)
    rr,rd=repeated(traw,train_df,base_cols); gr,gd=group(traw,train_df,base_cols); pr,pdeta=pseudo(traw,teraw,train_df,base_cols)
    metrics=pd.concat([rr,gr,pr],ignore_index=True); metrics.to_csv(EXPORT_DIR/"surface_metrics.csv",index=False); pd.concat([rd,gd,pdeta],ignore_index=True).to_csv(EXPORT_DIR/"predictions.csv",index=False)
    summary=metrics.groupby("family",as_index=False).agg(mean_delta=("delta","mean"),min_delta=("delta","min"),switches=("switches","sum"),rescue=("rescue","sum"),harm=("harm","sum"),net=("net","sum"),positive=("delta",lambda s:int((s>0).sum())),nonnegative=("delta",lambda s:int((s>=0).sum())))
    summary.to_csv(EXPORT_DIR/"summary.csv",index=False)

    td=pd.read_csv(V37); base=td.v10.astype(int).to_numpy(); tp=td.text_pred.astype(int).to_numpy(); prob=td.text_prob.to_numpy(float); cand,mask,conf=gate(base,tp,prob); ids=td.PassengerId.astype(int).to_numpy()
    pd.DataFrame({"PassengerId":ids,"Survived":cand}).to_csv(SUB_DIR/"submission_v38_v10_text_rescue.csv",index=False)
    test_detail=td.copy(); test_detail["v10_text_candidate"]=cand; test_detail["v10_text_switch"]=mask.astype(int); test_detail.to_csv(EXPORT_DIR/"test_details.csv",index=False)
    test_summary=pd.DataFrame([{"changed_vs_v10":int(mask.sum()),"test_positives":int(cand.sum()),"all_switches_text_seed_unanimous":int(np.all((td.loc[mask,'text_seed_positive_votes'].to_numpy()==0)|(td.loc[mask,'text_seed_positive_votes'].to_numpy()==6))) if mask.any() else 1}]); test_summary.to_csv(EXPORT_DIR/"test_summary.csv",index=False)
    print("\n=== summary ==="); print(summary.to_string(index=False,float_format=lambda x:f"{x:.5f}")); print("\n=== direct v10 test switches ==="); print(test_detail[test_detail.v10_text_switch==1][["PassengerId","v10","text_pred","text_confidence","text_seed_positive_votes","v10_text_candidate"]].to_string(index=False)); print("\nNo Kaggle submission was performed.")


if __name__=="__main__": main()
