"""v30: direct repeated matched audit of EB CatBoost vs trusted v5.

Motivation
----------
v29 nested selection showed two things at once:
1) the selector itself only gained +3 correct vs always-v5;
2) static EB CatBoost was the best untouched-outer candidate (0.85185) but was
   never selected by the inner selector.

This script removes selection from the question and directly compares exactly
two predeclared candidates on identical 5-fold splits across five seeds:
  * trusted v5 robust vote
  * FamilyFare + EB25 CatBoost

Seed 2901 is reused from v29 immutable OOF artifacts. Seeds 42/123/777/2026 are
trained here from scratch. No test predictions and no Kaggle submission.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from audit_group_survival import add_group_survival, build_base_frame, build_models as build_v1_models, model_feature_columns
from feature_ablation_v6 import build_models as build_typed_models
from group_key_audit_v6 import helper_keys, peer_feature
from gunes_exact_foldsafe_v11 import structural_frames
from model_zoo_screen_v4 import build_optional_model
from model_zoo_screen_v5 import make_model
from partial_pooling_v27 import EB_COLS, eb_features
from pseudo_test_relational_v14 import role_flags
from tabpfn_finalist_v8 import prepare_data


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "exports" / "v30"
V29_DIR = BASE_DIR / "exports" / "v29"
SEEDS = [42, 123, 777, 2026, 2901]
FAMILYFARE_COLS = ["FamilyFareAny", "FamilyFareMean", "FamilyFareSmooth", "FamilyFareCount"]


def majority3(a, b, c):
    return ((a.astype(int) + b.astype(int) + c.astype(int)) >= 2).astype(int)


def add_familyfare(ref, app, exclude_self):
    stats = peer_feature(ref, app, group_col="FamilyFareGroup_v6", wcg_only=False, exclude_self=exclude_self)
    out = app.copy()
    for source, dest in zip(["Any", "Mean", "Smooth", "Count"], FAMILYFARE_COLS):
        out[dest] = stats[source].to_numpy()
    return out


def add_block(df, block, cols):
    out = df.copy(); b = block.reset_index(drop=True)
    for c in cols:
        out[c] = b[c].to_numpy()
    return out


def prepare_rel(train_raw, test_raw):
    rel_train, _, _, _, _ = structural_frames(train_raw, test_raw)
    rel_train["IsWomanChild"] = role_flags(train_raw)
    trh, _ = helper_keys(train_raw, test_raw)
    return rel_train.merge(trh[["PassengerId", "FamilyFareGroup_v6"]], on="PassengerId", how="left")


def predict_fold(train_raw, train_v1, train_v2, rel_train, v1_cols, v2_cols, tr_idx, va_idx, seed):
    y = train_raw["Survived"].astype(int).to_numpy()

    # trusted v5 robust
    tr1=train_v1.iloc[tr_idx].copy(); va1=train_v1.iloc[va_idx].copy()
    tr1["GroupSurvival"] = add_group_survival(tr1,tr1,exclude_self=True).to_numpy()
    va1["GroupSurvival"] = add_group_survival(tr1,va1,exclude_self=False).to_numpy()
    v1p=[]
    for m in build_v1_models(seed, profile="v1").values():
        m.fit(tr1[v1_cols],y[tr_idx]); v1p.append(np.asarray(m.predict_proba(va1[v1_cols]))[:,1])
    v1_prob=np.mean(v1p,axis=0)

    tr2=train_v2.iloc[tr_idx].copy(); va2=train_v2.iloc[va_idx].copy()
    tr2["GroupSurvival"] = add_group_survival(tr2,tr2,exclude_self=True).to_numpy()
    va2["GroupSurvival"] = add_group_survival(tr2,va2,exclude_self=False).to_numpy()
    tab=build_optional_model("TabICLv2"); tab.fit(tr2[v2_cols],y[tr_idx]); tabp=np.asarray(tab.predict_proba(va2[v2_cols]))[:,1]
    v4b=.90*v1_prob+.10*tabp
    rule=make_model("RuleFit",seed+101); rule.fit(tr2[v2_cols],y[tr_idx]); rp=np.asarray(rule.predict_proba(va2[v2_cols]))[:,1]
    mlpps=[]
    for s in [42,142,242]:
        mlp=make_model("MLP_PLR",s+seed); mlp.fit(tr2[v2_cols],y[tr_idx]); mlpps.append(np.asarray(mlp.predict_proba(va2[v2_cols]))[:,1])
    mp=np.mean(mlpps,axis=0)
    v5=majority3(v4b>.5,rp>.5,mp>.5); v5score=(v4b+rp+mp)/3

    # FamilyFare + EB25 CatBoost
    rr=rel_train.iloc[tr_idx].copy(); rv=rel_train.iloc[va_idx].copy()
    gs_tr=tr2["GroupSurvival"].to_numpy()
    trf=add_familyfare(tr2,tr2,True)
    vaf=add_familyfare(train_v2.iloc[tr_idx].assign(GroupSurvival=gs_tr),va2,False)
    tre,_=eb_features(rr,rr,exclude_self=True); vae,_=eb_features(rr,rv,exclude_self=False)
    trf=add_block(trf,tre,EB_COLS); vaf=add_block(vaf,vae,EB_COLS)
    cols=list(dict.fromkeys(v2_cols+FAMILYFARE_COLS+EB_COLS))
    cat=build_typed_models(seed+404)["CatBoost"]
    cat.fit(trf[cols],y[tr_idx]); ep=np.asarray(cat.predict_proba(vaf[cols]))[:,1]; eb=(ep>.5).astype(int)
    return v5,v5score,eb,ep


def paired_counts(y, a, b):
    # a=EB, b=v5
    rescue=int(np.sum((a==y)&(b!=y)))
    harm=int(np.sum((a!=y)&(b==y)))
    return rescue,harm,rescue-harm


def main():
    EXPORT_DIR.mkdir(parents=True,exist_ok=True)
    train_raw=pd.read_csv(DATA_DIR/"train.csv"); test_raw=pd.read_csv(DATA_DIR/"test.csv")
    y=train_raw["Survived"].astype(int).to_numpy()
    train_v1,_=build_base_frame(train_raw,test_raw,profile="v1"); v1_cols=model_feature_columns(train_v1)
    train_v2,_,_,v2_cols=prepare_data(); rel_train=prepare_rel(train_raw,test_raw)
    rows=[]; oof_frames=[]

    for split_seed in SEEDS:
        if split_seed==2901:
            z=pd.read_csv(V29_DIR/"nested_oof.csv")
            v5=z["v5_robust"].astype(int).to_numpy(); v5s=z["v5_robust_score"].to_numpy(float)
            eb=z["eb_cat"].astype(int).to_numpy(); ebs=z["eb_cat_score"].to_numpy(float)
            folds=z["outer_fold"].astype(int).to_numpy()
        else:
            folds=np.full(len(train_raw),-1,dtype=int)
            skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=split_seed)
            v5=np.zeros(len(train_raw),dtype=int); eb=np.zeros(len(train_raw),dtype=int)
            v5s=np.zeros(len(train_raw)); ebs=np.zeros(len(train_raw))
            for fold,(tr_idx,va_idx) in enumerate(skf.split(train_raw,y)):
                folds[va_idx]=fold
                print(f"seed={split_seed} fold={fold}",flush=True)
                a,ascore,b,bscore=predict_fold(train_raw,train_v1,train_v2,rel_train,v1_cols,v2_cols,tr_idx,va_idx,split_seed*100+fold)
                v5[va_idx]=a; v5s[va_idx]=ascore; eb[va_idx]=b; ebs[va_idx]=bscore
        rescue,harm,net=paired_counts(y,eb,v5)
        rows.extend([
            {"seed":split_seed,"candidate":"v5_robust","accuracy":accuracy_score(y,v5),"roc_auc":roc_auc_score(y,v5s),"rescue_vs_v5":0,"harm_vs_v5":0,"net_vs_v5":0},
            {"seed":split_seed,"candidate":"eb_cat","accuracy":accuracy_score(y,eb),"roc_auc":roc_auc_score(y,ebs),"rescue_vs_v5":rescue,"harm_vs_v5":harm,"net_vs_v5":net},
        ])
        oof_frames.append(pd.DataFrame({"PassengerId":train_raw["PassengerId"].astype(int),"Survived":y,"seed":split_seed,"fold":folds,"v5":v5,"v5_score":v5s,"eb_cat":eb,"eb_cat_score":ebs}))
        print(f"seed {split_seed}: v5={accuracy_score(y,v5):.5f} eb={accuracy_score(y,eb):.5f} rescue={rescue} harm={harm} net={net:+d}",flush=True)

    metrics=pd.DataFrame(rows); metrics.to_csv(EXPORT_DIR/"seed_metrics.csv",index=False)
    pd.concat(oof_frames,ignore_index=True).to_csv(EXPORT_DIR/"repeated_oof.csv",index=False)
    pivot=metrics.pivot(index="seed",columns="candidate",values="accuracy").reset_index()
    pivot["delta_eb_vs_v5"]=pivot["eb_cat"]-pivot["v5_robust"]
    pivot.to_csv(EXPORT_DIR/"paired_seed_deltas.csv",index=False)
    summary=pd.DataFrame([{
        "candidate":"eb_cat_vs_v5",
        "mean_v5_accuracy":float(pivot["v5_robust"].mean()),
        "mean_eb_accuracy":float(pivot["eb_cat"].mean()),
        "mean_delta":float(pivot["delta_eb_vs_v5"].mean()),
        "min_delta":float(pivot["delta_eb_vs_v5"].min()),
        "max_delta":float(pivot["delta_eb_vs_v5"].max()),
        "positive_seeds":int((pivot["delta_eb_vs_v5"]>0).sum()),
        "nonnegative_seeds":int((pivot["delta_eb_vs_v5"]>=0).sum()),
        "negative_seeds":int((pivot["delta_eb_vs_v5"]<0).sum()),
        "total_rescue":int(metrics[metrics.candidate=="eb_cat"].rescue_vs_v5.sum()),
        "total_harm":int(metrics[metrics.candidate=="eb_cat"].harm_vs_v5.sum()),
    }])
    summary.to_csv(EXPORT_DIR/"summary.csv",index=False)
    print("\n=== repeated matched summary ==="); print(pivot.to_string(index=False,float_format=lambda x:f"{x:.5f}")); print(summary.to_string(index=False,float_format=lambda x:f"{x:.5f}")); print("\nNo Kaggle submission was performed.")


if __name__=="__main__": main()
