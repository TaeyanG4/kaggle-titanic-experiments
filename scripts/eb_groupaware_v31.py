"""v31: group-aware and shift-weighted promotion audit for EB CatBoost.

Uses the exact saved v23 group-aware fold manifests (seeds 42 and 777) so the
comparison to v5 is paired and immutable. Only EB CatBoost is trained here.
The v23 grouping already blocks Ticket + FamilyGroup components; only 5 train
rows belong to FamilyFare groups that cross those components, so the stress is
effectively strict for the new EB representation.

Also evaluates v29 seed-2901 EB/v5 OOF under the two v26 cross-fitted
train-vs-test propensity weights.

No Kaggle submission is performed.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from audit_group_survival import add_group_survival
from feature_ablation_v6 import build_models as build_typed_models
from group_key_audit_v6 import helper_keys, peer_feature
from gunes_exact_foldsafe_v11 import structural_frames
from partial_pooling_v27 import EB_COLS, eb_features
from pseudo_test_relational_v14 import role_flags
from tabpfn_finalist_v8 import prepare_data


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "exports" / "v31"
V23_DIR = BASE_DIR / "exports" / "v23"
V26_DIR = BASE_DIR / "exports" / "v26"
V29_DIR = BASE_DIR / "exports" / "v29"
FAMILYFARE_COLS = ["FamilyFareAny", "FamilyFareMean", "FamilyFareSmooth", "FamilyFareCount"]


def add_familyfare(ref, app, exclude_self):
    stats=peer_feature(ref,app,group_col="FamilyFareGroup_v6",wcg_only=False,exclude_self=exclude_self)
    out=app.copy()
    for source,dest in zip(["Any","Mean","Smooth","Count"],FAMILYFARE_COLS): out[dest]=stats[source].to_numpy()
    return out


def add_block(df, block):
    out=df.copy(); b=block.reset_index(drop=True)
    for c in EB_COLS: out[c]=b[c].to_numpy()
    return out


def prepare_rel(train_raw,test_raw):
    rel_train,_,_,_,_=structural_frames(train_raw,test_raw)
    rel_train["IsWomanChild"]=role_flags(train_raw)
    trh,_=helper_keys(train_raw,test_raw)
    return rel_train.merge(trh[["PassengerId","FamilyFareGroup_v6"]],on="PassengerId",how="left")


def paired(y, new, base):
    rescue=int(np.sum((new==y)&(base!=y))); harm=int(np.sum((new!=y)&(base==y)))
    return rescue,harm,rescue-harm


def weighted_acc(y,p,w): return float(np.sum(w*(p==y))/np.sum(w))


def main():
    EXPORT_DIR.mkdir(parents=True,exist_ok=True)
    train_raw=pd.read_csv(DATA_DIR/"train.csv"); test_raw=pd.read_csv(DATA_DIR/"test.csv")
    train,_,_,base_cols=prepare_data(); rel=prepare_rel(train_raw,test_raw); y=train["Survived"].astype(int).to_numpy()
    cols=list(dict.fromkeys(base_cols+FAMILYFARE_COLS+EB_COLS))
    rows=[]; oof_frames=[]

    for seed in [42,777]:
        ref=pd.read_csv(V23_DIR/f"groupaware_oof_seed{seed}.csv")
        if not ref.PassengerId.astype(int).equals(train.PassengerId.astype(int)): raise ValueError("PassengerId mismatch")
        folds=ref.fold.astype(int).to_numpy(); v5=ref.v5_robust.astype(int).to_numpy()
        ebp=np.zeros(len(train)); eb=np.zeros(len(train),dtype=int)
        for fold in sorted(np.unique(folds)):
            tr_idx=np.flatnonzero(folds!=fold); va_idx=np.flatnonzero(folds==fold)
            tr=train.iloc[tr_idx].copy(); va=train.iloc[va_idx].copy(); rr=rel.iloc[tr_idx].copy(); rv=rel.iloc[va_idx].copy()
            gs_tr=add_group_survival(tr,tr,exclude_self=True).to_numpy(); gs_va=add_group_survival(tr,va,exclude_self=False).to_numpy()
            tr["GroupSurvival"]=gs_tr; va["GroupSurvival"]=gs_va
            tr=add_familyfare(tr,tr,True); va=add_familyfare(train.iloc[tr_idx].assign(GroupSurvival=gs_tr),va,False)
            tre,_=eb_features(rr,rr,exclude_self=True); vae,_=eb_features(rr,rv,exclude_self=False)
            tr=add_block(tr,tre); va=add_block(va,vae)
            model=build_typed_models(31000+seed+int(fold))["CatBoost"]
            model.fit(tr[cols],y[tr_idx]); p=np.asarray(model.predict_proba(va[cols]))[:,1]
            ebp[va_idx]=p; eb[va_idx]=(p>.5).astype(int)
        rescue,harm,net=paired(y,eb,v5)
        rows.append({"surface":f"group_seed{seed}","v5_accuracy":accuracy_score(y,v5),"eb_accuracy":accuracy_score(y,eb),"delta":accuracy_score(y,eb)-accuracy_score(y,v5),"eb_auc":roc_auc_score(y,ebp),"rescue":rescue,"harm":harm,"net_correct":net})
        oof_frames.append(pd.DataFrame({"PassengerId":train.PassengerId.astype(int),"Survived":y,"seed":seed,"fold":folds,"v5":v5,"eb_cat":eb,"eb_cat_score":ebp}))
        print(f"group seed {seed}: v5={accuracy_score(y,v5):.5f} eb={accuracy_score(y,eb):.5f} rescue={rescue} harm={harm} net={net:+d}",flush=True)

    # Shift-weighted audit on the untouched v29 outer OOF surface.
    n=pd.read_csv(V29_DIR/"nested_oof.csv"); v5=n.v5_robust.astype(int).to_numpy(); eb=n.eb_cat.astype(int).to_numpy()
    weights=pd.read_csv(V26_DIR/"train_shift_weights.csv")
    for wm in ["logistic","lightgbm"]:
        w=weights[weights.weight_model==wm].sort_values("train_row").weight.to_numpy(float)
        rows.append({"surface":f"shift_{wm}","v5_accuracy":weighted_acc(y,v5,w),"eb_accuracy":weighted_acc(y,eb,w),"delta":weighted_acc(y,eb,w)-weighted_acc(y,v5,w),"eb_auc":np.nan,"rescue":np.nan,"harm":np.nan,"net_correct":np.nan})

    result=pd.DataFrame(rows); result.to_csv(EXPORT_DIR/"promotion_surfaces.csv",index=False)
    pd.concat(oof_frames,ignore_index=True).to_csv(EXPORT_DIR/"groupaware_eb_oof.csv",index=False)
    summary=pd.DataFrame([{
        "group_mean_delta":float(result[result.surface.str.startswith('group_')].delta.mean()),
        "group_min_delta":float(result[result.surface.str.startswith('group_')].delta.min()),
        "group_positive":int((result[result.surface.str.startswith('group_')].delta>0).sum()),
        "shift_mean_delta":float(result[result.surface.str.startswith('shift_')].delta.mean()),
        "shift_min_delta":float(result[result.surface.str.startswith('shift_')].delta.min()),
    }])
    summary.to_csv(EXPORT_DIR/"summary.csv",index=False)
    print("\n=== EB promotion surfaces ==="); print(result.to_string(index=False,float_format=lambda x:f"{x:.5f}")); print(summary.to_string(index=False,float_format=lambda x:f"{x:.5f}")); print("\nNo Kaggle submission was performed.")


if __name__=="__main__": main()
