"""v28: transfer v27 empirical-Bayes relational features to stronger v2/FamilyFare trees.

Stage 1 screens 3 representations x 3 strong tree families on the immutable
seed-42 folds. Stage 2 confirms only the top 3 fixed-fold candidates on seeds
123/777/2026. This keeps selection degrees of freedom small.

No Kaggle submission is performed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from audit_group_survival import add_group_survival
from feature_ablation_v6 import build_models
from group_key_audit_v6 import helper_keys, peer_feature
from gunes_exact_foldsafe_v11 import structural_frames
from partial_pooling_v27 import EB_COLS, eb_features
from pseudo_test_relational_v14 import VARIANTS as REL_VARIANTS, relation_features, role_flags
from tabpfn_finalist_v8 import load_champion_context, prepare_data


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "exports" / "v28"
V26_DIR = BASE_DIR / "exports" / "v26"

TYPED_COLS = REL_VARIANTS["typed22"]
FAMILYFARE_COLS = ["FamilyFareAny", "FamilyFareMean", "FamilyFareSmooth", "FamilyFareCount"]
MODELS = ["GradientBoosting", "XGBoost", "CatBoost"]
VARIANTS = {
    "familyfare_typed22": TYPED_COLS,
    "familyfare_eb25": EB_COLS,
    "familyfare_typed22_eb25": TYPED_COLS + EB_COLS,
}


def add_familyfare(ref, app, exclude_self):
    stats = peer_feature(ref, app, group_col="FamilyFareGroup_v6", wcg_only=False, exclude_self=exclude_self)
    out = app.copy()
    for source, dest in zip(["Any", "Mean", "Smooth", "Count"], FAMILYFARE_COLS):
        out[dest] = stats[source].to_numpy()
    return out


def add_block(df: pd.DataFrame, block: pd.DataFrame, cols: list[str]):
    out = df.copy()
    b = block.reset_index(drop=True)
    for c in cols:
        out[c] = b[c].to_numpy()
    return out


def prepare_rel(train_raw, test_raw):
    rel_train, rel_test, _, _, _ = structural_frames(train_raw, test_raw)
    rel_train["IsWomanChild"] = role_flags(train_raw)
    rel_test["IsWomanChild"] = role_flags(test_raw)
    trh, teh = helper_keys(train_raw, test_raw)
    rel_train = rel_train.merge(trh[["PassengerId", "FamilyFareGroup_v6"]], on="PassengerId", how="left")
    rel_test = rel_test.merge(teh[["PassengerId", "FamilyFareGroup_v6"]], on="PassengerId", how="left")
    return rel_train, rel_test


def fold_features(train, rel_train, tr_idx, va_idx, base_cols, variant):
    tr = train.iloc[tr_idx].copy(); va = train.iloc[va_idx].copy()
    rr = rel_train.iloc[tr_idx].copy(); rv = rel_train.iloc[va_idx].copy()
    gs_tr = add_group_survival(tr, tr, exclude_self=True).to_numpy()
    gs_va = add_group_survival(tr, va, exclude_self=False).to_numpy()
    tr["GroupSurvival"] = gs_tr; va["GroupSurvival"] = gs_va
    tr = add_familyfare(tr, tr, True)
    va = add_familyfare(train.iloc[tr_idx].assign(GroupSurvival=gs_tr), va, False)

    cols = list(base_cols) + FAMILYFARE_COLS
    if "typed22" in variant:
        tr_t = relation_features(rr, rr, exclude_self=True, alpha=2.0)
        va_t = relation_features(rr, rv, exclude_self=False, alpha=2.0)
        tr = add_block(tr, tr_t, TYPED_COLS); va = add_block(va, va_t, TYPED_COLS)
        cols += TYPED_COLS
    if "eb25" in variant:
        tr_e, _ = eb_features(rr, rr, exclude_self=True)
        va_e, _ = eb_features(rr, rv, exclude_self=False)
        tr = add_block(tr, tr_e, EB_COLS); va = add_block(va, va_e, EB_COLS)
        cols += EB_COLS
    cols = list(dict.fromkeys(cols))
    return tr, va, cols


def eval_candidate(train, rel_train, folds, base_cols, variant, model_name, seed_offset):
    y = train["Survived"].astype(int).to_numpy()
    oof = np.zeros(len(train), dtype=float); fold_rows=[]
    for fold in sorted(np.unique(folds)):
        tr_idx=np.flatnonzero(folds!=fold); va_idx=np.flatnonzero(folds==fold)
        tr,va,cols=fold_features(train,rel_train,tr_idx,va_idx,base_cols,variant)
        model=build_models(seed_offset+int(fold))[model_name]
        model.fit(tr[cols],y[tr_idx]); p=model.predict_proba(va[cols])[:,1]; oof[va_idx]=p
        fold_rows.append({"fold":int(fold),"accuracy":accuracy_score(y[va_idx],p>.5),"roc_auc":roc_auc_score(y[va_idx],p)})
    f=pd.DataFrame(fold_rows)
    return {
        "accuracy": accuracy_score(y,oof>.5),
        "roc_auc": roc_auc_score(y,oof),
        "fold_acc_std": float(f.accuracy.std(ddof=0)),
        "oof":oof,
        "fold_rows":f,
    }


def main():
    EXPORT_DIR.mkdir(parents=True,exist_ok=True)
    train_raw=pd.read_csv(DATA_DIR/"train.csv"); test_raw=pd.read_csv(DATA_DIR/"test.csv")
    train,_,fixed_folds,base_cols=prepare_data(); rel_train,_=prepare_rel(train_raw,test_raw)
    y=train["Survived"].astype(int).to_numpy()
    zoo,_,champ,_,_=load_champion_context()

    fixed_rows=[]; fixed_oof={}; fold_rows=[]
    for variant in VARIANTS:
        for model_name in MODELS:
            print(f"=== fixed {variant} / {model_name} ===",flush=True)
            r=eval_candidate(train,rel_train,fixed_folds,base_cols,variant,model_name,28000)
            stem=f"{variant}__{model_name}"; fixed_oof[stem]=r["oof"]
            pred=(r["oof"]>.5).astype(int)
            fixed_rows.append({"candidate":stem,"accuracy":r["accuracy"],"roc_auc":r["roc_auc"],"fold_acc_std":r["fold_acc_std"],"unique_correct_vs_v5":int(np.sum((pred==y)&(champ!=y))),"unique_wrong_vs_v5":int(np.sum((pred!=y)&(champ==y)))})
            ff=r["fold_rows"].copy(); ff["candidate"]=stem; fold_rows.append(ff)
            print(f"acc={r['accuracy']:.5f} auc={r['roc_auc']:.5f} unique={fixed_rows[-1]['unique_correct_vs_v5']}/{fixed_rows[-1]['unique_wrong_vs_v5']}",flush=True)
    fixed=pd.DataFrame(fixed_rows).sort_values(["accuracy","roc_auc"],ascending=False)
    fixed.to_csv(EXPORT_DIR/"fixed_screen.csv",index=False)
    pd.concat(fold_rows,ignore_index=True).to_csv(EXPORT_DIR/"fixed_fold_metrics.csv",index=False)
    pd.DataFrame({"PassengerId":train["PassengerId"].astype(int),"Survived":y,**fixed_oof}).to_csv(EXPORT_DIR/"fixed_oof.csv",index=False)

    finalists=fixed.head(3)["candidate"].tolist()
    alt_rows=[]
    for seed in [123,777,2026]:
        folds=np.full(len(train),-1,dtype=int); skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
        for f,(_,va) in enumerate(skf.split(train,y)): folds[va]=f
        for stem in finalists:
            variant,model_name=stem.split("__",1)
            r=eval_candidate(train,rel_train,folds,base_cols,variant,model_name,seed*10)
            alt_rows.append({"seed":seed,"candidate":stem,"accuracy":r["accuracy"],"roc_auc":r["roc_auc"],"fold_acc_std":r["fold_acc_std"]})
            print(f"alt seed={seed} {stem}: acc={r['accuracy']:.5f} auc={r['roc_auc']:.5f}",flush=True)
    alt=pd.DataFrame(alt_rows); alt.to_csv(EXPORT_DIR/"alternate_confirmation.csv",index=False)

    conf=[]
    for stem in finalists:
        f=fixed[fixed.candidate==stem].iloc[0]; a=alt[alt.candidate==stem]
        vals=[float(f.accuracy),*a.accuracy.tolist()]
        conf.append({"candidate":stem,"fixed_accuracy":f.accuracy,"fixed_auc":f.roc_auc,"mean4_accuracy":float(np.mean(vals)),"min4_accuracy":float(np.min(vals)),"std4_accuracy":float(np.std(vals)),"alt_mean_auc":float(a.roc_auc.mean()),"unique_correct_vs_v5":int(f.unique_correct_vs_v5),"unique_wrong_vs_v5":int(f.unique_wrong_vs_v5)})
    confirmed=pd.DataFrame(conf).sort_values(["mean4_accuracy","fixed_accuracy"],ascending=False)
    confirmed.to_csv(EXPORT_DIR/"confirmed_ranking.csv",index=False)

    # v26 shift-weighted fixed-screen audit.
    weights=pd.read_csv(V26_DIR/"train_shift_weights.csv")
    wr=[]
    for wm in ["logistic","lightgbm"]:
        w=weights[weights.weight_model==wm].sort_values("train_row").weight.to_numpy(float)
        for stem in finalists:
            pred=(fixed_oof[stem]>.5).astype(int)
            wr.append({"weight_model":wm,"candidate":stem,"weighted_accuracy":float(np.sum(w*(pred==y))/np.sum(w))})
    weighted=pd.DataFrame(wr); weighted.to_csv(EXPORT_DIR/"shift_weighted_finalists.csv",index=False)

    print("\n=== fixed screen ==="); print(fixed.to_string(index=False,float_format=lambda v:f"{v:.5f}"))
    print("\n=== confirmed ranking ==="); print(confirmed.to_string(index=False,float_format=lambda v:f"{v:.5f}"))
    print("\n=== shift weighted ==="); print(weighted.to_string(index=False,float_format=lambda v:f"{v:.5f}"))
    print("\nNo Kaggle submission was performed.")


if __name__=="__main__": main()
