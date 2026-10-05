"""Group-aware stress validation for Titanic finalists.

Rows are connected when they share either Ticket or the project's FamilyGroup
(surname + family size). Connected components are kept intact with
StratifiedGroupKFold, preventing the same travel/family component from appearing
on both sides of a fold.

Finalists tested:
  * v5 robust vote (v4b + RuleFit + 3-seed MLP-PLR)
  * Deotte WCG+XGB Python port
  * FamilyFare+typed22 CatBoost
  * FamilyFare+typed22 HPO RandomForest config 6

This is a stress test, not the primary validation. No submission is performed.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier

from audit_group_survival import add_group_survival, build_base_frame, build_models as build_v1_models, model_feature_columns
from deotte_wcg_xgb_v21 import predict_split as deotte_predict_split
from feature_ablation_v6 import build_models as build_typed_models
from group_key_audit_v6 import peer_feature
from gunes_exact_foldsafe_v11 import structural_frames
from model_zoo_screen_v4 import build_optional_model
from model_zoo_screen_v5 import make_model
from pseudo_test_relational_v14 import VARIANTS as REL_VARIANTS, eligible_groups, relation_features, role_flags
from tabpfn_finalist_v8 import prepare_data


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "exports" / "v23"
TYPED_COLS = REL_VARIANTS["typed22"]
FAMILYFARE_COLS = ["FamilyFareAny", "FamilyFareMean", "FamilyFareSmooth", "FamilyFareCount"]


class DSU:
    def __init__(self, n):
        self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[b] = a


def connected_groups(raw: pd.DataFrame) -> np.ndarray:
    n = len(raw)
    dsu = DSU(n)
    surname = raw["Name"].str.split(",").str[0].str.strip()
    fsize = raw["SibSp"] + raw["Parch"] + 1
    family = surname + "_" + fsize.astype(str)
    for keys in [raw["Ticket"].astype(str), family.astype(str)]:
        buckets = {}
        for i, k in enumerate(keys):
            if k in buckets:
                dsu.union(i, buckets[k])
            else:
                buckets[k] = i
    roots = [dsu.find(i) for i in range(n)]
    remap = {r:j for j,r in enumerate(sorted(set(roots)))}
    return np.array([remap[r] for r in roots], dtype=int)


def add_familyfare(ref, app, exclude_self):
    stats = peer_feature(ref, app, group_col="FamilyFareGroup_v6", wcg_only=False, exclude_self=exclude_self)
    out = app.copy()
    for source, dest in zip(["Any", "Mean", "Smooth", "Count"], FAMILYFARE_COLS):
        out[dest] = stats[source].to_numpy()
    return out


def add_rel(df, rel):
    out = df.copy()
    for c in TYPED_COLS:
        out[c] = rel[c].to_numpy()
    return out


def majority3(a,b,c):
    return ((a.astype(int)+b.astype(int)+c.astype(int)) >= 2).astype(int)


def main():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    y = train_raw["Survived"].astype(int).to_numpy()
    groups = connected_groups(train_raw)

    train_v1, _ = build_base_frame(train_raw, test_raw, profile="v1")
    v1_cols = model_feature_columns(train_v1)
    train_v2, _, _, v2_cols = prepare_data()
    rel_train, _, _, _, _ = structural_frames(train_raw, test_raw)
    rel_train["IsWomanChild"] = role_flags(train_raw)
    typed_cols = list(dict.fromkeys(v2_cols + FAMILYFARE_COLS + TYPED_COLS))

    rows = []
    all_oof = []
    for split_seed in [42, 777]:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=split_seed)
        preds = {k:np.zeros(len(train_raw),dtype=int) for k in ["v5_robust","deotte","typed_cat","typed_rf6"]}
        scores = {k:np.zeros(len(train_raw),dtype=float) for k in preds}
        fold_id = np.full(len(train_raw), -1, dtype=int)
        print(f"\n=== group split seed {split_seed} ===", flush=True)

        for fold,(tr_idx,va_idx) in enumerate(sgkf.split(train_raw,y,groups)):
            fold_id[va_idx] = fold
            # ----- v1/v2 fold-safe matrices -----
            tr1=train_v1.iloc[tr_idx].copy(); va1=train_v1.iloc[va_idx].copy()
            tr1["GroupSurvival"] = add_group_survival(tr1,tr1,exclude_self=True).to_numpy()
            va1["GroupSurvival"] = add_group_survival(tr1,va1,exclude_self=False).to_numpy()
            v1_probs=[]
            for m in build_v1_models(split_seed*100+fold, profile="v1").values():
                m.fit(tr1[v1_cols], y[tr_idx]); v1_probs.append(m.predict_proba(va1[v1_cols])[:,1])
            v1_prob=np.mean(v1_probs,axis=0)

            tr2=train_v2.iloc[tr_idx].copy(); va2=train_v2.iloc[va_idx].copy()
            tr2["GroupSurvival"] = add_group_survival(tr2,tr2,exclude_self=True).to_numpy()
            va2["GroupSurvival"] = add_group_survival(tr2,va2,exclude_self=False).to_numpy()

            tabicl=build_optional_model("TabICLv2")
            tabicl.fit(tr2[v2_cols],y[tr_idx]); tab_p=tabicl.predict_proba(va2[v2_cols])[:,1]
            v4b=0.90*v1_prob+0.10*tab_p

            rule=make_model("RuleFit",split_seed*100+fold)
            rule.fit(tr2[v2_cols],y[tr_idx]); rule_p=rule.predict_proba(va2[v2_cols])[:,1]

            mlp_ps=[]
            for seed in [42,142,242]:
                mlp=make_model("MLP_PLR",seed+split_seed+fold)
                mlp.fit(tr2[v2_cols],y[tr_idx]); mlp_ps.append(mlp.predict_proba(va2[v2_cols])[:,1])
            mlp_p=np.mean(mlp_ps,axis=0)
            v5=majority3(v4b>.5, rule_p>.5, mlp_p>.5)
            preds["v5_robust"][va_idx]=v5
            scores["v5_robust"][va_idx]=(v4b+rule_p+mlp_p)/3

            # ----- typed FamilyFare representation -----
            trt=add_familyfare(tr2,tr2,True)
            vat=add_familyfare(train_v2.iloc[tr_idx].assign(GroupSurvival=tr2["GroupSurvival"].to_numpy()),va2,False)
            rr=rel_train.iloc[tr_idx].copy(); rv=rel_train.iloc[va_idx].copy()
            af,at=eligible_groups(rr,rv)
            rr_rel=relation_features(rr,rr,exclude_self=True,alpha=2.0,allowed_fam=af,allowed_tic=at)
            rv_rel=relation_features(rr,rv,exclude_self=False,alpha=2.0,allowed_fam=af,allowed_tic=at)
            trt=add_rel(trt,rr_rel); vat=add_rel(vat,rv_rel)

            cat=build_typed_models(split_seed*100+fold)["CatBoost"]
            cat.fit(trt[typed_cols],y[tr_idx]); cp=cat.predict_proba(vat[typed_cols])[:,1]
            preds["typed_cat"][va_idx]=(cp>.5).astype(int); scores["typed_cat"][va_idx]=cp

            rf=RandomForestClassifier(n_estimators=500,max_depth=7,min_samples_split=8,min_samples_leaf=4,max_features=.7,random_state=split_seed*100+fold,n_jobs=-1)
            rf.fit(trt[typed_cols],y[tr_idx]); rp=rf.predict_proba(vat[typed_cols])[:,1]
            preds["typed_rf6"][va_idx]=(rp>.5).astype(int); scores["typed_rf6"][va_idx]=rp

            # ----- Deotte -----
            db=deotte_predict_split(train_raw,tr_idx,va_idx,seed=split_seed*100+fold,imputer_fit_indices=tr_idx,special_link=True)
            preds["deotte"][va_idx]=db.wcg_both; scores["deotte"][va_idx]=db.score

            for name in preds:
                rows.append({"split_seed":split_seed,"fold":fold,"candidate":name,"accuracy":accuracy_score(y[va_idx],preds[name][va_idx]),"n_val":len(va_idx),"n_groups_val":len(np.unique(groups[va_idx]))})
            print("fold",fold,"n",len(va_idx)," | "," ".join(f"{k}={accuracy_score(y[va_idx],preds[k][va_idx]):.4f}" for k in preds),flush=True)

        frame=pd.DataFrame({"PassengerId":train_raw["PassengerId"].astype(int),"Survived":y,"group":groups,"fold":fold_id})
        for k in preds:
            frame[k]=preds[k]; frame[k+"_score"]=scores[k]
        frame.to_csv(EXPORT_DIR/f"groupaware_oof_seed{split_seed}.csv",index=False)
        all_oof.append(frame.assign(split_seed=split_seed))

    fold_df=pd.DataFrame(rows)
    fold_df.to_csv(EXPORT_DIR/"groupaware_fold_metrics.csv",index=False)
    summary=[]
    for seed in [42,777]:
        f=all_oof[[x.split_seed.iloc[0] for x in all_oof].index(seed)]
        for k in ["v5_robust","deotte","typed_cat","typed_rf6"]:
            summary.append({"split_seed":seed,"candidate":k,"accuracy":accuracy_score(y,f[k]),"roc_auc":roc_auc_score(y,f[k+"_score"])})
    s=pd.DataFrame(summary)
    overall=s.groupby("candidate",as_index=False).agg(mean_accuracy=("accuracy","mean"),min_accuracy=("accuracy","min"),std_accuracy=("accuracy","std"),mean_auc=("roc_auc","mean")).sort_values("mean_accuracy",ascending=False)
    s.to_csv(EXPORT_DIR/"groupaware_seed_summary.csv",index=False)
    overall.to_csv(EXPORT_DIR/"groupaware_summary.csv",index=False)
    print("\n=== group-aware summary ===")
    print(overall.to_string(index=False,float_format=lambda x:f"{x:.5f}"))
    print("\nNo Kaggle submission was performed.")


if __name__ == "__main__":
    main()
