"""Low-degree HPO sweep on the strongest fold-safe relational representation.

Representation: trusted v2 base + fold-safe GroupSurvival + FamilyFare peer
statistics + typed22 relational block.  Screening uses the immutable seed-42
manifest; only the top two configs per family are confirmed on two independent
StratifiedKFold seeds (123, 777).  This is intentionally small to avoid fitting
hundreds of choices to Titanic's 891 rows.

Families: XGBoost, LightGBM, RandomForest, ExtraTrees.
No Kaggle submission is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from audit_group_survival import add_group_survival
from group_key_audit_v6 import peer_feature
from gunes_exact_foldsafe_v11 import structural_frames
from pseudo_test_relational_v14 import VARIANTS as REL_VARIANTS, eligible_groups, relation_features, role_flags
from tabpfn_finalist_v8 import prepare_data


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "exports" / "v22"
SUB_DIR = BASE_DIR / "submissions"
FAMILYFARE_COLS = ["FamilyFareAny", "FamilyFareMean", "FamilyFareSmooth", "FamilyFareCount"]
TYPED_COLS = REL_VARIANTS["typed22"]


@dataclass
class FoldData:
    fold: int
    tr_idx: np.ndarray
    va_idx: np.ndarray
    xtr: np.ndarray
    ytr: np.ndarray
    xva: np.ndarray
    yva: np.ndarray


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


def build_fold_cache(train, rel_train, base_cols, folds):
    y = train["Survived"].astype(int).to_numpy()
    features = list(dict.fromkeys(base_cols + FAMILYFARE_COLS + TYPED_COLS))
    cache = []
    for fold in sorted(np.unique(folds)):
        tr_idx = np.flatnonzero(folds != fold)
        va_idx = np.flatnonzero(folds == fold)
        tr = train.iloc[tr_idx].copy()
        va = train.iloc[va_idx].copy()
        rr = rel_train.iloc[tr_idx].copy()
        rv = rel_train.iloc[va_idx].copy()

        tr["GroupSurvival"] = add_group_survival(tr, tr, exclude_self=True).to_numpy()
        va["GroupSurvival"] = add_group_survival(tr, va, exclude_self=False).to_numpy()
        tr = add_familyfare(tr, tr, True)
        va = add_familyfare(train.iloc[tr_idx].assign(GroupSurvival=tr["GroupSurvival"].to_numpy()), va, False)

        af, at = eligible_groups(rr, rv)
        tr_rel = relation_features(rr, rr, exclude_self=True, alpha=2.0, allowed_fam=af, allowed_tic=at)
        va_rel = relation_features(rr, rv, exclude_self=False, alpha=2.0, allowed_fam=af, allowed_tic=at)
        tr = add_rel(tr, tr_rel)
        va = add_rel(va, va_rel)
        cache.append(FoldData(int(fold), tr_idx, va_idx, tr[features].to_numpy(float), y[tr_idx], va[features].to_numpy(float), y[va_idx]))
    return cache, features


def configs():
    return {
        "XGBoost": [
            dict(n_estimators=160,max_depth=2,learning_rate=.04,min_child_weight=1,subsample=.90,colsample_bytree=.90,reg_lambda=3,reg_alpha=0,gamma=0),
            dict(n_estimators=240,max_depth=2,learning_rate=.03,min_child_weight=2,subsample=.85,colsample_bytree=.90,reg_lambda=5,reg_alpha=.05,gamma=0),
            dict(n_estimators=320,max_depth=2,learning_rate=.02,min_child_weight=3,subsample=.90,colsample_bytree=.80,reg_lambda=8,reg_alpha=.10,gamma=.02),
            dict(n_estimators=180,max_depth=3,learning_rate=.04,min_child_weight=2,subsample=.85,colsample_bytree=.85,reg_lambda=3,reg_alpha=.05,gamma=.02),
            dict(n_estimators=260,max_depth=3,learning_rate=.025,min_child_weight=3,subsample=.90,colsample_bytree=.90,reg_lambda=6,reg_alpha=.10,gamma=.05),
            dict(n_estimators=360,max_depth=3,learning_rate=.02,min_child_weight=4,subsample=.80,colsample_bytree=.85,reg_lambda=8,reg_alpha=.20,gamma=.05),
            dict(n_estimators=160,max_depth=4,learning_rate=.035,min_child_weight=3,subsample=.80,colsample_bytree=.80,reg_lambda=6,reg_alpha=.10,gamma=.05),
            dict(n_estimators=240,max_depth=4,learning_rate=.025,min_child_weight=4,subsample=.85,colsample_bytree=.85,reg_lambda=10,reg_alpha=.20,gamma=.10),
            dict(n_estimators=140,max_depth=1,learning_rate=.05,min_child_weight=1,subsample=1.0,colsample_bytree=1.0,reg_lambda=2,reg_alpha=0,gamma=0),
            dict(n_estimators=300,max_depth=1,learning_rate=.025,min_child_weight=1,subsample=.90,colsample_bytree=.90,reg_lambda=4,reg_alpha=.05,gamma=0),
            dict(n_estimators=220,max_depth=3,learning_rate=.03,min_child_weight=6,subsample=1.0,colsample_bytree=.75,reg_lambda=12,reg_alpha=.25,gamma=.10),
            dict(n_estimators=420,max_depth=2,learning_rate=.015,min_child_weight=4,subsample=.75,colsample_bytree=1.0,reg_lambda=10,reg_alpha=.15,gamma=.05),
        ],
        "LightGBM": [
            dict(n_estimators=180,num_leaves=5,max_depth=3,learning_rate=.04,min_child_samples=20,subsample=.9,colsample_bytree=.9,reg_lambda=3,reg_alpha=0),
            dict(n_estimators=260,num_leaves=5,max_depth=3,learning_rate=.025,min_child_samples=25,subsample=.85,colsample_bytree=.9,reg_lambda=5,reg_alpha=.05),
            dict(n_estimators=320,num_leaves=7,max_depth=3,learning_rate=.02,min_child_samples=30,subsample=.9,colsample_bytree=.85,reg_lambda=8,reg_alpha=.10),
            dict(n_estimators=180,num_leaves=7,max_depth=3,learning_rate=.04,min_child_samples=15,subsample=.85,colsample_bytree=.85,reg_lambda=3,reg_alpha=.05),
            dict(n_estimators=260,num_leaves=9,max_depth=4,learning_rate=.025,min_child_samples=20,subsample=.9,colsample_bytree=.9,reg_lambda=6,reg_alpha=.10),
            dict(n_estimators=340,num_leaves=9,max_depth=4,learning_rate=.018,min_child_samples=30,subsample=.8,colsample_bytree=.85,reg_lambda=10,reg_alpha=.20),
            dict(n_estimators=180,num_leaves=12,max_depth=4,learning_rate=.035,min_child_samples=25,subsample=.8,colsample_bytree=.8,reg_lambda=8,reg_alpha=.15),
            dict(n_estimators=280,num_leaves=12,max_depth=4,learning_rate=.022,min_child_samples=35,subsample=.9,colsample_bytree=.8,reg_lambda=12,reg_alpha=.25),
            dict(n_estimators=200,num_leaves=3,max_depth=2,learning_rate=.04,min_child_samples=20,subsample=1.0,colsample_bytree=1.0,reg_lambda=3,reg_alpha=0),
            dict(n_estimators=360,num_leaves=3,max_depth=2,learning_rate=.02,min_child_samples=25,subsample=.9,colsample_bytree=.9,reg_lambda=6,reg_alpha=.05),
        ],
        "RandomForest": [
            dict(n_estimators=500,max_depth=4,min_samples_split=4,min_samples_leaf=2,max_features="sqrt"),
            dict(n_estimators=500,max_depth=5,min_samples_split=4,min_samples_leaf=2,max_features="sqrt"),
            dict(n_estimators=500,max_depth=6,min_samples_split=6,min_samples_leaf=2,max_features="sqrt"),
            dict(n_estimators=500,max_depth=7,min_samples_split=6,min_samples_leaf=3,max_features="sqrt"),
            dict(n_estimators=500,max_depth=5,min_samples_split=6,min_samples_leaf=3,max_features=.7),
            dict(n_estimators=500,max_depth=6,min_samples_split=8,min_samples_leaf=3,max_features=.7),
            dict(n_estimators=500,max_depth=7,min_samples_split=8,min_samples_leaf=4,max_features=.7),
            dict(n_estimators=500,max_depth=8,min_samples_split=10,min_samples_leaf=4,max_features=1.0),
        ],
        "ExtraTrees": [
            dict(n_estimators=500,max_depth=4,min_samples_split=4,min_samples_leaf=2,max_features="sqrt"),
            dict(n_estimators=500,max_depth=5,min_samples_split=4,min_samples_leaf=2,max_features="sqrt"),
            dict(n_estimators=500,max_depth=6,min_samples_split=6,min_samples_leaf=2,max_features="sqrt"),
            dict(n_estimators=500,max_depth=7,min_samples_split=6,min_samples_leaf=3,max_features="sqrt"),
            dict(n_estimators=500,max_depth=5,min_samples_split=6,min_samples_leaf=3,max_features=.7),
            dict(n_estimators=500,max_depth=6,min_samples_split=8,min_samples_leaf=3,max_features=.7),
            dict(n_estimators=500,max_depth=7,min_samples_split=8,min_samples_leaf=4,max_features=.7),
            dict(n_estimators=500,max_depth=8,min_samples_split=10,min_samples_leaf=4,max_features=1.0),
        ],
    }


def make_model(family, cfg, seed):
    if family == "XGBoost":
        return XGBClassifier(**cfg, objective="binary:logistic", eval_metric="logloss", random_state=seed, n_jobs=-1, tree_method="hist", verbosity=0)
    if family == "LightGBM":
        return LGBMClassifier(**cfg, random_state=seed, n_jobs=-1, verbosity=-1)
    if family == "RandomForest":
        return RandomForestClassifier(**cfg, random_state=seed, n_jobs=-1, class_weight=None)
    if family == "ExtraTrees":
        return ExtraTreesClassifier(**cfg, random_state=seed, n_jobs=-1, class_weight=None)
    raise KeyError(family)


def eval_config(family, cfg, cache, n_rows, seed_offset=0):
    oof = np.zeros(n_rows, dtype=float)
    fold_acc = []
    for fd in cache:
        m = make_model(family, cfg, seed_offset + 42 + fd.fold)
        m.fit(fd.xtr, fd.ytr)
        p = m.predict_proba(fd.xva)[:, 1]
        oof[fd.va_idx] = p
        fold_acc.append(accuracy_score(fd.yva, p > .5))
    y = np.zeros(n_rows, dtype=int)
    for fd in cache:
        y[fd.va_idx] = fd.yva
    return {
        "accuracy": accuracy_score(y, oof > .5),
        "roc_auc": roc_auc_score(y, oof),
        "fold_std": float(np.std(fold_acc)),
        "oof": oof,
    }


def main():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    train_raw = pd.read_csv(DATA_DIR / "train.csv")
    test_raw = pd.read_csv(DATA_DIR / "test.csv")
    train, _, fixed_folds, base_cols = prepare_data()
    rel_train, _, _, _, _ = structural_frames(train_raw, test_raw)
    rel_train["IsWomanChild"] = role_flags(train_raw)
    y = train["Survived"].astype(int).to_numpy()

    fixed_cache, features = build_fold_cache(train, rel_train, base_cols, fixed_folds)
    screen_rows = []
    screen_oof = {}
    cfgs = configs()
    for family, family_cfgs in cfgs.items():
        print(f"\n=== {family} screening ({len(family_cfgs)}) ===", flush=True)
        for i, cfg in enumerate(family_cfgs):
            r = eval_config(family, cfg, fixed_cache, len(train), seed_offset=i*100)
            screen_rows.append({"family":family,"config_id":i,"accuracy":r["accuracy"],"roc_auc":r["roc_auc"],"fold_std":r["fold_std"],"config":json.dumps(cfg,sort_keys=True)})
            screen_oof[f"{family}_{i}"] = r["oof"]
            print(f"{i:02d} acc={r['accuracy']:.5f} auc={r['roc_auc']:.5f} std={r['fold_std']:.5f}", flush=True)

    screen = pd.DataFrame(screen_rows).sort_values(["accuracy","roc_auc"], ascending=False)
    screen.to_csv(EXPORT_DIR / "hpo_screen.csv", index=False)
    pd.DataFrame({"PassengerId":train["PassengerId"].astype(int),"Survived":y,**screen_oof}).to_csv(EXPORT_DIR / "hpo_screen_oof.csv",index=False)

    finalists = []
    for family in cfgs:
        top = screen[screen["family"]==family].head(2)
        for _, row in top.iterrows():
            finalists.append((family,int(row["config_id"]),cfgs[family][int(row["config_id"])]))

    alt_rows = []
    for split_seed in [123,777]:
        folds = np.full(len(train), -1, dtype=int)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=split_seed)
        for f, (_, va) in enumerate(skf.split(train, y)):
            folds[va] = f
        cache, _ = build_fold_cache(train, rel_train, base_cols, folds)
        print(f"\n=== alternate seed {split_seed} ===", flush=True)
        for family,cid,cfg in finalists:
            r = eval_config(family,cfg,cache,len(train),seed_offset=split_seed*10+cid*100)
            alt_rows.append({"split_seed":split_seed,"family":family,"config_id":cid,"accuracy":r["accuracy"],"roc_auc":r["roc_auc"],"fold_std":r["fold_std"]})
            print(f"{family}_{cid}: acc={r['accuracy']:.5f} auc={r['roc_auc']:.5f}", flush=True)
    alt = pd.DataFrame(alt_rows)
    alt.to_csv(EXPORT_DIR / "hpo_alternate_confirmation.csv", index=False)

    merged = []
    for family,cid,cfg in finalists:
        frow = screen[(screen.family==family)&(screen.config_id==cid)].iloc[0]
        a = alt[(alt.family==family)&(alt.config_id==cid)]
        merged.append({
            "family":family,"config_id":cid,
            "fixed_accuracy":frow.accuracy,"fixed_auc":frow.roc_auc,
            "alt_mean_accuracy":a.accuracy.mean(),"alt_min_accuracy":a.accuracy.min(),"alt_mean_auc":a.roc_auc.mean(),
            "mean_all3_accuracy":np.mean([frow.accuracy,*a.accuracy.tolist()]),
            "config":json.dumps(cfg,sort_keys=True),
        })
    conf = pd.DataFrame(merged).sort_values(["mean_all3_accuracy","fixed_accuracy","alt_mean_auc"], ascending=False)
    conf.to_csv(EXPORT_DIR / "hpo_confirmed_ranking.csv", index=False)
    print("\n=== confirmed ranking ===")
    print(conf.to_string(index=False,float_format=lambda x:f"{x:.5f}"))
    print("\nNo Kaggle submission was performed.")


if __name__ == "__main__":
    main()
