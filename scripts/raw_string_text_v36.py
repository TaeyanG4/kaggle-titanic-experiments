"""v36: raw Name/Ticket/Cabin sparse-text probe.

Residual forensics (v35) found stable hard cases concentrated in slices such as
first-class men and several raw ticket-prefix groups. Existing pipelines reduce
Name/Ticket/Cabin to hand-engineered summaries. This probe keeps the raw strings
and learns character n-grams with a sparse logistic model.

Two deliberately small representations are tested:
  * text_only: Name + Ticket + Cabin raw strings
  * text_context: same strings plus literal Sex/Pclass/Embarked context tokens

Validation surfaces:
  * six paired repeated Stratified 5-fold seeds;
  * the existing two Ticket+Family group-aware splits;
  * the five predeclared matched pseudo-test splits;
  * stable-hard-case rescue from v35.

Vectorizers are fit inside each training fold. No target-derived text features,
no test labels, and no Kaggle submission.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from groupaware_validation_v23 import connected_groups
from pseudo_test_relational_v14 import build_pseudo_splits


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "exports" / "v36"

SEEDS = [42, 123, 777, 2026, 31415, 27182]
VARIANTS = ["text_only", "text_context"]


def clean_piece(x) -> str:
    if pd.isna(x):
        return "<MISSING>"
    return str(x).strip()


def row_text(df: pd.DataFrame, variant: str) -> list[str]:
    out = []
    for _, r in df.iterrows():
        base = (
            f"NAME={clean_piece(r['Name'])} "
            f"TICKET={clean_piece(r['Ticket'])} "
            f"CABIN={clean_piece(r['Cabin'])}"
        )
        if variant == "text_context":
            base += (
                f" SEX={clean_piece(r['Sex'])}"
                f" PCLASS={clean_piece(r['Pclass'])}"
                f" EMBARKED={clean_piece(r['Embarked'])}"
            )
        out.append(base)
    return out


def fit_predict(train_df: pd.DataFrame, tr_idx: np.ndarray, va_idx: np.ndarray, variant: str, seed: int):
    y = train_df["Survived"].astype(int).to_numpy()
    tr_text = row_text(train_df.iloc[tr_idx], variant)
    va_text = row_text(train_df.iloc[va_idx], variant)
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        max_features=12000,
        sublinear_tf=True,
        lowercase=True,
        norm="l2",
    )
    xtr = vec.fit_transform(tr_text)
    xva = vec.transform(va_text)
    model = LogisticRegression(
        C=1.5,
        max_iter=3000,
        solver="liblinear",
        random_state=seed,
    )
    model.fit(xtr, y[tr_idx])
    prob = model.predict_proba(xva)[:, 1]
    return (prob >= 0.5).astype(int), prob, len(vec.vocabulary_)


def load_v5(seed: int) -> pd.DataFrame:
    if seed == 42:
        p = pd.read_csv(BASE_DIR / "exports" / "v33" / "predictions.csv")
        p = p[p["surface"] == "repeated_42"][["PassengerId", "Survived", "v5"]].copy()
        return p.sort_values("PassengerId").reset_index(drop=True)
    if seed in [123, 777, 2026]:
        p = pd.read_csv(BASE_DIR / "exports" / "v31" / f"v5_oof_seed{seed}.csv")
        return p[["PassengerId", "Survived", "v5_pred"]].rename(columns={"v5_pred": "v5"}).sort_values("PassengerId").reset_index(drop=True)
    if seed == 31415:
        p = pd.read_csv(BASE_DIR / "exports" / "v29" / "nested_oof.csv")
        return p[["PassengerId", "Survived", "v5_pred"]].rename(columns={"v5_pred": "v5"}).sort_values("PassengerId").reset_index(drop=True)
    if seed == 27182:
        p = pd.read_csv(BASE_DIR / "exports" / "v29_seed27182" / "nested_oof.csv")
        return p[["PassengerId", "Survived", "v5_pred"]].rename(columns={"v5_pred": "v5"}).sort_values("PassengerId").reset_index(drop=True)
    raise KeyError(seed)


def paired_stats(y, new, base):
    rescue = int(np.sum((new == y) & (base != y)))
    harm = int(np.sum((new != y) & (base == y)))
    changed = rescue + harm
    return rescue, harm, rescue - harm, changed, (rescue / changed if changed else np.nan)


def repeated_audit(train: pd.DataFrame):
    y = train["Survived"].astype(int).to_numpy()
    rows = []; details = []
    for seed in SEEDS:
        v5df = load_v5(seed)
        if not np.array_equal(v5df["PassengerId"].astype(int).to_numpy(), train["PassengerId"].astype(int).to_numpy()):
            raise ValueError(f"v5 order mismatch seed {seed}")
        v5 = v5df["v5"].astype(int).to_numpy()
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        fold_id = np.full(len(train), -1, dtype=int)
        for variant in VARIANTS:
            pred = np.zeros(len(train), dtype=int); prob = np.zeros(len(train), dtype=float); vocab=[]
            for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
                fold_id[va_idx] = fold
                p, s, n_vocab = fit_predict(train, tr_idx, va_idx, variant, seed * 10 + fold)
                pred[va_idx] = p; prob[va_idx] = s; vocab.append(n_vocab)
            r,h,n,d,prec = paired_stats(y,pred,v5)
            rows.append({
                "surface": f"repeated_{seed}", "variant": variant,
                "accuracy": accuracy_score(y,pred), "roc_auc": roc_auc_score(y,prob),
                "v5_accuracy": accuracy_score(y,v5), "delta_vs_v5": accuracy_score(y,pred)-accuracy_score(y,v5),
                "rescue":r,"harm":h,"net":n,"changed_vs_v5":d,"switch_precision":prec,
                "mean_vocab_size":float(np.mean(vocab)),
            })
            details.append(pd.DataFrame({
                "PassengerId":train["PassengerId"].astype(int),"Survived":y,"surface":f"repeated_{seed}",
                "variant":variant,"fold":fold_id,"text_pred":pred,"text_prob":prob,"v5":v5,
            }))
            print(f"repeated {seed} {variant}: acc={accuracy_score(y,pred):.5f} v5={accuracy_score(y,v5):.5f} net={n:+d}",flush=True)
    return pd.DataFrame(rows), pd.concat(details, ignore_index=True)


def group_audit(train: pd.DataFrame):
    y=train["Survived"].astype(int).to_numpy(); groups=connected_groups(train); rows=[]; details=[]
    for seed in [42,777]:
        old=pd.read_csv(BASE_DIR/"exports"/"v23"/f"groupaware_oof_seed{seed}.csv").sort_values("PassengerId").reset_index(drop=True)
        v5=old["v5_robust"].astype(int).to_numpy()
        sgkf=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
        for variant in VARIANTS:
            pred=np.zeros(len(train),dtype=int); prob=np.zeros(len(train),dtype=float); fold_id=np.full(len(train),-1,dtype=int)
            for fold,(tr_idx,va_idx) in enumerate(sgkf.split(train,y,groups)):
                fold_id[va_idx]=fold; p,s,_=fit_predict(train,tr_idx,va_idx,variant,36000+seed+fold); pred[va_idx]=p; prob[va_idx]=s
            r,h,n,d,prec=paired_stats(y,pred,v5)
            rows.append({"surface":f"group_{seed}","variant":variant,"accuracy":accuracy_score(y,pred),"roc_auc":roc_auc_score(y,prob),"v5_accuracy":accuracy_score(y,v5),"delta_vs_v5":accuracy_score(y,pred)-accuracy_score(y,v5),"rescue":r,"harm":h,"net":n,"changed_vs_v5":d,"switch_precision":prec})
            details.append(pd.DataFrame({"PassengerId":train.PassengerId.astype(int),"Survived":y,"surface":f"group_{seed}","variant":variant,"fold":fold_id,"text_pred":pred,"text_prob":prob,"v5":v5}))
            print(f"group {seed} {variant}: acc={accuracy_score(y,pred):.5f} v5={accuracy_score(y,v5):.5f} net={n:+d}",flush=True)
    return pd.DataFrame(rows),pd.concat(details,ignore_index=True)


def pseudo_audit(train: pd.DataFrame, test: pd.DataFrame):
    y=train["Survived"].astype(int).to_numpy(); saved=pd.read_csv(BASE_DIR/"exports"/"v19"/"pseudotest_consensus_predictions.csv"); _,splits=build_pseudo_splits(train,test); rows=[]; details=[]
    for split,(_,va_idx,_) in enumerate(splits):
        tr_idx=np.setdiff1d(np.arange(len(train)),va_idx); ref=saved[saved["split"]==split].reset_index(drop=True); ids=train.iloc[va_idx].PassengerId.astype(int).to_numpy(); v5=ref.v5.astype(int).to_numpy(); yy=y[va_idx]
        for variant in VARIANTS:
            p,s,_=fit_predict(train,tr_idx,va_idx,variant,37000+split); r,h,n,d,prec=paired_stats(yy,p,v5)
            rows.append({"surface":f"pseudo_{split}","variant":variant,"accuracy":accuracy_score(yy,p),"roc_auc":roc_auc_score(yy,s),"v5_accuracy":accuracy_score(yy,v5),"delta_vs_v5":accuracy_score(yy,p)-accuracy_score(yy,v5),"rescue":r,"harm":h,"net":n,"changed_vs_v5":d,"switch_precision":prec})
            details.append(pd.DataFrame({"PassengerId":ids,"Survived":yy,"surface":f"pseudo_{split}","variant":variant,"text_pred":p,"text_prob":s,"v5":v5}))
            print(f"pseudo {split} {variant}: acc={accuracy_score(yy,p):.5f} v5={accuracy_score(yy,v5):.5f} net={n:+d}",flush=True)
    return pd.DataFrame(rows),pd.concat(details,ignore_index=True)


def hard_case_audit(repeated_detail: pd.DataFrame):
    hard=pd.read_csv(BASE_DIR/"exports"/"v35"/"passenger_hardness.csv")[["PassengerId","consensus_stable_hard"]]
    z=repeated_detail.merge(hard,on="PassengerId",how="left"); z=z[z.consensus_stable_hard==1].copy(); rows=[]
    for (surface,variant),g in z.groupby(["surface","variant"]):
        y=g.Survived.astype(int).to_numpy(); p=g.text_pred.astype(int).to_numpy(); v=g.v5.astype(int).to_numpy(); r,h,n,d,prec=paired_stats(y,p,v)
        rows.append({"surface":surface,"variant":variant,"n_hard":len(g),"text_accuracy_hard":accuracy_score(y,p),"v5_accuracy_hard":accuracy_score(y,v),"rescue":r,"harm":h,"net":n,"switch_precision":prec})
    return pd.DataFrame(rows)


def main():
    EXPORT_DIR.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(DATA_DIR/"train.csv"); test=pd.read_csv(DATA_DIR/"test.csv")
    rr,rd=repeated_audit(train); gr,gd=group_audit(train); pr,pdeta=pseudo_audit(train,test)
    allm=pd.concat([rr,gr,pr],ignore_index=True); allm.to_csv(EXPORT_DIR/"surface_metrics.csv",index=False)
    pd.concat([rd,gd,pdeta],ignore_index=True).to_csv(EXPORT_DIR/"predictions.csv",index=False)
    hard=hard_case_audit(rd); hard.to_csv(EXPORT_DIR/"stable_hard_audit.csv",index=False)
    summary=[]
    for variant in VARIANTS:
        q=allm[allm.variant==variant]
        for family,prefix in [("repeated","repeated_"),("group","group_"),("pseudo","pseudo_")]:
            z=q[q.surface.str.startswith(prefix)]
            summary.append({"variant":variant,"surface_family":family,"mean_accuracy":float(z.accuracy.mean()),"mean_delta_vs_v5":float(z.delta_vs_v5.mean()),"min_delta_vs_v5":float(z.delta_vs_v5.min()),"positive_surfaces":int((z.delta_vs_v5>0).sum()),"nonnegative_surfaces":int((z.delta_vs_v5>=0).sum()),"mean_switch_precision":float(z.switch_precision.mean()),"total_net":int(z.net.sum())})
    sm=pd.DataFrame(summary); sm.to_csv(EXPORT_DIR/"summary.csv",index=False)
    print("\n=== v36 summary ==="); print(sm.to_string(index=False,float_format=lambda x:f"{x:.5f}"))
    print("\n=== stable hard audit ==="); print(hard.groupby("variant",as_index=False).agg(mean_text_accuracy_hard=("text_accuracy_hard","mean"),mean_v5_accuracy_hard=("v5_accuracy_hard","mean"),total_rescue=("rescue","sum"),total_harm=("harm","sum"),total_net=("net","sum")).to_string(index=False,float_format=lambda x:f"{x:.5f}"))
    print("\nNo Kaggle submission was performed.")


if __name__=="__main__": main()
