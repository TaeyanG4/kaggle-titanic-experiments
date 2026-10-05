"""v39: identify which raw-string field drives the v38 text rescue signal.

We keep the frozen confidence gate (>= 0.75) and compare five sparse char-ngram
representations: Name only, Ticket only, Cabin only, Name+Ticket, and the full
Name+Ticket+Cabin view. v10-analogue base predictions are reused from v38 so the
only changing factor is the raw string source.

The real-test analysis asks whether PassengerId 980 is supported by multiple
independent raw fields or only one fragile lexical pattern. No submission.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from groupaware_validation_v23 import connected_groups
from pseudo_test_relational_v14 import build_pseudo_splits


BASE_DIR=Path(__file__).resolve().parents[1]
DATA_DIR=BASE_DIR/"data"
EXPORT_DIR=BASE_DIR/"exports"/"v39"
V38=BASE_DIR/"exports"/"v38"/"predictions.csv"
THRESHOLD=0.75
SEEDS=[42,123,777,2026,31415,27182]
FIELDS={
    "name_only":["Name"],
    "ticket_only":["Ticket"],
    "cabin_only":["Cabin"],
    "name_ticket":["Name","Ticket"],
    "full_text":["Name","Ticket","Cabin"],
}


def clean(x): return "<MISSING>" if pd.isna(x) else str(x).strip()
def texts(df,fields):
    return [" ".join(f"{c.upper()}={clean(r[c])}" for c in fields) for _,r in df.iterrows()]


def fit_predict(train,tr_idx,va_idx,fields,seed):
    y=train.Survived.astype(int).to_numpy(); vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(2,5),min_df=2,max_features=12000,sublinear_tf=True,lowercase=True,norm="l2")
    xtr=vec.fit_transform(texts(train.iloc[tr_idx],fields)); xva=vec.transform(texts(train.iloc[va_idx],fields)); m=LogisticRegression(C=1.5,max_iter=3000,solver="liblinear",random_state=seed); m.fit(xtr,y[tr_idx]); p=m.predict_proba(xva)[:,1]; return (p>=.5).astype(int),p


def fit_test(train,test,y,tr_idx,fields,seed):
    vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(2,5),min_df=2,max_features=12000,sublinear_tf=True,lowercase=True,norm="l2")
    xtr=vec.fit_transform(texts(train.iloc[tr_idx],fields)); xte=vec.transform(texts(test,fields)); m=LogisticRegression(C=1.5,max_iter=3000,solver="liblinear",random_state=seed); m.fit(xtr,y[tr_idx]); return m.predict_proba(xte)[:,1]


def gate(base,pred,prob):
    conf=np.maximum(prob,1-prob); mask=(pred!=base)&(conf>=THRESHOLD); out=base.copy(); out[mask]=pred[mask]; return out,mask,conf


def paired(y,new,base):
    r=int(np.sum((new==y)&(base!=y))); h=int(np.sum((new!=y)&(base==y))); return r,h,r-h


def base_surface(surface,ids):
    p=pd.read_csv(V38); q=p[p.surface==surface].set_index("PassengerId").loc[ids].reset_index(); return q.base.astype(int).to_numpy()


def main():
    EXPORT_DIR.mkdir(parents=True,exist_ok=True); train=pd.read_csv(DATA_DIR/"train.csv"); test=pd.read_csv(DATA_DIR/"test.csv"); y=train.Survived.astype(int).to_numpy(); groups=connected_groups(train); rows=[]

    # Repeated surfaces.
    for seed in SEEDS:
        skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
        for name,fields in FIELDS.items():
            pred=np.zeros(len(train),int); prob=np.zeros(len(train),float)
            for fold,(tr,va) in enumerate(skf.split(train,y)):
                pp,ss=fit_predict(train,tr,va,fields,39000+seed+fold); pred[va]=pp; prob[va]=ss
            ids=train.PassengerId.astype(int).to_numpy(); base=base_surface(f"repeated_{seed}",ids); out,mask,conf=gate(base,pred,prob); r,h,n=paired(y,out,base)
            rows.append({"surface":f"repeated_{seed}","family":"repeated","variant":name,"switches":int(mask.sum()),"rescue":r,"harm":h,"net":n,"precision":r/(r+h) if r+h else np.nan})

    # Group-aware surfaces.
    for seed in [42,777]:
        sg=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed)
        for name,fields in FIELDS.items():
            pred=np.zeros(len(train),int); prob=np.zeros(len(train),float)
            for fold,(tr,va) in enumerate(sg.split(train,y,groups)):
                pp,ss=fit_predict(train,tr,va,fields,40000+seed+fold); pred[va]=pp; prob[va]=ss
            ids=train.PassengerId.astype(int).to_numpy(); base=base_surface(f"group_{seed}",ids); out,mask,conf=gate(base,pred,prob); r,h,n=paired(y,out,base)
            rows.append({"surface":f"group_{seed}","family":"group","variant":name,"switches":int(mask.sum()),"rescue":r,"harm":h,"net":n,"precision":r/(r+h) if r+h else np.nan})

    # Pseudo surfaces.
    _,splits=build_pseudo_splits(train,test)
    for split,(_,va,_) in enumerate(splits):
        tr=np.setdiff1d(np.arange(len(train)),va); ids=train.iloc[va].PassengerId.astype(int).to_numpy(); yy=y[va]; base=base_surface(f"pseudo_{split}",ids)
        for name,fields in FIELDS.items():
            pred,prob=fit_predict(train,tr,va,fields,41000+split); out,mask,conf=gate(base,pred,prob); r,h,n=paired(yy,out,base)
            rows.append({"surface":f"pseudo_{split}","family":"pseudo","variant":name,"switches":int(mask.sum()),"rescue":r,"harm":h,"net":n,"precision":r/(r+h) if r+h else np.nan})

    metrics=pd.DataFrame(rows); metrics.to_csv(EXPORT_DIR/"surface_metrics.csv",index=False)
    summary=metrics.groupby(["variant","family"],as_index=False).agg(switches=("switches","sum"),rescue=("rescue","sum"),harm=("harm","sum"),net=("net","sum"),positive_surfaces=("net",lambda s:int((s>0).sum())),negative_surfaces=("net",lambda s:int((s<0).sum())))
    summary["switch_precision"]=summary.rescue/(summary.rescue+summary.harm).replace(0,np.nan); summary.to_csv(EXPORT_DIR/"summary.csv",index=False)

    # Real-test bagged field predictions.
    v10=pd.read_csv(BASE_DIR/"submissions"/"submission_v10_score_0.81578.csv").sort_values("PassengerId").reset_index(drop=True); base=v10.Survived.astype(int).to_numpy(); test_ids=test.sort_values("PassengerId").PassengerId.astype(int).to_numpy(); test_rows=[]
    detail=pd.DataFrame({"PassengerId":test_ids,"v10":base})
    for name,fields in FIELDS.items():
        seed_probs=[]
        for seed in SEEDS:
            skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed); fps=[]
            for fold,(tr,_) in enumerate(skf.split(train,y)): fps.append(fit_test(train,test,y,tr,fields,42000+seed+fold))
            seed_probs.append(np.mean(fps,axis=0))
        prob=np.mean(np.vstack(seed_probs),axis=0); pred=(prob>=.5).astype(int); out,mask,conf=gate(base,pred,prob)
        detail[f"{name}_prob"]=prob; detail[f"{name}_pred"]=pred; detail[f"{name}_conf"]=conf; detail[f"{name}_switch"]=mask.astype(int)
        test_rows.append({"variant":name,"switches_vs_v10":int(mask.sum()),"switched_ids":",".join(map(str,test_ids[mask].tolist())),"test_positives_after_gate":int(out.sum())})
    detail.to_csv(EXPORT_DIR/"test_field_predictions.csv",index=False); pd.DataFrame(test_rows).to_csv(EXPORT_DIR/"test_summary.csv",index=False)
    p980=detail[detail.PassengerId==980].copy(); p980.to_csv(EXPORT_DIR/"passenger980_fields.csv",index=False)

    print("=== field ablation validation ==="); print(summary.to_string(index=False,float_format=lambda x:f"{x:.4f}")); print("\n=== test switches ==="); print(pd.DataFrame(test_rows).to_string(index=False)); print("\n=== Passenger 980 ==="); print(p980.to_string(index=False)); print("\nNo Kaggle submission was performed.")


if __name__=="__main__": main()
