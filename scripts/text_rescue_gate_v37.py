"""v37: predeclared high-confidence raw-text rescue gate.

Development evidence from v36 repeated OOF showed that raw text is weak overall
but occasionally makes very high-confidence corrections to v5. We therefore
freeze one low-degree-of-freedom rule before touching real-test predictions:

  switch v5 -> text_only iff text_only disagrees with v5 and confidence >= 0.75

The threshold is validated unchanged on group-aware and matched pseudo-test
surfaces, then a 6-seed x 5-fold bagged text probability is built for the real
test. Mandatory headroom is computed versus v10. No Kaggle submission occurs.
"""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from raw_string_text_v36 import fit_predict, row_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "exports" / "v37"
SUB_DIR = BASE_DIR / "submissions"
SEEDS = [42,123,777,2026,31415,27182]
THRESHOLD = 0.75


def gate(base: np.ndarray, text_pred: np.ndarray, text_prob: np.ndarray):
    conf = np.maximum(text_prob, 1-text_prob)
    mask = (text_pred != base) & (conf >= THRESHOLD)
    out = base.copy(); out[mask] = text_pred[mask]
    return out, mask, conf


def paired(y,new,base):
    rescue=int(np.sum((new==y)&(base!=y))); harm=int(np.sum((new!=y)&(base==y)))
    return rescue,harm,rescue-harm


def validate_saved():
    p=pd.read_csv(BASE_DIR/"exports"/"v36"/"predictions.csv")
    p=p[p.variant=="text_only"].copy(); rows=[]
    for surface,g in p.groupby("surface"):
        y=g.Survived.astype(int).to_numpy(); base=g.v5.astype(int).to_numpy(); tp=g.text_pred.astype(int).to_numpy(); prob=g.text_prob.to_numpy(float)
        out,mask,conf=gate(base,tp,prob); r,h,n=paired(y,out,base)
        family=surface.split('_')[0]
        rows.append({"surface":surface,"family":family,"v5_accuracy":float(np.mean(base==y)),"gated_accuracy":float(np.mean(out==y)),"delta":float(np.mean(out==y)-np.mean(base==y)),"switches":int(mask.sum()),"rescue":r,"harm":h,"net":n,"mean_switch_confidence":float(conf[mask].mean()) if mask.any() else np.nan})
    return pd.DataFrame(rows)


def fold_test_prob(train,test,y,tr_idx,seed):
    tr_text=row_text(train.iloc[tr_idx],"text_only"); te_text=row_text(test,"text_only")
    vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(2,5),min_df=2,max_features=12000,sublinear_tf=True,lowercase=True,norm="l2")
    xtr=vec.fit_transform(tr_text); xte=vec.transform(te_text)
    model=LogisticRegression(C=1.5,max_iter=3000,solver="liblinear",random_state=seed)
    model.fit(xtr,y[tr_idx]); return model.predict_proba(xte)[:,1]


def required_precision(d,gap):
    if d==0: return math.inf, math.inf
    w=math.floor((d+gap)/2)+1
    return w,w/d


def main():
    EXPORT_DIR.mkdir(parents=True,exist_ok=True); SUB_DIR.mkdir(parents=True,exist_ok=True)
    metrics=validate_saved(); metrics.to_csv(EXPORT_DIR/"validation_surfaces.csv",index=False)
    fam=metrics.groupby("family",as_index=False).agg(mean_delta=("delta","mean"),min_delta=("delta","min"),switches=("switches","sum"),rescue=("rescue","sum"),harm=("harm","sum"),net=("net","sum"),positive=("delta",lambda s:int((s>0).sum())),nonnegative=("delta",lambda s:int((s>=0).sum())))
    fam.to_csv(EXPORT_DIR/"validation_summary.csv",index=False)

    train=pd.read_csv(DATA_DIR/"train.csv"); test=pd.read_csv(DATA_DIR/"test.csv"); y=train.Survived.astype(int).to_numpy()
    seed_probs=[]; seed_labels=[]; seed_rows=[]
    for seed in SEEDS:
        skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed); fps=[]
        for fold,(tr_idx,_) in enumerate(skf.split(train,y)):
            print(f"seed={seed} fold={fold}",flush=True); fps.append(fold_test_prob(train,test,y,tr_idx,38000+seed+fold))
        prob=np.mean(fps,axis=0); lab=(prob>=.5).astype(int); seed_probs.append(prob); seed_labels.append(lab); seed_rows.append({"seed":seed,"mean_probability":float(prob.mean()),"positive_labels":int(lab.sum()),"high_confidence_rows":int(np.sum(np.maximum(prob,1-prob)>=THRESHOLD))})
    seed_probs=np.vstack(seed_probs); seed_labels=np.vstack(seed_labels); mean_prob=seed_probs.mean(axis=0); text_pred=(mean_prob>=.5).astype(int)

    v5=pd.read_csv(SUB_DIR/"submission_v5_score_0.79665.csv").sort_values("PassengerId").reset_index(drop=True); v10=pd.read_csv(SUB_DIR/"submission_v10_score_0.81578.csv").sort_values("PassengerId").reset_index(drop=True); ids=test.sort_values("PassengerId").PassengerId.astype(int).to_numpy(); base=v5.Survived.astype(int).to_numpy(); champion=v10.Survived.astype(int).to_numpy()
    cand,mask,conf=gate(base,text_pred,mean_prob)
    out=pd.DataFrame({"PassengerId":ids,"Survived":cand}); out.to_csv(SUB_DIR/"submission_v37_text_rescue_gate.csv",index=False)
    details=pd.DataFrame({"PassengerId":ids,"v5":base,"v10":champion,"text_prob":mean_prob,"text_pred":text_pred,"text_confidence":conf,"switch":mask.astype(int),"candidate":cand,"text_seed_positive_votes":seed_labels.sum(axis=0)})
    for i,s in enumerate(SEEDS): details[f"prob_seed_{s}"]=seed_probs[i]
    details.to_csv(EXPORT_DIR/"test_details.csv",index=False); pd.DataFrame(seed_rows).to_csv(EXPORT_DIR/"test_seed_stability.csv",index=False)

    d=int(mask.sum()); gap=8; wmin,prec=required_precision(d,gap)
    head=pd.DataFrame([{"changed_vs_v5":d,"changed_vs_v10":int(np.sum(cand!=champion)),"champion_gap_correct":gap,"wins_needed_among_changed_to_beat_v10":wmin,"required_switch_precision":prec,"mathematically_can_beat_v10":int(wmin<=d),"test_positives":int(cand.sum()),"all_switches_seed_unanimous":int(np.all((seed_labels[:,mask].sum(axis=0)==0)|(seed_labels[:,mask].sum(axis=0)==len(SEEDS)))) if d else 1}])
    head.to_csv(EXPORT_DIR/"headroom.csv",index=False)
    print("=== validation summary ==="); print(fam.to_string(index=False,float_format=lambda x:f"{x:.5f}")); print("\n=== headroom ==="); print(head.to_string(index=False,float_format=lambda x:f"{x:.5f}"));
    if d: print("\n=== switched test rows ==="); print(details[details.switch==1][["PassengerId","v5","v10","text_pred","text_confidence","text_seed_positive_votes","candidate"]].to_string(index=False))
    print("\nNo Kaggle submission was performed.")


if __name__=="__main__": main()
