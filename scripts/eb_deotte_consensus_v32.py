"""v32: zero-training probe for majority(v5, Deotte, EB CatBoost).

Hypothesis: EB is strong on ordinary/repeated and shift-weighted validation but
weak when relational groups are completely unseen. Deotte shows the opposite
strength on group-aware validation. A 2-of-3 vote with trusted v5 may retain
only disagreements supported by two distinct inductive biases.

This script uses immutable saved OOF artifacts only. No model training and no
Kaggle submission.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

BASE_DIR=Path(__file__).resolve().parents[1]
EXPORT_DIR=BASE_DIR/"exports"/"v32"

def majority(a,b,c): return ((a.astype(int)+b.astype(int)+c.astype(int))>=2).astype(int)
def paired(y,new,base):
    rescue=int(np.sum((new==y)&(base!=y))); harm=int(np.sum((new!=y)&(base==y)))
    return rescue,harm,rescue-harm
def weighted(y,p,w): return float(np.sum(w*(p==y))/np.sum(w))

def main():
    EXPORT_DIR.mkdir(parents=True,exist_ok=True)
    rows=[]

    # Fixed historical project fold surface.
    z=pd.read_csv(BASE_DIR/"exports"/"v5"/"model_zoo_oof.csv")
    mlp=pd.read_csv(BASE_DIR/"exports"/"v5"/"seed_probes"/"MLP_PLR_seed_mean__oof.csv")
    de=pd.read_csv(BASE_DIR/"exports"/"v21"/"deotte_fixed_oof.csv")
    eb=pd.read_csv(BASE_DIR/"exports"/"v28"/"fixed_oof.csv")
    y=z.Survived.astype(int).to_numpy()
    v5=(((z["v4b__Champion"].to_numpy()>.5).astype(int)+(z.RuleFit.to_numpy()>.5).astype(int)+(mlp.probability.to_numpy()>.5).astype(int))>=2).astype(int)
    dp=de.wcg_both.astype(int).to_numpy(); ep=(eb["familyfare_eb25__CatBoost"].to_numpy()>.5).astype(int)
    m=majority(v5,dp,ep); r,h,n=paired(y,m,v5)
    rows.append({"surface":"fixed_original","v5_accuracy":accuracy_score(y,v5),"consensus_accuracy":accuracy_score(y,m),"delta":accuracy_score(y,m)-accuracy_score(y,v5),"rescue":r,"harm":h,"net":n})

    # Untouched v29 outer OOF.
    o=pd.read_csv(BASE_DIR/"exports"/"v29"/"nested_oof.csv"); y=o.Survived.astype(int).to_numpy(); v5=o.v5_robust.astype(int).to_numpy(); dp=o.deotte.astype(int).to_numpy(); ep=o.eb_cat.astype(int).to_numpy(); m=majority(v5,dp,ep); r,h,n=paired(y,m,v5)
    rows.append({"surface":"outer2901","v5_accuracy":accuracy_score(y,v5),"consensus_accuracy":accuracy_score(y,m),"delta":accuracy_score(y,m)-accuracy_score(y,v5),"rescue":r,"harm":h,"net":n})

    # Existing group-aware surfaces, pairing v23 v5/deotte with v31 EB.
    gnew=pd.read_csv(BASE_DIR/"exports"/"v31"/"groupaware_eb_oof.csv")
    for seed in [42,777]:
        old=pd.read_csv(BASE_DIR/"exports"/"v23"/f"groupaware_oof_seed{seed}.csv")
        en=gnew[gnew.seed==seed].sort_values("PassengerId")
        old=old.sort_values("PassengerId")
        y=old.Survived.astype(int).to_numpy(); v5=old.v5_robust.astype(int).to_numpy(); dp=old.deotte.astype(int).to_numpy(); ep=en.eb_cat.astype(int).to_numpy(); m=majority(v5,dp,ep); r,h,n=paired(y,m,v5)
        rows.append({"surface":f"group_seed{seed}","v5_accuracy":accuracy_score(y,v5),"consensus_accuracy":accuracy_score(y,m),"delta":accuracy_score(y,m)-accuracy_score(y,v5),"rescue":r,"harm":h,"net":n})

    # v26 cross-fitted propensity weights applied to v29 outer OOF.
    o=pd.read_csv(BASE_DIR/"exports"/"v29"/"nested_oof.csv"); y=o.Survived.astype(int).to_numpy(); v5=o.v5_robust.astype(int).to_numpy(); m=majority(v5,o.deotte.astype(int).to_numpy(),o.eb_cat.astype(int).to_numpy())
    wdf=pd.read_csv(BASE_DIR/"exports"/"v26"/"train_shift_weights.csv")
    for wm in ["logistic","lightgbm"]:
        w=wdf[wdf.weight_model==wm].sort_values("train_row").weight.to_numpy(float)
        rows.append({"surface":f"shift_{wm}","v5_accuracy":weighted(y,v5,w),"consensus_accuracy":weighted(y,m,w),"delta":weighted(y,m,w)-weighted(y,v5,w),"rescue":np.nan,"harm":np.nan,"net":np.nan})

    out=pd.DataFrame(rows); out.to_csv(EXPORT_DIR/"surface_metrics.csv",index=False)
    summary=pd.DataFrame([{
        "mean_delta_all":float(out.delta.mean()),"min_delta_all":float(out.delta.min()),"positive_surfaces":int((out.delta>0).sum()),"nonnegative_surfaces":int((out.delta>=0).sum()),"negative_surfaces":int((out.delta<0).sum()),
        "mean_group_delta":float(out[out.surface.str.startswith('group_')].delta.mean()),"mean_standard_delta":float(out[~out.surface.str.startswith('group_')].delta.mean())
    }]); summary.to_csv(EXPORT_DIR/"summary.csv",index=False)
    print(out.to_string(index=False,float_format=lambda x:f"{x:.5f}")); print(); print(summary.to_string(index=False,float_format=lambda x:f"{x:.5f}")); print("\nNo Kaggle submission was performed.")

if __name__=="__main__": main()
