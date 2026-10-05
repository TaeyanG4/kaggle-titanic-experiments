"""v40: X-only passenger graph centrality probe.

v17 already tested connected-component counts and target-smoothed component
survival. It did NOT test structural centrality. This experiment adds a purely
X-derived projected passenger graph built from shared Ticket, meaningful Family
(surname + family size), and Cabin tokens.

Features include degree, weighted degree, PageRank, k-core, articulation status,
approximate betweenness, triangles, average-neighbor degree, component size and
component density. The graph is transductive X-only: fold-validation covariates
are visible when constructing graph structure, just as real test covariates are
visible at inference; no validation/test labels enter graph construction.

Marginal value is tested by adding this block to the strong fixed
FamilyFare+typed22+EB CatBoost representation on fixed, matched-pseudo, and
group-aware surfaces. No submission.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import re

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from feature_ablation_v6 import build_models as build_tree_models
from groupaware_validation_v23 import connected_groups
from partial_pooling_transfer_v28 import fold_features, prepare_rel
from pseudo_test_relational_v14 import build_pseudo_splits
from tabpfn_finalist_v8 import prepare_data


BASE_DIR=Path(__file__).resolve().parents[1]
DATA_DIR=BASE_DIR/"data"
EXPORT_DIR=BASE_DIR/"exports"/"v40"
VARIANT="familyfare_typed22_eb25"

GRAPH_COLS=[
    "GraphDegree","GraphWeightedDegree","GraphPageRank","GraphCore","GraphArticulation",
    "GraphBetweenness","GraphTriangles","GraphAvgNeighborDegree","GraphComponentSize",
    "GraphComponentDensity","GraphRelationTypes","GraphBridgeIncidents",
]


def surname(name): return re.sub(r"[^A-Za-z ]","",str(name).split(",")[0]).strip().upper()


def relation_groups(df: pd.DataFrame):
    n=len(df); groups=[]
    # Ticket groups.
    for _,idx in df.groupby(df.Ticket.astype(str),sort=False).groups.items():
        ids=list(idx)
        if len(ids)>1: groups.append(("ticket",ids))
    # Family key only when family size > 1.
    fs=(df.SibSp+df.Parch+1).astype(int); fk=df.Name.map(surname)+"_"+fs.astype(str)
    tmp=pd.DataFrame({"key":fk,"fs":fs},index=df.index)
    for key,idx in tmp.groupby("key",sort=False).groups.items():
        ids=list(idx)
        if len(ids)>1 and int(tmp.loc[ids,"fs"].median())>1: groups.append(("family",ids))
    # Exact cabin tokens. Multi-cabin strings contribute each token.
    cabin_map={}
    for i,v in df.Cabin.items():
        if pd.isna(v): continue
        for token in str(v).split(): cabin_map.setdefault(token,[]).append(i)
    for _,ids in cabin_map.items():
        if len(ids)>1: groups.append(("cabin",ids))
    return groups


def graph_block(ref_raw: pd.DataFrame, inf_raw: pd.DataFrame):
    combo=pd.concat([ref_raw.assign(_side="ref"),inf_raw.assign(_side="inf")],ignore_index=True,sort=False)
    G=nx.Graph(); G.add_nodes_from(range(len(combo))); rel_types=[set() for _ in range(len(combo))]
    for typ,ids in relation_groups(combo):
        for i in ids: rel_types[i].add(typ)
        for a,b in combinations(ids,2):
            if G.has_edge(a,b): G[a][b]["weight"]+=1.0
            else: G.add_edge(a,b,weight=1.0)
    deg=dict(G.degree()); wdeg=dict(G.degree(weight="weight")); pr=nx.pagerank(G,alpha=.85,weight="weight") if G.number_of_edges() else {i:1/len(G) for i in G.nodes}
    core=nx.core_number(G) if G.number_of_edges() else {i:0 for i in G.nodes}; art=set(nx.articulation_points(G)) if G.number_of_edges() else set(); tri=nx.triangles(G); avg=nx.average_neighbor_degree(G,weight="weight") if G.number_of_edges() else {i:0.0 for i in G.nodes}
    # Approximate betweenness is enough for a cheap falsification probe.
    if G.number_of_edges():
        k=min(80,len(G)); btw=nx.betweenness_centrality(G,k=k,normalized=True,weight=None,seed=40)
        bridge_inc={i:0 for i in G.nodes}
        for a,b in nx.bridges(G): bridge_inc[a]+=1; bridge_inc[b]+=1
    else:
        btw={i:0.0 for i in G.nodes}; bridge_inc={i:0 for i in G.nodes}
    comp_size={}; comp_density={}
    for comp in nx.connected_components(G):
        nodes=list(comp); sg=G.subgraph(nodes); size=len(nodes); dens=nx.density(sg) if size>1 else 0.0
        for i in nodes: comp_size[i]=size; comp_density[i]=dens
    rows=[]
    for i in range(len(combo)):
        rows.append({
            "GraphDegree":float(deg.get(i,0)),"GraphWeightedDegree":float(wdeg.get(i,0)),"GraphPageRank":float(pr.get(i,0)),"GraphCore":float(core.get(i,0)),"GraphArticulation":float(i in art),"GraphBetweenness":float(btw.get(i,0)),"GraphTriangles":float(tri.get(i,0)),"GraphAvgNeighborDegree":float(avg.get(i,0)),"GraphComponentSize":float(comp_size.get(i,1)),"GraphComponentDensity":float(comp_density.get(i,0)),"GraphRelationTypes":float(len(rel_types[i])),"GraphBridgeIncidents":float(bridge_inc.get(i,0)),
        })
    block=pd.DataFrame(rows)
    return block.iloc[:len(ref_raw)].reset_index(drop=True),block.iloc[len(ref_raw):].reset_index(drop=True)


def eval_split(train_raw,train,rel,base_cols,y,tr_idx,va_idx,seed):
    tr,va,cols=fold_features(train,rel,tr_idx,va_idx,base_cols,VARIANT)
    gtr,gva=graph_block(train_raw.iloc[tr_idx].reset_index(drop=True),train_raw.iloc[va_idx].reset_index(drop=True))
    for c in GRAPH_COLS: tr[c]=gtr[c].to_numpy(); va[c]=gva[c].to_numpy()
    base_model=build_tree_models(seed)["CatBoost"]; graph_model=build_tree_models(seed+1000)["CatBoost"]
    base_model.fit(tr[cols],y[tr_idx]); pb=np.asarray(base_model.predict_proba(va[cols]))[:,1]
    gcols=cols+GRAPH_COLS; graph_model.fit(tr[gcols],y[tr_idx]); pg=np.asarray(graph_model.predict_proba(va[gcols]))[:,1]
    return pb,pg


def main():
    EXPORT_DIR.mkdir(parents=True,exist_ok=True); traw=pd.read_csv(DATA_DIR/"train.csv"); teraw=pd.read_csv(DATA_DIR/"test.csv"); train,_,folds,base_cols=prepare_data(); rel,_=prepare_rel(traw,teraw); y=train.Survived.astype(int).to_numpy(); rows=[]; details=[]

    # Trusted fixed folds.
    for f in sorted(np.unique(folds)):
        tr=np.flatnonzero(folds!=f); va=np.flatnonzero(folds==f); pb,pg=eval_split(traw,train,rel,base_cols,y,tr,va,40000+int(f)); rows.append({"surface":"fixed","split":int(f),"base_acc":accuracy_score(y[va],pb>.5),"graph_acc":accuracy_score(y[va],pg>.5),"delta":accuracy_score(y[va],pg>.5)-accuracy_score(y[va],pb>.5),"base_auc":roc_auc_score(y[va],pb),"graph_auc":roc_auc_score(y[va],pg)}); details.append(pd.DataFrame({"PassengerId":traw.iloc[va].PassengerId.astype(int),"Survived":y[va],"surface":"fixed","split":int(f),"base_prob":pb,"graph_prob":pg}))

    # Matched pseudo tests.
    _,pseudo=build_pseudo_splits(traw,teraw)
    for s,(_,va,_) in enumerate(pseudo):
        tr=np.setdiff1d(np.arange(len(train)),va); pb,pg=eval_split(traw,train,rel,base_cols,y,tr,va,41000+s); rows.append({"surface":"pseudo","split":s,"base_acc":accuracy_score(y[va],pb>.5),"graph_acc":accuracy_score(y[va],pg>.5),"delta":accuracy_score(y[va],pg>.5)-accuracy_score(y[va],pb>.5),"base_auc":roc_auc_score(y[va],pb),"graph_auc":roc_auc_score(y[va],pg)}); details.append(pd.DataFrame({"PassengerId":traw.iloc[va].PassengerId.astype(int),"Survived":y[va],"surface":"pseudo","split":s,"base_prob":pb,"graph_prob":pg}))

    # Group-aware stress.
    groups=connected_groups(traw)
    for seed in [42,777]:
        sg=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed); base=np.zeros(len(train)); graph=np.zeros(len(train))
        for f,(tr,va) in enumerate(sg.split(train,y,groups)):
            pb,pg=eval_split(traw,train,rel,base_cols,y,tr,va,42000+seed+f); base[va]=pb; graph[va]=pg
        rows.append({"surface":"group","split":seed,"base_acc":accuracy_score(y,base>.5),"graph_acc":accuracy_score(y,graph>.5),"delta":accuracy_score(y,graph>.5)-accuracy_score(y,base>.5),"base_auc":roc_auc_score(y,base),"graph_auc":roc_auc_score(y,graph)}); details.append(pd.DataFrame({"PassengerId":traw.PassengerId.astype(int),"Survived":y,"surface":"group","split":seed,"base_prob":base,"graph_prob":graph}))

    m=pd.DataFrame(rows); m.to_csv(EXPORT_DIR/"metrics.csv",index=False); pd.concat(details,ignore_index=True).to_csv(EXPORT_DIR/"predictions.csv",index=False)
    sm=m.groupby("surface",as_index=False).agg(mean_base_acc=("base_acc","mean"),mean_graph_acc=("graph_acc","mean"),mean_delta=("delta","mean"),min_delta=("delta","min"),positive=("delta",lambda s:int((s>0).sum())),negative=("delta",lambda s:int((s<0).sum())),mean_base_auc=("base_auc","mean"),mean_graph_auc=("graph_auc","mean")); sm.to_csv(EXPORT_DIR/"summary.csv",index=False)
    print("=== graph centrality summary ==="); print(sm.to_string(index=False,float_format=lambda x:f"{x:.5f}")); print("\n=== all surfaces ==="); print(m.to_string(index=False,float_format=lambda x:f"{x:.5f}")); print("\nNo Kaggle submission was performed.")


if __name__=="__main__": main()
