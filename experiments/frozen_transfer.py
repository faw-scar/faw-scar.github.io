"""Real feature-model refits for genotype/planting/time transfer and regression.

Uses only cached pretrained features, training-only scaling, no test tuning.
Temporal transfer excludes overlapping physical plants across train and test:
train on one timepoint on training plants, test on other timepoint on held-out plants.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cv_eval import fit_ordinal, fit_logreg, metrics
import os
ROOT=Path(os.environ.get("FAW_DATA_ROOT", "data")).resolve(); HERE=ROOT/'reviewer_revision'
f=pd.read_csv(ROOT/'folds_plant.csv')
m=pd.read_csv(ROOT/'metadata.csv').fillna('')
g=f.merge(m,on='image_path',validate='one_to_one',suffixes=('','_meta'))
g=g[g.group.str.startswith('gh_')].reset_index(drop=True)
cache=np.load(ROOT/'features_dinov2_vitb.npz',allow_pickle=True)
index={str(p):i for i,p in enumerate(cache['paths'])}
sel=np.array([index[p] for p in g.image_path]); X=cache['feat'][sel]; y=g.injury_score.to_numpy()
assert np.array_equal(cache['y'][sel],y)
tasks=[]
for e in sorted(g.gh_entry.unique()): tasks.append(('genotype',str(e),g.gh_entry.ne(e).to_numpy(),g.gh_entry.eq(e).to_numpy()))
for p in sorted(g.planting.unique()): tasks.append(('planting',p,g.planting.ne(p).to_numpy(),g.planting.eq(p).to_numpy()))
for a,b in [('7day','14day'),('14day','7day')]:
    for fold in range(10): tasks.append((f'temporal_{a}_to_{b}',str(fold),(g.timepoint.eq(a)&g.fold_rep0.ne(fold)).to_numpy(),(g.timepoint.eq(b)&g.fold_rep0.eq(fold)).to_numpy()))
for fold in range(10): tasks.append(('plant_cv_regression',str(fold),g.fold_rep0.ne(fold).to_numpy(),g.fold_rep0.eq(fold).to_numpy()))

def one(family,fold,tr,te,head):
    assert tr.any() and te.any()
    assert not set(g.loc[tr,'group']) & set(g.loc[te,'group'])
    if family=='genotype': assert not set(g.loc[tr,'gh_entry']) & set(g.loc[te,'gh_entry'])
    if family=='planting': assert not set(g.loc[tr,'planting']) & set(g.loc[te,'planting'])
    if head=='ordinal': pred=fit_ordinal(X[tr],y[tr],X[te])
    elif head=='categorical': pred=fit_logreg(X[tr],y[tr],X[te])
    else:
        sc=StandardScaler().fit(X[tr]); reg=Ridge(alpha=1).fit(sc.transform(X[tr]),y[tr])
        pred=np.clip(np.floor(reg.predict(sc.transform(X[te]))+.5),1,9).astype(int)
    row=dict(family=family,fold=fold,head=head,n_train=int(tr.sum()),n_test=int(te.sum()),train_plants=g.loc[tr,'group'].nunique(),test_plants=g.loc[te,'group'].nunique(),**metrics(y[te],pred))
    row['qwk']=float(cohen_kappa_score(y[te],pred,labels=np.arange(1,10),weights='quadratic'))
    p=g.loc[te,['image_path','group','timepoint','gh_entry']].copy()
    p['family']=family; p['fold']=fold; p['head']=head; p['y_true']=y[te]; p['y_pred']=pred
    return row,p

if __name__=='__main__':
    jobs=[]
    for task in tasks:
        heads=['regression'] if task[0]=='plant_cv_regression' else ['ordinal','categorical','regression']
        for head in heads: jobs.append((*task,head))
    out=Parallel(n_jobs=8,verbose=10)(delayed(one)(*args) for args in jobs)
    pd.DataFrame([r for r,p in out]).to_csv(HERE/'frozen_transfer_metrics.csv',index=False)
    pd.concat([p for r,p in out],ignore_index=True).to_csv(HERE/'frozen_transfer_predictions.csv',index=False)
    print(f'Completed {len(out)} actual training/test fits; no physical-plant overlap in any split.',flush=True)
