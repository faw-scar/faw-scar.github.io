"""Physical-plant-stratified greenhouse folds; field identities remain unknown."""
import argparse,os
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def build(metadata, n_splits=10, repeats=2, seed=42):
    d=metadata.copy().fillna('')
    required={'image_path','injury_score','source_set','planting','plant_id','timepoint'}
    if not required.issubset(d.columns): raise ValueError(f'Missing columns: {required-set(d.columns)}')
    if not d.image_path.is_unique: raise ValueError('image_path must be unique')
    numeric=pd.to_numeric(d.injury_score,errors='raise')
    if not numeric.between(1,9).all() or not (numeric==np.floor(numeric)).all(): raise ValueError('Davis grades must be integers 1–9')
    d['injury_score']=numeric.astype(int)
    gh=d.source_set.astype(str).str.startswith('greenhouse')
    if (d.loc[gh,'planting'].astype(str).eq('')|d.loc[gh,'plant_id'].astype(str).eq('')).any(): raise ValueError('Every greenhouse image needs planting and physical plant_id')
    d['group']='field_'+d.image_path
    d.loc[gh,'group']='gh_'+d.loc[gh,'planting'].astype(str)+'_'+d.loc[gh,'plant_id'].astype(str)
    if d.group.nunique()<n_splits: raise ValueError('Fewer groups than requested folds')
    for rep in range(repeats):
        split=StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=seed+rep)
        assignments=np.full(len(d),-1,dtype=int)
        for fold,(_,test) in enumerate(split.split(np.zeros(len(d)),d.injury_score,d.group)): assignments[test]=fold
        d[f'fold_rep{rep}']=assignments
        if not d.groupby('group')[f'fold_rep{rep}'].nunique().eq(1).all(): raise AssertionError('physical plant overlap')
    return d[['image_path','injury_score','group']+[f'fold_rep{r}' for r in range(repeats)]]

if __name__=='__main__':
    root=Path(os.environ.get('FAW_DATA_ROOT','data')).resolve()
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--metadata',type=Path,default=root/'metadata.csv')
    ap.add_argument('--folds',type=int,default=10)
    ap.add_argument('--repeats',type=int,default=2)
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    result=build(pd.read_csv(args.metadata,dtype=str),args.folds,args.repeats,args.seed)
    root.mkdir(parents=True,exist_ok=True)
    result.to_csv(root/'folds_plant.csv',index=False)
    result.to_csv(root/'folds.csv',index=False)
    print(f'{len(result)} images; {result.group.nunique()} groups. Greenhouse physical plants remain intact across DAI. Field groups are per image.')
