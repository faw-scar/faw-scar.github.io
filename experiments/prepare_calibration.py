from pathlib import Path
import os,pandas as pd
ROOT=Path(os.environ.get('FAW_DATA_ROOT','data')).resolve()
HERE=ROOT/'reviewer_revision';HERE.mkdir(parents=True,exist_ok=True)
f=pd.read_csv(ROOT/'folds_plant.csv');g=f[f.group.str.startswith('gh_')]
if g.empty:raise ValueError('No physical greenhouse plants found')
g[['image_path','injury_score','group','fold_rep0']].to_csv(HERE/'folds_calibration.csv',index=False)
fold_ids=sorted(g.fold_rep0.unique())
tasks=[dict(task=i,family='independent_calibration',config='P',fold=int(fold),folds=str(HERE/'folds_calibration.csv'),tag='_review_independent_calibration_seed42',seed=42,epochs=30,img_size=392,bs=16,lr=1e-4,no_aug=False) for i,fold in enumerate(fold_ids)]
pd.DataFrame(tasks).to_csv(HERE/'calibration_tasks.csv',index=False)
print(f'Prepared {len(tasks)} independent training/calibration/test GPU folds.')
