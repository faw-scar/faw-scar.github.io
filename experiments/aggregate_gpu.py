"""Aggregate only genuine reviewer runs; explicitly disclose failed/missing tasks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, balanced_accuracy_score
import os
ROOT=Path(os.environ.get("FAW_DATA_ROOT", "data")).resolve()
HERE=ROOT/"reviewer_revision"
manifest=pd.read_csv(HERE/'gpu_tasks.csv')
rows=[]; status=[]
for task in manifest.to_dict('records'):
    stem=f"{task['config']}{task['tag']}_rep0_fold{task['fold']}"
    result=ROOT/'results_ft'/f'{stem}.json'
    pred=ROOT/'oof_ft_probs'/f'{stem}.csv'
    state='complete' if result.exists() and pred.exists() else 'missing_or_failed'
    status.append(dict(task=task['task'],family=task['family'],fold=task['fold'],state=state))
    if state=='complete':
        row=json.loads(result.read_text()); row['family']=task['family']; row['task']=task['task']
        p=pd.read_csv(pred)
        assert len(p)==row['n_test'] and p.image_path.is_unique
        assert np.allclose(p.filter(regex=r'^ordinal_p[1-9]$').sum(axis=1),1,atol=1e-5)
        row['qwk_reported_by_training']=row['qwk']
        row['qwk']=float(cohen_kappa_score(p.y_true,p.y_pred,labels=np.arange(1,10),weights='quadratic'))
        rows.append(row)
pd.DataFrame(status).to_csv(HERE/'gpu_completion.csv',index=False)
lines=['# GPU reviewer experiment results', '', f'Completed {len(rows)} / {len(manifest)} planned runs. Missing runs remain unresolved. Smoke tests are excluded.', '', 'Fixed protocol: seed 42, 30 final epochs, 392 px, batch 16, AdamW LR 1e-4, identical corrected physical-plant folds for adaptation/ablation comparisons. Genotype transfer is greenhouse-only, trained with the eighth genotype entirely excluded. These are single-seed comparisons at fixed hyperparameters, not tuned-best full-fine-tuning or a nested model-selection estimate.', '', '| Family | Folds complete | Mean fold QWK | Mean MAE | Mean trainable params | Mean peak train VRAM GiB | Mean train seconds/fold |', '|---|---:|---:|---:|---:|---:|---:|']
if rows:
    r=pd.DataFrame(rows); r.to_csv(HERE/'gpu_per_fold_results.csv',index=False)
    for family,d in r.groupby('family'):
        expected=int((manifest.family==family).sum())
        lines.append(f"| {family} | {len(d)}/{expected} | {d.qwk.mean():.3f} | {d.mae.mean():.3f} | {d.trainable_params.mean():.0f} | {d.peak_train_vram_bytes.mean()/2**30:.2f} | {d.train_seconds.mean():.1f} |")
    summary=[]
    meta=pd.read_csv(ROOT/'metadata.csv').fillna('')
    for family in manifest.family.unique():
        tasks=manifest[manifest.family==family]
        if not all(next(s['state'] for s in status if s['task']==i)=='complete' for i in tasks.task): continue
        dfs=[]
        for t in tasks.to_dict('records'):
            p=ROOT/'oof_ft_probs'/f"{t['config']}{t['tag']}_rep0_fold{t['fold']}.csv"
            dfs.append(pd.read_csv(p))
        p=pd.concat(dfs,ignore_index=True)
        assert p.image_path.is_unique
        p.to_csv(HERE/f'oof_{family}.csv',index=False)
        for cohort in ['all','greenhouse']:
            q=p.merge(meta[['image_path','source_set','planting','plant_id','timepoint','gh_entry']],on='image_path',validate='one_to_one')
            if cohort=='greenhouse': q=q[q.source_set=='greenhouse2025'].copy()
            if q.empty: continue
            error=np.abs(q.y_true-q.y_pred)
            summary.append(dict(family=family,cohort=cohort,unit='image',n=len(q),qwk=cohen_kappa_score(q.y_true,q.y_pred,labels=np.arange(1,10),weights='quadratic'),mae=error.mean(),within1=(error<=1).mean(),rs_balanced_accuracy=balanced_accuracy_score(q.y_true>=5,q.y_pred>=5)))
            if cohort=='greenhouse':
                t=q.groupby(['planting','plant_id','timepoint'],as_index=False).agg(y_true=('y_true','first'),median=('y_pred','median'))
                t['y_pred']=np.floor(t['median']+.5).astype(int)
                summary.append(dict(family=family,cohort=cohort,unit='plant_timepoint_median',n=len(t),qwk=cohen_kappa_score(t.y_true,t.y_pred,labels=np.arange(1,10),weights='quadratic'),mae=mean_absolute_error(t.y_true,t.y_pred),within1=(np.abs(t.y_true-t.y_pred)<=1).mean(),rs_balanced_accuracy=balanced_accuracy_score(t.y_true>=5,t.y_pred>=5)))
    pd.DataFrame(summary).to_csv(HERE/'gpu_pooled_results.csv',index=False)
lines += ['', 'Efficiency values are measured during these actual runs, including the real input pipeline. Inference throughput is fold-evaluation throughput, not a separately warmed deployment benchmark. Peak allocated VRAM excludes framework-reserved memory and host RAM. No human agreement, external-site validity, or tag-mask intervention can be inferred from these experiments.', '', 'Full results: `gpu_per_fold_results.csv`; complete-family predictions: `oof_<family>.csv`; cohort summaries: `gpu_pooled_results.csv`; task failures: `gpu_completion.csv`.']
(HERE/'GPU_RESULTS.md').write_text('\n'.join(lines)+'\n')
print(f'Aggregated {len(rows)} / {len(manifest)} real experiment runs.')
