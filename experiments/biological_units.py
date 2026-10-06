"""Evaluate saved physical-plant OOF predictions; no retraining or test-time tuning.

Median hard-grade aggregation is fixed before this analysis. Round halves upward.
Physical plants can change grade with time; retain timepoint labels and equal-weight
plants, rather than assign one invented nine-class label to a longitudinal plant.
Frozen repeat 0 matches adapted folds. Field identity is unknown, so excluded.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import os
ROOT=Path(os.environ.get("FAW_DATA_ROOT", "data")).resolve()
OUT=ROOT/'reviewer_revision'
B=2000
rng=np.random.default_rng(42)
f=pd.read_csv(ROOT/'folds_plant.csv')
m=pd.read_csv(ROOT/'metadata.csv').fillna('')
meta=f.merge(m,on='image_path',validate='one_to_one',suffixes=('','_meta'))
gh=meta[meta.group.str.startswith('gh_')].copy()
assert len(gh)==4737 and gh.group.nunique()==84
assert gh.groupby(['group','timepoint']).injury_score.nunique().eq(1).all(), 'non-consensus unit labels'
assert gh.groupby('group').fold_rep0.nunique().eq(1).all()
plants=sorted(gh.group.unique())
plant_index={g:i for i,g in enumerate(plants)}
boot=rng.multinomial(len(plants),np.full(len(plants),1/len(plants)),size=B)
models={}
for cfg in ['P','M3','M1']:
    files=sorted((ROOT/'oof_ft_probs').glob(f'{cfg}_plant_rep0_fold*.csv'))
    assert len(files)==10, f'incomplete {cfg} folds'
    models[cfg]=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
frozen=pd.read_csv(ROOT/'oof_predictions_plant.csv')
for (backbone,head),d in frozen[frozen.rep==0].groupby(['backbone','head']):
    if head=='ordinal' or head=='majority': models[f'{backbone}_{head}']=d.copy()

def metrics(cm):
    cm=np.asarray(cm,dtype=float).reshape(-1,9,9)
    total=cm.sum((1,2)); norm=cm/total[:,None,None]
    dist=np.abs(np.arange(9)[:,None]-np.arange(9)[None,:])
    expect=norm.sum(2)[:,:,None]*norm.sum(1)[:,None,:]
    den=(expect*dist**2).sum((1,2))
    qwk=np.divide((norm*dist**2).sum((1,2)),den,out=np.full(len(den),np.nan),where=den>0)
    rb=norm[:,:4,:4].sum((1,2))/np.maximum(norm[:,:4,:].sum((1,2)),1e-12)
    sb=norm[:,4:,4:].sum((1,2))/np.maximum(norm[:,4:,:].sum((1,2)),1e-12)
    return dict(qwk=1-qwk,mae=(norm*dist).sum((1,2)),within1=(norm*(dist<=1)).sum((1,2)),within2=(norm*(dist<=2)).sum((1,2)),rs_balanced_accuracy=(rb+sb)/2,catastrophic_rate=(norm*(dist>=3)).sum((1,2)))

def summarize(model,unit,d,weights):
    clusters=np.zeros((len(plants),81))
    pi=d.group.map(plant_index).to_numpy()
    cell=(d.y_true.to_numpy(dtype=int)-1)*9+d.y_pred.to_numpy(dtype=int)-1
    np.add.at(clusters,(pi,cell),np.asarray(weights))
    point=metrics(clusters.sum(0)); estimates=metrics(boot@clusters)
    row=dict(model=model,unit=unit,n_rows=len(d),physical_plants=len(plants))
    for key,value in point.items():
        row[key]=float(value[0]); row[key+'_lo'],row[key+'_hi']=np.nanquantile(estimates[key],[.025,.975])
    return row,estimates

rows=[]; views=[]; predrows=[]; genorows=[]; saved={}; fewviews=[]
for name,p in models.items():
    assert p.image_path.is_unique, f'duplicate OOF rows: {name}'
    d=gh.merge(p[['image_path','y_true','y_pred']],on='image_path',validate='one_to_one')
    assert len(d)==len(gh), f'missing greenhouse predictions: {name}'
    assert np.array_equal(d.injury_score.to_numpy(),d.y_true.to_numpy())
    t=d.groupby(['group','timepoint'],as_index=False).agg(y_true=('y_true','first'),median_pred=('y_pred','median'),n_views=('y_pred','size'),gh_entry=('gh_entry','first'))
    t['y_pred']=np.floor(t.median_pred+.5).astype(int)
    t['model']=name
    predrows.append(t)
    units=[('greenhouse_image',d,np.ones(len(d))),('equal_plant_images',d,1/d.groupby('group').image_path.transform('size')),('plant_timepoint_median',t,np.ones(len(t))),('equal_plant_timepoints',t,1/t.groupby('group').timepoint.transform('size'))]
    for unit,data,w in units:
        row,estimates=summarize(name,unit,data,w)
        rows.append(row); saved[(name,unit)]=estimates
    for (plant,timepoint),v in d.groupby(['group','timepoint']):
        counts=v.y_pred.value_counts()
        binary=(v.y_pred>=5)
        views.append(dict(model=name,group=plant,timepoint=timepoint,views=len(v),grade_std=float(v.y_pred.std(ddof=0)),grade_range=int(v.y_pred.max()-v.y_pred.min()),modal_grade_fraction=float(counts.max()/len(v)),rs_modal_fraction=float(max(binary.mean(),1-binary.mean()))))
    # Genotypes are descriptive fixed-cohort means, not nine-class targets.
    gp=t.groupby('gh_entry',as_index=False).agg(expert_mean=('y_true','mean'),model_mean=('y_pred','mean'),plant_timepoints=('timepoint','size'))
    rho=spearmanr(gp.expert_mean,gp.model_mean).statistic
    genorows.append(dict(model=name,genotypes=len(gp),spearman=float(rho),mean_absolute_difference=float(np.abs(gp.expert_mean-gp.model_mean).mean()),interpretation='descriptive ranking of represented genotypes; not genotype-held-out training'))
    gp['model']=name; gp.to_csv(OUT/f'genotype_means_{name}.csv',index=False)
    # View sampling is a separate sensitivity analysis, not independent plant replication.
    for k in [1,3,5]:
        values=[]
        for repeat in range(100):
            sample=[]
            for (_, _),v in d.groupby(['group','timepoint']):
                pp=rng.choice(v.y_pred.to_numpy(),size=min(k,len(v)),replace=False)
                sample.append((int(v.y_true.iloc[0]),int(np.floor(np.median(pp)+.5))))
            arr=np.array(sample); cm=np.bincount((arr[:,0]-1)*9+arr[:,1]-1,minlength=81)
            values.append(float(metrics(cm)['qwk'][0]))
        fewviews.append(dict(model=name,views_per_timepoint=k,qwk_mean=float(np.mean(values)),view_sampling_lo=float(np.quantile(values,.025)),view_sampling_hi=float(np.quantile(values,.975)),repeats=100,interpretation='variation from sampled views conditional on these 168 observations; not biological CI'))

pd.DataFrame(rows).to_csv(OUT/'biological_unit_metrics.csv',index=False)
pd.concat(predrows).to_csv(OUT/'plant_timepoint_predictions.csv',index=False)
pd.DataFrame(views).to_csv(OUT/'view_consistency.csv',index=False)
pd.DataFrame(genorows).to_csv(OUT/'genotype_descriptive_metrics.csv',index=False)
pd.DataFrame(fewviews).to_csv(OUT/'few_view_sensitivity.csv',index=False)
# Unique biological support per test fold and per grade, including absent grades.
support=[]
for fold in range(10):
    test=gh[gh.fold_rep0==fold]
    for grade in range(1,10):
        v=test[test.injury_score==grade]
        support.append(dict(fold=fold,grade=grade,images=len(v),physical_plants=v.group.nunique(),plant_timepoints=v[['group','timepoint']].drop_duplicates().shape[0]))
pd.DataFrame(support).to_csv(OUT/'fold_grade_support.csv',index=False)
paired=[]
base='dinov2_vitb_ordinal'
for name in ['P','M3','M1']:
    for unit in ['equal_plant_images','plant_timepoint_median']:
        delta=saved[(name,unit)]['qwk']-saved[(base,unit)]['qwk']
        observed=next(r['qwk'] for r in rows if r['model']==name and r['unit']==unit)-next(r['qwk'] for r in rows if r['model']==base and r['unit']==unit)
        lo,hi=np.nanquantile(delta,[.025,.975])
        paired.append(dict(model=name,baseline=base,unit=unit,delta_qwk=observed,lo=float(lo),hi=float(hi)))
pd.DataFrame(paired).to_csv(OUT/'paired_cluster_differences.csv',index=False)
lines=['# Biological-unit evaluation','', 'Generated from existing physical-plant OOF predictions; greenhouse cohort only: **4,737 images, 84 physical plants, 168 plant × timepoint observations**. Frozen repeat 0 matches the adapted folds. All methods use saved hard-grade predictions; R/S is severity-derived at 4/5, not the auxiliary binary head.', '', 'Aggregation: median predicted grade within plant × timepoint, with half grades rounded upward. Each unit has one consensus true label. Plants can change label over time, so no invented single longitudinal Davis grade is used. Physical-plant weighting retains timepoint labels and assigns each plant total weight one. Genotype mean scores/ranks are descriptive, not unseen-genotype validation.', '', '95% percentile confidence intervals resample 84 whole physical plants (2,000 draws; seed 42), with identical draws for paired differences. Field images are excluded because plant identities are unknown. Conditional bootstrap uncertainty does not include training-seed uncertainty.', '', '| Model | Evaluation unit | QWK [95% CI] | MAE | Within-1 | R/S balanced accuracy |','|---|---|---|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['model']} | {r['unit']} | {r['qwk']:.3f} [{r['qwk_lo']:.3f}, {r['qwk_hi']:.3f}] | {r['mae']:.3f} | {r['within1']:.3f} | {r['rs_balanced_accuracy']:.3f} |")
lines += ['', '## Adaptation gains under matched plant-cluster resampling', '', '| Model | Unit | ΔQWK vs frozen DINOv2 | 95% CI |', '|---|---|---:|---|']
for r in paired: lines.append(f"| {r['model']} | {r['unit']} | {r['delta_qwk']:.3f} | [{r['lo']:.3f}, {r['hi']:.3f}] |")
lines += ['', '## Grade support and view sensitivity', '', '`fold_grade_support.csv` includes zero-support test grades; missing grades are not concealed. `view_consistency.csv` reports grade dispersion and R/S consistency per observation. `few_view_sensitivity.csv` compares 1/3/5 randomly sampled views over 100 repeats; its ranges describe view sampling, not biological confidence intervals.', '', '## Limits', '', 'Median aggregation was selected as a simple predefined analysis rule, not optimized on these results. Do not infer multi-environment resistance or unseen-genotype generalization from descriptive genotype rankings. Report greenhouse-only values separately from the mixed-source headline benchmark.']
(OUT/'BIOLOGICAL_UNIT_RESULTS.md').write_text('\n'.join(lines)+'\n')
print(pd.DataFrame(rows)[['model','unit','qwk','mae','within1']].to_string(index=False))
