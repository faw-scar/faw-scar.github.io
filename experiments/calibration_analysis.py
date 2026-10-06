"""Audit independently trained/calibrated GPU folds and assess honest prediction sets.

Conformal scores use RAW ordinal probabilities, so calibration labels are never reused
to fit the predictor underlying conformal scores. Temperatures are for a separate
probability-calibration comparison. Cluster scores are max across timepoints for each
physical plant; finite-sample coverage requires exchangeable plants, not arbitrary field
shift or guaranteed per-grade coverage. No temperature-fitted conformal claim is made.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import os
ROOT=Path(os.environ.get("FAW_DATA_ROOT", "data")).resolve(); HERE=ROOT/"reviewer_revision"
meta=pd.read_csv(ROOT/'metadata.csv').fillna('')
folds=pd.read_csv(HERE/'folds_calibration.csv')
tasks=pd.read_csv(HERE/'calibration_tasks.csv')
rows=[]; audits=[]; sets=[]; coverage=[]

def ece_binary(p,y,w,bins=15):
    result=0.
    for i in range(bins):
        idx=np.minimum((p*bins).astype(int),bins-1)==i
        if idx.any(): result+=w[idx].sum()/w.sum()*abs(np.average(p[idx],weights=w[idx])-np.average(y[idx],weights=w[idx]))
    return float(result)

for task in tasks.to_dict('records'):
    stem=f"{task['config']}{task['tag']}_rep0_fold{task['fold']}"
    pred=ROOT/'oof_ft_probs'/f'{stem}.csv'; cp=ROOT/'oof_ft_probs'/f'{stem}_calibration.csv'
    if not pred.exists() or not cp.exists(): continue
    p=pd.read_csv(pred).merge(folds[['image_path','group']],on='image_path',validate='one_to_one')
    cal=pd.read_csv(cp)
    train=folds[(folds.fold_rep0!=task['fold'])&~folds.image_path.isin(cal.image_path)]
    assert not set(p.group)&set(cal.group)
    assert not set(p.group)&set(train.group)
    assert not set(cal.group)&set(train.group)
    assert set(p.image_path)==set(folds[folds.fold_rep0==task['fold']].image_path)
    audits.append(dict(fold=task['fold'],train_plants=train.group.nunique(),calibration_plants=cal.group.nunique(),test_plants=p.group.nunique(),overlap=0))
    y=(p.y_true>=5).to_numpy(dtype=int)
    w=(1/p.groupby('group').image_path.transform('size')).to_numpy()
    for name,col in [('binary_raw','p_susceptible'),('binary_calibrated','binary_cal_susceptible'),('ordinal_mass_raw','ordinal_binary_raw_susceptible'),('ordinal_mass_calibrated','ordinal_binary_cal_susceptible')]:
        q=p[col].to_numpy(); eps=np.clip(q,1e-12,1-1e-12)
        rows.append(dict(fold=task['fold'],output=name,n=len(p),ece=ece_binary(q,y,w),brier=float(np.average((q-y)**2,weights=w)),nll=float(np.average(-(y*np.log(eps)+(1-y)*np.log(1-eps)),weights=w)),error_rate=float(np.average((q>=.5)!=y,weights=w))))
    def observations(d):
        cols=[f'ordinal_p{i}' for i in range(1,10)]
        q=d.merge(meta[['image_path','timepoint']],on='image_path',validate='one_to_one')
        assert q.groupby(['group','timepoint']).y_true.nunique().eq(1).all()
        t=q.groupby(['group','timepoint'],as_index=False).agg({'y_true':'first',**{c:'mean' for c in cols}})
        return t,t[cols].to_numpy()
    ct,cprobs=observations(cal); tt,tprobs=observations(p)
    ct['score']=1-cprobs[np.arange(len(ct)),ct.y_true.to_numpy(dtype=int)-1]
    scores=ct.groupby('group').score.max().sort_values().to_numpy()
    for target in [.8,.9,.95]:
        rank=int(np.ceil((len(scores)+1)*target))
        quantile=float(scores[rank-1]) if rank<=len(scores) else 1.
        included=tprobs>=1-quantile
        # Empty-set repair only adds labels; coverage cannot decrease.
        for idx in np.flatnonzero(~included.any(1)): included[idx,tprobs[idx].argmax()]=True
        true=(tt.y_true.to_numpy(dtype=int)-1)
        hit=included[np.arange(len(tt)),true]
        sizes=included.sum(1)
        perplant=pd.DataFrame(dict(group=tt.group,hit=hit)).groupby('group').hit.all()
        coverage.append(dict(fold=task['fold'],target=target,calibration_plants=len(scores),quantile=quantile,plant_all_timepoints_coverage=float(perplant.mean()),observation_coverage=float(hit.mean()),mean_set_size=float(sizes.mean()),n_test_plants=len(perplant),n_test_observations=len(tt)))
        for idx,r in tt.iterrows():
            grades=(np.flatnonzero(included[idx])+1).tolist()
            sets.append(dict(fold=task['fold'],target=target,group=r.group,timepoint=r.timepoint,y_true=r.y_true,prediction_set=' '.join(map(str,grades)),interval_low=min(grades),interval_high=max(grades),covered=bool(hit[idx])))
pd.DataFrame(audits).to_csv(HERE/'independent_calibration_audit.csv',index=False)
pd.DataFrame(rows).to_csv(HERE/'fair_binary_calibration.csv',index=False)
pd.DataFrame(coverage).to_csv(HERE/'cluster_conformal_coverage.csv',index=False)
pd.DataFrame(sets).to_csv(HERE/'cluster_conformal_sets.csv',index=False)
lines=['# Independent calibration and cluster prediction sets', '', f'Completed {len(audits)} / 10 independent GPU folds. Missing folds remain unresolved.', '', 'Training, temperature-calibration, and test physical plants are verified mutually disjoint. Both binary head and ordinal-derived mass use identical calibration images, with a separate temperature fitted for each output. Test metrics equal-weight physical plants. Calibration labels do not train the backbone.', '', 'Conformal prediction uses raw ordinal probabilities averaged per plant × timepoint and the maximum nonconformity across timepoints in each calibration plant. Raw scores avoid reusing labels to fit temperature and conformal thresholds. Reported prediction sets target 80/90/95% simultaneous plant-level coverage under exchangeability; subgroup and out-of-domain coverage are not guaranteed. Small calibration cohorts can force all-nine-grade sets at high target coverage.', '', 'Outputs: `fair_binary_calibration.csv`, `independent_calibration_audit.csv`, `cluster_conformal_coverage.csv`, and `cluster_conformal_sets.csv`. The fixed ±1 band remains a tolerance band, irrespective of these separately evaluated prediction sets.']
(HERE/'CALIBRATION_RESULTS.md').write_text('\n'.join(lines)+'\n')
print(f'Audited {len(audits)} independent calibration folds.')
