from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import cohen_kappa_score
import os
ROOT=Path(os.environ.get("FAW_DATA_ROOT", "data")).resolve(); HERE=ROOT/"reviewer_revision"
manifest=pd.read_csv(HERE/'field_tasks.csv');rows=[];test_ids=None
for t in manifest.to_dict('records'):
    path=ROOT/'oof_ft_probs'/f"{t['config']}{t['tag']}_rep0_fold{t['fold']}.csv"
    if not path.exists():continue
    p=pd.read_csv(path);ids=set(p.image_path)
    if test_ids is None:test_ids=ids
    assert ids==test_ids,'test cohort changed across budgets or seeds'
    k=int(t['tag'].rsplit('_k',1)[1]);err=np.abs(p.y_true-p.y_pred)
    rows.append(dict(seed=t['seed'],k=k,n_test=len(p),qwk=cohen_kappa_score(p.y_true,p.y_pred,labels=np.arange(1,10),weights='quadratic'),mae=err.mean(),within1=(err<=1).mean(),within2=(err<=2).mean()))
r=pd.DataFrame(rows);r.to_csv(HERE/'fixed_field_results.csv',index=False)
lines=['# Fixed-test repeated field adaptation','',f'Completed {len(rows)} / 15 actual GPU runs. All completed runs share the same held-out field image identities.', '', 'Three seeds combine support-sampling and training randomness. Report individual runs and descriptive SD; three replicates do not support strong uncertainty claims. Visual duplicate auditing does not establish physical field plant independence. No external-site validation or human study is performed.']
if rows:
    s=r.groupby('k')[['qwk','mae','within1']].agg(['mean','std','count'])
    s.to_csv(HERE/'fixed_field_summary.csv')
    lines += ['', '```', s.round(3).to_string(), '```']
(HERE/'FIXED_FIELD_RESULTS.md').write_text('\n'.join(lines)+'\n')
print(f'Aggregated {len(rows)} fixed-test GPU runs.')
