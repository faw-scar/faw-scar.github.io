"""Build auditable greenhouse-only genotype splits and matched GPU task manifest."""
from pathlib import Path
import json
import pandas as pd
import os
ROOT = Path(os.environ.get("FAW_DATA_ROOT", "data")).resolve()
OUT = ROOT / 'reviewer_revision'
OUT.mkdir(parents=True, exist_ok=True)
f = pd.read_csv(ROOT / 'folds_plant.csv')
m = pd.read_csv(ROOT / 'metadata.csv')
d = f.merge(m, on='image_path', validate='one_to_one', suffixes=('', '_meta'))
g = d[d.group.str.startswith('gh_')].copy()
assert len(g) == 4737 and g.group.nunique() == 84
assert g.gh_entry.notna().all()
assert g.groupby('group').gh_entry.nunique().eq(1).all()
entries = sorted(g.gh_entry.unique())
g['fold_rep0'] = g.gh_entry.map({e: i for i, e in enumerate(entries)})
cols = ['image_path', 'injury_score', 'group', 'fold_rep0']
g[cols].to_csv(OUT / 'folds_genotype.csv', index=False)
f[cols].to_csv(OUT / 'folds_matched_plant.csv', index=False)
checks=[]
for fold,e in enumerate(entries):
    te, tr = g[g.fold_rep0 == fold], g[g.fold_rep0 != fold]
    assert not set(te.group) & set(tr.group)
    assert not set(te.gh_entry) & set(tr.gh_entry)
    checks.append(dict(fold=fold, entry=int(e), test_images=len(te), test_plants=te.group.nunique(), train_plants=tr.group.nunique(), test_grades=sorted(te.injury_score.unique().tolist()), train_grades=sorted(tr.injury_score.unique().tolist())))
(OUT / 'genotype_split_audit.json').write_text(json.dumps(checks, indent=2))
tasks=[]
def add(family, config, folds, count, no_aug=False):
    for fold in range(count):
        tasks.append(dict(task=len(tasks), family=family, config=config, fold=fold, folds=str(folds), tag=f'_review_{family}_seed42', seed=42, epochs=30, img_size=392, bs=16, lr=1e-4, no_aug=no_aug))
add('genotype', 'M3', OUT / 'folds_genotype.csv', 8)
add('matched_lora', 'M3', OUT / 'folds_matched_plant.csv', 10)
add('matched_full', 'FULL_M3', OUT / 'folds_matched_plant.csv', 10)
add('matched_product', 'P', OUT / 'folds_matched_plant.csv', 10)
add('ce_fh', 'CE_FH', OUT / 'folds_matched_plant.csv', 10)
add('ce_fh_dist', 'CE_FH_DIST', OUT / 'folds_matched_plant.csv', 10)
add('corn', 'B', OUT / 'folds_matched_plant.csv', 10)
add('product_noaug', 'P', OUT / 'folds_matched_plant.csv', 10, True)
pd.DataFrame(tasks).to_csv(OUT / 'gpu_tasks.csv', index=False)
# Smoke inputs use disjoint real plant groups, small cohorts, and distinct result tags.
smoke=pd.concat([f[f.fold_rep0 == 0].sample(32, random_state=42), f[f.fold_rep0 != 0].sample(32, random_state=42)])
assert len(smoke[smoke.fold_rep0 == 0]) == 32
assert len(smoke[smoke.fold_rep0 != 0]) == 32
assert not set(smoke[smoke.fold_rep0 == 0].group) & set(smoke[smoke.fold_rep0 != 0].group)
smoke[cols].to_csv(OUT / 'folds_smoke.csv', index=False)
print(f'Prepared {len(tasks)} GPU runs; {len(entries)} truly held-out genotypes; all split overlap checks pass.')
