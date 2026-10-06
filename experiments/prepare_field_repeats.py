"""Fixed test cohort, nested field-label budgets, three seeds, conservative pHash clusters.

Visual clusters reduce detected duplicate contamination; they are NOT physical plant IDs.
The cutoff (64-bit pHash Hamming <=6) is fixed before performance is computed.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from PIL import Image,ImageOps
from scipy.fft import dctn
from sklearn.model_selection import StratifiedGroupKFold
import os
ROOT=Path(os.environ.get("FAW_DATA_ROOT", "data")).resolve(); HERE=ROOT/'reviewer_revision'
HERE.mkdir(parents=True, exist_ok=True)
f=pd.read_csv(ROOT/'folds_plant.csv')
field=f[f.group.str.startswith('field_')].reset_index(drop=True)
gh=f[f.group.str.startswith('gh_')].copy()
assert len(field)==299
hashes=[]; exact=[]
for path in field.image_path:
    imfile=ROOT/path
    exact.append(hashlib.sha256(imfile.read_bytes()).hexdigest())
    with Image.open(imfile) as im:
        a=np.asarray(ImageOps.exif_transpose(im).convert('L').resize((32,32)),dtype=float)
    c=dctn(a,norm='ortho')[:8,:8].ravel()
    hashes.append(int(''.join('1' if v>np.median(c[1:]) else '0' for v in c),2))
parent=list(range(len(field)))
def find(i):
    while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
    return i
pairs=[]
for i in range(len(field)):
    for j in range(i):
        distance=(hashes[i]^hashes[j]).bit_count()
        if exact[i]==exact[j] or distance<=6:
            parent[find(i)]=find(j)
            pairs.append(dict(image_a=field.image_path.iloc[i],image_b=field.image_path.iloc[j],phash_distance=distance,exact_duplicate=exact[i]==exact[j]))
field['visual_cluster']=[f'field_visual_{find(i)}' for i in range(len(field))]
pd.DataFrame(pairs,columns=['image_a','image_b','phash_distance','exact_duplicate']).to_csv(HERE/'field_duplicate_candidates.csv',index=False)
field[['image_path','injury_score','visual_cluster']].to_csv(HERE/'field_visual_clusters.csv',index=False)
# Select one predefined half-split; never select by model accuracy.
sg=StratifiedGroupKFold(n_splits=2,shuffle=True,random_state=20261006)
train_idx,test_idx=next(sg.split(np.zeros(len(field)),field.injury_score,field.visual_cluster))
pool,test=field.iloc[train_idx],field.iloc[test_idx]
assert len(pool)>=100, 'insufficient fixed support pool'
assert not set(pool.visual_cluster)&set(test.visual_cluster)
tasks=[]; manifests=[]
for seed in [42,43,44]:
    # Fixed random order shared across budgets within seed; deterministic nested support sets.
    order=pool.sample(frac=1,random_state=seed)
    for k in [0,10,25,50,100]:
        support=order.head(k)
        train=pd.concat([gh,support],ignore_index=True)
        train['fold_rep0']=1; te=test.copy(); te['fold_rep0']=0
        d=pd.concat([train,te],ignore_index=True)
        name=f'folds_fixed_field_seed{seed}_k{k}.csv'
        d[['image_path','injury_score','group','fold_rep0']].to_csv(HERE/name,index=False)
        assert not set(train.image_path)&set(te.image_path)
        tasks.append(dict(task=len(tasks),family='fixed_field_transfer',config='M3',fold=0,folds=str(HERE/name),tag=f'_review_fixed_field_seed{seed}_k{k}',seed=seed,epochs=30,img_size=392,bs=16,lr=1e-4,no_aug=False))
        manifests.append(dict(seed=seed,k=k,support_images=len(support),test_images=len(te),support_visual_clusters=support.visual_cluster.nunique() if k else 0,test_visual_clusters=te.visual_cluster.nunique()))
pd.DataFrame(tasks).to_csv(HERE/'field_tasks.csv',index=False)
pd.DataFrame(manifests).to_csv(HERE/'fixed_field_split_audit.csv',index=False)
(HERE/'FIELD_REPEAT_PROTOCOL.md').write_text(f'''# Repeated field adaptation protocol

All 15 runs use the same **{len(test)}-image field test cohort**, separated from a {len(pool)}-image support pool by predefined visual duplicate clusters. Budgets 0/10/25/50/100 are nested within each of seeds 42/43/44. All greenhouse plants remain training data. Test images never select hyperparameters or epochs.

A fixed SHA-256 exact-duplicate and 64-bit pHash Hamming-distance <=6 audit identified {len(pairs)} candidate pairs. Visual clusters are not recovered physical plant identities and do not guarantee plant-independent field evaluation. Candidate pairs and split manifests are retained. This is internal single-site domain transfer, not external validation. The field cohort lacks Davis grades 4 and 9.

Variation across the three seeds combines support-sample and training randomness; it does not estimate external-site or biological-cluster uncertainty. Present individual runs and descriptive dispersion, not strong inferential CIs from only three seeds.
''')
print(f'Prepared 15 fixed-test GPU runs, {len(test)} test images, {len(pairs)} duplicate candidate pairs.')
