from pathlib import Path
import os, subprocess, sys, json
import pandas as pd
ROOT=Path(os.environ.get("FAW_DATA_ROOT", "data")).resolve()
HERE=ROOT/"reviewer_revision"
idx=int(os.environ['SLURM_ARRAY_TASK_ID'])
smoke='--smoke' in sys.argv
if '--smoke-calibration' in sys.argv:
    task=dict(config='P',fold=0,folds=str(HERE/'folds_smoke.csv'),tag='_review_cal_smoke',seed=42,epochs=1,img_size=392,bs=16,lr=1e-4,no_aug=False,family='independent_calibration')
elif smoke:
    task=dict(config=['M3','FULL_M3'][idx],fold=0,folds=str(HERE/'folds_smoke.csv'),tag='_review_smoke',seed=42,epochs=1,img_size=392,bs=16,lr=1e-4,no_aug=False)
else:
    task=pd.read_csv(HERE/('field_tasks.csv' if '--field-batch' in sys.argv else 'calibration_tasks.csv' if '--calibration-batch' in sys.argv else 'gpu_tasks.csv')).iloc[idx].to_dict()
print('TASK',json.dumps(task),flush=True)
output=HERE.parent/'results_ft'/f"{task['config']}{task['tag']}_rep0_fold{task['fold']}.json"
pred=HERE.parent/'oof_ft_probs'/f"{task['config']}{task['tag']}_rep0_fold{task['fold']}.csv"
if output.exists() and pred.exists():
    print(f'Already complete: {output}',flush=True)
    raise SystemExit(0)
cmd=[sys.executable,'-u',str(Path(__file__).resolve().parents[1]/'train.py'),'--dump_probs']
if not smoke:
    cmd += ['--save_weights',str(HERE/'checkpoints'/f"{task['config']}{task['tag']}_fold{task['fold']}.pth")]
for key in ['config','fold','folds','tag','seed','epochs','img_size','bs','lr']:
    cmd += ['--'+key,str(task[key])]
if task['no_aug']: cmd.append('--no_aug')
if task.get('family')=='independent_calibration': cmd.append('--calibration')
subprocess.run(cmd,check=True)
