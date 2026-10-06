#!/bin/bash
#SBATCH --job-name=faw_review
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
set -euo pipefail
: "${FAW_CODE_ROOT:?Set FAW_CODE_ROOT to the repository directory}"
: "${FAW_DATA_ROOT:?Set FAW_DATA_ROOT to the dataset directory}"
export OMP_NUM_THREADS=8 PYTHONHASHSEED=42
cd "$FAW_CODE_ROOT"
python -u experiments/run_task.py "$@"
