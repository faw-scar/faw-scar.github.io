# faw-scar.github.io

SCAR: ordinal scoring of fall armyworm leaf injury in maize with DINOv2, LoRA, and biological-unit evaluation.

This repository contains reusable research code and reviewer experiment protocols. Raw images, private metadata, pretrained/model checkpoints, credentials, and generated training artifacts are excluded. This repository does not currently publish a GitHub Pages website.

## Scientific scope

The study contains 5,036 curated images: 4,737 greenhouse images from **84 physical plants**, and 299 field images without recoverable physical plant identities. Images are technical views, not independent biological replicates. Physical plants stay together across scoring timepoints in greenhouse evaluation. Field image grouping is not a plant-level independence guarantee.

A fixed ±1 output is a **tolerance band**, not a calibrated prediction interval. Independent human agreement and breeder usability are not measured. Genotype ranking within represented entries does not establish unseen-genotype generalization. Current reviewer GPU experiments test genuine genotype holdouts, matched full/LoRA adaptation, loss components, CORN, augmentation, independent calibration, and repeated fixed-test field transfer.

## Setup

Python 3.11 was used. Install a PyTorch/CUDA build appropriate for your hardware, then:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export FAW_DATA_ROOT=/absolute/path/to/curated_dataset
```

The first backbone load downloads the corresponding `timm` pretrained checkpoint unless it is already cached. For offline cluster execution, prepopulate the cache and set `HF_HOME`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1` in your job environment.

## Dataset contract

`FAW_DATA_ROOT` contains images at relative `image_path` locations and a curated `metadata.csv` with:

| Column | Meaning |
|---|---|
| `image_path` | Relative image path; unique per photograph |
| `injury_score` | Integer Davis grade 1–9 |
| `source_set` | `greenhouse2025` or `summer2026` in this study |
| `planting`, `plant_id` | Together identify a physical greenhouse plant |
| `timepoint` | `7day` or `14day` |
| `gh_entry` | Genotype entry identifier |
| `rater` | Photographer/record provenance; not proof of independent scoring |

Each image inherits its plant × timepoint consensus grade. Physical plant identity persists across both DAI timepoints. Curate the metadata before building folds; the published pipeline does not certify filtering accuracy.

## Main pipeline

```bash
python build_folds.py --metadata "$FAW_DATA_ROOT/metadata.csv"
python extract_features.py --backbones dinov2_vitb convnextv2_b resnet50
python cv_eval.py
python train.py --config M3 --folds "$FAW_DATA_ROOT/folds_plant.csv" \
  --fold 0 --img_size 392 --epochs 30 --bs 16 --lr 1e-4 --seed 42 \
  --tag _plant --dump_probs --save_weights "$FAW_DATA_ROOT/checkpoints/M3_fold0.pth"
```

Repeat training for folds 0–9. Configurations include categorical (`M1`), Frank–Hall ordinal (`M3`), multi-task (`P`), genuine CORN (`B`), matched full fine-tuning (`FULL_M3`), and component ablations (`CE_FH`, `CE_FH_DIST`). There is no claim to implement weight-sharing CORAL.

Training evaluates the fixed final epoch, records the seed and GPU/time/memory measurements, and never selects an epoch using the test labels. Declared trainable parameter counts include the architecture's heads; inactive losses can leave some heads without gradient updates. Ordinal probabilities are exported separately from CE-head probabilities; use `ordinal_p1`–`ordinal_p9` for ordinal outputs.

## Reviewer experiments

For the original study's curated metadata and fold manifest:

```bash
python experiments/prepare_experiments.py
export FAW_CODE_ROOT="$PWD"
sbatch --array=0-1%1 experiments/run_gpu.sbatch --smoke
sbatch --array=0-77%2 experiments/run_gpu.sbatch
```

Supply your cluster's account, partition, and GPU-type options when submitting. Activate the environment before submission. The study-specific preparation checks expected cohort counts to prevent silently evaluating the wrong data. Generated manifests, results, and checkpoints stay under `$FAW_DATA_ROOT/reviewer_revision` or other dataset-local output directories.

The 78-run main manifest covers eight genotype holdouts, ten matched LoRA folds, ten full-fine-tuning folds, and five ten-fold multi-task/component/CORN/augmentation families. Adaptation comparisons use identical manifests, 392 px, 30 epochs, batch 16, and seed 42. The full-fine-tuning comparator uses matched fixed settings; it is not a separately tuned optimum.

```bash
python experiments/aggregate_gpu.py
python experiments/biological_units.py
python experiments/frozen_transfer.py
python experiments/prepare_calibration.py
sbatch --array=0-9%1 experiments/run_gpu.sbatch --calibration-batch
python experiments/calibration_analysis.py
python experiments/prepare_field_repeats.py
sbatch --array=0-14%1 experiments/run_gpu.sbatch --field-batch
python experiments/aggregate_field.py
```

Run aggregation only after the associated jobs finish. Missing/failed tasks remain visible. Feature transfer/refit runs use CPUs because pretrained image features are already cached; LoRA, full fine-tuning, calibration, and few-shot adaptation use real GPUs.

Independent calibration uses disjoint training, calibration, and test physical plants. Binary-head and ordinal-derived probabilities use the same calibration cohort with separate fitted temperatures. Cluster-conformal sets use **raw** ordinal scores to avoid fitting temperature and conformal thresholds on the same labels.

Field repeats share one fixed 149-image test cohort, three seeds, and nested 0/10/25/50/100 support budgets. SHA-256/pHash visual grouping reduces detected duplication but does not recover biological plant identity. The field cohort lacks grades 4 and 9.

## Inference

```bash
python predict.py --checkpoint /path/to/model.pth --image /path/to/whorl.jpg
```

Inference reports ordinal grade, uncalibrated model probabilities, fixed tolerance bands, and Davis-derived low-/high-injury probabilities. It does not infer independently established genetic resistance or reject all unsupported images. Supply an appropriate validated checkpoint; no model weights are distributed here.

## Reviewer response and evidence

- [All 150 reviewer concerns](docs/hostile_review.md)
- [Concern-by-concern status](docs/revision_status.md)
- [Completed biological-unit analysis](docs/biological_unit_results.md)
- [Completed 103 feature-model refits](docs/frozen_transfer_results.md)
- [Validation and genotype–severity confounding](docs/validation.md)
- [Repeated field-transfer protocol](docs/field_repeat_protocol.md)

These documents distinguish completed evidence from GPU jobs in progress and unaddressed empirical gaps. No human re-scoring/user study is planned. Important limitations are answered by qualified claims rather than invented observations. Documentation is a snapshot dated 2026-10-06; refresh it after new experiments finish.
