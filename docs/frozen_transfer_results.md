# Actual cached-feature training experiments

Completed **103 actual refits** of pretrained DINOv2 features with training-only standardization. These are real classifier/regression fits; GPU LoRA transfer is separate and still running. All evaluation splits exclude test physical plants from training. Hyperparameters were fixed before evaluation.

| Evaluation | Head | Mean fold QWK | Pooled OOF QWK | MAE |
|---|---|---:|---:|---:|
| genotype | categorical | 0.189 | 0.351 | 1.665 |
| genotype | ordinal | 0.217 | 0.367 | 1.623 |
| genotype | regression | 0.248 | 0.341 | 1.565 |
| plant_cv_regression | regression | 0.753 | 0.772 | 1.030 |
| planting | categorical | 0.637 | 0.650 | 1.282 |
| planting | ordinal | 0.642 | 0.658 | 1.243 |
| planting | regression | 0.672 | 0.682 | 1.209 |
| temporal_14day_to_7day | categorical | 0.524 | 0.548 | 1.454 |
| temporal_14day_to_7day | ordinal | 0.598 | 0.624 | 1.322 |
| temporal_14day_to_7day | regression | 0.561 | 0.586 | 1.402 |
| temporal_7day_to_14day | categorical | 0.626 | 0.641 | 1.286 |
| temporal_7day_to_14day | ordinal | 0.621 | 0.636 | 1.248 |
| temporal_7day_to_14day | regression | 0.555 | 0.578 | 1.350 |

## Interpretation limits

Genotype holdout is genuine train-on-seven/test-on-eighth, not removal of a genotype from existing ranks. Only eight entries are available. Holding out entry 1 also removes all training examples of grades 1 and 2, so its difficulty combines genotype shift and absent severity support. Per-genotype QWK uses narrower severity marginals than pooled QWK; report both, rather than selecting the more favorable summary.

Temporal experiments train on one DAI on training plants and test the other DAI on different plants. They establish neither same-plant progression prediction nor external-site transfer. Planting holdout has only three test plantings. Ridge regression uses training-only scaling, fixed alpha=1, and rounded/clipped Davis predictions.
