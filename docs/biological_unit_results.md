# Biological-unit evaluation

Generated from existing physical-plant OOF predictions; greenhouse cohort only: **4,737 images, 84 physical plants, 168 plant × timepoint observations**. Frozen repeat 0 matches the adapted folds. All methods use saved hard-grade predictions; R/S is severity-derived at 4/5, not the auxiliary binary head.

Aggregation: median predicted grade within plant × timepoint, with half grades rounded upward. Each unit has one consensus true label. Plants can change label over time, so no invented single longitudinal Davis grade is used. Physical-plant weighting retains timepoint labels and assigns each plant total weight one. Genotype mean scores/ranks are descriptive, not unseen-genotype validation.

95% percentile confidence intervals resample 84 whole physical plants (2,000 draws; seed 42), with identical draws for paired differences. Field images are excluded because plant identities are unknown. Conditional bootstrap uncertainty does not include training-seed uncertainty.

| Model | Evaluation unit | QWK [95% CI] | MAE | Within-1 | R/S balanced accuracy |
|---|---|---|---:|---:|---:|
| P | greenhouse_image | 0.834 [0.784, 0.868] | 0.847 | 0.824 | 0.886 |
| P | equal_plant_images | 0.828 [0.778, 0.864] | 0.860 | 0.823 | 0.882 |
| P | plant_timepoint_median | 0.892 [0.853, 0.920] | 0.673 | 0.887 | 0.896 |
| P | equal_plant_timepoints | 0.892 [0.853, 0.920] | 0.673 | 0.887 | 0.896 |
| M3 | greenhouse_image | 0.840 [0.792, 0.872] | 0.834 | 0.836 | 0.896 |
| M3 | equal_plant_images | 0.835 [0.786, 0.869] | 0.845 | 0.834 | 0.894 |
| M3 | plant_timepoint_median | 0.890 [0.847, 0.920] | 0.708 | 0.893 | 0.913 |
| M3 | equal_plant_timepoints | 0.890 [0.847, 0.920] | 0.708 | 0.893 | 0.913 |
| M1 | greenhouse_image | 0.804 [0.749, 0.842] | 0.942 | 0.795 | 0.887 |
| M1 | equal_plant_images | 0.800 [0.746, 0.838] | 0.950 | 0.796 | 0.885 |
| M1 | plant_timepoint_median | 0.898 [0.859, 0.923] | 0.696 | 0.875 | 0.943 |
| M1 | equal_plant_timepoints | 0.898 [0.859, 0.923] | 0.696 | 0.875 | 0.943 |
| convnextv2_b_majority | greenhouse_image | -0.000 [-0.000, 0.000] | 1.652 | 0.562 | 0.500 |
| convnextv2_b_majority | equal_plant_images | -0.000 [-0.000, 0.000] | 1.625 | 0.580 | 0.500 |
| convnextv2_b_majority | plant_timepoint_median | 0.000 [-0.000, 0.000] | 1.649 | 0.571 | 0.500 |
| convnextv2_b_majority | equal_plant_timepoints | 0.000 [-0.000, 0.000] | 1.649 | 0.571 | 0.500 |
| convnextv2_b_ordinal | greenhouse_image | 0.626 [0.547, 0.686] | 1.319 | 0.683 | 0.783 |
| convnextv2_b_ordinal | equal_plant_images | 0.622 [0.544, 0.682] | 1.322 | 0.682 | 0.782 |
| convnextv2_b_ordinal | plant_timepoint_median | 0.871 [0.824, 0.907] | 0.720 | 0.875 | 0.896 |
| convnextv2_b_ordinal | equal_plant_timepoints | 0.871 [0.824, 0.907] | 0.720 | 0.875 | 0.896 |
| dinov2_vitb_ordinal | greenhouse_image | 0.765 [0.701, 0.810] | 1.048 | 0.757 | 0.883 |
| dinov2_vitb_ordinal | equal_plant_images | 0.761 [0.695, 0.806] | 1.060 | 0.754 | 0.883 |
| dinov2_vitb_ordinal | plant_timepoint_median | 0.891 [0.848, 0.920] | 0.750 | 0.881 | 0.956 |
| dinov2_vitb_ordinal | equal_plant_timepoints | 0.891 [0.848, 0.920] | 0.750 | 0.881 | 0.956 |
| resnet50_ordinal | greenhouse_image | 0.458 [0.380, 0.521] | 1.542 | 0.615 | 0.692 |
| resnet50_ordinal | equal_plant_images | 0.453 [0.377, 0.517] | 1.539 | 0.616 | 0.689 |
| resnet50_ordinal | plant_timepoint_median | 0.668 [0.560, 0.756] | 1.077 | 0.738 | 0.680 |
| resnet50_ordinal | equal_plant_timepoints | 0.668 [0.560, 0.756] | 1.077 | 0.738 | 0.680 |

## Adaptation gains under matched plant-cluster resampling

| Model | Unit | ΔQWK vs frozen DINOv2 | 95% CI |
|---|---|---:|---|
| P | equal_plant_images | 0.067 | [0.044, 0.094] |
| P | plant_timepoint_median | 0.001 | [-0.020, 0.022] |
| M3 | equal_plant_images | 0.075 | [0.054, 0.100] |
| M3 | plant_timepoint_median | -0.001 | [-0.024, 0.023] |
| M1 | equal_plant_images | 0.039 | [0.020, 0.061] |
| M1 | plant_timepoint_median | 0.007 | [-0.012, 0.025] |

## Grade support and view sensitivity

`fold_grade_support.csv` includes zero-support test grades; missing grades are not concealed. `view_consistency.csv` reports grade dispersion and R/S consistency per observation. `few_view_sensitivity.csv` compares 1/3/5 randomly sampled views over 100 repeats; its ranges describe view sampling, not biological confidence intervals.

## Limits

Median aggregation was selected as a simple predefined analysis rule, not optimized on these results. Do not infer multi-environment resistance or unseen-genotype generalization from descriptive genotype rankings. Report greenhouse-only values separately from the mixed-source headline benchmark.
