# Validation of first reviewer analyses

Saved greenhouse image/equal-plant QWK, MAE, binary balanced accuracy, and plant × timepoint median QWK independently match scikit-learn. Eight genotype holdouts cover 84 plants and 4,737 images with no train/test genotype or physical-plant overlap. Test/support grade counts are retained.

**Important discovered confounding:** training-grade absences by held-out entry are {1: [1, 2], 2: [], 3: [], 4: [], 6: [], 7: [], 8: [], 9: []}. Holding out entry 1 removes all grade-1 and grade-2 training examples. That fold tests joint genotype and unseen-severity shift, not isolated genotype morphology. This finding must accompany genotype-held-out claims; absence is not repaired by leaking held-out examples into training.

The 78 main GPU tasks have distinct result/checkpoint names. QWK in final reviewer summaries uses explicit nine-grade support. LoRA/full fine-tuning and independent-calibration smoke runs completed on NVIDIA A100-SXM4-80GB hardware; their accuracies are excluded from scientific results.
