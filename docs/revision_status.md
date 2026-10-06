# Reviewer revision status — all 150 concerns

**User constraints:** no human studies or independent re-scoring. Real GPU training required. No fabricated observations, human agreement, external cohort, or recovered field identity.

## Execution

- Main GPU array: `16736471`, 78 runs, max 2 concurrent A100s; genuine genotype holdout, matched full/LoRA, component/CORN/no-augmentation.
- Independent-calibration GPU array: `16736501`, 10 runs, max 1 concurrent A100; smoke `16736477` passed.
- Fixed-test field GPU array: `16736521`, 15 runs, after main array, max 1 concurrent A100.
- Automatic result jobs: `16736472`, `16736506`, `16736524`; missing/failed runs stay unresolved.
- CPU refit job: `16736497`, **completed 103 real fits**.
- Total planned scientific GPU runs: **103**. Smoke tests are separate and excluded.

## Evidence already computed

- SCAR plant × timepoint median QWK 0.892; frozen DINOv2 0.891. Matched cluster-bootstrap ΔQWK 0.001 [−0.020, 0.022]: no established multi-view advantage.
- All grades 1/2 occur in entry 1. Its holdout removes those grades from training: genotype and severity shift are confounded.
- Historical calibration generator fits and evaluates temperature on pooled OOF: independent-calibration claim removed locally; genuine separate-plant experiment running.
- Field duplicate scan: SHA-256/pHash, 299 images, zero candidate pairs at Hamming <=6. Same-plant identity is still unknown.

## Individual response and completion status

### 1. 5,036 images are not 5,036 independent samples.

**Status:** Computed; interpretation limits remain.

**Reviewer request:** The greenhouse cohort contains only 84 physical plants, about 55 views per plant, and 8 genotypes. Report plants and grade support in each fold; aggregate predictions per plant × timepoint; compare image-weighted and plant-weighted performance; use plant-cluster confidence intervals. Grouped splitting prevents repeated-view leakage but does not remove pseudoreplication in image-level summaries.

**Current response/work:** Biological-unit metrics, whole-plant bootstrap, per-fold grade support and view-sensitivity outputs completed; 84 plants remain the biological sample size. No field identity guarantee.

### 2. Why are 84 plants enough?

**Status:** Computed; interpretation limits remain.

**Reviewer request:** Extreme grades have few biological observations. Explain 10-fold CV, grade support, fold stability, possible 5-fold alternatives, and the exact stratification algorithm when physical plants carry different labels at 7 and 14 DAI.

**Current response/work:** Biological-unit metrics, whole-plant bootstrap, per-fold grade support and view-sensitivity outputs completed; 84 plants remain the biological sample size. No field identity guarantee.

### 3. Field images grouped individually remain a leakage vulnerability.

**Status:** Partial audit; intervention evidence open.

**Reviewer request:** Investigate repeated angles, sequential capture, neighboring plants, sessions, and near duplicates. Replace unqualified “leakage-free” with greenhouse physical-plant grouping and disclosed field image grouping.

**Current response/work:** Field exact-hash/pHash scan completed (no candidate pairs at predeclared threshold). This does not recover field plant IDs. Tag masking and validated injury/background intervention remain unresolved.

### 4. Greenhouse-to-field failure can dominate the review.

**Status:** Repeated fixed-test GPU queued; historical result corrected.

**Reviewer request:** Define whether the application targets greenhouse or field use; report device, lighting, distance, occlusion, invalid-image, other-pest, and healthy-maize limitations. Update this criticism to distinguish the completed frozen and LoRA transfer results.

**Current response/work:** Three seeds, five nested budgets, fixed 149-image field test, visual-duplicate audit. Field plant identities unknown; three seeds and one site do not establish external validity.

### 5. Generalization may be overclaimed.

**Status:** Repeated fixed-test GPU queued; historical result corrected.

**Reviewer request:** Distinguish in-distribution grouped validation, cross-timepoint transfer, and cross-domain field transfer. Pooled source CV is internal validation, not external validation.

**Current response/work:** Three seeds, five nested budgets, fixed 149-image field test, visual-duplicate audit. Field plant identities unknown; three seeds and one site do not establish external validity.

### 6. Backbone coverage is limited.

**Status:** GPU running/queued.

**Reviewer request:** Justify DINOv2, ConvNeXt-V2, and ResNet-50; consider standard ViT, CLIP/SigLIP, MAE, Swin, EfficientNet, MobileNet, or DINOv3. Highest priority: a full-fine-tuning DINOv2 comparator.

**Current response/work:** Matched full DINOv2, LoRA ordinal and multi-task models on ten identical folds; measured parameters, VRAM/time. A tuned-best adapted CNN comparator remains open.

### 7. Actual LoRA efficiency evidence is missing.

**Status:** GPU running/queued.

**Reviewer request:** Quantify trainable parameters, parameter fraction, peak VRAM, training time per fold, inference cost, and performance against full fine-tuning.

**Current response/work:** Matched full DINOv2, LoRA ordinal and multi-task models on ten identical folds; measured parameters, VRAM/time. A tuned-best adapted CNN comparator remains open.

### 8. Is SCAR a new method?

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Frame known DINOv2, LoRA, Frank–Hall, distance loss, auxiliary binary head, class balancing, and calibration as an applied phenotyping pipeline and benchmark rather than a novel architecture.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 9. “Product head” is nonstandard.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Prefer “multi-task ordinal head” unless a precise mathematical rationale exists.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 10. Loss weights appear hand-selected.

**Status:** GPU running/queued; additional sensitivity open.

**Reviewer request:** Explain the provenance of 0.4 CE, 0.6 ordinal, 0.1 distance, and 0.5 binary; distinguish literature defaults, validation selection, and test-driven selection. Assess sensitivity.

**Current response/work:** CE+Frank–Hall, distance, binary, pure ordinal and genuine CORN comparisons. Fixed weights and LoRA rank; broad weight/rank/imbalance sweeps remain open. No false CORAL implementation claim.

### 11. Loss-component ablation is incomplete.

**Status:** GPU running/queued; additional sensitivity open.

**Reviewer request:** Compare ordinal only, CE + ordinal, added distance, and added R/S task. Report severity and binary calibration outcomes.

**Current response/work:** CE+Frank–Hall, distance, binary, pure ordinal and genuine CORN comparisons. Fixed weights and LoRA rank; broad weight/rank/imbalance sweeps remain open. No false CORAL implementation claim.

### 12. Why Frank–Hall rather than CORAL/CORN?

**Status:** GPU running/queued; additional sensitivity open.

**Reviewer request:** Benchmark genuine implementations, ordinal logistic regression, EMD, or explain the choice. A shared loss with unconstrained independent threshold weights is not a genuine CORAL implementation.

**Current response/work:** CE+Frank–Hall, distance, binary, pure ordinal and genuine CORN comparisons. Fixed weights and LoRA rank; broad weight/rank/imbalance sweeps remain open. No false CORAL implementation claim.

### 13. Rank consistency conflicts with unconstrained thresholds.

**Status:** GPU running/queued; additional sensitivity open.

**Reviewer request:** Explain Frank–Hall nonmonotonicity and avoid implying strict rank consistency.

**Current response/work:** CE+Frank–Hall, distance, binary, pure ordinal and genuine CORN comparisons. Fixed weights and LoRA rank; broad weight/rank/imbalance sweeps remain open. No false CORAL implementation claim.

### 14. Exact accuracy is only approximately 39%.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Justify ±1 tolerance agronomically and avoid replacement-of-experts claims. Report how near-boundary errors affect advancement decisions.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 15. The 4/5 boundary matters more than global agreement.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Report false advancement/rejection costs, ROC/PR, PPV/NPV, validation-selected operating points, and threshold tradeoffs.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 16. “Resistant/susceptible” may overstate biology.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Use low-/high-injury or explicitly Davis-derived categories; distinguish observed injury from validated genetic resistance across environments.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 17. Perfect genotype ranking over eight genotypes is limited evidence.

**Status:** GPU running; feature refits completed.

**Reviewer request:** Determine whether all genotypes appeared in training. Perform train-on-seven/test-on-unseen-eighth evaluation; deleting one genotype from existing rankings is not that experiment.

**Current response/work:** Eight genuine genotype-disjoint GPU holdouts. 103 feature refits completed. Holding out entry 1 removes training grades 1/2; report joint genotype/severity shift.

### 18. Plant grouping does not prevent genotype confounding.

**Status:** GPU running; feature refits completed.

**Reviewer request:** Evaluate unseen genetic material or limit inference to represented genotype distributions.

**Current response/work:** Eight genuine genotype-disjoint GPU holdouts. 103 feature refits completed. Holding out entry 1 removes training grades 1/2; report joint genotype/severity shift.

### 19. Identical scores on 90 shared units require provenance.

**Status:** Address through limitations; no human study.

**Reviewer request:** Were raters blinded and independent, jointly scoring, verifying, or using a common record? Consensus records are not independent agreement evidence.

**Current response/work:** User excludes human studies. Existing consensus records cannot estimate independent human agreement; no expert-equivalence, validated breeder usability, or measured total labor-saving claim.

### 20. Inter-rater variability motivates the task but is not measured.

**Status:** Address through limitations; no human study.

**Reviewer request:** Independently re-score a subset if feasible; compute human–human QWK and intra-rater reliability.

**Current response/work:** User excludes human studies. Existing consensus records cannot estimate independent human agreement; no expert-equivalence, validated breeder usability, or measured total labor-saving claim.

### 21. No human baseline limits interpretation.

**Status:** Address through limitations; no human study.

**Reviewer request:** Compare model–human agreement with human reproducibility before claiming expert-level performance.

**Current response/work:** User excludes human studies. Existing consensus records cannot estimate independent human agreement; no expert-equivalence, validated breeder usability, or measured total labor-saving claim.

### 22. Kappa interpretation thresholds are heuristic.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Avoid treating κ > 0.80 as a universal biological adequacy standard.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 23. Why QWK is primary needs justification.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Explain ordinal agreement relative to MAE, rank correlation, C-index, weighted F1, or EMD.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 24. QWK depends on severity prevalence.

**Status:** Computed; interpretation limits remain.

**Reviewer request:** Consider balanced resampling, per-grade metrics, macro-MAE, and prevalence sensitivity.

**Current response/work:** Biological-unit metrics, whole-plant bootstrap, per-fold grade support and view-sensitivity outputs completed; 84 plants remain the biological sample size. No field identity guarantee.

### 25. CV-fold inference remains dependent.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Clarify corrected t-test versus Wilcoxon, fold-based effect sizes, plant-bootstrap differences, and why adapted models have one repeat versus frozen models' two.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 26. Pairing only the first frozen repeat needs justification.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Assess sensitivity to the second repeat or evaluate all models on identical repeated splits.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 27. Model-selection leakage is unclear.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Document selection of backbones, LoRA rank, learning rate, augmentation, weights, and model configuration; distinguish benchmark estimates from nested model-selection estimates.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 28. Define exactly what temperature calibration scales.

**Status:** Independent-calibration GPU running.

**Reviewer request:** Specify binary versus severity logits, global/fold-specific temperatures, calibration cohort size, optimizer, objective, bins, and uncertainty.

**Current response/work:** Separate training/calibration/test plants; fair binary/ordinal temperatures; raw-ordinal cluster-conformal sets. Historical pooled-OOF calibration is exploratory and fixed ±1 is a tolerance band.

### 29. Fixed ±1 is not a calibrated interval.

**Status:** Independent-calibration GPU running.

**Reviewer request:** Temperature scaling does not confer interval coverage. Use “severity tolerance band” and reserve calibrated language for validated probabilities.

**Current response/work:** Separate training/calibration/test plants; fair binary/ordinal temperatures; raw-ordinal cluster-conformal sets. Historical pooled-OOF calibration is exploratory and fixed ±1 is a tolerance band.

### 30. Consider conformal prediction.

**Status:** Independent-calibration GPU running.

**Reviewer request:** If claiming prediction sets, evaluate coverage targets, empirical coverage, set size, per-grade and boundary coverage, with group-disjoint calibration.

**Current response/work:** Separate training/calibration/test plants; fair binary/ordinal temperatures; raw-ordinal cluster-conformal sets. Historical pooled-OOF calibration is exploratory and fixed ±1 is a tolerance band.

### 31. ECE alone is insufficient.

**Status:** Independent-calibration GPU running.

**Reviewer request:** Report NLL/Brier, classwise or adaptive ECE, reliability curves, and source-specific calibration; pooled calibration does not establish calibration under shift.

**Current response/work:** Separate training/calibration/test plants; fair binary/ordinal temperatures; raw-ordinal cluster-conformal sets. Historical pooled-OOF calibration is exploratory and fixed ±1 is a tolerance band.

### 32. Entropy triage is weak.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** AUROC 0.58 and 35% error capture from 30% reviewed images provide modest enrichment. Tone down utility claims.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 33. Evaluate selective prediction properly.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Risk–coverage curves, AURC/E-AURC, and selective error are preferable to one review-budget point.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 34. The VLM extension may distract.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** One example establishes no improvement or explanation faithfulness. Remove from the main story or keep entirely supplementary; disclose hallucination and confidence limitations.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 35. Validate the cleaning gate.

**Status:** No human audit; computational checks remain open.

**Reviewer request:** Record model, prompt, threshold, removed-image review, false positives/negatives, cleaning-before-splitting, and possible selection bias.

**Current response/work:** Do not invent cleaning precision/recall or visibility labels. Automated sensitivity can be evaluated, but human-verified filtering/label-visibility accuracy cannot be claimed.

### 36. Shortcut learning remains plausible.

**Status:** Partial audit; intervention evidence open.

**Reviewer request:** Audit tags, whorl crops, background-only inputs, soil/pots, saliency, and nearest neighbors. Intervention tests are stronger evidence than visual saliency alone.

**Current response/work:** Field exact-hash/pHash scan completed (no candidate pairs at predeclared threshold). This does not recover field plant IDs. Tag masking and validated injury/background intervention remain unresolved.

### 37. Planting/session leakage may remain.

**Status:** Actual refits completed; GPU extension open.

**Reviewer request:** Perform leave-one-planting-out or acquisition-session-held-out evaluation.

**Current response/work:** 103 training-only-scaled feature-model fits include planting holdout, physical-plant-disjoint temporal transfer and ridge regression. Report fold means and pooled scores separately.

### 38. Photographer comparison may be confounded.

**Status:** Partial audit; intervention evidence open.

**Reviewer request:** Examine photographer × planting × genotype × timepoint contingency tables; similar stratified QWK does not remove confounding.

**Current response/work:** Field exact-hash/pHash scan completed (no candidate pairs at predeclared threshold). This does not recover field plant IDs. Tag masking and validated injury/background intervention remain unresolved.

### 39. Timepoint stratification is not temporal transfer.

**Status:** Actual refits completed; GPU extension open.

**Reviewer request:** Train at 7 DAI and test at 14, and vice versa, with an explicit physical-plant overlap policy.

**Current response/work:** 103 training-only-scaled feature-model fits include planting holdout, physical-plant-disjoint temporal transfer and ridge regression. Report fold means and pooled scores separately.

### 40. Field grades 4 and 9 are absent.

**Status:** Repeated fixed-test GPU queued; historical result corrected.

**Reviewer request:** Full-spectrum field validity and the 4/5 boundary are not demonstrated.

**Current response/work:** Three seeds, five nested budgets, fixed 149-image field test, visual-duplicate audit. Field plant identities unknown; three seeds and one site do not establish external validity.

### 41. Few-shot transfer can be contaminated or unstable.

**Status:** Repeated fixed-test GPU queued; historical result corrected.

**Reviewer request:** Document sampling, train/calibration/test disjointness, near duplicates, seeds, repeated budgets, and uncertainty bars.

**Current response/work:** Three seeds, five nested budgets, fixed 149-image field test, visual-duplicate audit. Field plant identities unknown; three seeds and one site do not establish external validity.

### 42. No independent external site is tested.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Consider another location, year, camera, breeding program, or natural infestation cohort.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 43. Artificial infestation differs from natural injury.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Controlled 30-neonate conditions do not represent heterogeneous pest timing, density, or mixed damage.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 44. Scope is whorl-stage injury.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Avoid extrapolation to tassel/ear injury, other stages, or other pests.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 45. Capture protocol needs detail.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Camera, resolution, distance, lighting, angle, compression, orientation, view selection, and photographs per plant/timepoint.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 46. Explain 392 × 392 resolution.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Consider 224/392/518 sensitivity to small lesions and pinholes.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 47. Augmentation claims need ablation.

**Status:** No-augmentation GPU queued; detailed variants open.

**Reviewer request:** Compare no augmentation, geometric only, mild photometric, and strong photometric under corrected physical-plant splits.

**Current response/work:** Matched multi-task augmentation/no-augmentation GPU runs queued; geometric/photometric/background component studies still open.

### 48. Background randomization needs quantitative results.

**Status:** No-augmentation GPU queued; detailed variants open.

**Reviewer request:** Report ExG/GrabCut and white/blur/random background experiments, including negative outcomes.

**Current response/work:** Matched multi-task augmentation/no-augmentation GPU runs queued; geometric/photometric/background component studies still open.

### 49. Combined imbalance corrections need justification.

**Status:** GPU running/queued; additional sensitivity open.

**Reviewer request:** Weighted sampling, effective-number weights, and label smoothing may overcompensate; assess separately.

**Current response/work:** CE+Frank–Hall, distance, binary, pure ordinal and genuine CORN comparisons. Fixed weights and LoRA rank; broad weight/rank/imbalance sweeps remain open. No false CORAL implementation claim.

### 50. Explain β = 0.999.

**Status:** GPU running/queued; additional sensitivity open.

**Reviewer request:** Provide provenance or tuning rationale.

**Current response/work:** CE+Frank–Hall, distance, binary, pure ordinal and genuine CORN comparisons. Fixed weights and LoRA rank; broad weight/rank/imbalance sweeps remain open. No false CORAL implementation claim.

### 51. Explain LoRA insertion and settings.

**Status:** GPU running/queued; additional sensitivity open.

**Reviewer request:** qkv/proj, rank 16, alpha 32, dropout 0.05; consider rank and insertion sensitivity.

**Current response/work:** CE+Frank–Hall, distance, binary, pure ordinal and genuine CORN comparisons. Fixed weights and LoRA rank; broad weight/rank/imbalance sweeps remain open. No false CORAL implementation claim.

### 52. Frozen CNN competitors may be weak baselines.

**Status:** GPU running/queued.

**Reviewer request:** Include at least one end-to-end adapted CNN and full-fine-tuned DINOv2.

**Current response/work:** Matched full DINOv2, LoRA ordinal and multi-task models on ten identical folds; measured parameters, VRAM/time. A tuned-best adapted CNN comparator remains open.

### 53. Separate representation advantage from adaptation advantage.

**Status:** GPU running/queued.

**Reviewer request:** Fairly compare frozen backbones and adapted backbones within explicit experiment families.

**Current response/work:** Matched full DINOv2, LoRA ordinal and multi-task models on ten identical folds; measured parameters, VRAM/time. A tuned-best adapted CNN comparator remains open.

### 54. Majority prediction is a weak baseline.

**Status:** Actual refits completed; GPU extension open.

**Reviewer request:** Add regression, shallow ML, handcrafted features, or an appropriate simple task-specific model.

**Current response/work:** 103 training-only-scaled feature-model fits include planting holdout, physical-plant-disjoint temporal transfer and ridge regression. Report fold means and pooled scores separately.

### 55. Why classification rather than scalar regression?

**Status:** Actual refits completed; GPU extension open.

**Reviewer request:** Compare MSE/Huber regression, rounding/clipping to Davis grades, against ordinal classification using QWK and MAE.

**Current response/work:** 103 training-only-scaled feature-model fits include planting holdout, physical-plant-disjoint temporal transfer and ridge regression. Report fold means and pooled scores separately.

### 56. Could damage-area estimation suffice?

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Evaluate interpretable image-processing/segmentation damage fraction if suitable labels or masks exist.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 57. Breeder-facing utility lacks direct user evidence.

**Status:** Address through limitations; no human study.

**Reviewer request:** Use “breeder-oriented output” unless actual usability or decision studies demonstrate utility.

**Current response/work:** User excludes human studies. Existing consensus records cannot estimate independent human agreement; no expert-equivalence, validated breeder usability, or measured total labor-saving claim.

### 58. Throughput and labor-saving evidence are missing.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Measure inference latency, hardware, acquisition time, manual scoring time, and processing throughput.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 59. Photography itself may be the bottleneck.

**Status:** Address through limitations; no human study.

**Reviewer request:** Address framing requirements, real-time operation, operational cost, and total workflow time.

**Current response/work:** User excludes human studies. Existing consensus records cannot estimate independent human agreement; no expert-equivalence, validated breeder usability, or measured total labor-saving claim.

### 60. Dozens of views per plant are unrealistic deployment requirements.

**Status:** Computed; interpretation limits remain.

**Reviewer request:** Clarify single-image inference; report within-plant prediction dispersion and R/S consistency, and multi-view aggregation.

**Current response/work:** Biological-unit metrics, whole-plant bootstrap, per-fold grade support and view-sensitivity outputs completed; 84 plants remain the biological sample size. No field identity guarantee.

### 61. Unequal view counts distort image-weighted metrics.

**Status:** Computed; interpretation limits remain.

**Reviewer request:** Report equal weighting per physical plant and genotype, alongside conventional image-level results.

**Current response/work:** Biological-unit metrics, whole-plant bootstrap, per-fold grade support and view-sensitivity outputs completed; 84 plants remain the biological sample size. No field identity guarantee.

### 62. Show representative errors.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Include adjacent-grade errors, severe errors, boundary crossings, and field failures, with explanations.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 63. Report catastrophic errors.

**Status:** Computed; interpretation limits remain.

**Reviewer request:** Count |error| ≥ 3, investigate quality, occlusion, environment, and label visibility.

**Current response/work:** Biological-unit metrics, whole-plant bootstrap, per-fold grade support and view-sensitivity outputs completed; 84 plants remain the biological sample size. No field identity guarantee.

### 64. Break down errors by source.

**Status:** Computed; interpretation limits remain.

**Reviewer request:** Greenhouse versus field catastrophic-error rates.

**Current response/work:** Biological-unit metrics, whole-plant bootstrap, per-fold grade support and view-sensitivity outputs completed; 84 plants remain the biological sample size. No field identity guarantee.

### 65. Break down errors by planting/year.

**Status:** Computed; interpretation limits remain.

**Reviewer request:** Identify batch-specific failure or confounding.

**Current response/work:** Biological-unit metrics, whole-plant bootstrap, per-fold grade support and view-sensitivity outputs completed; 84 plants remain the biological sample size. No field identity guarantee.

### 66. Break down errors by genotype.

**Status:** Computed; interpretation limits remain.

**Reviewer request:** Identify performance gaps across germplasm.

**Current response/work:** Biological-unit metrics, whole-plant bootstrap, per-fold grade support and view-sensitivity outputs completed; 84 plants remain the biological sample size. No field identity guarantee.

### 67. Genotype morphology may confound severity.

**Status:** GPU running; feature refits completed.

**Reviewer request:** Leaf architecture, color, waxiness, vigor; genotype-held-out training is a stronger test.

**Current response/work:** Eight genuine genotype-disjoint GPU holdouts. 103 feature refits completed. Holding out entry 1 removes training grades 1/2; report joint genotype/severity shift.

### 68. Calibration may vary by genotype.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Explore if supported by sample size.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 69. Disclose binary prevalence.

**Status:** Computed; interpretation limits remain.

**Reviewer request:** Report low-/high-injury proportions and denominators.

**Current response/work:** Biological-unit metrics, whole-plant bootstrap, per-fold grade support and view-sensitivity outputs completed; 84 plants remain the biological sample size. No field identity guarantee.

### 70. Binary metrics may conflict across text and figures.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Reconcile binary-head predictions versus severity-threshold decisions and their exact confusion counts.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 71. The chosen SCAR model is not the highest-QWK variant.

**Status:** GPU running/queued.

**Reviewer request:** Justify 0.846 product versus 0.852 ordinal based on validated outputs, not severity superiority.

**Current response/work:** Matched full DINOv2, LoRA ordinal and multi-task models on ten identical folds; measured parameters, VRAM/time. A tuned-best adapted CNN comparator remains open.

### 72. The binary head may not change decisions.

**Status:** Independent-calibration GPU running.

**Reviewer request:** Compare appropriately calibrated ordinal-derived probabilities with calibrated binary-head probabilities.

**Current response/work:** Separate training/calibration/test plants; fair binary/ordinal temperatures; raw-ordinal cluster-conformal sets. Historical pooled-OOF calibration is exploratory and fixed ±1 is a tolerance band.

### 73. Calibration comparisons must be fair.

**Status:** Independent-calibration GPU running.

**Reviewer request:** Raw ordinal mass versus calibrated binary outputs is not a fair comparison; fit both using the same disjoint calibration cohort.

**Current response/work:** Separate training/calibration/test plants; fair binary/ordinal temperatures; raw-ordinal cluster-conformal sets. Historical pooled-OOF calibration is exploratory and fixed ±1 is a tolerance band.

### 74. Temperature fitting must not reuse evaluation observations.

**Status:** Independent-calibration GPU running.

**Reviewer request:** Show the nested training/calibration/test sequence explicitly.

**Current response/work:** Separate training/calibration/test plants; fair binary/ordinal temperatures; raw-ordinal cluster-conformal sets. Historical pooled-OOF calibration is exploratory and fixed ±1 is a tolerance band.

### 75. Explain temperature scaling for ordinal output.

**Status:** Independent-calibration GPU running.

**Reviewer request:** Specify transformations and reconstruction of ordinal class probabilities.

**Current response/work:** Separate training/calibration/test plants; fair binary/ordinal temperatures; raw-ordinal cluster-conformal sets. Historical pooled-OOF calibration is exploratory and fixed ±1 is a tolerance band.

### 76. ECE depends on the 15-bin choice.

**Status:** Independent-calibration GPU running.

**Reviewer request:** Assess adaptive ECE or binning sensitivity.

**Current response/work:** Separate training/calibration/test plants; fair binary/ordinal temperatures; raw-ordinal cluster-conformal sets. Historical pooled-OOF calibration is exploratory and fixed ±1 is a tolerance band.

### 77. Calibration metrics need uncertainty.

**Status:** Independent-calibration GPU running.

**Reviewer request:** Use appropriate physical-plant bootstrap CIs.

**Current response/work:** Separate training/calibration/test plants; fair binary/ordinal temperatures; raw-ordinal cluster-conformal sets. Historical pooled-OOF calibration is exploratory and fixed ±1 is a tolerance band.

### 78. Assess calibration under field shift.

**Status:** Independent-calibration GPU running.

**Reviewer request:** Report field ECE/Brier separately.

**Current response/work:** Separate training/calibration/test plants; fair binary/ordinal temperatures; raw-ordinal cluster-conformal sets. Historical pooled-OOF calibration is exploratory and fixed ±1 is a tolerance band.

### 79. Deployment needs an explicit validity boundary.

**Status:** Partial audit; intervention evidence open.

**Reviewer request:** Clarify actual app rejection behavior for non-maize, wrong stage, blur, other pests, or unsupported conditions; an optional VLM gate does not establish app-wide validation.

**Current response/work:** Field exact-hash/pHash scan completed (no candidate pairs at predeclared threshold). This does not recover field plant IDs. Tag masking and validated injury/background intervention remain unresolved.

### 80. “Available from authors” weakens reproducibility.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Prefer persistent repositories, DOI, weights, manifests, environment, seeds, exact commits, and metadata schema.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 81. Reconcile “released” with access-on-request wording.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Use one accurate availability statement.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 82. Specify licensing and redistribution rights.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Clarify USDA imagery access and metadata restrictions.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 83. Record implementation details.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** PyTorch/timm versions, GPU, workers, seeds, preprocessing, scheduler, checkpoint selection, inference.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 84. Explain epoch/checkpoint selection.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Was validation QWK used, fixed epoch count, or early stopping? The held-out test fold must not select checkpoints.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 85. Field-adaptation methods need full detail.

**Status:** Repeated fixed-test GPU queued; historical result corrected.

**Reviewer request:** Make sampling and training as reproducible as the headline CV.

**Current response/work:** Three seeds, five nested budgets, fixed 149-image field test, visual-duplicate audit. Field plant identities unknown; three seeds and one site do not establish external validity.

### 86. Few-shot uncertainty is missing.

**Status:** Repeated fixed-test GPU queued; historical result corrected.

**Reviewer request:** Quantify variability across sampled support sets and seeds.

**Current response/work:** Three seeds, five nested budgets, fixed 149-image field test, visual-duplicate audit. Field plant identities unknown; three seeds and one site do not establish external validity.

### 87. Statistical gains need agronomic meaning.

**Status:** GPU running/queued; additional sensitivity open.

**Reviewer request:** Interpret +0.033 QWK in terms of grade error and boundary decisions.

**Current response/work:** CE+Frank–Hall, distance, binary, pure ordinal and genuine CORN comparisons. Fixed weights and LoRA rank; broad weight/rank/imbalance sweeps remain open. No false CORAL implementation claim.

### 88. Large fold-based Cohen's d values need explanation.

**Status:** GPU running/queued.

**Reviewer request:** Supplement standardized fold differences with cluster-bootstrap ΔQWK intervals.

**Current response/work:** Matched full DINOv2, LoRA ordinal and multi-task models on ten identical folds; measured parameters, VRAM/time. A tuned-best adapted CNN comparator remains open.

### 89. Define the Holm correction family.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** State all included comparisons.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 90. Choose a primary test.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Corrected resampled t-test can be primary; Wilcoxon a sensitivity analysis. Avoid opportunistic interpretation of whichever is significant.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 91. “Modest but reliable” overstates a nonsignificant primary test.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Prefer “directionally consistent but not significant under the primary corrected test.”

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 92. Abstract significance claims must match comparisons.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Specify the model family and correction used.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 93. Literature novelty claims need support.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Explain search scope and avoid universal “first” claims unsupported by the reviewed evidence.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 94. Comparison tables may look selective.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** State inclusion criteria; distinguish unrelated ordinal papers from directly comparable crop-injury studies; avoid a marketing-style checklist.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 95. Tangential in-review SAGE self-citation may be removable.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Keep only if it contributes necessary context.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 96. “Adaptive Ranking” needs a natural method meaning.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Clarify what is adapted and ranked, and whether a ranking loss exists.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 97. Reconcile dataset years.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Distinguish image acquisition years from broader agronomic trial history.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 98. Define replicated trials.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Separate biological replicates, technical views, plantings, field blocks, and design randomization.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 99. Add agronomic experimental detail.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Conditions, photoperiod, temperature, planting layout, genotype replication, infestation, age, nutrients, blocks, seasons.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 100. Eight genotypes limit breeding claims.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Frame as a methodological phenotyping proof of concept.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 101. Longitudinal scores are correlated.

**Status:** Computed; interpretation limits remain.

**Reviewer request:** Explain growth morphology, correlation of 7/14 DAI labels, and grouping across timepoints.

**Current response/work:** Biological-unit metrics, whole-plant bootstrap, per-fold grade support and view-sensitivity outputs completed; 84 plants remain the biological sample size. No field identity guarantee.

### 102. Longitudinal prediction is a possible extension.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Predict 14 DAI injury from 7 DAI or model paired images, without treating this as required evidence for the present task.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 103. Timepoint sample imbalance matters.

**Status:** Computed; interpretation limits remain.

**Reviewer request:** Report 7/14 DAI image and biological-unit counts and performance.

**Current response/work:** Biological-unit metrics, whole-plant bootstrap, per-fold grade support and view-sensitivity outputs completed; 84 plants remain the biological sample size. No field identity guarantee.

### 104. Justify the 1–4/5–9 binary cutoff.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Cite relevant practice or experimental protocol; avoid claiming universality.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 105. Show Davis rubric semantics.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Provide grade descriptions alongside representative imagery.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 106. Examples may expose acquisition artifacts.

**Status:** Partial audit; intervention evidence open.

**Reviewer request:** Inspect sample grids for systematic pots, tags, hands, backgrounds, and lighting by grade.

**Current response/work:** Field exact-hash/pHash scan completed (no candidate pairs at predeclared threshold). This does not recover field plant IDs. Tag masking and validated injury/background intervention remain unresolved.

### 107. Saliency should include successes and failures.

**Status:** Partial audit; intervention evidence open.

**Reviewer request:** Useful descriptive evidence, without causal claims.

**Current response/work:** Field exact-hash/pHash scan completed (no candidate pairs at predeclared threshold). This does not recover field plant IDs. Tag masking and validated injury/background intervention remain unresolved.

### 108. Background-only intervention is stronger than saliency.

**Status:** Partial audit; intervention evidence open.

**Reviewer request:** Prioritize direct shortcut testing.

**Current response/work:** Field exact-hash/pHash scan completed (no candidate pairs at predeclared threshold). This does not recover field plant IDs. Tag masking and validated injury/background intervention remain unresolved.

### 109. Audit duplicates and near duplicates.

**Status:** Partial audit; intervention evidence open.

**Reviewer request:** Use exact hashes, perceptual hashes, or embedding neighbors, with verification across groups.

**Current response/work:** Field exact-hash/pHash scan completed (no candidate pairs at predeclared threshold). This does not recover field plant IDs. Tag masking and validated injury/background intervention remain unresolved.

### 110. Ensure augmented variants never cross splits.

**Status:** Partial audit; intervention evidence open.

**Reviewer request:** State online training-only augmentation.

**Current response/work:** Field exact-hash/pHash scan completed (no candidate pairs at predeclared threshold). This does not recover field plant IDs. Tag masking and validated injury/background intervention remain unresolved.

### 111. Folders and filenames encode labels.

**Status:** Partial audit; intervention evidence open.

**Reviewer request:** Verify the model consumes pixels only and metadata cannot leak through preprocessing.

**Current response/work:** Field exact-hash/pHash scan completed (no candidate pairs at predeclared threshold). This does not recover field plant IDs. Tag masking and validated injury/background intervention remain unresolved.

### 112. Tags remaining in valid photographs can leak information.

**Status:** Partial audit; intervention evidence open.

**Reviewer request:** Tag-only removal is insufficient; test tag masking.

**Current response/work:** Field exact-hash/pHash scan completed (no candidate pairs at predeclared threshold). This does not recover field plant IDs. Tag masking and validated injury/background intervention remain unresolved.

### 113. Hands/rulers and other objects may be shortcuts.

**Status:** Partial audit; intervention evidence open.

**Reviewer request:** Include these in the capture-cue audit.

**Current response/work:** Field exact-hash/pHash scan completed (no candidate pairs at predeclared threshold). This does not recover field plant IDs. Tag masking and validated injury/background intervention remain unresolved.

### 114. Test cropped leaf/whorl images.

**Status:** Partial audit; intervention evidence open.

**Reviewer request:** Compare against whole-image inputs with a validated mask/crop method.

**Current response/work:** Field exact-hash/pHash scan completed (no candidate pairs at predeclared threshold). This does not recover field plant IDs. Tag masking and validated injury/background intervention remain unresolved.

### 115. Non-destructive imaging is not unique over visual scoring.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Avoid presenting a shared property as a demonstrated advantage.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 116. Local labels may reduce scalability.

**Status:** Repeated fixed-test GPU queued; historical result corrected.

**Reviewer request:** Discuss amortized local adaptation rather than implying universal zero-label deployment; incorporate the completed zero-shot LoRA result.

**Current response/work:** Three seeds, five nested budgets, fixed 149-image field test, visual-duplicate audit. Field plant identities unknown; three seeds and one site do not establish external validity.

### 117. Adaptation frequency is unknown.

**Status:** Repeated fixed-test GPU queued; historical result corrected.

**Reviewer request:** Per farm/year/camera requirements remain untested.

**Current response/work:** Three seeds, five nested budgets, fixed 149-image field test, visual-duplicate audit. Field plant identities unknown; three seeds and one site do not establish external validity.

### 118. Few-shot adaptation may be a central contribution.

**Status:** Repeated fixed-test GPU queued; historical result corrected.

**Reviewer request:** Elevate the quantified frozen-versus-LoRA transfer experiment while preserving its uncertainty limitations.

**Current response/work:** Three seeds, five nested budgets, fixed 149-image field test, visual-duplicate audit. Field plant identities unknown; three seeds and one site do not establish external validity.

### 119. Explain unsuccessful CORAL alignment.

**Status:** Repeated fixed-test GPU queued; historical result corrected.

**Reviewer request:** Distinguish feature-covariance alignment from CORAL ordinal classification; document layer, preprocessing, data, and method.

**Current response/work:** Three seeds, five nested budgets, fixed 149-image field test, visual-duplicate audit. Field plant identities unknown; three seeds and one site do not establish external validity.

### 120. Self-supervised/test-time adaptation are future options.

**Status:** Repeated fixed-test GPU queued; historical result corrected.

**Reviewer request:** Consider self-training or entropy minimization without suggesting they were tested.

**Current response/work:** Three seeds, five nested budgets, fixed 149-image field test, visual-duplicate audit. Field plant identities unknown; three seeds and one site do not establish external validity.

### 121. The paper may be diffuse.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Center grouped biological data, ordinal scoring, adaptation, robust evaluation, field shift, and decision outputs; move VLM/app/extended analyses to supplement.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 122. INSIGHT boxes may conflict with journal style.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Check norms before retaining editorial presentation elements.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 123. Figure 1 can make the VLM look central.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Remove or clearly mark as exploratory/supplementary.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 124. Presence detection is not a severity baseline.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Use its contrast for task definition, not performance superiority.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 125. Make hypotheses explicit.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Separate representation, adaptation, ordinal loss, decision calibration, and field adaptation hypotheses.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 126. Label exploratory analyses honestly.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Distinguish prespecified comparisons from analyses added after observing results.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 127. Provide per-fold values.

**Status:** Computed; interpretation limits remain.

**Reviewer request:** Include metrics and predictions in supplementary/reproducible artifacts.

**Current response/work:** Biological-unit metrics, whole-plant bootstrap, per-fold grade support and view-sensitivity outputs completed; 84 plants remain the biological sample size. No field identity guarantee.

### 128. Fine-tuning seeds add variance beyond CV.

**Status:** Repeated fixed-test GPU queued; historical result corrected.

**Reviewer request:** Repeat representative splits or all CV across seeds if feasible.

**Current response/work:** Three seeds, five nested budgets, fixed 149-image field test, visual-duplicate audit. Field plant identities unknown; three seeds and one site do not establish external validity.

### 129. Separate fold and seed variance.

**Status:** Repeated fixed-test GPU queued; historical result corrected.

**Reviewer request:** Do not imply one captures the other.

**Current response/work:** Three seeds, five nested budgets, fixed 149-image field test, visual-duplicate audit. Field plant identities unknown; three seeds and one site do not establish external validity.

### 130. Binary metric differences need CIs.

**Status:** Independent-calibration GPU running.

**Reviewer request:** Use matched cluster resampling where identity is known.

**Current response/work:** Separate training/calibration/test plants; fair binary/ordinal temperatures; raw-ordinal cluster-conformal sets. Historical pooled-OOF calibration is exploratory and fixed ±1 is a tolerance band.

### 131. Rare-grade performance needs biological-unit uncertainty.

**Status:** Computed; interpretation limits remain.

**Reviewer request:** Include unique plant support and cluster intervals.

**Current response/work:** Biological-unit metrics, whole-plant bootstrap, per-fold grade support and view-sensitivity outputs completed; 84 plants remain the biological sample size. No field identity guarantee.

### 132. Image bootstrap is inappropriate for repeated greenhouse views.

**Status:** Computed; interpretation limits remain.

**Reviewer request:** Use plant clusters; disclose that field clustering cannot be recovered.

**Current response/work:** Biological-unit metrics, whole-plant bootstrap, per-fold grade support and view-sensitivity outputs completed; 84 plants remain the biological sample size. No field identity guarantee.

### 133. Genotype inference has only eight genotype units.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Plant resampling conditional on fixed genotypes does not establish a broader genotype population.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 134. Explain the genotype-rank interval.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Clarify what is resampled and avoid overly precise generalization claims.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 135. Deleting one item preserves an already perfect ranking easily.

**Status:** GPU running; feature refits completed.

**Reviewer request:** Leave-one-genotype-out rank analysis is not unseen-genotype learning.

**Current response/work:** Eight genuine genotype-disjoint GPU holdouts. 103 feature refits completed. Holding out entry 1 removes training grades 1/2; report joint genotype/severity shift.

### 136. Document image label inheritance.

**Status:** Computed; interpretation limits remain.

**Reviewer request:** Explain one consensus plant-timepoint score assigned to many views.

**Current response/work:** Biological-unit metrics, whole-plant bootstrap, per-fold grade support and view-sensitivity outputs completed; 84 plants remain the biological sample size. No field identity guarantee.

### 137. Inherited labels may be invisible in a particular view.

**Status:** No human audit; computational checks remain open.

**Reviewer request:** Audit injury visibility and acknowledge image-label noise.

**Current response/work:** Do not invent cleaning precision/recall or visibility labels. Automated sensitivity can be evaluated, but human-verified filtering/label-visibility accuracy cannot be claimed.

### 138. Multi-view aggregation may be more biologically valid.

**Status:** Computed; interpretation limits remain.

**Reviewer request:** Evaluate a few photos per plant, median predictions, and diminishing returns.

**Current response/work:** Biological-unit metrics, whole-plant bootstrap, per-fold grade support and view-sensitivity outputs completed; 84 plants remain the biological sample size. No field identity guarantee.

### 139. Why train at image rather than plant level?

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Discuss multi-instance modeling as an extension if not evaluated.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 140. Displaying probability 1.00 may imply certainty.

**Status:** Independent-calibration GPU running.

**Reviewer request:** Explain rounding and calibration limits in the app.

**Current response/work:** Separate training/calibration/test plants; fair binary/ordinal temperatures; raw-ordinal cluster-conformal sets. Historical pooled-OOF calibration is exploratory and fixed ±1 is a tolerance band.

### 141. Rare low-injury calibration matters.

**Status:** Independent-calibration GPU running.

**Reviewer request:** Report classwise behavior where support allows.

**Current response/work:** Separate training/calibration/test plants; fair binary/ordinal temperatures; raw-ordinal cluster-conformal sets. Historical pooled-OOF calibration is exploratory and fixed ±1 is a tolerance band.

### 142. Describe calibration imbalance.

**Status:** Independent-calibration GPU running.

**Reviewer request:** Report label support and weighting policy.

**Current response/work:** Separate training/calibration/test plants; fair binary/ordinal temperatures; raw-ordinal cluster-conformal sets. Historical pooled-OOF calibration is exploratory and fixed ±1 is a tolerance band.

### 143. Alternative calibration methods are possible.

**Status:** Independent-calibration GPU running.

**Reviewer request:** Isotonic/Dirichlet/vector scaling can be future or comparative work, without inflated scope.

**Current response/work:** Separate training/calibration/test plants; fair binary/ordinal temperatures; raw-ordinal cluster-conformal sets. Historical pooled-OOF calibration is exploratory and fixed ±1 is a tolerance band.

### 144. Benchmark-selected model estimates are not untouched selection estimates.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Clarify outcome-driven choice of DINOv2/LoRA and limits of internal CV.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 145. Separate inherited from tuned hyperparameters.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Record each setting's provenance.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 146. No locked final cohort exists.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Consider independent planting/year/site; do not rebrand a repeatedly inspected cohort as untouched.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 147. The field cohort could help but lacks plant identity.

**Status:** Partial audit; intervention evidence open.

**Reviewer request:** Keep the identity caveat attached to field conclusions.

**Current response/work:** Field exact-hash/pHash scan completed (no candidate pairs at predeclared threshold). This does not recover field plant IDs. Tag masking and validated injury/background intervention remain unresolved.

### 148. Internal versus external validation should be explicit.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Use consistent terminology throughout.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 149. Conclusion must not imply replacement readiness.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** A step toward higher-throughput screening is defensible; demonstrated expert replacement is not.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.

### 150. Make Plant Phenomics fit explicit.

**Status:** Documentation/method audit; some evidence missing.

**Reviewer request:** Center biology, units of inference, breeding decisions, acquisition practicalities, and field validity over generic AI language.

**Current response/work:** Use concern-specific qualified wording below; no new external cohort or human observations. Do not equate a response with completed empirical validation.
