# SCAR / FAW paper: hostile journal review and revision checklist

Source: reviewer critique supplied by Tirtho on 2026-10-06. This is a structured transcription of its verdict, all 150 numbered concerns, and prioritized requests, with wording condensed for tracking. It is not a verbatim transcript. References to “PlantPhenomics (7)” in the supplied review are document-reference placeholders, not verified bibliographic citations.

## Reviewer perspective

I read this as if I were a **hostile/rough journal reviewer**, not as a collaborator trying to defend the work. The paper has a strong applied story, but a tough reviewer can still attack it from several directions. The biggest vulnerability is that the manuscript repeatedly looks like a **5,036-image study**, while biologically it is primarily an **84-greenhouse-plant study with many repeated images**, plus 299 field images without recoverable plant identity. Acknowledging this does not stop a reviewer from making it the central criticism.

## Likely overall verdict: Major Revision

> The manuscript addresses an important phenotyping problem and contains several careful methodological choices, including ordinal modeling, group-aware cross-validation, corrected statistical tests, and explicit domain-shift analysis. However, the biological sample size is small relative to the apparent image count, external field validation is weak, several deployment-oriented claims exceed the evidence, and some methodological choices and ablations are insufficiently justified. I would require substantial revision before the claims about breeder-facing utility, calibration, generalization, and resistance screening can be supported.

Major revision is more likely than outright rejection if the journal values applied plant phenotyping.

## Important update since the review text

The completed October 6 transfer experiment changes concerns 4, 5, 116, 118, and the concluding review wording: **frozen** greenhouse-only DINOv2 achieves QWK 0.125, but greenhouse-only **LoRA ordinal** achieves 0.626, rising to 0.809 after adding 100 field labels. Its curve is not strictly monotonic. The manuscript prose was corrected and pushed in commit `9571119`. The broader concerns about field identity, varying test cohorts, single-run estimates, and independent external validation still apply.

## All 150 reviewer concerns

### Biological independence and evaluation design

1. **5,036 images are not 5,036 independent samples.** The greenhouse cohort contains only 84 physical plants, about 55 views per plant, and 8 genotypes. Report plants and grade support in each fold; aggregate predictions per plant × timepoint; compare image-weighted and plant-weighted performance; use plant-cluster confidence intervals. Grouped splitting prevents repeated-view leakage but does not remove pseudoreplication in image-level summaries.
2. **Why are 84 plants enough?** Extreme grades have few biological observations. Explain 10-fold CV, grade support, fold stability, possible 5-fold alternatives, and the exact stratification algorithm when physical plants carry different labels at 7 and 14 DAI.
3. **Field images grouped individually remain a leakage vulnerability.** Investigate repeated angles, sequential capture, neighboring plants, sessions, and near duplicates. Replace unqualified “leakage-free” with greenhouse physical-plant grouping and disclosed field image grouping.
4. **Greenhouse-to-field failure can dominate the review.** Define whether the application targets greenhouse or field use; report device, lighting, distance, occlusion, invalid-image, other-pest, and healthy-maize limitations. Update this criticism to distinguish the completed frozen and LoRA transfer results.
5. **Generalization may be overclaimed.** Distinguish in-distribution grouped validation, cross-timepoint transfer, and cross-domain field transfer. Pooled source CV is internal validation, not external validation.
6. **Backbone coverage is limited.** Justify DINOv2, ConvNeXt-V2, and ResNet-50; consider standard ViT, CLIP/SigLIP, MAE, Swin, EfficientNet, MobileNet, or DINOv3. Highest priority: a full-fine-tuning DINOv2 comparator.
7. **Actual LoRA efficiency evidence is missing.** Quantify trainable parameters, parameter fraction, peak VRAM, training time per fold, inference cost, and performance against full fine-tuning.
8. **Is SCAR a new method?** Frame known DINOv2, LoRA, Frank–Hall, distance loss, auxiliary binary head, class balancing, and calibration as an applied phenotyping pipeline and benchmark rather than a novel architecture.
9. **“Product head” is nonstandard.** Prefer “multi-task ordinal head” unless a precise mathematical rationale exists.
10. **Loss weights appear hand-selected.** Explain the provenance of 0.4 CE, 0.6 ordinal, 0.1 distance, and 0.5 binary; distinguish literature defaults, validation selection, and test-driven selection. Assess sensitivity.
11. **Loss-component ablation is incomplete.** Compare ordinal only, CE + ordinal, added distance, and added R/S task. Report severity and binary calibration outcomes.
12. **Why Frank–Hall rather than CORAL/CORN?** Benchmark genuine implementations, ordinal logistic regression, EMD, or explain the choice. A shared loss with unconstrained independent threshold weights is not a genuine CORAL implementation.
13. **Rank consistency conflicts with unconstrained thresholds.** Explain Frank–Hall nonmonotonicity and avoid implying strict rank consistency.
14. **Exact accuracy is only approximately 39%.** Justify ±1 tolerance agronomically and avoid replacement-of-experts claims. Report how near-boundary errors affect advancement decisions.
15. **The 4/5 boundary matters more than global agreement.** Report false advancement/rejection costs, ROC/PR, PPV/NPV, validation-selected operating points, and threshold tradeoffs.
16. **“Resistant/susceptible” may overstate biology.** Use low-/high-injury or explicitly Davis-derived categories; distinguish observed injury from validated genetic resistance across environments.
17. **Perfect genotype ranking over eight genotypes is limited evidence.** Determine whether all genotypes appeared in training. Perform train-on-seven/test-on-unseen-eighth evaluation; deleting one genotype from existing rankings is not that experiment.
18. **Plant grouping does not prevent genotype confounding.** Evaluate unseen genetic material or limit inference to represented genotype distributions.
19. **Identical scores on 90 shared units require provenance.** Were raters blinded and independent, jointly scoring, verifying, or using a common record? Consensus records are not independent agreement evidence.
20. **Inter-rater variability motivates the task but is not measured.** Independently re-score a subset if feasible; compute human–human QWK and intra-rater reliability.
21. **No human baseline limits interpretation.** Compare model–human agreement with human reproducibility before claiming expert-level performance.
22. **Kappa interpretation thresholds are heuristic.** Avoid treating κ > 0.80 as a universal biological adequacy standard.
23. **Why QWK is primary needs justification.** Explain ordinal agreement relative to MAE, rank correlation, C-index, weighted F1, or EMD.
24. **QWK depends on severity prevalence.** Consider balanced resampling, per-grade metrics, macro-MAE, and prevalence sensitivity.
25. **CV-fold inference remains dependent.** Clarify corrected t-test versus Wilcoxon, fold-based effect sizes, plant-bootstrap differences, and why adapted models have one repeat versus frozen models' two.
26. **Pairing only the first frozen repeat needs justification.** Assess sensitivity to the second repeat or evaluate all models on identical repeated splits.
27. **Model-selection leakage is unclear.** Document selection of backbones, LoRA rank, learning rate, augmentation, weights, and model configuration; distinguish benchmark estimates from nested model-selection estimates.
28. **Define exactly what temperature calibration scales.** Specify binary versus severity logits, global/fold-specific temperatures, calibration cohort size, optimizer, objective, bins, and uncertainty.
29. **Fixed ±1 is not a calibrated interval.** Temperature scaling does not confer interval coverage. Use “severity tolerance band” and reserve calibrated language for validated probabilities.
30. **Consider conformal prediction.** If claiming prediction sets, evaluate coverage targets, empirical coverage, set size, per-grade and boundary coverage, with group-disjoint calibration.
31. **ECE alone is insufficient.** Report NLL/Brier, classwise or adaptive ECE, reliability curves, and source-specific calibration; pooled calibration does not establish calibration under shift.
32. **Entropy triage is weak.** AUROC 0.58 and 35% error capture from 30% reviewed images provide modest enrichment. Tone down utility claims.
33. **Evaluate selective prediction properly.** Risk–coverage curves, AURC/E-AURC, and selective error are preferable to one review-budget point.
34. **The VLM extension may distract.** One example establishes no improvement or explanation faithfulness. Remove from the main story or keep entirely supplementary; disclose hallucination and confidence limitations.
35. **Validate the cleaning gate.** Record model, prompt, threshold, removed-image review, false positives/negatives, cleaning-before-splitting, and possible selection bias.
36. **Shortcut learning remains plausible.** Audit tags, whorl crops, background-only inputs, soil/pots, saliency, and nearest neighbors. Intervention tests are stronger evidence than visual saliency alone.
37. **Planting/session leakage may remain.** Perform leave-one-planting-out or acquisition-session-held-out evaluation.
38. **Photographer comparison may be confounded.** Examine photographer × planting × genotype × timepoint contingency tables; similar stratified QWK does not remove confounding.
39. **Timepoint stratification is not temporal transfer.** Train at 7 DAI and test at 14, and vice versa, with an explicit physical-plant overlap policy.
40. **Field grades 4 and 9 are absent.** Full-spectrum field validity and the 4/5 boundary are not demonstrated.
41. **Few-shot transfer can be contaminated or unstable.** Document sampling, train/calibration/test disjointness, near duplicates, seeds, repeated budgets, and uncertainty bars.
42. **No independent external site is tested.** Consider another location, year, camera, breeding program, or natural infestation cohort.
43. **Artificial infestation differs from natural injury.** Controlled 30-neonate conditions do not represent heterogeneous pest timing, density, or mixed damage.
44. **Scope is whorl-stage injury.** Avoid extrapolation to tassel/ear injury, other stages, or other pests.
45. **Capture protocol needs detail.** Camera, resolution, distance, lighting, angle, compression, orientation, view selection, and photographs per plant/timepoint.
46. **Explain 392 × 392 resolution.** Consider 224/392/518 sensitivity to small lesions and pinholes.
47. **Augmentation claims need ablation.** Compare no augmentation, geometric only, mild photometric, and strong photometric under corrected physical-plant splits.
48. **Background randomization needs quantitative results.** Report ExG/GrabCut and white/blur/random background experiments, including negative outcomes.
49. **Combined imbalance corrections need justification.** Weighted sampling, effective-number weights, and label smoothing may overcompensate; assess separately.
50. **Explain β = 0.999.** Provide provenance or tuning rationale.
51. **Explain LoRA insertion and settings.** qkv/proj, rank 16, alpha 32, dropout 0.05; consider rank and insertion sensitivity.
52. **Frozen CNN competitors may be weak baselines.** Include at least one end-to-end adapted CNN and full-fine-tuned DINOv2.
53. **Separate representation advantage from adaptation advantage.** Fairly compare frozen backbones and adapted backbones within explicit experiment families.
54. **Majority prediction is a weak baseline.** Add regression, shallow ML, handcrafted features, or an appropriate simple task-specific model.
55. **Why classification rather than scalar regression?** Compare MSE/Huber regression, rounding/clipping to Davis grades, against ordinal classification using QWK and MAE.
56. **Could damage-area estimation suffice?** Evaluate interpretable image-processing/segmentation damage fraction if suitable labels or masks exist.
57. **Breeder-facing utility lacks direct user evidence.** Use “breeder-oriented output” unless actual usability or decision studies demonstrate utility.
58. **Throughput and labor-saving evidence are missing.** Measure inference latency, hardware, acquisition time, manual scoring time, and processing throughput.
59. **Photography itself may be the bottleneck.** Address framing requirements, real-time operation, operational cost, and total workflow time.
60. **Dozens of views per plant are unrealistic deployment requirements.** Clarify single-image inference; report within-plant prediction dispersion and R/S consistency, and multi-view aggregation.
61. **Unequal view counts distort image-weighted metrics.** Report equal weighting per physical plant and genotype, alongside conventional image-level results.
62. **Show representative errors.** Include adjacent-grade errors, severe errors, boundary crossings, and field failures, with explanations.
63. **Report catastrophic errors.** Count |error| ≥ 3, investigate quality, occlusion, environment, and label visibility.
64. **Break down errors by source.** Greenhouse versus field catastrophic-error rates.
65. **Break down errors by planting/year.** Identify batch-specific failure or confounding.
66. **Break down errors by genotype.** Identify performance gaps across germplasm.
67. **Genotype morphology may confound severity.** Leaf architecture, color, waxiness, vigor; genotype-held-out training is a stronger test.
68. **Calibration may vary by genotype.** Explore if supported by sample size.
69. **Disclose binary prevalence.** Report low-/high-injury proportions and denominators.
70. **Binary metrics may conflict across text and figures.** Reconcile binary-head predictions versus severity-threshold decisions and their exact confusion counts.
71. **The chosen SCAR model is not the highest-QWK variant.** Justify 0.846 product versus 0.852 ordinal based on validated outputs, not severity superiority.
72. **The binary head may not change decisions.** Compare appropriately calibrated ordinal-derived probabilities with calibrated binary-head probabilities.
73. **Calibration comparisons must be fair.** Raw ordinal mass versus calibrated binary outputs is not a fair comparison; fit both using the same disjoint calibration cohort.
74. **Temperature fitting must not reuse evaluation observations.** Show the nested training/calibration/test sequence explicitly.
75. **Explain temperature scaling for ordinal output.** Specify transformations and reconstruction of ordinal class probabilities.
76. **ECE depends on the 15-bin choice.** Assess adaptive ECE or binning sensitivity.
77. **Calibration metrics need uncertainty.** Use appropriate physical-plant bootstrap CIs.
78. **Assess calibration under field shift.** Report field ECE/Brier separately.
79. **Deployment needs an explicit validity boundary.** Clarify actual app rejection behavior for non-maize, wrong stage, blur, other pests, or unsupported conditions; an optional VLM gate does not establish app-wide validation.
80. **“Available from authors” weakens reproducibility.** Prefer persistent repositories, DOI, weights, manifests, environment, seeds, exact commits, and metadata schema.
81. **Reconcile “released” with access-on-request wording.** Use one accurate availability statement.
82. **Specify licensing and redistribution rights.** Clarify USDA imagery access and metadata restrictions.
83. **Record implementation details.** PyTorch/timm versions, GPU, workers, seeds, preprocessing, scheduler, checkpoint selection, inference.
84. **Explain epoch/checkpoint selection.** Was validation QWK used, fixed epoch count, or early stopping? The held-out test fold must not select checkpoints.
85. **Field-adaptation methods need full detail.** Make sampling and training as reproducible as the headline CV.
86. **Few-shot uncertainty is missing.** Quantify variability across sampled support sets and seeds.
87. **Statistical gains need agronomic meaning.** Interpret +0.033 QWK in terms of grade error and boundary decisions.
88. **Large fold-based Cohen's d values need explanation.** Supplement standardized fold differences with cluster-bootstrap ΔQWK intervals.
89. **Define the Holm correction family.** State all included comparisons.
90. **Choose a primary test.** Corrected resampled t-test can be primary; Wilcoxon a sensitivity analysis. Avoid opportunistic interpretation of whichever is significant.
91. **“Modest but reliable” overstates a nonsignificant primary test.** Prefer “directionally consistent but not significant under the primary corrected test.”
92. **Abstract significance claims must match comparisons.** Specify the model family and correction used.
93. **Literature novelty claims need support.** Explain search scope and avoid universal “first” claims unsupported by the reviewed evidence.
94. **Comparison tables may look selective.** State inclusion criteria; distinguish unrelated ordinal papers from directly comparable crop-injury studies; avoid a marketing-style checklist.
95. **Tangential in-review SAGE self-citation may be removable.** Keep only if it contributes necessary context.
96. **“Adaptive Ranking” needs a natural method meaning.** Clarify what is adapted and ranked, and whether a ranking loss exists.
97. **Reconcile dataset years.** Distinguish image acquisition years from broader agronomic trial history.
98. **Define replicated trials.** Separate biological replicates, technical views, plantings, field blocks, and design randomization.
99. **Add agronomic experimental detail.** Conditions, photoperiod, temperature, planting layout, genotype replication, infestation, age, nutrients, blocks, seasons.
100. **Eight genotypes limit breeding claims.** Frame as a methodological phenotyping proof of concept.
101. **Longitudinal scores are correlated.** Explain growth morphology, correlation of 7/14 DAI labels, and grouping across timepoints.
102. **Longitudinal prediction is a possible extension.** Predict 14 DAI injury from 7 DAI or model paired images, without treating this as required evidence for the present task.
103. **Timepoint sample imbalance matters.** Report 7/14 DAI image and biological-unit counts and performance.
104. **Justify the 1–4/5–9 binary cutoff.** Cite relevant practice or experimental protocol; avoid claiming universality.
105. **Show Davis rubric semantics.** Provide grade descriptions alongside representative imagery.
106. **Examples may expose acquisition artifacts.** Inspect sample grids for systematic pots, tags, hands, backgrounds, and lighting by grade.
107. **Saliency should include successes and failures.** Useful descriptive evidence, without causal claims.
108. **Background-only intervention is stronger than saliency.** Prioritize direct shortcut testing.
109. **Audit duplicates and near duplicates.** Use exact hashes, perceptual hashes, or embedding neighbors, with verification across groups.
110. **Ensure augmented variants never cross splits.** State online training-only augmentation.
111. **Folders and filenames encode labels.** Verify the model consumes pixels only and metadata cannot leak through preprocessing.
112. **Tags remaining in valid photographs can leak information.** Tag-only removal is insufficient; test tag masking.
113. **Hands/rulers and other objects may be shortcuts.** Include these in the capture-cue audit.
114. **Test cropped leaf/whorl images.** Compare against whole-image inputs with a validated mask/crop method.
115. **Non-destructive imaging is not unique over visual scoring.** Avoid presenting a shared property as a demonstrated advantage.
116. **Local labels may reduce scalability.** Discuss amortized local adaptation rather than implying universal zero-label deployment; incorporate the completed zero-shot LoRA result.
117. **Adaptation frequency is unknown.** Per farm/year/camera requirements remain untested.
118. **Few-shot adaptation may be a central contribution.** Elevate the quantified frozen-versus-LoRA transfer experiment while preserving its uncertainty limitations.
119. **Explain unsuccessful CORAL alignment.** Distinguish feature-covariance alignment from CORAL ordinal classification; document layer, preprocessing, data, and method.
120. **Self-supervised/test-time adaptation are future options.** Consider self-training or entropy minimization without suggesting they were tested.
121. **The paper may be diffuse.** Center grouped biological data, ordinal scoring, adaptation, robust evaluation, field shift, and decision outputs; move VLM/app/extended analyses to supplement.
122. **INSIGHT boxes may conflict with journal style.** Check norms before retaining editorial presentation elements.
123. **Figure 1 can make the VLM look central.** Remove or clearly mark as exploratory/supplementary.
124. **Presence detection is not a severity baseline.** Use its contrast for task definition, not performance superiority.
125. **Make hypotheses explicit.** Separate representation, adaptation, ordinal loss, decision calibration, and field adaptation hypotheses.
126. **Label exploratory analyses honestly.** Distinguish prespecified comparisons from analyses added after observing results.
127. **Provide per-fold values.** Include metrics and predictions in supplementary/reproducible artifacts.
128. **Fine-tuning seeds add variance beyond CV.** Repeat representative splits or all CV across seeds if feasible.
129. **Separate fold and seed variance.** Do not imply one captures the other.
130. **Binary metric differences need CIs.** Use matched cluster resampling where identity is known.
131. **Rare-grade performance needs biological-unit uncertainty.** Include unique plant support and cluster intervals.
132. **Image bootstrap is inappropriate for repeated greenhouse views.** Use plant clusters; disclose that field clustering cannot be recovered.
133. **Genotype inference has only eight genotype units.** Plant resampling conditional on fixed genotypes does not establish a broader genotype population.
134. **Explain the genotype-rank interval.** Clarify what is resampled and avoid overly precise generalization claims.
135. **Deleting one item preserves an already perfect ranking easily.** Leave-one-genotype-out rank analysis is not unseen-genotype learning.
136. **Document image label inheritance.** Explain one consensus plant-timepoint score assigned to many views.
137. **Inherited labels may be invisible in a particular view.** Audit injury visibility and acknowledge image-label noise.
138. **Multi-view aggregation may be more biologically valid.** Evaluate a few photos per plant, median predictions, and diminishing returns.
139. **Why train at image rather than plant level?** Discuss multi-instance modeling as an extension if not evaluated.
140. **Displaying probability 1.00 may imply certainty.** Explain rounding and calibration limits in the app.
141. **Rare low-injury calibration matters.** Report classwise behavior where support allows.
142. **Describe calibration imbalance.** Report label support and weighting policy.
143. **Alternative calibration methods are possible.** Isotonic/Dirichlet/vector scaling can be future or comparative work, without inflated scope.
144. **Benchmark-selected model estimates are not untouched selection estimates.** Clarify outcome-driven choice of DINOv2/LoRA and limits of internal CV.
145. **Separate inherited from tuned hyperparameters.** Record each setting's provenance.
146. **No locked final cohort exists.** Consider independent planting/year/site; do not rebrand a repeatedly inspected cohort as untouched.
147. **The field cohort could help but lacks plant identity.** Keep the identity caveat attached to field conclusions.
148. **Internal versus external validation should be explicit.** Use consistent terminology throughout.
149. **Conclusion must not imply replacement readiness.** A step toward higher-throughput screening is defensible; demonstrated expert replacement is not.
150. **Make Plant Phenomics fit explicit.** Center biology, units of inference, breeding decisions, acquisition practicalities, and field validity over generic AI language.

## Fifteen most likely reviewer questions

1. How robust is a 5,036-image study when 84 plants underlie 4,737 greenhouse images?
2. What are plant-level rather than image-level results?
3. Does training generalize to an entirely unseen genotype?
4. Why call field evaluation leakage-free without physical plant identities?
5. How does the headline pooled QWK compare with genuine frozen and LoRA field transfer?
6. Where is independent external validation?
7. Why compare adapted DINOv2 primarily with frozen competitors?
8. Where are full-fine-tuning and parameter/memory/time comparisons?
9. How were loss weights selected and components validated?
10. Why Frank–Hall rather than CORAL/CORN/regression?
11. Where is human–human agreement?
12. Were the two raters independent despite identical shared records?
13. Why call ±1 calibrated if it is a fixed tolerance band?
14. Could tags/backgrounds/sessions/genotype morphology explain performance?
15. What demonstrates breeder utility or throughput improvement?

## Experiments requested before submission, ranked

1. Plant-level and plant × timepoint evaluation, equal biological-unit weighting, cluster CIs.
2. Genuine genotype-held-out learning: train on seven, test unseen eighth, repeat eight times.
3. Full DINOv2 fine-tuning against LoRA, with parameters/VRAM/time.
4. Shortcut audit: tag masking, plant crop, background-only inputs.
5. Independent human re-scoring: human–human and model–human agreement.
6. Loss-component ablations: CE/ordinal/distance/binary.
7. Fair calibration comparison with the same independent calibration observations for each output.
8. Conformal prediction, or consistently call ±1 a tolerance band.
9. Repeated few-shot field adaptation with fixed held-out test set and uncertainty.
10. Leave-one-planting-out / cross-session evaluation.
11. True 7→14 and 14→7 transfer, distinguishing same-plant longitudinal from unseen-plant temporal transfer.
12. Within-plant view consistency and few-view aggregation.
13. At least one strong adapted ConvNeXt/ResNet comparator.
14. Augmentation ablation using physical-plant splits.
15. Inference and total-workflow throughput analysis.

## Overall assessment

The study has unusually good awareness of clustering, ordinal evaluation, domain shift, and statistical correction. Those admissions are valuable, but not substitutes for the requested evidence.

> The model engineering is competent, but much of the apparent sample size arises from repeated images of a small biological cohort; validation is largely internal; unseen-genotype generalization is not established; field transfer and adaptation estimates have unresolved identity and uncertainty limitations; and breeder-oriented calibration/deployment has not yet been validated operationally.

Neutralize plant-level evaluation, genotype-held-out learning, adaptation fairness, shortcut tests, and honest uncertainty terminology first. Track evidence and unfinished work in `REVISION_STATUS.md`; do not treat reviewer suggestions as completed experiments or mandatory claims to defend.
