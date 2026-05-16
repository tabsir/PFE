# Figure Selection For Thesis: `thesis_artifacts_ood_lowerthrd`

## Main Recommendation

Keep `6` figures in the main thesis text and use the family-detection table instead of forcing a weak figure into the results chapter.

### 1. Distribution Shift Across Splits

- Figure file: `figures/feature_mean_shift_across_splits.png`
- Keep in: setup section or the opening of the results chapter
- Why keep it: this is the cleanest figure showing that `test_ood` is not just another held-out split but a genuinely shifted distribution

**Interpretation**

The `test_ood` split shows a visible shift in several dominant traffic features, especially throughput, TCP window maxima, and flow duration. In contrast, `test` stays much closer to `train` for most features. This figure provides the strongest visual justification for why OOD evaluation must be interpreted separately from the in-distribution test set.

**Suggested caption**

Feature mean shift across train, test, and test_ood splits. The held-out-family `test_ood` split exhibits clear deviations in several high-activity traffic features, confirming that the open-set benchmark introduces a real distribution shift beyond ordinary train-test variation.

**Suggested paragraph**

Figure X compares the mean values of the most active continuous traffic features across the `train`, `test`, and `test_ood` splits. The ordinary `test` split remains relatively close to the training distribution, whereas `test_ood` shows clear deviations in throughput-related variables, TCP window maxima, and flow duration features. This confirms that the held-out-family benchmark is not simply a class split, but also a distribution-shifted setting. As a result, performance on `test_ood` should be interpreted as a genuine open-set generalization result rather than a standard in-distribution evaluation.

### 2. Validation Overview Across Epochs

- Figure file: `figures/v3_validation_overview.png`
- Keep in: training / model-selection subsection
- Why keep it: it summarizes how the checkpoint behaved across epochs and explains why the chosen model is stable rather than a lucky single-epoch result

**Interpretation**

The current-attack PR-AUC stays near `0.97` across epochs, the unknown-risk branch remains high and stable, and present detection precision stays at `1.0` with recall near `0.90` under the validation-selected operating point. By contrast, future-warning PR-AUC remains very low, which already signals that future forecasting is the hardest task. The figure supports the conclusion that the final checkpoint is stable on present detection and novelty detection, while future warning remains the unresolved component.

**Suggested caption**

Validation overview across refinement epochs, showing the stability of present-attack detection, unknown-risk detection, and family acceptance metrics, together with the weaker performance of the future-warning task.

**Suggested paragraph**

Figure X summarizes the validation behaviour of the multitask model across the refinement stage. Present-attack detection remains highly stable across epochs, with current PR-AUC near `0.97` and a consistently strong F1 score under the validation-selected threshold. Unknown-risk detection is also stable and remains close to saturation, whereas the future-warning PR-AUC stays much lower throughout training. This pattern indicates that the final checkpoint is not selected because of an isolated spike in one epoch; instead, it reflects a robust model for present detection and zero-day detection, while future warning remains the least mature task.

### 3. In-Distribution Present-Attack Curves

- Figure file: `figures/test_current_curves.png`
- Keep in: main results section
- Why keep it: this is the clearest figure for the benign-vs-malicious task on the normal test split

**Interpretation**

The PR and ROC curves are both strong on `test`, with `PR-AUC = 0.936` and `AUC = 0.984`. The curve shape shows that the model preserves high precision until recall becomes very large, which matches the thresholded result of `precision = 0.629`, `recall = 0.930`, and `F1 = 0.751` at the lowered threshold. This is the right figure to support the claim that present attack detection is reliable in-distribution.

**Suggested caption**

Precision-recall and ROC curves for present attack detection on the in-distribution `test` split. The model achieves strong ranking performance, indicating reliable discrimination between benign and attack traffic under ordinary test conditions.

**Suggested paragraph**

Figure X presents the precision-recall and ROC curves for the benign-vs-malicious task on the in-distribution `test` split. The model achieves `PR-AUC = 0.936` and `ROC-AUC = 0.984`, which indicates strong ranking quality over a wide range of thresholds. This result is consistent with the thresholded operating point reported in the quantitative tables, where the lowered decision threshold emphasizes recall (`0.930`) while maintaining a reasonable F1 score (`0.751`). Overall, the figure shows that the present-attack head remains effective when evaluation stays close to the training distribution.

### 4. Zero-Day Detection On The Held-Out OOD Split

- Figure file: `figures/test_ood_ood_curves.png`
- Keep in: main results section
- Why keep it: this is the single strongest figure in the whole artifact and best supports the open-set thesis claim

**Interpretation**

The unknown-risk head separates OOD attack windows almost perfectly on `test_ood`, with `PR-AUC = 1.000` and `AUC = 0.998`. The PR curve remains essentially flat at precision `~1.0` across almost the entire recall range, which is unusually strong for an open-set detection problem. This is the figure that most directly supports the claim that the model learned a useful zero-day detector.

**Suggested caption**

Precision-recall and ROC curves for unknown-risk detection on the held-out `test_ood` split. The explicit unknown-risk head shows near-perfect ranking performance, demonstrating strong separation between known-or-benign traffic and held-out attack families.

**Suggested paragraph**

Figure X shows the precision-recall and ROC curves for zero-day detection on the held-out-family `test_ood` split. The unknown-risk branch achieves near-perfect ranking performance with `PR-AUC = 0.999975` and `ROC-AUC = 0.998332`, indicating extremely strong separation between unknown-labelled attack traffic and the non-unknown reference set. This result is the clearest evidence that the open-set component of the model generalizes beyond the families used for supervised training. Even under the aggressive recall-oriented threshold setting used in this artifact, the OOD head remains highly reliable and constitutes the strongest result of the full multitask system.

### 5. Failure Of The Closed-Set Present-Attack Head Under OOD Shift

- Figure file: `figures/test_ood_current_curves.png`
- Keep in: main results section immediately after the zero-day figure
- Why keep it: this is the best contrast figure because it shows why the unknown-risk head is necessary

**Interpretation**

Although the PR curve looks superficially strong and `PR-AUC = 0.969`, the ROC curve collapses to `AUC = 0.354`, which is worse than random ranking. This matches the summary metrics showing a benign false-positive rate of `0.9585` on `test_ood`. The figure demonstrates that the ordinary closed-set current-attack head does not transfer reliably under OOD shift, even though the dedicated unknown-risk head does.

**Suggested caption**

Precision-recall and ROC curves for present attack detection on the held-out `test_ood` split. Despite a high PR-AUC, the ROC curve collapses, revealing that the closed-set present-attack head is not robust under OOD shift.

**Suggested paragraph**

Figure X highlights an important failure mode of the closed-set present-attack head on the `test_ood` split. While the precision-recall curve remains high because the split is dominated by attack windows, the ROC-AUC falls to `0.354`, which indicates poor ranking quality and effectively inverted discrimination behaviour under distribution shift. This apparent contradiction is resolved by the confusion statistics, which show a very high benign false-positive rate at the selected threshold. Therefore, the figure demonstrates that strong open-set performance does not come from the present-attack head alone; it depends on the dedicated unknown-risk pathway.

### 6. Future Warning Under OOD Shift

- Figure file: `figures/test_ood_future_15m_curves.png`
- Keep in: limitations subsection or future-warning subsection
- Why keep it: this is the most honest way to show that future forecasting is still weak under OOD conditions

**Interpretation**

The `15m` future-warning curves are clearly much weaker than the present-detection and zero-day curves, with `PR-AUC = 0.229` and `AUC = 0.473`. The ROC curve stays around or below the diagonal, which is consistent with the thresholded result of zero precision, recall, and F1 on `test_ood`. This figure is valuable because it shows a real limitation of the model rather than only its strongest results.

**Suggested caption**

Precision-recall and ROC curves for `15m` future warning on the held-out `test_ood` split. Future forecasting degrades substantially under OOD shift and does not provide a usable warning signal at the selected threshold.

**Suggested paragraph**

Figure X reports the `15m` future-warning curves on the held-out `test_ood` split. In contrast to the strong zero-day detection result, the future-warning head generalizes poorly under this shift, with `PR-AUC = 0.228908` and `ROC-AUC = 0.473239`. The thresholded evaluation confirms this weakness, as the deployed operating point yields zero precision, zero recall, and zero F1 on this split. This result suggests that early warning of future attacks remains the least transferable task in the multitask setting and requires further work in both representation learning and threshold design.

## Family Detection Recommendation

For family detection, the best choice is to keep the **table** from `final_report_tables.tex` and **not** use a main-text figure from this artifact.

- Table file: `final_report_tables.tex`
- Relevant table: family detection
- Why not force a figure: this artifact does not contain a strong known-family confusion matrix for the in-distribution test split, and the available OOD gate plots are mostly rejection diagnostics rather than interpretable family-classification results

**Suggested paragraph for family results**

Family attribution remained the weakest closed-set classification component of the pipeline and is best summarized numerically rather than visually for this artifact. On the `test` split, raw known-family accuracy reached `0.2149`, while accepted known-family accuracy increased to `0.4477` at the cost of low coverage (`0.0775`), showing that the model makes useful family assignments only on a restricted subset of confident cases. On `test_ood`, family metrics are not applicable because the split is composed of held-out attack families, so the correct behaviour is rejection rather than assignment to a known class. For this reason, the family-detection results are more clearly presented as a table than as a main-text figure.

## Appendix-Only Figures

These are worth keeping only in the appendix or supplementary material.

### A. Split Composition

- Figure file: `figures/split_known_unknown_distribution.png`
- Why appendix-only: useful for explaining the benchmark design, but not necessary in the core results flow

**Interpretation**

This figure shows that `test_ood` is composed almost entirely of unknown attack windows, whereas `train` and `test` contain benign and known-attack mixtures. It helps explain why some thresholded metrics behave differently on `test_ood`, especially for the present-attack head.

### B. Family Gate Tradeoff

- Figure file: `figures/test_ood_known_gate_tradeoff.png`
- Why appendix-only: the plot is diagnostically useful, but visually weak for main-text presentation because accepted known accuracy and known coverage are nearly zero across the OOD regime

**Interpretation**

At the selected known-confidence threshold (`0.401`), unknown recall is near `0.97`, while known coverage is effectively suppressed. This confirms that on the OOD split the gate acts primarily as a rejection mechanism rather than a family classifier.

## Short Final Advice

If you want the cleanest thesis presentation, keep the following in the main text:

1. `feature_mean_shift_across_splits.png`
2. `v3_validation_overview.png`
3. `test_current_curves.png`
4. `test_ood_ood_curves.png`
5. `test_ood_current_curves.png`
6. `test_ood_future_15m_curves.png`

For family detection, use the table instead of a figure.