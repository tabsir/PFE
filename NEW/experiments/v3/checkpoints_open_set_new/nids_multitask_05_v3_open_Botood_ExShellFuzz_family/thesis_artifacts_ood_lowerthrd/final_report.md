# Final Report: `thesis_artifacts_ood_lowerthrd`

## Scope And Operating Point

This report consolidates the evaluation results for the `validation`, `test`, and `test_ood` splits in `thesis_artifacts_ood_lowerthrd`.

- Applied thresholds: current attack = `0.500000`, known-family gate = `0.401270`, unknown-risk = `0.003000`
- Future thresholds: `10m = 0.133898`, `15m = 0.220142`
- Stored validation thresholds inside the checkpoint were much stricter for present detection and zero-day detection (`current = 0.926637`, `ood = 0.488446`)

The main consequence is that this artifact should be read as a recall-oriented operating point: it preserves very high recall for present-attack and zero-day detection, but it accepts more false positives and lowers precision compared with the stricter checkpoint-selected thresholds.

## Executive Summary

The lowered thresholds produce a usable high-recall operating point on `validation` and `test` for both present attack detection and zero-day detection. On `test`, the model reaches `0.9298` recall and `0.7505` F1 for benign-vs-malicious detection, while zero-day detection reaches `0.9991` recall and `0.7264` F1. This is the most balanced split under the chosen thresholds.

The `test_ood` split tells a different story. The explicit zero-day head remains strong there, with `PR-AUC = 0.999975`, `recall = 0.8383`, and `F1 = 0.9120`, but the present benign-vs-malicious head becomes unreliable as a discriminator because `AUC` drops to `0.3536` and benign false-positive rate rises to `0.9585`. In practice, that means the OOD split is being handled mostly by the unknown-risk pathway rather than by the closed-set present-attack pathway.

Family detection is still the weakest closed-set classification task. The model protects unknown-family recall by being highly selective about assigning a known family, which keeps coverage low on `validation` (`0.0366`) and only moderately better on `test` (`0.0775`). On `test_ood`, known-family metrics are correctly `n/a` because the split contains no known-family samples.

Future warning remains the hardest task overall. It is somewhat useful on `test`, where macro recall reaches `0.9344`, but precision stays very low (`0.0455`) and the macro F1 remains only `0.0867`. On `test_ood`, future warning does not trigger any positive predictions at the chosen thresholds, so macro precision, recall, and F1 all collapse to `0.0000`.

## 1. Benign Vs Malicious Detection

This task is strong on `validation` and `test` in ranking terms and stays recall-heavy at the lowered `0.5` decision threshold.

- `validation`: `PR-AUC = 0.9689`, `AUC = 0.9912`, `precision = 0.6090`, `recall = 0.9635`, `F1 = 0.7463`, with benign `FPR = 0.0593`
- `test`: `PR-AUC = 0.9363`, `AUC = 0.9836`, `precision = 0.6292`, `recall = 0.9298`, `F1 = 0.7505`, with benign `FPR = 0.0539`
- `test_ood`: `PR-AUC = 0.9693`, but `AUC = 0.3536` and benign `FPR = 0.9585`, despite `precision = 0.9855`, `recall = 0.9697`, and `F1 = 0.9775`

The `test_ood` result is the critical caveat. The high precision and recall there do not indicate healthy separation between benign and malicious traffic. The very low `AUC` and the `95.85%` benign false-positive rate show that the present-attack head is effectively over-firing on OOD data at this threshold.

## 2. Family Detection

Family detection is evaluated only for known families. The operating pattern is conservative: the model rejects many samples instead of assigning a known family label, which preserves unknown recall but keeps family coverage low.

- `validation`: raw known-family accuracy is `0.0710`; accepted known-family accuracy is `0.1016` with only `0.0366` coverage; family-gate unknown recall is `0.7864`
- `test`: raw known-family accuracy improves to `0.2149`; accepted known-family accuracy improves to `0.4477`; coverage rises to `0.0775`; family-gate unknown recall remains stable at `0.7877`
- `test_ood`: raw and accepted known-family accuracy are `n/a` because there are no known-family samples; family-gate unknown recall rises to `0.9671`

So the family head is useful mainly as a guarded acceptance layer on in-distribution traffic. It works better on `test` than on `validation`, but it still covers only a small fraction of known-family traffic. On `test_ood`, the correct behavior is rejection, and that is what the gate mostly does.

## 3. Zero-Day Detection

Zero-day performance is the best part of this configuration, especially because the deployed unknown-risk decision uses the explicit unknown head (`raw_unknown_head_only`) rather than reconstruction-only novelty.

- `validation`: `PR-AUC = 0.8698`, `AUC = 0.9990`, `precision = 0.3077`, `recall = 0.9903`, `F1 = 0.4695`, benign-and-known `FPR = 0.0022`
- `test`: `PR-AUC = 0.7499`, `AUC = 0.9990`, `precision = 0.5707`, `recall = 0.9991`, `F1 = 0.7264`, benign-and-known `FPR = 0.0048`
- `test_ood`: `PR-AUC = 0.999975`, `AUC = 0.9983`, `precision = 1.0000`, `recall = 0.8383`, `F1 = 0.9120`, benign-and-known `FPR = 0.0000`

Two points matter here. First, the lowered `0.003` unknown-risk threshold pushes recall extremely high on `validation` and `test`, but precision drops, especially on `validation`. Second, on the true OOD split the unknown head is doing exactly what it should do: it cleanly separates held-out families from benign-and-known traffic with essentially perfect ranking and zero observed false positives at the chosen threshold.

## 4. Future Warning

Future warning is still the weakest head. Even where recall is high, precision remains very low, so the task is not yet operating at a practically clean warning point.

- `validation`: macro `PR-AUC = 0.0271`, macro `AUC = 0.7995`, macro `precision = 0.0210`, macro `recall = 0.6014`, macro `F1 = 0.0406`, macro benign `FPR = 0.0821`
- `test`: macro `PR-AUC = 0.1813`, macro `AUC = 0.9739`, macro `precision = 0.0455`, macro `recall = 0.9344`, macro `F1 = 0.0867`, macro benign `FPR = 0.1004`
- `test_ood`: macro `PR-AUC = 0.2139`, macro `AUC = 0.5031`, but macro `precision = 0.0000`, `recall = 0.0000`, and `F1 = 0.0000`

The horizon-level pattern is consistent on `validation` and `test`: the `15m` horizon is slightly better than `10m` in F1 and PR-AUC. On `test_ood`, neither horizon produces any positive detections, and the split has only `40` valid positives after the pre-onset exclusion gap, so that result should be treated as a failure under severe distribution shift and limited support.

## Cross-Split Tables

### Benign Vs Malicious

| Split | PR-AUC | AUC | Precision | Recall | F1 | Benign FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.968867 | 0.991248 | 0.609041 | 0.963512 | 0.746326 | 0.059259 |
| Test | 0.936335 | 0.983613 | 0.629222 | 0.929816 | 0.750541 | 0.053916 |
| Test OOD | 0.969299 | 0.353647 | 0.985530 | 0.969678 | 0.977540 | 0.958549 |

### Family Detection

| Split | Raw Known-Family Accuracy | Accepted Accuracy | Coverage | Family-Gate Unknown Recall |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.070951 | 0.101644 | 0.036625 | 0.786408 |
| Test | 0.214943 | 0.447691 | 0.077491 | 0.787678 |
| Test OOD | n/a | n/a | n/a | 0.967139 |

### Zero-Day Detection

| Split | PR-AUC | AUC | Precision | Recall | F1 | Benign-and-Known FPR | Unknown Positives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.869778 | 0.998978 | 0.307692 | 0.990291 | 0.469505 | 0.002175 | 206 |
| Test | 0.749851 | 0.998961 | 0.570655 | 0.999052 | 0.726396 | 0.004848 | 1055 |
| Test OOD | 0.999975 | 0.998332 | 1.000000 | 0.838310 | 0.912044 | 0.000000 | 12994 |

### Future Warning

| Split | Macro PR-AUC | Macro AUC | Macro Precision | Macro Recall | Macro F1 | Macro Benign FPR | Valid Positives | 10m F1 | 15m F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.027109 | 0.799477 | 0.021001 | 0.601402 | 0.040551 | 0.082102 | 642 | 0.034296 | 0.046807 |
| Test | 0.181272 | 0.973864 | 0.045511 | 0.934392 | 0.086737 | 0.100417 | 834 | 0.077002 | 0.096471 |
| Test OOD | 0.213910 | 0.503067 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 40 | 0.000000 | 0.000000 |

## Final Takeaway

For this `ood_lowerthrd` artifact, the best overall story is that lowering the thresholds successfully turns the model into a high-recall detector for present attacks and zero-day traffic on `validation` and `test`. The strongest result is zero-day detection, especially on `test_ood`, where the unknown head remains highly reliable.

The main limitations are equally clear. Family attribution is still sparse because the model only accepts a small subset of known-family predictions, and future warning remains too low-precision to present as a mature operational capability. The `test_ood` benign-vs-malicious result should also be treated with caution: despite high thresholded precision and recall, the closed-set present-attack head is not robust on OOD traffic at this operating point.