# Final Best Test Task Summary

This file combines the saved aggregate test metrics from `final_eval_test.json` with a one-off recomputation of the missing per-family test metrics.

## Thresholds
- Current threshold: 0.926637
- Known-family threshold: 0.401270
- Unknown-risk threshold: 0.488446
- Future threshold 10m: 0.133898
- Future threshold 15m: 0.220142

## Counts
- Sequence count: 164611
- Present-attack positives: 14747
- Unknown positives: 1055
- Known-family samples: 13692
- Valid future positives: 834
- Raw future positives: 2121
- Future positives ignored near onset: 1287

## Benign Vs Malicious
| Metric | Value |
| --- | ---: |
| pr_auc | 0.936335 |
| auc | 0.983613 |
| precision | 1.000000 |
| recall | 0.747338 |
| f1 | 0.855402 |
| false_positive_rate | 0.000000 |
| brier_score | 0.087592 |
| ece | 0.239350 |

## Zero-Day / Unknown Risk
| Metric | Value |
| --- | ---: |
| pr_auc | 0.749851 |
| auc | 0.998961 |
| precision | 0.714678 |
| recall | 0.987678 |
| f1 | 0.829288 |
| false_positive_rate | 0.002543 |
| brier_score | 0.002447 |
| ece | 0.002478 |

## Future Warning Macro
| Metric | Value |
| --- | ---: |
| pr_auc | 0.181272 |
| auc | 0.973864 |
| precision | 0.045511 |
| recall | 0.934392 |
| f1 | 0.086737 |
| false_positive_rate | 0.100417 |
| brier_score | 0.058323 |
| ece | 0.074395 |

### Future By Horizon
| Horizon | PR-AUC | ROC-AUC | Precision | Recall | F1 | Benign FPR | Mean lead (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10m | 0.165969 | 0.973642 | 0.040146 | 0.939528 | 0.077002 | 0.102976 | 4.194245 |
| 15m | 0.196575 | 0.974085 | 0.050876 | 0.929257 | 0.096471 | 0.097859 | 5.746615 |

## Family Classification Overview
| Metric | Value |
| --- | ---: |
| raw_accuracy | 0.214943 |
| raw_macro_f1 | 0.394130 |
| accepted_accuracy | 0.241379 |
| accepted_macro_f1 | 0.375251 |
| known_coverage | 0.048715 |
| unknown_recall | 0.761137 |

## Raw Per-Family Metrics
| Family | Support | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DoS / DDoS | 10100 | 0.272727 | 0.000297 | 0.000593 | 0.950849 | 0.930903 |
| Brute Force | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Infiltration | 2848 | 0.996255 | 0.840590 | 0.911826 | 0.990888 | 0.990674 |
| Web Attacks | 54 | 0.193548 | 1.000000 | 0.324324 | 0.998570 | 0.707745 |
| Analysis / Backdoors | 89 | 0.000870 | 0.101124 | 0.001726 | 0.067304 | 0.014639 |
| Reconnaissance | 253 | 0.665414 | 0.699605 | 0.682081 | 0.810629 | 0.675960 |
| Worms / Generic | 348 | 0.801047 | 0.879310 | 0.838356 | 0.929452 | 0.895177 |

## Accepted Per-Family Metrics
| Family | Accepted Support | Coverage | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DoS / DDoS | 464 | 0.045941 | 0.000000 | 0.000000 | 0.000000 | 0.973320 | 0.968252 |
| Brute Force | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Infiltration | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Web Attacks | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Analysis / Backdoors | 18 | 0.202247 | 0.000000 | 0.000000 | 0.000000 | 0.217514 | 0.016793 |
| Reconnaissance | 141 | 0.557312 | 0.773256 | 0.525692 | 0.625882 | 0.937208 | 0.919154 |
| Worms / Generic | 44 | 0.126437 | 0.666667 | 0.080460 | 0.143590 | 0.776485 | 0.595131 |