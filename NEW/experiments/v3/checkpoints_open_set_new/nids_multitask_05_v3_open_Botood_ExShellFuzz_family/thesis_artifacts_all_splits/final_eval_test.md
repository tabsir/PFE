# Evaluation Summary: test

- Checkpoint: /home/aka/PFE-code/NEW/experiments/v3/checkpoints_open_set_new/nids_multitask_05_v3_open_Botood_ExShellFuzz_family/nids_multitask_best.pt
- Run mode: open_set_benchmark
- Thesis claim policy: open_set_attack_detection_with_held_out_family_benchmark
- Decision policy: two_stage_current_then_novelty
- Novelty score mode: combined_mae_mfm
- Unknown-risk score mode: raw_unknown_head_only
- Task activation: {'current_attack_head_active': True, 'future_head_active': True, 'family_head_active': True, 'unknown_head_active': True, 'reconstruction_auxiliary_active': True, 'reconstruction_novelty_active': True}
- Unknown-head supervision active: True
- Hybrid reconstruction-backed unknown risk: True
- Pseudo-zero-day families: ['Exploits / Shellcode', 'Fuzzers']
- Future horizons: [10, 15]
- Calibration bins: 10
- Current threshold: 0.926637
- Known threshold: 0.401270
- Future thresholds by horizon: {'10m': 0.13389787077903748, '15m': 0.2201424390077591}
- Unknown-risk threshold: 0.488446

## Present Detection
- PR-AUC: 0.936335
- AUC: 0.983613
- Precision: 1.000000
- Recall: 0.747338
- F1: 0.855402
- Benign FPR: 0.000000
- Brier score: 0.087592
- ECE (10 bins): 0.239350

## Novelty And Unknown-Risk
- Decision score: raw unknown-head probability only
- Reconstruction novelty score: combined_mae_mfm_reconstruction_mse
- Explicit unknown head available in checkpoint: True
- Unknown-head supervision active during training: True
- Unknown-labelled positives on this split: 1055
- PR-AUC: 0.749851
- AUC: 0.998961
- Precision: 0.714678
- Recall: 0.987678
- F1: 0.829288
- Benign-and-known FPR: 0.002543
- Brier score: 0.002447
- ECE (10 bins): 0.002478
- Raw unknown-head PR-AUC: 0.749851
- Raw unknown-head recall: 0.987678
- Reconstruction-only novelty PR-AUC: 0.041711
- Reconstruction-only novelty recall: 0.912796

## Future Warning
- Pre-onset exclusion gap minutes: 1.000
- Valid future positives on this split: 834 (raw=2121, ignored_near_onset=1287)
- Macro PR-AUC: 0.181272
- Macro AUC: 0.973864
- Macro precision: 0.045511
- Macro recall: 0.934392
- Macro F1: 0.086737
- Macro benign FPR: 0.100417
- Macro Brier score: 0.058323
- Macro ECE (10 bins): 0.074395
- 10m: threshold=0.133898, valid_count=148577, valid_pos=678, ignored_near_onset_pos=1287, PR-AUC=0.165969, AUC=0.973642, precision=0.040146, recall=0.939528, F1=0.077002, benign_FPR=0.102976, Brier=0.056508, ECE=0.072160, mean_detected_lead=4.194245
- 15m: threshold=0.220142, valid_count=148577, valid_pos=834, ignored_near_onset_pos=1287, PR-AUC=0.196575, AUC=0.974085, precision=0.050876, recall=0.929257, F1=0.096471, benign_FPR=0.097859, Brier=0.060137, ECE=0.076629, mean_detected_lead=5.746615

## Family Acceptance
- Raw known-family accuracy: 0.214943
- Raw known-family macro F1: 0.394130
- Accepted known-family accuracy: 0.241379
- Accepted known-family macro F1: 0.375251
- Known-family coverage: 0.048715
- Family-gate unknown recall: 0.761137

### Raw Known-Family Metrics By Label
| Family | Support | Predicted | Precision | Recall | F1 | PR-AUC | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DoS / DDoS | 10100 | 11 | 0.272727 | 0.000297 | 0.000593 | 0.930903 | 0.950849 |
| Brute Force | 0 | 12 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Infiltration | 2848 | 2403 | 0.996255 | 0.840590 | 0.911826 | 0.990674 | 0.990888 |
| Web Attacks | 54 | 279 | 0.193548 | 1.000000 | 0.324324 | 0.707745 | 0.998570 |
| Analysis / Backdoors | 89 | 10339 | 0.000870 | 0.101124 | 0.001726 | 0.014639 | 0.067304 |
| Reconnaissance | 253 | 266 | 0.665414 | 0.699605 | 0.682081 | 0.675960 | 0.810629 |
| Worms / Generic | 348 | 382 | 0.801047 | 0.879310 | 0.838356 | 0.895177 | 0.929452 |

### Accepted Known-Family Metrics By Label
| Family | Support | Accepted support | Coverage | Precision | Recall | F1 | PR-AUC | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DoS / DDoS | 10100 | 464 | 0.045941 | 0.000000 | 0.000000 | 0.000000 | 0.968252 | 0.973320 |
| Brute Force | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Infiltration | 2848 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Web Attacks | 54 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Analysis / Backdoors | 89 | 18 | 0.202247 | 0.000000 | 0.000000 | 0.000000 | 0.016793 | 0.217514 |
| Reconnaissance | 253 | 141 | 0.557312 | 0.773256 | 0.525692 | 0.625882 | 0.919154 | 0.937208 |
| Worms / Generic | 348 | 44 | 0.126437 | 0.666667 | 0.080460 | 0.143590 | 0.595131 | 0.776485 |
