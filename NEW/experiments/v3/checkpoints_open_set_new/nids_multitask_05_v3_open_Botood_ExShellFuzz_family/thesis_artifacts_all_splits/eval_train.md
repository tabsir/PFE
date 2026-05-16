# Evaluation Summary: train

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
- PR-AUC: 0.979209
- AUC: 0.993114
- Precision: 0.999991
- Recall: 0.844443
- F1: 0.915658
- Benign FPR: 0.000001
- Brier score: 0.083625
- ECE (10 bins): 0.234162

## Novelty And Unknown-Risk
- Decision score: raw unknown-head probability only
- Reconstruction novelty score: combined_mae_mfm_reconstruction_mse
- Explicit unknown head available in checkpoint: True
- Unknown-head supervision active during training: True
- Unknown-labelled positives on this split: 3677
- PR-AUC: 0.960738
- AUC: 0.999907
- Precision: 0.862538
- Recall: 1.000000
- F1: 0.926196
- Benign-and-known FPR: 0.000641
- Brier score: 0.000586
- ECE (10 bins): 0.000746
- Raw unknown-head PR-AUC: 0.960738
- Raw unknown-head recall: 1.000000
- Reconstruction-only novelty PR-AUC: 0.024528
- Reconstruction-only novelty recall: 0.863748

## Future Warning
- Pre-onset exclusion gap minutes: 1.000
- Valid future positives on this split: 3096 (raw=6984, ignored_near_onset=3888)
- Macro PR-AUC: 0.338462
- Macro AUC: 0.989771
- Macro precision: 0.042472
- Macro recall: 0.987465
- Macro F1: 0.081367
- Macro benign FPR: 0.078363
- Macro Brier score: 0.044833
- Macro ECE (10 bins): 0.059260
- 10m: threshold=0.133898, valid_count=781820, valid_pos=2373, ignored_near_onset_pos=3888, PR-AUC=0.317761, AUC=0.990264, precision=0.036103, recall=0.989465, F1=0.069664, benign_FPR=0.080426, Brier=0.043324, ECE=0.057243, mean_detected_lead=4.597836
- 15m: threshold=0.220142, valid_count=781820, valid_pos=3096, ignored_near_onset_pos=3888, PR-AUC=0.359163, AUC=0.989279, precision=0.048841, recall=0.985465, F1=0.093069, benign_FPR=0.076300, Brier=0.046342, ECE=0.061277, mean_detected_lead=6.401303

## Family Acceptance
- Raw known-family accuracy: 0.115413
- Raw known-family macro F1: 0.481414
- Accepted known-family accuracy: 0.131115
- Accepted known-family macro F1: 0.388741
- Known-family coverage: 0.058457
- Family-gate unknown recall: 0.920044

### Raw Known-Family Metrics By Label
| Family | Support | Predicted | Precision | Recall | F1 | PR-AUC | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DoS / DDoS | 75149 | 4 | 1.000000 | 0.000053 | 0.000106 | 0.999628 | 0.999666 |
| Brute Force | 35947 | 6 | 0.000000 | 0.000000 | 0.000000 | 0.999912 | 0.999980 |
| Infiltration | 14571 | 12454 | 0.999277 | 0.854094 | 0.920999 | 0.995524 | 0.995942 |
| Web Attacks | 413 | 1087 | 0.379945 | 1.000000 | 0.550667 | 0.983928 | 0.999948 |
| Analysis / Backdoors | 264 | 112613 | 0.002344 | 1.000000 | 0.004678 | 1.000000 | 1.000000 |
| Reconnaissance | 774 | 870 | 0.888506 | 0.998708 | 0.940389 | 0.996001 | 0.999959 |
| Worms / Generic | 874 | 958 | 0.911273 | 0.998856 | 0.953057 | 0.987032 | 0.999944 |

### Accepted Known-Family Metrics By Label
| Family | Support | Accepted support | Coverage | Precision | Recall | F1 | PR-AUC | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DoS / DDoS | 75149 | 6501 | 0.086508 | 0.000000 | 0.000000 | 0.000000 | 0.998172 | 0.994260 |
| Brute Force | 35947 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Infiltration | 14571 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Web Attacks | 413 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Analysis / Backdoors | 264 | 104 | 0.393939 | 0.015892 | 0.393939 | 0.030552 | 1.000000 | 1.000000 |
| Reconnaissance | 774 | 708 | 0.914729 | 0.963265 | 0.914729 | 0.938370 | 0.997066 | 0.999815 |
| Worms / Generic | 874 | 169 | 0.193364 | 0.871134 | 0.193364 | 0.316479 | 0.940806 | 0.999025 |
