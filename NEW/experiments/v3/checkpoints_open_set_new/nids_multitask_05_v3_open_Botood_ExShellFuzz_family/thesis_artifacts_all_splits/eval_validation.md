# Evaluation Summary: validation

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
- PR-AUC: 0.968867
- AUC: 0.991248
- Precision: 1.000000
- Recall: 0.900011
- F1: 0.947374
- Benign FPR: 0.000000
- Brier score: 0.088287
- ECE (10 bins): 0.250917

## Novelty And Unknown-Risk
- Decision score: raw unknown-head probability only
- Reconstruction novelty score: combined_mae_mfm_reconstruction_mse
- Explicit unknown head available in checkpoint: True
- Unknown-head supervision active during training: True
- Unknown-labelled positives on this split: 206
- PR-AUC: 0.869778
- AUC: 0.998978
- Precision: 0.793774
- Recall: 0.990291
- F1: 0.881210
- Benign-and-known FPR: 0.000251
- Brier score: 0.000237
- ECE (10 bins): 0.000308
- Raw unknown-head PR-AUC: 0.869778
- Raw unknown-head recall: 0.990291
- Reconstruction-only novelty PR-AUC: 0.004183
- Reconstruction-only novelty recall: 0.815534

## Future Warning
- Pre-onset exclusion gap minutes: 1.000
- Valid future positives on this split: 642 (raw=839, ignored_near_onset=197)
- Macro PR-AUC: 0.027109
- Macro AUC: 0.799477
- Macro precision: 0.021001
- Macro recall: 0.601402
- Macro F1: 0.040551
- Macro benign FPR: 0.082102
- Macro Brier score: 0.045005
- Macro ECE (10 bins): 0.058160
- 10m: threshold=0.133898, valid_count=192602, valid_pos=480, ignored_near_onset_pos=197, PR-AUC=0.023161, AUC=0.801378, precision=0.017652, recall=0.600000, F1=0.034296, benign_FPR=0.083421, Brier=0.042446, ECE=0.055341, mean_detected_lead=5.116988
- 15m: threshold=0.220142, valid_count=192602, valid_pos=642, ignored_near_onset_pos=197, PR-AUC=0.031056, AUC=0.797576, precision=0.024349, recall=0.602804, F1=0.046807, benign_FPR=0.080782, Brier=0.047564, ECE=0.060979, mean_detected_lead=7.104583

## Family Acceptance
- Raw known-family accuracy: 0.070951
- Raw known-family macro F1: 0.426502
- Accepted known-family accuracy: 0.032258
- Accepted known-family macro F1: 0.334367
- Known-family coverage: 0.032246
- Family-gate unknown recall: 0.766990

### Raw Known-Family Metrics By Label
| Family | Support | Predicted | Precision | Recall | F1 | PR-AUC | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DoS / DDoS | 16771 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.977682 | 0.834873 |
| Brute Force | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Infiltration | 1343 | 1163 | 1.000000 | 0.865972 | 0.928172 | 0.991316 | 0.991720 |
| Web Attacks | 99 | 242 | 0.409091 | 1.000000 | 0.580645 | 0.803650 | 0.999326 |
| Analysis / Backdoors | 19 | 16816 | 0.000535 | 0.473684 | 0.001069 | 0.135460 | 0.324007 |
| Reconnaissance | 27 | 33 | 0.666667 | 0.814815 | 0.733333 | 0.863520 | 0.965079 |
| Worms / Generic | 7 | 12 | 0.250000 | 0.428571 | 0.315789 | 0.424864 | 0.763905 |

### Accepted Known-Family Metrics By Label
| Family | Support | Accepted support | Coverage | Precision | Recall | F1 | PR-AUC | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DoS / DDoS | 16771 | 566 | 0.033749 | 0.000000 | 0.000000 | 0.000000 | 0.999942 | 0.998540 |
| Brute Force | 0 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Infiltration | 1343 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Web Attacks | 99 | 0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| Analysis / Backdoors | 19 | 6 | 0.315789 | 0.007042 | 0.210526 | 0.013629 | 0.669585 | 0.676387 |
| Reconnaissance | 27 | 15 | 0.555556 | 0.736842 | 0.518519 | 0.608696 | 0.952504 | 0.998606 |
| Worms / Generic | 7 | 2 | 0.285714 | 0.500000 | 0.142857 | 0.222222 | 0.501307 | 0.512777 |
