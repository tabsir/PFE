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
- Bootstrap samples: 1000
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
- PR-AUC 95% bootstrap CI: [0.933482, 0.939039] (valid bootstrap samples=1000)
- Recall 95% bootstrap CI: [0.740545, 0.753944] (valid bootstrap samples=1000)
- F1 95% bootstrap CI: [0.850934, 0.859713] (valid bootstrap samples=1000)

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
- PR-AUC 95% bootstrap CI: [0.717483, 0.781929] (valid bootstrap samples=1000)
- Recall 95% bootstrap CI: [0.980749, 0.994203] (valid bootstrap samples=1000)
- F1 95% bootstrap CI: [0.812827, 0.843617] (valid bootstrap samples=1000)
- Raw unknown-head PR-AUC: 0.749851
- Raw unknown-head recall: 0.987678
- Raw unknown-head PR-AUC 95% bootstrap CI: [0.717435, 0.781927] (valid bootstrap samples=1000)
- Reconstruction-only novelty PR-AUC: 0.042225
- Reconstruction-only novelty recall: 0.911848
- Reconstruction-only novelty PR-AUC 95% bootstrap CI: [0.036541, 0.048849] (valid bootstrap samples=1000)

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
- Macro PR-AUC 95% bootstrap CI: [0.167724, 0.194997] (valid bootstrap samples=1000)
- Macro recall 95% bootstrap CI: [0.921305, 0.947182] (valid bootstrap samples=1000)
- Macro F1 95% bootstrap CI: [0.082751, 0.091080] (valid bootstrap samples=1000)
- 10m: threshold=0.133898, valid_count=148577, valid_pos=678, ignored_near_onset_pos=1287, PR-AUC=0.165969, AUC=0.973642, precision=0.040146, recall=0.939528, F1=0.077002, benign_FPR=0.102976, Brier=0.056508, ECE=0.072160, mean_detected_lead=4.194245
- 10m bootstrap 95% CI: PR-AUC=[0.148492, 0.183339] (valid bootstrap samples=1000), recall=[0.920453, 0.957448] (valid bootstrap samples=1000), F1=[0.071629, 0.082576] (valid bootstrap samples=1000)
- 15m: threshold=0.220142, valid_count=148577, valid_pos=834, ignored_near_onset_pos=1287, PR-AUC=0.196575, AUC=0.974085, precision=0.050876, recall=0.929257, F1=0.096471, benign_FPR=0.097859, Brier=0.060137, ECE=0.076629, mean_detected_lead=5.746615
- 15m bootstrap 95% CI: PR-AUC=[0.177526, 0.216382] (valid bootstrap samples=1000), recall=[0.910950, 0.946997] (valid bootstrap samples=1000), F1=[0.090483, 0.102775] (valid bootstrap samples=1000)

## Family Acceptance
- Raw known-family accuracy: 0.214943
- Accepted known-family accuracy: 0.241379
- Known-family coverage: 0.048715
- Family-gate unknown recall: 0.761137