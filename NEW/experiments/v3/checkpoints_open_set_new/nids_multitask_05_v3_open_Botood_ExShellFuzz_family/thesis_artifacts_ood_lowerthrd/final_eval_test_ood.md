# Evaluation Summary: test_ood

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
- Current threshold: 0.500000
- Known threshold: 0.401270
- Future thresholds by horizon: {'10m': 0.13389787077903748, '15m': 0.2201424390077591}
- Unknown-risk threshold: 0.003000
- Validation thresholds stored in checkpoint: {'current': 0.9266373515129089, 'known': 0.4012698531150818, 'future': {'10m': 0.13389787077903748, '15m': 0.2201424390077591}, 'ood': 0.4884459674358368}
- CLI threshold overrides: {'current': 0.5, 'known': None, 'future': None, 'ood': 0.003}

## Present Detection
- PR-AUC: 0.969299
- AUC: 0.353647
- Precision: 0.985530
- Recall: 0.969678
- F1: 0.977540
- Benign FPR: 0.958549
- Brier score: 0.197617
- ECE (10 bins): 0.426147
- PR-AUC 95% bootstrap CI: [0.963595, 0.974555] (valid bootstrap samples=1000)
- Recall 95% bootstrap CI: [0.966700, 0.972464] (valid bootstrap samples=1000)
- F1 95% bootstrap CI: [0.975707, 0.979279] (valid bootstrap samples=1000)

## Novelty And Unknown-Risk
- Decision score: raw unknown-head probability only
- Reconstruction novelty score: combined_mae_mfm_reconstruction_mse
- Explicit unknown head available in checkpoint: True
- Unknown-head supervision active during training: True
- Unknown-labelled positives on this split: 12994
- PR-AUC: 0.999975
- AUC: 0.998332
- Precision: 1.000000
- Recall: 0.838310
- F1: 0.912044
- Benign-and-known FPR: 0.000000
- Brier score: 0.972103
- ECE (10 bins): 0.978701
- PR-AUC 95% bootstrap CI: [0.999966, 0.999984] (valid bootstrap samples=1000)
- Recall 95% bootstrap CI: [0.831451, 0.844766] (valid bootstrap samples=1000)
- F1 95% bootstrap CI: [0.907970, 0.915852] (valid bootstrap samples=1000)
- Raw unknown-head PR-AUC: 0.999975
- Raw unknown-head recall: 0.838310
- Raw unknown-head PR-AUC 95% bootstrap CI: [0.999965, 0.999984] (valid bootstrap samples=1000)
- Reconstruction-only novelty PR-AUC: 0.990072
- Reconstruction-only novelty recall: 1.000000
- Reconstruction-only novelty PR-AUC 95% bootstrap CI: [0.987068, 0.992774] (valid bootstrap samples=1000)

## Future Warning
- Pre-onset exclusion gap minutes: 1.000
- Valid future positives on this split: 40 (raw=51, ignored_near_onset=11)
- Macro PR-AUC: 0.213910
- Macro AUC: 0.503067
- Macro precision: 0.000000
- Macro recall: 0.000000
- Macro F1: 0.000000
- Macro benign FPR: 0.000000
- Macro Brier score: 0.190706
- Macro ECE (10 bins): 0.188452
- Macro PR-AUC 95% bootstrap CI: [0.157460, 0.301782] (valid bootstrap samples=1000)
- Macro recall 95% bootstrap CI: [0.000000, 0.000000] (valid bootstrap samples=1000)
- Macro F1 95% bootstrap CI: [0.000000, 0.000000] (valid bootstrap samples=1000)
- 10m: threshold=0.133898, valid_count=182, valid_pos=30, ignored_near_onset_pos=11, PR-AUC=0.198912, AUC=0.532895, precision=0.000000, recall=0.000000, F1=0.000000, benign_FPR=0.000000, Brier=0.163681, ECE=0.161721, mean_detected_lead=nan
- 10m bootstrap 95% CI: PR-AUC=[0.123389, 0.339697] (valid bootstrap samples=1000), recall=[0.000000, 0.000000] (valid bootstrap samples=1000), F1=[0.000000, 0.000000] (valid bootstrap samples=1000)
- 15m: threshold=0.220142, valid_count=182, valid_pos=40, ignored_near_onset_pos=11, PR-AUC=0.228908, AUC=0.473239, precision=0.000000, recall=0.000000, F1=0.000000, benign_FPR=0.000000, Brier=0.217730, ECE=0.215183, mean_detected_lead=nan
- 15m bootstrap 95% CI: PR-AUC=[0.153015, 0.341033] (valid bootstrap samples=1000), recall=[0.000000, 0.000000] (valid bootstrap samples=1000), F1=[0.000000, 0.000000] (valid bootstrap samples=1000)

## Family Acceptance
- Raw known-family accuracy: n/a
- Accepted known-family accuracy: n/a
- Known-family coverage: n/a
- Family-gate unknown recall: 0.967139