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
- Bootstrap samples: 1000
- Current threshold: 0.500000
- Known threshold: 0.401270
- Future thresholds by horizon: {'10m': 0.13389787077903748, '15m': 0.2201424390077591}
- Unknown-risk threshold: 0.003000
- Validation thresholds stored in checkpoint: {'current': 0.9266373515129089, 'known': 0.4012698531150818, 'future': {'10m': 0.13389787077903748, '15m': 0.2201424390077591}, 'ood': 0.4884459674358368}
- CLI threshold overrides: {'current': 0.5, 'known': None, 'future': None, 'ood': 0.003}

## Present Detection
- PR-AUC: 0.968867
- AUC: 0.991248
- Precision: 0.609041
- Recall: 0.963512
- F1: 0.746326
- Benign FPR: 0.059259
- Brier score: 0.088287
- ECE (10 bins): 0.250917
- PR-AUC 95% bootstrap CI: [0.966942, 0.970636] (valid bootstrap samples=1000)
- Recall 95% bootstrap CI: [0.960690, 0.966059] (valid bootstrap samples=1000)
- F1 95% bootstrap CI: [0.741790, 0.750550] (valid bootstrap samples=1000)

## Novelty And Unknown-Risk
- Decision score: raw unknown-head probability only
- Reconstruction novelty score: combined_mae_mfm_reconstruction_mse
- Explicit unknown head available in checkpoint: True
- Unknown-head supervision active during training: True
- Unknown-labelled positives on this split: 206
- PR-AUC: 0.869778
- AUC: 0.998978
- Precision: 0.307692
- Recall: 0.990291
- F1: 0.469505
- Benign-and-known FPR: 0.002175
- Brier score: 0.000237
- ECE (10 bins): 0.000308
- PR-AUC 95% bootstrap CI: [0.813817, 0.921639] (valid bootstrap samples=1000)
- Recall 95% bootstrap CI: [0.975121, 1.000000] (valid bootstrap samples=1000)
- F1 95% bootstrap CI: [0.427420, 0.507907] (valid bootstrap samples=1000)
- Raw unknown-head PR-AUC: 0.869778
- Raw unknown-head recall: 0.990291
- Raw unknown-head PR-AUC 95% bootstrap CI: [0.810229, 0.919218] (valid bootstrap samples=1000)
- Reconstruction-only novelty PR-AUC: 0.004234
- Reconstruction-only novelty recall: 1.000000
- Reconstruction-only novelty PR-AUC 95% bootstrap CI: [0.002499, 0.012948] (valid bootstrap samples=1000)

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
- Macro PR-AUC 95% bootstrap CI: [0.023123, 0.031629] (valid bootstrap samples=1000)
- Macro recall 95% bootstrap CI: [0.573008, 0.629704] (valid bootstrap samples=1000)
- Macro F1 95% bootstrap CI: [0.037424, 0.043502] (valid bootstrap samples=1000)
- 10m: threshold=0.133898, valid_count=192602, valid_pos=480, ignored_near_onset_pos=197, PR-AUC=0.023161, AUC=0.801378, precision=0.017652, recall=0.600000, F1=0.034296, benign_FPR=0.083421, Brier=0.042446, ECE=0.055341, mean_detected_lead=5.116988
- 10m bootstrap 95% CI: PR-AUC=[0.018226, 0.029065] (valid bootstrap samples=1000), recall=[0.558170, 0.644719] (valid bootstrap samples=1000), F1=[0.030705, 0.038098] (valid bootstrap samples=1000)
- 15m: threshold=0.220142, valid_count=192602, valid_pos=642, ignored_near_onset_pos=197, PR-AUC=0.031056, AUC=0.797576, precision=0.024349, recall=0.602804, F1=0.046807, benign_FPR=0.080782, Brier=0.047564, ECE=0.060979, mean_detected_lead=7.104583
- 15m bootstrap 95% CI: PR-AUC=[0.025181, 0.037464] (valid bootstrap samples=1000), recall=[0.565852, 0.638129] (valid bootstrap samples=1000), F1=[0.042303, 0.051369] (valid bootstrap samples=1000)

## Family Acceptance
- Raw known-family accuracy: 0.070951
- Accepted known-family accuracy: 0.101644
- Known-family coverage: 0.036625
- Family-gate unknown recall: 0.786408