# Evaluation Summary: validation

- Checkpoint: /home/aka/PFE-code/NEW/experiments/v3/checkpoints_open_set/nids_multitask_05_v3_open_web_recon_main/nids_multitask_best.pt
- Run mode: open_set_benchmark
- Thesis claim policy: open_set_attack_detection_with_held_out_family_benchmark
- Decision policy: two_stage_current_then_novelty
- Novelty score mode: combined_mae_mfm
- Unknown-risk score mode: hybrid_max_raw_unknown_head_and_reconstruction_percentile
- Task activation: {'current_attack_head_active': True, 'future_head_active': True, 'family_head_active': True, 'unknown_head_active': True, 'reconstruction_auxiliary_active': True, 'reconstruction_novelty_active': True}
- Unknown-head supervision active: True
- Hybrid reconstruction-backed unknown risk: True
- Pseudo-zero-day families: ['Web Attacks', 'Reconnaissance']
- Future horizons: [10, 15]
- Calibration bins: 10
- Current threshold: 0.664229
- Known threshold: 0.943616
- Future thresholds by horizon: {'10m': 0.002018195576965809, '15m': 0.0035831190180033445}
- Unknown-risk threshold: 0.015170

## Present Detection
- PR-AUC: 0.978415
- AUC: 0.992868
- Precision: 0.956714
- Recall: 0.950032
- F1: 0.953361
- Benign FPR: 0.004118
- Brier score: 0.047070
- ECE (10 bins): 0.149505

## Novelty And Unknown-Risk
- Decision score: max(raw unknown-head probability, reconstruction novelty percentile)
- Reconstruction calibration: benign empirical percentile with validation MAE/MFM mask ratios 0.300/0.100
- Reconstruction novelty score: combined_mae_mfm_reconstruction_mse
- Explicit unknown head available in checkpoint: True
- Unknown-head supervision active during training: True
- Unknown-labelled positives on this split: 126
- PR-AUC: 0.110062
- AUC: 0.406814
- Precision: 0.000628
- Recall: 1.000000
- F1: 0.001256
- Benign-and-known FPR: 0.949253
- Brier score: 0.308638
- ECE (10 bins): 0.463962
- Raw unknown-head PR-AUC: 0.139984
- Raw unknown-head recall: 0.309524
- Reconstruction-only novelty PR-AUC: 0.000749
- Reconstruction-only novelty recall: 1.000000

## Future Warning
- Macro PR-AUC: 0.032554
- Macro AUC: 0.829529
- Macro precision: 0.027936
- Macro recall: 0.500667
- Macro F1: 0.052888
- Macro benign FPR: 0.068779
- Macro Brier score: 0.016525
- Macro ECE (10 bins): 0.018189
- 10m: threshold=0.002018, PR-AUC=0.028984, AUC=0.826228, precision=0.024890, recall=0.500739, F1=0.047423, benign_FPR=0.069128, Brier=0.015998, ECE=0.017672, mean_detected_lead=3.331631
- 15m: threshold=0.003583, PR-AUC=0.036123, AUC=0.832829, precision=0.030983, recall=0.500596, F1=0.058354, benign_FPR=0.068431, Brier=0.017052, ECE=0.018705, mean_detected_lead=5.115675

## Family Acceptance
- Raw known-family accuracy: 0.082525
- Accepted known-family accuracy: 0.850877
- Known-family coverage: 0.006214
- Family-gate unknown recall: 0.269841