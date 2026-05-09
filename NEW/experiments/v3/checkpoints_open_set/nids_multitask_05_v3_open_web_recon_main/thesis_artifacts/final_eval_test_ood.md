# Evaluation Summary: test_ood

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
- PR-AUC: 0.937269
- AUC: 0.000222
- Precision: 0.000000
- Recall: 0.000000
- F1: 0.000000
- Benign FPR: 0.041451
- Brier score: 0.916042
- ECE (10 bins): 0.955627

## Novelty And Unknown-Risk
- Decision score: max(raw unknown-head probability, reconstruction novelty percentile)
- Reconstruction calibration: benign empirical percentile with validation MAE/MFM mask ratios 0.300/0.100
- Reconstruction novelty score: combined_mae_mfm_reconstruction_mse
- Explicit unknown head available in checkpoint: True
- Unknown-head supervision active during training: True
- Unknown-labelled positives on this split: 12994
- PR-AUC: 0.999535
- AUC: 0.983404
- Precision: 0.985364
- Recall: 1.000000
- F1: 0.992628
- Benign-and-known FPR: 1.000000
- Brier score: 0.353233
- ECE (10 bins): 0.586608
- Raw unknown-head PR-AUC: 0.937256
- Raw unknown-head recall: 0.000000
- Reconstruction-only novelty PR-AUC: 0.999535
- Reconstruction-only novelty recall: 1.000000

## Future Warning
- Macro PR-AUC: 0.194925
- Macro AUC: 0.376237
- Macro precision: 0.178539
- Macro recall: 0.479197
- Macro F1: 0.259295
- Macro benign FPR: 0.690234
- Macro Brier score: 0.233613
- Macro ECE (10 bins): 0.226641
- 10m: threshold=0.002018, PR-AUC=0.178509, AUC=0.386553, precision=0.158730, recall=0.487805, F1=0.239521, benign_FPR=0.697368, Brier=0.208609, ECE=0.201987, mean_detected_lead=7.130025
- 15m: threshold=0.003583, PR-AUC=0.211341, AUC=0.365921, precision=0.198347, recall=0.470588, F1=0.279070, benign_FPR=0.683099, Brier=0.258616, ECE=0.251294, mean_detected_lead=8.149822

## Family Acceptance
- Raw known-family accuracy: n/a
- Accepted known-family accuracy: n/a
- Known-family coverage: n/a
- Family-gate unknown recall: 0.000000