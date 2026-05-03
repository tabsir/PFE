# Evaluation Summary: validation

- Checkpoint: NEW/experiments/v2/checkpoints/nids_multitask_05_v2_run1/nids_multitask_best.pt
- Pseudo-zero-day families: ['DoS / DDoS']
- Current threshold: 0.978673
- Known threshold: 0.329343
- Future threshold: 0.000023

## Present Detection
- PR-AUC: 0.979546
- AUC: 0.993437
- Precision: 1.000000
- Recall: 0.850043
- F1: 0.918944
- Benign FPR: 0.000000

## Future Warning
- PR-AUC: 0.021430
- AUC: 0.836276
- Precision: 0.009567
- Recall: 0.700224
- F1: 0.018876
- Benign FPR: 0.168457

## Family And Open-Set
- Raw known-family accuracy: 0.941799
- Accepted known-family accuracy: 0.717172
- Known-family coverage: 0.116402
- Unknown-warning recall: 0.800012