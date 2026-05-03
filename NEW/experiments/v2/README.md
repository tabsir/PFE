# NIDS Experiment V2

This folder keeps a versioned downstream experiment separate from the original `NEW/src/05_train_multitask_nids.py` and `NEW/src/06_infer_nids.py` files.

## Files

- `src/05_train_multitask_nids_v2.py`
- `src/06_infer_nids_v2.py`
- `src/07_generate_thesis_artifacts_v2.py`

## Default Output Location

- `checkpoints/nids_multitask_05_v2/`

The v2 training script writes the following files into that checkpoint directory when you run it:

- `nids_multitask_best.pt`
- `nids_multitask_epoch_N.pt`
- `known_attack_labels.json`
- `v2_experiment_config.json`

## What Changed In V2

- Calibrates the current threshold on validation, like the original script.
- Adds future-threshold calibration instead of keeping future at a fixed 0.50.
- Adds known-threshold calibration for the known-vs-unknown confidence gate.
- Supports pseudo-zero-day validation by holding out one or more attack families from known-family supervision.
- Adds stronger family and future weighting knobs.
- Adds family-aware sampling to reduce collapse onto frequent attack families.

## Run Training

```bash
python NEW/experiments/v2/src/05_train_multitask_nids_v2.py --enable-future-task
```

To choose held-out families explicitly:

```bash
python NEW/experiments/v2/src/05_train_multitask_nids_v2.py \
  --enable-future-task \
  --pseudo-zero-day-families "Infiltration" "Botnets"
```

## Run Inference

```bash
python NEW/experiments/v2/src/06_infer_nids_v2.py --split test --only-attacks
```

## Notes

- The original scripts remain untouched.
- The v2 inference script uses the v2 checkpoint directory by default.
- The thesis-artifact generator rebuilds epoch history, exports evaluation summaries, and creates a broad figure set under `checkpoints/nids_multitask_05_v2/thesis_artifacts/` by default.

## Generate Thesis Artifacts

```bash
python NEW/experiments/v2/src/07_generate_thesis_artifacts_v2.py
```

Default outputs:

- `checkpoints/nids_multitask_05_v2/thesis_artifacts/v2_validation_metrics_by_epoch.csv`
- `checkpoints/nids_multitask_05_v2/thesis_artifacts/final_eval_test.json`
- `checkpoints/nids_multitask_05_v2/thesis_artifacts/final_eval_test_ood.json`
- `checkpoints/nids_multitask_05_v2/thesis_artifacts/figures/*.png`
- `checkpoints/nids_multitask_05_v2/thesis_artifacts/thesis_figure_manifest.md`

The generator currently covers:

- validation history curves
- threshold and loss curves
- split distribution charts
- attack-family distribution charts
- known-vs-unknown composition charts
- train feature correlation matrix
- feature mean shift across splits
- present detection PR and ROC curves
- future warning PR and ROC curves
- threshold sweeps
- score histograms
- confusion matrices
- reliability diagrams
- future lead-time histograms
- known-vs-unknown gate tradeoff curves
- family confusion matrices