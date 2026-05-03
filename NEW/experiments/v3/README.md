# NIDS Experiment V3

This folder keeps a stronger downstream experiment isolated from the original scripts and from `NEW/experiments/v2`.

## Files

- `src/03_stt_architecture_v3.py`
- `src/05_train_multitask_nids_v3.py`
- `src/06_infer_nids_v3.py`
- `src/07_generate_thesis_artifacts_v3.py`

## Default Output Location

- `checkpoints/nids_multitask_05_v3_full/`

The v3 training script writes the following files into that checkpoint directory:

- `nids_multitask_best.pt`
- `nids_multitask_epoch_N.pt`
- `known_attack_labels.json`
- `v3_experiment_config.json`

## What Changed In V3

- Adds a dedicated OOD or unknown-attack head instead of relying only on the family-confidence gate.
- Calibrates current, known, and OOD thresholds on validation, plus one future threshold per configured horizon.
- Keeps pseudo-zero-day family holdout support for validation realism.
- Rotates additional surrogate unknown families across epochs during training unless you disable it.
- Uses more recall-oriented defaults for present detection.
- Replaces the single future-warning task with multi-horizon early warning defaults at 1, 3, and 5 minutes.
- Uses stronger future-positive weighting to reduce missed early warnings.
- Exports OOD metrics and OOD figures in the thesis artifact generator.

## Important Note

If you want the new OOD head and multi-horizon future head to work, you must train a new v3 checkpoint. An old v2 checkpoint cannot gain those heads without retraining.

## Run Training

```bash
python NEW/experiments/v3/src/05_train_multitask_nids_v3.py \
  --enable-future-task \
  --future-horizons-minutes 1 3 5
```

To choose held-out families explicitly:

```bash
python NEW/experiments/v3/src/05_train_multitask_nids_v3.py \
  --enable-future-task \
  --future-horizons-minutes 1 3 5 \
  --pseudo-zero-day-families "Infiltration" "Botnets"
```

To control the extra surrogate-unknown rotation used during training:

```bash
python NEW/experiments/v3/src/05_train_multitask_nids_v3.py \
  --enable-future-task \
  --future-horizons-minutes 1 3 5 \
  --pseudo-zero-day-rotation-size 2
```

Disable that rotation entirely with `--disable-pseudo-zero-day-rotation`.

## Run Inference

```bash
python NEW/experiments/v3/src/06_infer_nids_v3.py --split test --only-attacks
```

You can also override thresholds from the CLI:

```bash
python NEW/experiments/v3/src/06_infer_nids_v3.py \
  --split test_ood \
  --current-threshold 0.35 \
  --known-threshold 0.70 \
  --future-threshold 0.30 \
  --ood-threshold 0.40
```

Inference always uses the future horizons stored in the checkpoint. The CLI `--future-threshold` override is broadcast to every trained horizon.

## Generate Thesis Artifacts

```bash
python NEW/experiments/v3/src/07_generate_thesis_artifacts_v3.py
```

Default outputs:

- `checkpoints/nids_multitask_05_v3_full/thesis_artifacts/v3_validation_metrics_by_epoch.csv`
- `checkpoints/nids_multitask_05_v3_full/thesis_artifacts/v3_validation_metrics_by_epoch.json`
- `checkpoints/nids_multitask_05_v3_full/thesis_artifacts/final_eval_test.json`
- `checkpoints/nids_multitask_05_v3_full/thesis_artifacts/final_eval_test_ood.json`
- `checkpoints/nids_multitask_05_v3_full/thesis_artifacts/figures/*.png`
- `checkpoints/nids_multitask_05_v3_full/thesis_artifacts/thesis_figure_manifest.md`

The generator now covers:

- validation history curves including OOD metrics
- threshold and loss curves including OOD threshold and OOD loss
- split distribution charts
- attack-family distribution charts
- known-vs-unknown composition charts
- train feature correlation matrix
- feature mean shift across splits
- present detection PR and ROC curves
- OOD or unknown-warning PR and ROC curves
- future warning PR and ROC curves per horizon
- threshold sweeps
- score histograms
- confusion matrices
- reliability diagrams
- future lead-time histograms per horizon
- known-vs-unknown gate tradeoff curves
- family confusion matrices

## Notes

- The original scripts remain untouched.
- The v3 inference script uses the v3 checkpoint directory by default.
- The v3 artifact generator exports the explicitly trained OOD-head metrics instead of only the family-gate unknown recall.
- The checkpoint and artifact defaults now point to `nids_multitask_05_v3_full`, which is the multi-horizon v3 training output.