# Defense Demo Sequence

## Why 06 Exists

`06_infer_nids.py` is the qualitative inference layer of the pipeline.

- `04_train_foundation.py` learns generic traffic representations.
- `05_train_multitask_nids.py` fine-tunes the downstream detector and produces the official checkpoints and aggregate metrics.
- `06_infer_nids.py` loads the saved best checkpoint and turns model outputs into human-readable alerts window by window.

So 06 is not the main benchmarking script. Its role is to simulate deployment behavior and provide a live demo.

Use 05-derived evaluation reports for official numbers, and use 06 for the live replay.

## Files To Open During The Defense

1. `NEW/checkpoints/nids_multitask_05_with_future/final_eval_test.md`
2. `NEW/checkpoints/nids_multitask_05_with_future/demo_test_detected_attacks.txt`
3. `NEW/checkpoints/nids_multitask_05_with_future/final_eval_test_ood.md`

## What To Say First

Start from the official test report:

- Present PR-AUC: `0.955018`
- Present AUC: `0.986513`
- Precision: `1.000000`
- Recall: `0.727538`
- Benign FPR: `0.000000`

Suggested sentence:

"This is the formal aggregate evaluation on the full standard test split. The model is very strong for present attack detection, with PR-AUC 0.955, AUC 0.987, perfect precision at the transferred validation threshold, and zero benign false positives on test."

## Live Demo Command

Run this in the terminal for the live replay:

```bash
python NEW/src/06_infer_nids.py \
  --checkpoint NEW/checkpoints/nids_multitask_05_with_future/nids_multitask_best.pt \
  --split test \
  --only-attacks \
  --status-filter known_attack \
  --max-sequences 1 \
  --batch-size 128 \
  --num-workers 0
```

This command shows one ground-truth attack window that the model actually predicts as a known attack.

## What To Comment On Live

When the replay prints the sequence, comment on these lines:

1. `Current status: known_attack`
2. `Current attack probability: 0.9560`
3. `Known-family confidence: 0.8429`
4. `Ground truth current label: attack`
5. `Ground truth attack type: Exploits`

Suggested sentence:

"Here the model is not just ranking the sample highly; at the deployed threshold it actively classifies the window as an attack, with 0.956 attack probability. The ground truth is also attack, so this is a true positive in the live replay."

## What Not To Oversell

- The family head is weaker than the binary detector, so do not make the family label the main claim.
- The strongest claim is present attack detection, not precise family attribution.

## If They Ask About Zero-Day Or OOD

Open `NEW/checkpoints/nids_multitask_05_with_future/final_eval_test_ood.md` and say:

- The model still has ranking signal on OOD samples because PR-AUC is high.
- But at the transferred threshold, unknown recall is currently `0.000000`.
- So zero-day detection is not yet a validated final claim.

Suggested sentence:

"For the OOD split, the model still separates positives in score space, but the current deployed threshold and unknown-warning behavior do not yet convert that into operational detections. So I present zero-day handling as promising but still incomplete."

## Backup File

If you do not want to run the live command during the defense, use the saved replay file:

`NEW/checkpoints/nids_multitask_05_with_future/demo_test_detected_attacks.txt`