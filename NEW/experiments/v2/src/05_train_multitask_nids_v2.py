import argparse
import json
import math
import os
import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
NEW_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = NEW_DIR / "src"


def load_source_module(module_name, filename):
    module_path = SRC_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base_train = load_source_module("base_train_multitask_nids", "05_train_multitask_nids.py")
st_data_loader = load_source_module("st_data_loader", "02_st_data_loader.py")
stt_architecture = load_source_module("stt_architecture", "03_stt_architecture.py")

SpatioTemporalNIDSDataset = st_data_loader.SpatioTemporalNIDSDataset
SpatioTemporalTransformer = stt_architecture.SpatioTemporalTransformer
NIDSMultiTaskModel = stt_architecture.NIDSMultiTaskModel

DEFAULT_TRAIN_DIR = str(NEW_DIR / "data" / "nids_src_grouped" / "train")
DEFAULT_VALID_DIR = str(NEW_DIR / "data" / "nids_src_grouped" / "validation")
DEFAULT_TEST_DIR = str(NEW_DIR / "data" / "nids_src_grouped" / "test")
DEFAULT_STATS_PATH = str(NEW_DIR / "nids_normalization_stats.json")
DEFAULT_DOWNSTREAM_CHECKPOINT_DIR = str(EXPERIMENT_DIR / "checkpoints" / "nids_multitask_05_v2")
DEFAULT_FOUNDATION_CHECKPOINT = str(NEW_DIR / "checkpoints" / "stt_best.pt")
DEFAULT_MIN_KNOWN_ATTACK_COUNT = base_train.DEFAULT_MIN_KNOWN_ATTACK_COUNT
DEFAULT_TRAIN_TARGET_POSITIVE_RATE = base_train.DEFAULT_TRAIN_TARGET_POSITIVE_RATE
DEFAULT_THRESHOLD_TARGET_RECALL = base_train.DEFAULT_THRESHOLD_TARGET_RECALL
DEFAULT_FUTURE_THRESHOLD_TARGET_RECALL = 0.70
DEFAULT_KNOWN_TARGET_UNKNOWN_RECALL = 0.80
DEFAULT_UNKNOWN_FAMILY_LOSS_WEIGHT = base_train.DEFAULT_UNKNOWN_FAMILY_LOSS_WEIGHT
DEFAULT_FAMILY_LOSS_WEIGHT = 1.00
DEFAULT_FUTURE_LOSS_WEIGHT = 1.00
DEFAULT_FAMILY_SAMPLER_POWER = 0.50
DEFAULT_FUTURE_POSITIVE_BOOST = 1.00
DEFAULT_FUTURE_HORIZON_MINUTES = 5
DEFAULT_PSEUDO_ZERO_DAY_FAMILY_COUNT = 1

CAT_VOCABS = [256, 256, 256, 65536, 65536, 256, 256, 256, 256]
PERCENTILE_CANDIDATES = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the v2 downstream multitask NIDS model with calibrated thresholds."
    )
    parser.add_argument(
        "--foundation-checkpoint",
        default=DEFAULT_FOUNDATION_CHECKPOINT,
        help="Foundation checkpoint used to initialize the backbone.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_DOWNSTREAM_CHECKPOINT_DIR,
        help="Directory where v2 checkpoints and metadata will be written.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Checkpoint file or output directory to resume from.",
    )
    parser.add_argument(
        "--min-known-attack-count",
        type=int,
        default=DEFAULT_MIN_KNOWN_ATTACK_COUNT,
        help="Minimum malicious window count required for a family to stay supervised.",
    )
    parser.add_argument(
        "--enable-future-task",
        action="store_true",
        help="Enable the future-attack prediction head.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=32,
        help="Sequence length for downstream windows.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=16,
        help="Stride used to build downstream windows.",
    )
    parser.add_argument(
        "--current-label-rule",
        choices=["any_attack", "last_half_attack"],
        default="last_half_attack",
        help="Rule used to mark a sequence window as malicious for the current attack task.",
    )
    parser.add_argument(
        "--rebuild-caches",
        action="store_true",
        help="Ignore cached sequence windows and downstream targets, then rebuild them.",
    )
    parser.add_argument(
        "--train-target-positive-rate",
        type=float,
        default=DEFAULT_TRAIN_TARGET_POSITIVE_RATE,
        help="Target positive rate seen by the training sampler.",
    )
    parser.add_argument(
        "--threshold-target-recall",
        type=float,
        default=DEFAULT_THRESHOLD_TARGET_RECALL,
        help="Recall target used to pick the current-attack threshold on validation.",
    )
    parser.add_argument(
        "--future-threshold-target-recall",
        type=float,
        default=DEFAULT_FUTURE_THRESHOLD_TARGET_RECALL,
        help="Recall target used to pick the future-warning threshold on validation.",
    )
    parser.add_argument(
        "--known-target-unknown-recall",
        type=float,
        default=DEFAULT_KNOWN_TARGET_UNKNOWN_RECALL,
        help="Target recall for held-out pseudo-zero-day families when calibrating the known gate.",
    )
    parser.add_argument(
        "--unknown-family-loss-weight",
        type=float,
        default=DEFAULT_UNKNOWN_FAMILY_LOSS_WEIGHT,
        help="Weight for the open-set family regularizer on unknown windows.",
    )
    parser.add_argument(
        "--family-loss-weight",
        type=float,
        default=DEFAULT_FAMILY_LOSS_WEIGHT,
        help="Weight for the known-family classification loss.",
    )
    parser.add_argument(
        "--future-loss-weight",
        type=float,
        default=DEFAULT_FUTURE_LOSS_WEIGHT,
        help="Weight for the future-warning loss when the future head is enabled.",
    )
    parser.add_argument(
        "--future-horizon-minutes",
        type=int,
        default=DEFAULT_FUTURE_HORIZON_MINUTES,
        help="Forecasting horizon in minutes used to label future-attack positives.",
    )
    parser.add_argument(
        "--family-sampler-power",
        type=float,
        default=DEFAULT_FAMILY_SAMPLER_POWER,
        help="Exponent used to upweight rare attack families in the sampler.",
    )
    parser.add_argument(
        "--future-positive-boost",
        type=float,
        default=DEFAULT_FUTURE_POSITIVE_BOOST,
        help=(
            "Multiplier applied in the sampler to benign windows that are followed by an attack "
            "within the future horizon. Values above 1.0 make the future head see more rare positives."
        ),
    )
    parser.add_argument(
        "--pseudo-zero-day-families",
        nargs="*",
        default=None,
        help=(
            "Mapped attack families to hold out from known-family supervision. "
            "Leave unset to auto-select overlapping train/validation families."
        ),
    )
    parser.add_argument(
        "--pseudo-zero-day-family-count",
        type=int,
        default=DEFAULT_PSEUDO_ZERO_DAY_FAMILY_COUNT,
        help="How many families to auto-select when --pseudo-zero-day-families is unset. Set to 0 to disable.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of downstream training epochs.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for downstream training.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="DataLoader worker count.",
    )
    return parser.parse_args()


def normalize_attack_family_names(family_names):
    normalized = []
    seen = set()
    for family_name in family_names or []:
        mapped = base_train.map_attack_family(family_name)
        if mapped == "Benign" or mapped in seen:
            continue
        normalized.append(mapped)
        seen.add(mapped)
    return normalized


def select_pseudo_zero_day_families(requested_families, train_counts, valid_counts, family_count):
    if requested_families:
        selected = []
        for family_name in normalize_attack_family_names(requested_families):
            train_count = int(train_counts.get(family_name, 0))
            valid_count = int(valid_counts.get(family_name, 0))
            if train_count == 0 or valid_count == 0:
                print(
                    f"Skipping pseudo-zero-day family '{family_name}' because it does not appear in both train and validation.",
                    flush=True,
                )
                continue
            selected.append(family_name)
        return selected

    if family_count <= 0:
        return []

    candidates = []
    for family_name in sorted(set(train_counts) & set(valid_counts)):
        if family_name == "Benign":
            continue
        train_count = int(train_counts.get(family_name, 0))
        valid_count = int(valid_counts.get(family_name, 0))
        if train_count == 0 or valid_count == 0:
            continue
        candidates.append((valid_count, train_count, family_name))

    candidates.sort(reverse=True)
    return [family_name for _, _, family_name in candidates[:family_count]]


class VariantDownstreamNIDSDataset(base_train.DownstreamNIDSDataset):
    def __init__(self, *args, held_out_attack_families=None, **kwargs):
        self.held_out_attack_families = set(normalize_attack_family_names(held_out_attack_families))
        super().__init__(*args, **kwargs)

    def _build_known_attack_vocab(self, min_known_attack_count):
        attack_counter = Counter()
        for raw_attack_id, current_label in zip(self.sequence_attack_ids, self.sequence_current_labels):
            if current_label == 0:
                continue
            attack_name = base_train.map_attack_family(self.raw_attack_names[int(raw_attack_id)])
            if attack_name == "Benign" or attack_name in self.held_out_attack_families:
                continue
            attack_counter[attack_name] += 1

        known_attack_names = [
            attack_name
            for attack_name in base_train.PROJECT_ATTACK_FAMILY_ORDER
            if attack_name not in self.held_out_attack_families
            and attack_counter.get(attack_name, 0) >= min_known_attack_count
        ]

        for attack_name in sorted(attack_counter):
            if attack_name in base_train.PROJECT_ATTACK_FAMILY_ORDER:
                continue
            if attack_counter[attack_name] >= min_known_attack_count:
                known_attack_names.append(attack_name)

        return {attack_name: idx for idx, attack_name in enumerate(known_attack_names)}


def build_family_balanced_sample_weights(
    dataset,
    target_positive_rate,
    family_sampler_power,
    future_positive_boost=1.0,
):
    sample_weights, observed_positive_rate, effective_positive_rate = (
        base_train.build_target_rate_sample_weights(
            dataset.sequence_current_labels.astype(np.float32),
            target_positive_rate,
        )
    )
    sample_weights = np.asarray(sample_weights, dtype=np.float32)
    balanced_weights = sample_weights.copy()
    family_factors = {}

    if family_sampler_power > 0.0:
        attack_counts = Counter(
            attack_family
            for label_value, attack_family in zip(
                dataset.sequence_current_labels,
                dataset.sequence_attack_families,
            )
            if int(label_value) == 1 and attack_family != "Benign"
        )
        if attack_counts:
            max_count = float(max(attack_counts.values()))
            family_factors = {
                attack_family: float((max_count / count) ** family_sampler_power)
                for attack_family, count in attack_counts.items()
            }

            for idx, (label_value, attack_family) in enumerate(
                zip(dataset.sequence_current_labels, dataset.sequence_attack_families)
            ):
                if int(label_value) == 0:
                    continue
                balanced_weights[idx] *= family_factors.get(attack_family, 1.0)

    benign_mask = np.asarray(dataset.sequence_current_labels == 0, dtype=bool)
    future_positive_mask = benign_mask & np.asarray(dataset.future_attack_targets == 1, dtype=bool)
    if future_positive_boost != 1.0 and future_positive_mask.any():
        balanced_weights[future_positive_mask] *= float(future_positive_boost)

    mean_weight = float(balanced_weights.mean())
    if mean_weight > 0:
        balanced_weights /= mean_weight

    positive_mask = dataset.sequence_current_labels == 1
    effective_positive_rate = float(
        balanced_weights[positive_mask].sum() / max(balanced_weights.sum(), 1e-12)
    )
    benign_weight_total = float(balanced_weights[benign_mask].sum()) if benign_mask.any() else 0.0
    observed_future_positive_rate = float(dataset.future_attack_targets[benign_mask].mean()) if benign_mask.any() else 0.0
    effective_future_positive_rate = (
        float(balanced_weights[future_positive_mask].sum() / max(benign_weight_total, 1e-12))
        if benign_weight_total > 0.0
        else 0.0
    )
    sampler_stats = {
        "future_positive_count": int(future_positive_mask.sum()),
        "observed_future_positive_rate": observed_future_positive_rate,
        "effective_future_positive_rate": effective_future_positive_rate,
    }
    return (
        balanced_weights.astype(np.float32),
        observed_positive_rate,
        effective_positive_rate,
        family_factors,
        sampler_stats,
    )


def build_confidence_candidates(*arrays):
    candidate_parts = [np.linspace(0.0, 1.0, 101)]
    for values in arrays:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            continue
        candidate_parts.append(np.percentile(values, PERCENTILE_CANDIDATES))
        candidate_parts.append(values)
    return np.unique(np.concatenate(candidate_parts))


def select_known_threshold(
    known_current_probabilities,
    known_confidences,
    known_predictions,
    known_targets,
    unknown_current_probabilities,
    unknown_confidences,
    current_threshold,
    target_unknown_recall,
    default_threshold=0.55,
):
    known_current_probabilities = np.asarray(known_current_probabilities, dtype=np.float64)
    known_confidences = np.asarray(known_confidences, dtype=np.float64)
    known_predictions = np.asarray(known_predictions, dtype=np.int64)
    known_targets = np.asarray(known_targets, dtype=np.int64)
    unknown_current_probabilities = np.asarray(unknown_current_probabilities, dtype=np.float64)
    unknown_confidences = np.asarray(unknown_confidences, dtype=np.float64)

    if known_confidences.size == 0 and unknown_confidences.size == 0:
        return {
            "threshold": float(default_threshold),
            "accepted_accuracy": 0.0,
            "known_coverage": 0.0,
            "balanced_score": 0.0,
            "unknown_recall": 0.0,
            "meets_target_unknown_recall": False,
            "selection_policy": "balanced_known_accuracy_at_target_unknown_recall",
            "target_unknown_recall": float(target_unknown_recall),
        }

    candidates = build_confidence_candidates(known_confidences, unknown_confidences)
    selected = None
    fallback = None

    for threshold_value in np.sort(candidates):
        known_gate = (
            (known_current_probabilities >= current_threshold)
            & (known_confidences >= threshold_value)
        )
        accepted_accuracy = float(
            (known_predictions[known_gate] == known_targets[known_gate]).mean()
        ) if known_gate.any() else 0.0
        known_coverage = float(known_gate.mean()) if known_confidences.size else 0.0
        balanced_score = 2.0 * accepted_accuracy * known_coverage / max(
            accepted_accuracy + known_coverage,
            1e-9,
        )

        if unknown_confidences.size:
            unknown_warning = (
                (unknown_current_probabilities >= current_threshold)
                & (unknown_confidences < threshold_value)
            )
            unknown_recall = float(unknown_warning.mean())
        else:
            unknown_recall = 0.0

        record = {
            "threshold": float(threshold_value),
            "accepted_accuracy": accepted_accuracy,
            "known_coverage": known_coverage,
            "balanced_score": float(balanced_score),
            "unknown_recall": unknown_recall,
            "meets_target_unknown_recall": (
                float(unknown_recall) >= float(target_unknown_recall)
                if unknown_confidences.size
                else True
            ),
            "selection_policy": "balanced_known_accuracy_at_target_unknown_recall",
            "target_unknown_recall": float(target_unknown_recall),
        }

        fallback_key = (
            record["unknown_recall"],
            record["balanced_score"],
            record["accepted_accuracy"],
            record["known_coverage"],
            -record["threshold"],
        )
        if fallback is None:
            fallback = record
        else:
            current_fallback_key = (
                fallback["unknown_recall"],
                fallback["balanced_score"],
                fallback["accepted_accuracy"],
                fallback["known_coverage"],
                -fallback["threshold"],
            )
            if fallback_key > current_fallback_key:
                fallback = record

        if not record["meets_target_unknown_recall"]:
            continue

        selected_key = (
            record["balanced_score"],
            record["accepted_accuracy"],
            record["known_coverage"],
            -record["threshold"],
        )
        if selected is None:
            selected = record
        else:
            current_selected_key = (
                selected["balanced_score"],
                selected["accepted_accuracy"],
                selected["known_coverage"],
                -selected["threshold"],
            )
            if selected_key > current_selected_key:
                selected = record

    return selected or fallback


def evaluate_downstream_v2(model, data_loader, device, thresholds):
    model.eval()
    metric_totals = {"total": 0.0, "current": 0.0, "family": 0.0, "future": 0.0, "unknown": 0.0}
    current_probs = []
    current_targets = []
    future_probs = []
    future_targets = []
    future_leads = []
    family_predictions = []
    family_targets = []
    known_confidences = []
    known_current_probs = []
    unknown_confidences = []
    unknown_current_probs = []
    family_head_enabled = False

    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader), desc="Downstream validation", leave=False):
            cont = batch["continuous"].to(device, non_blocking=True)
            cat = batch["categorical"].to(device, non_blocking=True)
            current_target = batch["label"].to(device, non_blocking=True)
            future_target = batch["future_attack"].to(device, non_blocking=True)
            known_attack_target = batch["known_attack_id"].to(device, non_blocking=True)
            unknown_target = batch["unknown_attack_target"].to(device, non_blocking=True)
            future_lead = batch["future_lead_minutes"].to(device, non_blocking=True)

            outputs = model(cont, cat, apply_mfm=False)
            batch_losses = base_train.compute_multitask_losses(
                outputs,
                {
                    "label": current_target,
                    "future_attack": future_target,
                    "known_attack_id": known_attack_target,
                    "unknown_attack_target": unknown_target,
                },
                evaluate_downstream_v2.current_loss_fn,
                evaluate_downstream_v2.family_loss_fn,
                evaluate_downstream_v2.future_loss_fn,
                evaluate_downstream_v2.unknown_family_loss_fn,
                evaluate_downstream_v2.loss_weights,
            )

            for loss_name, loss_value in batch_losses.items():
                metric_totals[loss_name] += float(loss_value.item())

            current_probability = torch.sigmoid(outputs["current_attack_logits"]).cpu().numpy()
            current_target_np = current_target.cpu().numpy()
            current_probs.append(current_probability)
            current_targets.append(current_target_np)

            benign_mask = current_target == 0
            if outputs.get("future_attack_logits") is not None and benign_mask.any():
                future_probability = torch.sigmoid(outputs["future_attack_logits"][benign_mask]).cpu().numpy()
                future_probs.append(future_probability)
                future_targets.append(future_target[benign_mask].cpu().numpy())
                future_leads.append(future_lead[benign_mask].cpu().numpy())

            if outputs["attack_family_logits"] is not None:
                family_head_enabled = True
                family_probability = torch.softmax(outputs["attack_family_logits"], dim=-1).cpu().numpy()
                family_confidence = family_probability.max(axis=-1)
                family_prediction = family_probability.argmax(axis=-1)
                known_target_np = known_attack_target.cpu().numpy()
                unknown_target_np = unknown_target.cpu().numpy()

                known_mask = known_target_np >= 0
                if known_mask.any():
                    family_predictions.append(family_prediction[known_mask])
                    family_targets.append(known_target_np[known_mask])
                    known_confidences.append(family_confidence[known_mask])
                    known_current_probs.append(current_probability[known_mask])

                unknown_mask = unknown_target_np == 1
                if unknown_mask.any():
                    unknown_confidences.append(family_confidence[unknown_mask])
                    unknown_current_probs.append(current_probability[unknown_mask])

    for loss_name in metric_totals:
        metric_totals[loss_name] /= max(len(data_loader), 1)

    current_probabilities = np.concatenate(current_probs) if current_probs else np.array([])
    current_labels = np.concatenate(current_targets) if current_targets else np.array([])
    current_metrics = base_train.compute_binary_metrics(
        current_labels,
        current_probabilities,
        thresholds["current"],
    )

    threshold_target_recall = getattr(
        evaluate_downstream_v2,
        "threshold_target_recall",
        DEFAULT_THRESHOLD_TARGET_RECALL,
    )
    current_selection = base_train.select_threshold_for_target_recall(
        current_labels,
        current_probabilities,
        threshold_target_recall,
    )
    best_current_metrics = base_train.compute_binary_metrics(
        current_labels,
        current_probabilities,
        current_selection["threshold"],
    )
    best_current_metrics.update(current_selection)

    future_probabilities = np.concatenate(future_probs) if future_probs else np.array([])
    future_labels = np.concatenate(future_targets) if future_targets else np.array([])
    future_metrics = base_train.compute_binary_metrics(
        future_labels,
        future_probabilities,
        thresholds["future"],
    )

    future_threshold_target_recall = getattr(
        evaluate_downstream_v2,
        "future_threshold_target_recall",
        DEFAULT_FUTURE_THRESHOLD_TARGET_RECALL,
    )
    future_selection = base_train.select_threshold_for_target_recall(
        future_labels,
        future_probabilities,
        future_threshold_target_recall,
    )
    best_future_metrics = base_train.compute_binary_metrics(
        future_labels,
        future_probabilities,
        future_selection["threshold"],
    )
    best_future_metrics.update(future_selection)

    future_lead_values = np.concatenate(future_leads) if future_leads else np.array([])
    future_hits = (
        (future_probabilities >= best_future_metrics["threshold"])
        & (future_labels == 1)
    )
    mean_future_lead = float(future_lead_values[future_hits].mean()) if future_hits.any() else float("nan")

    family_target_array = np.concatenate(family_targets) if family_targets else np.array([])
    family_prediction_array = np.concatenate(family_predictions) if family_predictions else np.array([])
    raw_known_accuracy = float(
        (family_prediction_array == family_target_array).mean()
    ) if family_target_array.size else float("nan")

    known_confidence_array = np.concatenate(known_confidences) if known_confidences else np.array([])
    known_current_prob_array = np.concatenate(known_current_probs) if known_current_probs else np.array([])
    unknown_confidence_array = np.concatenate(unknown_confidences) if unknown_confidences else np.array([])
    unknown_current_prob_array = np.concatenate(unknown_current_probs) if unknown_current_probs else np.array([])

    if family_head_enabled:
        best_known_metrics = select_known_threshold(
            known_current_probabilities=known_current_prob_array,
            known_confidences=known_confidence_array,
            known_predictions=family_prediction_array,
            known_targets=family_target_array,
            unknown_current_probabilities=unknown_current_prob_array,
            unknown_confidences=unknown_confidence_array,
            current_threshold=best_current_metrics["threshold"],
            target_unknown_recall=getattr(
                evaluate_downstream_v2,
                "known_target_unknown_recall",
                DEFAULT_KNOWN_TARGET_UNKNOWN_RECALL,
            ),
            default_threshold=thresholds["known"],
        )
    else:
        best_known_metrics = {
            "threshold": float(thresholds["known"]),
            "accepted_accuracy": 0.0,
            "known_coverage": 0.0,
            "balanced_score": 0.0,
            "unknown_recall": 0.0,
            "meets_target_unknown_recall": False,
            "selection_policy": "no_family_head",
            "target_unknown_recall": getattr(
                evaluate_downstream_v2,
                "known_target_unknown_recall",
                DEFAULT_KNOWN_TARGET_UNKNOWN_RECALL,
            ),
        }

    return {
        "loss": metric_totals,
        "current": current_metrics,
        "best_current": best_current_metrics,
        "future": future_metrics,
        "best_future": best_future_metrics,
        "known_family_accuracy": raw_known_accuracy,
        "known_family_accepted_accuracy": best_known_metrics["accepted_accuracy"],
        "known_family_coverage": best_known_metrics["known_coverage"],
        "best_known": best_known_metrics,
        "unknown_warning_recall": best_known_metrics["unknown_recall"],
        "mean_future_lead_minutes": mean_future_lead,
    }


def build_validation_rank(validation_metrics, future_task_enabled):
    best_current = validation_metrics["best_current"]
    best_future = validation_metrics["best_future"]
    best_known = validation_metrics["best_known"]
    return (
        float(np.nan_to_num(best_current["pr_auc"], nan=0.0)),
        float(np.nan_to_num(best_current["f1"], nan=0.0)),
        float(np.nan_to_num(best_future["pr_auc"], nan=0.0)) if future_task_enabled else 0.0,
        float(np.nan_to_num(best_known["balanced_score"], nan=0.0)),
        float(np.nan_to_num(best_known["unknown_recall"], nan=0.0)),
        float(np.nan_to_num(best_known["accepted_accuracy"], nan=0.0)),
        float(np.nan_to_num(best_current["auc"], nan=0.0)),
    )


def ensure_variant_checkpoint_dir_compatible(checkpoint_dir, expected_config, device):
    best_checkpoint_path = Path(checkpoint_dir) / "nids_multitask_best.pt"
    if not best_checkpoint_path.exists():
        return

    checkpoint = base_train.load_trusted_checkpoint(best_checkpoint_path, device)
    mismatches = []
    missing_fields = []

    for field_name, expected_value in expected_config.items():
        existing_value = checkpoint.get(field_name)
        if existing_value is None:
            missing_fields.append(field_name)
            continue

        if isinstance(expected_value, list):
            if list(existing_value) != list(expected_value):
                mismatches.append(
                    f"{field_name}={existing_value!r} in {best_checkpoint_path.name} vs requested {expected_value!r}"
                )
        elif existing_value != expected_value:
            mismatches.append(
                f"{field_name}={existing_value!r} in {best_checkpoint_path.name} vs requested {expected_value!r}"
            )

    if mismatches:
        mismatch_summary = "; ".join(mismatches)
        raise ValueError(
            "Output directory already contains v2 checkpoints for a different configuration. "
            "Use a fresh --output-dir or resume the matching run. "
            f"{mismatch_summary}"
        )

    if missing_fields:
        print(
            "Warning: existing best checkpoint is missing v2 configuration metadata "
            f"{missing_fields}; directory compatibility could not be fully verified.",
            flush=True,
        )


def dump_variant_config(checkpoint_dir, config_payload):
    config_path = Path(checkpoint_dir) / "v2_experiment_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as handle:
        json.dump(config_payload, handle, indent=2)


def build_training_checkpoint_payload(
    *,
    epoch,
    model,
    optimizer,
    scheduler,
    foundation_checkpoint,
    checkpoint_dir,
    known_attack_labels,
    pseudo_zero_day_families,
    future_horizon_minutes,
    future_task_enabled,
    seq_len,
    stride,
    current_label_rule,
    train_target_positive_rate,
    threshold_target_recall,
    future_threshold_target_recall,
    known_target_unknown_recall,
    unknown_family_loss_weight,
    family_loss_weight,
    future_loss_weight,
    family_sampler_power,
    future_positive_boost,
    loss_weights,
    thresholds,
    validation_score=None,
    validation_rank=None,
    validation_metrics=None,
    interruption_state=None,
):
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "foundation_checkpoint": foundation_checkpoint,
        "output_dir": checkpoint_dir,
        "known_attack_labels": known_attack_labels,
        "pseudo_zero_day_families": pseudo_zero_day_families,
        "future_horizon_minutes": future_horizon_minutes,
        "future_task_enabled": future_task_enabled,
        "seq_len": seq_len,
        "stride": stride,
        "current_label_rule": current_label_rule,
        "train_target_positive_rate": train_target_positive_rate,
        "threshold_target_recall": threshold_target_recall,
        "future_threshold_target_recall": future_threshold_target_recall,
        "known_target_unknown_recall": known_target_unknown_recall,
        "unknown_family_loss_weight": unknown_family_loss_weight,
        "family_loss_weight": family_loss_weight,
        "future_loss_weight": future_loss_weight,
        "family_sampler_power": family_sampler_power,
        "future_positive_boost": future_positive_boost,
        "loss_weights": loss_weights,
        "thresholds": thresholds,
        "best_threshold": thresholds["current"],
    }

    if validation_score is not None:
        payload["validation_score"] = validation_score
    if validation_rank is not None:
        payload["validation_rank"] = list(validation_rank)
    if validation_metrics is not None:
        payload["validation_metrics"] = validation_metrics
    if interruption_state is not None:
        payload.update(interruption_state)

    return payload


def train_multitask_nids_v2():
    args = parse_args()
    print("Starting downstream NIDS training v2...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = args.epochs
    batch_size = args.batch_size
    seq_len = args.seq_len
    stride = args.stride
    clip_value = 5.0
    warmup_epochs = 2
    freeze_backbone_epochs = 2
    backbone_lr = 5e-5
    head_lr = 2e-4
    weight_decay = 1e-4
    future_horizon_minutes = args.future_horizon_minutes
    min_known_attack_count = args.min_known_attack_count
    future_task_enabled = args.enable_future_task
    num_workers = args.num_workers
    current_label_rule = args.current_label_rule
    rebuild_caches = args.rebuild_caches
    train_target_positive_rate = args.train_target_positive_rate
    threshold_target_recall = args.threshold_target_recall
    future_threshold_target_recall = args.future_threshold_target_recall
    known_target_unknown_recall = args.known_target_unknown_recall
    unknown_family_loss_weight = args.unknown_family_loss_weight
    family_loss_weight = args.family_loss_weight
    future_loss_weight = args.future_loss_weight if future_task_enabled else 0.0
    family_sampler_power = args.family_sampler_power
    future_positive_boost = args.future_positive_boost

    if not 0.0 < train_target_positive_rate < 1.0:
        raise ValueError(
            f"train_target_positive_rate must be in (0, 1), got {train_target_positive_rate}"
        )
    if not 0.0 < threshold_target_recall <= 1.0:
        raise ValueError(
            f"threshold_target_recall must be in (0, 1], got {threshold_target_recall}"
        )
    if not 0.0 < future_threshold_target_recall <= 1.0:
        raise ValueError(
            f"future_threshold_target_recall must be in (0, 1], got {future_threshold_target_recall}"
        )
    if not 0.0 <= known_target_unknown_recall <= 1.0:
        raise ValueError(
            f"known_target_unknown_recall must be in [0, 1], got {known_target_unknown_recall}"
        )
    if unknown_family_loss_weight < 0.0:
        raise ValueError(
            f"unknown_family_loss_weight must be >= 0, got {unknown_family_loss_weight}"
        )
    if family_loss_weight < 0.0:
        raise ValueError(f"family_loss_weight must be >= 0, got {family_loss_weight}")
    if future_loss_weight < 0.0:
        raise ValueError(f"future_loss_weight must be >= 0, got {future_loss_weight}")
    if family_sampler_power < 0.0:
        raise ValueError(f"family_sampler_power must be >= 0, got {family_sampler_power}")
    if future_positive_boost <= 0.0:
        raise ValueError(f"future_positive_boost must be > 0, got {future_positive_boost}")
    if future_horizon_minutes <= 0:
        raise ValueError(f"future_horizon_minutes must be > 0, got {future_horizon_minutes}")

    loss_weights = {
        "current": 2.0,
        "family": family_loss_weight,
        "future": future_loss_weight,
        "unknown": unknown_family_loss_weight,
    }
    thresholds = {
        "current": 0.50,
        "known": 0.55,
        "future": 0.50,
    }

    train_dir = DEFAULT_TRAIN_DIR
    valid_dir = DEFAULT_VALID_DIR
    test_dir = DEFAULT_TEST_DIR
    stats_path = DEFAULT_STATS_PATH
    checkpoint_dir = args.output_dir
    foundation_checkpoint = args.foundation_checkpoint
    resume_checkpoint = args.resume_checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)

    base_expected_config = {
        "seq_len": seq_len,
        "stride": stride,
        "current_label_rule": current_label_rule,
        "future_task_enabled": future_task_enabled,
        "future_horizon_minutes": future_horizon_minutes,
        "train_target_positive_rate": train_target_positive_rate,
        "threshold_target_recall": threshold_target_recall,
        "future_threshold_target_recall": future_threshold_target_recall,
        "known_target_unknown_recall": known_target_unknown_recall,
        "unknown_family_loss_weight": unknown_family_loss_weight,
        "family_loss_weight": family_loss_weight,
        "future_loss_weight": future_loss_weight,
        "family_sampler_power": family_sampler_power,
        "future_positive_boost": future_positive_boost,
        "pseudo_zero_day_families": normalize_attack_family_names(args.pseudo_zero_day_families),
        "pseudo_zero_day_family_count": int(args.pseudo_zero_day_family_count),
    }

    print(f"Foundation checkpoint: {foundation_checkpoint}", flush=True)
    print(f"Downstream output directory: {checkpoint_dir}", flush=True)
    print(f"Future task enabled: {future_task_enabled}", flush=True)
    print(f"Future horizon minutes: {future_horizon_minutes}", flush=True)
    print(f"Window config: seq_len={seq_len}, stride={stride}", flush=True)
    print(f"Current label rule: {current_label_rule}", flush=True)
    print(f"Train target positive rate: {train_target_positive_rate:.3f}", flush=True)
    print(f"Current threshold target recall: {threshold_target_recall:.3f}", flush=True)
    print(f"Future threshold target recall: {future_threshold_target_recall:.3f}", flush=True)
    print(f"Known target unknown recall: {known_target_unknown_recall:.3f}", flush=True)
    print(f"Family loss weight: {family_loss_weight:.3f}", flush=True)
    print(f"Future loss weight: {future_loss_weight:.3f}", flush=True)
    print(f"Unknown family loss weight: {unknown_family_loss_weight:.3f}", flush=True)
    print(f"Family sampler power: {family_sampler_power:.3f}", flush=True)
    print(f"Future positive boost: {future_positive_boost:.3f}", flush=True)
    print(f"Rebuild caches: {rebuild_caches}", flush=True)

    ensure_variant_checkpoint_dir_compatible(checkpoint_dir, base_expected_config, device)

    validation_path = valid_dir if os.path.exists(valid_dir) else test_dir
    if validation_path == test_dir:
        print(
            "Validation split introuvable. Utilisation temporaire du split test comme validation downstream.",
            flush=True,
        )

    print("Loading base train dataset...", flush=True)
    train_base_dataset = SpatioTemporalNIDSDataset(
        arrow_dir_path=train_dir,
        stats_path=stats_path,
        seq_len=seq_len,
        stride=stride,
        clip_value=clip_value,
        rebuild_sequence_cache=rebuild_caches,
    )
    print(f"Base train dataset ready: {len(train_base_dataset)} sequences", flush=True)

    print("Loading base validation dataset...", flush=True)
    valid_base_dataset = SpatioTemporalNIDSDataset(
        arrow_dir_path=validation_path,
        stats_path=stats_path,
        seq_len=seq_len,
        stride=stride,
        clip_value=clip_value,
        rebuild_sequence_cache=rebuild_caches,
    )
    print(f"Base validation dataset ready: {len(valid_base_dataset)} sequences", flush=True)

    print("Inspecting attack-family overlap for pseudo-zero-day selection...", flush=True)
    train_probe_dataset = VariantDownstreamNIDSDataset(
        train_base_dataset,
        future_horizon_minutes=future_horizon_minutes,
        known_attack_to_idx={},
        current_label_rule=current_label_rule,
        rebuild_target_cache=rebuild_caches,
        held_out_attack_families=[],
    )
    valid_probe_dataset = VariantDownstreamNIDSDataset(
        valid_base_dataset,
        future_horizon_minutes=future_horizon_minutes,
        known_attack_to_idx={},
        current_label_rule=current_label_rule,
        rebuild_target_cache=rebuild_caches,
        held_out_attack_families=[],
    )
    pseudo_zero_day_families = select_pseudo_zero_day_families(
        args.pseudo_zero_day_families,
        train_probe_dataset.attack_family_counts,
        valid_probe_dataset.attack_family_counts,
        args.pseudo_zero_day_family_count,
    )
    print(f"Pseudo-zero-day families: {pseudo_zero_day_families}", flush=True)

    print("Preparing downstream train targets...", flush=True)
    train_dataset = VariantDownstreamNIDSDataset(
        train_base_dataset,
        future_horizon_minutes=future_horizon_minutes,
        min_known_attack_count=min_known_attack_count,
        current_label_rule=current_label_rule,
        rebuild_target_cache=rebuild_caches,
        held_out_attack_families=pseudo_zero_day_families,
    )
    print(f"Downstream train dataset ready: {len(train_dataset)} sequences", flush=True)
    print(f"Mapped attack families in train: {dict(train_dataset.attack_family_counts)}", flush=True)
    print(f"Known attack families: {train_dataset.known_attack_names}", flush=True)

    print("Preparing downstream validation targets...", flush=True)
    valid_dataset = VariantDownstreamNIDSDataset(
        valid_base_dataset,
        future_horizon_minutes=future_horizon_minutes,
        known_attack_to_idx=train_dataset.known_attack_to_idx,
        current_label_rule=current_label_rule,
        rebuild_target_cache=rebuild_caches,
        held_out_attack_families=pseudo_zero_day_families,
    )
    print(f"Downstream validation dataset ready: {len(valid_dataset)} sequences", flush=True)
    print(f"Mapped attack families in validation: {dict(valid_dataset.attack_family_counts)}", flush=True)

    dump_variant_config(
        checkpoint_dir,
        {
            **base_expected_config,
            "pseudo_zero_day_families": pseudo_zero_day_families,
            "known_attack_labels": train_dataset.known_attack_names,
            "train_attack_family_counts": dict(train_dataset.attack_family_counts),
            "validation_attack_family_counts": dict(valid_dataset.attack_family_counts),
        },
    )

    attack_vocab_path = os.path.join(checkpoint_dir, "known_attack_labels.json")
    with open(attack_vocab_path, "w") as handle:
        json.dump({"known_attack_labels": train_dataset.known_attack_names}, handle, indent=2)

    sample_weights, observed_positive_rate, effective_positive_rate, family_sampler_factors, sampler_stats = (
        build_family_balanced_sample_weights(
            train_dataset,
            train_target_positive_rate,
            family_sampler_power,
            future_positive_boost,
        )
    )
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )
    print(
        "Train sampler configured: "
        f"observed_positive_rate={observed_positive_rate:.4f}, "
        f"target_positive_rate={train_target_positive_rate:.4f}, "
        f"effective_positive_rate={effective_positive_rate:.4f}",
        flush=True,
    )
    if sampler_stats["future_positive_count"] > 0:
        print(
            "Future-positive benign sampling: "
            f"count={sampler_stats['future_positive_count']}, "
            f"observed_rate={sampler_stats['observed_future_positive_rate']:.6f}, "
            f"effective_rate={sampler_stats['effective_future_positive_rate']:.6f}",
            flush=True,
        )
    if family_sampler_factors:
        print(f"Family sampler factors: {family_sampler_factors}", flush=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    print(f"Train loader ready: {len(train_loader)} batches per epoch", flush=True)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    print(f"Validation loader ready: {len(valid_loader)} batches", flush=True)

    num_cont = len(train_base_dataset.cont_cols)
    backbone = SpatioTemporalTransformer(
        num_cont_features=num_cont,
        cat_vocab_sizes=CAT_VOCABS,
        seq_len=seq_len,
        init_mae=0.10,
        init_mfm=0.00,
    )
    print(f"Loading foundation weights from {foundation_checkpoint}", flush=True)
    base_train.load_foundation_checkpoint(backbone, foundation_checkpoint, device)
    model = NIDSMultiTaskModel(
        backbone=backbone,
        num_known_attack_classes=len(train_dataset.known_attack_names),
        use_future_head=future_task_enabled,
    ).to(device)
    print(f"Downstream model ready on {device}", flush=True)

    benign_mask = train_dataset.sequence_current_labels == 0
    if future_task_enabled:
        future_pos_weight = torch.tensor(
            base_train.build_pos_weight(train_dataset.future_attack_targets[benign_mask]),
            dtype=torch.float32,
            device=device,
        )
    else:
        future_pos_weight = None

    if train_dataset.known_attack_names:
        attack_counter = Counter(
            train_dataset.known_attack_targets[train_dataset.known_attack_targets >= 0].tolist()
        )
        family_weights = np.ones(len(train_dataset.known_attack_names), dtype=np.float32)
        total_known = sum(attack_counter.values())
        for attack_idx, attack_count in attack_counter.items():
            family_weights[attack_idx] = total_known / max(len(train_dataset.known_attack_names) * attack_count, 1)
        family_weight_tensor = torch.tensor(family_weights, dtype=torch.float32, device=device)
    else:
        family_weight_tensor = None

    positive_rate = float(train_dataset.sequence_current_labels.mean())
    focal_alpha = 1.0 - positive_rate
    current_loss_fn = base_train.FocalLoss(alpha=focal_alpha, gamma=3.0)
    future_loss_fn = nn.BCEWithLogitsLoss(pos_weight=future_pos_weight) if future_task_enabled else None
    family_loss_fn = (
        nn.CrossEntropyLoss(weight=family_weight_tensor, label_smoothing=0.1)
        if family_weight_tensor is not None
        else None
    )
    unknown_family_loss_fn = (
        base_train.compute_unknown_family_regularization_loss
        if train_dataset.known_attack_names
        else None
    )

    evaluate_downstream_v2.current_loss_fn = current_loss_fn
    evaluate_downstream_v2.future_loss_fn = future_loss_fn
    evaluate_downstream_v2.family_loss_fn = family_loss_fn
    evaluate_downstream_v2.unknown_family_loss_fn = unknown_family_loss_fn
    evaluate_downstream_v2.loss_weights = loss_weights
    evaluate_downstream_v2.threshold_target_recall = threshold_target_recall
    evaluate_downstream_v2.future_threshold_target_recall = future_threshold_target_recall
    evaluate_downstream_v2.known_target_unknown_recall = known_target_unknown_recall

    backbone_parameters = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.") and parameter.requires_grad
    ]
    optimizer = optim.AdamW(
        [
            {"params": backbone_parameters, "lr": backbone_lr},
            {"params": head_parameters, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )

    total_steps = max(len(train_loader) * epochs, 1)
    warmup_steps = max(len(train_loader) * warmup_epochs, 1)

    def lr_lambda(step):
        if step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    start_epoch = 0
    best_rank = (-float("inf"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    best_rank = base_train.load_best_validation_rank(checkpoint_dir, device, best_rank)

    if best_rank[0] > -float("inf"):
        print(f"Existing best validation rank: {best_rank}", flush=True)

    if resume_checkpoint:
        resume_path, resume_state = base_train.resolve_resume_checkpoint(resume_checkpoint, device)
        checkpoint_attack_labels = list(resume_state.get("known_attack_labels", []))
        if checkpoint_attack_labels != list(train_dataset.known_attack_names):
            raise ValueError(
                "Resume checkpoint attack-label vocabulary does not match the current dataset. "
                f"Checkpoint labels: {checkpoint_attack_labels} | Dataset labels: {train_dataset.known_attack_names}"
            )

        checkpoint_future_task_enabled = bool(resume_state.get("future_task_enabled", True))
        if checkpoint_future_task_enabled != future_task_enabled:
            raise ValueError(
                "Resume checkpoint future-task setting does not match the current configuration. "
                f"Checkpoint future_task_enabled={checkpoint_future_task_enabled} | "
                f"Requested future_task_enabled={future_task_enabled}"
            )

        checkpoint_seq_len = int(resume_state.get("seq_len", 32))
        checkpoint_stride = int(resume_state.get("stride", 16))
        checkpoint_label_rule = str(resume_state.get("current_label_rule", "any_attack"))
        if checkpoint_seq_len != seq_len or checkpoint_stride != stride or checkpoint_label_rule != current_label_rule:
            raise ValueError(
                "Resume checkpoint window configuration does not match the current configuration. "
                f"Checkpoint seq_len={checkpoint_seq_len}, stride={checkpoint_stride}, label_rule={checkpoint_label_rule} | "
                f"Requested seq_len={seq_len}, stride={stride}, label_rule={current_label_rule}"
            )

        for field_name, expected_value in (
            ("future_horizon_minutes", future_horizon_minutes),
            ("train_target_positive_rate", train_target_positive_rate),
            ("threshold_target_recall", threshold_target_recall),
            ("future_threshold_target_recall", future_threshold_target_recall),
            ("known_target_unknown_recall", known_target_unknown_recall),
            ("unknown_family_loss_weight", unknown_family_loss_weight),
            ("family_loss_weight", family_loss_weight),
            ("future_loss_weight", future_loss_weight),
            ("family_sampler_power", family_sampler_power),
            ("future_positive_boost", future_positive_boost),
        ):
            existing_value = resume_state.get(field_name)
            if existing_value is None:
                continue
            if float(existing_value) != float(expected_value):
                raise ValueError(
                    f"Resume checkpoint {field_name}={existing_value} | requested {expected_value}"
                )

        checkpoint_pseudo_zero_day_families = list(resume_state.get("pseudo_zero_day_families", []))
        if checkpoint_pseudo_zero_day_families != pseudo_zero_day_families:
            raise ValueError(
                "Resume checkpoint pseudo-zero-day families do not match the current configuration. "
                f"Checkpoint families: {checkpoint_pseudo_zero_day_families} | Requested families: {pseudo_zero_day_families}"
            )

        model.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scheduler.load_state_dict(resume_state["scheduler_state_dict"])
        thresholds = dict(resume_state.get("thresholds", thresholds))

        foundation_checkpoint = resume_state.get("foundation_checkpoint", foundation_checkpoint)
        resume_epoch = resume_state.get("resume_epoch")
        if resume_epoch is not None:
            start_epoch = int(resume_epoch)
        else:
            start_epoch = int(resume_state.get("epoch", -1)) + 1
        resume_metrics = resume_state.get("validation_metrics")
        if resume_metrics is not None:
            if "best_future" in resume_metrics and "best_known" in resume_metrics:
                resume_rank = build_validation_rank(resume_metrics, future_task_enabled)
            else:
                resume_score = float(resume_state.get("validation_score", -float("inf")))
                resume_rank = (resume_score, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        else:
            resume_score = float(resume_state.get("validation_score", -float("inf")))
            resume_rank = (resume_score, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        best_rank = base_train.load_best_validation_rank(checkpoint_dir, device, resume_rank)

        print(f"Resumed downstream training from {resume_path}", flush=True)
        if resume_state.get("interrupted"):
            interrupted_after_step = int(resume_state.get("interrupted_after_step", 0))
            steps_per_epoch = int(resume_state.get("steps_per_epoch", 0))
            print(
                "Resume checkpoint was saved during an interrupted epoch; the epoch will be restarted from the saved state.",
                flush=True,
            )
            if steps_per_epoch > 0:
                print(
                    f"Interrupted progress: {interrupted_after_step}/{steps_per_epoch} batches completed in epoch {start_epoch + 1}.",
                    flush=True,
                )
        print(f"Next epoch: {start_epoch + 1}/{epochs}", flush=True)
        print(f"Current best validation rank: {best_rank}", flush=True)

        if start_epoch >= epochs:
            print("Resume checkpoint already reached the requested total epoch count. Nothing to do.", flush=True)
            return

    for epoch in range(start_epoch, epochs):
        print(f"Starting downstream epoch {epoch + 1}/{epochs}...", flush=True)
        backbone_trainable = epoch >= freeze_backbone_epochs
        base_train.set_backbone_trainable(model, backbone_trainable)
        model.train()

        running_losses = {"total": 0.0, "current": 0.0, "family": 0.0, "future": 0.0, "unknown": 0.0}
        progress_bar = None
        completed_steps = 0
        validation_metrics = None
        validation_rank = None
        validation_score = None

        try:
            progress_bar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Downstream Epoch {epoch + 1}/{epochs}",
                file=sys.stdout,
            )

            for step, batch in progress_bar:
                cont = batch["continuous"].to(device, non_blocking=True)
                cat = batch["categorical"].to(device, non_blocking=True)
                label = batch["label"].to(device, non_blocking=True)
                future_attack = batch["future_attack"].to(device, non_blocking=True)
                known_attack_id = batch["known_attack_id"].to(device, non_blocking=True)
                unknown_attack_target = batch["unknown_attack_target"].to(device, non_blocking=True)

                outputs = model(cont, cat, apply_mfm=False)
                losses = base_train.compute_multitask_losses(
                    outputs,
                    {
                        "label": label,
                        "future_attack": future_attack,
                        "known_attack_id": known_attack_id,
                        "unknown_attack_target": unknown_attack_target,
                    },
                    current_loss_fn,
                    family_loss_fn,
                    future_loss_fn,
                    unknown_family_loss_fn,
                    loss_weights,
                )

                optimizer.zero_grad()
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                completed_steps = step + 1
                for loss_name, loss_value in losses.items():
                    running_losses[loss_name] += float(loss_value.item())

                progress_bar.set_postfix(
                    {
                        "loss": f"{running_losses['total'] / (step + 1):.4f}",
                        "current": f"{running_losses['current'] / (step + 1):.4f}",
                        "family": f"{running_losses['family'] / (step + 1):.4f}",
                        "future": f"{running_losses['future'] / (step + 1):.4f}",
                        "unknown": f"{running_losses['unknown'] / (step + 1):.4f}",
                    }
                )

            progress_bar.close()
            progress_bar = None

            validation_metrics = evaluate_downstream_v2(model, valid_loader, device, thresholds)
            best_current = validation_metrics["best_current"]
            best_future = validation_metrics["best_future"]
            best_known = validation_metrics["best_known"]
            thresholds["current"] = best_current["threshold"]
            thresholds["known"] = best_known["threshold"]
            thresholds["future"] = best_future["threshold"] if future_task_enabled else 0.0
            validation_rank = build_validation_rank(validation_metrics, future_task_enabled)
            validation_score = validation_rank[0]

            checkpoint_payload = build_training_checkpoint_payload(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                foundation_checkpoint=foundation_checkpoint,
                checkpoint_dir=checkpoint_dir,
                known_attack_labels=train_dataset.known_attack_names,
                pseudo_zero_day_families=pseudo_zero_day_families,
                future_horizon_minutes=future_horizon_minutes,
                future_task_enabled=future_task_enabled,
                seq_len=seq_len,
                stride=stride,
                current_label_rule=current_label_rule,
                train_target_positive_rate=train_target_positive_rate,
                threshold_target_recall=threshold_target_recall,
                future_threshold_target_recall=future_threshold_target_recall,
                known_target_unknown_recall=known_target_unknown_recall,
                unknown_family_loss_weight=unknown_family_loss_weight,
                family_loss_weight=family_loss_weight,
                future_loss_weight=future_loss_weight,
                family_sampler_power=family_sampler_power,
                future_positive_boost=future_positive_boost,
                loss_weights=loss_weights,
                thresholds=thresholds,
                validation_score=validation_score,
                validation_rank=validation_rank,
                validation_metrics=validation_metrics,
            )

            epoch_checkpoint = os.path.join(checkpoint_dir, f"nids_multitask_epoch_{epoch + 1}.pt")
            base_train.atomic_torch_save(checkpoint_payload, epoch_checkpoint)

            if validation_rank > best_rank:
                best_rank = validation_rank
                best_checkpoint_path = os.path.join(checkpoint_dir, "nids_multitask_best.pt")
                base_train.atomic_torch_save(checkpoint_payload, best_checkpoint_path)

            print(
                f" Downstream epoch {epoch + 1} complete. "
                f"CurrentThresh={best_current['threshold']:.3f} "
                f"KnownThresh={best_known['threshold']:.3f} "
                f"FutureThresh={best_future['threshold']:.3f} | "
                f"CurrentPRAUC={best_current['pr_auc']:.4f} CurrentF1={best_current['f1']:.4f} | "
                f"FuturePRAUC={best_future['pr_auc']:.4f} FutureF1={best_future['f1']:.4f} | "
                f"KnownAcceptedAcc={best_known['accepted_accuracy']:.4f} KnownCoverage={best_known['known_coverage']:.4f} | "
                f"UnknownRecall={best_known['unknown_recall']:.4f} | "
                f"RawKnownAcc={validation_metrics['known_family_accuracy']:.4f} | "
                f"ScorePRAUC={validation_score:.4f}",
                flush=True,
            )
        except KeyboardInterrupt:
            if progress_bar is not None:
                progress_bar.close()

            interruption_state = {
                "interrupted": True,
                "resume_epoch": epoch,
                "interrupted_after_step": completed_steps,
                "steps_per_epoch": len(train_loader),
                "epoch_completion_ratio": (
                    float(completed_steps / max(len(train_loader), 1))
                    if len(train_loader) > 0
                    else 0.0
                ),
            }

            checkpoint_payload = build_training_checkpoint_payload(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                foundation_checkpoint=foundation_checkpoint,
                checkpoint_dir=checkpoint_dir,
                known_attack_labels=train_dataset.known_attack_names,
                pseudo_zero_day_families=pseudo_zero_day_families,
                future_horizon_minutes=future_horizon_minutes,
                future_task_enabled=future_task_enabled,
                seq_len=seq_len,
                stride=stride,
                current_label_rule=current_label_rule,
                train_target_positive_rate=train_target_positive_rate,
                threshold_target_recall=threshold_target_recall,
                future_threshold_target_recall=future_threshold_target_recall,
                known_target_unknown_recall=known_target_unknown_recall,
                unknown_family_loss_weight=unknown_family_loss_weight,
                family_loss_weight=family_loss_weight,
                future_loss_weight=future_loss_weight,
                family_sampler_power=family_sampler_power,
                future_positive_boost=future_positive_boost,
                loss_weights=loss_weights,
                thresholds=thresholds,
                validation_score=validation_score,
                validation_rank=validation_rank,
                validation_metrics=validation_metrics,
                interruption_state=interruption_state,
            )

            interrupt_checkpoint = os.path.join(checkpoint_dir, f"nids_multitask_epoch_{epoch + 1}.pt")
            base_train.atomic_torch_save(checkpoint_payload, interrupt_checkpoint)

            print(
                f"Training interrupted. Saved resume checkpoint to {interrupt_checkpoint} after {completed_steps}/{len(train_loader)} batches of epoch {epoch + 1}.",
                flush=True,
            )
            print(
                "Resume with --resume-checkpoint pointing to this checkpoint file or the checkpoint directory.",
                flush=True,
            )
            return


if __name__ == "__main__":
    train_multitask_nids_v2()