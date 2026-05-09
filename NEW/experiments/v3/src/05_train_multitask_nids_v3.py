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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

EXPERIMENT_SRC_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = EXPERIMENT_SRC_DIR.parent
NEW_DIR = Path(__file__).resolve().parents[3]
SRC_DIR = NEW_DIR / "src"


def load_module_from_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_source_module(module_name, filename):
    return load_module_from_path(module_name, SRC_DIR / filename)


base_train = load_source_module("base_train_multitask_nids", "05_train_multitask_nids.py")
st_data_loader = load_source_module("st_data_loader", "02_st_data_loader.py")
stt_architecture = load_module_from_path("stt_architecture_v3", EXPERIMENT_SRC_DIR / "03_stt_architecture_v3.py")

SpatioTemporalNIDSDataset = st_data_loader.SpatioTemporalNIDSDataset
SpatioTemporalTransformer = stt_architecture.SpatioTemporalTransformer
NIDSMultiTaskModel = stt_architecture.NIDSMultiTaskModel

DEFAULT_TRAIN_DIR = str(NEW_DIR / "data" / "nids_src_grouped" / "train")
DEFAULT_VALID_DIR = str(NEW_DIR / "data" / "nids_src_grouped" / "validation")
DEFAULT_TEST_DIR = str(NEW_DIR / "data" / "nids_src_grouped" / "test")
DEFAULT_STATS_PATH = str(NEW_DIR / "nids_normalization_stats.json")
DEFAULT_DOWNSTREAM_CHECKPOINT_DIR = str(EXPERIMENT_DIR / "checkpoints_future" / "nids_multitask_05_v3_full")
DEFAULT_FUTURE_REFINEMENT_CHECKPOINT_DIR = str(
    EXPERIMENT_DIR / "checkpoints_future" / "nids_multitask_05_v3_future_refinement"
)
DEFAULT_FAMILY_REFINEMENT_CHECKPOINT_DIR = str(
    EXPERIMENT_DIR / "checkpoints_future" / "nids_multitask_05_v3_family_refinement"
)
DEFAULT_FOUNDATION_CHECKPOINT = str(NEW_DIR / "checkpoints" / "stt_best.pt")
DEFAULT_MIN_KNOWN_ATTACK_COUNT = 1
DEFAULT_TRAIN_TARGET_POSITIVE_RATE = base_train.DEFAULT_TRAIN_TARGET_POSITIVE_RATE
DEFAULT_THRESHOLD_TARGET_RECALL = 0.90
DEFAULT_FUTURE_THRESHOLD_TARGET_RECALL = 0.60
DEFAULT_OOD_THRESHOLD_TARGET_RECALL = 0.80
DEFAULT_KNOWN_TARGET_UNKNOWN_RECALL = 0.75
DEFAULT_UNKNOWN_FAMILY_LOSS_WEIGHT = base_train.DEFAULT_UNKNOWN_FAMILY_LOSS_WEIGHT
DEFAULT_OOD_LOSS_WEIGHT = 1.5
DEFAULT_FAMILY_LOSS_WEIGHT = 1.5
DEFAULT_FUTURE_LOSS_WEIGHT = 1.2
DEFAULT_FUTURE_PRE_ONSET_EXCLUSION_GAP_MINUTES = 0.0
DEFAULT_RECONSTRUCTION_LOSS_WEIGHT = 0.25
DEFAULT_FAMILY_SAMPLER_POWER = 0.50
DEFAULT_FUTURE_POSITIVE_BOOST = 2.00
DEFAULT_FUTURE_REFINEMENT_EPOCHS = 4
DEFAULT_FUTURE_REFINEMENT_LR = 1e-3
DEFAULT_FUTURE_REFINEMENT_TARGET_POSITIVE_RATE = 0.20
DEFAULT_FUTURE_REFINEMENT_MIN_PR_AUC_GAIN = 0.001
DEFAULT_FUTURE_REFINEMENT_MAX_CURRENT_PR_AUC_DROP = 0.002
DEFAULT_FUTURE_REFINEMENT_MAX_CURRENT_F1_DROP = 0.005
DEFAULT_FUTURE_REFINEMENT_MAX_OOD_PR_AUC_DROP = 0.010
DEFAULT_FUTURE_REFINEMENT_MAX_OOD_RECALL_DROP = 0.020
DEFAULT_FUTURE_REFINEMENT_MAX_KNOWN_BALANCED_DROP = 0.010
DEFAULT_FUTURE_REFINEMENT_MAX_KNOWN_COVERAGE_DROP = 0.020
DEFAULT_FAMILY_REFINEMENT_EPOCHS = 8
DEFAULT_FAMILY_REFINEMENT_LR = 1e-3
DEFAULT_FAMILY_REFINEMENT_SAMPLER_POWER = 0.20
DEFAULT_FAMILY_REFINEMENT_MAX_FAMILY_BOOST = 4.0
DEFAULT_FAMILY_REFINEMENT_LABEL_SMOOTHING = 0.02
DEFAULT_FAMILY_REFINEMENT_PROJECTION_LR_SCALE = 0.25
DEFAULT_FAMILY_REFINEMENT_MAX_CURRENT_PR_AUC_DROP = 0.002
DEFAULT_FAMILY_REFINEMENT_MAX_CURRENT_F1_DROP = 0.005
DEFAULT_FAMILY_REFINEMENT_MAX_FUTURE_PR_AUC_DROP = 0.002
DEFAULT_FAMILY_REFINEMENT_MAX_FUTURE_F1_DROP = 0.005
DEFAULT_FUTURE_HORIZONS_MINUTES = [1, 3, 5]
DEFAULT_FUTURE_HORIZON_MINUTES = DEFAULT_FUTURE_HORIZONS_MINUTES[-1]
DEFAULT_PSEUDO_ZERO_DAY_FAMILY_COUNT = 0
DEFAULT_ROTATE_PSEUDO_ZERO_DAY_FAMILIES = False
DEFAULT_PSEUDO_ZERO_DAY_ROTATION_SIZE = 0
RUN_MODE_CLOSED_SET = "closed_set_deployment"
RUN_MODE_OPEN_SET = "open_set_benchmark"
DEFAULT_RUN_MODE = RUN_MODE_CLOSED_SET
DEFAULT_OPEN_SET_PSEUDO_ZERO_DAY_FAMILY_COUNT = 2
DEFAULT_OPEN_SET_ROTATE_PSEUDO_ZERO_DAY_FAMILIES = True
DEFAULT_OPEN_SET_PSEUDO_ZERO_DAY_ROTATION_SIZE = 1
DEFAULT_USE_RECONSTRUCTION_HYBRID_OOD = True
DEFAULT_UNKNOWN_RISK_SCORE_MODE = None
DEFAULT_OOD_THRESHOLD_SELECTION_POLICY = "target_recall"
DEFAULT_OOD_MAX_FPR = 0.01
DEFAULT_RECONSTRUCTION_VALIDATION_MASK_SEED = 1729
DEFAULT_NOVELTY_SCORE_MODE = "combined_mae_mfm"
DEFAULT_DECISION_POLICY = "two_stage_current_then_novelty"
DEFAULT_UNKNOWN_HEAD_POLICY = "conditional_on_unknown_labels"
DEFAULT_CLOSED_SET_THESIS_CLAIM = "known_attack_detection_with_novelty_backed_unknown_risk"
DEFAULT_OPEN_SET_THESIS_CLAIM = "open_set_attack_detection_with_held_out_family_benchmark"
VALIDATION_RANK_VERSION = 2
DEFAULT_SELECTION_MIN_OOD_RECALL = 0.70
DEFAULT_SELECTION_MIN_KNOWN_COVERAGE = 0.10
DEFAULT_SELECTION_MIN_FUTURE_PR_AUC = 0.02

UNKNOWN_RISK_SCORE_MODE_HYBRID = "hybrid_max_raw_unknown_head_and_reconstruction_percentile"
UNKNOWN_RISK_SCORE_MODE_RAW = "raw_unknown_head_only"
UNKNOWN_RISK_SCORE_MODE_RECON = "reconstruction_percentile_only"

OOD_THRESHOLD_SELECTION_POLICY_TARGET_RECALL = "target_recall"
OOD_THRESHOLD_SELECTION_POLICY_MAX_FPR = "max_fpr"

CAT_VOCABS = [256, 256, 256, 65536, 65536, 256, 256, 256, 256]
PERCENTILE_CANDIDATES = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the v3 downstream multitask NIDS model with a dedicated OOD head and configurable future horizon."
    )
    parser.add_argument(
        "--foundation-checkpoint",
        default=DEFAULT_FOUNDATION_CHECKPOINT,
        help="Foundation checkpoint used to initialize the backbone.",
    )
    parser.add_argument(
        "--run-mode",
        choices=[RUN_MODE_CLOSED_SET, RUN_MODE_OPEN_SET],
        default=DEFAULT_RUN_MODE,
        help=(
            "Training contract for this run. closed_set_deployment keeps all observed train families known by default; "
            "open_set_benchmark enables held-out-family unknown evaluation by default."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_DOWNSTREAM_CHECKPOINT_DIR,
        help="Directory where v3 checkpoints and metadata will be written.",
    )
    parser.add_argument(
        "--future-refinement-output-dir",
        default=DEFAULT_FUTURE_REFINEMENT_CHECKPOINT_DIR,
        help="Directory where future-only refinement checkpoints and metadata will be written.",
    )
    parser.add_argument(
        "--family-refinement-output-dir",
        default=DEFAULT_FAMILY_REFINEMENT_CHECKPOINT_DIR,
        help="Directory where family-only refinement checkpoints and metadata will be written.",
    )
    parser.add_argument(
        "--refinement-source-dir",
        default=None,
        help=(
            "Optional checkpoint directory whose nids_multitask_best.pt will seed future/family refinement. "
            "Use this for future-horizon transfer experiments without reusing the main output directory."
        ),
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
        help="Minimum malicious window count required for a family to stay supervised. Default 1 keeps all observed train families supervised.",
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
        "--ood-threshold-target-recall",
        type=float,
        default=DEFAULT_OOD_THRESHOLD_TARGET_RECALL,
        help="Recall target used to pick the OOD or unknown-warning threshold on validation.",
    )
    parser.add_argument(
        "--unknown-risk-score-mode",
        choices=[
            UNKNOWN_RISK_SCORE_MODE_HYBRID,
            UNKNOWN_RISK_SCORE_MODE_RAW,
            UNKNOWN_RISK_SCORE_MODE_RECON,
        ],
        default=DEFAULT_UNKNOWN_RISK_SCORE_MODE,
        help=(
            "Decision-time score used for unknown-risk alerting. Leave unset to infer it from the model "
            "capabilities and reconstruction setting."
        ),
    )
    parser.add_argument(
        "--ood-threshold-selection-policy",
        choices=[
            OOD_THRESHOLD_SELECTION_POLICY_TARGET_RECALL,
            OOD_THRESHOLD_SELECTION_POLICY_MAX_FPR,
        ],
        default=DEFAULT_OOD_THRESHOLD_SELECTION_POLICY,
        help="Policy used to pick the OOD or unknown-warning threshold on validation.",
    )
    parser.add_argument(
        "--ood-max-fpr",
        type=float,
        default=DEFAULT_OOD_MAX_FPR,
        help="Maximum validation false-positive rate allowed when --ood-threshold-selection-policy=max_fpr.",
    )
    parser.add_argument(
        "--known-target-unknown-recall",
        type=float,
        default=DEFAULT_KNOWN_TARGET_UNKNOWN_RECALL,
        help="Fallback target unknown recall for the family-confidence gate on held-out pseudo-zero-day families.",
    )
    parser.add_argument(
        "--unknown-family-loss-weight",
        type=float,
        default=DEFAULT_UNKNOWN_FAMILY_LOSS_WEIGHT,
        help="Weight for the open-set family regularizer on unknown windows.",
    )
    parser.add_argument(
        "--ood-loss-weight",
        type=float,
        default=DEFAULT_OOD_LOSS_WEIGHT,
        help="Weight for the explicit OOD or unknown-attack detection head.",
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
        "--reconstruction-loss-weight",
        type=float,
        default=DEFAULT_RECONSTRUCTION_LOSS_WEIGHT,
        help="Weight for the auxiliary masked-reconstruction loss inherited from foundation training.",
    )
    parser.add_argument(
        "--disable-reconstruction-hybrid-ood",
        action="store_false",
        dest="use_reconstruction_hybrid_ood",
        help="Disable reconstruction-backed OOD fusion and keep the explicit unknown head only.",
    )
    parser.set_defaults(use_reconstruction_hybrid_ood=DEFAULT_USE_RECONSTRUCTION_HYBRID_OOD)
    parser.add_argument(
        "--future-horizons-minutes",
        nargs="+",
        type=int,
        default=None,
        help="Forecasting horizons in minutes used to label and predict future-attack warnings.",
    )
    parser.add_argument(
        "--future-pre-onset-exclusion-gap-minutes",
        type=float,
        default=DEFAULT_FUTURE_PRE_ONSET_EXCLUSION_GAP_MINUTES,
        help=(
            "Ignore future-task supervision when the first upcoming attack starts less than this many minutes "
            "after the benign window ends. This changes only the future task."
        ),
    )
    parser.add_argument(
        "--future-horizon-minutes",
        type=int,
        default=None,
        help="Deprecated single future horizon override. Use --future-horizons-minutes instead.",
    )
    parser.add_argument(
        "--family-sampler-power",
        type=float,
        default=DEFAULT_FAMILY_SAMPLER_POWER,
        help="Exponent used to upweight rare attack families in the sampler.",
    )
    parser.add_argument(
        "--future-refinement-epochs",
        type=int,
        default=DEFAULT_FUTURE_REFINEMENT_EPOCHS,
        help=(
            "Extra epochs that train only the future head on benign windows after multitask training. "
            "The backbone, current head, OOD head, and family head stay frozen."
        ),
    )
    parser.add_argument(
        "--future-refinement-lr",
        type=float,
        default=DEFAULT_FUTURE_REFINEMENT_LR,
        help="Learning rate for the future-only refinement stage.",
    )
    parser.add_argument(
        "--future-refinement-target-positive-rate",
        type=float,
        default=DEFAULT_FUTURE_REFINEMENT_TARGET_POSITIVE_RATE,
        help=(
            "Target future-positive rate used by the future-only refinement sampler over benign windows."
        ),
    )
    parser.add_argument(
        "--family-refinement-epochs",
        type=int,
        default=DEFAULT_FAMILY_REFINEMENT_EPOCHS,
        help=(
            "Extra epochs that train only the family head on known attack windows after multitask training. "
            "Backbone and the other heads stay frozen so current and future predictions remain unchanged."
        ),
    )
    parser.add_argument(
        "--family-refinement-lr",
        type=float,
        default=DEFAULT_FAMILY_REFINEMENT_LR,
        help="Learning rate for the family-only refinement stage.",
    )
    parser.add_argument(
        "--family-refinement-sampler-power",
        type=float,
        default=DEFAULT_FAMILY_REFINEMENT_SAMPLER_POWER,
        help="Mild balancing exponent used only during family-only refinement.",
    )
    parser.add_argument(
        "--family-refinement-max-family-boost",
        type=float,
        default=DEFAULT_FAMILY_REFINEMENT_MAX_FAMILY_BOOST,
        help="Maximum per-family sampling boost applied during family-only refinement.",
    )
    parser.add_argument(
        "--family-refinement-label-smoothing",
        type=float,
        default=DEFAULT_FAMILY_REFINEMENT_LABEL_SMOOTHING,
        help="Label smoothing used only for the family-only refinement loss.",
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
            "Leave unset to use the default closed-set setup with no held-out families."
        ),
    )
    parser.add_argument(
        "--pseudo-zero-day-family-count",
        type=int,
        default=None,
        help="How many families to auto-select when --pseudo-zero-day-families is unset. Default depends on --run-mode.",
    )
    parser.add_argument(
        "--pseudo-zero-day-rotation-size",
        type=int,
        default=None,
        help="How many additional known families to temporarily mask as unknown each training epoch. Default depends on --run-mode.",
    )
    parser.add_argument(
        "--enable-pseudo-zero-day-rotation",
        action="store_true",
        dest="rotate_pseudo_zero_day_families",
        help="Enable epoch-wise rotation of additional surrogate unknown families during training.",
    )
    parser.add_argument(
        "--disable-pseudo-zero-day-rotation",
        action="store_false",
        dest="rotate_pseudo_zero_day_families",
        help="Disable epoch-wise rotation of additional surrogate unknown families during training.",
    )
    parser.set_defaults(rotate_pseudo_zero_day_families=None)
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


def resolve_run_mode_settings(args):
    run_mode = str(args.run_mode)
    requested_pseudo_zero_day_families = normalize_attack_family_names(args.pseudo_zero_day_families)

    if run_mode == RUN_MODE_OPEN_SET:
        default_family_count = DEFAULT_OPEN_SET_PSEUDO_ZERO_DAY_FAMILY_COUNT
        default_rotate = DEFAULT_OPEN_SET_ROTATE_PSEUDO_ZERO_DAY_FAMILIES
        default_rotation_size = DEFAULT_OPEN_SET_PSEUDO_ZERO_DAY_ROTATION_SIZE
        thesis_claim = DEFAULT_OPEN_SET_THESIS_CLAIM
    else:
        default_family_count = DEFAULT_PSEUDO_ZERO_DAY_FAMILY_COUNT
        default_rotate = DEFAULT_ROTATE_PSEUDO_ZERO_DAY_FAMILIES
        default_rotation_size = DEFAULT_PSEUDO_ZERO_DAY_ROTATION_SIZE
        thesis_claim = DEFAULT_CLOSED_SET_THESIS_CLAIM

    pseudo_zero_day_family_count = (
        default_family_count
        if args.pseudo_zero_day_family_count is None
        else int(args.pseudo_zero_day_family_count)
    )
    rotate_pseudo_zero_day_families = (
        default_rotate
        if args.rotate_pseudo_zero_day_families is None
        else bool(args.rotate_pseudo_zero_day_families)
    )
    pseudo_zero_day_rotation_size = (
        default_rotation_size
        if args.pseudo_zero_day_rotation_size is None
        else int(args.pseudo_zero_day_rotation_size)
    )

    return {
        "run_mode": run_mode,
        "requested_pseudo_zero_day_families": requested_pseudo_zero_day_families,
        "pseudo_zero_day_family_count": pseudo_zero_day_family_count,
        "rotate_pseudo_zero_day_families": rotate_pseudo_zero_day_families,
        "pseudo_zero_day_rotation_size": pseudo_zero_day_rotation_size,
        "novelty_score_mode": DEFAULT_NOVELTY_SCORE_MODE,
        "decision_policy": DEFAULT_DECISION_POLICY,
        "unknown_head_policy": DEFAULT_UNKNOWN_HEAD_POLICY,
        "thesis_claim": thesis_claim,
    }


def normalize_future_horizons_minutes(future_horizons_minutes=None, fallback_single_horizon=None):
    if future_horizons_minutes:
        horizons = future_horizons_minutes
    elif fallback_single_horizon is not None:
        horizons = [fallback_single_horizon]
    else:
        horizons = DEFAULT_FUTURE_HORIZONS_MINUTES

    normalized = sorted({int(value) for value in horizons})
    if not normalized:
        raise ValueError("At least one future horizon must be provided.")
    if normalized[0] <= 0:
        raise ValueError(f"Future horizons must be strictly positive, got {normalized}")
    return normalized


def build_future_horizon_labels(future_horizons_minutes):
    return [f"{int(value)}m" for value in future_horizons_minutes]


def normalize_future_thresholds(thresholds, future_horizons_minutes, default_threshold=0.50):
    labels = build_future_horizon_labels(future_horizons_minutes)
    if thresholds is None:
        return {label: float(default_threshold) for label in labels}

    if isinstance(thresholds, dict):
        normalized = {}
        for idx, label in enumerate(labels):
            horizon_value = future_horizons_minutes[idx]
            value = thresholds.get(label)
            if value is None:
                value = thresholds.get(str(horizon_value))
            if value is None:
                value = thresholds.get(horizon_value)
            if value is None:
                value = thresholds.get("future")
            if value is None:
                value = thresholds.get("default", default_threshold)
            normalized[label] = float(value)
        return normalized

    if np.isscalar(thresholds):
        return {label: float(thresholds) for label in labels}

    threshold_values = [float(value) for value in thresholds]
    if len(threshold_values) == 1:
        return {label: threshold_values[0] for label in labels}
    if len(threshold_values) != len(labels):
        raise ValueError(
            "Future threshold count does not match the configured future horizons. "
            f"Thresholds={threshold_values} | Horizons={future_horizons_minutes}"
        )
    return {label: value for label, value in zip(labels, threshold_values)}


def resolve_unknown_risk_score_mode(requested_mode, use_reconstruction_hybrid_ood, unknown_head_active):
    if requested_mode is not None:
        return str(requested_mode)
    if use_reconstruction_hybrid_ood and unknown_head_active:
        return UNKNOWN_RISK_SCORE_MODE_HYBRID
    if use_reconstruction_hybrid_ood:
        return UNKNOWN_RISK_SCORE_MODE_RECON
    return UNKNOWN_RISK_SCORE_MODE_RAW


def select_ood_threshold(labels, probabilities, selection_policy, target_recall, max_fpr):
    if selection_policy == OOD_THRESHOLD_SELECTION_POLICY_MAX_FPR:
        return base_train.select_threshold_for_max_fpr(labels, probabilities, max_fpr)
    return base_train.select_threshold_for_target_recall(labels, probabilities, target_recall)


def aggregate_future_metrics(metrics_by_horizon):
    if not metrics_by_horizon:
        return {
            "auc": float("nan"),
            "pr_auc": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "false_positive_rate": float("nan"),
            "positive_rate": float("nan"),
            "threshold": float("nan"),
            "horizon_count": 0,
        }

    numeric_fields = [
        "auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "positive_rate",
        "threshold",
    ]
    aggregate = {"horizon_count": len(metrics_by_horizon)}
    for field_name in numeric_fields:
        values = np.asarray(
            [float(metrics.get(field_name, float("nan"))) for metrics in metrics_by_horizon.values()],
            dtype=np.float64,
        )
        aggregate[field_name] = float(np.nanmean(values)) if values.size and not np.isnan(values).all() else float("nan")

    if any("target_recall" in metrics for metrics in metrics_by_horizon.values()):
        target_values = np.asarray(
            [float(metrics.get("target_recall", float("nan"))) for metrics in metrics_by_horizon.values()],
            dtype=np.float64,
        )
        aggregate["target_recall"] = (
            float(np.nanmean(target_values))
            if target_values.size and not np.isnan(target_values).all()
            else float("nan")
        )
        aggregate["selection_policy"] = "macro_average_precision_at_target_recall"

    meets_target = [float(bool(metrics.get("meets_target_recall", False))) for metrics in metrics_by_horizon.values()]
    aggregate["meets_target_recall"] = bool(np.mean(meets_target) == 1.0) if meets_target else False
    return aggregate


def format_metric_value(value, precision=4):
    return f"{float(value):.{precision}f}" if value == value else "nan"


def summarize_future_metrics_by_horizon(metrics_by_horizon):
    if not metrics_by_horizon:
        return "disabled"
    parts = []
    for horizon_label, metrics in metrics_by_horizon.items():
        parts.append(
            f"{horizon_label}(th={format_metric_value(metrics.get('threshold', float('nan')), 3)}, "
            f"pr={format_metric_value(metrics.get('pr_auc', float('nan')), 4)}, "
            f"f1={format_metric_value(metrics.get('f1', float('nan')), 4)})"
        )
    return "; ".join(parts)


def compute_masked_future_loss(future_logits, future_targets, future_supervision_mask, future_loss_fn, fallback_tensor):
    if future_logits is None or future_loss_fn is None:
        return fallback_tensor.new_zeros(())

    if future_targets.ndim == 1:
        future_targets = future_targets.unsqueeze(-1)
    if future_logits.ndim == 1:
        future_logits = future_logits.unsqueeze(-1)

    if future_supervision_mask is None:
        valid_mask = torch.ones_like(future_targets, dtype=torch.bool)
    else:
        if future_supervision_mask.ndim == 1:
            future_supervision_mask = future_supervision_mask.unsqueeze(-1)
        valid_mask = future_supervision_mask > 0.5

    if not bool(valid_mask.any()):
        return fallback_tensor.new_zeros(())

    per_element_loss = F.binary_cross_entropy_with_logits(
        future_logits,
        future_targets,
        pos_weight=getattr(future_loss_fn, "pos_weight", None),
        reduction="none",
    )
    masked_loss = per_element_loss[valid_mask]
    if masked_loss.numel() == 0:
        return fallback_tensor.new_zeros(())
    if getattr(future_loss_fn, "reduction", "mean") == "sum":
        return masked_loss.sum()
    return masked_loss.mean()


def compute_multiclass_macro_f1(targets, predictions):
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if targets.size == 0 or predictions.size == 0 or targets.shape[0] != predictions.shape[0]:
        return float("nan")

    valid_mask = (targets >= 0) & (predictions >= 0)
    if not valid_mask.any():
        return float("nan")

    targets = targets[valid_mask]
    predictions = predictions[valid_mask]
    labels = np.unique(np.concatenate([targets, predictions]))
    if labels.size == 0:
        return float("nan")

    macro_f1_values = []
    for label in labels.tolist():
        true_positive = int(np.sum((predictions == label) & (targets == label)))
        false_positive = int(np.sum((predictions == label) & (targets != label)))
        false_negative = int(np.sum((predictions != label) & (targets == label)))
        denominator = (2 * true_positive) + false_positive + false_negative
        macro_f1_values.append(0.0 if denominator <= 0 else (2.0 * true_positive) / denominator)

    return float(np.mean(np.asarray(macro_f1_values, dtype=np.float64)))


def compute_multiclass_metrics_by_label(targets, predictions, label_names=None, accepted_mask=None):
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if targets.size == 0 or predictions.size == 0 or targets.shape[0] != predictions.shape[0]:
        return {}

    valid_mask = targets >= 0
    if not valid_mask.any():
        return {}

    targets = targets[valid_mask]
    predictions = predictions[valid_mask]
    gated_predictions = predictions.copy()
    accepted_mask_array = None
    if accepted_mask is not None:
        accepted_mask_array = np.asarray(accepted_mask, dtype=bool)
        if accepted_mask_array.shape[0] != valid_mask.shape[0]:
            return {}
        accepted_mask_array = accepted_mask_array[valid_mask]
        gated_predictions[~accepted_mask_array] = -1

    resolved_label_names = list(label_names or [])
    required_label_count = 0
    if targets.size:
        required_label_count = max(required_label_count, int(targets.max()) + 1)
    non_negative_predictions = predictions[predictions >= 0]
    if non_negative_predictions.size:
        required_label_count = max(required_label_count, int(non_negative_predictions.max()) + 1)
    if len(resolved_label_names) < required_label_count:
        resolved_label_names.extend(
            [f"attack_{label_idx}" for label_idx in range(len(resolved_label_names), required_label_count)]
        )

    metrics_by_label = {}
    for label_idx, label_name in enumerate(resolved_label_names):
        support = int(np.sum(targets == label_idx))
        predicted_count = int(np.sum(gated_predictions == label_idx))
        true_positive = int(np.sum((gated_predictions == label_idx) & (targets == label_idx)))
        false_positive = int(np.sum((gated_predictions == label_idx) & (targets != label_idx)))
        false_negative = int(np.sum((gated_predictions != label_idx) & (targets == label_idx)))
        precision = true_positive / predicted_count if predicted_count > 0 else 0.0
        recall = true_positive / support if support > 0 else 0.0
        f1 = 0.0
        if precision > 0.0 or recall > 0.0:
            f1 = (2.0 * precision * recall) / max(precision + recall, 1e-9)

        record = {
            "label_index": int(label_idx),
            "support": support,
            "predicted_count": predicted_count,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        if accepted_mask_array is not None:
            accepted_support = int(np.sum((targets == label_idx) & accepted_mask_array))
            record["accepted_support"] = accepted_support
            record["coverage"] = float(accepted_support / support) if support > 0 else 0.0

        metrics_by_label[str(label_name)] = record

    return metrics_by_label


def add_ovr_curve_metrics_by_label(metrics_by_label, targets, probability_matrix, label_names=None):
    targets = np.asarray(targets, dtype=np.int64)
    probability_matrix = np.asarray(probability_matrix, dtype=np.float64)
    if targets.size == 0 or probability_matrix.size == 0 or probability_matrix.shape[0] != targets.shape[0]:
        return metrics_by_label

    resolved_label_names = list(label_names or [])
    if len(resolved_label_names) < probability_matrix.shape[1]:
        resolved_label_names.extend(
            [f"attack_{label_idx}" for label_idx in range(len(resolved_label_names), probability_matrix.shape[1])]
        )

    for label_idx in range(probability_matrix.shape[1]):
        label_name = str(resolved_label_names[label_idx])
        if label_name not in metrics_by_label:
            metrics_by_label[label_name] = {
                "label_index": int(label_idx),
                "support": int(np.sum(targets == label_idx)),
                "predicted_count": 0,
                "true_positive": 0,
                "false_positive": 0,
                "false_negative": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }

        binary_targets = (targets == label_idx).astype(np.int64)
        class_probabilities = probability_matrix[:, label_idx]
        curve_metrics = base_train.compute_binary_metrics(binary_targets, class_probabilities, threshold=0.5)
        metrics_by_label[label_name]["auc"] = safe_metric(curve_metrics.get("auc"))
        metrics_by_label[label_name]["roc_auc"] = safe_metric(curve_metrics.get("auc"))
        metrics_by_label[label_name]["pr_auc"] = safe_metric(curve_metrics.get("pr_auc"))
        metrics_by_label[label_name]["score_threshold"] = 0.5

    return metrics_by_label


def select_epoch_unknown_attack_families(rotation_pool, rotation_size, epoch_index):
    if rotation_size <= 0 or not rotation_pool:
        return []

    rotation_size = min(int(rotation_size), len(rotation_pool))
    start_index = int((epoch_index * rotation_size) % len(rotation_pool))
    return [rotation_pool[(start_index + offset) % len(rotation_pool)] for offset in range(rotation_size)]


class VariantDownstreamNIDSDataset(base_train.DownstreamNIDSDataset):
    def __init__(
        self,
        base_dataset,
        future_horizon_minutes=None,
        future_horizons_minutes=None,
        future_pre_onset_exclusion_gap_minutes=DEFAULT_FUTURE_PRE_ONSET_EXCLUSION_GAP_MINUTES,
        known_attack_to_idx=None,
        min_known_attack_count=100,
        max_sequences=None,
        current_label_rule="last_half_attack",
        rebuild_target_cache=False,
        held_out_attack_families=None,
        epoch_unknown_attack_families=None,
    ):
        self.base_dataset = base_dataset
        self.future_horizons_minutes = normalize_future_horizons_minutes(
            future_horizons_minutes,
            future_horizon_minutes,
        )
        self.future_horizon_labels = build_future_horizon_labels(self.future_horizons_minutes)
        self.future_horizon_minutes = int(self.future_horizons_minutes[-1])
        self.future_horizon_ms = int(self.future_horizon_minutes * 60 * 1000)
        self.future_horizon_ms_values = np.asarray(self.future_horizons_minutes, dtype=np.int64) * 60_000
        self.future_pre_onset_exclusion_gap_minutes = max(0.0, float(future_pre_onset_exclusion_gap_minutes))
        self.future_pre_onset_exclusion_gap_ms = int(round(self.future_pre_onset_exclusion_gap_minutes * 60_000.0))
        self.sequence_ranges = np.asarray(base_dataset.sequence_ranges[:max_sequences], dtype=np.int64)
        self.max_sequences = max_sequences
        self.current_label_rule = current_label_rule
        self.rebuild_target_cache = rebuild_target_cache
        self.held_out_attack_families = set(normalize_attack_family_names(held_out_attack_families))
        self.epoch_unknown_attack_families = set()
        self.active_unknown_attack_families = sorted(self.held_out_attack_families)
        self.cache_dir = os.path.dirname(base_dataset.sequence_cache_path)
        cache_suffix = (
            f"seq{base_dataset.seq_len}_stride{base_dataset.stride}_"
            f"h{'-'.join(str(value) for value in self.future_horizons_minutes)}m_"
            f"gap{self.future_pre_onset_exclusion_gap_minutes:g}m_{self.current_label_rule}"
        )
        if max_sequences is not None:
            cache_suffix += f"_first{max_sequences}"
        self.target_cache_path = os.path.join(self.cache_dir, f"downstream_targets_{cache_suffix}.npz")
        self.target_meta_path = os.path.join(self.cache_dir, f"downstream_targets_{cache_suffix}_attacks.json")

        if self.base_dataset.group_col is None:
            print("Warning: grouped sequence ids are missing. Future-warning targets fall back to dataset order until ETL is regenerated.")

        (
            self.sequence_current_labels,
            self.sequence_attack_ids,
            self.sequence_start_times,
            self.sequence_end_times,
            self.sequence_group_ids,
            self.future_attack_targets,
            self.future_lead_minutes,
            self.future_supervision_mask,
            self.raw_attack_names,
        ) = self._load_or_build_targets()

        self.future_attack_targets = np.asarray(self.future_attack_targets, dtype=np.int8)
        if self.future_attack_targets.ndim == 1:
            self.future_attack_targets = self.future_attack_targets.reshape(-1, 1)

        self.future_lead_minutes = np.asarray(self.future_lead_minutes, dtype=np.float32)
        if self.future_lead_minutes.ndim == 1:
            self.future_lead_minutes = self.future_lead_minutes.reshape(-1, 1)

        self.future_supervision_mask = np.asarray(self.future_supervision_mask, dtype=np.int8)
        if self.future_supervision_mask.ndim == 1:
            self.future_supervision_mask = self.future_supervision_mask.reshape(-1, 1)

        self.future_effective_targets = np.logical_and(
            self.future_attack_targets == 1,
            self.future_supervision_mask == 1,
        ).astype(np.int8)

        if len(self.sequence_ranges):
            self.future_attack_any_targets = self.future_effective_targets.max(axis=1).astype(np.int8)
            self.future_supervision_any_mask = self.future_supervision_mask.any(axis=1).astype(np.int8)
            masked_future_leads = np.where(self.future_effective_targets == 1, self.future_lead_minutes, np.nan)
            self.future_lead_minutes_any = np.full(len(self.sequence_ranges), -1.0, dtype=np.float32)
            valid_future_any = np.asarray(self.future_attack_any_targets == 1, dtype=bool)
            if valid_future_any.any():
                self.future_lead_minutes_any[valid_future_any] = np.nanmax(
                    masked_future_leads[valid_future_any],
                    axis=1,
                ).astype(np.float32)
        else:
            self.future_attack_any_targets = np.zeros(0, dtype=np.int8)
            self.future_supervision_any_mask = np.zeros(0, dtype=np.int8)
            self.future_lead_minutes_any = np.zeros(0, dtype=np.float32)

        if known_attack_to_idx is None:
            self.known_attack_to_idx = self._build_known_attack_vocab(min_known_attack_count)
        else:
            self.known_attack_to_idx = dict(known_attack_to_idx)

        self.known_attack_names = [None] * len(self.known_attack_to_idx)
        for attack_name, attack_idx in self.known_attack_to_idx.items():
            self.known_attack_names[attack_idx] = attack_name

        self.set_epoch_unknown_attack_families(epoch_unknown_attack_families)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        item.update({
            "label": torch.tensor(self.sequence_current_labels[idx], dtype=torch.long),
            "attack": self.raw_attack_names[int(self.sequence_attack_ids[idx])],
            "known_attack_id": torch.tensor(self.known_attack_targets[idx], dtype=torch.long),
            "future_attack": torch.tensor(self.future_attack_targets[idx], dtype=torch.float32),
            "future_attack_any": torch.tensor(self.future_attack_any_targets[idx], dtype=torch.float32),
            "future_lead_minutes": torch.tensor(self.future_lead_minutes[idx], dtype=torch.float32),
            "future_lead_minutes_any": torch.tensor(self.future_lead_minutes_any[idx], dtype=torch.float32),
            "future_supervision_mask": torch.tensor(self.future_supervision_mask[idx], dtype=torch.float32),
            "unknown_attack_target": torch.tensor(self.unknown_attack_targets[idx], dtype=torch.float32),
        })
        return item

    def _build_future_targets(self, sequence_current_labels, sequence_start_times, sequence_end_times, sequence_group_ids):
        num_sequences = len(sequence_current_labels)
        num_horizons = len(self.future_horizons_minutes)
        future_attack_targets = np.zeros((num_sequences, num_horizons), dtype=np.int8)
        future_lead_minutes = np.full((num_sequences, num_horizons), -1.0, dtype=np.float32)
        future_supervision_mask = np.ones((num_sequences, num_horizons), dtype=np.int8)

        if num_sequences == 0:
            return future_attack_targets, future_lead_minutes, future_supervision_mask

        group_change_indices = np.flatnonzero(sequence_group_ids[1:] != sequence_group_ids[:-1]) + 1
        group_starts = np.concatenate([[0], group_change_indices])
        group_ends = np.concatenate([group_change_indices, [num_sequences]])

        for group_start, group_end in zip(group_starts, group_ends):
            group_labels = sequence_current_labels[group_start:group_end]
            group_start_times = sequence_start_times[group_start:group_end]
            group_end_times = sequence_end_times[group_start:group_end]
            malicious_start_times = group_start_times[group_labels == 1]

            if malicious_start_times.size == 0:
                continue

            for local_idx in range(group_end - group_start):
                if group_labels[local_idx] == 1:
                    continue

                window_end_time = group_end_times[local_idx]
                first_future_attack = np.searchsorted(malicious_start_times, window_end_time, side="right")
                if first_future_attack >= malicious_start_times.size:
                    continue

                lead_ms = int(malicious_start_times[first_future_attack] - window_end_time)
                horizon_hits = lead_ms <= self.future_horizon_ms_values
                if not horizon_hits.any():
                    continue

                future_attack_targets[group_start + local_idx, horizon_hits] = 1
                future_lead_minutes[group_start + local_idx, horizon_hits] = float(lead_ms / 60_000.0)

                if self.future_pre_onset_exclusion_gap_ms > 0 and lead_ms < self.future_pre_onset_exclusion_gap_ms:
                    future_supervision_mask[group_start + local_idx, horizon_hits] = 0

        return future_attack_targets, future_lead_minutes, future_supervision_mask

    def _load_or_build_targets(self):
        if (not self.rebuild_target_cache and os.path.exists(self.target_cache_path)
                and os.path.exists(self.target_meta_path)):
            print(f"Loading cached downstream targets: {self.target_cache_path}", flush=True)
            cached = np.load(self.target_cache_path)
            with open(self.target_meta_path, "r") as handle:
                target_meta = json.load(handle)
            raw_attack_names = target_meta["raw_attack_names"]
            future_attack_targets = cached["future_attack_targets"]
            future_lead_minutes = cached["future_lead_minutes"]
            future_supervision_mask = (
                cached["future_supervision_mask"]
                if "future_supervision_mask" in cached
                else np.ones_like(future_attack_targets, dtype=np.int8)
            )
            if future_attack_targets.ndim == 1:
                future_attack_targets = future_attack_targets.reshape(-1, 1)
            if future_lead_minutes.ndim == 1:
                future_lead_minutes = future_lead_minutes.reshape(-1, 1)
            if future_supervision_mask.ndim == 1:
                future_supervision_mask = future_supervision_mask.reshape(-1, 1)
            return (
                cached["sequence_current_labels"],
                cached["sequence_attack_ids"],
                cached["sequence_start_times"],
                cached["sequence_end_times"],
                cached["sequence_group_ids"],
                future_attack_targets,
                future_lead_minutes,
                future_supervision_mask,
                raw_attack_names,
            )

        if self.rebuild_target_cache and os.path.exists(self.target_cache_path):
            print(f"Rebuilding downstream target cache: {self.target_cache_path}", flush=True)

        print("Building downstream targets cache. This can take a while on the first run...", flush=True)
        row_limit = len(self.base_dataset.data)
        if len(self.sequence_ranges) > 0:
            row_limit = int(self.sequence_ranges[-1][1])

        row_labels = self._load_numeric_column(self.base_dataset.label_col, row_limit, np.int8)
        row_start_times = self._load_numeric_column(self.base_dataset.start_time_col, row_limit, np.int64)
        row_end_times = self._load_numeric_column(self.base_dataset.end_time_col, row_limit, np.int64)

        if self.base_dataset.group_col is not None:
            row_group_ids = self._load_numeric_column(self.base_dataset.group_col, row_limit, np.uint64)
        else:
            row_group_ids = None

        row_attack_ids, raw_attack_names = self._build_row_attack_ids(row_limit, row_labels)

        num_sequences = len(self.sequence_ranges)
        sequence_current_labels = np.zeros(num_sequences, dtype=np.int8)
        sequence_attack_ids = np.zeros(num_sequences, dtype=np.int16)
        sequence_start_times = np.zeros(num_sequences, dtype=np.int64)
        sequence_end_times = np.zeros(num_sequences, dtype=np.int64)
        sequence_group_ids = np.zeros(num_sequences, dtype=np.uint64)

        for idx, (start_idx, end_idx) in enumerate(tqdm(self.sequence_ranges, desc="Building downstream targets")):
            start_idx = int(start_idx)
            end_idx = int(end_idx)
            window_labels = row_labels[start_idx:end_idx]
            sequence_start_times[idx] = row_start_times[start_idx]
            sequence_end_times[idx] = row_end_times[end_idx - 1]
            sequence_group_ids[idx] = row_group_ids[start_idx] if row_group_ids is not None else 0

            attack_offsets = np.flatnonzero(window_labels > 0)
            if self._window_has_current_attack(attack_offsets, len(window_labels)):
                sequence_current_labels[idx] = 1
                last_attack_offset = int(attack_offsets[-1])
                sequence_attack_ids[idx] = row_attack_ids[start_idx + last_attack_offset]

        future_attack_targets, future_lead_minutes, future_supervision_mask = self._build_future_targets(
            sequence_current_labels,
            sequence_start_times,
            sequence_end_times,
            sequence_group_ids,
        )

        np.savez_compressed(
            self.target_cache_path,
            sequence_current_labels=sequence_current_labels,
            sequence_attack_ids=sequence_attack_ids,
            sequence_start_times=sequence_start_times,
            sequence_end_times=sequence_end_times,
            sequence_group_ids=sequence_group_ids,
            future_attack_targets=future_attack_targets,
            future_lead_minutes=future_lead_minutes,
            future_supervision_mask=future_supervision_mask,
        )
        with open(self.target_meta_path, "w") as handle:
            json.dump(
                {
                    "raw_attack_names": raw_attack_names,
                    "current_label_rule": self.current_label_rule,
                    "future_horizons_minutes": self.future_horizons_minutes,
                    "future_pre_onset_exclusion_gap_minutes": self.future_pre_onset_exclusion_gap_minutes,
                },
                handle,
                indent=2,
            )

        print(f"Saved downstream targets cache: {self.target_cache_path}", flush=True)

        return (
            sequence_current_labels,
            sequence_attack_ids,
            sequence_start_times,
            sequence_end_times,
            sequence_group_ids,
            future_attack_targets,
            future_lead_minutes,
            future_supervision_mask,
            raw_attack_names,
        )

    def _refresh_attack_family_targets(self):
        self.known_attack_targets = np.full(len(self.sequence_ranges), -1, dtype=np.int64)
        self.unknown_attack_targets = np.zeros(len(self.sequence_ranges), dtype=np.int64)
        self.sequence_attack_families = ["Benign"] * len(self.sequence_ranges)
        self.attack_family_counts = Counter()
        self.supervised_attack_family_counts = Counter()

        active_unknown_families = self.held_out_attack_families | self.epoch_unknown_attack_families
        for idx, raw_attack_id in enumerate(self.sequence_attack_ids):
            if self.sequence_current_labels[idx] == 0:
                continue

            attack_family = base_train.map_attack_family(self.raw_attack_names[int(raw_attack_id)])
            self.sequence_attack_families[idx] = attack_family
            self.attack_family_counts[attack_family] += 1

            if attack_family in active_unknown_families or attack_family not in self.known_attack_to_idx:
                self.unknown_attack_targets[idx] = 1
                continue

            self.known_attack_targets[idx] = self.known_attack_to_idx[attack_family]
            self.supervised_attack_family_counts[attack_family] += 1

        self.active_unknown_attack_families = sorted(active_unknown_families)

    def set_epoch_unknown_attack_families(self, family_names):
        normalized = set(normalize_attack_family_names(family_names))
        normalized.difference_update(self.held_out_attack_families)
        self.epoch_unknown_attack_families = normalized
        self._refresh_attack_family_targets()

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
    valid_future_benign_mask = benign_mask & np.asarray(dataset.future_supervision_any_mask == 1, dtype=bool)
    future_positive_mask = valid_future_benign_mask & np.asarray(dataset.future_attack_any_targets == 1, dtype=bool)
    if future_positive_boost != 1.0 and future_positive_mask.any():
        balanced_weights[future_positive_mask] *= float(future_positive_boost)

    mean_weight = float(balanced_weights.mean())
    if mean_weight > 0:
        balanced_weights /= mean_weight

    positive_mask = dataset.sequence_current_labels == 1
    effective_positive_rate = float(
        balanced_weights[positive_mask].sum() / max(balanced_weights.sum(), 1e-12)
    )
    benign_weight_total = float(balanced_weights[valid_future_benign_mask].sum()) if valid_future_benign_mask.any() else 0.0
    observed_future_positive_rate = (
        float(dataset.future_attack_any_targets[valid_future_benign_mask].mean())
        if valid_future_benign_mask.any()
        else 0.0
    )
    effective_future_positive_rate = (
        float(balanced_weights[future_positive_mask].sum() / max(benign_weight_total, 1e-12))
        if benign_weight_total > 0.0
        else 0.0
    )
    sampler_stats = {
        "future_positive_count": int(future_positive_mask.sum()),
        "future_ignored_near_onset_count": int(benign_mask.sum() - valid_future_benign_mask.sum()),
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


def build_future_pos_weight_tensor(dataset, device):
    benign_mask = dataset.sequence_current_labels == 0
    future_targets = np.asarray(dataset.future_attack_targets[benign_mask], dtype=np.int64)
    future_supervision_mask = np.asarray(dataset.future_supervision_mask[benign_mask], dtype=np.int8)
    if future_targets.ndim == 1:
        future_targets = future_targets.reshape(-1, 1)
    if future_supervision_mask.ndim == 1:
        future_supervision_mask = future_supervision_mask.reshape(-1, 1)

    if future_targets.size == 0:
        pos_weights = np.ones(len(dataset.future_horizons_minutes), dtype=np.float32)
    else:
        pos_weight_values = []
        for horizon_idx in range(future_targets.shape[1]):
            horizon_valid_mask = future_supervision_mask[:, horizon_idx] == 1
            if not horizon_valid_mask.any():
                pos_weight_values.append(1.0)
                continue
            pos_weight_values.append(base_train.build_pos_weight(future_targets[horizon_valid_mask, horizon_idx]))
        pos_weights = np.asarray(pos_weight_values, dtype=np.float32)
    return torch.tensor(pos_weights, dtype=torch.float32, device=device)


def build_future_refinement_loader(
    dataset,
    batch_size,
    num_workers,
    device,
    target_positive_rate,
):
    benign_indices = np.flatnonzero(
        (dataset.sequence_current_labels == 0)
        & (np.asarray(dataset.future_supervision_any_mask == 1, dtype=bool))
    )
    if benign_indices.size == 0:
        return None, 0, 0, 0.0, 0.0

    benign_future_targets = np.asarray(dataset.future_attack_any_targets[benign_indices], dtype=np.int64)
    future_positive_count = int(benign_future_targets.sum())
    if future_positive_count <= 0:
        return None, int(benign_indices.size), 0, 0.0, 0.0

    sample_weights, observed_positive_rate, effective_positive_rate = (
        base_train.build_target_rate_sample_weights(
            benign_future_targets,
            target_positive_rate,
        )
    )
    subset = Subset(dataset, benign_indices.tolist())
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(np.asarray(sample_weights, dtype=np.float32)),
        num_samples=len(sample_weights),
        replacement=True,
    )
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    return (
        loader,
        int(benign_indices.size),
        future_positive_count,
        float(observed_positive_rate),
        float(effective_positive_rate),
    )


def set_future_refinement_trainable(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    if model.future_attack_head is not None:
        for parameter in model.future_attack_head.parameters():
            parameter.requires_grad = True


def future_refinement_candidate_is_acceptable(reference_validation_metrics, candidate_validation_metrics):
    reference_future = reference_validation_metrics.get("best_future") or {}
    candidate_future = candidate_validation_metrics.get("best_future") or {}
    reference_current = reference_validation_metrics["best_current"]
    candidate_current = candidate_validation_metrics["best_current"]
    reference_ood = reference_validation_metrics["best_ood"]
    candidate_ood = candidate_validation_metrics["best_ood"]
    reference_known = reference_validation_metrics["best_known"]
    candidate_known = candidate_validation_metrics["best_known"]

    reference_future_pr_auc = safe_metric(reference_future.get("pr_auc"))
    candidate_future_pr_auc = safe_metric(candidate_future.get("pr_auc"))
    future_pr_auc_gain = candidate_future_pr_auc - reference_future_pr_auc

    reasons = []
    if future_pr_auc_gain < DEFAULT_FUTURE_REFINEMENT_MIN_PR_AUC_GAIN:
        reasons.append(
            "future_pr_auc_gain="
            f"{future_pr_auc_gain:.4f} < {DEFAULT_FUTURE_REFINEMENT_MIN_PR_AUC_GAIN:.4f}"
        )

    current_pr_auc_drop = safe_metric(reference_current.get("pr_auc")) - safe_metric(candidate_current.get("pr_auc"))
    if current_pr_auc_drop > DEFAULT_FUTURE_REFINEMENT_MAX_CURRENT_PR_AUC_DROP:
        reasons.append(
            "current_pr_auc_drop="
            f"{current_pr_auc_drop:.4f} > {DEFAULT_FUTURE_REFINEMENT_MAX_CURRENT_PR_AUC_DROP:.4f}"
        )

    current_f1_drop = safe_metric(reference_current.get("f1")) - safe_metric(candidate_current.get("f1"))
    if current_f1_drop > DEFAULT_FUTURE_REFINEMENT_MAX_CURRENT_F1_DROP:
        reasons.append(
            "current_f1_drop="
            f"{current_f1_drop:.4f} > {DEFAULT_FUTURE_REFINEMENT_MAX_CURRENT_F1_DROP:.4f}"
        )

    reference_ood_positive_rate = safe_metric(reference_ood.get("positive_rate"))
    if reference_ood_positive_rate > 0.0:
        ood_pr_auc_drop = safe_metric(reference_ood.get("pr_auc")) - safe_metric(candidate_ood.get("pr_auc"))
        if ood_pr_auc_drop > DEFAULT_FUTURE_REFINEMENT_MAX_OOD_PR_AUC_DROP:
            reasons.append(
                "ood_pr_auc_drop="
                f"{ood_pr_auc_drop:.4f} > {DEFAULT_FUTURE_REFINEMENT_MAX_OOD_PR_AUC_DROP:.4f}"
            )

        ood_recall_drop = safe_metric(reference_ood.get("recall")) - safe_metric(candidate_ood.get("recall"))
        if ood_recall_drop > DEFAULT_FUTURE_REFINEMENT_MAX_OOD_RECALL_DROP:
            reasons.append(
                "ood_recall_drop="
                f"{ood_recall_drop:.4f} > {DEFAULT_FUTURE_REFINEMENT_MAX_OOD_RECALL_DROP:.4f}"
            )

    known_balanced_drop = safe_metric(reference_known.get("balanced_score")) - safe_metric(
        candidate_known.get("balanced_score")
    )
    if known_balanced_drop > DEFAULT_FUTURE_REFINEMENT_MAX_KNOWN_BALANCED_DROP:
        reasons.append(
            "known_balanced_drop="
            f"{known_balanced_drop:.4f} > {DEFAULT_FUTURE_REFINEMENT_MAX_KNOWN_BALANCED_DROP:.4f}"
        )

    known_coverage_drop = safe_metric(reference_known.get("known_coverage")) - safe_metric(
        candidate_known.get("known_coverage")
    )
    if known_coverage_drop > DEFAULT_FUTURE_REFINEMENT_MAX_KNOWN_COVERAGE_DROP:
        reasons.append(
            "known_coverage_drop="
            f"{known_coverage_drop:.4f} > {DEFAULT_FUTURE_REFINEMENT_MAX_KNOWN_COVERAGE_DROP:.4f}"
        )

    return len(reasons) == 0, future_pr_auc_gain, reasons


def run_future_refinement_stage(
    *,
    model,
    device,
    train_dataset,
    valid_loader,
    batch_size,
    num_workers,
    future_refinement_epochs,
    future_refinement_lr,
    future_refinement_target_positive_rate,
    thresholds,
    checkpoint_dir,
    build_checkpoint_payload,
    build_manifest_payload,
    validation_rank_builder,
    best_rank,
    reference_validation_metrics,
    checkpoint_epoch_offset=-1,
):
    if model.future_attack_head is None or future_refinement_epochs <= 0:
        return best_rank, None, None, None

    (
        refinement_loader,
        benign_window_count,
        future_positive_count,
        observed_positive_rate,
        effective_positive_rate,
    ) = build_future_refinement_loader(
        train_dataset,
        batch_size,
        num_workers,
        device,
        future_refinement_target_positive_rate,
    )
    if refinement_loader is None or benign_window_count <= 0 or future_positive_count <= 0:
        return best_rank, None, None, None

    print(
        "Starting future-only refinement: "
        f"epochs={future_refinement_epochs}, benign_windows={benign_window_count}, "
        f"future_positive_windows={future_positive_count}",
        flush=True,
    )
    print(
        "Future refinement sampler: "
        f"observed_positive_rate={observed_positive_rate:.4f}, "
        f"target_positive_rate={future_refinement_target_positive_rate:.4f}, "
        f"effective_positive_rate={effective_positive_rate:.4f}",
        flush=True,
    )

    set_future_refinement_trainable(model)
    future_parameters = [parameter for parameter in model.future_attack_head.parameters() if parameter.requires_grad]
    optimizer = optim.AdamW(future_parameters, lr=future_refinement_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    future_loss_fn = nn.BCEWithLogitsLoss(pos_weight=build_future_pos_weight_tensor(train_dataset, device))

    accepted_selection = None
    accepted_metrics = None
    accepted_rank = None
    accepted_score = None

    for refinement_epoch in range(future_refinement_epochs):
        model.eval()
        model.future_attack_head.train()
        running_future_loss = 0.0
        progress_bar = tqdm(
            enumerate(refinement_loader),
            total=len(refinement_loader),
            desc=f"Future Refinement {refinement_epoch + 1}/{future_refinement_epochs}",
            file=sys.stdout,
        )

        for step, batch in progress_bar:
            cont = batch["continuous"].to(device, non_blocking=True)
            cat = batch["categorical"].to(device, non_blocking=True)
            future_targets = batch["future_attack"].float().to(device, non_blocking=True)
            future_supervision_mask = batch["future_supervision_mask"].float().to(device, non_blocking=True)
            if future_targets.ndim == 1:
                future_targets = future_targets.unsqueeze(-1)
            if future_supervision_mask.ndim == 1:
                future_supervision_mask = future_supervision_mask.unsqueeze(-1)

            outputs = model(
                cont,
                cat,
                apply_mfm=False,
                compute_reconstruction=False,
                reconstruction_mask=None,
                reconstruction_apply_mfm=False,
            )
            future_logits = outputs["future_attack_logits"]
            if future_logits is not None and future_logits.ndim == 1:
                future_logits = future_logits.unsqueeze(-1)
            if future_logits is None:
                continue

            future_loss = compute_masked_future_loss(
                future_logits,
                future_targets,
                future_supervision_mask,
                future_loss_fn,
                outputs["current_attack_logits"],
            )

            optimizer.zero_grad()
            future_loss.backward()
            torch.nn.utils.clip_grad_norm_(future_parameters, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_future_loss += float(future_loss.item())
            progress_bar.set_postfix({"future": f"{running_future_loss / (step + 1):.4f}"})

        progress_bar.close()

        model.eval()
        last_validation_metrics = evaluate_downstream_v3(model, valid_loader, device, thresholds)
        best_current = last_validation_metrics["best_current"]
        best_ood = last_validation_metrics["best_ood"]
        best_future = last_validation_metrics.get("best_future") or aggregate_future_metrics({})
        best_known = last_validation_metrics["best_known"]

        thresholds["current"] = best_current["threshold"]
        thresholds["known"] = best_known["threshold"]
        thresholds["future"] = {
            horizon_label: metrics["threshold"]
            for horizon_label, metrics in last_validation_metrics.get("best_future_by_horizon", {}).items()
        }
        thresholds["ood"] = best_ood["threshold"]

        last_validation_rank = validation_rank_builder(last_validation_metrics)
        last_validation_score = last_validation_rank[1]

        checkpoint_payload = build_checkpoint_payload(
            optimizer,
            scheduler,
            validation_score=last_validation_score,
            validation_rank=last_validation_rank,
            validation_metrics=last_validation_metrics,
            payload_epoch=checkpoint_epoch_offset + refinement_epoch + 1,
        )
        refinement_checkpoint = os.path.join(
            checkpoint_dir,
            f"nids_multitask_future_refine_epoch_{refinement_epoch + 1}.pt",
        )
        base_train.atomic_torch_save(checkpoint_payload, refinement_checkpoint)
        dump_run_manifest(checkpoint_dir, build_manifest_payload(last_validation_metrics))

        accepted, future_pr_auc_gain, rejection_reasons = future_refinement_candidate_is_acceptable(
            reference_validation_metrics,
            last_validation_metrics,
        )
        candidate_selection = (
            safe_metric(best_future.get("pr_auc")),
            safe_metric(best_future.get("f1")),
            last_validation_score,
        )

        if accepted and (accepted_selection is None or candidate_selection > accepted_selection):
            accepted_selection = candidate_selection
            accepted_metrics = last_validation_metrics
            accepted_rank = last_validation_rank
            accepted_score = last_validation_score
            best_rank = last_validation_rank
            best_checkpoint_path = os.path.join(checkpoint_dir, "nids_multitask_best.pt")
            base_train.atomic_torch_save(checkpoint_payload, best_checkpoint_path)
            decision_text = f"accepted future_gain={future_pr_auc_gain:.4f}"
        else:
            reason_text = "; ".join(rejection_reasons) if rejection_reasons else "not better than current accepted future refinement"
            decision_text = f"rejected {reason_text}"

        print(
            f" Future refinement {refinement_epoch + 1} complete. "
            f"CurrentPRAUC={best_current['pr_auc']:.4f} CurrentF1={best_current['f1']:.4f} | "
            f"FutureMacroPRAUC={format_metric_value(best_future.get('pr_auc', float('nan')), 4)} "
            f"FutureMacroF1={format_metric_value(best_future.get('f1', float('nan')), 4)} | "
            f"KnownAcceptedAcc={best_known['accepted_accuracy']:.4f} KnownCoverage={best_known['known_coverage']:.4f} | "
            f"CompositeScore={last_validation_score:.4f} | {decision_text}",
            flush=True,
        )

    return best_rank, accepted_metrics, accepted_rank, accepted_score


def build_family_refinement_loader(
    dataset,
    batch_size,
    num_workers,
    device,
    sampler_power,
    max_family_boost,
):
    known_indices = np.flatnonzero(dataset.known_attack_targets >= 0)
    if known_indices.size == 0:
        return None, 0, {}

    subset = Subset(dataset, known_indices.tolist())
    known_targets = dataset.known_attack_targets[known_indices]
    family_counts = Counter(int(value) for value in known_targets.tolist())
    family_sampler_factors = {}

    if sampler_power > 0.0 and family_counts:
        max_count = float(max(family_counts.values()))
        family_sampler_factors = {
            int(family_idx): min(
                float((max_count / count) ** sampler_power),
                float(max_family_boost),
            )
            for family_idx, count in family_counts.items()
        }
        sample_weights = np.asarray(
            [family_sampler_factors.get(int(target), 1.0) for target in known_targets],
            dtype=np.float32,
        )
        mean_weight = float(sample_weights.mean())
        if mean_weight > 0.0:
            sample_weights /= mean_weight
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=num_workers > 0,
        )
    else:
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=num_workers > 0,
        )

    readable_factors = {
        dataset.known_attack_names[family_idx]: factor
        for family_idx, factor in family_sampler_factors.items()
    }
    return loader, int(known_indices.size), readable_factors


def set_family_refinement_trainable(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module_name in ("pool_norm", "shared_projection", "attack_family_head"):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.requires_grad = True


def family_refinement_candidate_is_acceptable(reference_validation_metrics, candidate_validation_metrics):
    if reference_validation_metrics is None:
        return True, []

    reference_current = reference_validation_metrics["best_current"]
    candidate_current = candidate_validation_metrics["best_current"]
    reference_future = reference_validation_metrics.get("best_future") or {}
    candidate_future = candidate_validation_metrics.get("best_future") or {}

    reasons = []

    current_pr_auc_drop = safe_metric(reference_current.get("pr_auc")) - safe_metric(candidate_current.get("pr_auc"))
    if current_pr_auc_drop > DEFAULT_FAMILY_REFINEMENT_MAX_CURRENT_PR_AUC_DROP:
        reasons.append(
            "current_pr_auc_drop="
            f"{current_pr_auc_drop:.4f} > {DEFAULT_FAMILY_REFINEMENT_MAX_CURRENT_PR_AUC_DROP:.4f}"
        )

    current_f1_drop = safe_metric(reference_current.get("f1")) - safe_metric(candidate_current.get("f1"))
    if current_f1_drop > DEFAULT_FAMILY_REFINEMENT_MAX_CURRENT_F1_DROP:
        reasons.append(
            "current_f1_drop="
            f"{current_f1_drop:.4f} > {DEFAULT_FAMILY_REFINEMENT_MAX_CURRENT_F1_DROP:.4f}"
        )

    reference_future_pr_auc = safe_metric(reference_future.get("pr_auc"))
    candidate_future_pr_auc = safe_metric(candidate_future.get("pr_auc"))
    future_pr_auc_drop = reference_future_pr_auc - candidate_future_pr_auc
    if future_pr_auc_drop > DEFAULT_FAMILY_REFINEMENT_MAX_FUTURE_PR_AUC_DROP:
        reasons.append(
            "future_pr_auc_drop="
            f"{future_pr_auc_drop:.4f} > {DEFAULT_FAMILY_REFINEMENT_MAX_FUTURE_PR_AUC_DROP:.4f}"
        )

    future_f1_drop = safe_metric(reference_future.get("f1")) - safe_metric(candidate_future.get("f1"))
    if future_f1_drop > DEFAULT_FAMILY_REFINEMENT_MAX_FUTURE_F1_DROP:
        reasons.append(
            "future_f1_drop="
            f"{future_f1_drop:.4f} > {DEFAULT_FAMILY_REFINEMENT_MAX_FUTURE_F1_DROP:.4f}"
        )

    return len(reasons) == 0, reasons


def run_family_refinement_stage(
    *,
    model,
    device,
    train_dataset,
    valid_loader,
    batch_size,
    num_workers,
    family_refinement_epochs,
    family_refinement_lr,
    family_refinement_sampler_power,
    family_refinement_max_family_boost,
    family_refinement_label_smoothing,
    thresholds,
    checkpoint_dir,
    build_checkpoint_payload,
    build_manifest_payload,
    validation_rank_builder,
    best_rank,
    reference_validation_metrics=None,
    checkpoint_epoch_offset=-1,
):
    if model.attack_family_head is None or family_refinement_epochs <= 0:
        return best_rank, None, None, None

    refinement_loader, known_window_count, family_sampler_factors = build_family_refinement_loader(
        train_dataset,
        batch_size,
        num_workers,
        device,
        family_refinement_sampler_power,
        family_refinement_max_family_boost,
    )
    if refinement_loader is None or known_window_count <= 0:
        return best_rank, None, None, None

    print(
        f"Starting family-only refinement: epochs={family_refinement_epochs}, known_windows={known_window_count}",
        flush=True,
    )
    if family_sampler_factors:
        print(f"Family refinement sampler factors: {family_sampler_factors}", flush=True)

    set_family_refinement_trainable(model)
    projection_parameters = []
    for module_name in ("pool_norm", "shared_projection"):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        projection_parameters.extend([parameter for parameter in module.parameters() if parameter.requires_grad])
    family_head_parameters = [parameter for parameter in model.attack_family_head.parameters() if parameter.requires_grad]
    trainable_family_parameters = projection_parameters + family_head_parameters
    optimizer_param_groups = []
    if projection_parameters:
        optimizer_param_groups.append(
            {
                "params": projection_parameters,
                "lr": family_refinement_lr * DEFAULT_FAMILY_REFINEMENT_PROJECTION_LR_SCALE,
            }
        )
    if family_head_parameters:
        optimizer_param_groups.append({"params": family_head_parameters, "lr": family_refinement_lr})
    optimizer = optim.AdamW(optimizer_param_groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    family_weight_tensor = build_family_weight_tensor(train_dataset, device)
    family_loss_fn = nn.CrossEntropyLoss(
        weight=family_weight_tensor,
        label_smoothing=family_refinement_label_smoothing,
    )

    print(
        "Family refinement trainable blocks: pool_norm, shared_projection, attack_family_head | "
        f"projection_lr={family_refinement_lr * DEFAULT_FAMILY_REFINEMENT_PROJECTION_LR_SCALE:.5f} | "
        f"head_lr={family_refinement_lr:.5f}",
        flush=True,
    )

    last_validation_metrics = None
    last_validation_rank = None
    last_validation_score = None
    accepted_metrics = reference_validation_metrics
    accepted_rank = best_rank
    accepted_score = best_rank[1] if len(best_rank) > 1 else -float("inf")

    for refinement_epoch in range(family_refinement_epochs):
        model.eval()
        if getattr(model, "pool_norm", None) is not None:
            model.pool_norm.train()
        if getattr(model, "shared_projection", None) is not None:
            model.shared_projection.train()
        model.attack_family_head.train()
        running_family_loss = 0.0
        progress_bar = tqdm(
            enumerate(refinement_loader),
            total=len(refinement_loader),
            desc=f"Family Refinement {refinement_epoch + 1}/{family_refinement_epochs}",
            file=sys.stdout,
        )

        for step, batch in progress_bar:
            cont = batch["continuous"].to(device, non_blocking=True)
            cat = batch["categorical"].to(device, non_blocking=True)
            known_attack_id = batch["known_attack_id"].to(device, non_blocking=True)

            outputs = model(
                cont,
                cat,
                apply_mfm=False,
                compute_reconstruction=False,
                reconstruction_mask=None,
                reconstruction_apply_mfm=False,
            )
            family_logits = outputs["attack_family_logits"]
            family_loss = family_loss_fn(family_logits, known_attack_id)

            optimizer.zero_grad()
            family_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_family_parameters, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_family_loss += float(family_loss.item())
            progress_bar.set_postfix({"family": f"{running_family_loss / (step + 1):.4f}"})

        progress_bar.close()

        model.eval()
        last_validation_metrics = evaluate_downstream_v3(model, valid_loader, device, thresholds)
        best_current = last_validation_metrics["best_current"]
        best_ood = last_validation_metrics["best_ood"]
        best_future = last_validation_metrics.get("best_future") or aggregate_future_metrics({})
        best_known = last_validation_metrics["best_known"]

        thresholds["current"] = best_current["threshold"]
        thresholds["known"] = best_known["threshold"]
        thresholds["future"] = {
            horizon_label: metrics["threshold"]
            for horizon_label, metrics in last_validation_metrics.get("best_future_by_horizon", {}).items()
        }
        thresholds["ood"] = best_ood["threshold"]

        last_validation_rank = validation_rank_builder(last_validation_metrics)
        last_validation_score = last_validation_rank[1]

        checkpoint_payload = build_checkpoint_payload(
            optimizer,
            scheduler,
            validation_score=last_validation_score,
            validation_rank=last_validation_rank,
            validation_metrics=last_validation_metrics,
            payload_epoch=checkpoint_epoch_offset + refinement_epoch + 1,
        )
        refinement_checkpoint = os.path.join(
            checkpoint_dir,
            f"nids_multitask_family_refine_epoch_{refinement_epoch + 1}.pt",
        )
        base_train.atomic_torch_save(checkpoint_payload, refinement_checkpoint)

        dump_run_manifest(checkpoint_dir, build_manifest_payload(last_validation_metrics))

        accepted, rejection_reasons = family_refinement_candidate_is_acceptable(
            reference_validation_metrics,
            last_validation_metrics,
        )
        if accepted and last_validation_rank > accepted_rank:
            accepted_metrics = last_validation_metrics
            accepted_rank = last_validation_rank
            accepted_score = last_validation_score
            best_rank = last_validation_rank
            best_checkpoint_path = os.path.join(checkpoint_dir, "nids_multitask_best.pt")
            base_train.atomic_torch_save(checkpoint_payload, best_checkpoint_path)
            decision_text = "accepted"
        else:
            reason_text = "; ".join(rejection_reasons) if rejection_reasons else "not better than current accepted family refinement"
            decision_text = f"rejected {reason_text}"

        print(
            f" Family refinement {refinement_epoch + 1} complete. "
            f"CurrentPRAUC={best_current['pr_auc']:.4f} CurrentF1={best_current['f1']:.4f} | "
            f"FutureMacroPRAUC={format_metric_value(best_future.get('pr_auc', float('nan')), 4)} "
            f"FutureMacroF1={format_metric_value(best_future.get('f1', float('nan')), 4)} | "
            f"RawKnownMacroF1={format_metric_value(last_validation_metrics.get('known_family_macro_f1', float('nan')), 4)} "
            f"AcceptedKnownMacroF1={format_metric_value(last_validation_metrics.get('known_family_accepted_macro_f1', float('nan')), 4)} | "
            f"KnownAcceptedAcc={best_known['accepted_accuracy']:.4f} KnownCoverage={best_known['known_coverage']:.4f} | "
            f"RawKnownAcc={last_validation_metrics['known_family_accuracy']:.4f} | "
            f"FamilySelectionScore={last_validation_score:.4f} | {decision_text}",
            flush=True,
        )

    return best_rank, accepted_metrics, accepted_rank, accepted_score


def build_family_weight_tensor(dataset, device):
    if not dataset.known_attack_names:
        return None

    attack_counter = Counter(dataset.known_attack_targets[dataset.known_attack_targets >= 0].tolist())
    if not attack_counter:
        return None

    family_weights = np.ones(len(dataset.known_attack_names), dtype=np.float32)
    total_known = sum(attack_counter.values())
    for attack_idx, attack_count in attack_counter.items():
        family_weights[attack_idx] = total_known / max(len(dataset.known_attack_names) * attack_count, 1)
    return torch.tensor(family_weights, dtype=torch.float32, device=device)


def build_train_loader_for_epoch(
    dataset,
    batch_size,
    num_workers,
    device,
    train_target_positive_rate,
    family_sampler_power,
    future_positive_boost,
):
    sample_weights, observed_positive_rate, effective_positive_rate, family_sampler_factors, sampler_stats = (
        build_family_balanced_sample_weights(
            dataset,
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
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    return (
        train_loader,
        observed_positive_rate,
        effective_positive_rate,
        family_sampler_factors,
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
    usable_fallback = None
    fallback = None

    for threshold_value in np.sort(candidates):
        known_gate = (
            (known_current_probabilities >= current_threshold)
            & (known_confidences >= threshold_value)
        )
        accepted_accuracy = float(
            (known_predictions[known_gate] == known_targets[known_gate]).mean()
        ) if known_gate.any() else 0.0
        accepted_macro_f1 = compute_multiclass_macro_f1(
            known_targets[known_gate],
            known_predictions[known_gate],
        )
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
            "accepted_macro_f1": safe_metric(accepted_macro_f1),
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
            record["accepted_macro_f1"],
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
                fallback.get("accepted_macro_f1", 0.0),
                fallback["accepted_accuracy"],
                fallback["known_coverage"],
                -fallback["threshold"],
            )
            if fallback_key > current_fallback_key:
                fallback = record

        if known_gate.any():
            usable_fallback_key = (
                record["unknown_recall"],
                record["balanced_score"],
                record["accepted_macro_f1"],
                record["accepted_accuracy"],
                record["known_coverage"],
                -record["threshold"],
            )
            if usable_fallback is None:
                usable_fallback = record
            else:
                current_usable_fallback_key = (
                    usable_fallback["unknown_recall"],
                    usable_fallback["balanced_score"],
                    usable_fallback.get("accepted_macro_f1", 0.0),
                    usable_fallback["accepted_accuracy"],
                    usable_fallback["known_coverage"],
                    -usable_fallback["threshold"],
                )
                if usable_fallback_key > current_usable_fallback_key:
                    usable_fallback = record

        if not record["meets_target_unknown_recall"] or not known_gate.any():
            continue

        selected_key = (
            record["balanced_score"],
            record["accepted_macro_f1"],
            record["accepted_accuracy"],
            record["known_coverage"],
            -record["threshold"],
        )
        if selected is None:
            selected = record
        else:
            current_selected_key = (
                selected["balanced_score"],
                selected.get("accepted_macro_f1", 0.0),
                selected["accepted_accuracy"],
                selected["known_coverage"],
                -selected["threshold"],
            )
            if selected_key > current_selected_key:
                selected = record

    if selected is not None:
        return selected
    if usable_fallback is not None:
        usable_fallback = dict(usable_fallback)
        usable_fallback["selection_policy"] = "best_effort_nonzero_known_coverage"
        return usable_fallback
    return fallback


def compute_multitask_losses_v3(
    outputs,
    batch,
    current_loss_fn,
    family_loss_fn,
    future_loss_fn,
    ood_loss_fn,
    unknown_family_loss_fn,
    loss_weights,
):
    current_targets = batch["label"].float()
    future_targets = batch["future_attack"].float()
    future_supervision_mask = batch.get("future_supervision_mask")
    if future_targets.ndim == 1:
        future_targets = future_targets.unsqueeze(-1)
    if future_supervision_mask is not None and future_supervision_mask.ndim == 1:
        future_supervision_mask = future_supervision_mask.unsqueeze(-1)
    known_attack_targets = batch["known_attack_id"]
    unknown_attack_targets = batch["unknown_attack_target"].float()

    current_loss = current_loss_fn(outputs["current_attack_logits"], current_targets)

    future_mask = current_targets == 0
    future_logits = outputs.get("future_attack_logits")
    if future_logits is not None and future_logits.ndim == 1:
        future_logits = future_logits.unsqueeze(-1)
    if future_logits is not None and future_loss_fn is not None and future_mask.any():
        future_loss = compute_masked_future_loss(
            future_logits[future_mask],
            future_targets[future_mask],
            future_supervision_mask[future_mask] if future_supervision_mask is not None else None,
            future_loss_fn,
            outputs["current_attack_logits"],
        )
    else:
        future_loss = outputs["current_attack_logits"].new_zeros(())

    known_attack_mask = known_attack_targets >= 0
    if outputs["attack_family_logits"] is not None and family_loss_fn is not None and known_attack_mask.any():
        family_loss = family_loss_fn(outputs["attack_family_logits"][known_attack_mask], known_attack_targets[known_attack_mask])
    else:
        family_loss = outputs["current_attack_logits"].new_zeros(())

    if outputs.get("unknown_attack_logits") is not None and ood_loss_fn is not None:
        ood_loss = ood_loss_fn(outputs["unknown_attack_logits"], unknown_attack_targets)
    else:
        ood_loss = outputs["current_attack_logits"].new_zeros(())

    unknown_attack_mask = unknown_attack_targets > 0.5
    if (
        outputs["attack_family_logits"] is not None
        and unknown_family_loss_fn is not None
        and unknown_attack_mask.any()
    ):
        unknown_regularizer = unknown_family_loss_fn(outputs["attack_family_logits"][unknown_attack_mask])
    else:
        unknown_regularizer = outputs["current_attack_logits"].new_zeros(())

    if outputs.get("reconstructed_cont") is not None and outputs.get("reconstruction_mask") is not None:
        reconstruction_metrics = stt_architecture.compute_reconstruction_metrics(
            outputs["reconstructed_cont"],
            batch["continuous"].float(),
            outputs["reconstruction_mask"],
        )
        reconstruction_loss = reconstruction_metrics["train_loss"]
        reconstruction_masked_mse = reconstruction_metrics["masked_mse"]
        reconstruction_full_mse = reconstruction_metrics["full_mse"]
    else:
        reconstruction_loss = outputs["current_attack_logits"].new_zeros(())
        reconstruction_masked_mse = outputs["current_attack_logits"].new_zeros(())
        reconstruction_full_mse = outputs["current_attack_logits"].new_zeros(())

    total_loss = (
        loss_weights["current"] * current_loss
        + loss_weights["family"] * family_loss
        + loss_weights["future"] * future_loss
        + loss_weights["ood"] * ood_loss
        + loss_weights.get("reconstruction", 0.0) * reconstruction_loss
        + loss_weights.get("unknown_regularizer", 0.0) * unknown_regularizer
    )

    return {
        "total": total_loss,
        "current": current_loss,
        "family": family_loss,
        "future": future_loss,
        "ood": ood_loss,
        "reconstruction": reconstruction_loss,
        "reconstruction_masked_mse": reconstruction_masked_mse,
        "reconstruction_full_mse": reconstruction_full_mse,
        "unknown_regularizer": unknown_regularizer,
    }


def evaluate_downstream_v3(model, data_loader, device, thresholds):
    model.eval()
    metric_totals = {
        "total": 0.0,
        "current": 0.0,
        "family": 0.0,
        "future": 0.0,
        "ood": 0.0,
        "reconstruction": 0.0,
        "reconstruction_masked_mse": 0.0,
        "reconstruction_full_mse": 0.0,
        "unknown_regularizer": 0.0,
    }
    current_probs = []
    current_targets = []
    future_probs = []
    future_targets = []
    future_leads = []
    future_supervision_masks = []
    raw_ood_probs = []
    ood_targets = []
    reconstruction_scores = []
    mae_reconstruction_scores = []
    mfm_reconstruction_scores = []
    family_predictions = []
    family_targets = []
    family_probability_vectors = []
    known_confidences = []
    known_current_probs = []
    unknown_confidences = []
    unknown_current_probs = []
    family_head_enabled = False
    known_attack_labels = list(getattr(data_loader.dataset, "known_attack_names", []))
    future_horizons_minutes = list(
        getattr(
            data_loader.dataset,
            "future_horizons_minutes",
            [getattr(data_loader.dataset, "future_horizon_minutes", DEFAULT_FUTURE_HORIZON_MINUTES)],
        )
    )
    future_horizon_labels = build_future_horizon_labels(future_horizons_minutes)
    future_thresholds = normalize_future_thresholds(thresholds.get("future"), future_horizons_minutes)
    reconstruction_validation_mae_mask_ratio = float(
        getattr(
            evaluate_downstream_v3,
            "reconstruction_validation_mae_mask_ratio",
            getattr(model.backbone, "mae_mask_ratio", 0.30),
        )
    )
    reconstruction_validation_mask_seed = int(
        getattr(
            evaluate_downstream_v3,
            "reconstruction_validation_mask_seed",
            DEFAULT_RECONSTRUCTION_VALIDATION_MASK_SEED,
        )
    )
    reconstruction_validation_mfm_mask_ratio = float(
        getattr(
            evaluate_downstream_v3,
            "reconstruction_validation_mfm_mask_ratio",
            0.10,
        )
    )
    use_reconstruction_hybrid_ood = bool(
        getattr(
            evaluate_downstream_v3,
            "use_reconstruction_hybrid_ood",
            DEFAULT_USE_RECONSTRUCTION_HYBRID_OOD,
        )
    )
    unknown_head_active = bool(
        getattr(
            evaluate_downstream_v3,
            "unknown_head_active",
            getattr(model, "unknown_attack_head", None) is not None,
        )
    )
    unknown_risk_score_mode = str(
        getattr(
            evaluate_downstream_v3,
            "unknown_risk_score_mode",
            resolve_unknown_risk_score_mode(
                None,
                use_reconstruction_hybrid_ood,
                unknown_head_active,
            ),
        )
    )
    novelty_score_mode = str(
        getattr(
            evaluate_downstream_v3,
            "novelty_score_mode",
            DEFAULT_NOVELTY_SCORE_MODE,
        )
    )
    decision_policy = str(
        getattr(
            evaluate_downstream_v3,
            "decision_policy",
            DEFAULT_DECISION_POLICY,
        )
    )
    compute_reconstruction = bool(
        use_reconstruction_hybrid_ood or evaluate_downstream_v3.loss_weights.get("reconstruction", 0.0) > 0.0
    )

    with torch.no_grad():
        for batch_index, batch in enumerate(
            tqdm(data_loader, total=len(data_loader), desc="Downstream validation", leave=False)
        ):
            cont = batch["continuous"].to(device, non_blocking=True)
            cat = batch["categorical"].to(device, non_blocking=True)
            current_target = batch["label"].to(device, non_blocking=True)
            future_target = batch["future_attack"].to(device, non_blocking=True)
            if future_target.ndim == 1:
                future_target = future_target.unsqueeze(-1)
            known_attack_target = batch["known_attack_id"].to(device, non_blocking=True)
            unknown_target = batch["unknown_attack_target"].to(device, non_blocking=True)
            future_lead = batch["future_lead_minutes"].to(device, non_blocking=True)
            if future_lead.ndim == 1:
                future_lead = future_lead.unsqueeze(-1)
            future_supervision_mask = batch["future_supervision_mask"].to(device, non_blocking=True)
            if future_supervision_mask.ndim == 1:
                future_supervision_mask = future_supervision_mask.unsqueeze(-1)

            reconstruction_mask = None
            reconstruction_mfm_mask = None
            if compute_reconstruction:
                reconstruction_mask = stt_architecture.build_fixed_spatial_mask(
                    cont.shape[0],
                    cont.shape[1],
                    reconstruction_validation_mae_mask_ratio,
                    device,
                    reconstruction_validation_mask_seed + batch_index,
                )
                if novelty_score_mode == DEFAULT_NOVELTY_SCORE_MODE:
                    reconstruction_mfm_mask = stt_architecture.build_fixed_spatial_mask(
                        cont.shape[0],
                        cont.shape[1],
                        reconstruction_validation_mfm_mask_ratio,
                        device,
                        reconstruction_validation_mask_seed + 100_000 + batch_index,
                    )

            if compute_reconstruction and novelty_score_mode == DEFAULT_NOVELTY_SCORE_MODE:
                outputs = stt_architecture.compute_combined_reconstruction_outputs(
                    model,
                    cont,
                    cat,
                    reconstruction_mask,
                    reconstruction_mfm_mask,
                )
            else:
                outputs = model(
                    cont,
                    cat,
                    apply_mfm=False,
                    compute_reconstruction=compute_reconstruction,
                    reconstruction_mask=reconstruction_mask,
                    reconstruction_apply_mfm=False,
                )
            batch_losses = compute_multitask_losses_v3(
                outputs,
                {
                    "continuous": cont,
                    "label": current_target,
                    "future_attack": future_target,
                    "future_supervision_mask": future_supervision_mask,
                    "known_attack_id": known_attack_target,
                    "unknown_attack_target": unknown_target,
                },
                evaluate_downstream_v3.current_loss_fn,
                evaluate_downstream_v3.family_loss_fn,
                evaluate_downstream_v3.future_loss_fn,
                evaluate_downstream_v3.ood_loss_fn,
                evaluate_downstream_v3.unknown_family_loss_fn,
                evaluate_downstream_v3.loss_weights,
            )

            for loss_name, loss_value in batch_losses.items():
                metric_totals[loss_name] += float(loss_value.item())

            current_probability = torch.sigmoid(outputs["current_attack_logits"]).cpu().numpy()
            current_target_np = current_target.cpu().numpy()
            current_probs.append(current_probability)
            current_targets.append(current_target_np)

            if outputs.get("unknown_attack_logits") is not None:
                ood_probability = torch.sigmoid(outputs["unknown_attack_logits"]).cpu().numpy()
            else:
                ood_probability = np.zeros_like(current_probability)
            raw_ood_probs.append(ood_probability)
            ood_targets.append(unknown_target.cpu().numpy())

            if outputs.get("reconstruction_score") is not None:
                reconstruction_scores.append(outputs["reconstruction_score"].cpu().numpy())
                mae_reconstruction_scores.append(
                    outputs.get("mae_reconstruction_score", outputs["reconstruction_score"]).cpu().numpy()
                )
                mfm_reconstruction_scores.append(
                    outputs.get("mfm_reconstruction_score", outputs["reconstruction_score"].new_zeros(outputs["reconstruction_score"].shape)).cpu().numpy()
                )
            else:
                reconstruction_scores.append(np.zeros_like(current_probability, dtype=np.float32))
                mae_reconstruction_scores.append(np.zeros_like(current_probability, dtype=np.float32))
                mfm_reconstruction_scores.append(np.zeros_like(current_probability, dtype=np.float32))

            benign_mask = current_target == 0
            future_logits = outputs.get("future_attack_logits")
            if future_logits is not None and future_logits.ndim == 1:
                future_logits = future_logits.unsqueeze(-1)
            if future_logits is not None and benign_mask.any():
                future_probability = torch.sigmoid(future_logits[benign_mask]).cpu().numpy()
                if future_probability.ndim == 1:
                    future_probability = future_probability.reshape(-1, 1)
                future_probs.append(future_probability)
                future_targets.append(future_target[benign_mask].cpu().numpy())
                future_leads.append(future_lead[benign_mask].cpu().numpy())
                future_supervision_masks.append(future_supervision_mask[benign_mask].cpu().numpy())

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
                    family_probability_vectors.append(family_probability[known_mask])
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
        evaluate_downstream_v3,
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

    raw_ood_probabilities = np.concatenate(raw_ood_probs) if raw_ood_probs else np.zeros_like(current_probabilities)
    ood_labels = np.concatenate(ood_targets) if ood_targets else np.array([])
    reconstruction_score_values = (
        np.concatenate(reconstruction_scores) if reconstruction_scores else np.zeros_like(current_probabilities)
    )
    mae_reconstruction_score_values = (
        np.concatenate(mae_reconstruction_scores) if mae_reconstruction_scores else np.zeros_like(current_probabilities)
    )
    mfm_reconstruction_score_values = (
        np.concatenate(mfm_reconstruction_scores) if mfm_reconstruction_scores else np.zeros_like(current_probabilities)
    )
    reconstruction_calibration = stt_architecture.build_reconstruction_calibration(
        reconstruction_score_values,
        current_labels,
    )
    if reconstruction_calibration is not None:
        reconstruction_calibration["score_name"] = f"{novelty_score_mode}_reconstruction_mse"
    reconstruction_probabilities = stt_architecture.reconstruction_scores_to_percentiles(
        reconstruction_score_values,
        reconstruction_calibration,
    )
    ood_probabilities = stt_architecture.resolve_unknown_risk_probabilities(
        raw_ood_probabilities,
        reconstruction_probabilities,
        unknown_risk_score_mode,
    )
    ood_metrics = base_train.compute_binary_metrics(
        ood_labels,
        ood_probabilities,
        thresholds["ood"],
    )

    ood_threshold_target_recall = getattr(
        evaluate_downstream_v3,
        "ood_threshold_target_recall",
        DEFAULT_OOD_THRESHOLD_TARGET_RECALL,
    )
    ood_threshold_selection_policy = getattr(
        evaluate_downstream_v3,
        "ood_threshold_selection_policy",
        DEFAULT_OOD_THRESHOLD_SELECTION_POLICY,
    )
    ood_max_fpr = getattr(
        evaluate_downstream_v3,
        "ood_max_fpr",
        DEFAULT_OOD_MAX_FPR,
    )
    ood_selection = select_ood_threshold(
        ood_labels,
        ood_probabilities,
        ood_threshold_selection_policy,
        ood_threshold_target_recall,
        ood_max_fpr,
    )
    best_ood_metrics = base_train.compute_binary_metrics(
        ood_labels,
        ood_probabilities,
        ood_selection["threshold"],
    )
    best_ood_metrics.update(ood_selection)

    ood_head_metrics = base_train.compute_binary_metrics(
        ood_labels,
        raw_ood_probabilities,
        thresholds["ood"],
    )
    ood_head_selection = select_ood_threshold(
        ood_labels,
        raw_ood_probabilities,
        ood_threshold_selection_policy,
        ood_threshold_target_recall,
        ood_max_fpr,
    )
    best_ood_head_metrics = base_train.compute_binary_metrics(
        ood_labels,
        raw_ood_probabilities,
        ood_head_selection["threshold"],
    )
    best_ood_head_metrics.update(ood_head_selection)

    reconstruction_unknown_metrics = base_train.compute_binary_metrics(
        ood_labels,
        reconstruction_probabilities,
        thresholds["ood"],
    )
    reconstruction_unknown_selection = select_ood_threshold(
        ood_labels,
        reconstruction_probabilities,
        ood_threshold_selection_policy,
        ood_threshold_target_recall,
        ood_max_fpr,
    )
    best_reconstruction_unknown_metrics = base_train.compute_binary_metrics(
        ood_labels,
        reconstruction_probabilities,
        reconstruction_unknown_selection["threshold"],
    )
    best_reconstruction_unknown_metrics.update(reconstruction_unknown_selection)

    future_probabilities = (
        np.concatenate(future_probs, axis=0)
        if future_probs else np.zeros((0, len(future_horizons_minutes)), dtype=np.float32)
    )
    future_labels = (
        np.concatenate(future_targets, axis=0)
        if future_targets else np.zeros((0, len(future_horizons_minutes)), dtype=np.float32)
    )
    future_lead_values = (
        np.concatenate(future_leads, axis=0)
        if future_leads else np.zeros((0, len(future_horizons_minutes)), dtype=np.float32)
    )
    future_supervision_values = (
        np.concatenate(future_supervision_masks, axis=0)
        if future_supervision_masks else np.zeros((0, len(future_horizons_minutes)), dtype=np.float32)
    )

    future_threshold_target_recall = getattr(
        evaluate_downstream_v3,
        "future_threshold_target_recall",
        DEFAULT_FUTURE_THRESHOLD_TARGET_RECALL,
    )
    future_metrics_by_horizon = {}
    best_future_by_horizon = {}
    mean_future_lead_by_horizon = {}
    for horizon_idx, horizon_label in enumerate(future_horizon_labels):
        horizon_labels = future_labels[:, horizon_idx] if future_labels.size else np.array([])
        horizon_probs = future_probabilities[:, horizon_idx] if future_probabilities.size else np.array([])
        horizon_leads = future_lead_values[:, horizon_idx] if future_lead_values.size else np.array([])
        horizon_supervision_mask = (
            future_supervision_values[:, horizon_idx].astype(bool)
            if future_supervision_values.size
            else np.zeros(0, dtype=bool)
        )
        filtered_horizon_labels = horizon_labels[horizon_supervision_mask] if horizon_supervision_mask.size else np.array([])
        filtered_horizon_probs = horizon_probs[horizon_supervision_mask] if horizon_supervision_mask.size else np.array([])
        filtered_horizon_leads = horizon_leads[horizon_supervision_mask] if horizon_supervision_mask.size else np.array([])
        horizon_threshold = future_thresholds[horizon_label]

        future_metrics_by_horizon[horizon_label] = base_train.compute_binary_metrics(
            filtered_horizon_labels,
            filtered_horizon_probs,
            horizon_threshold,
        )
        horizon_selection = base_train.select_threshold_for_target_recall(
            filtered_horizon_labels,
            filtered_horizon_probs,
            future_threshold_target_recall,
        )
        horizon_best_metrics = base_train.compute_binary_metrics(
            filtered_horizon_labels,
            filtered_horizon_probs,
            horizon_selection["threshold"],
        )
        horizon_best_metrics.update(horizon_selection)
        raw_positive_count = int(np.sum(np.asarray(horizon_labels) == 1)) if np.asarray(horizon_labels).size else 0
        valid_positive_count = int(np.sum(np.asarray(filtered_horizon_labels) == 1)) if np.asarray(filtered_horizon_labels).size else 0
        ignored_near_onset_positive_count = max(raw_positive_count - valid_positive_count, 0)
        future_metrics_by_horizon[horizon_label].update({
            "valid_count": int(filtered_horizon_labels.shape[0]),
            "raw_positive_count": raw_positive_count,
            "valid_positive_count": valid_positive_count,
            "ignored_near_onset_positive_count": ignored_near_onset_positive_count,
            "future_pre_onset_exclusion_gap_minutes": float(
                getattr(evaluate_downstream_v3, "future_pre_onset_exclusion_gap_minutes", 0.0)
            ),
        })
        horizon_best_metrics.update({
            "valid_count": int(filtered_horizon_labels.shape[0]),
            "raw_positive_count": raw_positive_count,
            "valid_positive_count": valid_positive_count,
            "ignored_near_onset_positive_count": ignored_near_onset_positive_count,
            "future_pre_onset_exclusion_gap_minutes": float(
                getattr(evaluate_downstream_v3, "future_pre_onset_exclusion_gap_minutes", 0.0)
            ),
        })
        best_future_by_horizon[horizon_label] = horizon_best_metrics

        horizon_hits = (
            (np.asarray(filtered_horizon_probs) >= horizon_best_metrics["threshold"])
            & (np.asarray(filtered_horizon_labels) == 1)
        )
        mean_future_lead_by_horizon[horizon_label] = (
            float(np.asarray(filtered_horizon_leads)[horizon_hits].mean())
            if np.asarray(filtered_horizon_leads).size and horizon_hits.any()
            else float("nan")
        )

    future_metrics = aggregate_future_metrics(future_metrics_by_horizon) if future_metrics_by_horizon else None
    best_future_metrics = aggregate_future_metrics(best_future_by_horizon) if best_future_by_horizon else None
    if future_metrics is not None:
        future_metrics["thresholds"] = dict(future_thresholds)
    if best_future_metrics is not None:
        best_future_metrics["thresholds"] = {
            horizon_label: metrics["threshold"] for horizon_label, metrics in best_future_by_horizon.items()
        }
    mean_future_lead = (
        float(np.nanmean(np.asarray(list(mean_future_lead_by_horizon.values()), dtype=np.float64)))
        if mean_future_lead_by_horizon and not np.isnan(np.asarray(list(mean_future_lead_by_horizon.values()), dtype=np.float64)).all()
        else float("nan")
    )

    family_target_array = np.concatenate(family_targets) if family_targets else np.array([])
    family_prediction_array = np.concatenate(family_predictions) if family_predictions else np.array([])
    family_probability_array = (
        np.concatenate(family_probability_vectors, axis=0)
        if family_probability_vectors
        else np.zeros((0, len(known_attack_labels)), dtype=np.float64)
    )
    raw_known_accuracy = float(
        (family_prediction_array == family_target_array).mean()
    ) if family_target_array.size else float("nan")
    raw_known_macro_f1 = compute_multiclass_macro_f1(
        family_target_array,
        family_prediction_array,
    )
    raw_known_metrics_by_label = compute_multiclass_metrics_by_label(
        family_target_array,
        family_prediction_array,
        known_attack_labels,
    )
    raw_known_metrics_by_label = add_ovr_curve_metrics_by_label(
        raw_known_metrics_by_label,
        family_target_array,
        family_probability_array,
        known_attack_labels,
    )

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
                evaluate_downstream_v3,
                "known_target_unknown_recall",
                DEFAULT_KNOWN_TARGET_UNKNOWN_RECALL,
            ),
            default_threshold=thresholds["known"],
        )
    else:
        best_known_metrics = {
            "threshold": float(thresholds["known"]),
            "accepted_accuracy": 0.0,
            "accepted_macro_f1": 0.0,
            "known_coverage": 0.0,
            "balanced_score": 0.0,
            "unknown_recall": 0.0,
            "meets_target_unknown_recall": False,
            "selection_policy": "no_family_head",
            "target_unknown_recall": getattr(
                evaluate_downstream_v3,
                "known_target_unknown_recall",
                DEFAULT_KNOWN_TARGET_UNKNOWN_RECALL,
            ),
        }

    accepted_known_gate = (
        (known_current_prob_array >= best_current_metrics["threshold"])
        & (known_confidence_array >= best_known_metrics["threshold"])
    ) if family_target_array.size and known_confidence_array.size else np.zeros(0, dtype=bool)
    accepted_known_macro_f1 = compute_multiclass_macro_f1(
        family_target_array[accepted_known_gate],
        family_prediction_array[accepted_known_gate],
    ) if accepted_known_gate.size else float("nan")
    accepted_known_metrics_by_label = compute_multiclass_metrics_by_label(
        family_target_array,
        family_prediction_array,
        known_attack_labels,
        accepted_mask=accepted_known_gate,
    )
    accepted_family_target_array = family_target_array[accepted_known_gate] if accepted_known_gate.size else np.array([])
    accepted_family_probability_array = (
        family_probability_array[accepted_known_gate]
        if accepted_known_gate.size and family_probability_array.size
        else np.zeros((0, family_probability_array.shape[1] if family_probability_array.ndim == 2 else 0), dtype=np.float64)
    )
    accepted_known_metrics_by_label = add_ovr_curve_metrics_by_label(
        accepted_known_metrics_by_label,
        accepted_family_target_array,
        accepted_family_probability_array,
        known_attack_labels,
    )

    return {
        "loss": metric_totals,
        "current": current_metrics,
        "best_current": best_current_metrics,
        "ood": ood_metrics,
        "best_ood": best_ood_metrics,
        "ood_head": ood_head_metrics,
        "best_ood_head": best_ood_head_metrics,
        "reconstruction_unknown": reconstruction_unknown_metrics,
        "best_reconstruction_unknown": best_reconstruction_unknown_metrics,
        "reconstruction_calibration": reconstruction_calibration,
        "reconstruction_score_mode": novelty_score_mode,
        "decision_policy": decision_policy,
        "unknown_head_active": unknown_head_active,
        "unknown_label_positive_count": int(np.asarray(ood_labels).sum()) if np.asarray(ood_labels).size else 0,
        "reconstruction_score_mean": float(np.asarray(reconstruction_score_values).mean()) if np.asarray(reconstruction_score_values).size else 0.0,
        "mae_reconstruction_score_mean": float(np.asarray(mae_reconstruction_score_values).mean()) if np.asarray(mae_reconstruction_score_values).size else 0.0,
        "mfm_reconstruction_score_mean": float(np.asarray(mfm_reconstruction_score_values).mean()) if np.asarray(mfm_reconstruction_score_values).size else 0.0,
        "ood_score_mode": unknown_risk_score_mode if (use_reconstruction_hybrid_ood or unknown_head_active) else "disabled",
        "unknown_risk_score_mode": unknown_risk_score_mode if (use_reconstruction_hybrid_ood or unknown_head_active) else "disabled",
        "ood_threshold_selection_policy": str(ood_threshold_selection_policy),
        "ood_max_fpr": float(ood_max_fpr),
        "future": future_metrics,
        "best_future": best_future_metrics,
        "future_by_horizon": future_metrics_by_horizon,
        "best_future_by_horizon": best_future_by_horizon,
        "future_horizons_minutes": future_horizons_minutes,
        "future_horizon_labels": future_horizon_labels,
        "known_family_accuracy": raw_known_accuracy,
        "known_family_macro_f1": raw_known_macro_f1,
        "known_family_metrics_by_label": raw_known_metrics_by_label,
        "known_family_accepted_accuracy": best_known_metrics["accepted_accuracy"],
        "known_family_accepted_macro_f1": safe_metric(
            best_known_metrics.get("accepted_macro_f1", accepted_known_macro_f1)
        ),
        "known_family_accepted_metrics_by_label": accepted_known_metrics_by_label,
        "known_family_coverage": best_known_metrics["known_coverage"],
        "best_known": best_known_metrics,
        "unknown_warning_recall": ood_metrics["recall"],
        "best_unknown_warning_recall": best_ood_metrics["recall"],
        "mean_future_lead_minutes": mean_future_lead,
        "mean_future_lead_minutes_by_horizon": mean_future_lead_by_horizon,
    }


def safe_metric(value):
    return float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))


def build_validation_rank(
    validation_metrics,
    future_task_enabled,
    *,
    ood_threshold_target_recall=DEFAULT_OOD_THRESHOLD_TARGET_RECALL,
):
    best_current = validation_metrics["best_current"]
    best_ood = validation_metrics["best_ood"]
    best_future = validation_metrics.get("best_future") or {}
    best_known = validation_metrics["best_known"]

    current_pr_auc = safe_metric(best_current.get("pr_auc"))
    current_f1 = safe_metric(best_current.get("f1"))
    ood_pr_auc = safe_metric(best_ood.get("pr_auc"))
    ood_f1 = safe_metric(best_ood.get("f1"))
    ood_recall = safe_metric(best_ood.get("recall"))
    future_pr_auc = safe_metric(best_future.get("pr_auc")) if future_task_enabled else 0.0
    future_f1 = safe_metric(best_future.get("f1")) if future_task_enabled else 0.0
    known_balanced_score = safe_metric(best_known.get("balanced_score"))
    known_coverage = safe_metric(best_known.get("known_coverage"))
    future_positive_rate = safe_metric(best_future.get("positive_rate")) if future_task_enabled else 0.0
    ood_positive_rate = safe_metric(best_ood.get("positive_rate"))
    ood_task_active = ood_positive_rate > 0.0

    ood_recall_floor = max(
        DEFAULT_SELECTION_MIN_OOD_RECALL,
        float(ood_threshold_target_recall) - 0.05,
    )
    known_coverage_floor = DEFAULT_SELECTION_MIN_KNOWN_COVERAGE
    future_pr_auc_floor = (
        max(DEFAULT_SELECTION_MIN_FUTURE_PR_AUC, 2.0 * future_positive_rate)
        if future_task_enabled
        else 0.0
    )

    floor_pass_count = float(known_coverage >= known_coverage_floor)
    if ood_task_active:
        floor_pass_count += float(ood_recall >= ood_recall_floor)
    if future_task_enabled:
        floor_pass_count += float(future_pr_auc >= future_pr_auc_floor)

    weighted_sum = 0.0
    total_weight = 0.0
    for metric_value, weight in (
        (current_pr_auc, 0.24),
        (current_f1, 0.08),
        (ood_pr_auc, 0.18 if ood_task_active else 0.0),
        (ood_f1, 0.10 if ood_task_active else 0.0),
        (ood_recall, 0.08 if ood_task_active else 0.0),
        (known_balanced_score, 0.16),
        (known_coverage, 0.06),
        (future_pr_auc, 0.07 if future_task_enabled else 0.0),
        (future_f1, 0.03 if future_task_enabled else 0.0),
    ):
        if weight <= 0.0:
            continue
        weighted_sum += metric_value * weight
        total_weight += weight

    composite_score = weighted_sum / total_weight if total_weight > 0.0 else 0.0
    return (
        floor_pass_count,
        composite_score,
        current_pr_auc,
        ood_pr_auc,
        future_pr_auc,
        known_balanced_score,
        ood_recall,
        known_coverage,
        current_f1,
        ood_f1,
        future_f1,
    )


def build_family_refinement_rank(validation_metrics):
    best_known = validation_metrics["best_known"]
    balanced_score = safe_metric(best_known.get("balanced_score"))
    accepted_macro_f1 = safe_metric(best_known.get("accepted_macro_f1"))
    accepted_accuracy = safe_metric(best_known.get("accepted_accuracy"))
    raw_macro_f1 = safe_metric(validation_metrics.get("known_family_macro_f1"))
    raw_known_accuracy = safe_metric(validation_metrics.get("known_family_accuracy"))
    known_coverage = safe_metric(best_known.get("known_coverage"))
    family_unknown_recall = safe_metric(best_known.get("unknown_recall"))
    unknown_label_positive_count = int(validation_metrics.get("unknown_label_positive_count", 0))
    target_unknown_recall = safe_metric(best_known.get("target_unknown_recall"))
    unknown_recall_floor = max(0.0, target_unknown_recall - 0.05)
    unknown_recall_pass = 1.0
    if unknown_label_positive_count > 0:
        unknown_recall_pass = float(family_unknown_recall >= unknown_recall_floor)

    return (
        unknown_recall_pass,
        balanced_score,
        accepted_macro_f1,
        accepted_accuracy,
        raw_macro_f1,
        raw_known_accuracy,
        known_coverage,
        family_unknown_recall,
    )


def load_best_validation_rank(checkpoint_dir, device, fallback_rank):
    best_checkpoint_path = Path(checkpoint_dir) / "nids_multitask_best.pt"
    if not best_checkpoint_path.exists():
        return fallback_rank

    try:
        best_checkpoint = base_train.load_trusted_checkpoint(best_checkpoint_path, device)
        stored_rank = best_checkpoint.get("validation_rank")
        stored_rank_version = int(best_checkpoint.get("validation_rank_version", 0))
        if stored_rank is not None and stored_rank_version == VALIDATION_RANK_VERSION:
            return tuple(float(value) for value in stored_rank)

        validation_metrics = best_checkpoint.get("validation_metrics")
        if validation_metrics is not None:
            return build_validation_rank(
                validation_metrics,
                bool(best_checkpoint.get("future_task_enabled", True)),
                ood_threshold_target_recall=float(
                    best_checkpoint.get(
                        "ood_threshold_target_recall",
                        DEFAULT_OOD_THRESHOLD_TARGET_RECALL,
                    )
                ),
            )

        fallback_score = float(best_checkpoint.get("validation_score", fallback_rank[1]))
        return (0.0, fallback_score, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    except Exception as exc:
        print(f"Warning: could not load best checkpoint {best_checkpoint_path}: {exc}", flush=True)
        return fallback_rank


def load_best_multitask_checkpoint(checkpoint_dir, device):
    best_checkpoint_path = Path(checkpoint_dir) / "nids_multitask_best.pt"
    if not best_checkpoint_path.exists():
        return None, None

    checkpoint = base_train.load_trusted_checkpoint(best_checkpoint_path, device)
    return best_checkpoint_path, checkpoint


def restore_best_checkpoint_for_refinement(
    checkpoint_dir,
    device,
    model,
    thresholds,
    future_horizons_minutes,
    stage_name,
):
    best_checkpoint_path, best_checkpoint = load_best_multitask_checkpoint(checkpoint_dir, device)
    if best_checkpoint is None:
        print(
            f"Warning: no saved best checkpoint was found; {stage_name} will start from the last in-memory state.",
            flush=True,
        )
        return thresholds, None, None

    source_state_dict = dict(best_checkpoint["model_state_dict"])
    expected_state_dict = model.state_dict()
    dropped_future_head_keys = []
    for key, value in list(source_state_dict.items()):
        expected_value = expected_state_dict.get(key)
        if expected_value is None:
            continue
        if expected_value.shape != value.shape:
            if key.startswith("future_attack_head."):
                dropped_future_head_keys.append(key)
                del source_state_dict[key]
                continue
            raise ValueError(
                f"Checkpoint tensor shape mismatch for {key}: {tuple(value.shape)} vs expected {tuple(expected_value.shape)}"
            )

    incompatible_missing, incompatible_unexpected = model.load_state_dict(source_state_dict, strict=False)
    incompatible_missing = [key for key in incompatible_missing if not key.startswith("future_attack_head.")]
    incompatible_unexpected = [key for key in incompatible_unexpected if not key.startswith("future_attack_head.")]
    if incompatible_missing or incompatible_unexpected:
        raise ValueError(
            "Refinement checkpoint is incompatible with the requested architecture. "
            f"Missing keys: {incompatible_missing} | Unexpected keys: {incompatible_unexpected}"
        )
    if dropped_future_head_keys:
        print(
            "Future-head shape mismatch detected; keeping the current future head initialization for refinement. "
            f"Dropped keys: {dropped_future_head_keys}",
            flush=True,
        )

    restored_thresholds = dict(best_checkpoint.get("thresholds", thresholds))
    restored_thresholds["future"] = normalize_future_thresholds(
        restored_thresholds.get("future"),
        future_horizons_minutes,
        default_threshold=0.50,
    )
    print(
        f"Starting {stage_name} from best checkpoint: "
        f"{best_checkpoint_path.name} (epoch {int(best_checkpoint.get('epoch', -1)) + 1})",
        flush=True,
    )
    return restored_thresholds, best_checkpoint_path, best_checkpoint


def seed_refinement_checkpoint_dir(
    source_checkpoint_path,
    source_checkpoint,
    target_checkpoint_dir,
    *,
    config_payload=None,
    manifest_payload=None,
    checkpoint_overrides=None,
):
    if source_checkpoint is None:
        return None, None

    os.makedirs(target_checkpoint_dir, exist_ok=True)
    seeded_checkpoint = dict(source_checkpoint)
    if checkpoint_overrides:
        seeded_checkpoint.update(checkpoint_overrides)
    seeded_checkpoint["output_dir"] = target_checkpoint_dir
    target_best_path = Path(target_checkpoint_dir) / "nids_multitask_best.pt"
    base_train.atomic_torch_save(seeded_checkpoint, target_best_path)

    if config_payload is not None:
        dump_variant_config(target_checkpoint_dir, config_payload)
    if manifest_payload is not None:
        dump_run_manifest(target_checkpoint_dir, manifest_payload)

    print(
        "Seeded refinement directory: "
        f"{target_checkpoint_dir} from {source_checkpoint_path}",
        flush=True,
    )
    return target_best_path, seeded_checkpoint


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
            "Output directory already contains v3 checkpoints for a different configuration. "
            "Use a fresh --output-dir or resume the matching run. "
            f"{mismatch_summary}"
        )

    if missing_fields:
        print(
            "Warning: existing best checkpoint is missing v3 configuration metadata "
            f"{missing_fields}; directory compatibility could not be fully verified.",
            flush=True,
        )


def dump_variant_config(checkpoint_dir, config_payload):
    config_path = Path(checkpoint_dir) / "v3_experiment_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as handle:
        json.dump(config_payload, handle, indent=2)


def dump_run_manifest(checkpoint_dir, manifest_payload):
    manifest_path = Path(checkpoint_dir) / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as handle:
        json.dump(manifest_payload, handle, indent=2)


def build_task_activation_metadata(
    *,
    future_task_enabled,
    family_head_enabled,
    unknown_head_active,
    use_reconstruction_hybrid_ood,
    reconstruction_loss_weight,
):
    return {
        "current_attack_head_active": True,
        "future_head_active": bool(future_task_enabled),
        "family_head_active": bool(family_head_enabled),
        "unknown_head_active": bool(unknown_head_active),
        "reconstruction_auxiliary_active": float(reconstruction_loss_weight) > 0.0,
        "reconstruction_novelty_active": bool(use_reconstruction_hybrid_ood),
    }


def build_run_manifest_payload(
    *,
    run_mode,
    thesis_claim,
    novelty_score_mode,
    decision_policy,
    unknown_head_policy,
    task_activation,
    requested_pseudo_zero_day_families,
    pseudo_zero_day_families,
    rotate_pseudo_zero_day_families,
    pseudo_zero_day_rotation_size,
    future_horizons_minutes,
    thresholds,
    loss_weights,
    known_attack_labels,
    unknown_risk_score_mode,
    ood_threshold_selection_policy,
    ood_max_fpr,
    validation_metrics=None,
):
    manifest_payload = {
        "run_mode": run_mode,
        "thesis_claim": thesis_claim,
        "novelty_score_mode": novelty_score_mode,
        "decision_policy": decision_policy,
        "unknown_head_policy": unknown_head_policy,
        "task_activation": task_activation,
        "requested_pseudo_zero_day_families": list(requested_pseudo_zero_day_families),
        "effective_pseudo_zero_day_families": list(pseudo_zero_day_families),
        "rotate_pseudo_zero_day_families": bool(rotate_pseudo_zero_day_families),
        "pseudo_zero_day_rotation_size": int(pseudo_zero_day_rotation_size),
        "future_horizons_minutes": list(future_horizons_minutes),
        "thresholds": thresholds,
        "loss_weights": loss_weights,
        "known_attack_labels": list(known_attack_labels),
        "unknown_risk_score_mode": str(unknown_risk_score_mode),
        "ood_threshold_selection_policy": str(ood_threshold_selection_policy),
        "ood_max_fpr": float(ood_max_fpr),
    }
    if validation_metrics is not None:
        manifest_payload["validation_summary"] = {
            "current_pr_auc": safe_metric(validation_metrics["best_current"].get("pr_auc", 0.0)),
            "unknown_risk_pr_auc": safe_metric(validation_metrics["best_ood"].get("pr_auc", 0.0)),
            "future_pr_auc": safe_metric((validation_metrics.get("best_future") or {}).get("pr_auc", 0.0)),
            "known_family_macro_f1": safe_metric(validation_metrics.get("known_family_macro_f1", 0.0)),
            "known_family_accepted_accuracy": safe_metric(
                validation_metrics.get("known_family_accepted_accuracy", 0.0)
            ),
            "known_family_accepted_macro_f1": safe_metric(
                validation_metrics.get("known_family_accepted_macro_f1", 0.0)
            ),
            "known_family_balanced_score": safe_metric(
                (validation_metrics.get("best_known") or {}).get("balanced_score", 0.0)
            ),
            "unknown_label_positive_count": int(validation_metrics.get("unknown_label_positive_count", 0)),
        }
        manifest_payload["known_family_metrics_by_label"] = dict(
            validation_metrics.get("known_family_metrics_by_label", {})
        )
        manifest_payload["known_family_accepted_metrics_by_label"] = dict(
            validation_metrics.get("known_family_accepted_metrics_by_label", {})
        )
    return manifest_payload


def build_training_checkpoint_payload(
    *,
    epoch,
    model,
    optimizer,
    scheduler,
    foundation_checkpoint,
    checkpoint_dir,
    known_attack_labels,
    requested_pseudo_zero_day_families,
    pseudo_zero_day_family_count,
    pseudo_zero_day_families,
    epoch_unknown_attack_families,
    rotation_family_pool,
    rotate_pseudo_zero_day_families,
    pseudo_zero_day_rotation_size,
    future_horizon_minutes,
    future_horizons_minutes,
    future_pre_onset_exclusion_gap_minutes=DEFAULT_FUTURE_PRE_ONSET_EXCLUSION_GAP_MINUTES,
    future_task_enabled,
    seq_len,
    stride,
    current_label_rule,
    train_target_positive_rate,
    threshold_target_recall,
    future_threshold_target_recall,
    ood_threshold_target_recall,
    known_target_unknown_recall,
    unknown_family_loss_weight,
    ood_loss_weight,
    family_loss_weight,
    future_loss_weight,
    reconstruction_loss_weight,
    family_sampler_power,
    future_refinement_epochs,
    future_refinement_lr,
    future_refinement_target_positive_rate,
    future_positive_boost,
    run_mode,
    thesis_claim,
    novelty_score_mode,
    decision_policy,
    unknown_head_policy,
    task_activation,
    use_reconstruction_hybrid_ood,
    unknown_risk_score_mode,
    unknown_head_active,
    ood_threshold_selection_policy,
    ood_max_fpr,
    reconstruction_train_mae_mask_ratio,
    reconstruction_train_mfm_mask_ratio,
    reconstruction_validation_mae_mask_ratio,
    reconstruction_validation_mfm_mask_ratio,
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
        "requested_pseudo_zero_day_families": requested_pseudo_zero_day_families,
        "pseudo_zero_day_family_count": pseudo_zero_day_family_count,
        "pseudo_zero_day_families": pseudo_zero_day_families,
        "epoch_unknown_attack_families": epoch_unknown_attack_families,
        "rotation_family_pool": rotation_family_pool,
        "rotate_pseudo_zero_day_families": rotate_pseudo_zero_day_families,
        "pseudo_zero_day_rotation_size": pseudo_zero_day_rotation_size,
        "future_horizon_minutes": future_horizon_minutes,
        "future_horizons_minutes": future_horizons_minutes,
        "future_pre_onset_exclusion_gap_minutes": float(future_pre_onset_exclusion_gap_minutes),
        "future_task_enabled": future_task_enabled,
        "seq_len": seq_len,
        "stride": stride,
        "current_label_rule": current_label_rule,
        "train_target_positive_rate": train_target_positive_rate,
        "threshold_target_recall": threshold_target_recall,
        "future_threshold_target_recall": future_threshold_target_recall,
        "ood_threshold_target_recall": ood_threshold_target_recall,
        "known_target_unknown_recall": known_target_unknown_recall,
        "unknown_family_loss_weight": unknown_family_loss_weight,
        "ood_loss_weight": ood_loss_weight,
        "family_loss_weight": family_loss_weight,
        "future_loss_weight": future_loss_weight,
        "reconstruction_loss_weight": reconstruction_loss_weight,
        "family_sampler_power": family_sampler_power,
        "future_refinement_epochs": future_refinement_epochs,
        "future_refinement_lr": future_refinement_lr,
        "future_refinement_target_positive_rate": future_refinement_target_positive_rate,
        "future_positive_boost": future_positive_boost,
        "run_mode": run_mode,
        "thesis_claim": thesis_claim,
        "novelty_score_mode": novelty_score_mode,
        "decision_policy": decision_policy,
        "unknown_head_policy": unknown_head_policy,
        "task_activation": task_activation,
        "use_reconstruction_hybrid_ood": use_reconstruction_hybrid_ood,
        "unknown_risk_score_mode": str(unknown_risk_score_mode),
        "reconstruction_train_mae_mask_ratio": reconstruction_train_mae_mask_ratio,
        "reconstruction_train_mfm_mask_ratio": reconstruction_train_mfm_mask_ratio,
        "reconstruction_validation_mae_mask_ratio": reconstruction_validation_mae_mask_ratio,
        "reconstruction_validation_mfm_mask_ratio": reconstruction_validation_mfm_mask_ratio,
        "loss_weights": loss_weights,
        "thresholds": thresholds,
        "best_threshold": thresholds["current"],
        "use_ood_head": bool(unknown_head_active),
        "unknown_head_active": bool(unknown_head_active),
        "validation_rank_version": VALIDATION_RANK_VERSION,
        "ood_threshold_selection_policy": str(ood_threshold_selection_policy),
        "ood_max_fpr": float(ood_max_fpr),
    }

    if validation_score is not None:
        payload["validation_score"] = validation_score
    if validation_rank is not None:
        payload["validation_rank"] = list(validation_rank)
    if validation_metrics is not None:
        payload["validation_metrics"] = validation_metrics
        payload["reconstruction_calibration"] = validation_metrics.get("reconstruction_calibration")
        payload["unknown_risk_score_mode"] = validation_metrics.get("unknown_risk_score_mode")
    if interruption_state is not None:
        payload.update(interruption_state)

    return payload


def load_foundation_mask_ratios(checkpoint_path, device):
    ratios = {
        "train_mae_mask_ratio": 0.10,
        "train_mfm_mask_ratio": 0.00,
        "validation_mae_mask_ratio": 0.30,
        "validation_mfm_mask_ratio": 0.10,
    }
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return ratios

    checkpoint = base_train.load_trusted_checkpoint(checkpoint_path, device)
    for field_name, fallback in list(ratios.items()):
        field_value = checkpoint.get(field_name)
        if field_value is None:
            ratios[field_name] = float(fallback)
            continue
        ratios[field_name] = float(field_value)
    return ratios


def train_multitask_nids_v3():
    args = parse_args()
    print("Starting downstream NIDS training v3...", flush=True)
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
    future_horizons_minutes = normalize_future_horizons_minutes(
        args.future_horizons_minutes,
        args.future_horizon_minutes,
    )
    future_horizon_minutes = future_horizons_minutes[-1]
    min_known_attack_count = args.min_known_attack_count
    future_task_enabled = args.enable_future_task
    num_workers = args.num_workers
    current_label_rule = args.current_label_rule
    rebuild_caches = args.rebuild_caches
    train_target_positive_rate = args.train_target_positive_rate
    threshold_target_recall = args.threshold_target_recall
    future_threshold_target_recall = args.future_threshold_target_recall
    future_pre_onset_exclusion_gap_minutes = float(args.future_pre_onset_exclusion_gap_minutes)
    ood_threshold_target_recall = args.ood_threshold_target_recall
    ood_threshold_selection_policy = str(args.ood_threshold_selection_policy)
    ood_max_fpr = float(args.ood_max_fpr)
    known_target_unknown_recall = args.known_target_unknown_recall
    unknown_family_loss_weight = args.unknown_family_loss_weight
    ood_loss_weight = args.ood_loss_weight
    family_loss_weight = args.family_loss_weight
    future_loss_weight = args.future_loss_weight if future_task_enabled else 0.0
    reconstruction_loss_weight = args.reconstruction_loss_weight
    family_sampler_power = args.family_sampler_power
    future_refinement_epochs = args.future_refinement_epochs
    future_refinement_lr = args.future_refinement_lr
    future_refinement_target_positive_rate = args.future_refinement_target_positive_rate
    family_refinement_epochs = args.family_refinement_epochs
    family_refinement_lr = args.family_refinement_lr
    family_refinement_sampler_power = args.family_refinement_sampler_power
    family_refinement_max_family_boost = args.family_refinement_max_family_boost
    family_refinement_label_smoothing = args.family_refinement_label_smoothing
    future_positive_boost = args.future_positive_boost
    use_reconstruction_hybrid_ood = bool(args.use_reconstruction_hybrid_ood)
    run_mode_settings = resolve_run_mode_settings(args)
    run_mode = run_mode_settings["run_mode"]
    requested_pseudo_zero_day_families = run_mode_settings["requested_pseudo_zero_day_families"]
    pseudo_zero_day_family_count = int(run_mode_settings["pseudo_zero_day_family_count"])
    rotate_pseudo_zero_day_families = bool(run_mode_settings["rotate_pseudo_zero_day_families"])
    pseudo_zero_day_rotation_size = int(run_mode_settings["pseudo_zero_day_rotation_size"])
    novelty_score_mode = run_mode_settings["novelty_score_mode"]
    decision_policy = run_mode_settings["decision_policy"]
    unknown_head_policy = run_mode_settings["unknown_head_policy"]
    thesis_claim = run_mode_settings["thesis_claim"]

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
    if future_pre_onset_exclusion_gap_minutes < 0.0:
        raise ValueError(
            "future_pre_onset_exclusion_gap_minutes must be >= 0, got "
            f"{future_pre_onset_exclusion_gap_minutes}"
        )
    if not 0.0 < ood_threshold_target_recall <= 1.0:
        raise ValueError(
            f"ood_threshold_target_recall must be in (0, 1], got {ood_threshold_target_recall}"
        )
    if ood_threshold_selection_policy not in {
        OOD_THRESHOLD_SELECTION_POLICY_TARGET_RECALL,
        OOD_THRESHOLD_SELECTION_POLICY_MAX_FPR,
    }:
        raise ValueError(
            "ood_threshold_selection_policy must be one of "
            f"{OOD_THRESHOLD_SELECTION_POLICY_TARGET_RECALL!r} or {OOD_THRESHOLD_SELECTION_POLICY_MAX_FPR!r}, "
            f"got {ood_threshold_selection_policy}"
        )
    if not 0.0 <= ood_max_fpr <= 1.0:
        raise ValueError(f"ood_max_fpr must be in [0, 1], got {ood_max_fpr}")
    if not 0.0 <= known_target_unknown_recall <= 1.0:
        raise ValueError(
            f"known_target_unknown_recall must be in [0, 1], got {known_target_unknown_recall}"
        )
    if unknown_family_loss_weight < 0.0:
        raise ValueError(
            f"unknown_family_loss_weight must be >= 0, got {unknown_family_loss_weight}"
        )
    if ood_loss_weight < 0.0:
        raise ValueError(f"ood_loss_weight must be >= 0, got {ood_loss_weight}")
    if family_loss_weight < 0.0:
        raise ValueError(f"family_loss_weight must be >= 0, got {family_loss_weight}")
    if future_loss_weight < 0.0:
        raise ValueError(f"future_loss_weight must be >= 0, got {future_loss_weight}")
    if reconstruction_loss_weight < 0.0:
        raise ValueError(
            f"reconstruction_loss_weight must be >= 0, got {reconstruction_loss_weight}"
        )
    if family_sampler_power < 0.0:
        raise ValueError(f"family_sampler_power must be >= 0, got {family_sampler_power}")
    if future_refinement_epochs < 0:
        raise ValueError(f"future_refinement_epochs must be >= 0, got {future_refinement_epochs}")
    if future_refinement_lr <= 0.0:
        raise ValueError(f"future_refinement_lr must be > 0, got {future_refinement_lr}")
    if not 0.0 < future_refinement_target_positive_rate < 1.0:
        raise ValueError(
            "future_refinement_target_positive_rate must be in (0, 1), got "
            f"{future_refinement_target_positive_rate}"
        )
    if family_refinement_epochs < 0:
        raise ValueError(f"family_refinement_epochs must be >= 0, got {family_refinement_epochs}")
    if family_refinement_lr <= 0.0:
        raise ValueError(f"family_refinement_lr must be > 0, got {family_refinement_lr}")
    if family_refinement_sampler_power < 0.0:
        raise ValueError(
            f"family_refinement_sampler_power must be >= 0, got {family_refinement_sampler_power}"
        )
    if family_refinement_max_family_boost < 1.0:
        raise ValueError(
            f"family_refinement_max_family_boost must be >= 1, got {family_refinement_max_family_boost}"
        )
    if not 0.0 <= family_refinement_label_smoothing < 1.0:
        raise ValueError(
            "family_refinement_label_smoothing must be in [0, 1), got "
            f"{family_refinement_label_smoothing}"
        )
    if future_positive_boost <= 0.0:
        raise ValueError(f"future_positive_boost must be > 0, got {future_positive_boost}")
    if pseudo_zero_day_rotation_size < 0:
        raise ValueError(
            f"pseudo_zero_day_rotation_size must be >= 0, got {pseudo_zero_day_rotation_size}"
        )

    loss_weights = {
        "current": 2.0,
        "family": family_loss_weight,
        "future": future_loss_weight,
        "ood": ood_loss_weight,
        "reconstruction": reconstruction_loss_weight,
        "unknown_regularizer": unknown_family_loss_weight,
    }
    thresholds = {
        "current": 0.50,
        "known": 0.55,
        "future": normalize_future_thresholds(None, future_horizons_minutes, default_threshold=0.50),
        "ood": 0.50,
    }

    train_dir = DEFAULT_TRAIN_DIR
    valid_dir = DEFAULT_VALID_DIR
    test_dir = DEFAULT_TEST_DIR
    stats_path = DEFAULT_STATS_PATH
    checkpoint_dir = args.output_dir
    future_refinement_output_dir = args.future_refinement_output_dir
    family_refinement_output_dir = args.family_refinement_output_dir
    refinement_source_dir = args.refinement_source_dir
    foundation_checkpoint = args.foundation_checkpoint
    resume_checkpoint = args.resume_checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)

    active_output_dirs = [os.path.abspath(checkpoint_dir)]
    if future_task_enabled or family_refinement_epochs > 0:
        active_output_dirs.append(os.path.abspath(future_refinement_output_dir))
    if family_refinement_epochs > 0:
        active_output_dirs.append(os.path.abspath(family_refinement_output_dir))
    if len(active_output_dirs) != len(set(active_output_dirs)):
        raise ValueError(
            "Main, future-refinement, and family-refinement output directories must be distinct when their stages are active."
        )

    # The baseline output directory should stay reusable across future-only and family-only reruns.
    base_expected_config = {
        "run_mode": run_mode,
        "seq_len": seq_len,
        "stride": stride,
        "current_label_rule": current_label_rule,
        "future_task_enabled": future_task_enabled,
        "future_horizons_minutes": future_horizons_minutes,
        "future_horizon_minutes": future_horizon_minutes,
        "future_pre_onset_exclusion_gap_minutes": future_pre_onset_exclusion_gap_minutes,
        "train_target_positive_rate": train_target_positive_rate,
        "threshold_target_recall": threshold_target_recall,
        "future_threshold_target_recall": future_threshold_target_recall,
        "ood_threshold_target_recall": ood_threshold_target_recall,
        "ood_threshold_selection_policy": ood_threshold_selection_policy,
        "ood_max_fpr": ood_max_fpr,
        "known_target_unknown_recall": known_target_unknown_recall,
        "unknown_family_loss_weight": unknown_family_loss_weight,
        "ood_loss_weight": ood_loss_weight,
        "family_loss_weight": family_loss_weight,
        "future_loss_weight": future_loss_weight,
        "reconstruction_loss_weight": reconstruction_loss_weight,
        "family_sampler_power": family_sampler_power,
        "future_positive_boost": future_positive_boost,
        "novelty_score_mode": novelty_score_mode,
        "decision_policy": decision_policy,
        "unknown_head_policy": unknown_head_policy,
        "thesis_claim": thesis_claim,
        "use_reconstruction_hybrid_ood": use_reconstruction_hybrid_ood,
        "rotate_pseudo_zero_day_families": rotate_pseudo_zero_day_families,
        "pseudo_zero_day_rotation_size": pseudo_zero_day_rotation_size,
        "requested_pseudo_zero_day_families": requested_pseudo_zero_day_families,
        "pseudo_zero_day_family_count": pseudo_zero_day_family_count,
    }

    print(f"Foundation checkpoint: {foundation_checkpoint}", flush=True)
    print(f"Downstream output directory: {checkpoint_dir}", flush=True)
    print(f"Future refinement output directory: {future_refinement_output_dir}", flush=True)
    print(f"Family refinement output directory: {family_refinement_output_dir}", flush=True)
    if refinement_source_dir:
        print(f"Refinement source directory: {refinement_source_dir}", flush=True)
    print(f"Run mode: {run_mode}", flush=True)
    print(f"Thesis claim policy: {thesis_claim}", flush=True)
    print(f"Future task enabled: {future_task_enabled}", flush=True)
    print(f"Future horizons minutes: {future_horizons_minutes}", flush=True)
    print(
        f"Future pre-onset exclusion gap minutes: {future_pre_onset_exclusion_gap_minutes:.3f}",
        flush=True,
    )
    print(f"Window config: seq_len={seq_len}, stride={stride}", flush=True)
    print(f"Current label rule: {current_label_rule}", flush=True)
    print(f"Train target positive rate: {train_target_positive_rate:.3f}", flush=True)
    print(f"Current threshold target recall: {threshold_target_recall:.3f}", flush=True)
    print(f"Future threshold target recall: {future_threshold_target_recall:.3f}", flush=True)
    print(f"OOD threshold target recall: {ood_threshold_target_recall:.3f}", flush=True)
    print(f"OOD threshold selection policy: {ood_threshold_selection_policy}", flush=True)
    print(f"OOD max FPR: {ood_max_fpr:.4f}", flush=True)
    print(f"Known target unknown recall: {known_target_unknown_recall:.3f}", flush=True)
    print(f"OOD loss weight: {ood_loss_weight:.3f}", flush=True)
    print(f"Family loss weight: {family_loss_weight:.3f}", flush=True)
    print(f"Future loss weight: {future_loss_weight:.3f}", flush=True)
    print(f"Reconstruction loss weight: {reconstruction_loss_weight:.3f}", flush=True)
    print(f"Unknown family loss weight: {unknown_family_loss_weight:.3f}", flush=True)
    print(f"Novelty score mode: {novelty_score_mode}", flush=True)
    print(f"Decision policy: {decision_policy}", flush=True)
    print(f"Unknown head policy: {unknown_head_policy}", flush=True)
    print(f"Hybrid reconstruction-backed unknown risk: {use_reconstruction_hybrid_ood}", flush=True)
    print(f"Family sampler power: {family_sampler_power:.3f}", flush=True)
    print(
        "Future refinement: "
        f"epochs={future_refinement_epochs}, lr={future_refinement_lr:.5f}, "
        f"target_positive_rate={future_refinement_target_positive_rate:.3f}",
        flush=True,
    )
    print(
        "Family refinement: "
        f"epochs={family_refinement_epochs}, lr={family_refinement_lr:.5f}, "
        f"sampler_power={family_refinement_sampler_power:.3f}, "
        f"max_boost={family_refinement_max_family_boost:.2f}, "
        f"label_smoothing={family_refinement_label_smoothing:.3f}",
        flush=True,
    )
    print(f"Future positive boost: {future_positive_boost:.3f}", flush=True)
    print(f"Rotate pseudo-zero-day families: {rotate_pseudo_zero_day_families}", flush=True)
    print(f"Pseudo-zero-day rotation size: {pseudo_zero_day_rotation_size}", flush=True)
    print(f"Requested pseudo-zero-day families: {requested_pseudo_zero_day_families}", flush=True)
    print(f"Pseudo-zero-day auto-select count: {pseudo_zero_day_family_count}", flush=True)
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
        future_horizons_minutes=future_horizons_minutes,
        future_pre_onset_exclusion_gap_minutes=future_pre_onset_exclusion_gap_minutes,
        known_attack_to_idx={},
        current_label_rule=current_label_rule,
        rebuild_target_cache=rebuild_caches,
        held_out_attack_families=[],
    )
    valid_probe_dataset = VariantDownstreamNIDSDataset(
        valid_base_dataset,
        future_horizons_minutes=future_horizons_minutes,
        future_pre_onset_exclusion_gap_minutes=future_pre_onset_exclusion_gap_minutes,
        known_attack_to_idx={},
        current_label_rule=current_label_rule,
        rebuild_target_cache=rebuild_caches,
        held_out_attack_families=[],
    )
    pseudo_zero_day_families = select_pseudo_zero_day_families(
        requested_pseudo_zero_day_families,
        train_probe_dataset.attack_family_counts,
        valid_probe_dataset.attack_family_counts,
        pseudo_zero_day_family_count,
    )
    if pseudo_zero_day_families:
        print(f"Pseudo-zero-day families: {pseudo_zero_day_families}", flush=True)
    else:
        print("Pseudo-zero-day families: disabled; all observed train families stay supervised.", flush=True)

    print("Preparing downstream train targets...", flush=True)
    train_dataset = VariantDownstreamNIDSDataset(
        train_base_dataset,
        future_horizons_minutes=future_horizons_minutes,
        future_pre_onset_exclusion_gap_minutes=future_pre_onset_exclusion_gap_minutes,
        min_known_attack_count=min_known_attack_count,
        current_label_rule=current_label_rule,
        rebuild_target_cache=rebuild_caches,
        held_out_attack_families=pseudo_zero_day_families,
    )
    print(f"Downstream train dataset ready: {len(train_dataset)} sequences", flush=True)
    print(f"Mapped attack families in train: {dict(train_dataset.attack_family_counts)}", flush=True)
    print(f"Known attack families: {train_dataset.known_attack_names}", flush=True)
    print(f"Unknown train windows: {int(train_dataset.unknown_attack_targets.sum())}", flush=True)

    print("Preparing downstream validation targets...", flush=True)
    valid_dataset = VariantDownstreamNIDSDataset(
        valid_base_dataset,
        future_horizons_minutes=future_horizons_minutes,
        future_pre_onset_exclusion_gap_minutes=future_pre_onset_exclusion_gap_minutes,
        known_attack_to_idx=train_dataset.known_attack_to_idx,
        current_label_rule=current_label_rule,
        rebuild_target_cache=rebuild_caches,
        held_out_attack_families=pseudo_zero_day_families,
    )
    print(f"Downstream validation dataset ready: {len(valid_dataset)} sequences", flush=True)
    print(f"Mapped attack families in validation: {dict(valid_dataset.attack_family_counts)}", flush=True)
    print(f"Unknown validation windows: {int(valid_dataset.unknown_attack_targets.sum())}", flush=True)

    train_unknown_positive_count = int(train_dataset.unknown_attack_targets.sum())
    valid_unknown_positive_count = int(valid_dataset.unknown_attack_targets.sum())
    rotation_family_pool = [
        family_name for family_name in train_dataset.known_attack_names if family_name not in pseudo_zero_day_families
    ]
    unknown_head_active = (
        train_unknown_positive_count > 0
        or (
            rotate_pseudo_zero_day_families
            and pseudo_zero_day_rotation_size > 0
            and bool(rotation_family_pool)
        )
    )
    unknown_risk_score_mode = resolve_unknown_risk_score_mode(
        args.unknown_risk_score_mode,
        use_reconstruction_hybrid_ood,
        unknown_head_active,
    )
    base_expected_config["unknown_risk_score_mode"] = unknown_risk_score_mode
    task_activation = build_task_activation_metadata(
        future_task_enabled=future_task_enabled,
        family_head_enabled=bool(train_dataset.known_attack_names),
        unknown_head_active=unknown_head_active,
        use_reconstruction_hybrid_ood=use_reconstruction_hybrid_ood,
        reconstruction_loss_weight=reconstruction_loss_weight,
    )
    print(f"Task activation: {task_activation}", flush=True)
    print(f"Unknown-risk score mode: {unknown_risk_score_mode}", flush=True)
    if not unknown_head_active:
        print(
            "Unknown head disabled for this run because the training split contains no unknown-labelled windows.",
            flush=True,
        )
    if rotate_pseudo_zero_day_families and rotation_family_pool:
        print(f"Epoch rotation family pool: {rotation_family_pool}", flush=True)
    else:
        print("Epoch rotation family pool: disabled", flush=True)

    variant_config_payload = {
        **base_expected_config,
        "effective_pseudo_zero_day_families": pseudo_zero_day_families,
        "known_attack_labels": train_dataset.known_attack_names,
        "train_attack_family_counts": dict(train_dataset.attack_family_counts),
        "validation_attack_family_counts": dict(valid_dataset.attack_family_counts),
        "task_activation": task_activation,
        "train_unknown_positive_count": train_unknown_positive_count,
        "validation_unknown_positive_count": valid_unknown_positive_count,
        "refinement_source_dir": refinement_source_dir,
        "future_refinement_output_dir": future_refinement_output_dir,
        "family_refinement_output_dir": family_refinement_output_dir,
    }

    dump_variant_config(checkpoint_dir, variant_config_payload)

    dump_run_manifest(
        checkpoint_dir,
        build_run_manifest_payload(
            run_mode=run_mode,
            thesis_claim=thesis_claim,
            novelty_score_mode=novelty_score_mode,
            decision_policy=decision_policy,
            unknown_head_policy=unknown_head_policy,
            task_activation=task_activation,
            requested_pseudo_zero_day_families=requested_pseudo_zero_day_families,
            pseudo_zero_day_families=pseudo_zero_day_families,
            rotate_pseudo_zero_day_families=rotate_pseudo_zero_day_families,
            pseudo_zero_day_rotation_size=pseudo_zero_day_rotation_size,
            future_horizons_minutes=future_horizons_minutes,
            thresholds=thresholds,
            loss_weights=loss_weights,
            known_attack_labels=train_dataset.known_attack_names,
            unknown_risk_score_mode=unknown_risk_score_mode,
            ood_threshold_selection_policy=ood_threshold_selection_policy,
            ood_max_fpr=ood_max_fpr,
        ),
    )

    attack_vocab_path = os.path.join(checkpoint_dir, "known_attack_labels.json")
    with open(attack_vocab_path, "w") as handle:
        json.dump({"known_attack_labels": train_dataset.known_attack_names}, handle, indent=2)
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
    foundation_mask_ratios = load_foundation_mask_ratios(foundation_checkpoint, device)
    backbone = SpatioTemporalTransformer(
        num_cont_features=num_cont,
        cat_vocab_sizes=CAT_VOCABS,
        seq_len=seq_len,
        init_mae=foundation_mask_ratios["train_mae_mask_ratio"],
        init_mfm=foundation_mask_ratios["train_mfm_mask_ratio"],
    )
    print(f"Loading foundation weights from {foundation_checkpoint}", flush=True)
    base_train.load_foundation_checkpoint(backbone, foundation_checkpoint, device)
    backbone.mae_mask_ratio = foundation_mask_ratios["train_mae_mask_ratio"]
    backbone.mfm_layer.mask_ratio = foundation_mask_ratios["train_mfm_mask_ratio"]
    print(
        "Foundation mask ratios restored: "
        f"train(MAE/MFM)={foundation_mask_ratios['train_mae_mask_ratio']:.3f}/{foundation_mask_ratios['train_mfm_mask_ratio']:.3f} | "
        f"validation(MAE/MFM)={foundation_mask_ratios['validation_mae_mask_ratio']:.3f}/{foundation_mask_ratios['validation_mfm_mask_ratio']:.3f}",
        flush=True,
    )
    model = NIDSMultiTaskModel(
        backbone=backbone,
        num_known_attack_classes=len(train_dataset.known_attack_names),
        use_future_head=future_task_enabled,
        use_ood_head=unknown_head_active,
        future_horizons_minutes=future_horizons_minutes,
    ).to(device)
    print(f"Downstream model ready on {device}", flush=True)

    if future_task_enabled:
        future_pos_weight = build_future_pos_weight_tensor(train_dataset, device)
    else:
        future_pos_weight = None

    positive_rate = float(train_dataset.sequence_current_labels.mean())
    focal_alpha = 1.0 - positive_rate
    current_loss_fn = base_train.FocalLoss(alpha=focal_alpha, gamma=3.0)
    future_loss_fn = nn.BCEWithLogitsLoss(pos_weight=future_pos_weight) if future_task_enabled else None
    ood_loss_fn = None
    if unknown_head_active and train_unknown_positive_count > 0:
        unknown_positive_count = max(train_unknown_positive_count, 1)
        known_positive_count = max(int(len(train_dataset)) - train_unknown_positive_count, 1)
        ood_pos_weight = torch.tensor(
            [known_positive_count / unknown_positive_count],
            device=device,
            dtype=torch.float32,
        )
        ood_loss_fn = nn.BCEWithLogitsLoss(pos_weight=ood_pos_weight)
    family_weight_tensor = build_family_weight_tensor(train_dataset, device)
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
    evaluate_downstream_v3.current_loss_fn = current_loss_fn
    evaluate_downstream_v3.future_loss_fn = future_loss_fn
    evaluate_downstream_v3.ood_loss_fn = ood_loss_fn
    evaluate_downstream_v3.family_loss_fn = family_loss_fn
    evaluate_downstream_v3.unknown_family_loss_fn = unknown_family_loss_fn
    evaluate_downstream_v3.loss_weights = loss_weights
    evaluate_downstream_v3.threshold_target_recall = threshold_target_recall
    evaluate_downstream_v3.future_threshold_target_recall = future_threshold_target_recall
    evaluate_downstream_v3.future_pre_onset_exclusion_gap_minutes = future_pre_onset_exclusion_gap_minutes
    evaluate_downstream_v3.ood_threshold_target_recall = ood_threshold_target_recall
    evaluate_downstream_v3.ood_threshold_selection_policy = ood_threshold_selection_policy
    evaluate_downstream_v3.ood_max_fpr = ood_max_fpr
    evaluate_downstream_v3.known_target_unknown_recall = known_target_unknown_recall
    evaluate_downstream_v3.future_horizons_minutes = future_horizons_minutes
    evaluate_downstream_v3.use_reconstruction_hybrid_ood = use_reconstruction_hybrid_ood
    evaluate_downstream_v3.unknown_risk_score_mode = unknown_risk_score_mode
    evaluate_downstream_v3.unknown_head_active = unknown_head_active
    evaluate_downstream_v3.novelty_score_mode = novelty_score_mode
    evaluate_downstream_v3.decision_policy = decision_policy
    evaluate_downstream_v3.reconstruction_validation_mae_mask_ratio = foundation_mask_ratios[
        "validation_mae_mask_ratio"
    ]
    evaluate_downstream_v3.reconstruction_validation_mfm_mask_ratio = foundation_mask_ratios[
        "validation_mfm_mask_ratio"
    ]
    evaluate_downstream_v3.reconstruction_validation_mask_seed = DEFAULT_RECONSTRUCTION_VALIDATION_MASK_SEED

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

    estimated_steps_per_epoch = max(int(math.ceil(max(len(train_dataset), 1) / batch_size)), 1)
    total_steps = max(estimated_steps_per_epoch * epochs, 1)
    warmup_steps = max(estimated_steps_per_epoch * warmup_epochs, 1)

    def lr_lambda(step):
        if step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    start_epoch = 0
    best_rank = (-1.0, -float("inf"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    best_rank = load_best_validation_rank(checkpoint_dir, device, best_rank)

    if best_rank[1] > -float("inf"):
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

        checkpoint_future_horizons = normalize_future_horizons_minutes(
            resume_state.get("future_horizons_minutes"),
            resume_state.get("future_horizon_minutes"),
        )
        if checkpoint_future_horizons != future_horizons_minutes:
            raise ValueError(
                "Resume checkpoint future-horizon configuration does not match the current configuration. "
                f"Checkpoint future_horizons_minutes={checkpoint_future_horizons} | Requested future_horizons_minutes={future_horizons_minutes}"
            )

        for field_name, expected_value in (
            ("run_mode", run_mode),
            ("train_target_positive_rate", train_target_positive_rate),
            ("threshold_target_recall", threshold_target_recall),
            ("future_threshold_target_recall", future_threshold_target_recall),
            ("ood_threshold_target_recall", ood_threshold_target_recall),
            ("ood_threshold_selection_policy", ood_threshold_selection_policy),
            ("ood_max_fpr", ood_max_fpr),
            ("known_target_unknown_recall", known_target_unknown_recall),
            ("unknown_family_loss_weight", unknown_family_loss_weight),
            ("ood_loss_weight", ood_loss_weight),
            ("family_loss_weight", family_loss_weight),
            ("future_loss_weight", future_loss_weight),
            ("reconstruction_loss_weight", reconstruction_loss_weight),
            ("family_sampler_power", family_sampler_power),
            ("future_positive_boost", future_positive_boost),
            ("novelty_score_mode", novelty_score_mode),
            ("decision_policy", decision_policy),
            ("unknown_head_policy", unknown_head_policy),
            ("thesis_claim", thesis_claim),
            ("use_reconstruction_hybrid_ood", use_reconstruction_hybrid_ood),
            ("unknown_risk_score_mode", unknown_risk_score_mode),
            ("unknown_head_active", unknown_head_active),
            ("rotate_pseudo_zero_day_families", rotate_pseudo_zero_day_families),
            ("pseudo_zero_day_rotation_size", pseudo_zero_day_rotation_size),
            ("pseudo_zero_day_family_count", pseudo_zero_day_family_count),
        ):
            existing_value = resume_state.get(field_name)
            if existing_value is None:
                continue
            if isinstance(expected_value, float):
                values_match = float(existing_value) == float(expected_value)
            else:
                values_match = existing_value == expected_value
            if not values_match:
                raise ValueError(
                    f"Resume checkpoint {field_name}={existing_value} | requested {expected_value}"
                )

        checkpoint_requested_pseudo_zero_day_families = list(
            resume_state.get("requested_pseudo_zero_day_families", [])
        )
        if checkpoint_requested_pseudo_zero_day_families != requested_pseudo_zero_day_families:
            raise ValueError(
                "Resume checkpoint requested pseudo-zero-day families do not match the current configuration. "
                f"Checkpoint families: {checkpoint_requested_pseudo_zero_day_families} | Requested families: {requested_pseudo_zero_day_families}"
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
        thresholds["future"] = normalize_future_thresholds(
            thresholds.get("future"),
            future_horizons_minutes,
            default_threshold=0.50,
        )

        foundation_checkpoint = resume_state.get("foundation_checkpoint", foundation_checkpoint)
        resume_epoch = resume_state.get("resume_epoch")
        if resume_epoch is not None:
            start_epoch = int(resume_epoch)
        else:
            start_epoch = int(resume_state.get("epoch", -1)) + 1
        resume_metrics = resume_state.get("validation_metrics")
        if resume_metrics is not None:
            if "best_future" in resume_metrics and "best_known" in resume_metrics and "best_ood" in resume_metrics:
                resume_rank = build_validation_rank(
                    resume_metrics,
                    future_task_enabled,
                    ood_threshold_target_recall=ood_threshold_target_recall,
                )
            else:
                resume_score = float(resume_state.get("validation_score", -float("inf")))
                resume_rank = (0.0, resume_score, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        else:
            resume_score = float(resume_state.get("validation_score", -float("inf")))
            resume_rank = (0.0, resume_score, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        best_rank = load_best_validation_rank(checkpoint_dir, device, resume_rank)

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

    def build_manifest_payload_for_current_state(current_validation_metrics):
        return build_run_manifest_payload(
            run_mode=run_mode,
            thesis_claim=thesis_claim,
            novelty_score_mode=novelty_score_mode,
            decision_policy=decision_policy,
            unknown_head_policy=unknown_head_policy,
            task_activation=task_activation,
            requested_pseudo_zero_day_families=requested_pseudo_zero_day_families,
            pseudo_zero_day_families=pseudo_zero_day_families,
            rotate_pseudo_zero_day_families=rotate_pseudo_zero_day_families,
            pseudo_zero_day_rotation_size=pseudo_zero_day_rotation_size,
            future_horizons_minutes=future_horizons_minutes,
            thresholds=thresholds,
            loss_weights=loss_weights,
            known_attack_labels=train_dataset.known_attack_names,
            unknown_risk_score_mode=unknown_risk_score_mode,
            ood_threshold_selection_policy=ood_threshold_selection_policy,
            ood_max_fpr=ood_max_fpr,
            validation_metrics=current_validation_metrics,
        )

    def build_checkpoint_payload_for_current_state(
        current_optimizer,
        current_scheduler,
        *,
        validation_score,
        validation_rank,
        validation_metrics,
        payload_epoch,
        active_checkpoint_dir=None,
    ):
        payload_checkpoint_dir = active_checkpoint_dir or checkpoint_dir
        return build_training_checkpoint_payload(
            epoch=payload_epoch,
            model=model,
            optimizer=current_optimizer,
            scheduler=current_scheduler,
            foundation_checkpoint=foundation_checkpoint,
            checkpoint_dir=payload_checkpoint_dir,
            known_attack_labels=train_dataset.known_attack_names,
            requested_pseudo_zero_day_families=requested_pseudo_zero_day_families,
            pseudo_zero_day_family_count=pseudo_zero_day_family_count,
            pseudo_zero_day_families=pseudo_zero_day_families,
            epoch_unknown_attack_families=sorted(train_dataset.epoch_unknown_attack_families),
            rotation_family_pool=rotation_family_pool,
            rotate_pseudo_zero_day_families=rotate_pseudo_zero_day_families,
            pseudo_zero_day_rotation_size=pseudo_zero_day_rotation_size,
            future_horizon_minutes=future_horizon_minutes,
            future_horizons_minutes=future_horizons_minutes,
            future_pre_onset_exclusion_gap_minutes=future_pre_onset_exclusion_gap_minutes,
            future_task_enabled=future_task_enabled,
            seq_len=seq_len,
            stride=stride,
            current_label_rule=current_label_rule,
            train_target_positive_rate=train_target_positive_rate,
            threshold_target_recall=threshold_target_recall,
            future_threshold_target_recall=future_threshold_target_recall,
            ood_threshold_target_recall=ood_threshold_target_recall,
            known_target_unknown_recall=known_target_unknown_recall,
            unknown_family_loss_weight=unknown_family_loss_weight,
            ood_loss_weight=ood_loss_weight,
            family_loss_weight=family_loss_weight,
            future_loss_weight=future_loss_weight,
            reconstruction_loss_weight=reconstruction_loss_weight,
            family_sampler_power=family_sampler_power,
            future_refinement_epochs=future_refinement_epochs,
            future_refinement_lr=future_refinement_lr,
            future_refinement_target_positive_rate=future_refinement_target_positive_rate,
            future_positive_boost=future_positive_boost,
            run_mode=run_mode,
            thesis_claim=thesis_claim,
            novelty_score_mode=novelty_score_mode,
            decision_policy=decision_policy,
            unknown_head_policy=unknown_head_policy,
            task_activation=task_activation,
            use_reconstruction_hybrid_ood=use_reconstruction_hybrid_ood,
            unknown_risk_score_mode=unknown_risk_score_mode,
            unknown_head_active=unknown_head_active,
            ood_threshold_selection_policy=ood_threshold_selection_policy,
            ood_max_fpr=ood_max_fpr,
            reconstruction_train_mae_mask_ratio=foundation_mask_ratios["train_mae_mask_ratio"],
            reconstruction_train_mfm_mask_ratio=foundation_mask_ratios["train_mfm_mask_ratio"],
            reconstruction_validation_mae_mask_ratio=foundation_mask_ratios["validation_mae_mask_ratio"],
            reconstruction_validation_mfm_mask_ratio=foundation_mask_ratios["validation_mfm_mask_ratio"],
            loss_weights=loss_weights,
            thresholds=thresholds,
            validation_score=validation_score,
            validation_rank=validation_rank,
            validation_metrics=validation_metrics,
        )

    for epoch in range(start_epoch, epochs):
        print(f"Starting downstream epoch {epoch + 1}/{epochs}...", flush=True)
        epoch_unknown_attack_families = []
        if rotate_pseudo_zero_day_families and rotation_family_pool:
            epoch_unknown_attack_families = select_epoch_unknown_attack_families(
                rotation_family_pool,
                pseudo_zero_day_rotation_size,
                epoch,
            )
        train_dataset.set_epoch_unknown_attack_families(epoch_unknown_attack_families)
        print(
            f"Epoch {epoch + 1} surrogate unknown families: {train_dataset.active_unknown_attack_families}",
            flush=True,
        )

        train_loader, observed_positive_rate, effective_positive_rate, family_sampler_factors, sampler_stats = (
            build_train_loader_for_epoch(
                train_dataset,
                batch_size,
                num_workers,
                device,
                train_target_positive_rate,
                family_sampler_power,
                future_positive_boost,
            )
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
        print(f"Train loader ready: {len(train_loader)} batches per epoch", flush=True)

        family_weight_tensor = build_family_weight_tensor(train_dataset, device)
        epoch_unknown_positive_count = int(train_dataset.unknown_attack_targets.sum())
        if unknown_head_active and epoch_unknown_positive_count > 0:
            ood_pos_weight = torch.tensor(
                base_train.build_pos_weight(train_dataset.unknown_attack_targets),
                dtype=torch.float32,
                device=device,
            )
            ood_loss_fn = nn.BCEWithLogitsLoss(pos_weight=ood_pos_weight)
        else:
            ood_loss_fn = None
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

        evaluate_downstream_v3.ood_loss_fn = ood_loss_fn
        evaluate_downstream_v3.family_loss_fn = family_loss_fn
        evaluate_downstream_v3.unknown_family_loss_fn = unknown_family_loss_fn

        backbone_trainable = epoch >= freeze_backbone_epochs
        base_train.set_backbone_trainable(model, backbone_trainable)
        model.train()

        running_losses = {
            "total": 0.0,
            "current": 0.0,
            "family": 0.0,
            "future": 0.0,
            "ood": 0.0,
            "reconstruction": 0.0,
            "reconstruction_masked_mse": 0.0,
            "reconstruction_full_mse": 0.0,
            "unknown_regularizer": 0.0,
        }
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

                outputs = model(
                    cont,
                    cat,
                    apply_mfm=False,
                    compute_reconstruction=reconstruction_loss_weight > 0.0,
                    reconstruction_mask=None,
                    reconstruction_apply_mfm=None,
                )
                losses = compute_multitask_losses_v3(
                    outputs,
                    {
                        "continuous": cont,
                        "label": label,
                        "future_attack": future_attack,
                        "known_attack_id": known_attack_id,
                        "unknown_attack_target": unknown_attack_target,
                    },
                    current_loss_fn,
                    family_loss_fn,
                    future_loss_fn,
                    ood_loss_fn,
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
                        "ood": f"{running_losses['ood'] / (step + 1):.4f}",
                        "recon": f"{running_losses['reconstruction'] / (step + 1):.4f}",
                        "ureg": f"{running_losses['unknown_regularizer'] / (step + 1):.4f}",
                    }
                )

            progress_bar.close()
            progress_bar = None

            validation_metrics = evaluate_downstream_v3(model, valid_loader, device, thresholds)
            best_current = validation_metrics["best_current"]
            best_ood = validation_metrics["best_ood"]
            best_future = validation_metrics.get("best_future") or aggregate_future_metrics({})
            best_known = validation_metrics["best_known"]
            thresholds["current"] = best_current["threshold"]
            thresholds["known"] = best_known["threshold"]
            thresholds["future"] = (
                {
                    horizon_label: metrics["threshold"]
                    for horizon_label, metrics in validation_metrics.get("best_future_by_horizon", {}).items()
                }
                if future_task_enabled
                else {}
            )
            thresholds["ood"] = best_ood["threshold"]
            validation_rank = build_validation_rank(
                validation_metrics,
                future_task_enabled,
                ood_threshold_target_recall=ood_threshold_target_recall,
            )
            validation_score = validation_rank[1]

            dump_run_manifest(
                checkpoint_dir,
                build_run_manifest_payload(
                    run_mode=run_mode,
                    thesis_claim=thesis_claim,
                    novelty_score_mode=novelty_score_mode,
                    decision_policy=decision_policy,
                    unknown_head_policy=unknown_head_policy,
                    task_activation=task_activation,
                    requested_pseudo_zero_day_families=requested_pseudo_zero_day_families,
                    pseudo_zero_day_families=pseudo_zero_day_families,
                    rotate_pseudo_zero_day_families=rotate_pseudo_zero_day_families,
                    pseudo_zero_day_rotation_size=pseudo_zero_day_rotation_size,
                    future_horizons_minutes=future_horizons_minutes,
                    thresholds=thresholds,
                    loss_weights=loss_weights,
                    known_attack_labels=train_dataset.known_attack_names,
                    unknown_risk_score_mode=unknown_risk_score_mode,
                    ood_threshold_selection_policy=ood_threshold_selection_policy,
                    ood_max_fpr=ood_max_fpr,
                    validation_metrics=validation_metrics,
                ),
            )

            checkpoint_payload = build_training_checkpoint_payload(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                foundation_checkpoint=foundation_checkpoint,
                checkpoint_dir=checkpoint_dir,
                known_attack_labels=train_dataset.known_attack_names,
                requested_pseudo_zero_day_families=requested_pseudo_zero_day_families,
                pseudo_zero_day_family_count=pseudo_zero_day_family_count,
                pseudo_zero_day_families=pseudo_zero_day_families,
                epoch_unknown_attack_families=sorted(train_dataset.epoch_unknown_attack_families),
                rotation_family_pool=rotation_family_pool,
                rotate_pseudo_zero_day_families=rotate_pseudo_zero_day_families,
                pseudo_zero_day_rotation_size=pseudo_zero_day_rotation_size,
                future_horizon_minutes=future_horizon_minutes,
                future_horizons_minutes=future_horizons_minutes,
                future_pre_onset_exclusion_gap_minutes=future_pre_onset_exclusion_gap_minutes,
                future_task_enabled=future_task_enabled,
                seq_len=seq_len,
                stride=stride,
                current_label_rule=current_label_rule,
                train_target_positive_rate=train_target_positive_rate,
                threshold_target_recall=threshold_target_recall,
                future_threshold_target_recall=future_threshold_target_recall,
                ood_threshold_target_recall=ood_threshold_target_recall,
                known_target_unknown_recall=known_target_unknown_recall,
                unknown_family_loss_weight=unknown_family_loss_weight,
                ood_loss_weight=ood_loss_weight,
                family_loss_weight=family_loss_weight,
                future_loss_weight=future_loss_weight,
                reconstruction_loss_weight=reconstruction_loss_weight,
                family_sampler_power=family_sampler_power,
                future_refinement_epochs=future_refinement_epochs,
                future_refinement_lr=future_refinement_lr,
                future_refinement_target_positive_rate=future_refinement_target_positive_rate,
                future_positive_boost=future_positive_boost,
                run_mode=run_mode,
                thesis_claim=thesis_claim,
                novelty_score_mode=novelty_score_mode,
                decision_policy=decision_policy,
                unknown_head_policy=unknown_head_policy,
                task_activation=task_activation,
                use_reconstruction_hybrid_ood=use_reconstruction_hybrid_ood,
                unknown_risk_score_mode=unknown_risk_score_mode,
                unknown_head_active=unknown_head_active,
                ood_threshold_selection_policy=ood_threshold_selection_policy,
                ood_max_fpr=ood_max_fpr,
                reconstruction_train_mae_mask_ratio=foundation_mask_ratios["train_mae_mask_ratio"],
                reconstruction_train_mfm_mask_ratio=foundation_mask_ratios["train_mfm_mask_ratio"],
                reconstruction_validation_mae_mask_ratio=foundation_mask_ratios["validation_mae_mask_ratio"],
                reconstruction_validation_mfm_mask_ratio=foundation_mask_ratios["validation_mfm_mask_ratio"],
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
                f"UnknownRiskThresh={best_ood['threshold']:.3f} "
                f"KnownThresh={best_known['threshold']:.3f} "
                f"CurrentPRAUC={best_current['pr_auc']:.4f} CurrentF1={best_current['f1']:.4f} | "
                f"UnknownRiskPRAUC={best_ood['pr_auc']:.4f} UnknownRiskF1={best_ood['f1']:.4f} UnknownRiskRecall={best_ood['recall']:.4f} | "
                f"RawUnknownHeadPRAUC={validation_metrics['best_ood_head']['pr_auc']:.4f} ReconNoveltyPRAUC={validation_metrics['best_reconstruction_unknown']['pr_auc']:.4f} | "
                f"FutureMacroPRAUC={format_metric_value(best_future.get('pr_auc', float('nan')), 4)} "
                f"FutureMacroF1={format_metric_value(best_future.get('f1', float('nan')), 4)} | "
                f"KnownAcceptedAcc={best_known['accepted_accuracy']:.4f} KnownCoverage={best_known['known_coverage']:.4f} | "
                f"FamilyUnknownRecall={best_known['unknown_recall']:.4f} | "
                f"RawKnownAcc={validation_metrics['known_family_accuracy']:.4f} | "
                f"UnknownHeadActive={validation_metrics['unknown_head_active']} "
                f"NoveltyMode={validation_metrics['reconstruction_score_mode']} | "
                f"ReconLoss={validation_metrics['loss'].get('reconstruction', 0.0):.4f} | "
                f"CompositeScore={validation_score:.4f}",
                flush=True,
            )
            if future_task_enabled:
                print(
                    f" Future horizons: {summarize_future_metrics_by_horizon(validation_metrics.get('best_future_by_horizon', {}))}",
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
                requested_pseudo_zero_day_families=requested_pseudo_zero_day_families,
                pseudo_zero_day_family_count=pseudo_zero_day_family_count,
                pseudo_zero_day_families=pseudo_zero_day_families,
                epoch_unknown_attack_families=sorted(train_dataset.epoch_unknown_attack_families),
                rotation_family_pool=rotation_family_pool,
                rotate_pseudo_zero_day_families=rotate_pseudo_zero_day_families,
                pseudo_zero_day_rotation_size=pseudo_zero_day_rotation_size,
                future_horizon_minutes=future_horizon_minutes,
                future_horizons_minutes=future_horizons_minutes,
                future_pre_onset_exclusion_gap_minutes=future_pre_onset_exclusion_gap_minutes,
                future_task_enabled=future_task_enabled,
                seq_len=seq_len,
                stride=stride,
                current_label_rule=current_label_rule,
                train_target_positive_rate=train_target_positive_rate,
                threshold_target_recall=threshold_target_recall,
                future_threshold_target_recall=future_threshold_target_recall,
                ood_threshold_target_recall=ood_threshold_target_recall,
                known_target_unknown_recall=known_target_unknown_recall,
                unknown_family_loss_weight=unknown_family_loss_weight,
                ood_loss_weight=ood_loss_weight,
                family_loss_weight=family_loss_weight,
                future_loss_weight=future_loss_weight,
                reconstruction_loss_weight=reconstruction_loss_weight,
                family_sampler_power=family_sampler_power,
                future_refinement_epochs=future_refinement_epochs,
                future_refinement_lr=future_refinement_lr,
                future_refinement_target_positive_rate=future_refinement_target_positive_rate,
                future_positive_boost=future_positive_boost,
                run_mode=run_mode,
                thesis_claim=thesis_claim,
                novelty_score_mode=novelty_score_mode,
                decision_policy=decision_policy,
                unknown_head_policy=unknown_head_policy,
                task_activation=task_activation,
                use_reconstruction_hybrid_ood=use_reconstruction_hybrid_ood,
                unknown_risk_score_mode=unknown_risk_score_mode,
                unknown_head_active=unknown_head_active,
                ood_threshold_selection_policy=ood_threshold_selection_policy,
                ood_max_fpr=ood_max_fpr,
                reconstruction_train_mae_mask_ratio=foundation_mask_ratios["train_mae_mask_ratio"],
                reconstruction_train_mfm_mask_ratio=foundation_mask_ratios["train_mfm_mask_ratio"],
                reconstruction_validation_mae_mask_ratio=foundation_mask_ratios["validation_mae_mask_ratio"],
                reconstruction_validation_mfm_mask_ratio=foundation_mask_ratios["validation_mfm_mask_ratio"],
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

    future_refinement_seeded_rank = best_rank
    future_refinement_source_dir = refinement_source_dir or checkpoint_dir
    future_stage_enabled = future_task_enabled and future_refinement_epochs > 0
    family_stage_enabled = family_refinement_epochs > 0
    existing_future_refinement_best = Path(future_refinement_output_dir) / "nids_multitask_best.pt"
    use_existing_future_refinement_best = (
        future_task_enabled
        and family_stage_enabled
        and not future_stage_enabled
        and existing_future_refinement_best.exists()
    )
    future_stage_seed_required = future_task_enabled and (
        future_refinement_epochs > 0
        or (family_refinement_epochs > 0 and not use_existing_future_refinement_best)
    )

    if use_existing_future_refinement_best:
        future_refinement_source_dir = future_refinement_output_dir
        future_refinement_seeded_rank = load_best_validation_rank(
            future_refinement_output_dir,
            device,
            best_rank,
        )
        print(
            "Using existing future refinement best checkpoint as the family refinement source: "
            f"{existing_future_refinement_best}",
            flush=True,
        )

    if future_stage_seed_required:
        os.makedirs(future_refinement_output_dir, exist_ok=True)
        thresholds, source_best_path, multitask_best_state = restore_best_checkpoint_for_refinement(
            future_refinement_source_dir,
            device,
            model,
            thresholds,
            future_horizons_minutes,
            "future refinement",
        )

        reference_validation_metrics = None
        if multitask_best_state is not None:
            source_future_horizons = normalize_future_horizons_minutes(
                multitask_best_state.get("future_horizons_minutes"),
                multitask_best_state.get("future_horizon_minutes"),
            )
            if source_future_horizons == future_horizons_minutes:
                reference_validation_metrics = multitask_best_state.get("validation_metrics")
        if reference_validation_metrics is None:
            model.eval()
            reference_validation_metrics = evaluate_downstream_v3(model, valid_loader, device, thresholds)

        reference_validation_rank = build_validation_rank(
            reference_validation_metrics,
            future_task_enabled,
            ood_threshold_target_recall=ood_threshold_target_recall,
        )
        future_refinement_seeded_rank = reference_validation_rank

        if multitask_best_state is not None:
            seed_refinement_checkpoint_dir(
                source_best_path,
                multitask_best_state,
                future_refinement_output_dir,
                config_payload={
                    **variant_config_payload,
                    "refinement_stage": "future",
                    "refinement_source_dir": future_refinement_source_dir,
                },
                manifest_payload=build_run_manifest_payload(
                    run_mode=run_mode,
                    thesis_claim=thesis_claim,
                    novelty_score_mode=novelty_score_mode,
                    decision_policy=decision_policy,
                    unknown_head_policy=unknown_head_policy,
                    task_activation=task_activation,
                    requested_pseudo_zero_day_families=requested_pseudo_zero_day_families,
                    pseudo_zero_day_families=pseudo_zero_day_families,
                    rotate_pseudo_zero_day_families=rotate_pseudo_zero_day_families,
                    pseudo_zero_day_rotation_size=pseudo_zero_day_rotation_size,
                    future_horizons_minutes=future_horizons_minutes,
                    thresholds=thresholds,
                    loss_weights=loss_weights,
                    known_attack_labels=train_dataset.known_attack_names,
                    unknown_risk_score_mode=unknown_risk_score_mode,
                    ood_threshold_selection_policy=ood_threshold_selection_policy,
                    ood_max_fpr=ood_max_fpr,
                    validation_metrics=reference_validation_metrics,
                ),
                checkpoint_overrides={
                    "model_state_dict": model.state_dict(),
                    "thresholds": thresholds,
                    "future_horizons_minutes": future_horizons_minutes,
                    "future_horizon_minutes": future_horizon_minutes,
                    "validation_metrics": reference_validation_metrics,
                    "validation_rank": list(reference_validation_rank),
                    "validation_rank_version": VALIDATION_RANK_VERSION,
                    "validation_score": reference_validation_rank[1],
                },
            )

        if future_stage_enabled:
            future_refinement_seeded_rank, future_refined_metrics, _, _ = run_future_refinement_stage(
                model=model,
                device=device,
                train_dataset=train_dataset,
                valid_loader=valid_loader,
                batch_size=batch_size,
                num_workers=num_workers,
                future_refinement_epochs=future_refinement_epochs,
                future_refinement_lr=future_refinement_lr,
                future_refinement_target_positive_rate=future_refinement_target_positive_rate,
                thresholds=thresholds,
                checkpoint_dir=future_refinement_output_dir,
                build_checkpoint_payload=lambda current_optimizer, current_scheduler, *, validation_score, validation_rank, validation_metrics, payload_epoch: build_checkpoint_payload_for_current_state(
                    current_optimizer,
                    current_scheduler,
                    validation_score=validation_score,
                    validation_rank=validation_rank,
                    validation_metrics=validation_metrics,
                    payload_epoch=payload_epoch,
                    active_checkpoint_dir=future_refinement_output_dir,
                ),
                build_manifest_payload=build_manifest_payload_for_current_state,
                validation_rank_builder=lambda metrics: build_validation_rank(
                    metrics,
                    future_task_enabled,
                    ood_threshold_target_recall=ood_threshold_target_recall,
                ),
                best_rank=future_refinement_seeded_rank,
                reference_validation_metrics=reference_validation_metrics,
                checkpoint_epoch_offset=int(multitask_best_state.get("epoch", -1)) if multitask_best_state is not None else -1,
            )
            if future_refined_metrics is not None:
                print(
                    "Future refinement finished. "
                    f"FutureMacroPRAUC={safe_metric((future_refined_metrics.get('best_future') or {}).get('pr_auc', 0.0)):.4f} | "
                    f"FutureMacroF1={safe_metric((future_refined_metrics.get('best_future') or {}).get('f1', 0.0)):.4f}",
                    flush=True,
                )
        else:
            print(
                "Future refinement epochs set to 0. Keeping the seeded baseline checkpoint in the future refinement directory.",
                flush=True,
            )
        future_refinement_source_dir = future_refinement_output_dir

    family_refinement_source_dir = future_refinement_source_dir
    os.makedirs(family_refinement_output_dir, exist_ok=True)
    thresholds, multitask_best_path, multitask_best_state = restore_best_checkpoint_for_refinement(
        family_refinement_source_dir,
        device,
        model,
        thresholds,
        future_horizons_minutes,
        "family refinement",
    )

    reference_family_validation_metrics = None
    if multitask_best_state is not None:
        reference_family_validation_metrics = multitask_best_state.get("validation_metrics")
    if (
        reference_family_validation_metrics is None
        or "known_family_macro_f1" not in reference_family_validation_metrics
        or "accepted_macro_f1" not in (reference_family_validation_metrics.get("best_known") or {})
    ):
        model.eval()
        reference_family_validation_metrics = evaluate_downstream_v3(model, valid_loader, device, thresholds)

    family_refinement_stage_rank = (
        -1.0,
        -float("inf"),
        -float("inf"),
        -float("inf"),
        -float("inf"),
        -float("inf"),
        -float("inf"),
        -float("inf"),
    )
    if multitask_best_state is not None:
        seed_refinement_checkpoint_dir(
            multitask_best_path,
            multitask_best_state,
            family_refinement_output_dir,
            config_payload={
                **variant_config_payload,
                "refinement_stage": "family",
                "refinement_source_dir": family_refinement_source_dir,
            },
            manifest_payload=build_run_manifest_payload(
                run_mode=run_mode,
                thesis_claim=thesis_claim,
                novelty_score_mode=novelty_score_mode,
                decision_policy=decision_policy,
                unknown_head_policy=unknown_head_policy,
                task_activation=task_activation,
                requested_pseudo_zero_day_families=requested_pseudo_zero_day_families,
                pseudo_zero_day_families=pseudo_zero_day_families,
                rotate_pseudo_zero_day_families=rotate_pseudo_zero_day_families,
                pseudo_zero_day_rotation_size=pseudo_zero_day_rotation_size,
                future_horizons_minutes=future_horizons_minutes,
                thresholds=thresholds,
                loss_weights=loss_weights,
                known_attack_labels=train_dataset.known_attack_names,
                unknown_risk_score_mode=unknown_risk_score_mode,
                ood_threshold_selection_policy=ood_threshold_selection_policy,
                ood_max_fpr=ood_max_fpr,
                validation_metrics=reference_family_validation_metrics,
            ),
        )
        if reference_family_validation_metrics is not None:
            family_refinement_stage_rank = build_family_refinement_rank(reference_family_validation_metrics)

    family_refinement_stage_rank, family_refined_metrics, _, _ = run_family_refinement_stage(
        model=model,
        device=device,
        train_dataset=train_dataset,
        valid_loader=valid_loader,
        batch_size=batch_size,
        num_workers=num_workers,
        family_refinement_epochs=family_refinement_epochs,
        family_refinement_lr=family_refinement_lr,
        family_refinement_sampler_power=family_refinement_sampler_power,
        family_refinement_max_family_boost=family_refinement_max_family_boost,
        family_refinement_label_smoothing=family_refinement_label_smoothing,
        thresholds=thresholds,
        checkpoint_dir=family_refinement_output_dir,
        build_checkpoint_payload=lambda current_optimizer, current_scheduler, *, validation_score, validation_rank, validation_metrics, payload_epoch: build_checkpoint_payload_for_current_state(
            current_optimizer,
            current_scheduler,
            validation_score=validation_score,
            validation_rank=validation_rank,
            validation_metrics=validation_metrics,
            payload_epoch=payload_epoch,
            active_checkpoint_dir=family_refinement_output_dir,
        ),
        build_manifest_payload=build_manifest_payload_for_current_state,
        validation_rank_builder=build_family_refinement_rank,
        best_rank=family_refinement_stage_rank,
        reference_validation_metrics=reference_family_validation_metrics,
        checkpoint_epoch_offset=int(multitask_best_state.get("epoch", -1)) if multitask_best_state is not None else -1,
    )
    if family_refined_metrics is not None:
        print(
            "Family refinement finished. "
            f"RawKnownAcc={family_refined_metrics['known_family_accuracy']:.4f} | "
            f"RawKnownMacroF1={safe_metric(family_refined_metrics.get('known_family_macro_f1', 0.0)):.4f} | "
            f"KnownAcceptedAcc={family_refined_metrics['known_family_accepted_accuracy']:.4f} | "
            f"AcceptedKnownMacroF1={safe_metric(family_refined_metrics.get('known_family_accepted_macro_f1', 0.0)):.4f} | "
            f"KnownCoverage={family_refined_metrics['known_family_coverage']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    train_multitask_nids_v3()