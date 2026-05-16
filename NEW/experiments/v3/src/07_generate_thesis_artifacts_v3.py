import argparse
import gc
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm


EXPERIMENT_SRC_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = EXPERIMENT_SRC_DIR.parent
NEW_DIR = EXPERIMENT_DIR.parents[1]
SRC_DIR = NEW_DIR / "src"
DEFAULT_CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints" / "nids_multitask_05_v3_full"
DEFAULT_STATS_PATH = NEW_DIR / "nids_normalization_stats.json"
DEFAULT_DATA_ROOT = NEW_DIR / "data" / "nids_src_grouped"
CAT_VOCABS = [256, 256, 256, 65536, 65536, 256, 256, 256, 256]


def load_module_from_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base_train = load_module_from_path("base_train_multitask_nids", SRC_DIR / "05_train_multitask_nids.py")
st_data_loader = load_module_from_path("st_data_loader", SRC_DIR / "02_st_data_loader.py")
stt_architecture = load_module_from_path("stt_architecture_v3", EXPERIMENT_SRC_DIR / "03_stt_architecture_v3.py")
v3_train = load_module_from_path("train_multitask_nids_v3", EXPERIMENT_SRC_DIR / "05_train_multitask_nids_v3.py")

SpatioTemporalNIDSDataset = st_data_loader.SpatioTemporalNIDSDataset
SpatioTemporalTransformer = stt_architecture.SpatioTemporalTransformer
NIDSMultiTaskModel = stt_architecture.NIDSMultiTaskModel
DownstreamNIDSDataset = v3_train.VariantDownstreamNIDSDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate thesis-ready v3 evaluation artifacts and figures."
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="Checkpoint directory containing v3 epoch checkpoints and the best checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional explicit checkpoint path. Defaults to checkpoint-dir/nids_multitask_best.pt.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory for artifacts. Defaults to <checkpoint-dir>/thesis_artifacts.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["validation", "test", "test_ood"],
        help="Dataset splits to evaluate and plot. The test_ood split name is kept for compatibility and refers to the held-out-family benchmark split.",
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for evaluation.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count for evaluation.")
    parser.add_argument(
        "--current-threshold",
        type=float,
        default=None,
        help="Optional override for the current-attack threshold stored in the checkpoint.",
    )
    parser.add_argument(
        "--known-threshold",
        type=float,
        default=None,
        help="Optional override for the known-family gate threshold stored in the checkpoint.",
    )
    parser.add_argument(
        "--future-threshold",
        type=float,
        default=None,
        help="Optional scalar override applied to every future-warning horizon threshold stored in the checkpoint.",
    )
    parser.add_argument(
        "--ood-threshold",
        type=float,
        default=None,
        help="Optional override for the novelty or unknown-risk threshold stored in the checkpoint.",
    )
    parser.add_argument(
        "--correlation-sample-rows",
        type=int,
        default=50000,
        help="Number of raw train rows to sample for the continuous-feature correlation matrix.",
    )
    parser.add_argument(
        "--feature-shift-sample-rows",
        type=int,
        default=20000,
        help="Number of raw rows per split to sample for feature-shift plots.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Figure DPI.",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Number of equal-width bins used for ECE and reliability diagrams.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=0,
        help="Optional number of bootstrap resamples for 95%% confidence intervals on PR-AUC, recall, and F1.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="Random seed used for bootstrap confidence intervals.",
    )
    parser.add_argument(
        "--progress-log-interval",
        type=int,
        default=25,
        help="Write an explicit progress log every N evaluation batches. Set to 0 to disable.",
    )
    return parser.parse_args()


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def to_serializable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(val) for val in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
    return value


def dump_json(path, payload):
    with open(path, "w") as handle:
        json.dump(to_serializable(payload), handle, indent=2)


def resolve_checkpoint_path(checkpoint_dir, checkpoint_path):
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
    else:
        path = Path(checkpoint_dir) / "nids_multitask_best.pt"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def checkpoint_uses_ood_head(checkpoint):
    if "use_ood_head" in checkpoint:
        return bool(checkpoint["use_ood_head"])
    model_state_dict = checkpoint.get("model_state_dict", {})
    return any(key.startswith("unknown_attack_head.") for key in model_state_dict)


def resolve_unknown_risk_score_mode(use_ood_head, use_reconstruction_hybrid_ood):
    if use_reconstruction_hybrid_ood and use_ood_head:
        return "hybrid_max_raw_unknown_head_and_reconstruction_percentile"
    if use_reconstruction_hybrid_ood:
        return "reconstruction_percentile_only"
    if use_ood_head:
        return "raw_unknown_head_only"
    return "disabled"


def resolve_unknown_risk_probabilities(raw_unknown_probabilities, reconstruction_probabilities, score_mode):
    return stt_architecture.resolve_unknown_risk_probabilities(
        raw_unknown_probabilities,
        reconstruction_probabilities,
        score_mode,
    )


def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    thresholds = checkpoint.get("thresholds", {"current": 0.5, "known": 0.55, "future": 0.5, "ood": 0.5})
    known_attack_labels = checkpoint.get("known_attack_labels", [])
    pseudo_zero_day_families = checkpoint.get("pseudo_zero_day_families", [])
    future_horizons_minutes = v3_train.normalize_future_horizons_minutes(
        checkpoint.get("future_horizons_minutes"),
        checkpoint.get("future_horizon_minutes", 5),
    )
    thresholds = dict(thresholds)
    thresholds["future"] = v3_train.normalize_future_thresholds(
        thresholds.get("future"),
        future_horizons_minutes,
    )
    future_task_enabled = bool(checkpoint.get("future_task_enabled", True))
    use_ood_head = checkpoint_uses_ood_head(checkpoint)
    seq_len = int(checkpoint.get("seq_len", 32))
    stride = int(checkpoint.get("stride", 16))
    reconstruction_calibration = checkpoint.get("reconstruction_calibration")
    use_reconstruction_hybrid_ood = bool(
        checkpoint.get("use_reconstruction_hybrid_ood", False) and reconstruction_calibration is not None
    )
    unknown_risk_score_mode = str(
        checkpoint.get(
            "unknown_risk_score_mode",
            resolve_unknown_risk_score_mode(use_ood_head, use_reconstruction_hybrid_ood),
        )
    )
    return {
        "checkpoint": checkpoint,
        "thresholds": dict(thresholds),
        "validation_thresholds": dict(thresholds),
        "known_attack_labels": known_attack_labels,
        "pseudo_zero_day_families": pseudo_zero_day_families,
        "future_horizons_minutes": future_horizons_minutes,
        "future_horizon_minutes": int(future_horizons_minutes[-1]),
        "future_pre_onset_exclusion_gap_minutes": float(
            checkpoint.get(
                "future_pre_onset_exclusion_gap_minutes",
                v3_train.DEFAULT_FUTURE_PRE_ONSET_EXCLUSION_GAP_MINUTES,
            )
        ),
        "future_horizon_labels": v3_train.build_future_horizon_labels(future_horizons_minutes),
        "future_task_enabled": future_task_enabled,
        "run_mode": str(checkpoint.get("run_mode", v3_train.RUN_MODE_CLOSED_SET)),
        "thesis_claim": str(checkpoint.get("thesis_claim", v3_train.DEFAULT_CLOSED_SET_THESIS_CLAIM)),
        "novelty_score_mode": str(checkpoint.get("novelty_score_mode", v3_train.DEFAULT_NOVELTY_SCORE_MODE)),
        "decision_policy": str(checkpoint.get("decision_policy", v3_train.DEFAULT_DECISION_POLICY)),
        "task_activation": dict(checkpoint.get("task_activation", {})),
        "unknown_head_active": bool(checkpoint.get("unknown_head_active", use_ood_head)),
        "use_ood_head": use_ood_head,
        "reconstruction_calibration": reconstruction_calibration,
        "use_reconstruction_hybrid_ood": use_reconstruction_hybrid_ood,
        "unknown_risk_score_mode": unknown_risk_score_mode,
        "ood_threshold_selection_policy": str(
            checkpoint.get("ood_threshold_selection_policy", "target_recall")
        ),
        "ood_max_fpr": float(checkpoint.get("ood_max_fpr", 0.01)),
        "reconstruction_train_mae_mask_ratio": float(
            checkpoint.get("reconstruction_train_mae_mask_ratio", 0.10)
        ),
        "reconstruction_train_mfm_mask_ratio": float(
            checkpoint.get("reconstruction_train_mfm_mask_ratio", 0.00)
        ),
        "reconstruction_validation_mae_mask_ratio": float(
            checkpoint.get("reconstruction_validation_mae_mask_ratio", 0.30)
        ),
        "reconstruction_validation_mfm_mask_ratio": float(
            checkpoint.get("reconstruction_validation_mfm_mask_ratio", 0.10)
        ),
        "seq_len": seq_len,
        "stride": stride,
        "threshold_target_recall": float(
            checkpoint.get("threshold_target_recall", base_train.DEFAULT_THRESHOLD_TARGET_RECALL)
        ),
        "future_threshold_target_recall": float(
            checkpoint.get(
                "future_threshold_target_recall",
                v3_train.DEFAULT_FUTURE_THRESHOLD_TARGET_RECALL,
            )
        ),
        "ood_threshold_target_recall": float(
            checkpoint.get(
                "ood_threshold_target_recall",
                v3_train.DEFAULT_OOD_THRESHOLD_TARGET_RECALL,
            )
        ),
        "known_target_unknown_recall": float(
            checkpoint.get(
                "known_target_unknown_recall",
                v3_train.DEFAULT_KNOWN_TARGET_UNKNOWN_RECALL,
            )
        ),
    }


def apply_threshold_overrides(
    thresholds,
    future_horizons_minutes,
    current_threshold=None,
    known_threshold=None,
    future_threshold=None,
    ood_threshold=None,
):
    applied_thresholds = dict(thresholds)
    if current_threshold is not None:
        applied_thresholds["current"] = float(current_threshold)
    if known_threshold is not None:
        applied_thresholds["known"] = float(known_threshold)
    applied_thresholds["future"] = v3_train.normalize_future_thresholds(
        applied_thresholds.get("future"),
        future_horizons_minutes,
    )
    if future_threshold is not None:
        applied_thresholds["future"] = {
            label: float(future_threshold)
            for label in v3_train.build_future_horizon_labels(future_horizons_minutes)
        }
    if ood_threshold is not None:
        applied_thresholds["ood"] = float(ood_threshold)
    if applied_thresholds.get("ood") is None:
        applied_thresholds["ood"] = 0.50
    return applied_thresholds


def load_model(checkpoint_bundle, dataset, device):
    checkpoint = checkpoint_bundle["checkpoint"]
    seq_len = checkpoint_bundle["seq_len"]
    future_task_enabled = checkpoint_bundle["future_task_enabled"]
    num_cont = len(dataset.base_dataset.cont_cols)
    model = NIDSMultiTaskModel(
        backbone=SpatioTemporalTransformer(
            num_cont_features=num_cont,
            cat_vocab_sizes=CAT_VOCABS,
            seq_len=seq_len,
            init_mae=checkpoint_bundle.get("reconstruction_train_mae_mask_ratio", 0.10),
            init_mfm=checkpoint_bundle.get("reconstruction_train_mfm_mask_ratio", 0.00),
        ),
        num_known_attack_classes=len(checkpoint_bundle["known_attack_labels"]),
        use_future_head=future_task_enabled,
        use_ood_head=checkpoint_bundle.get("use_ood_head", True),
        future_horizons_minutes=checkpoint_bundle["future_horizons_minutes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def resolve_split_path(split_name):
    split_path = DEFAULT_DATA_ROOT / split_name
    if split_path.exists():
        return split_path
    return None


def build_downstream_dataset(split_name, checkpoint_bundle):
    split_path = resolve_split_path(split_name)
    if split_path is None:
        return None
    base_dataset = SpatioTemporalNIDSDataset(
        arrow_dir_path=str(split_path),
        stats_path=str(DEFAULT_STATS_PATH),
        seq_len=checkpoint_bundle["seq_len"],
        stride=checkpoint_bundle["stride"],
        clip_value=5.0,
    )
    known_attack_to_idx = {
        attack_name: idx for idx, attack_name in enumerate(checkpoint_bundle["known_attack_labels"])
    }
    downstream_dataset = DownstreamNIDSDataset(
        base_dataset=base_dataset,
        future_horizons_minutes=checkpoint_bundle["future_horizons_minutes"],
        future_pre_onset_exclusion_gap_minutes=checkpoint_bundle[
            "future_pre_onset_exclusion_gap_minutes"
        ],
        known_attack_to_idx=known_attack_to_idx,
        max_sequences=None,
    )
    return downstream_dataset


def build_dataset_profile(split_name, checkpoint_bundle):
    dataset = build_downstream_dataset(split_name, checkpoint_bundle)
    if dataset is None:
        return None

    family_counts = Counter(
        attack_family
        for attack_family, label_value in zip(dataset.sequence_attack_families, dataset.sequence_current_labels)
        if int(label_value) == 1 and attack_family != "Benign"
    )
    unknown_positive_count = int(dataset.unknown_attack_targets.sum())
    known_family_count = int((dataset.known_attack_targets >= 0).sum())
    future_target_matrix = np.asarray(dataset.future_attack_targets, dtype=np.float32)
    future_supervision_mask = np.asarray(dataset.future_supervision_mask, dtype=bool)
    valid_future_target_matrix = future_target_matrix.astype(bool) & future_supervision_mask
    raw_future_target_matrix = future_target_matrix.astype(bool)
    future_positive_count = int(valid_future_target_matrix.any(axis=1).sum())
    future_raw_positive_count = int(raw_future_target_matrix.any(axis=1).sum())
    future_positive_counts_by_horizon = {
        horizon_label: int(valid_future_target_matrix[:, horizon_idx].sum())
        for horizon_idx, horizon_label in enumerate(dataset.future_horizon_labels)
    }
    future_raw_positive_counts_by_horizon = {
        horizon_label: int(raw_future_target_matrix[:, horizon_idx].sum())
        for horizon_idx, horizon_label in enumerate(dataset.future_horizon_labels)
    }
    future_ignored_near_onset_positive_counts_by_horizon = {
        horizon_label: max(
            future_raw_positive_counts_by_horizon[horizon_label]
            - future_positive_counts_by_horizon[horizon_label],
            0,
        )
        for horizon_label in dataset.future_horizon_labels
    }

    return {
        "split": split_name,
        "sequence_count": int(len(dataset)),
        "current_positive_count": int(dataset.sequence_current_labels.sum()),
        "current_positive_rate": float(dataset.sequence_current_labels.mean()) if len(dataset) else 0.0,
        "unknown_positive_count": unknown_positive_count,
        "known_family_count": known_family_count,
        "future_positive_count": future_positive_count,
        "future_raw_positive_count": future_raw_positive_count,
        "future_ignored_near_onset_positive_count": max(
            future_raw_positive_count - future_positive_count,
            0,
        ),
        "future_positive_counts_by_horizon": future_positive_counts_by_horizon,
        "future_raw_positive_counts_by_horizon": future_raw_positive_counts_by_horizon,
        "future_ignored_near_onset_positive_counts_by_horizon": future_ignored_near_onset_positive_counts_by_horizon,
        "attack_family_counts": dict(family_counts),
    }


def binary_pr_curve(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive_count = int((labels == 1).sum())
    if labels.size == 0 or positive_count == 0:
        return np.array([0.0, 1.0]), np.array([1.0, 1.0])

    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels == 1)
    false_positives = np.cumsum(sorted_labels == 0)
    precision = true_positives / np.maximum(true_positives + false_positives, 1)
    recall = true_positives / positive_count
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return recall, precision


def binary_roc_curve(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive_count = int((labels == 1).sum())
    negative_count = int((labels == 0).sum())
    if labels.size == 0 or positive_count == 0 or negative_count == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels == 1)
    false_positives = np.cumsum(sorted_labels == 0)
    tpr = np.concatenate([[0.0], true_positives / positive_count, [1.0]])
    fpr = np.concatenate([[0.0], false_positives / negative_count, [1.0]])
    return fpr, tpr


def threshold_sweep(labels, scores, threshold_candidates=None):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0:
        return pd.DataFrame(columns=["threshold", "precision", "recall", "f1", "false_positive_rate"])

    if threshold_candidates is None:
        # The sweep is only used for diagnostic figures, so avoid using every
        # raw score as a threshold candidate on very large splits.
        threshold_candidates = np.unique(
            np.concatenate([
                np.linspace(0.0, 1.0, 101),
                np.percentile(scores, [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]),
                np.quantile(scores, np.linspace(0.0, 1.0, 301)),
            ])
        )

    threshold_candidates = np.sort(np.unique(np.asarray(threshold_candidates, dtype=np.float64)))
    if threshold_candidates.size == 0:
        return pd.DataFrame(columns=["threshold", "precision", "recall", "f1", "false_positive_rate"])

    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    positive_prefix = np.concatenate([[0], np.cumsum(sorted_labels == 1, dtype=np.int64)])
    negative_prefix = np.concatenate([[0], np.cumsum(sorted_labels == 0, dtype=np.int64)])
    positive_total = int(positive_prefix[-1])
    negative_total = int(negative_prefix[-1])

    threshold_indices = np.searchsorted(sorted_scores, threshold_candidates, side="left")
    true_positive = positive_total - positive_prefix[threshold_indices]
    false_positive = negative_total - negative_prefix[threshold_indices]
    predicted_positive = true_positive + false_positive

    precision = true_positive / np.maximum(predicted_positive, 1)
    recall = true_positive / max(positive_total, 1)
    f1 = np.where(
        precision + recall > 0.0,
        2.0 * precision * recall / np.maximum(precision + recall, 1e-12),
        0.0,
    )
    false_positive_rate = false_positive / max(negative_total, 1)

    return pd.DataFrame(
        {
            "threshold": threshold_candidates.astype(np.float64),
            "precision": precision.astype(np.float64),
            "recall": recall.astype(np.float64),
            "f1": f1.astype(np.float64),
            "false_positive_rate": false_positive_rate.astype(np.float64),
        }
    )


def prepare_binary_arrays(labels, scores):
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size != scores.size:
        raise ValueError(f"Mismatched labels and scores: {labels.size} vs {scores.size}")
    valid_mask = np.isfinite(labels) & np.isfinite(scores)
    if not valid_mask.all():
        labels = labels[valid_mask]
        scores = scores[valid_mask]
    return labels.astype(np.int64), scores.astype(np.float64)


def compute_brier_score(labels, scores):
    labels, scores = prepare_binary_arrays(labels, scores)
    if labels.size == 0:
        return float("nan")
    return float(np.mean((scores - labels) ** 2))


def compute_expected_calibration_error(labels, scores, bins=10):
    labels, scores = prepare_binary_arrays(labels, scores)
    if labels.size == 0:
        return float("nan")

    bin_edges = np.linspace(0.0, 1.0, int(bins) + 1)
    bin_indices = np.digitize(scores, bin_edges[1:-1], right=False)
    total_count = labels.size
    ece = 0.0
    for bin_index in range(int(bins)):
        mask = bin_indices == bin_index
        if not mask.any():
            continue
        confidence = float(scores[mask].mean())
        accuracy = float(labels[mask].mean())
        ece += (mask.sum() / total_count) * abs(confidence - accuracy)
    return float(ece)


def compute_binary_calibration_metrics(labels, scores, bins=10):
    return {
        "brier_score": compute_brier_score(labels, scores),
        "ece": compute_expected_calibration_error(labels, scores, bins=bins),
        "calibration_bins": int(bins),
    }


def summarize_bootstrap_metric(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "lower": float("nan"),
            "upper": float("nan"),
            "valid_samples": 0,
        }
    return {
        "lower": float(np.percentile(values, 2.5)),
        "upper": float(np.percentile(values, 97.5)),
        "valid_samples": int(values.size),
    }


def compute_binary_bootstrap_confidence_intervals(labels, scores, threshold, samples, seed):
    labels, scores = prepare_binary_arrays(labels, scores)
    if samples <= 0 or labels.size == 0:
        return None

    rng = np.random.default_rng(int(seed))
    sample_count = labels.size
    collected = {metric_name: [] for metric_name in ["pr_auc", "recall", "f1"]}

    for _ in range(int(samples)):
        sample_indices = rng.integers(0, sample_count, size=sample_count)
        sample_metrics = base_train.compute_binary_metrics(
            labels[sample_indices],
            scores[sample_indices],
            float(threshold),
        )
        for metric_name in collected:
            metric_value = float(sample_metrics.get(metric_name, float("nan")))
            if np.isfinite(metric_value):
                collected[metric_name].append(metric_value)

    return {
        "samples": int(samples),
        "seed": int(seed),
        **{
            metric_name: summarize_bootstrap_metric(metric_values)
            for metric_name, metric_values in collected.items()
        },
    }


def compute_future_bootstrap_confidence_intervals(
    future_labels,
    future_probabilities,
    future_thresholds,
    future_horizon_labels,
    samples,
    seed,
    future_supervision_masks=None,
):
    future_labels = np.asarray(future_labels, dtype=np.float64)
    future_probabilities = np.asarray(future_probabilities, dtype=np.float64)
    if samples <= 0 or future_labels.size == 0 or future_probabilities.size == 0:
        return None, {}

    if future_labels.ndim == 1:
        future_labels = future_labels.reshape(-1, 1)
    if future_probabilities.ndim == 1:
        future_probabilities = future_probabilities.reshape(-1, 1)
    if future_supervision_masks is not None:
        future_supervision_masks = np.asarray(future_supervision_masks, dtype=bool)
        if future_supervision_masks.ndim == 1:
            future_supervision_masks = future_supervision_masks.reshape(-1, 1)
    else:
        future_supervision_masks = np.ones_like(future_labels, dtype=bool)

    if future_labels.shape[0] == 0:
        return None, {}

    rng = np.random.default_rng(int(seed))
    macro_collected = {metric_name: [] for metric_name in ["pr_auc", "recall", "f1"]}
    per_horizon_collected = {
        horizon_label: {metric_name: [] for metric_name in ["pr_auc", "recall", "f1"]}
        for horizon_label in future_horizon_labels
    }

    for _ in range(int(samples)):
        sampled_metrics_by_horizon = {}
        for horizon_idx, horizon_label in enumerate(future_horizon_labels):
            horizon_mask = future_supervision_masks[:, horizon_idx].astype(bool)
            horizon_labels = future_labels[horizon_mask, horizon_idx].astype(np.int64)
            horizon_probabilities = future_probabilities[horizon_mask, horizon_idx]
            row_count = horizon_labels.shape[0]
            if row_count == 0:
                sampled_metrics = base_train.compute_binary_metrics(
                    np.array([], dtype=np.int64),
                    np.array([], dtype=np.float64),
                    float(future_thresholds[horizon_label]),
                )
                sampled_metrics_by_horizon[horizon_label] = sampled_metrics
                continue
            sample_indices = rng.integers(0, row_count, size=row_count)
            sampled_metrics = base_train.compute_binary_metrics(
                horizon_labels[sample_indices],
                horizon_probabilities[sample_indices],
                float(future_thresholds[horizon_label]),
            )
            sampled_metrics_by_horizon[horizon_label] = sampled_metrics
            for metric_name in per_horizon_collected[horizon_label]:
                metric_value = float(sampled_metrics.get(metric_name, float("nan")))
                if np.isfinite(metric_value):
                    per_horizon_collected[horizon_label][metric_name].append(metric_value)

        macro_metrics = v3_train.aggregate_future_metrics(sampled_metrics_by_horizon)
        for metric_name in macro_collected:
            metric_value = float(macro_metrics.get(metric_name, float("nan")))
            if np.isfinite(metric_value):
                macro_collected[metric_name].append(metric_value)

    macro_summary = {
        "samples": int(samples),
        "seed": int(seed),
        **{
            metric_name: summarize_bootstrap_metric(metric_values)
            for metric_name, metric_values in macro_collected.items()
        },
    }
    per_horizon_summary = {
        horizon_label: {
            "samples": int(samples),
            "seed": int(seed),
            **{
                metric_name: summarize_bootstrap_metric(metric_values)
                for metric_name, metric_values in metric_map.items()
            },
        }
        for horizon_label, metric_map in per_horizon_collected.items()
    }
    return macro_summary, per_horizon_summary


def compute_confusion_counts(labels, scores, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = scores >= float(threshold)
    true_negative = int(np.logical_and(predictions == 0, labels == 0).sum())
    false_positive = int(np.logical_and(predictions == 1, labels == 0).sum())
    false_negative = int(np.logical_and(predictions == 0, labels == 1).sum())
    true_positive = int(np.logical_and(predictions == 1, labels == 1).sum())
    return np.array([[true_negative, false_positive], [false_negative, true_positive]], dtype=np.int64)


def compute_known_gate_metrics_at_threshold(
    known_current_probabilities,
    known_confidences,
    known_predictions,
    known_targets,
    unknown_current_probabilities,
    unknown_confidences,
    current_threshold,
    known_threshold,
):
    known_current_probabilities = np.asarray(known_current_probabilities, dtype=np.float64)
    known_confidences = np.asarray(known_confidences, dtype=np.float64)
    known_predictions = np.asarray(known_predictions, dtype=np.int64)
    known_targets = np.asarray(known_targets, dtype=np.int64)
    unknown_current_probabilities = np.asarray(unknown_current_probabilities, dtype=np.float64)
    unknown_confidences = np.asarray(unknown_confidences, dtype=np.float64)

    known_gate = (
        (known_current_probabilities >= current_threshold)
        & (known_confidences >= known_threshold)
    )
    accepted_accuracy = float(
        (known_predictions[known_gate] == known_targets[known_gate]).mean()
    ) if known_gate.any() else float("nan")
    known_coverage = float(known_gate.mean()) if known_confidences.size else float("nan")

    if unknown_confidences.size:
        unknown_gate = (
            (unknown_current_probabilities >= current_threshold)
            & (unknown_confidences < known_threshold)
        )
        unknown_recall = float(unknown_gate.mean())
    else:
        unknown_recall = float("nan")

    balanced_score = (
        2.0 * accepted_accuracy * known_coverage / max(accepted_accuracy + known_coverage, 1e-9)
        if not np.isnan(accepted_accuracy) and not np.isnan(known_coverage)
        else float("nan")
    )

    return {
        "threshold": float(known_threshold),
        "accepted_accuracy": accepted_accuracy,
        "known_coverage": known_coverage,
        "balanced_score": balanced_score,
        "unknown_recall": unknown_recall,
    }


def collect_split_outputs(
    split_name,
    checkpoint_bundle,
    device,
    batch_size,
    num_workers,
    profile=None,
    calibration_bins=10,
    bootstrap_samples=0,
    bootstrap_seed=42,
    progress_log_interval=25,
):
    print(f"[{split_name}] Building evaluation dataset...", flush=True)
    dataset = build_downstream_dataset(split_name, checkpoint_bundle)
    if dataset is None:
        return None

    future_horizons_minutes = checkpoint_bundle["future_horizons_minutes"]
    future_horizon_labels = checkpoint_bundle["future_horizon_labels"]

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model = load_model(checkpoint_bundle, dataset, device)
    print(
        f"[{split_name}] Evaluating {len(dataset):,} sequences across {len(data_loader):,} batches on {device.type}...",
        flush=True,
    )

    current_probs = []
    current_labels = []
    future_probs = []
    future_labels = []
    future_leads = []
    future_supervision_masks = []
    raw_ood_probs = []
    ood_labels = []
    reconstruction_scores = []
    mae_reconstruction_scores = []
    mfm_reconstruction_scores = []
    known_confidences = []
    known_current_probs = []
    known_predictions = []
    known_targets = []
    known_family_probabilities = []
    unknown_confidences = []
    unknown_current_probs = []
    use_ood_head = bool(checkpoint_bundle.get("use_ood_head", True))
    use_reconstruction_hybrid_ood = bool(checkpoint_bundle.get("use_reconstruction_hybrid_ood", False))
    reconstruction_calibration = checkpoint_bundle.get("reconstruction_calibration")
    reconstruction_validation_mae_mask_ratio = float(
        checkpoint_bundle.get("reconstruction_validation_mae_mask_ratio", 0.30)
    )
    reconstruction_validation_mfm_mask_ratio = float(
        checkpoint_bundle.get("reconstruction_validation_mfm_mask_ratio", 0.10)
    )
    novelty_score_mode = str(
        checkpoint_bundle.get("novelty_score_mode", v3_train.DEFAULT_NOVELTY_SCORE_MODE)
    )
    ood_metric_available = bool(use_ood_head or use_reconstruction_hybrid_ood)

    use_tqdm = os.isatty(1)
    with torch.no_grad():
        for batch_index, batch in enumerate(
            tqdm(
                data_loader,
                total=len(data_loader),
                desc=f"Eval {split_name}",
                leave=False,
                disable=not use_tqdm,
            )
        ):
            completed_batches = batch_index + 1
            if progress_log_interval > 0 and (
                completed_batches % progress_log_interval == 0 or completed_batches == len(data_loader)
            ):
                progress_pct = 100.0 * completed_batches / max(len(data_loader), 1)
                print(
                    f"[{split_name}] Evaluation progress: {completed_batches:,}/{len(data_loader):,} batches ({progress_pct:.1f}%)",
                    flush=True,
                )
            cont = batch["continuous"].to(device, non_blocking=True)
            cat = batch["categorical"].to(device, non_blocking=True)
            reconstruction_mask = None
            reconstruction_mfm_mask = None
            if use_reconstruction_hybrid_ood:
                reconstruction_mask = stt_architecture.build_fixed_spatial_mask(
                    cont.shape[0],
                    cont.shape[1],
                    reconstruction_validation_mae_mask_ratio,
                    device,
                    v3_train.DEFAULT_RECONSTRUCTION_VALIDATION_MASK_SEED + batch_index,
                )
                if novelty_score_mode == v3_train.DEFAULT_NOVELTY_SCORE_MODE:
                    reconstruction_mfm_mask = stt_architecture.build_fixed_spatial_mask(
                        cont.shape[0],
                        cont.shape[1],
                        reconstruction_validation_mfm_mask_ratio,
                        device,
                        v3_train.DEFAULT_RECONSTRUCTION_VALIDATION_MASK_SEED + 100_000 + batch_index,
                    )
            if use_reconstruction_hybrid_ood and novelty_score_mode == v3_train.DEFAULT_NOVELTY_SCORE_MODE:
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
                    compute_reconstruction=use_reconstruction_hybrid_ood,
                    reconstruction_mask=reconstruction_mask,
                    reconstruction_apply_mfm=False,
                )

            current_probability = torch.sigmoid(outputs["current_attack_logits"]).cpu().numpy()
            current_target = batch["label"].cpu().numpy()
            current_probs.append(current_probability)
            current_labels.append(current_target)

            ood_labels.append(batch["unknown_attack_target"].cpu().numpy())
            if use_ood_head and outputs.get("unknown_attack_logits") is not None:
                ood_probability = torch.sigmoid(outputs["unknown_attack_logits"]).cpu().numpy()
            else:
                ood_probability = np.zeros_like(current_probability)
            raw_ood_probs.append(ood_probability)

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

            if checkpoint_bundle["future_task_enabled"] and outputs.get("future_attack_logits") is not None:
                future_logits = outputs["future_attack_logits"]
                if future_logits.ndim == 1:
                    future_logits = future_logits.unsqueeze(-1)
                benign_mask = batch["label"] == 0
                if benign_mask.any():
                    future_probability = torch.sigmoid(future_logits[benign_mask]).cpu().numpy()
                    if future_probability.ndim == 1:
                        future_probability = future_probability.reshape(-1, 1)
                    future_target = batch["future_attack"][benign_mask].cpu().numpy()
                    future_lead = batch["future_lead_minutes"][benign_mask].cpu().numpy()
                    future_supervision_mask = batch["future_supervision_mask"][benign_mask].cpu().numpy()
                    if future_target.ndim == 1:
                        future_target = future_target.reshape(-1, 1)
                    if future_lead.ndim == 1:
                        future_lead = future_lead.reshape(-1, 1)
                    if future_supervision_mask.ndim == 1:
                        future_supervision_mask = future_supervision_mask.reshape(-1, 1)
                    future_probs.append(future_probability)
                    future_labels.append(future_target)
                    future_leads.append(future_lead)
                    future_supervision_masks.append(future_supervision_mask)

            if outputs.get("attack_family_logits") is not None:
                family_probability = torch.softmax(outputs["attack_family_logits"], dim=-1).cpu().numpy()
                family_confidence = family_probability.max(axis=-1)
                family_prediction = family_probability.argmax(axis=-1)
                known_attack_target = batch["known_attack_id"].cpu().numpy()
                unknown_target = batch["unknown_attack_target"].cpu().numpy()

                known_mask = known_attack_target >= 0
                if known_mask.any():
                    known_confidences.append(family_confidence[known_mask])
                    known_current_probs.append(current_probability[known_mask])
                    known_predictions.append(family_prediction[known_mask])
                    known_targets.append(known_attack_target[known_mask])
                    known_family_probabilities.append(family_probability[known_mask])

                unknown_mask = unknown_target == 1
                if unknown_mask.any():
                    unknown_confidences.append(family_confidence[unknown_mask])
                    unknown_current_probs.append(current_probability[unknown_mask])

    arrays = {
        "current_probabilities": np.concatenate(current_probs) if current_probs else np.array([]),
        "current_labels": np.concatenate(current_labels) if current_labels else np.array([]),
        "future_probabilities": (
            np.concatenate(future_probs, axis=0)
            if future_probs else np.zeros((0, len(future_horizons_minutes)), dtype=np.float32)
        ),
        "future_labels": (
            np.concatenate(future_labels, axis=0)
            if future_labels else np.zeros((0, len(future_horizons_minutes)), dtype=np.float32)
        ),
        "future_leads": (
            np.concatenate(future_leads, axis=0)
            if future_leads else np.zeros((0, len(future_horizons_minutes)), dtype=np.float32)
        ),
        "future_supervision_masks": (
            np.concatenate(future_supervision_masks, axis=0)
            if future_supervision_masks else np.zeros((0, len(future_horizons_minutes)), dtype=np.float32)
        ),
        "raw_ood_head_probabilities": (
            np.concatenate(raw_ood_probs) if raw_ood_probs else np.array([])
        ),
        "ood_labels": np.concatenate(ood_labels) if ood_labels else np.array([]),
        "reconstruction_scores": (
            np.concatenate(reconstruction_scores) if reconstruction_scores else np.array([])
        ),
        "mae_reconstruction_scores": (
            np.concatenate(mae_reconstruction_scores) if mae_reconstruction_scores else np.array([])
        ),
        "mfm_reconstruction_scores": (
            np.concatenate(mfm_reconstruction_scores) if mfm_reconstruction_scores else np.array([])
        ),
        "known_confidences": np.concatenate(known_confidences) if known_confidences else np.array([]),
        "known_current_probabilities": np.concatenate(known_current_probs) if known_current_probs else np.array([]),
        "known_predictions": np.concatenate(known_predictions) if known_predictions else np.array([]),
        "known_targets": np.concatenate(known_targets) if known_targets else np.array([]),
        "known_family_probabilities": (
            np.concatenate(known_family_probabilities, axis=0)
            if known_family_probabilities
            else np.zeros((0, len(checkpoint_bundle["known_attack_labels"])), dtype=np.float32)
        ),
        "unknown_confidences": np.concatenate(unknown_confidences) if unknown_confidences else np.array([]),
        "unknown_current_probabilities": np.concatenate(unknown_current_probs) if unknown_current_probs else np.array([]),
    }
    arrays["reconstruction_probabilities"] = stt_architecture.reconstruction_scores_to_percentiles(
        arrays["reconstruction_scores"],
        reconstruction_calibration,
    )
    arrays["ood_probabilities"] = resolve_unknown_risk_probabilities(
        arrays["raw_ood_head_probabilities"],
        arrays["reconstruction_probabilities"],
        checkpoint_bundle.get(
            "unknown_risk_score_mode",
            resolve_unknown_risk_score_mode(use_ood_head, use_reconstruction_hybrid_ood),
        ),
    )

    if profile is None:
        profile = build_dataset_profile(split_name, checkpoint_bundle)
    thresholds = checkpoint_bundle["thresholds"]
    current_at_validation = base_train.compute_binary_metrics(
        arrays["current_labels"],
        arrays["current_probabilities"],
        thresholds["current"],
    )
    current_at_validation.update(
        compute_binary_calibration_metrics(
            arrays["current_labels"],
            arrays["current_probabilities"],
            bins=calibration_bins,
        )
    )
    current_oracle = base_train.select_threshold_for_target_recall(
        arrays["current_labels"],
        arrays["current_probabilities"],
        checkpoint_bundle["threshold_target_recall"],
    )

    if ood_metric_available and arrays["ood_labels"].size:
        ood_at_validation = base_train.compute_binary_metrics(
            arrays["ood_labels"],
            arrays["ood_probabilities"],
            thresholds["ood"],
        )
        ood_at_validation.update(
            compute_binary_calibration_metrics(
                arrays["ood_labels"],
                arrays["ood_probabilities"],
                bins=calibration_bins,
            )
        )
        ood_oracle = base_train.select_threshold_for_target_recall(
            arrays["ood_labels"],
            arrays["ood_probabilities"],
            checkpoint_bundle["ood_threshold_target_recall"],
        )
    else:
        ood_at_validation = None
        ood_oracle = None

    if use_ood_head and arrays["ood_labels"].size:
        ood_head_at_validation = base_train.compute_binary_metrics(
            arrays["ood_labels"],
            arrays["raw_ood_head_probabilities"],
            thresholds["ood"],
        )
        ood_head_at_validation.update(
            compute_binary_calibration_metrics(
                arrays["ood_labels"],
                arrays["raw_ood_head_probabilities"],
                bins=calibration_bins,
            )
        )
        ood_head_oracle = base_train.select_threshold_for_target_recall(
            arrays["ood_labels"],
            arrays["raw_ood_head_probabilities"],
            checkpoint_bundle["ood_threshold_target_recall"],
        )
    else:
        ood_head_at_validation = None
        ood_head_oracle = None

    if reconstruction_calibration is not None and arrays["ood_labels"].size:
        reconstruction_ood_at_validation = base_train.compute_binary_metrics(
            arrays["ood_labels"],
            arrays["reconstruction_probabilities"],
            thresholds["ood"],
        )
        reconstruction_ood_at_validation.update(
            compute_binary_calibration_metrics(
                arrays["ood_labels"],
                arrays["reconstruction_probabilities"],
                bins=calibration_bins,
            )
        )
        reconstruction_ood_oracle = base_train.select_threshold_for_target_recall(
            arrays["ood_labels"],
            arrays["reconstruction_probabilities"],
            checkpoint_bundle["ood_threshold_target_recall"],
        )
    else:
        reconstruction_ood_at_validation = None
        reconstruction_ood_oracle = None

    if checkpoint_bundle["future_task_enabled"]:
        future_at_validation_by_horizon = {}
        future_oracle_by_horizon = {}
        mean_future_lead_minutes_by_horizon = {}
        for horizon_idx, horizon_label in enumerate(future_horizon_labels):
            horizon_supervision_mask = arrays["future_supervision_masks"][:, horizon_idx].astype(bool)
            horizon_labels = arrays["future_labels"][horizon_supervision_mask, horizon_idx]
            horizon_probs = arrays["future_probabilities"][horizon_supervision_mask, horizon_idx]
            horizon_leads = arrays["future_leads"][horizon_supervision_mask, horizon_idx]
            horizon_threshold = thresholds["future"][horizon_label]
            raw_positive_count = int(arrays["future_labels"][:, horizon_idx].sum())
            valid_positive_count = int(horizon_labels.sum())
            ignored_near_onset_positive_count = max(raw_positive_count - valid_positive_count, 0)

            future_at_validation_by_horizon[horizon_label] = base_train.compute_binary_metrics(
                horizon_labels,
                horizon_probs,
                horizon_threshold,
            )
            future_at_validation_by_horizon[horizon_label].update(
                compute_binary_calibration_metrics(
                    horizon_labels,
                    horizon_probs,
                    bins=calibration_bins,
                )
            )
            future_at_validation_by_horizon[horizon_label].update(
                {
                    "valid_count": int(horizon_labels.shape[0]),
                    "raw_positive_count": raw_positive_count,
                    "valid_positive_count": valid_positive_count,
                    "ignored_near_onset_positive_count": ignored_near_onset_positive_count,
                    "future_pre_onset_exclusion_gap_minutes": float(
                        checkpoint_bundle["future_pre_onset_exclusion_gap_minutes"]
                    ),
                }
            )
            future_selection = base_train.select_threshold_for_target_recall(
                horizon_labels,
                horizon_probs,
                checkpoint_bundle["future_threshold_target_recall"],
            )
            future_oracle_metrics = base_train.compute_binary_metrics(
                horizon_labels,
                horizon_probs,
                future_selection["threshold"],
            )
            future_oracle_metrics.update(future_selection)
            future_oracle_metrics.update(
                {
                    "valid_count": int(horizon_labels.shape[0]),
                    "raw_positive_count": raw_positive_count,
                    "valid_positive_count": valid_positive_count,
                    "ignored_near_onset_positive_count": ignored_near_onset_positive_count,
                    "future_pre_onset_exclusion_gap_minutes": float(
                        checkpoint_bundle["future_pre_onset_exclusion_gap_minutes"]
                    ),
                }
            )
            future_oracle_by_horizon[horizon_label] = future_oracle_metrics

            future_hits = (horizon_probs >= horizon_threshold) & (horizon_labels == 1)
            mean_future_lead_minutes_by_horizon[horizon_label] = (
                float(horizon_leads[future_hits].mean()) if future_hits.any() else float("nan")
            )

        future_at_validation = v3_train.aggregate_future_metrics(future_at_validation_by_horizon)
        future_oracle = v3_train.aggregate_future_metrics(future_oracle_by_horizon)
        future_brier_values = np.asarray(
            [metrics.get("brier_score", float("nan")) for metrics in future_at_validation_by_horizon.values()],
            dtype=np.float64,
        )
        future_ece_values = np.asarray(
            [metrics.get("ece", float("nan")) for metrics in future_at_validation_by_horizon.values()],
            dtype=np.float64,
        )
        future_at_validation["thresholds"] = dict(thresholds["future"])
        future_at_validation["future_pre_onset_exclusion_gap_minutes"] = float(
            checkpoint_bundle["future_pre_onset_exclusion_gap_minutes"]
        )
        future_at_validation["brier_score"] = (
            float(np.nanmean(future_brier_values))
            if future_brier_values.size and not np.isnan(future_brier_values).all()
            else float("nan")
        )
        future_at_validation["ece"] = (
            float(np.nanmean(future_ece_values))
            if future_ece_values.size and not np.isnan(future_ece_values).all()
            else float("nan")
        )
        future_at_validation["calibration_bins"] = int(calibration_bins)
        future_oracle["thresholds"] = {
            horizon_label: metrics["threshold"] for horizon_label, metrics in future_oracle_by_horizon.items()
        }
        future_oracle["future_pre_onset_exclusion_gap_minutes"] = float(
            checkpoint_bundle["future_pre_onset_exclusion_gap_minutes"]
        )
        mean_future_lead_minutes = (
            float(np.nanmean(np.asarray(list(mean_future_lead_minutes_by_horizon.values()), dtype=np.float64)))
            if mean_future_lead_minutes_by_horizon and not np.isnan(np.asarray(list(mean_future_lead_minutes_by_horizon.values()), dtype=np.float64)).all()
            else float("nan")
        )
    else:
        future_at_validation = None
        future_oracle = None
        future_at_validation_by_horizon = {}
        future_oracle_by_horizon = {}
        mean_future_lead_minutes = float("nan")
        mean_future_lead_minutes_by_horizon = {}

    split_seed = int(bootstrap_seed) + sum(ord(character) for character in split_name)
    current_bootstrap = compute_binary_bootstrap_confidence_intervals(
        arrays["current_labels"],
        arrays["current_probabilities"],
        thresholds["current"],
        bootstrap_samples,
        split_seed,
    )
    if ood_metric_available and arrays["ood_labels"].size:
        ood_bootstrap = compute_binary_bootstrap_confidence_intervals(
            arrays["ood_labels"],
            arrays["ood_probabilities"],
            thresholds["ood"],
            bootstrap_samples,
            split_seed + 1,
        )
    else:
        ood_bootstrap = None

    if use_ood_head and arrays["ood_labels"].size:
        ood_head_bootstrap = compute_binary_bootstrap_confidence_intervals(
            arrays["ood_labels"],
            arrays["raw_ood_head_probabilities"],
            thresholds["ood"],
            bootstrap_samples,
            split_seed + 11,
        )
    else:
        ood_head_bootstrap = None

    if reconstruction_calibration is not None and arrays["ood_labels"].size:
        reconstruction_ood_bootstrap = compute_binary_bootstrap_confidence_intervals(
            arrays["ood_labels"],
            arrays["reconstruction_probabilities"],
            thresholds["ood"],
            bootstrap_samples,
            split_seed + 12,
        )
    else:
        reconstruction_ood_bootstrap = None

    if checkpoint_bundle["future_task_enabled"] and future_at_validation is not None:
        future_bootstrap, future_bootstrap_by_horizon = compute_future_bootstrap_confidence_intervals(
            arrays["future_labels"],
            arrays["future_probabilities"],
            thresholds["future"],
            future_horizon_labels,
            bootstrap_samples,
            split_seed + 2,
            future_supervision_masks=arrays["future_supervision_masks"],
        )
    else:
        future_bootstrap = None
        future_bootstrap_by_horizon = {}

    known_gate_at_validation = compute_known_gate_metrics_at_threshold(
        known_current_probabilities=arrays["known_current_probabilities"],
        known_confidences=arrays["known_confidences"],
        known_predictions=arrays["known_predictions"],
        known_targets=arrays["known_targets"],
        unknown_current_probabilities=arrays["unknown_current_probabilities"],
        unknown_confidences=arrays["unknown_confidences"],
        current_threshold=thresholds["current"],
        known_threshold=thresholds["known"],
    )
    known_oracle = v3_train.select_known_threshold(
        known_current_probabilities=arrays["known_current_probabilities"],
        known_confidences=arrays["known_confidences"],
        known_predictions=arrays["known_predictions"],
        known_targets=arrays["known_targets"],
        unknown_current_probabilities=arrays["unknown_current_probabilities"],
        unknown_confidences=arrays["unknown_confidences"],
        current_threshold=thresholds["current"],
        target_unknown_recall=checkpoint_bundle["known_target_unknown_recall"],
        default_threshold=thresholds["known"],
    )
    raw_known_accuracy = (
        float((arrays["known_predictions"] == arrays["known_targets"]).mean())
        if arrays["known_targets"].size
        else float("nan")
    )
    raw_known_macro_f1 = (
        float(v3_train.compute_multiclass_macro_f1(arrays["known_targets"], arrays["known_predictions"]))
        if arrays["known_targets"].size
        else float("nan")
    )
    raw_known_metrics_by_label = v3_train.compute_multiclass_metrics_by_label(
        arrays["known_targets"],
        arrays["known_predictions"],
        checkpoint_bundle["known_attack_labels"],
    )
    raw_known_metrics_by_label = v3_train.add_ovr_curve_metrics_by_label(
        raw_known_metrics_by_label,
        arrays["known_targets"],
        arrays["known_family_probabilities"],
        checkpoint_bundle["known_attack_labels"],
    )

    accepted_known_gate = (
        (arrays["known_current_probabilities"] >= thresholds["current"])
        & (arrays["known_confidences"] >= thresholds["known"])
    ) if arrays["known_targets"].size and arrays["known_confidences"].size else np.zeros(0, dtype=bool)
    accepted_known_macro_f1 = (
        float(
            v3_train.compute_multiclass_macro_f1(
                arrays["known_targets"][accepted_known_gate],
                arrays["known_predictions"][accepted_known_gate],
            )
        )
        if accepted_known_gate.size
        else float("nan")
    )
    accepted_known_metrics_by_label = v3_train.compute_multiclass_metrics_by_label(
        arrays["known_targets"],
        arrays["known_predictions"],
        checkpoint_bundle["known_attack_labels"],
        accepted_mask=accepted_known_gate,
    )
    accepted_known_targets = arrays["known_targets"][accepted_known_gate] if accepted_known_gate.size else np.array([])
    accepted_known_family_probabilities = (
        arrays["known_family_probabilities"][accepted_known_gate]
        if accepted_known_gate.size and arrays["known_family_probabilities"].size
        else np.zeros(
            (
                0,
                arrays["known_family_probabilities"].shape[1]
                if arrays["known_family_probabilities"].ndim == 2
                else 0,
            ),
            dtype=np.float64,
        )
    )
    accepted_known_metrics_by_label = v3_train.add_ovr_curve_metrics_by_label(
        accepted_known_metrics_by_label,
        accepted_known_targets,
        accepted_known_family_probabilities,
        checkpoint_bundle["known_attack_labels"],
    )

    summary = {
        "split": split_name,
        "checkpoint": str(checkpoint_bundle["checkpoint_path"]),
        "device": device.type,
        "future_task_enabled": checkpoint_bundle["future_task_enabled"],
        "seq_len": checkpoint_bundle["seq_len"],
        "stride": checkpoint_bundle["stride"],
        "pseudo_zero_day_families": checkpoint_bundle["pseudo_zero_day_families"],
        "thresholds_from_validation": checkpoint_bundle.get("validation_thresholds", thresholds),
        "thresholds_applied": thresholds,
        "threshold_overrides": checkpoint_bundle.get("threshold_overrides", {}),
        "calibration_bins": int(calibration_bins),
        "bootstrap_samples": int(bootstrap_samples),
        "threshold_target_recall": checkpoint_bundle["threshold_target_recall"],
        "future_threshold_target_recall": checkpoint_bundle["future_threshold_target_recall"],
        "future_pre_onset_exclusion_gap_minutes": checkpoint_bundle[
            "future_pre_onset_exclusion_gap_minutes"
        ],
        "ood_threshold_target_recall": checkpoint_bundle["ood_threshold_target_recall"],
        "known_target_unknown_recall": checkpoint_bundle["known_target_unknown_recall"],
        "run_mode": checkpoint_bundle.get("run_mode", v3_train.RUN_MODE_CLOSED_SET),
        "thesis_claim": checkpoint_bundle.get("thesis_claim", v3_train.DEFAULT_CLOSED_SET_THESIS_CLAIM),
        "decision_policy": checkpoint_bundle.get("decision_policy", v3_train.DEFAULT_DECISION_POLICY),
        "novelty_score_mode": novelty_score_mode,
        "task_activation": checkpoint_bundle.get("task_activation", {}),
        "unknown_head_active": checkpoint_bundle.get("unknown_head_active", use_ood_head),
        "use_ood_head": checkpoint_bundle.get("use_ood_head", True),
        "use_reconstruction_hybrid_ood": use_reconstruction_hybrid_ood,
        "ood_score_mode": checkpoint_bundle.get(
            "unknown_risk_score_mode",
            resolve_unknown_risk_score_mode(use_ood_head, use_reconstruction_hybrid_ood),
        ),
        "unknown_risk_score_mode": checkpoint_bundle.get(
            "unknown_risk_score_mode",
            resolve_unknown_risk_score_mode(use_ood_head, use_reconstruction_hybrid_ood),
        ),
        "reconstruction_calibration": checkpoint_bundle.get("reconstruction_calibration"),
        "reconstruction_validation_mae_mask_ratio": reconstruction_validation_mae_mask_ratio,
        "reconstruction_validation_mfm_mask_ratio": reconstruction_validation_mfm_mask_ratio,
        "future_horizons_minutes": future_horizons_minutes,
        "future_horizon_labels": future_horizon_labels,
        **profile,
        "current_at_validation_threshold": current_at_validation,
        "current_oracle_for_this_split_diagnostic_only": current_oracle,
        "ood_at_validation_threshold": ood_at_validation,
        "ood_oracle_for_this_split_diagnostic_only": ood_oracle,
        "ood_head_at_validation_threshold": ood_head_at_validation,
        "ood_head_oracle_for_this_split_diagnostic_only": ood_head_oracle,
        "reconstruction_ood_at_validation_threshold": reconstruction_ood_at_validation,
        "reconstruction_ood_oracle_for_this_split_diagnostic_only": reconstruction_ood_oracle,
        "future_at_validation_threshold": future_at_validation,
        "future_at_validation_threshold_by_horizon": future_at_validation_by_horizon,
        "future_oracle_for_this_split_diagnostic_only": future_oracle,
        "future_oracle_for_this_split_diagnostic_only_by_horizon": future_oracle_by_horizon,
        "known_family_accuracy": raw_known_accuracy,
        "known_family_macro_f1": raw_known_macro_f1,
        "known_family_metrics_by_label": raw_known_metrics_by_label,
        "known_gate_at_validation_threshold": known_gate_at_validation,
        "known_family_accepted_macro_f1": accepted_known_macro_f1,
        "known_family_accepted_metrics_by_label": accepted_known_metrics_by_label,
        "known_gate_oracle_for_this_split_diagnostic_only": known_oracle,
        "unknown_warning_recall": ood_at_validation["recall"] if ood_at_validation is not None else None,
        "unknown_label_positive_count": int(arrays["ood_labels"].sum()) if arrays["ood_labels"].size else 0,
        "reconstruction_score_mean": float(arrays["reconstruction_scores"].mean()) if arrays["reconstruction_scores"].size else 0.0,
        "mae_reconstruction_score_mean": float(arrays["mae_reconstruction_scores"].mean()) if arrays["mae_reconstruction_scores"].size else 0.0,
        "mfm_reconstruction_score_mean": float(arrays["mfm_reconstruction_scores"].mean()) if arrays["mfm_reconstruction_scores"].size else 0.0,
        "mean_future_lead_minutes": mean_future_lead_minutes,
        "mean_future_lead_minutes_by_horizon": mean_future_lead_minutes_by_horizon,
        "bootstrap_confidence_intervals": {
            "current": current_bootstrap,
            "ood": ood_bootstrap,
            "ood_head": ood_head_bootstrap,
            "reconstruction_ood": reconstruction_ood_bootstrap,
            "future": future_bootstrap,
            "future_by_horizon": future_bootstrap_by_horizon,
        } if bootstrap_samples > 0 else None,
    }
    print(f"[{split_name}] Evaluation complete.", flush=True)
    return {"summary": summary, "arrays": arrays, "profile": profile}


def load_epoch_history(checkpoint_dir, device):
    checkpoint_dir = Path(checkpoint_dir)
    epoch_files = sorted(
        list(checkpoint_dir.glob("nids_multitask_epoch_*.pt"))
        + list(checkpoint_dir.glob("nids_multitask_future_refine_epoch_*.pt"))
        + list(checkpoint_dir.glob("nids_multitask_family_refine_epoch_*.pt")),
        key=lambda path: (
            0 if path.name.startswith("nids_multitask_epoch_") else 1 if "future_refine" in path.name else 2,
            base_train.extract_epoch_index(path),
        ),
    )
    rows = []
    for epoch_path in epoch_files:
        try:
            checkpoint = torch.load(epoch_path, map_location=device, weights_only=False)
        except Exception as exc:
            print(f"Skipping unreadable checkpoint {epoch_path.name}: {exc}")
            continue

        validation_metrics = checkpoint.get("validation_metrics") or {}
        current_metrics = validation_metrics.get("current", {})
        best_current = validation_metrics.get("best_current", {})
        ood_metrics = validation_metrics.get("ood", {})
        best_ood = validation_metrics.get("best_ood", {})
        ood_head_metrics = validation_metrics.get("ood_head", {})
        best_ood_head = validation_metrics.get("best_ood_head", {})
        reconstruction_ood_metrics = validation_metrics.get("reconstruction_unknown", {})
        best_reconstruction_ood = validation_metrics.get("best_reconstruction_unknown", {})
        future_metrics = validation_metrics.get("future", {})
        best_future = validation_metrics.get("best_future", {})
        future_metrics_by_horizon = validation_metrics.get("future_by_horizon", {})
        best_future_by_horizon = validation_metrics.get("best_future_by_horizon", {})
        mean_future_lead_by_horizon = validation_metrics.get("mean_future_lead_minutes_by_horizon", {})
        best_known = validation_metrics.get("best_known", {})
        thresholds = checkpoint.get("thresholds", {})
        future_horizons_minutes = v3_train.normalize_future_horizons_minutes(
            checkpoint.get("future_horizons_minutes"),
            checkpoint.get("future_horizon_minutes", v3_train.DEFAULT_FUTURE_HORIZON_MINUTES),
        )
        future_horizon_labels = validation_metrics.get("future_horizon_labels") or v3_train.build_future_horizon_labels(future_horizons_minutes)
        future_thresholds = v3_train.normalize_future_thresholds(
            thresholds.get("future"),
            future_horizons_minutes,
        )
        loss_metrics = validation_metrics.get("loss", {})

        if "future_refine" in epoch_path.name:
            epoch_index = base_train.extract_epoch_index(epoch_path)
            stage_name = "future_refinement"
        elif "family_refine" in epoch_path.name:
            epoch_index = base_train.extract_epoch_index(epoch_path)
            stage_name = "family_refinement"
        else:
            epoch_index = int(checkpoint.get("epoch", base_train.extract_epoch_index(epoch_path)))
            stage_name = "main_training"
        future_threshold_values = np.asarray(list(future_thresholds.values()), dtype=np.float64)
        row = {
            "checkpoint_file": epoch_path.name,
            "stage_name": stage_name,
            "epoch_index_zero_based": epoch_index,
            "epoch_number": epoch_index + 1,
            "run_mode": str(checkpoint.get("run_mode", v3_train.RUN_MODE_CLOSED_SET)),
            "novelty_score_mode": str(
                checkpoint.get("novelty_score_mode", v3_train.DEFAULT_NOVELTY_SCORE_MODE)
            ),
            "unknown_head_active": bool(checkpoint.get("unknown_head_active", checkpoint.get("use_ood_head", True))),
            "validation_score": float(checkpoint.get("validation_score", float("nan"))),
            "current_threshold": float(thresholds.get("current", float("nan"))),
            "known_threshold": float(thresholds.get("known", float("nan"))),
            "future_threshold": (
                float(np.nanmean(future_threshold_values))
                if future_threshold_values.size and not np.isnan(future_threshold_values).all()
                else float("nan")
            ),
            "ood_threshold": float(thresholds.get("ood", float("nan"))),
            "current_auc": float(current_metrics.get("auc", float("nan"))),
            "current_pr_auc": float(current_metrics.get("pr_auc", float("nan"))),
            "current_precision": float(current_metrics.get("precision", float("nan"))),
            "current_recall": float(current_metrics.get("recall", float("nan"))),
            "current_f1": float(current_metrics.get("f1", float("nan"))),
            "current_false_positive_rate": float(current_metrics.get("false_positive_rate", float("nan"))),
            "best_current_auc": float(best_current.get("auc", float("nan"))),
            "best_current_pr_auc": float(best_current.get("pr_auc", float("nan"))),
            "best_current_precision": float(best_current.get("precision", float("nan"))),
            "best_current_recall": float(best_current.get("recall", float("nan"))),
            "best_current_f1": float(best_current.get("f1", float("nan"))),
            "best_current_false_positive_rate": float(best_current.get("false_positive_rate", float("nan"))),
            "best_current_threshold": float(best_current.get("threshold", float("nan"))),
            "ood_auc": float(ood_metrics.get("auc", float("nan"))),
            "ood_pr_auc": float(ood_metrics.get("pr_auc", float("nan"))),
            "ood_precision": float(ood_metrics.get("precision", float("nan"))),
            "ood_recall": float(ood_metrics.get("recall", float("nan"))),
            "ood_f1": float(ood_metrics.get("f1", float("nan"))),
            "ood_false_positive_rate": float(ood_metrics.get("false_positive_rate", float("nan"))),
            "best_ood_auc": float(best_ood.get("auc", float("nan"))),
            "best_ood_pr_auc": float(best_ood.get("pr_auc", float("nan"))),
            "best_ood_precision": float(best_ood.get("precision", float("nan"))),
            "best_ood_recall": float(best_ood.get("recall", float("nan"))),
            "best_ood_f1": float(best_ood.get("f1", float("nan"))),
            "best_ood_false_positive_rate": float(best_ood.get("false_positive_rate", float("nan"))),
            "best_ood_threshold": float(best_ood.get("threshold", float("nan"))),
            "ood_head_pr_auc": float(ood_head_metrics.get("pr_auc", float("nan"))),
            "best_ood_head_pr_auc": float(best_ood_head.get("pr_auc", float("nan"))),
            "reconstruction_ood_pr_auc": float(reconstruction_ood_metrics.get("pr_auc", float("nan"))),
            "best_reconstruction_ood_pr_auc": float(best_reconstruction_ood.get("pr_auc", float("nan"))),
            "future_auc": float(future_metrics.get("auc", float("nan"))),
            "future_pr_auc": float(future_metrics.get("pr_auc", float("nan"))),
            "future_precision": float(future_metrics.get("precision", float("nan"))),
            "future_recall": float(future_metrics.get("recall", float("nan"))),
            "future_f1": float(future_metrics.get("f1", float("nan"))),
            "future_false_positive_rate": float(future_metrics.get("false_positive_rate", float("nan"))),
            "best_future_auc": float(best_future.get("auc", float("nan"))),
            "best_future_pr_auc": float(best_future.get("pr_auc", float("nan"))),
            "best_future_precision": float(best_future.get("precision", float("nan"))),
            "best_future_recall": float(best_future.get("recall", float("nan"))),
            "best_future_f1": float(best_future.get("f1", float("nan"))),
            "best_future_false_positive_rate": float(best_future.get("false_positive_rate", float("nan"))),
            "best_future_threshold": float(best_future.get("threshold", float("nan"))),
            "future_pre_onset_exclusion_gap_minutes": float(
                checkpoint.get(
                    "future_pre_onset_exclusion_gap_minutes",
                    v3_train.DEFAULT_FUTURE_PRE_ONSET_EXCLUSION_GAP_MINUTES,
                )
            ),
            "known_family_accuracy": float(validation_metrics.get("known_family_accuracy", float("nan"))),
            "known_family_accepted_accuracy": float(validation_metrics.get("known_family_accepted_accuracy", float("nan"))),
            "known_family_coverage": float(validation_metrics.get("known_family_coverage", float("nan"))),
            "known_balanced_score": float(best_known.get("balanced_score", float("nan"))),
            "unknown_warning_recall": float(validation_metrics.get("unknown_warning_recall", float("nan"))),
            "unknown_label_positive_count": float(validation_metrics.get("unknown_label_positive_count", float("nan"))),
            "reconstruction_score_mean": float(validation_metrics.get("reconstruction_score_mean", float("nan"))),
            "mae_reconstruction_score_mean": float(
                validation_metrics.get("mae_reconstruction_score_mean", float("nan"))
            ),
            "mfm_reconstruction_score_mean": float(
                validation_metrics.get("mfm_reconstruction_score_mean", float("nan"))
            ),
            "mean_future_lead_minutes": float(validation_metrics.get("mean_future_lead_minutes", float("nan"))),
            "loss_total": float(loss_metrics.get("total", float("nan"))),
            "loss_current": float(loss_metrics.get("current", float("nan"))),
            "loss_family": float(loss_metrics.get("family", float("nan"))),
            "loss_future": float(loss_metrics.get("future", float("nan"))),
            "loss_ood": float(loss_metrics.get("ood", float("nan"))),
            "loss_reconstruction": float(loss_metrics.get("reconstruction", float("nan"))),
            "loss_reconstruction_masked_mse": float(
                loss_metrics.get("reconstruction_masked_mse", float("nan"))
            ),
            "loss_reconstruction_full_mse": float(
                loss_metrics.get("reconstruction_full_mse", float("nan"))
            ),
            "loss_unknown_regularizer": float(loss_metrics.get("unknown_regularizer", float("nan"))),
        }

        for horizon_label in future_horizon_labels:
            safe_label = horizon_label.replace(".", "_")
            horizon_metrics = future_metrics_by_horizon.get(horizon_label, {})
            best_horizon_metrics = best_future_by_horizon.get(horizon_label, {})
            row[f"future_threshold_{safe_label}"] = float(future_thresholds.get(horizon_label, float("nan")))
            row[f"future_auc_{safe_label}"] = float(horizon_metrics.get("auc", float("nan")))
            row[f"future_pr_auc_{safe_label}"] = float(horizon_metrics.get("pr_auc", float("nan")))
            row[f"future_precision_{safe_label}"] = float(horizon_metrics.get("precision", float("nan")))
            row[f"future_recall_{safe_label}"] = float(horizon_metrics.get("recall", float("nan")))
            row[f"future_f1_{safe_label}"] = float(horizon_metrics.get("f1", float("nan")))
            row[f"best_future_auc_{safe_label}"] = float(best_horizon_metrics.get("auc", float("nan")))
            row[f"best_future_pr_auc_{safe_label}"] = float(best_horizon_metrics.get("pr_auc", float("nan")))
            row[f"best_future_precision_{safe_label}"] = float(best_horizon_metrics.get("precision", float("nan")))
            row[f"best_future_recall_{safe_label}"] = float(best_horizon_metrics.get("recall", float("nan")))
            row[f"best_future_f1_{safe_label}"] = float(best_horizon_metrics.get("f1", float("nan")))
            row[f"best_future_threshold_{safe_label}"] = float(best_horizon_metrics.get("threshold", float("nan")))
            row[f"future_valid_count_{safe_label}"] = float(horizon_metrics.get("valid_count", float("nan")))
            row[f"future_raw_positive_count_{safe_label}"] = float(
                horizon_metrics.get("raw_positive_count", float("nan"))
            )
            row[f"future_valid_positive_count_{safe_label}"] = float(
                horizon_metrics.get("valid_positive_count", float("nan"))
            )
            row[f"future_ignored_near_onset_positive_count_{safe_label}"] = float(
                horizon_metrics.get("ignored_near_onset_positive_count", float("nan"))
            )
            row[f"mean_future_lead_minutes_{safe_label}"] = float(
                mean_future_lead_by_horizon.get(horizon_label, float("nan"))
            )

        rows.append(row)
    return pd.DataFrame(rows).sort_values("epoch_number") if rows else pd.DataFrame()


def collect_prefixed_history_columns(history_df, prefix):
    return sorted(column_name for column_name in history_df.columns if column_name.startswith(prefix))


def sample_continuous_rows(split_name, sample_rows, feature_names):
    split_path = resolve_split_path(split_name)
    if split_path is None:
        return None
    dataset = load_from_disk(str(split_path))
    total_rows = len(dataset)
    if total_rows == 0:
        return None

    sample_rows = min(int(sample_rows), total_rows)
    if sample_rows <= 0:
        return None
    if sample_rows == total_rows:
        selected = dataset
    else:
        indices = np.linspace(0, total_rows - 1, sample_rows, dtype=np.int64)
        selected = dataset.select(indices.tolist())

    data = {
        feature_name: np.asarray(selected[feature_name], dtype=np.float64)
        for feature_name in feature_names
    }
    frame = pd.DataFrame(data)
    frame = frame.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    frame = np.sign(frame) * np.log1p(np.abs(frame))
    return pd.DataFrame(frame, columns=feature_names)


def save_figure(fig, output_path, manifest_entries, title, description, dpi):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    manifest_entries.append(
        {
            "path": str(output_path),
            "title": title,
            "description": description,
        }
    )


def plot_validation_overview(history_df, output_path, manifest_entries, dpi):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    x = history_df["epoch_number"]
    future_pr_auc_columns = collect_prefixed_history_columns(history_df, "best_future_pr_auc_")
    future_auc_columns = collect_prefixed_history_columns(history_df, "best_future_auc_")

    axes[0, 0].plot(x, history_df["validation_score"], marker="o", label="Validation score")
    axes[0, 0].plot(x, history_df["best_current_pr_auc"], marker="o", label="Current PR-AUC")
    axes[0, 0].plot(x, history_df["best_ood_pr_auc"], marker="o", label="Unknown-risk PR-AUC")
    if "best_ood_head_pr_auc" in history_df:
        axes[0, 0].plot(x, history_df["best_ood_head_pr_auc"], marker="o", label="Raw unknown-head PR-AUC")
    if "best_reconstruction_ood_pr_auc" in history_df:
        axes[0, 0].plot(
            x,
            history_df["best_reconstruction_ood_pr_auc"],
            marker="o",
            label="Reconstruction novelty PR-AUC",
        )
    axes[0, 0].plot(x, history_df["best_future_pr_auc"], marker="o", label="Future PR-AUC (macro)")
    for column_name in future_pr_auc_columns:
        axes[0, 0].plot(
            x,
            history_df[column_name],
            marker="o",
            alpha=0.7,
            label=f"Future PR-AUC {column_name.removeprefix('best_future_pr_auc_')}",
        )
    axes[0, 0].set_title("Validation Score And PR-AUC")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(x, history_df["best_current_auc"], marker="o", label="Current AUC")
    axes[0, 1].plot(x, history_df["best_ood_auc"], marker="o", label="Unknown-risk AUC")
    axes[0, 1].plot(x, history_df["best_future_auc"], marker="o", label="Future AUC (macro)")
    for column_name in future_auc_columns:
        axes[0, 1].plot(
            x,
            history_df[column_name],
            marker="o",
            alpha=0.7,
            label=f"Future AUC {column_name.removeprefix('best_future_auc_')}",
        )
    axes[0, 1].set_title("AUC Curves")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(x, history_df["best_current_precision"], marker="o", label="Current precision")
    axes[1, 0].plot(x, history_df["best_current_recall"], marker="o", label="Current recall")
    axes[1, 0].plot(x, history_df["best_current_f1"], marker="o", label="Current F1")
    axes[1, 0].set_title("Present Detection Metrics")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(x, history_df["known_family_accuracy"], marker="o", label="Raw known accuracy")
    axes[1, 1].plot(x, history_df["known_family_accepted_accuracy"], marker="o", label="Accepted known accuracy")
    axes[1, 1].plot(x, history_df["best_ood_recall"], marker="o", label="Unknown-risk recall")
    axes[1, 1].set_title("Family And Novelty Metrics")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    save_figure(
        fig,
        output_path,
        manifest_entries,
        "Validation overview",
        "Training-time validation score, present detection PR-AUC, unknown-risk diagnostics, future-warning metrics, and family acceptance metrics across epochs.",
        dpi,
    )


def plot_validation_thresholds_and_losses(history_df, output_path, manifest_entries, dpi):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    x = history_df["epoch_number"]
    future_threshold_columns = collect_prefixed_history_columns(history_df, "future_threshold_")
    mean_future_lead_columns = collect_prefixed_history_columns(history_df, "mean_future_lead_minutes_")

    axes[0, 0].plot(x, history_df["current_threshold"], marker="o", label="Current")
    axes[0, 0].plot(x, history_df["known_threshold"], marker="o", label="Known")
    axes[0, 0].plot(x, history_df["future_threshold"], marker="o", label="Future (macro)")
    for column_name in future_threshold_columns:
        axes[0, 0].plot(
            x,
            history_df[column_name],
            marker="o",
            alpha=0.7,
            label=f"Future {column_name.removeprefix('future_threshold_')}",
        )
    axes[0, 0].plot(x, history_df["ood_threshold"], marker="o", label="Unknown-risk")
    axes[0, 0].set_title("Threshold Trajectories")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(x, history_df["loss_total"], marker="o", label="Total")
    axes[0, 1].plot(x, history_df["loss_current"], marker="o", label="Current")
    axes[0, 1].plot(x, history_df["loss_family"], marker="o", label="Family")
    axes[0, 1].plot(x, history_df["loss_future"], marker="o", label="Future")
    axes[0, 1].plot(x, history_df["loss_ood"], marker="o", label="Unknown head")
    if "loss_reconstruction" in history_df:
        axes[0, 1].plot(x, history_df["loss_reconstruction"], marker="o", label="Reconstruction")
    axes[0, 1].plot(x, history_df["loss_unknown_regularizer"], marker="o", label="Unknown regularizer")
    axes[0, 1].set_title("Validation Loss Components")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(x, history_df["known_family_coverage"], marker="o", label="Known coverage")
    axes[1, 0].plot(x, history_df["known_balanced_score"], marker="o", label="Known balanced score")
    axes[1, 0].plot(x, history_df["best_ood_f1"], marker="o", label="Unknown-risk F1")
    axes[1, 0].plot(x, history_df["mean_future_lead_minutes"], marker="o", label="Mean future lead (macro)")
    for column_name in mean_future_lead_columns:
        axes[1, 0].plot(
            x,
            history_df[column_name],
            marker="o",
            alpha=0.7,
            label=f"Lead {column_name.removeprefix('mean_future_lead_minutes_')}",
        )
    axes[1, 0].set_title("Coverage And Lead-Time Diagnostics")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(x, history_df["best_future_precision"], marker="o", label="Future precision")
    axes[1, 1].plot(x, history_df["best_future_recall"], marker="o", label="Future recall")
    axes[1, 1].plot(x, history_df["best_future_f1"], marker="o", label="Future F1")
    axes[1, 1].set_title("Future Warning Metrics")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    save_figure(
        fig,
        output_path,
        manifest_entries,
        "Thresholds and losses",
        "Current, family, future, and unknown-risk thresholds together with validation losses, coverage, and lead-time diagnostics across epochs.",
        dpi,
    )


def plot_binary_curves(labels, scores, auc_value, pr_auc_value, output_path, manifest_entries, title, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    recall, precision = binary_pr_curve(labels, scores)
    fpr, tpr = binary_roc_curve(labels, scores)

    axes[0].plot(recall, precision, color="tab:blue")
    axes[0].set_title(f"{title} PR Curve | PR-AUC={pr_auc_value:.3f}" if pr_auc_value == pr_auc_value else f"{title} PR Curve")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].grid(alpha=0.3)

    axes[1].plot(fpr, tpr, color="tab:orange")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="grey", alpha=0.6)
    axes[1].set_title(f"{title} ROC Curve | AUC={auc_value:.3f}" if auc_value == auc_value else f"{title} ROC Curve")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].grid(alpha=0.3)

    save_figure(fig, output_path, manifest_entries, title, f"Precision-recall and ROC curves for {title.lower()}.", dpi)


def plot_threshold_sweep(sweep_df, selected_threshold, output_path, manifest_entries, title, dpi):
    if sweep_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(sweep_df["threshold"], sweep_df["precision"], label="Precision")
    ax.plot(sweep_df["threshold"], sweep_df["recall"], label="Recall")
    ax.plot(sweep_df["threshold"], sweep_df["f1"], label="F1")
    ax.axvline(float(selected_threshold), color="black", linestyle="--", label=f"Selected={selected_threshold:.3f}")
    ax.set_title(title)
    ax.set_xlabel("Threshold")
    ax.grid(alpha=0.3)
    ax.legend()

    save_figure(fig, output_path, manifest_entries, title, f"Precision, recall, and F1 as a function of threshold for {title.lower()}.", dpi)


def plot_score_histogram(labels, scores, threshold, output_path, manifest_entries, title, dpi):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    negative_mask = labels == 0
    positive_mask = labels == 1
    if negative_mask.any():
        ax.hist(scores[negative_mask], bins=40, alpha=0.6, label="Negative", density=True)
    if positive_mask.any():
        ax.hist(scores[positive_mask], bins=40, alpha=0.6, label="Positive", density=True)
    ax.axvline(float(threshold), color="black", linestyle="--", label=f"Threshold={threshold:.3f}")
    ax.set_title(title)
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)

    save_figure(fig, output_path, manifest_entries, title, f"Score histogram by ground-truth label for {title.lower()}.", dpi)


def annotate_confusion_matrix(ax, matrix, fontsize=9, skip_zero=False):
    matrix = np.asarray(matrix, dtype=np.int64)
    total_count = int(matrix.sum())
    max_count = int(matrix.max()) if matrix.size else 0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            cell_count = int(matrix[row_idx, col_idx])
            if skip_zero and cell_count == 0:
                continue
            cell_pct = 100.0 * cell_count / total_count if total_count > 0 else 0.0
            text_color = "white" if cell_count >= max_count * 0.55 and max_count > 0 else "black"
            ax.text(
                col_idx,
                row_idx,
                f"{cell_count}\n{cell_pct:.1f}%",
                ha="center",
                va="center",
                color=text_color,
                fontsize=fontsize,
            )


def plot_confusion_matrix(matrix, output_path, manifest_entries, title, class_names, dpi):
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(class_names)), labels=class_names)
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    annotate_confusion_matrix(ax, matrix)

    save_figure(fig, output_path, manifest_entries, title, f"Confusion matrix for {title.lower()}.", dpi)


def plot_reliability_diagram(labels, scores, output_path, manifest_entries, title, dpi, bins=10):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0:
        return

    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    bin_indices = np.digitize(scores, bin_edges[1:-1], right=False)
    bin_centers = []
    empirical_rates = []
    counts = []
    for bin_index in range(bins):
        mask = bin_indices == bin_index
        if not mask.any():
            continue
        bin_centers.append(float(scores[mask].mean()))
        empirical_rates.append(float(labels[mask].mean()))
        counts.append(int(mask.sum()))

    if not bin_centers:
        return

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    ax.plot(bin_centers, empirical_rates, marker="o", label="Observed")
    for x_value, y_value, count in zip(bin_centers, empirical_rates, counts):
        ax.text(x_value, y_value, str(count), fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.legend()
    ax.grid(alpha=0.3)

    save_figure(fig, output_path, manifest_entries, title, f"Reliability diagram for {title.lower()}.", dpi)


def plot_known_gate_tradeoff(
    known_current_probabilities,
    known_confidences,
    known_predictions,
    known_targets,
    unknown_current_probabilities,
    unknown_confidences,
    current_threshold,
    selected_threshold,
    output_path,
    manifest_entries,
    title,
    dpi,
):
    candidate_parts = [np.linspace(0.0, 1.0, 101)]
    for values in [known_confidences, unknown_confidences]:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            continue
        candidate_parts.append(np.percentile(values, v3_train.PERCENTILE_CANDIDATES))
        candidate_parts.append(np.quantile(values, np.linspace(0.0, 1.0, 201)))
    candidates = np.unique(np.concatenate(candidate_parts))
    if candidates.size == 0:
        return

    rows = []
    for threshold_value in np.sort(candidates):
        metrics = compute_known_gate_metrics_at_threshold(
            known_current_probabilities,
            known_confidences,
            known_predictions,
            known_targets,
            unknown_current_probabilities,
            unknown_confidences,
            current_threshold,
            threshold_value,
        )
        rows.append(metrics)
    curve_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(curve_df["threshold"], curve_df["accepted_accuracy"], label="Accepted known accuracy")
    ax.plot(curve_df["threshold"], curve_df["known_coverage"], label="Known coverage")
    ax.plot(curve_df["threshold"], curve_df["unknown_recall"], label="Unknown recall")
    ax.axvline(float(selected_threshold), color="black", linestyle="--", label=f"Selected={selected_threshold:.3f}")
    ax.set_title(title)
    ax.set_xlabel("Known confidence threshold")
    ax.grid(alpha=0.3)
    ax.legend()

    save_figure(fig, output_path, manifest_entries, title, f"Trade-off curve for the known-versus-unknown confidence gate in {title.lower()}.", dpi)


def plot_family_confusion_matrix(known_targets, known_predictions, label_names, output_path, manifest_entries, title, dpi):
    if len(label_names) == 0 or len(known_targets) == 0:
        return

    matrix = np.zeros((len(label_names), len(label_names)), dtype=np.int64)
    for true_label, predicted_label in zip(known_targets, known_predictions):
        matrix[int(true_label), int(predicted_label)] += 1

    fig, ax = plt.subplots(figsize=(max(7, len(label_names) * 0.9), max(6, len(label_names) * 0.7)))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(label_names)), labels=label_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(label_names)), labels=label_names)
    ax.set_xlabel("Predicted family")
    ax.set_ylabel("True family")
    ax.set_title(title)
    annotate_confusion_matrix(ax, matrix, fontsize=8, skip_zero=True)

    save_figure(fig, output_path, manifest_entries, title, f"Confusion matrix across known attack families for {title.lower()}.", dpi)


def plot_future_lead_histogram(future_leads, future_labels, future_scores, threshold, output_path, manifest_entries, title, dpi):
    future_leads = np.asarray(future_leads, dtype=np.float64)
    future_labels = np.asarray(future_labels, dtype=np.int64)
    future_scores = np.asarray(future_scores, dtype=np.float64)
    if future_leads.size == 0:
        return

    positive_mask = future_labels == 1
    hit_mask = positive_mask & (future_scores >= threshold)
    if not positive_mask.any():
        return

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(future_leads[positive_mask], bins=40, alpha=0.6, label="All future positives", density=True)
    if hit_mask.any():
        ax.hist(future_leads[hit_mask], bins=40, alpha=0.6, label="Threshold hits", density=True)
    ax.set_title(title)
    ax.set_xlabel("Future lead time (minutes)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)

    save_figure(fig, output_path, manifest_entries, title, f"Distribution of future-event lead times for {title.lower()}.", dpi)


def plot_split_sequence_distribution(split_profiles, output_path, manifest_entries, dpi):
    rows = [profile for profile in split_profiles.values() if profile is not None]
    if not rows:
        return
    frame = pd.DataFrame(rows)
    splits = frame["split"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(splits, frame["sequence_count"], color="tab:blue")
    axes[0].set_title("Sequence Count By Split")
    axes[0].set_ylabel("Sequence count")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(splits, frame["current_positive_rate"], color="tab:orange")
    axes[1].set_title("Attack Positive Rate By Split")
    axes[1].set_ylabel("Positive rate")
    axes[1].tick_params(axis="x", rotation=20)

    save_figure(
        fig,
        output_path,
        manifest_entries,
        "Split distribution overview",
        "Sequence counts and attack positive rates across train/validation/test and the test_ood held-out-family benchmark split.",
        dpi,
    )


def plot_attack_family_distribution(split_profiles, output_path, manifest_entries, dpi, top_k=12):
    rows = [profile for profile in split_profiles.values() if profile is not None]
    if not rows:
        return

    total_counts = Counter()
    for profile in rows:
        total_counts.update(profile["attack_family_counts"])
    top_families = [family_name for family_name, _ in total_counts.most_common(top_k)]
    if not top_families:
        return

    x = np.arange(len(top_families))
    width = 0.8 / max(len(rows), 1)
    fig, ax = plt.subplots(figsize=(max(12, len(top_families) * 0.8), 5.5))
    for idx, profile in enumerate(rows):
        counts = [profile["attack_family_counts"].get(family_name, 0) for family_name in top_families]
        ax.bar(x + idx * width, counts, width=width, label=profile["split"])

    ax.set_xticks(x + width * (len(rows) - 1) / 2.0, labels=top_families, rotation=40, ha="right")
    ax.set_ylabel("Window count")
    ax.set_title("Attack Family Distribution By Split")
    ax.legend()

    save_figure(fig, output_path, manifest_entries, "Attack family distribution", "Grouped bar chart of attack-family window counts across splits.", dpi)


def plot_known_unknown_distribution(split_profiles, output_path, manifest_entries, dpi):
    rows = [profile for profile in split_profiles.values() if profile is not None]
    if not rows:
        return

    splits = [profile["split"] for profile in rows]
    benign_counts = [profile["sequence_count"] - profile["current_positive_count"] for profile in rows]
    known_counts = [profile["known_family_count"] for profile in rows]
    unknown_counts = [profile["unknown_positive_count"] for profile in rows]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(splits, benign_counts, label="Benign")
    ax.bar(splits, known_counts, bottom=benign_counts, label="Known attack")
    ax.bar(
        splits,
        unknown_counts,
        bottom=np.asarray(benign_counts) + np.asarray(known_counts),
        label="Unknown / held-out attack",
    )
    ax.set_title("Benign, Known, And Unknown Window Composition By Split")
    ax.set_ylabel("Window count")
    ax.legend()

    save_figure(fig, output_path, manifest_entries, "Known vs unknown composition", "Stacked bar chart showing benign, known-attack, and unknown-attack window counts across splits.", dpi)


def plot_feature_correlation_matrix(frame, output_path, manifest_entries, dpi):
    if frame is None or frame.empty:
        return
    corr = frame.corr()
    fig, ax = plt.subplots(figsize=(16, 14))
    image = ax.imshow(corr.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(corr.columns)), labels=corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(corr.columns)), labels=corr.columns, fontsize=7)
    ax.set_title("Train Continuous Feature Correlation Matrix")

    save_figure(fig, output_path, manifest_entries, "Feature correlation matrix", "Correlation matrix over log-transformed continuous training features sampled from the train split.", dpi)


def plot_feature_shift(frames_by_split, output_path, manifest_entries, dpi, top_k=12):
    valid_frames = {split_name: frame for split_name, frame in frames_by_split.items() if frame is not None and not frame.empty}
    if len(valid_frames) < 2:
        return

    train_frame = valid_frames.get("train")
    if train_frame is None:
        return

    baseline_mean = train_frame.mean()
    top_features = (
        baseline_mean.abs().sort_values(ascending=False).head(top_k).index.tolist()
    )

    x = np.arange(len(top_features))
    width = 0.8 / max(len(valid_frames), 1)
    fig, ax = plt.subplots(figsize=(max(12, len(top_features) * 0.8), 5.5))
    for idx, (split_name, frame) in enumerate(valid_frames.items()):
        mean_values = frame[top_features].mean().values
        ax.bar(x + idx * width, mean_values, width=width, label=split_name)

    ax.set_xticks(x + width * (len(valid_frames) - 1) / 2.0, labels=top_features, rotation=40, ha="right")
    ax.set_title("Feature Mean Shift Across Splits")
    ax.set_ylabel("Mean log-transformed value")
    ax.legend()

    save_figure(
        fig,
        output_path,
        manifest_entries,
        "Feature mean shift",
        "Mean value comparison for the most active continuous features across train/validation/test and the test_ood held-out-family benchmark split.",
        dpi,
    )


def format_metric_value(value):
    return f"{value:.6f}" if value == value else "n/a"


def format_bootstrap_interval(metric_summary):
    if not metric_summary or int(metric_summary.get("valid_samples", 0)) == 0:
        return "n/a"
    lower = format_metric_value(metric_summary.get("lower", float("nan")))
    upper = format_metric_value(metric_summary.get("upper", float("nan")))
    return f"[{lower}, {upper}] (valid bootstrap samples={int(metric_summary.get('valid_samples', 0))})"


def append_family_metrics_table(lines, heading, metrics_by_label, include_coverage=False):
    if not metrics_by_label:
        return

    lines.append(heading)
    if include_coverage:
        lines.append("| Family | Support | Accepted support | Coverage | Precision | Recall | F1 | PR-AUC | ROC-AUC |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for family_name, metrics in metrics_by_label.items():
            lines.append(
                f"| {family_name} | "
                f"{int(metrics.get('support', 0))} | "
                f"{int(metrics.get('accepted_support', 0))} | "
                f"{format_metric_value(metrics.get('coverage', float('nan')))} | "
                f"{format_metric_value(metrics.get('precision', float('nan')))} | "
                f"{format_metric_value(metrics.get('recall', float('nan')))} | "
                f"{format_metric_value(metrics.get('f1', float('nan')))} | "
                f"{format_metric_value(metrics.get('pr_auc', float('nan')))} | "
                f"{format_metric_value(metrics.get('roc_auc', float('nan')))} |"
            )
    else:
        lines.append("| Family | Support | Predicted | Precision | Recall | F1 | PR-AUC | ROC-AUC |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for family_name, metrics in metrics_by_label.items():
            lines.append(
                f"| {family_name} | "
                f"{int(metrics.get('support', 0))} | "
                f"{int(metrics.get('predicted_count', 0))} | "
                f"{format_metric_value(metrics.get('precision', float('nan')))} | "
                f"{format_metric_value(metrics.get('recall', float('nan')))} | "
                f"{format_metric_value(metrics.get('f1', float('nan')))} | "
                f"{format_metric_value(metrics.get('pr_auc', float('nan')))} | "
                f"{format_metric_value(metrics.get('roc_auc', float('nan')))} |"
            )
    lines.append("")


def write_markdown_summary(summary, output_path):
    lines = [f"# Evaluation Summary: {summary['split']}", ""]
    lines.append(f"- Checkpoint: {summary['checkpoint']}")
    lines.append(f"- Run mode: {summary.get('run_mode', v3_train.RUN_MODE_CLOSED_SET)}")
    lines.append(f"- Thesis claim policy: {summary.get('thesis_claim', v3_train.DEFAULT_CLOSED_SET_THESIS_CLAIM)}")
    lines.append(f"- Decision policy: {summary.get('decision_policy', v3_train.DEFAULT_DECISION_POLICY)}")
    lines.append(f"- Novelty score mode: {summary.get('novelty_score_mode', v3_train.DEFAULT_NOVELTY_SCORE_MODE)}")
    lines.append(f"- Unknown-risk score mode: {summary.get('unknown_risk_score_mode', 'disabled')}")
    lines.append(f"- Task activation: {summary.get('task_activation', {})}")
    lines.append(
        f"- Unknown-head supervision active: {summary.get('unknown_head_active', summary.get('use_ood_head', True))}"
    )
    lines.append(
        f"- Hybrid reconstruction-backed unknown risk: {summary.get('use_reconstruction_hybrid_ood', False)}"
    )
    lines.append(f"- Pseudo-zero-day families: {summary['pseudo_zero_day_families']}")
    lines.append(f"- Future horizons: {summary.get('future_horizons_minutes', [])}")
    lines.append(f"- Calibration bins: {summary.get('calibration_bins', 10)}")
    if int(summary.get("bootstrap_samples", 0)) > 0:
        lines.append(f"- Bootstrap samples: {int(summary['bootstrap_samples'])}")
    applied_thresholds = summary.get("thresholds_applied", summary["thresholds_from_validation"])
    lines.append(f"- Current threshold: {applied_thresholds['current']:.6f}")
    lines.append(f"- Known threshold: {applied_thresholds['known']:.6f}")
    if isinstance(applied_thresholds.get("future"), dict):
        lines.append(f"- Future thresholds by horizon: {applied_thresholds['future']}")
    else:
        lines.append(f"- Future threshold: {applied_thresholds['future']:.6f}")
    if summary.get("use_ood_head", True) or summary.get("use_reconstruction_hybrid_ood", False):
        lines.append(f"- Unknown-risk threshold: {applied_thresholds['ood']:.6f}")
    else:
        lines.append("- Unknown-risk threshold: not applicable for checkpoints without novelty or unknown-risk scoring")
    threshold_overrides = summary.get("threshold_overrides") or {}
    if any(value is not None for value in threshold_overrides.values()):
        lines.append(f"- Validation thresholds stored in checkpoint: {summary['thresholds_from_validation']}")
        lines.append(f"- CLI threshold overrides: {threshold_overrides}")
    lines.append("")

    current_metrics = summary["current_at_validation_threshold"]
    bootstrap_confidence_intervals = summary.get("bootstrap_confidence_intervals") or {}
    current_bootstrap = bootstrap_confidence_intervals.get("current") or {}
    ood_bootstrap = bootstrap_confidence_intervals.get("ood") or {}
    ood_head_bootstrap = bootstrap_confidence_intervals.get("ood_head") or {}
    reconstruction_ood_bootstrap = bootstrap_confidence_intervals.get("reconstruction_ood") or {}
    future_bootstrap = bootstrap_confidence_intervals.get("future") or {}
    future_bootstrap_by_horizon = bootstrap_confidence_intervals.get("future_by_horizon") or {}
    calibration_bins = int(summary.get("calibration_bins", 10))
    lines.append("## Present Detection")
    lines.append(f"- PR-AUC: {current_metrics['pr_auc']:.6f}" if current_metrics["pr_auc"] == current_metrics["pr_auc"] else "- PR-AUC: n/a")
    lines.append(f"- AUC: {current_metrics['auc']:.6f}" if current_metrics["auc"] == current_metrics["auc"] else "- AUC: n/a")
    lines.append(f"- Precision: {current_metrics['precision']:.6f}")
    lines.append(f"- Recall: {current_metrics['recall']:.6f}")
    lines.append(f"- F1: {current_metrics['f1']:.6f}")
    lines.append(f"- Benign FPR: {current_metrics['false_positive_rate']:.6f}")
    lines.append(f"- Brier score: {format_metric_value(current_metrics.get('brier_score', float('nan')))}")
    lines.append(f"- ECE ({calibration_bins} bins): {format_metric_value(current_metrics.get('ece', float('nan')))}")
    if current_bootstrap:
        lines.append(f"- PR-AUC 95% bootstrap CI: {format_bootstrap_interval(current_bootstrap.get('pr_auc'))}")
        lines.append(f"- Recall 95% bootstrap CI: {format_bootstrap_interval(current_bootstrap.get('recall'))}")
        lines.append(f"- F1 95% bootstrap CI: {format_bootstrap_interval(current_bootstrap.get('f1'))}")
    lines.append("")

    lines.append("## Novelty And Unknown-Risk")
    ood_metrics = summary["ood_at_validation_threshold"]
    ood_head_metrics = summary.get("ood_head_at_validation_threshold")
    reconstruction_ood_metrics = summary.get("reconstruction_ood_at_validation_threshold")
    ood_score_mode = summary.get("unknown_risk_score_mode", "disabled")
    reconstruction_score_name = (summary.get("reconstruction_calibration") or {}).get(
        "score_name",
        "reconstruction_score",
    )
    if ood_metrics is None:
        lines.append("- Unknown-risk path: not available in this checkpoint")
        lines.append("- Unknown-risk-specific metrics and figures are intentionally skipped")
    else:
        if ood_score_mode == "hybrid_max_raw_unknown_head_and_reconstruction_percentile":
            lines.append("- Decision score: max(raw unknown-head probability, reconstruction novelty percentile)")
            lines.append(
                "- Reconstruction calibration: benign empirical percentile "
                f"with validation MAE/MFM mask ratios "
                f"{summary.get('reconstruction_validation_mae_mask_ratio', float('nan')):.3f}/"
                f"{summary.get('reconstruction_validation_mfm_mask_ratio', float('nan')):.3f}"
            )
        elif ood_score_mode == "reconstruction_percentile_only":
            lines.append("- Decision score: reconstruction novelty percentile only")
        elif ood_score_mode == "raw_unknown_head_only":
            lines.append("- Decision score: raw unknown-head probability only")
        else:
            lines.append("- Decision score: disabled")
        lines.append(f"- Reconstruction novelty score: {reconstruction_score_name}")
        lines.append(f"- Explicit unknown head available in checkpoint: {summary.get('use_ood_head', True)}")
        lines.append(
            f"- Unknown-head supervision active during training: {summary.get('unknown_head_active', summary.get('use_ood_head', True))}"
        )
        lines.append(f"- Unknown-labelled positives on this split: {summary.get('unknown_label_positive_count', 0)}")
        lines.append(f"- PR-AUC: {ood_metrics['pr_auc']:.6f}" if ood_metrics["pr_auc"] == ood_metrics["pr_auc"] else "- PR-AUC: n/a")
        lines.append(f"- AUC: {ood_metrics['auc']:.6f}" if ood_metrics["auc"] == ood_metrics["auc"] else "- AUC: n/a")
        lines.append(f"- Precision: {ood_metrics['precision']:.6f}")
        lines.append(f"- Recall: {ood_metrics['recall']:.6f}")
        lines.append(f"- F1: {ood_metrics['f1']:.6f}")
        lines.append(f"- Benign-and-known FPR: {ood_metrics['false_positive_rate']:.6f}")
        lines.append(f"- Brier score: {format_metric_value(ood_metrics.get('brier_score', float('nan')))}")
        lines.append(f"- ECE ({calibration_bins} bins): {format_metric_value(ood_metrics.get('ece', float('nan')))}")
        if ood_bootstrap:
            lines.append(f"- PR-AUC 95% bootstrap CI: {format_bootstrap_interval(ood_bootstrap.get('pr_auc'))}")
            lines.append(f"- Recall 95% bootstrap CI: {format_bootstrap_interval(ood_bootstrap.get('recall'))}")
            lines.append(f"- F1 95% bootstrap CI: {format_bootstrap_interval(ood_bootstrap.get('f1'))}")
        if ood_head_metrics is not None:
            lines.append(
                f"- Raw unknown-head PR-AUC: {format_metric_value(ood_head_metrics.get('pr_auc', float('nan')))}"
            )
            lines.append(
                f"- Raw unknown-head recall: {format_metric_value(ood_head_metrics.get('recall', float('nan')))}"
            )
            if ood_head_bootstrap:
                lines.append(
                    f"- Raw unknown-head PR-AUC 95% bootstrap CI: {format_bootstrap_interval(ood_head_bootstrap.get('pr_auc'))}"
                )
        if reconstruction_ood_metrics is not None:
            lines.append(
                f"- Reconstruction-only novelty PR-AUC: {format_metric_value(reconstruction_ood_metrics.get('pr_auc', float('nan')))}"
            )
            lines.append(
                f"- Reconstruction-only novelty recall: {format_metric_value(reconstruction_ood_metrics.get('recall', float('nan')))}"
            )
            if reconstruction_ood_bootstrap:
                lines.append(
                    f"- Reconstruction-only novelty PR-AUC 95% bootstrap CI: {format_bootstrap_interval(reconstruction_ood_bootstrap.get('pr_auc'))}"
                )
    lines.append("")

    if summary["future_at_validation_threshold"] is not None:
        future_metrics = summary["future_at_validation_threshold"]
        lines.append("## Future Warning")
        lines.append(
            "- Pre-onset exclusion gap minutes: "
            f"{summary.get('future_pre_onset_exclusion_gap_minutes', 0.0):.3f}"
        )
        lines.append(
            "- Valid future positives on this split: "
            f"{summary.get('future_positive_count', 0)} "
            f"(raw={summary.get('future_raw_positive_count', 0)}, "
            f"ignored_near_onset={summary.get('future_ignored_near_onset_positive_count', 0)})"
        )
        lines.append(f"- Macro PR-AUC: {future_metrics['pr_auc']:.6f}" if future_metrics["pr_auc"] == future_metrics["pr_auc"] else "- Macro PR-AUC: n/a")
        lines.append(f"- Macro AUC: {future_metrics['auc']:.6f}" if future_metrics["auc"] == future_metrics["auc"] else "- Macro AUC: n/a")
        lines.append(f"- Macro precision: {future_metrics['precision']:.6f}")
        lines.append(f"- Macro recall: {future_metrics['recall']:.6f}")
        lines.append(f"- Macro F1: {future_metrics['f1']:.6f}")
        lines.append(f"- Macro benign FPR: {future_metrics['false_positive_rate']:.6f}")
        lines.append(f"- Macro Brier score: {format_metric_value(future_metrics.get('brier_score', float('nan')))}")
        lines.append(f"- Macro ECE ({calibration_bins} bins): {format_metric_value(future_metrics.get('ece', float('nan')))}")
        if future_bootstrap:
            lines.append(f"- Macro PR-AUC 95% bootstrap CI: {format_bootstrap_interval(future_bootstrap.get('pr_auc'))}")
            lines.append(f"- Macro recall 95% bootstrap CI: {format_bootstrap_interval(future_bootstrap.get('recall'))}")
            lines.append(f"- Macro F1 95% bootstrap CI: {format_bootstrap_interval(future_bootstrap.get('f1'))}")
        for horizon_label in summary.get("future_horizon_labels", []):
            horizon_metrics = summary.get("future_at_validation_threshold_by_horizon", {}).get(horizon_label, {})
            horizon_threshold = applied_thresholds.get("future", {}).get(horizon_label, float("nan"))
            lead_minutes = summary.get("mean_future_lead_minutes_by_horizon", {}).get(horizon_label, float("nan"))
            lines.append(
                f"- {horizon_label}: threshold={horizon_threshold:.6f}, "
                f"valid_count={int(horizon_metrics.get('valid_count', 0))}, "
                f"valid_pos={int(horizon_metrics.get('valid_positive_count', 0))}, "
                f"ignored_near_onset_pos={int(horizon_metrics.get('ignored_near_onset_positive_count', 0))}, "
                f"PR-AUC={horizon_metrics.get('pr_auc', float('nan')):.6f}, "
                f"AUC={horizon_metrics.get('auc', float('nan')):.6f}, "
                f"precision={horizon_metrics.get('precision', float('nan')):.6f}, "
                f"recall={horizon_metrics.get('recall', float('nan')):.6f}, "
                f"F1={horizon_metrics.get('f1', float('nan')):.6f}, "
                f"benign_FPR={horizon_metrics.get('false_positive_rate', float('nan')):.6f}, "
                f"Brier={format_metric_value(horizon_metrics.get('brier_score', float('nan')))}, "
                f"ECE={format_metric_value(horizon_metrics.get('ece', float('nan')))}, "
                f"mean_detected_lead={lead_minutes:.6f}"
            )
            horizon_bootstrap = future_bootstrap_by_horizon.get(horizon_label) or {}
            if horizon_bootstrap:
                lines.append(
                    f"- {horizon_label} bootstrap 95% CI: "
                    f"PR-AUC={format_bootstrap_interval(horizon_bootstrap.get('pr_auc'))}, "
                    f"recall={format_bootstrap_interval(horizon_bootstrap.get('recall'))}, "
                    f"F1={format_bootstrap_interval(horizon_bootstrap.get('f1'))}"
                )
        lines.append("")

    lines.append("## Family Acceptance")
    raw_known_accuracy = summary["known_family_accuracy"]
    if raw_known_accuracy is not None:
        lines.append(f"- Raw known-family accuracy: {raw_known_accuracy:.6f}" if raw_known_accuracy == raw_known_accuracy else "- Raw known-family accuracy: n/a")
    raw_known_macro_f1 = summary.get("known_family_macro_f1")
    if raw_known_macro_f1 is not None:
        lines.append(
            f"- Raw known-family macro F1: {raw_known_macro_f1:.6f}"
            if raw_known_macro_f1 == raw_known_macro_f1
            else "- Raw known-family macro F1: n/a"
        )
    gate_metrics = summary["known_gate_at_validation_threshold"]
    lines.append(f"- Accepted known-family accuracy: {gate_metrics['accepted_accuracy']:.6f}" if gate_metrics["accepted_accuracy"] == gate_metrics["accepted_accuracy"] else "- Accepted known-family accuracy: n/a")
    accepted_known_macro_f1 = summary.get("known_family_accepted_macro_f1")
    if accepted_known_macro_f1 is not None:
        lines.append(
            f"- Accepted known-family macro F1: {accepted_known_macro_f1:.6f}"
            if accepted_known_macro_f1 == accepted_known_macro_f1
            else "- Accepted known-family macro F1: n/a"
        )
    lines.append(f"- Known-family coverage: {gate_metrics['known_coverage']:.6f}" if gate_metrics["known_coverage"] == gate_metrics["known_coverage"] else "- Known-family coverage: n/a")
    lines.append(f"- Family-gate unknown recall: {gate_metrics['unknown_recall']:.6f}" if gate_metrics["unknown_recall"] == gate_metrics["unknown_recall"] else "- Family-gate unknown recall: n/a")
    lines.append("")

    append_family_metrics_table(
        lines,
        "### Raw Known-Family Metrics By Label",
        summary.get("known_family_metrics_by_label") or {},
        include_coverage=False,
    )
    append_family_metrics_table(
        lines,
        "### Accepted Known-Family Metrics By Label",
        summary.get("known_family_accepted_metrics_by_label") or {},
        include_coverage=True,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def write_manifest(manifest_entries, output_path):
    lines = ["# Thesis Figure Manifest", ""]
    for entry in manifest_entries:
        lines.append(f"- {entry['title']}: {entry['path']}")
        lines.append(f"  {entry['description']}")
    Path(output_path).write_text("\n".join(lines))


def main():
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir / "thesis_artifacts"
    figures_dir = ensure_dir(output_dir / "figures")
    ensure_dir(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = resolve_checkpoint_path(checkpoint_dir, args.checkpoint)
    checkpoint_bundle = load_checkpoint(checkpoint_path, device)
    checkpoint_bundle["checkpoint_path"] = checkpoint_path
    checkpoint_bundle["thresholds"] = apply_threshold_overrides(
        checkpoint_bundle["thresholds"],
        checkpoint_bundle["future_horizons_minutes"],
        current_threshold=args.current_threshold,
        known_threshold=args.known_threshold,
        future_threshold=args.future_threshold,
        ood_threshold=args.ood_threshold,
    )
    checkpoint_bundle["threshold_overrides"] = {
        "current": args.current_threshold,
        "known": args.known_threshold,
        "future": args.future_threshold,
        "ood": args.ood_threshold,
    }

    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Writing thesis artifacts to: {output_dir}")
    print(f"Active thresholds: {checkpoint_bundle['thresholds']}", flush=True)
    print(f"Run mode: {checkpoint_bundle['run_mode']}", flush=True)
    print(f"Thesis claim policy: {checkpoint_bundle['thesis_claim']}", flush=True)
    print(f"Decision policy: {checkpoint_bundle['decision_policy']}", flush=True)
    print(f"Novelty score mode: {checkpoint_bundle['novelty_score_mode']}", flush=True)
    print(f"Unknown-risk score mode: {checkpoint_bundle['unknown_risk_score_mode']}", flush=True)
    print(f"Task activation: {checkpoint_bundle['task_activation']}", flush=True)
    print(f"Explicit unknown head available: {checkpoint_bundle['use_ood_head']}", flush=True)
    print(f"Unknown-head supervision active: {checkpoint_bundle['unknown_head_active']}", flush=True)
    print(
        "Hybrid reconstruction-backed unknown risk: "
        f"{checkpoint_bundle['use_reconstruction_hybrid_ood']}",
        flush=True,
    )
    print(f"Future horizons minutes: {checkpoint_bundle['future_horizons_minutes']}", flush=True)

    manifest_entries = []
    profiled_splits = list(dict.fromkeys(["train", *args.splits]))

    history_df = load_epoch_history(checkpoint_dir, device)
    if not history_df.empty:
        history_csv = output_dir / "v3_validation_metrics_by_epoch.csv"
        history_json = output_dir / "v3_validation_metrics_by_epoch.json"
        history_df.to_csv(history_csv, index=False)
        dump_json(history_json, history_df.to_dict(orient="records"))
        plot_validation_overview(history_df, figures_dir / "v3_validation_overview.png", manifest_entries, args.dpi)
        plot_validation_thresholds_and_losses(
            history_df,
            figures_dir / "v3_validation_thresholds_and_losses.png",
            manifest_entries,
            args.dpi,
        )

    print("Building split profiles...", flush=True)
    split_profiles = {}
    for split_name in profiled_splits:
        print(f"Profiling split: {split_name}", flush=True)
        split_profiles[split_name] = build_dataset_profile(split_name, checkpoint_bundle)

    print("Rendering dataset-level figures...", flush=True)
    plot_split_sequence_distribution(
        split_profiles,
        figures_dir / "split_sequence_distribution.png",
        manifest_entries,
        args.dpi,
    )
    plot_attack_family_distribution(
        split_profiles,
        figures_dir / "split_attack_family_distribution.png",
        manifest_entries,
        args.dpi,
    )
    plot_known_unknown_distribution(
        split_profiles,
        figures_dir / "split_known_unknown_distribution.png",
        manifest_entries,
        args.dpi,
    )

    cont_features = json.loads(DEFAULT_STATS_PATH.read_text())["features"]
    corr_frame = sample_continuous_rows("train", args.correlation_sample_rows, cont_features)
    plot_feature_correlation_matrix(
        corr_frame,
        figures_dir / "train_feature_correlation_matrix.png",
        manifest_entries,
        args.dpi,
    )

    feature_shift_frames = {
        split_name: sample_continuous_rows(split_name, args.feature_shift_sample_rows, cont_features)
        for split_name in profiled_splits
    }
    plot_feature_shift(
        feature_shift_frames,
        figures_dir / "feature_mean_shift_across_splits.png",
        manifest_entries,
        args.dpi,
    )

    for split_name in args.splits:
        print(f"Starting split export: {split_name}", flush=True)
        payload = collect_split_outputs(
            split_name,
            checkpoint_bundle,
            device,
            args.batch_size,
            args.num_workers,
            profile=split_profiles.get(split_name),
            calibration_bins=args.calibration_bins,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            progress_log_interval=args.progress_log_interval,
        )
        if payload is None:
            print(f"Skipping missing split: {split_name}")
            continue

        summary = payload["summary"]
        arrays = payload["arrays"]
        applied_thresholds = summary["thresholds_applied"]

        if split_name == "test":
            summary_prefix = "final_eval_test"
        elif split_name == "test_ood":
            summary_prefix = "final_eval_test_ood"
        else:
            summary_prefix = f"eval_{split_name}"
        dump_json(output_dir / f"{summary_prefix}.json", summary)
        write_markdown_summary(summary, output_dir / f"{summary_prefix}.md")
        print(f"[{split_name}] Wrote summary artifacts: {summary_prefix}.json/.md", flush=True)
        print(f"[{split_name}] Rendering split figures...", flush=True)
        print(f"[{split_name}] Rendering present-attack figures...", flush=True)

        current_metrics = summary["current_at_validation_threshold"]
        plot_binary_curves(
            arrays["current_labels"],
            arrays["current_probabilities"],
            current_metrics["auc"],
            current_metrics["pr_auc"],
            figures_dir / f"{split_name}_current_curves.png",
            manifest_entries,
            f"{split_name} present-attack curves",
            args.dpi,
        )
        plot_threshold_sweep(
            threshold_sweep(arrays["current_labels"], arrays["current_probabilities"]),
            applied_thresholds["current"],
            figures_dir / f"{split_name}_current_threshold_sweep.png",
            manifest_entries,
            f"{split_name} present-attack threshold sweep",
            args.dpi,
        )
        plot_score_histogram(
            arrays["current_labels"],
            arrays["current_probabilities"],
            applied_thresholds["current"],
            figures_dir / f"{split_name}_current_score_histogram.png",
            manifest_entries,
            f"{split_name} present-attack score histogram",
            args.dpi,
        )
        plot_confusion_matrix(
            compute_confusion_counts(
                arrays["current_labels"],
                arrays["current_probabilities"],
                applied_thresholds["current"],
            ),
            figures_dir / f"{split_name}_current_confusion_matrix.png",
            manifest_entries,
            f"{split_name} present-attack confusion matrix",
            ["Benign", "Attack"],
            args.dpi,
        )
        plot_reliability_diagram(
            arrays["current_labels"],
            arrays["current_probabilities"],
            figures_dir / f"{split_name}_current_reliability.png",
            manifest_entries,
            f"{split_name} present-attack reliability diagram",
            args.dpi,
            bins=args.calibration_bins,
        )

        ood_metrics = summary["ood_at_validation_threshold"]
        if ood_metrics is not None:
            print(f"[{split_name}] Rendering unknown-risk figures...", flush=True)
            plot_binary_curves(
                arrays["ood_labels"],
                arrays["ood_probabilities"],
                ood_metrics["auc"],
                ood_metrics["pr_auc"],
                figures_dir / f"{split_name}_ood_curves.png",
                manifest_entries,
                f"{split_name} unknown-risk curves",
                args.dpi,
            )
            plot_threshold_sweep(
                threshold_sweep(arrays["ood_labels"], arrays["ood_probabilities"]),
                applied_thresholds["ood"],
                figures_dir / f"{split_name}_ood_threshold_sweep.png",
                manifest_entries,
                f"{split_name} unknown-risk threshold sweep",
                args.dpi,
            )
            plot_score_histogram(
                arrays["ood_labels"],
                arrays["ood_probabilities"],
                applied_thresholds["ood"],
                figures_dir / f"{split_name}_ood_score_histogram.png",
                manifest_entries,
                f"{split_name} unknown-risk score histogram",
                args.dpi,
            )
            plot_confusion_matrix(
                compute_confusion_counts(
                    arrays["ood_labels"],
                    arrays["ood_probabilities"],
                    applied_thresholds["ood"],
                ),
                figures_dir / f"{split_name}_ood_confusion_matrix.png",
                manifest_entries,
                f"{split_name} unknown-risk confusion matrix",
                ["Known or benign", "Unknown-labelled"],
                args.dpi,
            )
            plot_reliability_diagram(
                arrays["ood_labels"],
                arrays["ood_probabilities"],
                figures_dir / f"{split_name}_ood_reliability.png",
                manifest_entries,
                f"{split_name} unknown-risk reliability diagram",
                args.dpi,
                bins=args.calibration_bins,
            )

        if checkpoint_bundle["future_task_enabled"] and summary["future_at_validation_threshold"] is not None:
            print(f"[{split_name}] Rendering future-warning figures...", flush=True)
            for horizon_idx, horizon_label in enumerate(summary.get("future_horizon_labels", [])):
                horizon_metrics = summary.get("future_at_validation_threshold_by_horizon", {}).get(horizon_label)
                if horizon_metrics is None:
                    continue

                print(f"[{split_name}] Rendering future horizon {horizon_label}...", flush=True)

                horizon_threshold = applied_thresholds["future"][horizon_label]
                horizon_scores = arrays["future_probabilities"][:, horizon_idx]
                horizon_labels = arrays["future_labels"][:, horizon_idx]
                horizon_leads = arrays["future_leads"][:, horizon_idx]
                safe_horizon_label = horizon_label.replace(".", "_")
                figure_prefix = f"{split_name}_future_{safe_horizon_label}"

                plot_binary_curves(
                    horizon_labels,
                    horizon_scores,
                    horizon_metrics["auc"],
                    horizon_metrics["pr_auc"],
                    figures_dir / f"{figure_prefix}_curves.png",
                    manifest_entries,
                    f"{split_name} future-warning curves {horizon_label}",
                    args.dpi,
                )
                plot_threshold_sweep(
                    threshold_sweep(horizon_labels, horizon_scores),
                    horizon_threshold,
                    figures_dir / f"{figure_prefix}_threshold_sweep.png",
                    manifest_entries,
                    f"{split_name} future-warning threshold sweep {horizon_label}",
                    args.dpi,
                )
                plot_score_histogram(
                    horizon_labels,
                    horizon_scores,
                    horizon_threshold,
                    figures_dir / f"{figure_prefix}_score_histogram.png",
                    manifest_entries,
                    f"{split_name} future-warning score histogram {horizon_label}",
                    args.dpi,
                )
                plot_confusion_matrix(
                    compute_confusion_counts(
                        horizon_labels,
                        horizon_scores,
                        horizon_threshold,
                    ),
                    figures_dir / f"{figure_prefix}_confusion_matrix.png",
                    manifest_entries,
                    f"{split_name} future-warning confusion matrix {horizon_label}",
                    ["No future attack", "Future attack"],
                    args.dpi,
                )
                plot_reliability_diagram(
                    horizon_labels,
                    horizon_scores,
                    figures_dir / f"{figure_prefix}_reliability.png",
                    manifest_entries,
                    f"{split_name} future-warning reliability diagram {horizon_label}",
                    args.dpi,
                    bins=args.calibration_bins,
                )
                plot_future_lead_histogram(
                    horizon_leads,
                    horizon_labels,
                    horizon_scores,
                    horizon_threshold,
                    figures_dir / f"{figure_prefix}_lead_histogram.png",
                    manifest_entries,
                    f"{split_name} future lead-time histogram {horizon_label}",
                    args.dpi,
                )

        print(f"[{split_name}] Rendering family and known-gate figures...", flush=True)
        plot_known_gate_tradeoff(
            arrays["known_current_probabilities"],
            arrays["known_confidences"],
            arrays["known_predictions"],
            arrays["known_targets"],
            arrays["unknown_current_probabilities"],
            arrays["unknown_confidences"],
            applied_thresholds["current"],
            applied_thresholds["known"],
            figures_dir / f"{split_name}_known_gate_tradeoff.png",
            manifest_entries,
            f"{split_name} known-vs-unknown gate tradeoff",
            args.dpi,
        )
        plot_score_histogram(
            np.concatenate(
                [
                    np.zeros_like(arrays["known_confidences"], dtype=np.int64),
                    np.ones_like(arrays["unknown_confidences"], dtype=np.int64),
                ]
            ) if arrays["known_confidences"].size or arrays["unknown_confidences"].size else np.array([]),
            np.concatenate([arrays["known_confidences"], arrays["unknown_confidences"]])
            if arrays["known_confidences"].size or arrays["unknown_confidences"].size
            else np.array([]),
            applied_thresholds["known"],
            figures_dir / f"{split_name}_known_confidence_histogram.png",
            manifest_entries,
            f"{split_name} known-confidence histogram",
            args.dpi,
        )
        plot_family_confusion_matrix(
            arrays["known_targets"],
            arrays["known_predictions"],
            checkpoint_bundle["known_attack_labels"],
            figures_dir / f"{split_name}_family_confusion_matrix.png",
            manifest_entries,
            f"{split_name} family confusion matrix",
            args.dpi,
        )
        print(f"[{split_name}] Figure export complete.", flush=True)
        del payload
        release_memory()

    write_manifest(manifest_entries, output_dir / "thesis_figure_manifest.md")
    print(f"Generated {len(manifest_entries)} figure entries.")


if __name__ == "__main__":
    main()