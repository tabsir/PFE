import argparse
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
DEFAULT_CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints" / "nids_multitask_05_v2"
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
stt_architecture = load_module_from_path("stt_architecture", SRC_DIR / "03_stt_architecture.py")
v2_train = load_module_from_path("train_multitask_nids_v2", EXPERIMENT_SRC_DIR / "05_train_multitask_nids_v2.py")

SpatioTemporalNIDSDataset = st_data_loader.SpatioTemporalNIDSDataset
SpatioTemporalTransformer = stt_architecture.SpatioTemporalTransformer
NIDSMultiTaskModel = stt_architecture.NIDSMultiTaskModel
DownstreamNIDSDataset = base_train.DownstreamNIDSDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate thesis-ready v2 evaluation artifacts and figures."
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="Checkpoint directory containing v2 epoch checkpoints and the best checkpoint.",
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
        help="Dataset splits to evaluate and plot.",
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
        help="Optional override for the future-warning threshold stored in the checkpoint.",
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
    return parser.parse_args()


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    thresholds = checkpoint.get("thresholds", {"current": 0.5, "known": 0.55, "future": 0.5})
    known_attack_labels = checkpoint.get("known_attack_labels", [])
    pseudo_zero_day_families = checkpoint.get("pseudo_zero_day_families", [])
    future_horizon_minutes = int(checkpoint.get("future_horizon_minutes", 5))
    future_task_enabled = bool(checkpoint.get("future_task_enabled", True))
    seq_len = int(checkpoint.get("seq_len", 32))
    stride = int(checkpoint.get("stride", 16))
    return {
        "checkpoint": checkpoint,
        "thresholds": dict(thresholds),
        "validation_thresholds": dict(thresholds),
        "known_attack_labels": known_attack_labels,
        "pseudo_zero_day_families": pseudo_zero_day_families,
        "future_horizon_minutes": future_horizon_minutes,
        "future_task_enabled": future_task_enabled,
        "seq_len": seq_len,
        "stride": stride,
        "threshold_target_recall": float(
            checkpoint.get("threshold_target_recall", base_train.DEFAULT_THRESHOLD_TARGET_RECALL)
        ),
        "future_threshold_target_recall": float(
            checkpoint.get(
                "future_threshold_target_recall",
                v2_train.DEFAULT_FUTURE_THRESHOLD_TARGET_RECALL,
            )
        ),
        "known_target_unknown_recall": float(
            checkpoint.get(
                "known_target_unknown_recall",
                v2_train.DEFAULT_KNOWN_TARGET_UNKNOWN_RECALL,
            )
        ),
    }


def apply_threshold_overrides(thresholds, current_threshold=None, known_threshold=None, future_threshold=None):
    applied_thresholds = dict(thresholds)
    if current_threshold is not None:
        applied_thresholds["current"] = float(current_threshold)
    if known_threshold is not None:
        applied_thresholds["known"] = float(known_threshold)
    if future_threshold is not None:
        applied_thresholds["future"] = float(future_threshold)
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
            init_mae=0.10,
            init_mfm=0.00,
        ),
        num_known_attack_classes=len(checkpoint_bundle["known_attack_labels"]),
        use_future_head=future_task_enabled,
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
        future_horizon_minutes=checkpoint_bundle["future_horizon_minutes"],
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
    future_positive_count = int(dataset.future_attack_targets.sum())

    return {
        "split": split_name,
        "sequence_count": int(len(dataset)),
        "current_positive_count": int(dataset.sequence_current_labels.sum()),
        "current_positive_rate": float(dataset.sequence_current_labels.mean()) if len(dataset) else 0.0,
        "unknown_positive_count": unknown_positive_count,
        "known_family_count": known_family_count,
        "future_positive_count": future_positive_count,
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
        threshold_candidates = np.unique(
            np.concatenate([
                np.linspace(0.0, 1.0, 101),
                np.percentile(scores, [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]),
                scores,
            ])
        )

    rows = []
    for threshold_value in np.sort(threshold_candidates):
        metrics = base_train.compute_binary_metrics(labels, scores, float(threshold_value))
        rows.append(
            {
                "threshold": float(threshold_value),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "false_positive_rate": metrics["false_positive_rate"],
            }
        )
    return pd.DataFrame(rows)


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


def collect_split_outputs(split_name, checkpoint_bundle, device, batch_size, num_workers, profile=None):
    print(f"[{split_name}] Building evaluation dataset...", flush=True)
    dataset = build_downstream_dataset(split_name, checkpoint_bundle)
    if dataset is None:
        return None

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
    known_confidences = []
    known_current_probs = []
    known_predictions = []
    known_targets = []
    unknown_confidences = []
    unknown_current_probs = []

    with torch.no_grad():
        for batch in tqdm(
            data_loader,
            total=len(data_loader),
            desc=f"Eval {split_name}",
            leave=False,
        ):
            cont = batch["continuous"].to(device, non_blocking=True)
            cat = batch["categorical"].to(device, non_blocking=True)
            outputs = model(cont, cat, apply_mfm=False)

            current_probability = torch.sigmoid(outputs["current_attack_logits"]).cpu().numpy()
            current_target = batch["label"].cpu().numpy()
            current_probs.append(current_probability)
            current_labels.append(current_target)

            if checkpoint_bundle["future_task_enabled"] and outputs.get("future_attack_logits") is not None:
                benign_mask = batch["label"] == 0
                if benign_mask.any():
                    future_probability = torch.sigmoid(outputs["future_attack_logits"][benign_mask]).cpu().numpy()
                    future_probs.append(future_probability)
                    future_labels.append(batch["future_attack"][benign_mask].cpu().numpy())
                    future_leads.append(batch["future_lead_minutes"][benign_mask].cpu().numpy())

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

                unknown_mask = unknown_target == 1
                if unknown_mask.any():
                    unknown_confidences.append(family_confidence[unknown_mask])
                    unknown_current_probs.append(current_probability[unknown_mask])

    arrays = {
        "current_probabilities": np.concatenate(current_probs) if current_probs else np.array([]),
        "current_labels": np.concatenate(current_labels) if current_labels else np.array([]),
        "future_probabilities": np.concatenate(future_probs) if future_probs else np.array([]),
        "future_labels": np.concatenate(future_labels) if future_labels else np.array([]),
        "future_leads": np.concatenate(future_leads) if future_leads else np.array([]),
        "known_confidences": np.concatenate(known_confidences) if known_confidences else np.array([]),
        "known_current_probabilities": np.concatenate(known_current_probs) if known_current_probs else np.array([]),
        "known_predictions": np.concatenate(known_predictions) if known_predictions else np.array([]),
        "known_targets": np.concatenate(known_targets) if known_targets else np.array([]),
        "unknown_confidences": np.concatenate(unknown_confidences) if unknown_confidences else np.array([]),
        "unknown_current_probabilities": np.concatenate(unknown_current_probs) if unknown_current_probs else np.array([]),
    }

    if profile is None:
        profile = build_dataset_profile(split_name, checkpoint_bundle)
    thresholds = checkpoint_bundle["thresholds"]
    current_at_validation = base_train.compute_binary_metrics(
        arrays["current_labels"],
        arrays["current_probabilities"],
        thresholds["current"],
    )
    current_oracle = base_train.select_threshold_for_target_recall(
        arrays["current_labels"],
        arrays["current_probabilities"],
        checkpoint_bundle["threshold_target_recall"],
    )

    if checkpoint_bundle["future_task_enabled"]:
        future_at_validation = base_train.compute_binary_metrics(
            arrays["future_labels"],
            arrays["future_probabilities"],
            thresholds["future"],
        )
        future_oracle = base_train.select_threshold_for_target_recall(
            arrays["future_labels"],
            arrays["future_probabilities"],
            checkpoint_bundle["future_threshold_target_recall"],
        )
        future_hits = (
            (arrays["future_probabilities"] >= thresholds["future"]) & (arrays["future_labels"] == 1)
        )
        mean_future_lead_minutes = (
            float(arrays["future_leads"][future_hits].mean()) if future_hits.any() else float("nan")
        )
    else:
        future_at_validation = None
        future_oracle = None
        mean_future_lead_minutes = float("nan")

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
    known_oracle = v2_train.select_known_threshold(
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
        "threshold_target_recall": checkpoint_bundle["threshold_target_recall"],
        "future_threshold_target_recall": checkpoint_bundle["future_threshold_target_recall"],
        "known_target_unknown_recall": checkpoint_bundle["known_target_unknown_recall"],
        **profile,
        "current_at_validation_threshold": current_at_validation,
        "current_oracle_for_this_split_diagnostic_only": current_oracle,
        "future_at_validation_threshold": future_at_validation,
        "future_oracle_for_this_split_diagnostic_only": future_oracle,
        "known_family_accuracy": raw_known_accuracy,
        "known_gate_at_validation_threshold": known_gate_at_validation,
        "known_gate_oracle_for_this_split_diagnostic_only": known_oracle,
        "unknown_warning_recall": known_gate_at_validation["unknown_recall"],
        "mean_future_lead_minutes": mean_future_lead_minutes,
    }
    print(f"[{split_name}] Evaluation complete.", flush=True)
    return {"summary": summary, "arrays": arrays, "profile": profile}


def load_epoch_history(checkpoint_dir, device):
    checkpoint_dir = Path(checkpoint_dir)
    epoch_files = sorted(
        checkpoint_dir.glob("nids_multitask_epoch_*.pt"),
        key=base_train.extract_epoch_index,
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
        future_metrics = validation_metrics.get("future", {})
        best_future = validation_metrics.get("best_future", {})
        best_known = validation_metrics.get("best_known", {})
        thresholds = checkpoint.get("thresholds", {})
        loss_metrics = validation_metrics.get("loss", {})

        epoch_index = int(checkpoint.get("epoch", base_train.extract_epoch_index(epoch_path)))
        rows.append(
            {
                "checkpoint_file": epoch_path.name,
                "epoch_index_zero_based": epoch_index,
                "epoch_number": epoch_index + 1,
                "validation_score": float(checkpoint.get("validation_score", float("nan"))),
                "current_threshold": float(thresholds.get("current", float("nan"))),
                "known_threshold": float(thresholds.get("known", float("nan"))),
                "future_threshold": float(thresholds.get("future", float("nan"))),
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
                "known_family_accuracy": float(validation_metrics.get("known_family_accuracy", float("nan"))),
                "known_family_accepted_accuracy": float(validation_metrics.get("known_family_accepted_accuracy", float("nan"))),
                "known_family_coverage": float(validation_metrics.get("known_family_coverage", float("nan"))),
                "known_balanced_score": float(best_known.get("balanced_score", float("nan"))),
                "unknown_warning_recall": float(validation_metrics.get("unknown_warning_recall", float("nan"))),
                "mean_future_lead_minutes": float(validation_metrics.get("mean_future_lead_minutes", float("nan"))),
                "loss_total": float(loss_metrics.get("total", float("nan"))),
                "loss_current": float(loss_metrics.get("current", float("nan"))),
                "loss_family": float(loss_metrics.get("family", float("nan"))),
                "loss_future": float(loss_metrics.get("future", float("nan"))),
                "loss_unknown": float(loss_metrics.get("unknown", float("nan"))),
            }
        )
    return pd.DataFrame(rows).sort_values("epoch_number") if rows else pd.DataFrame()


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

    axes[0, 0].plot(x, history_df["validation_score"], marker="o", label="Validation score")
    axes[0, 0].plot(x, history_df["best_current_pr_auc"], marker="o", label="Current PR-AUC")
    axes[0, 0].plot(x, history_df["best_future_pr_auc"], marker="o", label="Future PR-AUC")
    axes[0, 0].set_title("Validation Score And PR-AUC")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(x, history_df["best_current_auc"], marker="o", label="Current AUC")
    axes[0, 1].plot(x, history_df["best_future_auc"], marker="o", label="Future AUC")
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
    axes[1, 1].plot(x, history_df["unknown_warning_recall"], marker="o", label="Unknown recall")
    axes[1, 1].set_title("Family And Unknown Metrics")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    save_figure(
        fig,
        output_path,
        manifest_entries,
        "Validation overview",
        "Training-time validation score, current/future PR-AUC, present detection metrics, and family/open-set metrics across epochs.",
        dpi,
    )


def plot_validation_thresholds_and_losses(history_df, output_path, manifest_entries, dpi):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    x = history_df["epoch_number"]

    axes[0, 0].plot(x, history_df["current_threshold"], marker="o", label="Current")
    axes[0, 0].plot(x, history_df["known_threshold"], marker="o", label="Known")
    axes[0, 0].plot(x, history_df["future_threshold"], marker="o", label="Future")
    axes[0, 0].set_title("Threshold Trajectories")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(x, history_df["loss_total"], marker="o", label="Total")
    axes[0, 1].plot(x, history_df["loss_current"], marker="o", label="Current")
    axes[0, 1].plot(x, history_df["loss_family"], marker="o", label="Family")
    axes[0, 1].plot(x, history_df["loss_future"], marker="o", label="Future")
    axes[0, 1].plot(x, history_df["loss_unknown"], marker="o", label="Unknown")
    axes[0, 1].set_title("Validation Loss Components")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(x, history_df["known_family_coverage"], marker="o", label="Known coverage")
    axes[1, 0].plot(x, history_df["known_balanced_score"], marker="o", label="Known balanced score")
    axes[1, 0].plot(x, history_df["mean_future_lead_minutes"], marker="o", label="Mean future lead")
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
        "Current/known/future threshold trajectories, validation losses, open-set coverage, and future-warning metrics across epochs.",
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


def plot_confusion_matrix(matrix, output_path, manifest_entries, title, class_names, dpi):
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(class_names)), labels=class_names)
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(col_idx, row_idx, int(matrix[row_idx, col_idx]), ha="center", va="center", color="black")

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
    candidates = v2_train.build_confidence_candidates(known_confidences, unknown_confidences)
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
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            if matrix[row_idx, col_idx] == 0:
                continue
            ax.text(col_idx, row_idx, int(matrix[row_idx, col_idx]), ha="center", va="center", color="black", fontsize=8)

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

    save_figure(fig, output_path, manifest_entries, "Split distribution overview", "Sequence counts and attack positive rates across train/validation/test/test_ood.", dpi)


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

    save_figure(fig, output_path, manifest_entries, "Feature mean shift", "Mean value comparison for the most active continuous features across train/validation/test/test_ood.", dpi)


def write_markdown_summary(summary, output_path):
    lines = [f"# Evaluation Summary: {summary['split']}", ""]
    lines.append(f"- Checkpoint: {summary['checkpoint']}")
    lines.append(f"- Pseudo-zero-day families: {summary['pseudo_zero_day_families']}")
    applied_thresholds = summary.get("thresholds_applied", summary["thresholds_from_validation"])
    lines.append(f"- Current threshold: {applied_thresholds['current']:.6f}")
    lines.append(f"- Known threshold: {applied_thresholds['known']:.6f}")
    lines.append(f"- Future threshold: {applied_thresholds['future']:.6f}")
    threshold_overrides = summary.get("threshold_overrides") or {}
    if any(value is not None for value in threshold_overrides.values()):
        lines.append(f"- Validation thresholds stored in checkpoint: {summary['thresholds_from_validation']}")
        lines.append(f"- CLI threshold overrides: {threshold_overrides}")
    lines.append("")

    current_metrics = summary["current_at_validation_threshold"]
    lines.append("## Present Detection")
    lines.append(f"- PR-AUC: {current_metrics['pr_auc']:.6f}" if current_metrics["pr_auc"] == current_metrics["pr_auc"] else "- PR-AUC: n/a")
    lines.append(f"- AUC: {current_metrics['auc']:.6f}" if current_metrics["auc"] == current_metrics["auc"] else "- AUC: n/a")
    lines.append(f"- Precision: {current_metrics['precision']:.6f}")
    lines.append(f"- Recall: {current_metrics['recall']:.6f}")
    lines.append(f"- F1: {current_metrics['f1']:.6f}")
    lines.append(f"- Benign FPR: {current_metrics['false_positive_rate']:.6f}")
    lines.append("")

    if summary["future_at_validation_threshold"] is not None:
        future_metrics = summary["future_at_validation_threshold"]
        lines.append("## Future Warning")
        lines.append(f"- PR-AUC: {future_metrics['pr_auc']:.6f}" if future_metrics["pr_auc"] == future_metrics["pr_auc"] else "- PR-AUC: n/a")
        lines.append(f"- AUC: {future_metrics['auc']:.6f}" if future_metrics["auc"] == future_metrics["auc"] else "- AUC: n/a")
        lines.append(f"- Precision: {future_metrics['precision']:.6f}")
        lines.append(f"- Recall: {future_metrics['recall']:.6f}")
        lines.append(f"- F1: {future_metrics['f1']:.6f}")
        lines.append(f"- Benign FPR: {future_metrics['false_positive_rate']:.6f}")
        lines.append("")

    lines.append("## Family And Open-Set")
    raw_known_accuracy = summary["known_family_accuracy"]
    if raw_known_accuracy is not None:
        lines.append(f"- Raw known-family accuracy: {raw_known_accuracy:.6f}" if raw_known_accuracy == raw_known_accuracy else "- Raw known-family accuracy: n/a")
    gate_metrics = summary["known_gate_at_validation_threshold"]
    lines.append(f"- Accepted known-family accuracy: {gate_metrics['accepted_accuracy']:.6f}" if gate_metrics["accepted_accuracy"] == gate_metrics["accepted_accuracy"] else "- Accepted known-family accuracy: n/a")
    lines.append(f"- Known-family coverage: {gate_metrics['known_coverage']:.6f}" if gate_metrics["known_coverage"] == gate_metrics["known_coverage"] else "- Known-family coverage: n/a")
    lines.append(f"- Unknown-warning recall: {gate_metrics['unknown_recall']:.6f}" if gate_metrics["unknown_recall"] == gate_metrics["unknown_recall"] else "- Unknown-warning recall: n/a")

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
        current_threshold=args.current_threshold,
        known_threshold=args.known_threshold,
        future_threshold=args.future_threshold,
    )
    checkpoint_bundle["threshold_overrides"] = {
        "current": args.current_threshold,
        "known": args.known_threshold,
        "future": args.future_threshold,
    }

    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Writing thesis artifacts to: {output_dir}")
    print(f"Active thresholds: {checkpoint_bundle['thresholds']}", flush=True)

    manifest_entries = []

    history_df = load_epoch_history(checkpoint_dir, device)
    if not history_df.empty:
        history_csv = output_dir / "v2_validation_metrics_by_epoch.csv"
        history_json = output_dir / "v2_validation_metrics_by_epoch.json"
        history_df.to_csv(history_csv, index=False)
        dump_json(history_json, history_df.to_dict(orient="records"))
        plot_validation_overview(history_df, figures_dir / "v2_validation_overview.png", manifest_entries, args.dpi)
        plot_validation_thresholds_and_losses(
            history_df,
            figures_dir / "v2_validation_thresholds_and_losses.png",
            manifest_entries,
            args.dpi,
        )

    print("Building split profiles...", flush=True)
    split_profiles = {}
    for split_name in ["train", *args.splits]:
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
        for split_name in ["train", *args.splits]
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
        )
        if payload is None:
            print(f"Skipping missing split: {split_name}")
            continue

        summary = payload["summary"]
        arrays = payload["arrays"]

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
            summary["thresholds_from_validation"]["current"],
            figures_dir / f"{split_name}_current_threshold_sweep.png",
            manifest_entries,
            f"{split_name} present-attack threshold sweep",
            args.dpi,
        )
        plot_score_histogram(
            arrays["current_labels"],
            arrays["current_probabilities"],
            summary["thresholds_from_validation"]["current"],
            figures_dir / f"{split_name}_current_score_histogram.png",
            manifest_entries,
            f"{split_name} present-attack score histogram",
            args.dpi,
        )
        plot_confusion_matrix(
            compute_confusion_counts(
                arrays["current_labels"],
                arrays["current_probabilities"],
                summary["thresholds_from_validation"]["current"],
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
        )

        if checkpoint_bundle["future_task_enabled"] and summary["future_at_validation_threshold"] is not None:
            future_metrics = summary["future_at_validation_threshold"]
            plot_binary_curves(
                arrays["future_labels"],
                arrays["future_probabilities"],
                future_metrics["auc"],
                future_metrics["pr_auc"],
                figures_dir / f"{split_name}_future_curves.png",
                manifest_entries,
                f"{split_name} future-warning curves",
                args.dpi,
            )
            plot_threshold_sweep(
                threshold_sweep(arrays["future_labels"], arrays["future_probabilities"]),
                summary["thresholds_from_validation"]["future"],
                figures_dir / f"{split_name}_future_threshold_sweep.png",
                manifest_entries,
                f"{split_name} future-warning threshold sweep",
                args.dpi,
            )
            plot_score_histogram(
                arrays["future_labels"],
                arrays["future_probabilities"],
                summary["thresholds_from_validation"]["future"],
                figures_dir / f"{split_name}_future_score_histogram.png",
                manifest_entries,
                f"{split_name} future-warning score histogram",
                args.dpi,
            )
            plot_confusion_matrix(
                compute_confusion_counts(
                    arrays["future_labels"],
                    arrays["future_probabilities"],
                    summary["thresholds_from_validation"]["future"],
                ),
                figures_dir / f"{split_name}_future_confusion_matrix.png",
                manifest_entries,
                f"{split_name} future-warning confusion matrix",
                ["No future attack", "Future attack"],
                args.dpi,
            )
            plot_reliability_diagram(
                arrays["future_labels"],
                arrays["future_probabilities"],
                figures_dir / f"{split_name}_future_reliability.png",
                manifest_entries,
                f"{split_name} future-warning reliability diagram",
                args.dpi,
            )
            plot_future_lead_histogram(
                arrays["future_leads"],
                arrays["future_labels"],
                arrays["future_probabilities"],
                summary["thresholds_from_validation"]["future"],
                figures_dir / f"{split_name}_future_lead_histogram.png",
                manifest_entries,
                f"{split_name} future lead-time histogram",
                args.dpi,
            )

        plot_known_gate_tradeoff(
            arrays["known_current_probabilities"],
            arrays["known_confidences"],
            arrays["known_predictions"],
            arrays["known_targets"],
            arrays["unknown_current_probabilities"],
            arrays["unknown_confidences"],
            summary["thresholds_from_validation"]["current"],
            summary["thresholds_from_validation"]["known"],
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
            summary["thresholds_from_validation"]["known"],
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

    write_manifest(manifest_entries, output_dir / "thesis_figure_manifest.md")
    print(f"Generated {len(manifest_entries)} figure entries.")


if __name__ == "__main__":
    main()