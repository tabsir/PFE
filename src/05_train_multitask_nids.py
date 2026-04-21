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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def load_local_module(module_name, filename):
    module_path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


st_data_loader = load_local_module("st_data_loader", "02_st_data_loader.py")
stt_architecture = load_local_module("stt_architecture", "03_stt_architecture.py")

SpatioTemporalNIDSDataset = st_data_loader.SpatioTemporalNIDSDataset
SpatioTemporalTransformer = stt_architecture.SpatioTemporalTransformer
NIDSMultiTaskModel = stt_architecture.NIDSMultiTaskModel


DEFAULT_TRAIN_DIR = "/home/aka/PFE-code/data/nids_transformer_split/train"
DEFAULT_VALID_DIR = "/home/aka/PFE-code/data/nids_transformer_split/validation"
DEFAULT_TEST_DIR = "/home/aka/PFE-code/data/nids_transformer_split/test"
DEFAULT_STATS_PATH = "nids_normalization_stats.json"
DEFAULT_DOWNSTREAM_CHECKPOINT_DIR = "/home/aka/PFE-code/checkpoints/nids_multitask"
DEFAULT_FOUNDATION_CHECKPOINT = "/home/aka/PFE-code/checkpoints/stt_best.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the downstream multitask NIDS model.")
    parser.add_argument(
        "--foundation-checkpoint",
        default=DEFAULT_FOUNDATION_CHECKPOINT,
        help="Foundation checkpoint used to initialize the backbone.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_DOWNSTREAM_CHECKPOINT_DIR,
        help="Directory where downstream checkpoints and metadata will be written.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of downstream training epochs.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for downstream training.")
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1), help="DataLoader worker count.")
    return parser.parse_args()


def compute_binary_auroc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    positive_mask = labels == 1
    negative_mask = labels == 0
    n_positive = int(positive_mask.sum())
    n_negative = int(negative_mask.sum())
    if n_positive == 0 or n_negative == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    ranks = np.empty_like(sorted_scores, dtype=np.float64)

    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[start:end] = average_rank
        start = end

    positive_rank_sum = ranks[sorted_labels == 1].sum()
    auc = (positive_rank_sum - (n_positive * (n_positive + 1) / 2)) / (n_positive * n_negative)
    return float(auc)


def compute_binary_metrics(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.size == 0:
        return {
            "auc": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "positive_rate": float("nan"),
        }

    predictions = probabilities >= threshold
    true_positive = int(np.logical_and(predictions == 1, labels == 1).sum())
    false_positive = int(np.logical_and(predictions == 1, labels == 0).sum())
    false_negative = int(np.logical_and(predictions == 0, labels == 1).sum())

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    auc = compute_binary_auroc(labels, probabilities) if np.unique(labels).size > 1 else float("nan")
    return {
        "auc": auc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "positive_rate": float(labels.mean()),
    }


def build_pos_weight(binary_targets):
    binary_targets = np.asarray(binary_targets, dtype=np.int64)
    positives = int(binary_targets.sum())
    negatives = int(binary_targets.size - positives)
    if positives == 0:
        return 1.0
    return max(negatives / positives, 1.0)


class DownstreamNIDSDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        future_horizon_minutes=5,
        known_attack_to_idx=None,
        min_known_attack_count=100,
        max_sequences=None,
    ):
        self.base_dataset = base_dataset
        self.future_horizon_minutes = future_horizon_minutes
        self.future_horizon_ms = int(future_horizon_minutes * 60 * 1000)
        self.sequence_ranges = np.asarray(base_dataset.sequence_ranges[:max_sequences], dtype=np.int64)
        self.max_sequences = max_sequences
        self.cache_dir = os.path.dirname(base_dataset.sequence_cache_path)
        cache_suffix = f"seq{base_dataset.seq_len}_stride{base_dataset.stride}_h{future_horizon_minutes}m"
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
            self.raw_attack_names,
        ) = self._load_or_build_targets()

        if known_attack_to_idx is None:
            self.known_attack_to_idx = self._build_known_attack_vocab(min_known_attack_count)
        else:
            self.known_attack_to_idx = dict(known_attack_to_idx)

        self.known_attack_names = [None] * len(self.known_attack_to_idx)
        for attack_name, attack_idx in self.known_attack_to_idx.items():
            self.known_attack_names[attack_idx] = attack_name

        self.known_attack_targets = np.full(len(self.sequence_ranges), -1, dtype=np.int64)
        self.unknown_attack_targets = np.zeros(len(self.sequence_ranges), dtype=np.int64)

        for idx, raw_attack_id in enumerate(self.sequence_attack_ids):
            if self.sequence_current_labels[idx] == 0:
                continue
            attack_name = self.raw_attack_names[int(raw_attack_id)]
            if attack_name in self.known_attack_to_idx:
                self.known_attack_targets[idx] = self.known_attack_to_idx[attack_name]
            else:
                self.unknown_attack_targets[idx] = 1

    def __len__(self):
        return len(self.sequence_ranges)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        item.update({
            "known_attack_id": torch.tensor(self.known_attack_targets[idx], dtype=torch.long),
            "future_attack": torch.tensor(self.future_attack_targets[idx], dtype=torch.float32),
            "future_lead_minutes": torch.tensor(self.future_lead_minutes[idx], dtype=torch.float32),
            "unknown_attack_target": torch.tensor(self.unknown_attack_targets[idx], dtype=torch.float32),
        })
        return item

    def _load_numeric_column(self, column_name, row_limit, dtype, batch_size=1_000_000):
        values = np.empty(row_limit, dtype=dtype)
        for batch_start in range(0, row_limit, batch_size):
            batch_end = min(batch_start + batch_size, row_limit)
            batch = self.base_dataset.data[batch_start:batch_end][column_name]
            values[batch_start:batch_end] = np.asarray(batch, dtype=dtype)
        return values

    def _build_row_attack_ids(self, row_limit, row_labels, batch_size=250_000):
        row_attack_ids = np.zeros(row_limit, dtype=np.int16)
        attack_name_to_id = {"Benign": 0}

        for batch_start in range(0, row_limit, batch_size):
            batch_end = min(batch_start + batch_size, row_limit)
            attack_batch = self.base_dataset.data[batch_start:batch_end][self.base_dataset.attack_col]
            label_batch = row_labels[batch_start:batch_end]
            for offset, (label_value, attack_name) in enumerate(zip(label_batch, attack_batch)):
                if label_value == 0:
                    continue
                attack_name = str(attack_name)
                if attack_name not in attack_name_to_id:
                    attack_name_to_id[attack_name] = len(attack_name_to_id)
                row_attack_ids[batch_start + offset] = attack_name_to_id[attack_name]

        raw_attack_names = [None] * len(attack_name_to_id)
        for attack_name, attack_id in attack_name_to_id.items():
            raw_attack_names[attack_id] = attack_name

        return row_attack_ids, raw_attack_names

    def _build_future_targets(self, sequence_current_labels, sequence_start_times, sequence_end_times, sequence_group_ids):
        future_attack_targets = np.zeros_like(sequence_current_labels, dtype=np.int8)
        future_lead_minutes = np.full(sequence_current_labels.shape, -1.0, dtype=np.float32)

        if len(sequence_current_labels) == 0:
            return future_attack_targets, future_lead_minutes

        group_change_indices = np.flatnonzero(sequence_group_ids[1:] != sequence_group_ids[:-1]) + 1
        group_starts = np.concatenate([[0], group_change_indices])
        group_ends = np.concatenate([group_change_indices, [len(sequence_current_labels)]])

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
                horizon_end = np.searchsorted(
                    malicious_start_times,
                    window_end_time + self.future_horizon_ms,
                    side="right",
                )

                if horizon_end > first_future_attack:
                    future_attack_targets[group_start + local_idx] = 1
                    lead_minutes = (malicious_start_times[first_future_attack] - window_end_time) / 60_000.0
                    future_lead_minutes[group_start + local_idx] = float(lead_minutes)

        return future_attack_targets, future_lead_minutes

    def _load_or_build_targets(self):
        if os.path.exists(self.target_cache_path) and os.path.exists(self.target_meta_path):
            print(f"Loading cached downstream targets: {self.target_cache_path}", flush=True)
            cached = np.load(self.target_cache_path)
            with open(self.target_meta_path, "r") as handle:
                raw_attack_names = json.load(handle)["raw_attack_names"]
            return (
                cached["sequence_current_labels"],
                cached["sequence_attack_ids"],
                cached["sequence_start_times"],
                cached["sequence_end_times"],
                cached["sequence_group_ids"],
                cached["future_attack_targets"],
                cached["future_lead_minutes"],
                raw_attack_names,
            )

        print("Building downstream targets cache. This can take a while on the first run...", flush=True)
        row_limit = len(self.base_dataset.data)
        if len(self.sequence_ranges) > 0:
            row_limit = int(self.sequence_ranges[-1][1])

        row_labels = self._load_numeric_column(self.base_dataset.label_col, row_limit, np.int8)
        row_start_times = self._load_numeric_column(self.base_dataset.start_time_col, row_limit, np.int64)
        row_end_times = self._load_numeric_column(self.base_dataset.end_time_col, row_limit, np.int64)

        if self.base_dataset.group_col is not None:
            row_group_ids = self._load_numeric_column(self.base_dataset.group_col, row_limit, np.int64)
        else:
            row_group_ids = None

        row_attack_ids, raw_attack_names = self._build_row_attack_ids(row_limit, row_labels)

        num_sequences = len(self.sequence_ranges)
        sequence_current_labels = np.zeros(num_sequences, dtype=np.int8)
        sequence_attack_ids = np.zeros(num_sequences, dtype=np.int16)
        sequence_start_times = np.zeros(num_sequences, dtype=np.int64)
        sequence_end_times = np.zeros(num_sequences, dtype=np.int64)
        sequence_group_ids = np.zeros(num_sequences, dtype=np.int64)

        for idx, (start_idx, end_idx) in enumerate(tqdm(self.sequence_ranges, desc="Building downstream targets")):
            start_idx = int(start_idx)
            end_idx = int(end_idx)
            window_labels = row_labels[start_idx:end_idx]
            sequence_start_times[idx] = row_start_times[start_idx]
            sequence_end_times[idx] = row_end_times[end_idx - 1]
            sequence_group_ids[idx] = row_group_ids[start_idx] if row_group_ids is not None else 0

            if window_labels.max() > 0:
                sequence_current_labels[idx] = 1
                last_attack_offset = np.flatnonzero(window_labels > 0)[-1]
                sequence_attack_ids[idx] = row_attack_ids[start_idx + last_attack_offset]

        future_attack_targets, future_lead_minutes = self._build_future_targets(
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
        )
        with open(self.target_meta_path, "w") as handle:
            json.dump({"raw_attack_names": raw_attack_names}, handle, indent=2)

        print(f"Saved downstream targets cache: {self.target_cache_path}", flush=True)

        return (
            sequence_current_labels,
            sequence_attack_ids,
            sequence_start_times,
            sequence_end_times,
            sequence_group_ids,
            future_attack_targets,
            future_lead_minutes,
            raw_attack_names,
        )

    def _build_known_attack_vocab(self, min_known_attack_count):
        attack_counter = Counter()
        for raw_attack_id, current_label in zip(self.sequence_attack_ids, self.sequence_current_labels):
            if current_label == 0:
                continue
            attack_name = self.raw_attack_names[int(raw_attack_id)]
            if attack_name == "Benign":
                continue
            attack_counter[attack_name] += 1

        known_attack_names = [
            attack_name
            for attack_name, count in sorted(attack_counter.items(), key=lambda item: (-item[1], item[0]))
            if count >= min_known_attack_count
        ]
        return {attack_name: idx for idx, attack_name in enumerate(known_attack_names)}


def set_backbone_trainable(model, trainable):
    for parameter in model.backbone.parameters():
        parameter.requires_grad = trainable


def compute_multitask_losses(outputs, batch, current_loss_fn, family_loss_fn, future_loss_fn, loss_weights):
    current_targets = batch["label"].float()
    future_targets = batch["future_attack"].float()
    known_attack_targets = batch["known_attack_id"]

    current_loss = current_loss_fn(outputs["current_attack_logits"], current_targets)

    future_mask = current_targets == 0
    if future_mask.any():
        future_loss = future_loss_fn(outputs["future_attack_logits"][future_mask], future_targets[future_mask])
    else:
        future_loss = outputs["current_attack_logits"].new_zeros(())

    known_attack_mask = known_attack_targets >= 0
    if outputs["attack_family_logits"] is not None and known_attack_mask.any():
        family_loss = family_loss_fn(outputs["attack_family_logits"][known_attack_mask], known_attack_targets[known_attack_mask])
    else:
        family_loss = outputs["current_attack_logits"].new_zeros(())

    total_loss = (
        loss_weights["current"] * current_loss
        + loss_weights["family"] * family_loss
        + loss_weights["future"] * future_loss
    )

    return {
        "total": total_loss,
        "current": current_loss,
        "family": family_loss,
        "future": future_loss,
    }


def evaluate_downstream(model, data_loader, device, thresholds):
    model.eval()
    metric_totals = {"total": 0.0, "current": 0.0, "family": 0.0, "future": 0.0}
    current_probs = []
    current_targets = []
    future_probs = []
    future_targets = []
    future_leads = []
    family_predictions = []
    family_targets = []
    unknown_warning_flags = []
    unknown_targets = []

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
            batch_losses = compute_multitask_losses(
                outputs,
                {
                    "label": current_target,
                    "future_attack": future_target,
                    "known_attack_id": known_attack_target,
                },
                evaluate_downstream.current_loss_fn,
                evaluate_downstream.family_loss_fn,
                evaluate_downstream.future_loss_fn,
                evaluate_downstream.loss_weights,
            )

            for loss_name, loss_value in batch_losses.items():
                metric_totals[loss_name] += float(loss_value.item())

            current_probability = torch.sigmoid(outputs["current_attack_logits"]).cpu().numpy()
            current_probs.append(current_probability)
            current_targets.append(current_target.cpu().numpy())

            benign_mask = current_target == 0
            if benign_mask.any():
                future_probability = torch.sigmoid(outputs["future_attack_logits"][benign_mask]).cpu().numpy()
                future_probs.append(future_probability)
                future_targets.append(future_target[benign_mask].cpu().numpy())
                future_leads.append(future_lead[benign_mask].cpu().numpy())

            if outputs["attack_family_logits"] is not None:
                family_probability = torch.softmax(outputs["attack_family_logits"], dim=-1)
                family_confidence, family_prediction = family_probability.max(dim=-1)
                known_mask = known_attack_target >= 0
                if known_mask.any():
                    family_predictions.append(family_prediction[known_mask].cpu().numpy())
                    family_targets.append(known_attack_target[known_mask].cpu().numpy())

                unknown_warning = (
                    torch.sigmoid(outputs["current_attack_logits"]) >= thresholds["current"]
                ) & (family_confidence < thresholds["known"])
            else:
                unknown_warning = torch.sigmoid(outputs["current_attack_logits"]) >= thresholds["current"]

            unknown_warning_flags.append(unknown_warning.cpu().numpy())
            unknown_targets.append(unknown_target.cpu().numpy())

    for loss_name in metric_totals:
        metric_totals[loss_name] /= max(len(data_loader), 1)

    current_probabilities = np.concatenate(current_probs) if current_probs else np.array([])
    current_labels = np.concatenate(current_targets) if current_targets else np.array([])
    current_metrics = compute_binary_metrics(current_labels, current_probabilities, thresholds["current"])

    future_probabilities = np.concatenate(future_probs) if future_probs else np.array([])
    future_labels = np.concatenate(future_targets) if future_targets else np.array([])
    future_metrics = compute_binary_metrics(future_labels, future_probabilities, thresholds["future"])

    future_lead_values = np.concatenate(future_leads) if future_leads else np.array([])
    future_hits = (future_probabilities >= thresholds["future"]) & (future_labels == 1)
    mean_future_lead = float(future_lead_values[future_hits].mean()) if future_hits.any() else float("nan")

    if family_targets:
        family_target_array = np.concatenate(family_targets)
        family_prediction_array = np.concatenate(family_predictions)
        known_family_accuracy = float((family_prediction_array == family_target_array).mean())
    else:
        known_family_accuracy = float("nan")

    unknown_warning_array = np.concatenate(unknown_warning_flags) if unknown_warning_flags else np.array([])
    unknown_target_array = np.concatenate(unknown_targets) if unknown_targets else np.array([])
    unknown_mask = unknown_target_array == 1
    if unknown_mask.any():
        unknown_warning_recall = float(unknown_warning_array[unknown_mask].mean())
    else:
        unknown_warning_recall = float("nan")

    return {
        "loss": metric_totals,
        "current": current_metrics,
        "future": future_metrics,
        "known_family_accuracy": known_family_accuracy,
        "unknown_warning_recall": unknown_warning_recall,
        "mean_future_lead_minutes": mean_future_lead,
    }


def load_foundation_checkpoint(backbone, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        print(f"No foundation checkpoint found at {checkpoint_path}. Downstream training starts from scratch.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    missing_keys, unexpected_keys = backbone.load_state_dict(state_dict, strict=False)
    print(f"Loaded foundation checkpoint: {checkpoint_path}")
    if missing_keys:
        print(f"Missing keys while loading backbone: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys while loading backbone: {unexpected_keys}")


def train_multitask_nids():
    args = parse_args()
    print("Starting downstream NIDS training...", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = args.epochs
    batch_size = args.batch_size
    seq_len = 32
    clip_value = 5.0
    warmup_epochs = 2
    freeze_backbone_epochs = 2
    backbone_lr = 5e-5
    head_lr = 2e-4
    weight_decay = 1e-4
    future_horizon_minutes = 5
    min_known_attack_count = 100
    num_workers = args.num_workers

    loss_weights = {
        "current": 1.0,
        "family": 1.0,
        "future": 0.75,
    }
    thresholds = {
        "current": 0.50,
        "known": 0.55,
        "future": 0.50,
    }

    train_dir = DEFAULT_TRAIN_DIR
    valid_dir = DEFAULT_VALID_DIR
    TEST_DIR = DEFAULT_TEST_DIR
    stats_path = DEFAULT_STATS_PATH
    checkpoint_dir = args.output_dir
    foundation_checkpoint = args.foundation_checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Foundation checkpoint: {foundation_checkpoint}", flush=True)
    print(f"Downstream output directory: {checkpoint_dir}", flush=True)

    validation_path = valid_dir if os.path.exists(valid_dir) else TEST_DIR
    if validation_path == TEST_DIR:
        print("Validation split introuvable. Utilisation temporaire du split test comme validation downstream.")

    print("Loading base train dataset...", flush=True)
    train_base_dataset = SpatioTemporalNIDSDataset(
        arrow_dir_path=train_dir,
        stats_path=stats_path,
        seq_len=seq_len,
        clip_value=clip_value,
    )
    print(f"Base train dataset ready: {len(train_base_dataset)} sequences", flush=True)

    print("Loading base validation dataset...", flush=True)
    valid_base_dataset = SpatioTemporalNIDSDataset(
        arrow_dir_path=validation_path,
        stats_path=stats_path,
        seq_len=seq_len,
        clip_value=clip_value,
    )
    print(f"Base validation dataset ready: {len(valid_base_dataset)} sequences", flush=True)

    print("Preparing downstream train targets...", flush=True)
    train_dataset = DownstreamNIDSDataset(
        train_base_dataset,
        future_horizon_minutes=future_horizon_minutes,
        min_known_attack_count=min_known_attack_count,
    )
    print(f"Downstream train dataset ready: {len(train_dataset)} sequences", flush=True)

    print("Preparing downstream validation targets...", flush=True)
    valid_dataset = DownstreamNIDSDataset(
        valid_base_dataset,
        future_horizon_minutes=future_horizon_minutes,
        known_attack_to_idx=train_dataset.known_attack_to_idx,
    )
    print(f"Downstream validation dataset ready: {len(valid_dataset)} sequences", flush=True)

    attack_vocab_path = os.path.join(checkpoint_dir, "known_attack_labels.json")
    with open(attack_vocab_path, "w") as handle:
        json.dump({"known_attack_labels": train_dataset.known_attack_names}, handle, indent=2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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
    cat_vocabs = [256, 256, 256, 65536, 65536, 256, 256, 256, 256]

    backbone = SpatioTemporalTransformer(
        num_cont_features=num_cont,
        cat_vocab_sizes=cat_vocabs,
        seq_len=seq_len,
        init_mae=0.10,
        init_mfm=0.00,
    )
    print(f"Loading foundation weights from {foundation_checkpoint}", flush=True)
    load_foundation_checkpoint(backbone, foundation_checkpoint, device)
    model = NIDSMultiTaskModel(
        backbone=backbone,
        num_known_attack_classes=len(train_dataset.known_attack_names),
    ).to(device)
    print(f"Downstream model ready on {device}", flush=True)

    current_pos_weight = torch.tensor(
        build_pos_weight(train_dataset.sequence_current_labels),
        dtype=torch.float32,
        device=device,
    )
    benign_mask = train_dataset.sequence_current_labels == 0
    future_pos_weight = torch.tensor(
        build_pos_weight(train_dataset.future_attack_targets[benign_mask]),
        dtype=torch.float32,
        device=device,
    )

    if train_dataset.known_attack_names:
        attack_counter = Counter(train_dataset.known_attack_targets[train_dataset.known_attack_targets >= 0].tolist())
        family_weights = np.ones(len(train_dataset.known_attack_names), dtype=np.float32)
        total_known = sum(attack_counter.values())
        for attack_idx, attack_count in attack_counter.items():
            family_weights[attack_idx] = total_known / max(len(train_dataset.known_attack_names) * attack_count, 1)
        family_weight_tensor = torch.tensor(family_weights, dtype=torch.float32, device=device)
    else:
        family_weight_tensor = None

    current_loss_fn = nn.BCEWithLogitsLoss(pos_weight=current_pos_weight)
    future_loss_fn = nn.BCEWithLogitsLoss(pos_weight=future_pos_weight)
    family_loss_fn = nn.CrossEntropyLoss(weight=family_weight_tensor) if family_weight_tensor is not None else None

    evaluate_downstream.current_loss_fn = current_loss_fn
    evaluate_downstream.future_loss_fn = future_loss_fn
    evaluate_downstream.family_loss_fn = family_loss_fn
    evaluate_downstream.loss_weights = loss_weights

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
    best_score = -float("inf")

    for epoch in range(epochs):
        print(f"Starting downstream epoch {epoch + 1}/{epochs}...", flush=True)
        backbone_trainable = epoch >= freeze_backbone_epochs
        set_backbone_trainable(model, backbone_trainable)
        model.train()

        running_losses = {"total": 0.0, "current": 0.0, "family": 0.0, "future": 0.0}
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

            outputs = model(cont, cat, apply_mfm=False)
            losses = compute_multitask_losses(
                outputs,
                {
                    "label": label,
                    "future_attack": future_attack,
                    "known_attack_id": known_attack_id,
                },
                current_loss_fn,
                family_loss_fn,
                future_loss_fn,
                loss_weights,
            )

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            for loss_name, loss_value in losses.items():
                running_losses[loss_name] += float(loss_value.item())

            progress_bar.set_postfix({
                "loss": f"{running_losses['total'] / (step + 1):.4f}",
                "current": f"{running_losses['current'] / (step + 1):.4f}",
                "family": f"{running_losses['family'] / (step + 1):.4f}",
                "future": f"{running_losses['future'] / (step + 1):.4f}",
            })

        validation_metrics = evaluate_downstream(model, valid_loader, device, thresholds)
        validation_score = np.nan_to_num(validation_metrics["current"]["auc"], nan=0.0)
        validation_score += np.nan_to_num(validation_metrics["future"]["auc"], nan=0.0)
        validation_score += np.nan_to_num(validation_metrics["known_family_accuracy"], nan=0.0)
        validation_score += np.nan_to_num(validation_metrics["unknown_warning_recall"], nan=0.0)

        epoch_checkpoint = os.path.join(checkpoint_dir, f"nids_multitask_epoch_{epoch + 1}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "foundation_checkpoint": foundation_checkpoint,
            "output_dir": checkpoint_dir,
            "known_attack_labels": train_dataset.known_attack_names,
            "future_horizon_minutes": future_horizon_minutes,
            "thresholds": thresholds,
            "validation_score": validation_score,
            "validation_metrics": validation_metrics,
        }, epoch_checkpoint)

        if validation_score > best_score:
            best_score = validation_score
            best_checkpoint_path = os.path.join(checkpoint_dir, "nids_multitask_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "foundation_checkpoint": foundation_checkpoint,
                "output_dir": checkpoint_dir,
                "known_attack_labels": train_dataset.known_attack_names,
                "future_horizon_minutes": future_horizon_minutes,
                "thresholds": thresholds,
                "validation_score": validation_score,
                "validation_metrics": validation_metrics,
            }, best_checkpoint_path)

        print(
            f" Downstream epoch {epoch + 1} complete. "
            f"CurrentAUC: {validation_metrics['current']['auc']:.4f} | "
            f"CurrentF1: {validation_metrics['current']['f1']:.4f} | "
            f"KnownAcc: {validation_metrics['known_family_accuracy']:.4f} | "
            f"UnknownRecall: {validation_metrics['unknown_warning_recall']:.4f} | "
            f"FutureAUC: {validation_metrics['future']['auc']:.4f} | "
            f"Lead(min): {validation_metrics['mean_future_lead_minutes']:.2f}"
        )


if __name__ == "__main__":
    train_multitask_nids()