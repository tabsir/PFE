import argparse
import contextlib
import json
import math
import os
import importlib.util
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
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


DEFAULT_TRAIN_DIR = "/home/aka/PFE-code/NEW/data/nids_src_grouped/train"
DEFAULT_VALID_DIR = "/home/aka/PFE-code/NEW/data/nids_src_grouped/validation"
DEFAULT_TEST_DIR  = "/home/aka/PFE-code/NEW/data/nids_src_grouped/test"
DEFAULT_STATS_PATH = "/home/aka/PFE-code/NEW/nids_normalization_stats.json"
DEFAULT_DOWNSTREAM_CHECKPOINT_DIR = "/home/aka/PFE-code/NEW/checkpoints/nids_multitask_05"
DEFAULT_FOUNDATION_CHECKPOINT     = "/home/aka/PFE-code/NEW/checkpoints/stt_best.pt"
DEFAULT_MIN_KNOWN_ATTACK_COUNT = 5
DEFAULT_TRAIN_TARGET_POSITIVE_RATE = 0.20
DEFAULT_THRESHOLD_TARGET_RECALL = 0.85
DEFAULT_UNKNOWN_FAMILY_LOSS_WEIGHT = 0.25

PROJECT_ATTACK_FAMILY_ORDER = [
    "DoS / DDoS",
    "Brute Force",
    "Botnets",
    "Infiltration",
    "Web Attacks",
    "Fuzzers",
    "Analysis / Backdoors",
    "Exploits / Shellcode",
    "Reconnaissance",
    "Worms / Generic",
]

ATTACK_FAMILY_MAP = {
    "Analysis": "Analysis / Backdoors",
    "Backdoor": "Analysis / Backdoors",
    "Bot": "Botnets",
    "Brute_Force_-Web": "Web Attacks",
    "Brute_Force_-XSS": "Web Attacks",
    "DDOS_attack-HOIC": "DoS / DDoS",
    "DDOS_attack-LOIC-UDP": "DoS / DDoS",
    "DDoS_attacks-LOIC-HTTP": "DoS / DDoS",
    "DoS": "DoS / DDoS",
    "DoS_attacks-GoldenEye": "DoS / DDoS",
    "DoS_attacks-Hulk": "DoS / DDoS",
    "DoS_attacks-SlowHTTPTest": "DoS / DDoS",
    "DoS_attacks-Slowloris": "DoS / DDoS",
    "Exploits": "Exploits / Shellcode",
    "FTP-BruteForce": "Brute Force",
    "Generic": "Worms / Generic",
    "Infilteration": "Infiltration",
    "Infiltration": "Infiltration",
    "Reconnaissance": "Reconnaissance",
    "SQL_Injection": "Web Attacks",
    "SSH-Bruteforce": "Brute Force",
    "Shellcode": "Exploits / Shellcode",
    "Worms": "Worms / Generic",
}


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
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Checkpoint file or output directory to resume from. If a directory is provided, the latest readable epoch checkpoint is used.",
    )
    parser.add_argument(
        "--min-known-attack-count",
        type=int,
        default=DEFAULT_MIN_KNOWN_ATTACK_COUNT,
        help="Minimum number of malicious sequence windows required for a mapped attack family to become a supervised known class.",
    )
    parser.add_argument(
        "--enable-future-task",
        action="store_true",
        help="Enable the future-attack prediction head and include it in training and model selection.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=32,
        help="Sequence length for downstream windows. Changing this from 32 changes the experiment and may require cache rebuilds.",
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
        help="Ignore cached sequence windows and downstream targets, then rebuild them from the current split artifacts.",
    )
    parser.add_argument(
        "--train-target-positive-rate",
        type=float,
        default=DEFAULT_TRAIN_TARGET_POSITIVE_RATE,
        help="Target positive rate seen by the training sampler. Set below 0.5 to avoid near-balanced attack oversampling.",
    )
    parser.add_argument(
        "--threshold-target-recall",
        type=float,
        default=DEFAULT_THRESHOLD_TARGET_RECALL,
        help="Recall target used to pick the current-attack operating threshold on validation.",
    )
    parser.add_argument(
        "--unknown-family-loss-weight",
        type=float,
        default=DEFAULT_UNKNOWN_FAMILY_LOSS_WEIGHT,
        help="Weight for the open-set family regularizer that keeps unknown attacks from collapsing into known-family predictions.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of downstream training epochs.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for downstream training.")
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1), help="DataLoader worker count.")
    return parser.parse_args()


def map_attack_family(attack_name):
    attack_name = str(attack_name)
    if attack_name == "Benign":
        return "Benign"

    mapped_name = ATTACK_FAMILY_MAP.get(attack_name)
    if mapped_name is not None:
        return mapped_name

    lower_name = attack_name.lower()
    if "fuzzer" in lower_name:
        return "Fuzzers"
    if "recon" in lower_name:
        return "Reconnaissance"
    if "worm" in lower_name or "generic" in lower_name:
        return "Worms / Generic"
    if "shellcode" in lower_name or "exploit" in lower_name:
        return "Exploits / Shellcode"
    if "analysis" in lower_name or "backdoor" in lower_name:
        return "Analysis / Backdoors"
    if "sql" in lower_name or "xss" in lower_name or "web" in lower_name:
        return "Web Attacks"
    if "brute" in lower_name or "ssh" in lower_name or "ftp" in lower_name:
        return "Brute Force"
    if "dos" in lower_name or "ddos" in lower_name or "hoic" in lower_name or "loic" in lower_name:
        return "DoS / DDoS"
    if "bot" in lower_name:
        return "Botnets"
    if "infil" in lower_name:
        return "Infiltration"
    return attack_name


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


def compute_binary_pr_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    positive_count = int((labels == 1).sum())
    if labels.size == 0 or positive_count == 0:
        return float("nan")

    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels == 1)
    false_positives = np.cumsum(sorted_labels == 0)

    precision = true_positives / np.maximum(true_positives + false_positives, 1)
    recall = true_positives / positive_count
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.trapezoid(precision, recall))


def compute_binary_metrics(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.size == 0:
        return {
            "auc": float("nan"),
            "pr_auc": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "false_positive_rate": float("nan"),
            "positive_rate": float("nan"),
        }

    predictions = probabilities >= threshold
    true_positive = int(np.logical_and(predictions == 1, labels == 1).sum())
    false_positive = int(np.logical_and(predictions == 1, labels == 0).sum())
    false_negative = int(np.logical_and(predictions == 0, labels == 1).sum())
    true_negative = int(np.logical_and(predictions == 0, labels == 0).sum())

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    auc = compute_binary_auroc(labels, probabilities) if np.unique(labels).size > 1 else float("nan")
    pr_auc = compute_binary_pr_auc(labels, probabilities)
    false_positive_rate = false_positive / max(false_positive + true_negative, 1)
    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(false_positive_rate),
        "positive_rate": float(labels.mean()),
    }


def build_pos_weight(binary_targets):
    binary_targets = np.asarray(binary_targets, dtype=np.int64)
    positives = int(binary_targets.sum())
    negatives = int(binary_targets.size - positives)
    if positives == 0:
        return 1.0
    return max(negatives / positives, 1.0)


def build_target_rate_sample_weights(binary_targets, target_positive_rate):
    binary_targets = np.asarray(binary_targets, dtype=np.int64)
    if binary_targets.size == 0:
        return np.array([], dtype=np.float32), 0.0, 0.0

    positives = int(binary_targets.sum())
    negatives = int(binary_targets.size - positives)
    observed_positive_rate = positives / binary_targets.size

    if positives == 0 or negatives == 0:
        weights = np.ones(binary_targets.size, dtype=np.float32)
        return weights, float(observed_positive_rate), float(observed_positive_rate)

    positive_weight = (
        target_positive_rate * negatives
        / max(positives * (1.0 - target_positive_rate), 1e-12)
    )
    weights = np.where(binary_targets == 1, positive_weight, 1.0).astype(np.float32)
    effective_positive_rate = float(weights[binary_targets == 1].sum() / max(weights.sum(), 1e-12))
    return weights, float(observed_positive_rate), effective_positive_rate


class FocalLoss(nn.Module):
    """Binary focal loss for class-imbalanced detection tasks."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * (1.0 - p_t) ** self.gamma * bce
        return loss.mean()


def compute_unknown_family_regularization_loss(logits):
    if logits.numel() == 0 or logits.shape[-1] <= 1:
        return logits.new_zeros(())

    log_probs = F.log_softmax(logits, dim=-1)
    uniform_targets = torch.full_like(log_probs, 1.0 / log_probs.shape[-1])
    return F.kl_div(log_probs, uniform_targets, reduction="batchmean")


def select_threshold_for_target_recall(labels, probabilities, target_recall):
    """Pick the most precise threshold that still satisfies the requested recall."""
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if labels.size == 0 or int(labels.sum()) == 0:
        return {
            "threshold": 0.5,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "meets_target_recall": False,
            "selection_policy": "precision_at_target_recall",
            "target_recall": float(target_recall),
        }

    candidates = np.unique(np.concatenate([
        np.linspace(0.0, 1.0, 101),
        np.percentile(probs, [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]),
        probs,
    ]))

    selected = None
    fallback = None
    for t in np.sort(candidates)[::-1]:
        preds = probs >= t
        tp = int(np.logical_and(preds, labels == 1).sum())
        fp = int(np.logical_and(preds, labels == 0).sum())
        fn = int(np.logical_and(~preds, labels == 1).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
        record = {
            "threshold": float(t),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "meets_target_recall": float(recall) >= float(target_recall),
            "selection_policy": "precision_at_target_recall",
            "target_recall": float(target_recall),
        }

        if fallback is None or (record["recall"], record["precision"], record["threshold"]) > (
            fallback["recall"],
            fallback["precision"],
            fallback["threshold"],
        ):
            fallback = record

        if not record["meets_target_recall"]:
            continue

        if selected is None or (record["precision"], record["threshold"], record["f1"]) > (
            selected["precision"],
            selected["threshold"],
            selected["f1"],
        ):
            selected = record

    return selected or fallback


class DownstreamNIDSDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        future_horizon_minutes=5,
        known_attack_to_idx=None,
        min_known_attack_count=100,
        max_sequences=None,
        current_label_rule="last_half_attack",
        rebuild_target_cache=False,
    ):
        self.base_dataset = base_dataset
        self.future_horizon_minutes = future_horizon_minutes
        self.future_horizon_ms = int(future_horizon_minutes * 60 * 1000)
        self.sequence_ranges = np.asarray(base_dataset.sequence_ranges[:max_sequences], dtype=np.int64)
        self.max_sequences = max_sequences
        self.current_label_rule = current_label_rule
        self.rebuild_target_cache = rebuild_target_cache
        self.cache_dir = os.path.dirname(base_dataset.sequence_cache_path)
        cache_suffix = (
            f"seq{base_dataset.seq_len}_stride{base_dataset.stride}_"
            f"h{future_horizon_minutes}m_{self.current_label_rule}"
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
        self.sequence_attack_families = ["Benign"] * len(self.sequence_ranges)
        self.attack_family_counts = Counter()

        for idx, raw_attack_id in enumerate(self.sequence_attack_ids):
            if self.sequence_current_labels[idx] == 0:
                continue
            attack_family = map_attack_family(self.raw_attack_names[int(raw_attack_id)])
            self.sequence_attack_families[idx] = attack_family
            self.attack_family_counts[attack_family] += 1
            if attack_family in self.known_attack_to_idx:
                self.known_attack_targets[idx] = self.known_attack_to_idx[attack_family]
            else:
                self.unknown_attack_targets[idx] = 1

    def __len__(self):
        return len(self.sequence_ranges)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        # Align the current-attack target with the downstream labeling rule.
        item.update({
            "label": torch.tensor(self.sequence_current_labels[idx], dtype=torch.long),
            "attack": self.raw_attack_names[int(self.sequence_attack_ids[idx])],
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

    def _window_has_current_attack(self, attack_offsets, window_len):
        if attack_offsets.size == 0:
            return False
        if self.current_label_rule == "any_attack":
            return True
        if self.current_label_rule == "last_half_attack":
            return bool((attack_offsets >= (window_len // 2)).any())
        raise ValueError(f"Unsupported current_label_rule: {self.current_label_rule}")

    def _load_or_build_targets(self):
        if (not self.rebuild_target_cache and os.path.exists(self.target_cache_path)
                and os.path.exists(self.target_meta_path)):
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
            json.dump(
                {
                    "raw_attack_names": raw_attack_names,
                    "current_label_rule": self.current_label_rule,
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
            raw_attack_names,
        )

    def _build_known_attack_vocab(self, min_known_attack_count):
        attack_counter = Counter()
        for raw_attack_id, current_label in zip(self.sequence_attack_ids, self.sequence_current_labels):
            if current_label == 0:
                continue
            attack_name = map_attack_family(self.raw_attack_names[int(raw_attack_id)])
            if attack_name == "Benign":
                continue
            attack_counter[attack_name] += 1

        known_attack_names = [
            attack_name
            for attack_name in PROJECT_ATTACK_FAMILY_ORDER
            if attack_counter.get(attack_name, 0) >= min_known_attack_count
        ]

        for attack_name in sorted(attack_counter):
            if attack_name in PROJECT_ATTACK_FAMILY_ORDER:
                continue
            if attack_counter[attack_name] >= min_known_attack_count:
                known_attack_names.append(attack_name)
        return {attack_name: idx for idx, attack_name in enumerate(known_attack_names)}


def set_backbone_trainable(model, trainable):
    for parameter in model.backbone.parameters():
        parameter.requires_grad = trainable


def compute_multitask_losses(
    outputs,
    batch,
    current_loss_fn,
    family_loss_fn,
    future_loss_fn,
    unknown_family_loss_fn,
    loss_weights,
):
    current_targets = batch["label"].float()
    future_targets = batch["future_attack"].float()
    known_attack_targets = batch["known_attack_id"]
    unknown_attack_targets = batch.get("unknown_attack_target")

    current_loss = current_loss_fn(outputs["current_attack_logits"], current_targets)

    future_mask = current_targets == 0
    if outputs.get("future_attack_logits") is not None and future_loss_fn is not None and future_mask.any():
        future_loss = future_loss_fn(outputs["future_attack_logits"][future_mask], future_targets[future_mask])
    else:
        future_loss = outputs["current_attack_logits"].new_zeros(())

    known_attack_mask = known_attack_targets >= 0
    if outputs["attack_family_logits"] is not None and known_attack_mask.any():
        family_loss = family_loss_fn(outputs["attack_family_logits"][known_attack_mask], known_attack_targets[known_attack_mask])
    else:
        family_loss = outputs["current_attack_logits"].new_zeros(())

    unknown_attack_mask = unknown_attack_targets > 0.5 if unknown_attack_targets is not None else None
    if (
        outputs["attack_family_logits"] is not None
        and unknown_family_loss_fn is not None
        and unknown_attack_mask is not None
        and unknown_attack_mask.any()
    ):
        unknown_loss = unknown_family_loss_fn(outputs["attack_family_logits"][unknown_attack_mask])
    else:
        unknown_loss = outputs["current_attack_logits"].new_zeros(())

    total_loss = (
        loss_weights["current"] * current_loss
        + loss_weights["family"] * family_loss
        + loss_weights["future"] * future_loss
        + loss_weights.get("unknown", 0.0) * unknown_loss
    )

    return {
        "total": total_loss,
        "current": current_loss,
        "family": family_loss,
        "future": future_loss,
        "unknown": unknown_loss,
    }


def evaluate_downstream(model, data_loader, device, thresholds):
    model.eval()
    metric_totals = {"total": 0.0, "current": 0.0, "family": 0.0, "future": 0.0, "unknown": 0.0}
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
                    "unknown_attack_target": unknown_target,
                },
                evaluate_downstream.current_loss_fn,
                evaluate_downstream.family_loss_fn,
                evaluate_downstream.future_loss_fn,
                evaluate_downstream.unknown_family_loss_fn,
                evaluate_downstream.loss_weights,
            )

            for loss_name, loss_value in batch_losses.items():
                metric_totals[loss_name] += float(loss_value.item())

            current_probability = torch.sigmoid(outputs["current_attack_logits"]).cpu().numpy()
            current_probs.append(current_probability)
            current_targets.append(current_target.cpu().numpy())

            benign_mask = current_target == 0
            if outputs.get("future_attack_logits") is not None and benign_mask.any():
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

    threshold_target_recall = getattr(
        evaluate_downstream,
        "threshold_target_recall",
        DEFAULT_THRESHOLD_TARGET_RECALL,
    )
    threshold_selection = select_threshold_for_target_recall(
        current_labels,
        current_probabilities,
        threshold_target_recall,
    )
    best_current_metrics = compute_binary_metrics(
        current_labels,
        current_probabilities,
        threshold_selection["threshold"],
    )
    best_current_metrics.update(threshold_selection)

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
        "best_current": best_current_metrics,
        "future": future_metrics,
        "known_family_accuracy": known_family_accuracy,
        "unknown_warning_recall": unknown_warning_recall,
        "mean_future_lead_minutes": mean_future_lead,
    }


def load_foundation_checkpoint(backbone, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        print(f"No foundation checkpoint found at {checkpoint_path}. Downstream training starts from scratch.")
        return

    checkpoint = load_trusted_checkpoint(checkpoint_path, device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    checkpoint_pos_encoder = state_dict.get("pos_encoder")
    if checkpoint_pos_encoder is not None and checkpoint_pos_encoder.shape != backbone.pos_encoder.shape:
        resized_pos_encoder = F.interpolate(
            checkpoint_pos_encoder.transpose(1, 2),
            size=backbone.pos_encoder.shape[1],
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        state_dict = dict(state_dict)
        state_dict["pos_encoder"] = resized_pos_encoder
        print(
            "Resized foundation positional encoder "
            f"from seq_len={checkpoint_pos_encoder.shape[1]} to seq_len={backbone.pos_encoder.shape[1]}",
            flush=True,
        )
    missing_keys, unexpected_keys = backbone.load_state_dict(state_dict, strict=False)
    print(f"Loaded foundation checkpoint: {checkpoint_path}")
    if missing_keys:
        print(f"Missing keys while loading backbone: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys while loading backbone: {unexpected_keys}")


def atomic_torch_save(payload, destination_path):
    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_path = tempfile.mkstemp(
        dir=destination_path.parent,
        prefix=f".{destination_path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination_path)
    except Exception as exc:
        with contextlib.suppress(FileNotFoundError):
            os.remove(temp_path)
        raise RuntimeError(f"Failed to save checkpoint atomically: {destination_path}") from exc


def load_trusted_checkpoint(checkpoint_path, device):
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def build_validation_rank(validation_metrics, future_task_enabled):
    best_current = validation_metrics["best_current"]
    future_metrics = validation_metrics["future"]
    return (
        float(np.nan_to_num(best_current["pr_auc"], nan=0.0)),
        float(np.nan_to_num(best_current["f1"], nan=0.0)),
        float(np.nan_to_num(best_current["recall"], nan=0.0)),
        float(np.nan_to_num(future_metrics["pr_auc"], nan=0.0)) if future_task_enabled else 0.0,
        float(np.nan_to_num(validation_metrics["known_family_accuracy"], nan=0.0)),
        float(np.nan_to_num(validation_metrics["unknown_warning_recall"], nan=0.0)),
        float(np.nan_to_num(best_current["auc"], nan=0.0)),
    )


def load_best_validation_rank(checkpoint_dir, device, fallback_rank):
    best_checkpoint_path = Path(checkpoint_dir) / "nids_multitask_best.pt"
    if not best_checkpoint_path.exists():
        return fallback_rank

    try:
        best_checkpoint = load_trusted_checkpoint(best_checkpoint_path, device)
        stored_rank = best_checkpoint.get("validation_rank")
        if stored_rank is not None:
            return tuple(float(value) for value in stored_rank)

        validation_metrics = best_checkpoint.get("validation_metrics")
        if validation_metrics is not None:
            return build_validation_rank(
                validation_metrics,
                bool(best_checkpoint.get("future_task_enabled", True)),
            )

        fallback_score = float(best_checkpoint.get("validation_score", fallback_rank[0]))
        return (fallback_score, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    except Exception as exc:
        print(f"Warning: could not load best checkpoint {best_checkpoint_path}: {exc}", flush=True)
        return fallback_rank


def ensure_checkpoint_dir_compatible(
    checkpoint_dir,
    seq_len,
    stride,
    current_label_rule,
    future_task_enabled,
    train_target_positive_rate,
    threshold_target_recall,
    unknown_family_loss_weight,
    device,
):
    best_checkpoint_path = Path(checkpoint_dir) / "nids_multitask_best.pt"
    if not best_checkpoint_path.exists():
        return

    checkpoint = load_trusted_checkpoint(best_checkpoint_path, device)
    expected_config = {
        "seq_len": seq_len,
        "stride": stride,
        "current_label_rule": current_label_rule,
        "future_task_enabled": future_task_enabled,
        "train_target_positive_rate": train_target_positive_rate,
        "threshold_target_recall": threshold_target_recall,
        "unknown_family_loss_weight": unknown_family_loss_weight,
    }
    mismatches = []
    missing_fields = []

    for field_name, expected_value in expected_config.items():
        existing_value = checkpoint.get(field_name)
        if existing_value is None:
            missing_fields.append(field_name)
            continue
        if existing_value != expected_value:
            mismatches.append(
                f"{field_name}={existing_value!r} in {best_checkpoint_path.name} vs requested {expected_value!r}"
            )

    if mismatches:
        mismatch_summary = "; ".join(mismatches)
        raise ValueError(
            "Output directory already contains downstream checkpoints for a different configuration. "
            "Use a fresh --output-dir or resume the matching run. "
            f"{mismatch_summary}"
        )

    if missing_fields:
        print(
            "Warning: existing best checkpoint is missing configuration metadata "
            f"{missing_fields}; directory compatibility could not be fully verified.",
            flush=True,
        )


def extract_epoch_index(checkpoint_path):
    try:
        return int(Path(checkpoint_path).stem.rsplit("_", 1)[-1])
    except (TypeError, ValueError):
        return -1


def resolve_resume_checkpoint(resume_value, device):
    resume_path = Path(resume_value)
    if resume_path.is_dir():
        epoch_candidates = sorted(
            resume_path.glob("nids_multitask_epoch_*.pt"),
            key=extract_epoch_index,
            reverse=True,
        )
        for epoch_path in epoch_candidates:
            try:
                checkpoint = load_trusted_checkpoint(epoch_path, device)
                return epoch_path, checkpoint
            except Exception as exc:
                print(f"Skipping unreadable resume checkpoint {epoch_path}: {exc}", flush=True)
        raise FileNotFoundError(f"No readable epoch checkpoint found in {resume_path}")

    if not resume_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    checkpoint = load_trusted_checkpoint(resume_path, device)
    return resume_path, checkpoint


def train_multitask_nids():
    args = parse_args()
    print("Starting downstream NIDS training...", flush=True)
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
    future_horizon_minutes = 5
    min_known_attack_count = args.min_known_attack_count
    future_task_enabled = args.enable_future_task
    num_workers = args.num_workers
    current_label_rule = args.current_label_rule
    rebuild_caches = args.rebuild_caches
    train_target_positive_rate = args.train_target_positive_rate
    threshold_target_recall = args.threshold_target_recall
    unknown_family_loss_weight = args.unknown_family_loss_weight

    if not 0.0 < train_target_positive_rate < 1.0:
        raise ValueError(
            f"train_target_positive_rate must be in (0, 1), got {train_target_positive_rate}"
        )
    if not 0.0 < threshold_target_recall <= 1.0:
        raise ValueError(
            f"threshold_target_recall must be in (0, 1], got {threshold_target_recall}"
        )
    if unknown_family_loss_weight < 0.0:
        raise ValueError(
            f"unknown_family_loss_weight must be >= 0, got {unknown_family_loss_weight}"
        )

    loss_weights = {
        "current": 2.0, #detect attack is the primary task, so it gets the highest weight
        "family": 0.5,
        "future": 0.75 if future_task_enabled else 0.0,
        "unknown": unknown_family_loss_weight,
    }
    thresholds = {
        "current": 0.50, #If the model’s predicted probability for an attack is ≥ 0.5, it’s considered an attack 
        "known": 0.55,
        "future": 0.50,
    }

    train_dir = DEFAULT_TRAIN_DIR
    valid_dir = DEFAULT_VALID_DIR
    TEST_DIR = DEFAULT_TEST_DIR
    stats_path = DEFAULT_STATS_PATH
    checkpoint_dir = args.output_dir
    foundation_checkpoint = args.foundation_checkpoint
    resume_checkpoint = args.resume_checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Foundation checkpoint: {foundation_checkpoint}", flush=True)
    print(f"Downstream output directory: {checkpoint_dir}", flush=True)
    print(f"Future task enabled: {future_task_enabled}", flush=True)
    print(f"Window config: seq_len={seq_len}, stride={stride}", flush=True)
    print(f"Current label rule: {current_label_rule}", flush=True)
    print(f"Train target positive rate: {train_target_positive_rate:.3f}", flush=True)
    print(f"Threshold target recall: {threshold_target_recall:.3f}", flush=True)
    print(f"Unknown family loss weight: {unknown_family_loss_weight:.3f}", flush=True)
    print(f"Rebuild caches: {rebuild_caches}", flush=True)
    ensure_checkpoint_dir_compatible(
        checkpoint_dir,
        seq_len,
        stride,
        current_label_rule,
        future_task_enabled,
        train_target_positive_rate,
        threshold_target_recall,
        unknown_family_loss_weight,
        device,
    )

    validation_path = valid_dir if os.path.exists(valid_dir) else TEST_DIR
    if validation_path == TEST_DIR:
        print("Validation split introuvable. Utilisation temporaire du split test comme validation downstream.")

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

    print("Preparing downstream train targets...", flush=True)
    train_dataset = DownstreamNIDSDataset(
        train_base_dataset,
        future_horizon_minutes=future_horizon_minutes,
        min_known_attack_count=min_known_attack_count,
        current_label_rule=current_label_rule,
        rebuild_target_cache=rebuild_caches,
    )
    print(f"Downstream train dataset ready: {len(train_dataset)} sequences", flush=True)
    print(f"Mapped attack families in train: {dict(train_dataset.attack_family_counts)}", flush=True)
    print(f"Known attack families: {train_dataset.known_attack_names}", flush=True)

    print("Preparing downstream validation targets...", flush=True)
    valid_dataset = DownstreamNIDSDataset(
        valid_base_dataset,
        future_horizon_minutes=future_horizon_minutes,
        known_attack_to_idx=train_dataset.known_attack_to_idx,
        current_label_rule=current_label_rule,
        rebuild_target_cache=rebuild_caches,
    )
    print(f"Downstream validation dataset ready: {len(valid_dataset)} sequences", flush=True)
    print(f"Mapped attack families in validation: {dict(valid_dataset.attack_family_counts)}", flush=True)

    attack_vocab_path = os.path.join(checkpoint_dir, "known_attack_labels.json")
    with open(attack_vocab_path, "w") as handle:
        json.dump({"known_attack_labels": train_dataset.known_attack_names}, handle, indent=2)

    _labels = train_dataset.sequence_current_labels.astype(np.float32)
    _sample_weights, observed_positive_rate, effective_positive_rate = build_target_rate_sample_weights(
        _labels,
        train_target_positive_rate,
    )
    _sampler = WeightedRandomSampler(
        weights=torch.from_numpy(_sample_weights),
        num_samples=len(_sample_weights),
        replacement=True,
    )
    print(
        "Train sampler configured: "
        f"observed_positive_rate={observed_positive_rate:.4f}, "
        f"target_positive_rate={train_target_positive_rate:.4f}, "
        f"effective_positive_rate={effective_positive_rate:.4f}",
        flush=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=_sampler,
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
        use_future_head=future_task_enabled,
    ).to(device)
    print(f"Downstream model ready on {device}", flush=True)

    benign_mask = train_dataset.sequence_current_labels == 0
    if future_task_enabled:
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

    positive_rate = float(train_dataset.sequence_current_labels.mean())
    focal_alpha = 1.0 - positive_rate
    current_loss_fn = FocalLoss(alpha=focal_alpha, gamma=3.0)
    future_loss_fn = nn.BCEWithLogitsLoss(pos_weight=future_pos_weight) if future_task_enabled else None
    family_loss_fn = nn.CrossEntropyLoss(weight=family_weight_tensor, label_smoothing=0.1) if family_weight_tensor is not None else None
    unknown_family_loss_fn = compute_unknown_family_regularization_loss if train_dataset.known_attack_names else None

    evaluate_downstream.current_loss_fn = current_loss_fn
    evaluate_downstream.future_loss_fn = future_loss_fn
    evaluate_downstream.family_loss_fn = family_loss_fn
    evaluate_downstream.unknown_family_loss_fn = unknown_family_loss_fn
    evaluate_downstream.loss_weights = loss_weights
    evaluate_downstream.threshold_target_recall = threshold_target_recall

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
    best_rank = load_best_validation_rank(checkpoint_dir, device, best_rank)

    if best_rank[0] > -float("inf"):
        print(f"Existing best validation PR-AUC rank: {best_rank}", flush=True)

    if resume_checkpoint:
        resume_path, resume_state = resolve_resume_checkpoint(resume_checkpoint, device)
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

        checkpoint_train_target_positive_rate = resume_state.get("train_target_positive_rate")
        if (
            checkpoint_train_target_positive_rate is not None
            and float(checkpoint_train_target_positive_rate) != float(train_target_positive_rate)
        ):
            raise ValueError(
                "Resume checkpoint sampler configuration does not match the current configuration. "
                f"Checkpoint train_target_positive_rate={checkpoint_train_target_positive_rate} | "
                f"Requested train_target_positive_rate={train_target_positive_rate}"
            )

        checkpoint_threshold_target_recall = resume_state.get("threshold_target_recall")
        if (
            checkpoint_threshold_target_recall is not None
            and float(checkpoint_threshold_target_recall) != float(threshold_target_recall)
        ):
            raise ValueError(
                "Resume checkpoint threshold configuration does not match the current configuration. "
                f"Checkpoint threshold_target_recall={checkpoint_threshold_target_recall} | "
                f"Requested threshold_target_recall={threshold_target_recall}"
            )

        checkpoint_unknown_family_loss_weight = resume_state.get("unknown_family_loss_weight")
        if (
            checkpoint_unknown_family_loss_weight is not None
            and float(checkpoint_unknown_family_loss_weight) != float(unknown_family_loss_weight)
        ):
            raise ValueError(
                "Resume checkpoint unknown-family loss configuration does not match the current configuration. "
                f"Checkpoint unknown_family_loss_weight={checkpoint_unknown_family_loss_weight} | "
                f"Requested unknown_family_loss_weight={unknown_family_loss_weight}"
            )

        model.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        scheduler.load_state_dict(resume_state["scheduler_state_dict"])

        foundation_checkpoint = resume_state.get("foundation_checkpoint", foundation_checkpoint)
        start_epoch = int(resume_state.get("epoch", -1)) + 1
        resume_metrics = resume_state.get("validation_metrics")
        if resume_metrics is not None:
            resume_rank = build_validation_rank(resume_metrics, future_task_enabled)
        else:
            resume_score = float(resume_state.get("validation_score", -float("inf")))
            resume_rank = (resume_score, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        best_rank = load_best_validation_rank(checkpoint_dir, device, resume_rank)

        print(f"Resumed downstream training from {resume_path}", flush=True)
        print(f"Next epoch: {start_epoch + 1}/{epochs}", flush=True)
        print(f"Current best validation PR-AUC rank: {best_rank}", flush=True)

        if start_epoch >= epochs:
            print("Resume checkpoint already reached the requested total epoch count. Nothing to do.", flush=True)
            return

    for epoch in range(start_epoch, epochs):
        print(f"Starting downstream epoch {epoch + 1}/{epochs}...", flush=True)
        backbone_trainable = epoch >= freeze_backbone_epochs
        set_backbone_trainable(model, backbone_trainable)
        model.train()

        running_losses = {"total": 0.0, "current": 0.0, "family": 0.0, "future": 0.0, "unknown": 0.0}
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
            losses = compute_multitask_losses(
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

            for loss_name, loss_value in losses.items():
                running_losses[loss_name] += float(loss_value.item())

            progress_bar.set_postfix({
                "loss": f"{running_losses['total'] / (step + 1):.4f}",
                "current": f"{running_losses['current'] / (step + 1):.4f}",
                "family": f"{running_losses['family'] / (step + 1):.4f}",
                "future": f"{running_losses['future'] / (step + 1):.4f}",
                "unknown": f"{running_losses['unknown'] / (step + 1):.4f}",
            })

        validation_metrics = evaluate_downstream(model, valid_loader, device, thresholds)
        best_current = validation_metrics["best_current"]
        thresholds["current"] = best_current["threshold"]
        validation_rank = build_validation_rank(validation_metrics, future_task_enabled)
        validation_score = validation_rank[0]

        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "foundation_checkpoint": foundation_checkpoint,
            "output_dir": checkpoint_dir,
            "known_attack_labels": train_dataset.known_attack_names,
            "future_horizon_minutes": future_horizon_minutes,
            "future_task_enabled": future_task_enabled,
            "seq_len": seq_len,
            "stride": stride,
            "current_label_rule": current_label_rule,
            "train_target_positive_rate": train_target_positive_rate,
            "threshold_target_recall": threshold_target_recall,
            "unknown_family_loss_weight": unknown_family_loss_weight,
            "loss_weights": loss_weights,
            "thresholds": thresholds,
            "best_threshold": thresholds["current"],
            "validation_score": validation_score,
            "validation_rank": list(validation_rank),
            "validation_metrics": validation_metrics,
        }

        epoch_checkpoint = os.path.join(checkpoint_dir, f"nids_multitask_epoch_{epoch + 1}.pt")
        atomic_torch_save(checkpoint_payload, epoch_checkpoint)

        if validation_rank > best_rank:
            best_rank = validation_rank
            best_checkpoint_path = os.path.join(checkpoint_dir, "nids_multitask_best.pt")
            atomic_torch_save(checkpoint_payload, best_checkpoint_path)

        best_c = validation_metrics["best_current"]
        print(
            f" Downstream epoch {epoch + 1} complete. "
            f"BestThresh={best_c['threshold']:.3f} → "
            f"P={best_c['precision']:.4f} R={best_c['recall']:.4f} F1={best_c['f1']:.4f} | "
            f"AUC={best_c['auc']:.4f} PRAUC={best_c['pr_auc']:.4f} | "
            f"BenignFPR={best_c['false_positive_rate']:.4f} | "
            f"TargetRecall={best_c['target_recall']:.2f} Met={best_c['meets_target_recall']} | "
            f"KnownAcc={validation_metrics['known_family_accuracy']:.4f} | "
            f"UnknownRecall={validation_metrics['unknown_warning_recall']:.4f} | "
            f"FutureAUC={validation_metrics['future']['auc']:.4f} | "
            f"ScorePRAUC={validation_score:.4f}"
        )


if __name__ == "__main__":
    train_multitask_nids()