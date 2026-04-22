import argparse
import datetime as dt
import importlib.util
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def load_local_module(module_name, filename):
    module_path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


st_data_loader = load_local_module("st_data_loader", "02_st_data_loader.py")
stt_architecture = load_local_module("stt_architecture", "03_stt_architecture.py")
downstream_module = load_local_module("downstream_module", "05_train_multitask_nids.py")

SpatioTemporalNIDSDataset = st_data_loader.SpatioTemporalNIDSDataset
SpatioTemporalTransformer = stt_architecture.SpatioTemporalTransformer
NIDSMultiTaskModel = stt_architecture.NIDSMultiTaskModel
DownstreamNIDSDataset = downstream_module.DownstreamNIDSDataset


CAT_VOCABS = [256, 256, 256, 65536, 65536, 256, 256, 256, 256]
DEFAULT_CHECKPOINT = "/home/aka/PFE-code/checkpoints/nids_multitask/nids_multitask_best.pt"
DEFAULT_STATS = "/home/aka/PFE-code/nids_normalization_stats.json"
DEFAULT_DATA_ROOT = "/home/aka/PFE-code/data/nids_transformer_split"


def parse_args():
    parser = argparse.ArgumentParser(description="Run downstream NIDS inference and print human-readable alerts.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Path to the downstream multitask checkpoint.")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"], help="Dataset split to score.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for inference.") #see if i make it 512
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length used for the dataset.")
    parser.add_argument("--clip-value", type=float, default=5.0, help="Continuous feature clamp value.")
    parser.add_argument("--future-horizon-minutes", type=int, default=None, help="Override the future warning horizon stored in the checkpoint.")
    parser.add_argument("--max-sequences", type=int, default=32, help="Maximum number of sequences to print.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count for inference.")
    return parser.parse_args()


def resolve_split_path(split_name):
    requested_path = os.path.join(DEFAULT_DATA_ROOT, split_name)
    if os.path.exists(requested_path):
        return requested_path, split_name

    fallback_path = os.path.join(DEFAULT_DATA_ROOT, "test")
    print(f"Requested split '{split_name}' is not available. Falling back to 'test'.")
    return fallback_path, "test"


def format_timestamp(timestamp_ms):
    timestamp_ms = int(timestamp_ms)
    return dt.datetime.utcfromtimestamp(timestamp_ms / 1000.0).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_checkpoint(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    thresholds = checkpoint.get(
        "thresholds",
        {"current": 0.50, "known": 0.55, "future": 0.50},
    )
    known_attack_labels = checkpoint.get("known_attack_labels", [])
    future_horizon_minutes = checkpoint.get("future_horizon_minutes", 5)
    future_task_enabled = bool(checkpoint.get("future_task_enabled", True))
    return checkpoint, thresholds, known_attack_labels, future_horizon_minutes, future_task_enabled


def build_dataset(split_path, seq_len, clip_value, future_horizon_minutes, known_attack_labels, max_sequences):
    base_dataset = SpatioTemporalNIDSDataset(
        arrow_dir_path=split_path,
        stats_path=DEFAULT_STATS,
        seq_len=seq_len,
        clip_value=clip_value,
    )
    known_attack_to_idx = {attack_name: idx for idx, attack_name in enumerate(known_attack_labels)}
    downstream_dataset = DownstreamNIDSDataset(
        base_dataset=base_dataset,
        future_horizon_minutes=future_horizon_minutes,
        known_attack_to_idx=known_attack_to_idx,
        max_sequences=max_sequences,
    )
    return downstream_dataset


def load_model(checkpoint, dataset, device, seq_len, future_task_enabled):
    num_cont = len(dataset.base_dataset.cont_cols)
    model = NIDSMultiTaskModel(
        backbone=SpatioTemporalTransformer(
            num_cont_features=num_cont,
            cat_vocab_sizes=CAT_VOCABS,
            seq_len=seq_len,
            init_mae=0.10,
            init_mfm=0.00,
        ),
        num_known_attack_classes=len(checkpoint.get("known_attack_labels", [])),
        use_future_head=future_task_enabled,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def describe_prediction(sequence_idx, item, prediction, future_horizon_minutes):
    start_time = format_timestamp(item["start_time"])
    end_time = format_timestamp(item["end_time"])
    ground_truth_attack = item["attack"]
    current_ground_truth = "attack" if int(item["label"]) == 1 else "benign"
    future_truth = "yes" if float(item["future_attack"]) >= 0.5 else "no"

    lines = [
        f"Sequence {sequence_idx + 1}",
        f"Window: {start_time} -> {end_time}",
        f"Current status: {prediction['status']}",
        f"Predicted attack type: {prediction['attack_type']}",
        f"Current attack probability: {prediction['current_attack_probability']:.4f}",
    ]

    if "known_attack_confidence" in prediction:
        lines.append(f"Known-family confidence: {prediction['known_attack_confidence']:.4f}")

    if prediction.get("future_task_enabled", True):
        if prediction["future_warning"]:
            lines.append(
                f"Early warning: attack likely within the next {future_horizon_minutes} minutes "
                f"(probability={prediction['future_attack_probability']:.4f})"
            )
        else:
            lines.append(
                f"Early warning: no imminent attack predicted within the next {future_horizon_minutes} minutes "
                f"(probability={prediction['future_attack_probability']:.4f})"
            )
    else:
        lines.append("Early warning: disabled for this checkpoint")

    lines.extend([
        f"Ground truth current label: {current_ground_truth}",
        f"Ground truth attack type: {ground_truth_attack}",
    ])

    if prediction.get("future_task_enabled", True):
        lines.append(f"Ground truth future attack in horizon: {future_truth}")

    future_lead_minutes = float(item["future_lead_minutes"])
    if prediction.get("future_task_enabled", True) and future_lead_minutes >= 0:
        lines.append(f"Ground truth future lead time: {future_lead_minutes:.2f} minutes")

    return "\n".join(lines)


def summarize_predictions(predictions):
    status_counts = {}
    future_warning_count = 0
    future_task_enabled = any(prediction.get("future_task_enabled", True) for prediction in predictions)

    for prediction in predictions:
        status_counts[prediction["status"]] = status_counts.get(prediction["status"], 0) + 1
        future_warning_count += int(prediction["future_warning"])

    summary_lines = ["Inference summary"]
    for status_name in sorted(status_counts):
        summary_lines.append(f"{status_name}: {status_counts[status_name]}")
    if future_task_enabled:
        summary_lines.append(f"future warnings: {future_warning_count}")
    else:
        summary_lines.append("future warnings: disabled")
    return "\n".join(summary_lines)


def run_inference():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint, thresholds, known_attack_labels, checkpoint_horizon, future_task_enabled = load_checkpoint(args.checkpoint, device)
    future_horizon_minutes = args.future_horizon_minutes or checkpoint_horizon
    split_path, resolved_split = resolve_split_path(args.split)

    dataset = build_dataset(
        split_path=split_path,
        seq_len=args.seq_len,
        clip_value=args.clip_value,
        future_horizon_minutes=future_horizon_minutes,
        known_attack_labels=known_attack_labels,
        max_sequences=args.max_sequences,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = load_model(checkpoint, dataset, device, args.seq_len, future_task_enabled)

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Running inference on split: {resolved_split}")
    print(f"Sequences to inspect: {len(dataset)}")
    print(f"Known attack labels: {known_attack_labels}")
    print(f"Future task enabled: {future_task_enabled}")

    all_predictions = []
    item_index = 0

    with torch.no_grad():
        for batch in data_loader:
            cont = batch["continuous"].to(device, non_blocking=True)
            cat = batch["categorical"].to(device, non_blocking=True)
            outputs = model(cont, cat, apply_mfm=False)
            decoded_predictions = NIDSMultiTaskModel.decode_predictions(
                outputs,
                known_attack_labels,
                current_threshold=thresholds["current"],
                known_attack_threshold=thresholds["known"],
                future_threshold=thresholds["future"],
            )

            batch_size = len(decoded_predictions)
            for batch_idx in range(batch_size):
                item = {
                    "start_time": batch["start_time"][batch_idx].item(),
                    "end_time": batch["end_time"][batch_idx].item(),
                    "label": batch["label"][batch_idx].item(),
                    "attack": batch["attack"][batch_idx],
                    "future_attack": batch["future_attack"][batch_idx].item(),
                    "future_lead_minutes": batch["future_lead_minutes"][batch_idx].item(),
                }
                prediction = decoded_predictions[batch_idx]
                all_predictions.append(prediction)
                print(describe_prediction(item_index, item, prediction, future_horizon_minutes))
                print("-" * 72)
                item_index += 1

    print(summarize_predictions(all_predictions))


if __name__ == "__main__":
    run_inference()