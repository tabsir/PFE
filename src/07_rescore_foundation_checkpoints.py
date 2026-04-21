import argparse
import glob
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def load_local_module(module_name, filename):
    module_path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


st_data_loader = load_local_module("st_data_loader", "02_st_data_loader.py")
stt_architecture = load_local_module("stt_architecture", "03_stt_architecture.py")
foundation_train = load_local_module("foundation_train", "04_train_foundation.py")

SpatioTemporalNIDSDataset = st_data_loader.SpatioTemporalNIDSDataset
SpatioTemporalTransformer = stt_architecture.SpatioTemporalTransformer

DEFAULT_CHECKPOINT_DIRS = [
    "/home/aka/PFE-code/checkpoints",
    "/home/aka/PFE-code/checkpoints_test",
]
DEFAULT_STATS_PATH = "/home/aka/PFE-code/nids_normalization_stats.json"
DEFAULT_VALID_DIR = "/home/aka/PFE-code/data/nids_transformer_split/validation"
DEFAULT_TEST_DIR = "/home/aka/PFE-code/data/nids_transformer_split/test"
DEFAULT_DATA_SIGNATURE = "grouped_chronological_v1"
DEFAULT_VALIDATION_MAE_MASK_RATIO = 0.30
DEFAULT_SEQ_LEN = 32
DEFAULT_CLIP_VALUE = 5.0
DEFAULT_BATCH_SIZE = 512
DEFAULT_CAT_VOCABS = [256, 256, 256, 65536, 65536, 256, 256, 256, 256]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-evaluate foundation checkpoints with a fixed validation mask and rewrite stt_best.pt."
    )
    parser.add_argument(
        "--checkpoint-dirs",
        nargs="+",
        default=DEFAULT_CHECKPOINT_DIRS,
        help="Checkpoint directories containing stt_epoch_*.pt files.",
    )
    parser.add_argument(
        "--validation-mask-ratio",
        type=float,
        default=DEFAULT_VALIDATION_MAE_MASK_RATIO,
        help="Fixed MAE mask ratio used for rescoring validation.",
    )
    parser.add_argument(
        "--stats-path",
        default=DEFAULT_STATS_PATH,
        help="Normalization stats JSON used by the dataset.",
    )
    parser.add_argument(
        "--validation-dir",
        default=DEFAULT_VALID_DIR,
        help="Validation dataset directory.",
    )
    parser.add_argument(
        "--test-dir",
        default=DEFAULT_TEST_DIR,
        help="Fallback dataset directory if validation split is unavailable.",
    )
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN, help="Sequence length.")
    parser.add_argument("--clip-value", type=float, default=DEFAULT_CLIP_VALUE, help="Continuous feature clamp value.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Validation batch size.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device for rescoring.",
    )
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_checkpoint_paths(checkpoint_dir):
    checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, "stt_epoch_*.pt"))
    checkpoint_paths.sort(key=lambda path: int(path.split("_")[-1].split(".")[0]))
    return checkpoint_paths


def build_validation_loader(validation_dir, test_dir, stats_path, seq_len, clip_value, batch_size, device):
    validation_path = validation_dir if os.path.exists(validation_dir) else test_dir
    if validation_path == test_dir:
        print("Validation split introuvable. Utilisation temporaire du split test comme validation pour le rescoring.")

    validation_dataset = SpatioTemporalNIDSDataset(
        arrow_dir_path=validation_path,
        stats_path=stats_path,
        seq_len=seq_len,
        clip_value=clip_value,
    )
    num_workers = min(8, os.cpu_count() or 1)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    return validation_dataset, validation_loader


def evaluate_checkpoint(model, data_loader, device, mae_mask_ratio):
    model.eval()
    metrics = {
        "loss": 0.0,
        "masked_mse": 0.0,
        "full_mse": 0.0,
    }
    anomaly_scores = []
    anomaly_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader), desc="Validation", leave=False):
            cont = batch["continuous"].to(device, non_blocking=True)
            cat = batch["categorical"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            validation_mask = foundation_train.build_validation_mask(cont.shape[0], cont.shape[1], mae_mask_ratio, device)
            reconstructed, spatial_mask = model(cont, cat, spatial_mask=validation_mask, apply_mfm=False)
            batch_metrics = foundation_train.compute_reconstruction_metrics(reconstructed, cont, spatial_mask)

            metrics["loss"] += float(batch_metrics["train_loss"].item())
            metrics["masked_mse"] += float(batch_metrics["masked_mse"].item())
            metrics["full_mse"] += float(batch_metrics["full_mse"].item())

            sequence_scores = (reconstructed - cont).pow(2).mean(dim=(1, 2)).detach().cpu().numpy()
            anomaly_scores.append(sequence_scores)
            anomaly_labels.append((labels > 0).int().detach().cpu().numpy())

    for key in metrics:
        metrics[key] /= max(len(data_loader), 1)

    scores = np.concatenate(anomaly_scores) if anomaly_scores else np.array([])
    labels = np.concatenate(anomaly_labels) if anomaly_labels else np.array([])

    if scores.size > 0:
        benign_scores = scores[labels == 0]
        attack_scores = scores[labels == 1]
        metrics["benign_score_mean"] = float(benign_scores.mean()) if benign_scores.size else float("nan")
        metrics["attack_score_mean"] = float(attack_scores.mean()) if attack_scores.size else float("nan")
    else:
        metrics["benign_score_mean"] = float("nan")
        metrics["attack_score_mean"] = float("nan")

    if labels.size > 0 and np.unique(labels).size > 1:
        metrics["anomaly_auc"] = foundation_train.compute_binary_auroc(labels, scores)
    else:
        metrics["anomaly_auc"] = float("nan")

    return metrics


def build_best_checkpoint_payload(checkpoint, metrics, best_checkpoint_path, validation_mae_mask_ratio):
    return {
        "epoch": checkpoint.get("epoch"),
        "model_state_dict": checkpoint["model_state_dict"],
        "optimizer_state_dict": checkpoint.get("optimizer_state_dict"),
        "scheduler_state_dict": checkpoint.get("scheduler_state_dict"),
        "val_loss": metrics["loss"],
        "val_masked_mse": metrics["masked_mse"],
        "val_full_mse": metrics["full_mse"],
        "anomaly_auc": metrics["anomaly_auc"],
        "best_val_masked_mse": metrics["masked_mse"],
        "best_anomaly_auc": metrics["anomaly_auc"],
        "validation_mae_mask_ratio": validation_mae_mask_ratio,
        "rescored_from_checkpoint": best_checkpoint_path,
        "data_signature": checkpoint.get("data_signature", DEFAULT_DATA_SIGNATURE),
    }


def rescore_checkpoint_dir(checkpoint_dir, model, data_loader, device, validation_mae_mask_ratio):
    checkpoint_paths = get_checkpoint_paths(checkpoint_dir)
    if not checkpoint_paths:
        print(f"Aucun checkpoint d'époque trouvé dans {checkpoint_dir}.")
        return None

    best_val_masked_mse = float("inf")
    best_anomaly_auc = float("-inf")
    best_checkpoint_path = None
    best_checkpoint = None
    best_metrics = None
    summary_rows = []

    for checkpoint_path in tqdm(checkpoint_paths, desc=f"Rescoring {os.path.basename(checkpoint_dir)}"):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint.get("data_signature") not in (None, DEFAULT_DATA_SIGNATURE):
            print(f"Checkpoint ignoré (signature incompatible): {checkpoint_path}")
            continue

        model.load_state_dict(checkpoint["model_state_dict"])
        metrics = evaluate_checkpoint(model, data_loader, device, validation_mae_mask_ratio)

        is_best = foundation_train.is_better_checkpoint(metrics, best_val_masked_mse, best_anomaly_auc)
        if is_best:
            best_val_masked_mse = metrics["masked_mse"]
            best_anomaly_auc = metrics["anomaly_auc"]
            best_checkpoint_path = checkpoint_path
            best_checkpoint = checkpoint
            best_metrics = metrics

        checkpoint["rescored_val_loss"] = metrics["loss"]
        checkpoint["rescored_val_masked_mse"] = metrics["masked_mse"]
        checkpoint["rescored_val_full_mse"] = metrics["full_mse"]
        checkpoint["rescored_anomaly_auc"] = metrics["anomaly_auc"]
        checkpoint["rescored_benign_score_mean"] = metrics["benign_score_mean"]
        checkpoint["rescored_attack_score_mean"] = metrics["attack_score_mean"]
        checkpoint["best_val_masked_mse"] = best_val_masked_mse
        checkpoint["best_anomaly_auc"] = best_anomaly_auc
        checkpoint["validation_mae_mask_ratio"] = validation_mae_mask_ratio
        torch.save(checkpoint, checkpoint_path)

        summary_rows.append({
            "checkpoint": os.path.basename(checkpoint_path),
            "epoch": int(checkpoint.get("epoch", -1)) + 1,
            "rescored_val_loss": metrics["loss"],
            "rescored_val_masked_mse": metrics["masked_mse"],
            "rescored_val_full_mse": metrics["full_mse"],
            "rescored_anomaly_auc": metrics["anomaly_auc"],
            "is_best": is_best,
        })

    if best_checkpoint is None or best_metrics is None or best_checkpoint_path is None:
        print(f"Aucun checkpoint valide à rescorrer dans {checkpoint_dir}.")
        return None

    best_payload = build_best_checkpoint_payload(
        best_checkpoint,
        best_metrics,
        best_checkpoint_path,
        validation_mae_mask_ratio,
    )
    best_output_path = os.path.join(checkpoint_dir, "stt_best.pt")
    torch.save(best_payload, best_output_path)

    summary_path = os.path.join(checkpoint_dir, "foundation_rescore_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(
            {
                "validation_mae_mask_ratio": validation_mae_mask_ratio,
                "best_checkpoint": os.path.basename(best_checkpoint_path),
                "best_epoch": int(best_checkpoint.get("epoch", -1)) + 1,
                "best_val_masked_mse": best_metrics["masked_mse"],
                "best_anomaly_auc": best_metrics["anomaly_auc"],
                "rescored_checkpoints": summary_rows,
            },
            handle,
            indent=2,
        )

    print(
        f"{checkpoint_dir}: best={os.path.basename(best_checkpoint_path)} | "
        f"ValMaskedMSE={best_metrics['masked_mse']:.6f} | "
        f"AUC={best_metrics['anomaly_auc']:.4f}"
    )
    return {
        "checkpoint_dir": checkpoint_dir,
        "best_checkpoint": best_checkpoint_path,
        "best_epoch": int(best_checkpoint.get("epoch", -1)) + 1,
        "best_metrics": best_metrics,
        "summary_path": summary_path,
        "best_output_path": best_output_path,
    }


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Running foundation checkpoint rescoring on {device}.")

    validation_dataset, validation_loader = build_validation_loader(
        validation_dir=args.validation_dir,
        test_dir=args.test_dir,
        stats_path=args.stats_path,
        seq_len=args.seq_len,
        clip_value=args.clip_value,
        batch_size=args.batch_size,
        device=device,
    )
    print(f"Validation dataset ready: {len(validation_dataset)} sequences")

    model = SpatioTemporalTransformer(
        num_cont_features=len(validation_dataset.cont_cols),
        cat_vocab_sizes=DEFAULT_CAT_VOCABS,
        seq_len=args.seq_len,
        init_mae=0.10,
        init_mfm=0.00,
    ).to(device)

    results = []
    for checkpoint_dir in args.checkpoint_dirs:
        result = rescore_checkpoint_dir(
            checkpoint_dir=checkpoint_dir,
            model=model,
            data_loader=validation_loader,
            device=device,
            validation_mae_mask_ratio=args.validation_mask_ratio,
        )
        if result is not None:
            results.append(result)

    if not results:
        raise RuntimeError("Aucun checkpoint n'a pu être rescorré.")

    print("Rescoring terminé.")
    for result in results:
        print(
            f"{result['checkpoint_dir']} -> stt_best.pt = {os.path.basename(result['best_checkpoint'])} "
            f"(epoch {result['best_epoch']}, ValMaskedMSE={result['best_metrics']['masked_mse']:.6f}, "
            f"AUC={result['best_metrics']['anomaly_auc']:.4f})"
        )


if __name__ == "__main__":
    main()