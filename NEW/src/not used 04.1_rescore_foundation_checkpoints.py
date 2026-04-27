"""
04.1 – Foundation Checkpoint Rescoring
=======================================
Re-evaluates every stt_epoch_*.pt in the checkpoint directory with a *fixed*
validation mask (same tokens masked for all epochs) and selects the best
backbone by F1 → AUC → masked-MSE priority.  Writes the winner as
stt_best.pt in the same directory and dumps a summary JSON.
"""

import contextlib
import glob
import importlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Dynamic module loading (matches pattern used by 04 and 05)
# ---------------------------------------------------------------------------

def load_local_module(module_name, filename):
    module_path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINT_DIRS = [
    "/home/aka/PFE-code/NEW/checkpoints",
]
DEFAULT_STATS_PATH = "/home/aka/PFE-code/NEW/nids_normalization_stats.json"
DEFAULT_VALID_DIR = "/home/aka/PFE-code/NEW/data/nids_src_grouped/validation"
DEFAULT_TEST_DIR  = "/home/aka/PFE-code/NEW/data/nids_src_grouped/test"
DEFAULT_VALIDATION_MAE_MASK_RATIO = 0.30

DEFAULT_SEQ_LEN = 32
DEFAULT_STRIDE = 16
DEFAULT_CLIP_VALUE = 5.0
DEFAULT_BATCH_SIZE = 512

# MFM rescoring support
DEFAULT_VALIDATION_MFM_MASK_RATIO = 0.15


# ---------------------------------------------------------------------------
# Metric helpers (no external dependencies)
# ---------------------------------------------------------------------------

def compute_binary_auroc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    if pos == 0 or neg == 0:
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
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[start:end] = avg_rank
        start = end
    pos_rank_sum = ranks[sorted_labels == 1].sum()
    return float((pos_rank_sum - pos * (pos + 1) / 2) / (pos * neg))


def compute_best_f1_metrics(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size == 0 or scores.size == 0 or np.unique(labels).size < 2:
        return {"best_f1": float("nan"), "best_precision": float("nan"),
                "best_recall": float("nan"), "best_threshold": float("nan")}

    candidates = np.unique(np.concatenate([
        np.linspace(0.0, scores.max() + 1e-9, 200),
        np.percentile(scores, [10, 20, 30, 40, 50, 60, 70, 80, 90]),
    ]))
    best_f1, best_prec, best_rec, best_thresh = 0.0, 0.0, 0.0, float("nan")
    pos_mask = labels == 1
    for t in candidates:
        pred = scores >= t
        tp = int(np.logical_and(pred, pos_mask).sum())
        fp = int(np.logical_and(pred, ~pos_mask).sum())
        fn = int(np.logical_and(~pred, pos_mask).sum())
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1 = 2.0 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best_f1:
            best_f1, best_prec, best_rec, best_thresh = f1, prec, rec, float(t)
    return {"best_f1": float(best_f1), "best_precision": float(best_prec),
            "best_recall": float(best_rec), "best_threshold": best_thresh}


def _auc_safe(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return float("-inf")
    return float(v)


def is_better_ids_checkpoint(challenger, champion):
    """masked-MSE → AUC selection (foundation logic)."""
    eps = 1e-6
    c_mse = float(challenger.get("masked_mse", float("inf")))
    b_mse = float(champion.get("masked_mse", float("inf")))
    if c_mse < b_mse - eps:
        return True
    if math.isclose(c_mse, b_mse, rel_tol=0.0, abs_tol=eps):
        c_auc = _auc_safe(challenger.get("anomaly_auc"))
        b_auc = _auc_safe(champion.get("anomaly_auc"))
        return c_auc > b_auc + eps
    return False


# ---------------------------------------------------------------------------
# Atomic save
# ---------------------------------------------------------------------------

def atomic_torch_save(payload, destination_path):
    destination_path = Path(destination_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=destination_path.parent,
        prefix=f".{destination_path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            torch.save(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, destination_path)
    except Exception as exc:
        with contextlib.suppress(FileNotFoundError):
            os.remove(tmp)
        raise RuntimeError(f"Atomic save failed: {destination_path}") from exc


# ---------------------------------------------------------------------------
# Fixed validation mask (same across all checkpoint evaluations)
# ---------------------------------------------------------------------------


def build_fixed_validation_mask(num_sequences, seq_len, mask_ratio, device, seed=42):
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    mask = torch.rand(num_sequences, seq_len, generator=rng) < mask_ratio
    return mask.to(device)


# ---------------------------------------------------------------------------
# Per-checkpoint evaluation
# ---------------------------------------------------------------------------


def evaluate_checkpoint(model, data_loader, device, fixed_masks, mask_ratio, use_mfm=False):
    """
    Evaluates a loaded model with pre-generated fixed masks.
    If use_mfm=True, evaluates the MFM head (apply_mfm=True), else MAE (apply_mfm=False).
    """
    import torch.nn.functional as F

    model.eval()
    total_loss = 0.0
    total_masked_mse = 0.0
    n_batches = 0
    anomaly_scores = []
    anomaly_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="  Rescoring (MFM)" if use_mfm else "  Rescoring (MAE)", leave=False)):
            cont = batch["continuous"].to(device, non_blocking=True)
            cat  = batch["categorical"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            spatial_mask = fixed_masks[batch_idx].to(device)
            spatial_mask = spatial_mask[:cont.shape[0]]

            reconstructed, used_mask = model(cont, cat, spatial_mask=spatial_mask, apply_mfm=use_mfm)

            mse_raw     = (reconstructed - cont).pow(2)
            robust_raw  = F.smooth_l1_loss(reconstructed, cont, reduction="none", beta=1.0)
            mask_exp    = used_mask.unsqueeze(-1).float()
            denom       = mask_exp.sum().clamp(min=1.0) * reconstructed.shape[-1]
            masked_mse  = (mse_raw * mask_exp).sum() / denom
            masked_rob  = (robust_raw * mask_exp).sum() / denom
            full_rob    = robust_raw.mean()
            loss        = 0.85 * masked_rob + 0.15 * full_rob

            total_loss       += float(loss.item())
            total_masked_mse += float(masked_mse.item())
            n_batches        += 1

            seq_scores = (reconstructed - cont).pow(2).mean(dim=(1, 2)).cpu().numpy()
            anomaly_scores.append(seq_scores)
            anomaly_labels.append((labels > 0).int().cpu().numpy())

    n_batches = max(n_batches, 1)
    scores = np.concatenate(anomaly_scores) if anomaly_scores else np.array([])
    lbls   = np.concatenate(anomaly_labels) if anomaly_labels else np.array([])

    auc = compute_binary_auroc(lbls, scores) if lbls.size > 0 and np.unique(lbls).size > 1 else float("nan")
    f1_metrics = compute_best_f1_metrics(lbls, scores)

    return {
        "loss":          total_loss / n_batches,
        "masked_mse":    total_masked_mse / n_batches,
        "anomaly_auc":   auc,
        **f1_metrics,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running foundation checkpoint rescoring on {device}.", flush=True)

    # Load modules
    st_data_loader   = load_local_module("st_data_loader",   "02_st_data_loader.py")
    stt_architecture = load_local_module("stt_architecture", "03_stt_architecture.py")
    SpatioTemporalNIDSDataset  = st_data_loader.SpatioTemporalNIDSDataset
    SpatioTemporalTransformer  = stt_architecture.SpatioTemporalTransformer

    # Validation dataset
    valid_dir = DEFAULT_VALID_DIR if os.path.exists(DEFAULT_VALID_DIR) else DEFAULT_TEST_DIR
    valid_dataset = SpatioTemporalNIDSDataset(
        arrow_dir_path=valid_dir,
        stats_path=DEFAULT_STATS_PATH,
        seq_len=DEFAULT_SEQ_LEN,
        stride=DEFAULT_STRIDE,
        clip_value=DEFAULT_CLIP_VALUE,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=DEFAULT_BATCH_SIZE,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=device.type == "cuda",
    )
    print(f"Validation dataset ready: {len(valid_dataset)} sequences", flush=True)


    # Pre-build fixed masks for every batch (so all checkpoints see identical masks)
    print("Building fixed validation masks for MAE and MFM...", flush=True)
    fixed_masks_mae = []
    fixed_masks_mfm = []
    for batch in valid_loader:
        bsz = batch["continuous"].shape[0]
        rng_mae = torch.Generator()
        rng_mae.manual_seed(42 + len(fixed_masks_mae))
        m_mae = torch.rand(bsz, DEFAULT_SEQ_LEN, generator=rng_mae) < DEFAULT_VALIDATION_MAE_MASK_RATIO
        fixed_masks_mae.append(m_mae)

        rng_mfm = torch.Generator()
        rng_mfm.manual_seed(142 + len(fixed_masks_mfm))
        m_mfm = torch.rand(bsz, DEFAULT_SEQ_LEN, generator=rng_mfm) < DEFAULT_VALIDATION_MFM_MASK_RATIO
        fixed_masks_mfm.append(m_mfm)

    # Build a blank model (weights loaded per checkpoint)
    num_cont = len(valid_dataset.cont_cols)
    cat_vocabs = [256, 256, 256, 65536, 65536, 256, 256, 256, 256]
    model = SpatioTemporalTransformer(
        num_cont_features=num_cont,
        cat_vocab_sizes=cat_vocabs,
        seq_len=DEFAULT_SEQ_LEN,
        init_mae=DEFAULT_VALIDATION_MAE_MASK_RATIO,
        init_mfm=DEFAULT_VALIDATION_MFM_MASK_RATIO,
    ).to(device)

    # Collect all epoch checkpoints across all configured dirs
    all_epoch_checkpoints = []
    for checkpoint_dir in DEFAULT_CHECKPOINT_DIRS:
        pattern = os.path.join(checkpoint_dir, "stt_epoch_*.pt")
        found = sorted(glob.glob(pattern),
                       key=lambda p: int(Path(p).stem.rsplit("_", 1)[-1]))
        if found:
            all_epoch_checkpoints.extend([(checkpoint_dir, p) for p in found])
            print(f"Found {len(found)} epoch checkpoints in {checkpoint_dir}", flush=True)
        else:
            print(f"No epoch checkpoints found in {checkpoint_dir}.", flush=True)

    if not all_epoch_checkpoints:
        raise RuntimeError("No epoch checkpoints found in any configured directory.")

    # Score every checkpoint

    results = []
    best_combined_metric = None
    best_metrics = None
    best_checkpoint_path = None
    best_checkpoint_dir = None

    for checkpoint_dir, ckpt_path in all_epoch_checkpoints:
        epoch_num = int(Path(ckpt_path).stem.rsplit("_", 1)[-1])
        print(f"\nEvaluating epoch {epoch_num}: {ckpt_path}", flush=True)
        try:
            saved = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(saved["model_state_dict"])
        except Exception as exc:
            print(f"  Skipping (load error): {exc}", flush=True)
            continue

        metrics_mae = evaluate_checkpoint(model, valid_loader, device, fixed_masks_mae, DEFAULT_VALIDATION_MAE_MASK_RATIO, use_mfm=False)
        metrics_mfm = evaluate_checkpoint(model, valid_loader, device, fixed_masks_mfm, DEFAULT_VALIDATION_MFM_MASK_RATIO, use_mfm=True)

        combined_metric = metrics_mae["masked_mse"] + metrics_mfm["masked_mse"]

        entry = {
            "epoch":           epoch_num,
            "checkpoint_path": str(ckpt_path),
            "checkpoint_dir":  str(checkpoint_dir),
            "mae": metrics_mae,
            "mfm": metrics_mfm,
            "combined_masked_mse": combined_metric,
        }
        results.append(entry)

        print(
            f"  Epoch {epoch_num:3d} | MAE: masked_mse={metrics_mae['masked_mse']:.6f} | AUC={metrics_mae['anomaly_auc']:.4f} | F1={metrics_mae['best_f1']:.4f} "
            f"(P={metrics_mae['best_precision']:.4f} R={metrics_mae['best_recall']:.4f} thr={metrics_mae['best_threshold']:.4f})",
            flush=True,
        )
        print(
            f"                 | MFM: masked_mse={metrics_mfm['masked_mse']:.6f} | AUC={metrics_mfm['anomaly_auc']:.4f} | F1={metrics_mfm['best_f1']:.4f} "
            f"(P={metrics_mfm['best_precision']:.4f} R={metrics_mfm['best_recall']:.4f} thr={metrics_mfm['best_threshold']:.4f})",
            flush=True,
        )
        print(
            f"                 | Combined masked_mse={combined_metric:.6f}",
            flush=True,
        )

        # Select best checkpoint based on combined metric (lower is better)
        if best_combined_metric is None or combined_metric < best_combined_metric:
            best_combined_metric = combined_metric
            best_metrics = {"mae": metrics_mae, "mfm": metrics_mfm, "combined_masked_mse": combined_metric, "epoch": epoch_num}
            best_checkpoint_path = ckpt_path
            best_checkpoint_dir  = checkpoint_dir

    if best_checkpoint_path is None:
        raise RuntimeError("No checkpoint could be rescored.")

    print(f"\nBest checkpoint (by combined masked_mse): {best_checkpoint_path}", flush=True)
    print(
        f"  Combined masked_mse={best_metrics['combined_masked_mse']:.6f} | "
        f"MAE masked_mse={best_metrics['mae']['masked_mse']:.6f} | "
        f"MFM masked_mse={best_metrics['mfm']['masked_mse']:.6f}",
        flush=True,
    )



    # Write stt_best.pt (now based on combined metric)
    best_saved = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    best_payload = {
        "epoch":              best_metrics["epoch"],
        "model_state_dict":   best_saved["model_state_dict"],
        "val_masked_mse_mae": best_metrics["mae"]["masked_mse"],
        "val_masked_mse_mfm": best_metrics["mfm"]["masked_mse"],
        "val_combined_masked_mse": best_metrics["combined_masked_mse"],
        "anomaly_auc_mae":    best_metrics["mae"]["anomaly_auc"],
        "anomaly_auc_mfm":    best_metrics["mfm"]["anomaly_auc"],
        "best_f1_mae":        best_metrics["mae"]["best_f1"],
        "best_f1_mfm":        best_metrics["mfm"]["best_f1"],
        "best_precision_mae": best_metrics["mae"]["best_precision"],
        "best_precision_mfm": best_metrics["mfm"]["best_precision"],
        "best_recall_mae":    best_metrics["mae"]["best_recall"],
        "best_recall_mfm":    best_metrics["mfm"]["best_recall"],
        "best_threshold_mae": best_metrics["mae"]["best_threshold"],
        "best_threshold_mfm": best_metrics["mfm"]["best_threshold"],
        "validation_mae_mask_ratio": DEFAULT_VALIDATION_MAE_MASK_RATIO,
        "validation_mfm_mask_ratio": DEFAULT_VALIDATION_MFM_MASK_RATIO,
        "rescore_source":     str(best_checkpoint_path),
    }
    output_best = os.path.join(best_checkpoint_dir, "stt_best.pt")
    atomic_torch_save(best_payload, output_best)
    print(f"stt_best.pt written to {output_best}", flush=True)

    # Write summary JSON (with both MAE, MFM, and combined results)
    summary = {
        "best_epoch":              best_payload["epoch"],
        "best_checkpoint":         str(best_checkpoint_path),
        "best_masked_mse_mae":     best_metrics["mae"]["masked_mse"],
        "best_masked_mse_mfm":     best_metrics["mfm"]["masked_mse"],
        "best_combined_masked_mse": best_metrics["combined_masked_mse"],
        "best_anomaly_auc_mae":    best_metrics["mae"]["anomaly_auc"],
        "best_anomaly_auc_mfm":    best_metrics["mfm"]["anomaly_auc"],
        "best_f1_mae":             best_metrics["mae"]["best_f1"],
        "best_f1_mfm":             best_metrics["mfm"]["best_f1"],
        "best_precision_mae":      best_metrics["mae"]["best_precision"],
        "best_precision_mfm":      best_metrics["mfm"]["best_precision"],
        "best_recall_mae":         best_metrics["mae"]["best_recall"],
        "best_recall_mfm":         best_metrics["mfm"]["best_recall"],
        "best_threshold_mae":      best_metrics["mae"]["best_threshold"],
        "best_threshold_mfm":      best_metrics["mfm"]["best_threshold"],
        "mae_mask_ratio":          DEFAULT_VALIDATION_MAE_MASK_RATIO,
        "mfm_mask_ratio":          DEFAULT_VALIDATION_MFM_MASK_RATIO,
        "all_results":             results,
    }
    summary_path = os.path.join(best_checkpoint_dir, "foundation_rescore_summary_with_mfm.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Summary written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
import argparse
