import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import glob
import importlib
import numpy as np
from pathlib import Path


# 1. Chargement dynamique des modules
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


def get_progressive_ratios(epoch):
    """
    Variante plus agressive mais progressive pour comparer un masquage final plus fort.
    """
    if epoch < 5:
        return 0.10, 0.00

    if epoch < 15:
        progress = (epoch - 5) / 10
        mae = 0.10 + progress * (0.20 - 0.10)
        mfm = 0.00 + progress * (0.03 - 0.00)
        return mae, mfm

    if epoch < 30:
        progress = (epoch - 15) / 15
        mae = 0.20 + progress * (0.35 - 0.20)
        mfm = 0.03 + progress * (0.09 - 0.03)
        return mae, mfm

    progress = min((epoch - 30) / 20, 1.0)
    mae = 0.35 + progress * (0.45 - 0.35)
    mfm = 0.09 + progress * (0.15 - 0.09)
    return mae, mfm


def get_last_checkpoint(checkpoint_dir):
    """Recherche le fichier .pt avec l'époque la plus élevée pour reprendre l'entraînement."""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "stt_epoch_*.pt"))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return checkpoints[-1]


def build_validation_mask(batch_size, seq_len, mask_ratio, device):
    return torch.rand(batch_size, seq_len, device=device) < mask_ratio


def compute_reconstruction_metrics(reconstructed, target, spatial_mask):
    mse_raw = (reconstructed - target).pow(2)
    robust_raw = F.smooth_l1_loss(reconstructed, target, reduction='none', beta=1.0)

    mask_expanded = spatial_mask.unsqueeze(-1).float()
    masked_denominator = mask_expanded.sum().clamp(min=1.0) * reconstructed.shape[-1]

    masked_robust = (robust_raw * mask_expanded).sum() / masked_denominator
    full_robust = robust_raw.mean()
    masked_mse = (mse_raw * mask_expanded).sum() / masked_denominator
    full_mse = mse_raw.mean()

    return {
        'train_loss': 0.85 * masked_robust + 0.15 * full_robust,
        'masked_mse': masked_mse,
        'full_mse': full_mse,
    }


def compute_binary_auroc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    positive_mask = labels == 1
    negative_mask = labels == 0
    n_positive = int(positive_mask.sum())
    n_negative = int(negative_mask.sum())
    if n_positive == 0 or n_negative == 0:
        return float('nan')

    order = np.argsort(scores, kind='mergesort')
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


def _auc_for_selection(auc_value):
    if auc_value is None or math.isnan(auc_value):
        return float('-inf')
    return float(auc_value)


def is_better_checkpoint(validation_metrics, best_val_masked_mse, best_anomaly_auc, epsilon=1e-6):
    current_masked_mse = float(validation_metrics['masked_mse'])
    if current_masked_mse < best_val_masked_mse - epsilon:
        return True
    if math.isclose(current_masked_mse, best_val_masked_mse, rel_tol=0.0, abs_tol=epsilon):
        return _auc_for_selection(validation_metrics.get('anomaly_auc')) > _auc_for_selection(best_anomaly_auc)
    return False


def evaluate(model, data_loader, device, mae_mask_ratio):
    model.eval()
    metrics = {
        'loss': 0.0,
        'masked_mse': 0.0,
        'full_mse': 0.0,
    }
    anomaly_scores = []
    anomaly_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, total=len(data_loader), desc='Validation', leave=False):
            cont = batch['continuous'].to(device, non_blocking=True)
            cat = batch['categorical'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)

            validation_mask = build_validation_mask(cont.shape[0], cont.shape[1], mae_mask_ratio, device)
            reconstructed, spatial_mask = model(cont, cat, spatial_mask=validation_mask, apply_mfm=False)
            batch_metrics = compute_reconstruction_metrics(reconstructed, cont, spatial_mask)

            metrics['loss'] += batch_metrics['train_loss'].item()
            metrics['masked_mse'] += batch_metrics['masked_mse'].item()
            metrics['full_mse'] += batch_metrics['full_mse'].item()

            sequence_scores = (reconstructed - cont).pow(2).mean(dim=(1, 2)).detach().cpu().numpy()
            anomaly_scores.append(sequence_scores)
            anomaly_labels.append((labels > 0).int().detach().cpu().numpy())

    for key in metrics:
        metrics[key] /= len(data_loader)

    scores = np.concatenate(anomaly_scores) if anomaly_scores else np.array([])
    labels = np.concatenate(anomaly_labels) if anomaly_labels else np.array([])

    if scores.size > 0:
        benign_scores = scores[labels == 0]
        attack_scores = scores[labels == 1]
        metrics['benign_score_mean'] = float(benign_scores.mean()) if benign_scores.size else float('nan')
        metrics['attack_score_mean'] = float(attack_scores.mean()) if attack_scores.size else float('nan')
    else:
        metrics['benign_score_mean'] = float('nan')
        metrics['attack_score_mean'] = float('nan')

    if labels.size > 0 and np.unique(labels).size > 1:
        metrics['anomaly_auc'] = compute_binary_auroc(labels, scores)
    else:
        metrics['anomaly_auc'] = float('nan')

    return metrics


def train_foundation_test():
    # --- 1. Configuration & Hyperparamètres ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 50
    LR_WARMUP_EPOCHS = 4
    BATCH_SIZE = 512
    ACC_STEPS = 1
    LR = 1.5e-4
    SEQ_LEN = 32
    CLIP_VALUE = 5.0
    NUM_WORKERS = min(8, os.cpu_count() or 1)
    DATA_SIGNATURE = 'grouped_chronological_v1'
    VALIDATION_MAE_MASK_RATIO = 0.30

    TRAIN_DIR = "/home/aka/PFE-code/data/nids_transformer_split/train"
    VALID_DIR = "/home/aka/PFE-code/data/nids_transformer_split/validation"
    TEST_DIR = "/home/aka/PFE-code/data/nids_transformer_split/test"
    STATS_PATH = "nids_normalization_stats.json"
    CHECKPOINT_DIR = "./checkpoints_test"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # --- 2. Initialisation des Données ---
    validation_path = VALID_DIR if os.path.exists(VALID_DIR) else TEST_DIR
    if validation_path == TEST_DIR:
        print('Validation split introuvable. Utilisation temporaire du split test comme validation.')

    train_dataset = SpatioTemporalNIDSDataset(
        arrow_dir_path=TRAIN_DIR,
        stats_path=STATS_PATH,
        seq_len=SEQ_LEN,
        clip_value=CLIP_VALUE,
    )
    validation_dataset = SpatioTemporalNIDSDataset(
        arrow_dir_path=validation_path,
        stats_path=STATS_PATH,
        seq_len=SEQ_LEN,
        clip_value=CLIP_VALUE,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == 'cuda',
        persistent_workers=NUM_WORKERS > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=DEVICE.type == 'cuda',
        persistent_workers=NUM_WORKERS > 0,
    )

    # --- 3. Initialisation du Modèle & Optimisation ---
    NUM_CONT = len(train_dataset.cont_cols)
    CAT_VOCABS = [256, 256, 256, 65536, 65536, 256, 256, 256, 256]

    model = SpatioTemporalTransformer(
        num_cont_features=NUM_CONT,
        cat_vocab_sizes=CAT_VOCABS,
        seq_len=SEQ_LEN,
        init_mae=0.10,
        init_mfm=0.00,
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    train_steps_per_epoch = max(math.ceil(len(train_loader) / ACC_STEPS), 1)
    total_steps = EPOCHS * train_steps_per_epoch
    warmup_steps = LR_WARMUP_EPOCHS * train_steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- 4. Logique de Reprise Automatique (Checkpointing) ---
    start_epoch = 0
    best_val_masked_mse = float('inf')
    best_anomaly_auc = float('-inf')
    last_cp = get_last_checkpoint(CHECKPOINT_DIR)

    if last_cp:
        print(f"Checkpoint trouvé : {last_cp}. Chargement en cours...")
        checkpoint = torch.load(last_cp, map_location=DEVICE)
        if checkpoint.get('data_signature') == DATA_SIGNATURE:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_masked_mse = checkpoint.get('best_val_masked_mse', float('inf'))
            best_anomaly_auc = checkpoint.get('best_anomaly_auc', float('-inf'))
            print(f"Reprise prête à partir de l'époque {start_epoch + 1}/{EPOCHS}")
        else:
            print("Checkpoint ignoré: il provient d'une ancienne configuration de données.")
    else:
        print(f"Aucun checkpoint trouvé dans {CHECKPOINT_DIR}. Début d'un nouvel entraînement sur {DEVICE}.")

    # --- 5. Boucle d'Entraînement Principale ---
    for epoch in range(start_epoch, EPOCHS):
        model.train()

        mae_r, mfm_r = get_progressive_ratios(epoch)
        model.mae_mask_ratio = mae_r
        model.mfm_layer.mask_ratio = mfm_r

        epoch_loss = 0.0
        epoch_masked_mse = 0.0
        epoch_full_mse = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}")

        optimizer.zero_grad()

        for i, batch in progress_bar:
            cont = batch['continuous'].to(DEVICE, non_blocking=True)
            cat = batch['categorical'].to(DEVICE, non_blocking=True)

            reconstructed, spatial_mask = model(cont, cat)
            batch_metrics = compute_reconstruction_metrics(reconstructed, cont, spatial_mask)
            loss = batch_metrics['train_loss'] / ACC_STEPS

            loss.backward()

            if (i + 1) % ACC_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += batch_metrics['train_loss'].item()
            epoch_masked_mse += batch_metrics['masked_mse'].item()
            epoch_full_mse += batch_metrics['full_mse'].item()
            if i % 10 == 0:
                progress_bar.set_postfix({
                    "loss": f"{epoch_loss/(i+1):.4f}",
                    "masked_mse": f"{epoch_masked_mse/(i+1):.4f}",
                    "MAE": f"{mae_r:.2f}",
                    "MFM": f"{mfm_r:.2f}",
                })

        validation_metrics = evaluate(model, validation_loader, DEVICE, VALIDATION_MAE_MASK_RATIO)
        train_loss = epoch_loss / len(train_loader)
        train_masked_mse = epoch_masked_mse / len(train_loader)
        train_full_mse = epoch_full_mse / len(train_loader)

        is_best_checkpoint = is_better_checkpoint(validation_metrics, best_val_masked_mse, best_anomaly_auc)
        current_best_val_masked_mse = validation_metrics['masked_mse'] if is_best_checkpoint else best_val_masked_mse
        current_best_anomaly_auc = validation_metrics['anomaly_auc'] if is_best_checkpoint else best_anomaly_auc

        checkpoint_path = f"{CHECKPOINT_DIR}/stt_epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': train_loss,
            'train_masked_mse': train_masked_mse,
            'train_full_mse': train_full_mse,
            'val_loss': validation_metrics['loss'],
            'val_masked_mse': validation_metrics['masked_mse'],
            'val_full_mse': validation_metrics['full_mse'],
            'anomaly_auc': validation_metrics['anomaly_auc'],
            'best_val_masked_mse': current_best_val_masked_mse,
            'best_anomaly_auc': current_best_anomaly_auc,
            'validation_mae_mask_ratio': VALIDATION_MAE_MASK_RATIO,
            'data_signature': DATA_SIGNATURE,
        }, checkpoint_path)

        if is_best_checkpoint:
            best_val_masked_mse = validation_metrics['masked_mse']
            best_anomaly_auc = validation_metrics['anomaly_auc']
            best_checkpoint_path = os.path.join(CHECKPOINT_DIR, 'stt_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_masked_mse': validation_metrics['masked_mse'],
                'val_full_mse': validation_metrics['full_mse'],
                'anomaly_auc': validation_metrics['anomaly_auc'],
                'validation_mae_mask_ratio': VALIDATION_MAE_MASK_RATIO,
                'data_signature': DATA_SIGNATURE,
            }, best_checkpoint_path)

        print(
            f"Epoch {epoch+1} complete. "
            f"TrainLoss: {train_loss:.6f} | TrainMaskedMSE: {train_masked_mse:.6f} | "
            f"ValMaskedMSE: {validation_metrics['masked_mse']:.6f} | "
            f"ValFullMSE: {validation_metrics['full_mse']:.6f} | "
            f"AUC: {validation_metrics['anomaly_auc']:.4f} | "
            f"ValMaskRatio: {VALIDATION_MAE_MASK_RATIO:.2f} | Checkpoint: {checkpoint_path}"
        )
        print(
            f"Validation score gap -> benign: {validation_metrics['benign_score_mean']:.6f}, "
            f"attack: {validation_metrics['attack_score_mean']:.6f}"
        )


if __name__ == "__main__":
    train_foundation_test()