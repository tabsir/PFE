
import sys
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
    print(f"Chargement du module local '{module_name}' depuis {filename}...")
    module_path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


st_data_loader = load_local_module("st_data_loader", "02_st_data_loader.py")
stt_architecture = load_local_module("stt_architecture", "03_stt_architecture.py")

SpatioTemporalNIDSDataset = st_data_loader.SpatioTemporalNIDSDataset
SpatioTemporalTransformer = stt_architecture.SpatioTemporalTransformer

# def get_progressive_ratios(epoch):
  #  if epoch == 0:
   #      return 0.15, 0.00       # Phase 1: spatial only
   #  elif 1 <= epoch <= 5:
   #      return 0.15, 0.05       # Phase 2: introduce MFM gently
   #  elif 6 <= epoch < 15:
   #      progress = (epoch - 5) / (15 - 5)
   #      current_mae = 0.15 + progress * (0.40 - 0.15)
    #     current_mfm = 0.05 + progress * (0.15 - 0.05)
   #      return current_mae, current_mfm
    # else:
       #  return 0.40, 0.15  
def get_progressive_ratios(epoch):
    """
    Curriculum progressif pour stabiliser l'auto-encodage avant de durcir le masquage.
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
        mae = 0.20 + progress * (0.30 - 0.20)
        mfm = 0.03 + progress * (0.08 - 0.03)
        return mae, mfm

    progress = min((epoch - 30) / 20, 1.0)
    mae = 0.30 + progress * (0.35 - 0.30)
    mfm = 0.08 + progress * (0.10 - 0.08)
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


def build_fixed_validation_mask(batch_size, seq_len, mask_ratio, device, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return (torch.rand(batch_size, seq_len, generator=generator) < mask_ratio).to(device)


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


def compute_binary_pr_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    positive_count = int((labels == 1).sum())
    if labels.size == 0 or positive_count == 0:
        return float('nan')

    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels == 1)
    false_positives = np.cumsum(sorted_labels == 0)

    precision = true_positives / np.maximum(true_positives + false_positives, 1)
    recall = true_positives / positive_count
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return float(np.trapezoid(precision, recall))


def _metric_for_selection(metric_value):
    if metric_value is None or math.isnan(metric_value):
        return float('-inf')
    return float(metric_value)


def is_better_checkpoint(validation_metrics, best_metrics, epsilon=1e-6):
    current_combined_masked_mse = float(validation_metrics['combined_masked_mse'])
    best_combined_masked_mse = float(best_metrics.get('combined_masked_mse', float('inf')))
    if current_combined_masked_mse < best_combined_masked_mse - epsilon:
        return True
    if math.isclose(current_combined_masked_mse, best_combined_masked_mse, rel_tol=0.0, abs_tol=epsilon):
        current_pr_auc = _metric_for_selection(validation_metrics.get('pr_auc'))
        best_pr_auc = _metric_for_selection(best_metrics.get('pr_auc'))
        if current_pr_auc > best_pr_auc + epsilon:
            return True
        if math.isclose(current_pr_auc, best_pr_auc, rel_tol=0.0, abs_tol=epsilon):
            current_auc = _metric_for_selection(validation_metrics.get('anomaly_auc'))
            best_auc = _metric_for_selection(best_metrics.get('anomaly_auc'))
            if current_auc > best_auc + epsilon:
                return True
            if math.isclose(current_auc, best_auc, rel_tol=0.0, abs_tol=epsilon):
                current_masked_mse = float(validation_metrics['masked_mse'])
                best_masked_mse = float(best_metrics.get('masked_mse', float('inf')))
                return current_masked_mse < best_masked_mse - epsilon
    return False


def compute_best_f1_metrics(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    if labels.size == 0 or scores.size == 0 or np.unique(labels).size < 2:
        return {
            'best_f1': float('nan'),
            'best_precision': float('nan'),
            'best_recall': float('nan'),
            'best_threshold': float('nan'),
        }

    candidate_thresholds = np.unique(scores)
    candidate_thresholds = np.concatenate(([scores.max() + 1e-12], candidate_thresholds))

    best_f1 = float('-inf')
    best_precision = float('nan')
    best_recall = float('nan')
    best_threshold = float('nan')

    positive_mask = labels == 1
    for threshold in candidate_thresholds:
        predicted_positive = scores >= threshold
        tp = int(np.logical_and(predicted_positive, positive_mask).sum())
        fp = int(np.logical_and(predicted_positive, ~positive_mask).sum())
        fn = int(np.logical_and(~predicted_positive, positive_mask).sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            best_threshold = float(threshold)

    return {
        'best_f1': float(best_f1),
        'best_precision': float(best_precision),
        'best_recall': float(best_recall),
        'best_threshold': best_threshold,
    }


def evaluate(model, data_loader, device, mae_mask_ratio, validation_mask_seed=None, apply_mfm=False):
    model.eval()
    metrics = {
        'loss': 0.0,
        'masked_mse': 0.0,
        'full_mse': 0.0,
    }
    anomaly_scores = []
    anomaly_labels = []

    with torch.no_grad():
        for batch_index, batch in enumerate(tqdm(data_loader, total=len(data_loader), desc='Validation', leave=False, file=sys.stdout)):
            cont = batch['continuous'].to(device, non_blocking=True)
            cat = batch['categorical'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)

            if validation_mask_seed is None:
                validation_mask = build_validation_mask(cont.shape[0], cont.shape[1], mae_mask_ratio, device)
            else:
                validation_mask = build_fixed_validation_mask(
                    cont.shape[0],
                    cont.shape[1],
                    mae_mask_ratio,
                    device,
                    seed=validation_mask_seed + batch_index,
                )
            reconstructed, spatial_mask = model(cont, cat, spatial_mask=validation_mask, apply_mfm=apply_mfm)
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
    metrics['pr_auc'] = compute_binary_pr_auc(labels, scores)

    metrics.update(compute_best_f1_metrics(labels, scores))

    return metrics


def combine_validation_metrics(mae_metrics, mfm_metrics):
    combined = dict(mae_metrics)
    combined['mae_metrics'] = mae_metrics
    combined['mfm_metrics'] = mfm_metrics
    combined['combined_masked_mse'] = float(mae_metrics['masked_mse'] + mfm_metrics['masked_mse'])
    combined['combined_loss'] = float(mae_metrics['loss'] + mfm_metrics['loss'])
    combined['mfm_masked_mse'] = float(mfm_metrics['masked_mse'])
    combined['mfm_full_mse'] = float(mfm_metrics['full_mse'])
    combined['mfm_loss'] = float(mfm_metrics['loss'])
    combined['mfm_anomaly_auc'] = float(mfm_metrics['anomaly_auc'])
    combined['mfm_pr_auc'] = float(mfm_metrics['pr_auc'])
    combined['mfm_best_f1'] = float(mfm_metrics['best_f1'])
    combined['mfm_best_precision'] = float(mfm_metrics['best_precision'])
    combined['mfm_best_recall'] = float(mfm_metrics['best_recall'])
    combined['mfm_best_threshold'] = float(mfm_metrics['best_threshold'])
    return combined


def evaluate_both_masks(model, data_loader, device, mae_mask_ratio, mfm_mask_ratio, mae_seed, mfm_seed):
    mae_metrics = evaluate(
        model,
        data_loader,
        device,
        mae_mask_ratio,
        validation_mask_seed=mae_seed,
        apply_mfm=False,
    )
    mfm_metrics = evaluate(
        model,
        data_loader,
        device,
        mfm_mask_ratio,
        validation_mask_seed=mfm_seed,
        apply_mfm=True,
    )
    return combine_validation_metrics(mae_metrics, mfm_metrics)

def train_foundation():
    print("running train_foundation()...")
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
    INITIAL_TRAIN_MAE_MASK_RATIO, INITIAL_TRAIN_MFM_MASK_RATIO = get_progressive_ratios(0)
    VALIDATION_MAE_MASK_RATIO = 0.30
    VALIDATION_MFM_MASK_RATIO = 0.10
    VALIDATION_MAE_MASK_SEED = 42
    VALIDATION_MFM_MASK_SEED = 142
    
    TRAIN_DIR = "/home/aka/PFE-code/NEW/data/nids_src_grouped/train"
    VALID_DIR = "/home/aka/PFE-code/NEW/data/nids_src_grouped/validation"
    TEST_DIR  = "/home/aka/PFE-code/NEW/data/nids_src_grouped/test"
    STATS_PATH = "/home/aka/PFE-code/NEW/nids_normalization_stats.json"
    CHECKPOINT_DIR = "/home/aka/PFE-code/NEW/checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # --- 2. Initialisation des Données ---
    validation_path = VALID_DIR if os.path.exists(VALID_DIR) else TEST_DIR
    if validation_path == TEST_DIR:
        print('Validation split introuvable. Utilisation temporaire du split test comme validation.')

    train_dataset = SpatioTemporalNIDSDataset(
        arrow_dir_path=TRAIN_DIR,
        stats_path=STATS_PATH,
        seq_len=SEQ_LEN,
        stride=16,
        clip_value=CLIP_VALUE,
    )
    validation_dataset = SpatioTemporalNIDSDataset(
        arrow_dir_path=validation_path,
        stats_path=STATS_PATH,
        seq_len=SEQ_LEN,
        stride=16,
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
        init_mae=INITIAL_TRAIN_MAE_MASK_RATIO,
        init_mfm=INITIAL_TRAIN_MFM_MASK_RATIO
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
    best_metrics = None
    last_cp = get_last_checkpoint(CHECKPOINT_DIR)
    
    if last_cp:
        print(f"🔄 Checkpoint trouvé : {last_cp}. Chargement en cours...")
        checkpoint = torch.load(last_cp, map_location=DEVICE)
        if checkpoint.get('data_signature') == DATA_SIGNATURE:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_metrics = {
                'masked_mse': checkpoint.get('best_val_masked_mse', float('inf')),
                'mfm_masked_mse': checkpoint.get('best_mfm_masked_mse', checkpoint.get('best_val_masked_mse', float('inf'))),
                'combined_masked_mse': checkpoint.get('best_combined_masked_mse', checkpoint.get('best_val_masked_mse', float('inf'))),
                'anomaly_auc': checkpoint.get('best_anomaly_auc', float('-inf')),
                'pr_auc': checkpoint.get('best_pr_auc_running', checkpoint.get('pr_auc', float('nan'))),
                'best_f1': checkpoint.get('best_f1_running', checkpoint.get('best_f1', float('nan'))),
                'best_precision': checkpoint.get('best_precision_running', checkpoint.get('best_precision', float('nan'))),
                'best_recall': checkpoint.get('best_recall_running', checkpoint.get('best_recall', float('nan'))),
                'best_threshold': checkpoint.get('best_threshold_running', checkpoint.get('best_threshold', float('nan'))),
                'mfm_anomaly_auc': checkpoint.get('mfm_anomaly_auc', float('nan')),
                'mfm_pr_auc': checkpoint.get('mfm_pr_auc', float('nan')),
            }
            print(f" Reprise prête à partir de l'époque {start_epoch + 1}/{EPOCHS}")
        else:
            print(' Checkpoint ignoré: il provient d\'une ancienne configuration de données.')
    else:
        print(f" Aucun checkpoint trouvé. Début d'un nouvel entraînement sur {DEVICE}.")

    # --- 5. Boucle d'Entraînement Principale ---
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        
        # Progressive MAE + MFM corruption during every training epoch.
        mae_r, mfm_r = get_progressive_ratios(epoch)
        model.mae_mask_ratio = mae_r
        model.mfm_layer.mask_ratio = mfm_r
        
        
        epoch_loss = 0.0
        epoch_masked_mse = 0.0
        epoch_full_mse = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}", file=sys.stdout)
        
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
                    "MFM": f"{mfm_r:.2f}"
                })

        validation_metrics = evaluate_both_masks(
            model,
            validation_loader,
            DEVICE,
            VALIDATION_MAE_MASK_RATIO,
            VALIDATION_MFM_MASK_RATIO,
            VALIDATION_MAE_MASK_SEED,
            VALIDATION_MFM_MASK_SEED,
        )
        train_loss = epoch_loss / len(train_loader)
        train_masked_mse = epoch_masked_mse / len(train_loader)
        train_full_mse = epoch_full_mse / len(train_loader)

        if best_metrics is None:
            is_best_checkpoint = True
            best_metrics = validation_metrics
        else:
            is_best_checkpoint = is_better_checkpoint(validation_metrics, best_metrics)

            if is_best_checkpoint:
                best_metrics = validation_metrics

        current_best_val_masked_mse = best_metrics['masked_mse']
        current_best_mfm_masked_mse = best_metrics.get('mfm_masked_mse', float('inf'))
        current_best_combined_masked_mse = best_metrics.get('combined_masked_mse', float('inf'))
        current_best_anomaly_auc = best_metrics['anomaly_auc']
        current_best_pr_auc = best_metrics.get('pr_auc', float('nan'))
        current_best_f1 = best_metrics.get('best_f1', float('nan'))
        current_best_precision = best_metrics.get('best_precision', float('nan'))
        current_best_recall = best_metrics.get('best_recall', float('nan'))
        current_best_threshold = best_metrics.get('best_threshold', float('nan'))

        # Sauvegarde complète à la fin de l'époque
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
            'val_mfm_loss': validation_metrics['mfm_loss'],
            'val_combined_loss': validation_metrics['combined_loss'],
            'val_masked_mse': validation_metrics['masked_mse'],
            'val_mfm_masked_mse': validation_metrics['mfm_masked_mse'],
            'val_combined_masked_mse': validation_metrics['combined_masked_mse'],
            'val_full_mse': validation_metrics['full_mse'],
            'val_mfm_full_mse': validation_metrics['mfm_full_mse'],
            'anomaly_auc': validation_metrics['anomaly_auc'],
            'mfm_anomaly_auc': validation_metrics['mfm_anomaly_auc'],
            'pr_auc': validation_metrics['pr_auc'],
            'mfm_pr_auc': validation_metrics['mfm_pr_auc'],
            'best_f1': validation_metrics['best_f1'],
            'best_precision': validation_metrics['best_precision'],
            'best_recall': validation_metrics['best_recall'],
            'best_threshold': validation_metrics['best_threshold'],
            'mfm_best_f1': validation_metrics['mfm_best_f1'],
            'mfm_best_precision': validation_metrics['mfm_best_precision'],
            'mfm_best_recall': validation_metrics['mfm_best_recall'],
            'mfm_best_threshold': validation_metrics['mfm_best_threshold'],
            'best_val_masked_mse': current_best_val_masked_mse,
            'best_mfm_masked_mse': current_best_mfm_masked_mse,
            'best_combined_masked_mse': current_best_combined_masked_mse,
            'best_anomaly_auc': current_best_anomaly_auc,
            'best_pr_auc_running': current_best_pr_auc,
            'best_f1_running': current_best_f1,
            'best_precision_running': current_best_precision,
            'best_recall_running': current_best_recall,
            'best_threshold_running': current_best_threshold,
            'train_mae_mask_ratio': mae_r,
            'train_mfm_mask_ratio': mfm_r,
            'validation_mae_mask_ratio': VALIDATION_MAE_MASK_RATIO,
            'validation_mfm_mask_ratio': VALIDATION_MFM_MASK_RATIO,
            'data_signature': DATA_SIGNATURE,
        }, checkpoint_path)

        if is_best_checkpoint:
            best_checkpoint_path = os.path.join(CHECKPOINT_DIR, 'stt_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_masked_mse': validation_metrics['masked_mse'],
                'val_mfm_masked_mse': validation_metrics['mfm_masked_mse'],
                'val_combined_masked_mse': validation_metrics['combined_masked_mse'],
                'val_full_mse': validation_metrics['full_mse'],
                'val_mfm_full_mse': validation_metrics['mfm_full_mse'],
                'anomaly_auc': validation_metrics['anomaly_auc'],
                'mfm_anomaly_auc': validation_metrics['mfm_anomaly_auc'],
                'pr_auc': validation_metrics['pr_auc'],
                'mfm_pr_auc': validation_metrics['mfm_pr_auc'],
                'best_f1': validation_metrics['best_f1'],
                'best_precision': validation_metrics['best_precision'],
                'best_recall': validation_metrics['best_recall'],
                'best_threshold': validation_metrics['best_threshold'],
                'mfm_best_f1': validation_metrics['mfm_best_f1'],
                'mfm_best_precision': validation_metrics['mfm_best_precision'],
                'mfm_best_recall': validation_metrics['mfm_best_recall'],
                'mfm_best_threshold': validation_metrics['mfm_best_threshold'],
                'train_mae_mask_ratio': mae_r,
                'train_mfm_mask_ratio': mfm_r,
                'validation_mae_mask_ratio': VALIDATION_MAE_MASK_RATIO,
                'validation_mfm_mask_ratio': VALIDATION_MFM_MASK_RATIO,
                'data_signature': DATA_SIGNATURE,
            }, best_checkpoint_path)
        
        print(
            f" Epoch {epoch+1} complete. "
            f"TrainLoss: {train_loss:.6f} | TrainMaskedMSE: {train_masked_mse:.6f} | "
            f"ValMAE_MaskedMSE: {validation_metrics['masked_mse']:.6f} | "
            f"ValMFM_MaskedMSE: {validation_metrics['mfm_masked_mse']:.6f} | "
            f"ValCombinedMaskedMSE: {validation_metrics['combined_masked_mse']:.6f} | "
            f"ValFullMSE: {validation_metrics['full_mse']:.6f} | "
            f"PRAUC: {validation_metrics['pr_auc']:.4f} | "
            f"MFM_PRAUC: {validation_metrics['mfm_pr_auc']:.4f} | "
            f"AUC: {validation_metrics['anomaly_auc']:.4f} | "
            f"F1: {validation_metrics['best_f1']:.4f} | "
            f"MaskRatio Train/Val(MAE/MFM): {mae_r:.2f}/{mfm_r:.2f} | {VALIDATION_MAE_MASK_RATIO:.2f}/{VALIDATION_MFM_MASK_RATIO:.2f} | Checkpoint: {checkpoint_path}"
        )
        print(
            f" Validation score gap -> benign: {validation_metrics['benign_score_mean']:.6f}, "
            f"attack: {validation_metrics['attack_score_mean']:.6f}"
        )

if __name__ == "__main__":
    train_foundation()
    