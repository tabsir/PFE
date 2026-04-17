
import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler 
from tqdm import tqdm
import os
import glob
import importlib 

# 1. Chargement dynamique des modules
st_data_loader = importlib.import_module("02_st_data_loader")
stt_architecture = importlib.import_module("03_stt_architecture")

SpatioTemporalNIDSDataset = st_data_loader.SpatioTemporalNIDSDataset
SpatioTemporalTransformer = stt_architecture.SpatioTemporalTransformer

def get_progressive_ratios(epoch, warmup_epochs=10, 
                           start_mae=0.3, target_mae=0.7, 
                           start_mfm=0.1, target_mfm=0.4):
    """Calcule l'augmentation de la difficulté pour le Curriculum Learning."""
    if epoch < warmup_epochs:
        progress = epoch / warmup_epochs
        current_mae = start_mae + progress * (target_mae - start_mae)
        current_mfm = start_mfm + progress * (target_mfm - start_mfm)
    else:
        current_mae, current_mfm = target_mae, target_mfm
    return current_mae, current_mfm

def get_last_checkpoint(checkpoint_dir):
    """Recherche le fichier .pt avec l'époque la plus élevée pour reprendre l'entraînement."""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "stt_epoch_*.pt"))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return checkpoints[-1]

def train_foundation():
    # --- 1. Configuration & Hyperparamètres ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 50
    WARMUP_EPOCHS = 10
    BATCH_SIZE = 512       
    ACC_STEPS = 1  # Gradient Accumulation Steps         
    LR = 1e-4
    SEQ_LEN = 32
    
    TRAIN_DIR = "/home/aka/PFE-code/data/nids_transformer_split/train"
    STATS_PATH = "nids_normalization_stats.json"
    CHECKPOINT_DIR = "./checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # --- 2. Initialisation des Données ---
    train_dataset = SpatioTemporalNIDSDataset(arrow_dir_path=TRAIN_DIR, stats_path=STATS_PATH, seq_len=SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=10, pin_memory=True)

    # --- 3. Initialisation du Modèle & Optimisation ---
    NUM_CONT = len(train_dataset.cont_cols)
    CAT_VOCABS = [256, 256, 256, 65536, 65536, 256, 256, 256, 256]   
    
    model = SpatioTemporalTransformer(
        num_cont_features=NUM_CONT, 
        cat_vocab_sizes=CAT_VOCABS,
        seq_len=SEQ_LEN,
        init_mae=0.3, 
        init_mfm=0.1
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.MSELoss(reduction='none')
    
    # LR Scheduler: linear warmup then cosine decay
    total_steps = EPOCHS * (len(train_dataset) // BATCH_SIZE + 1)
    warmup_steps = WARMUP_EPOCHS * (len(train_dataset) // BATCH_SIZE + 1)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # --- 4. Logique de Reprise Automatique (Checkpointing) ---
    start_epoch = 0
    last_cp = get_last_checkpoint(CHECKPOINT_DIR)
    
    if last_cp:
        print(f"🔄 Checkpoint trouvé : {last_cp}. Chargement en cours...")
        checkpoint = torch.load(last_cp, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        #scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f" Reprise prête à partir de l'époque {start_epoch + 1}/{EPOCHS}")
    else:
        print(f" Aucun checkpoint trouvé. Début d'un nouvel entraînement sur {DEVICE}.")

    # --- 5. Boucle d'Entraînement Principale ---
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        
        # Application du Curriculum Learning (Manquant dans le code 2, rajouté ici)
        mae_r, mfm_r = get_progressive_ratios(epoch, WARMUP_EPOCHS)
        model.mae_mask_ratio = mae_r
        model.mfm_layer.mask_ratio = mfm_r
        
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        optimizer.zero_grad()

        for i, batch in progress_bar:
            cont = batch['continuous'].to(DEVICE, non_blocking=True)
            cat = batch['categorical'].to(DEVICE, non_blocking=True)
            
            # 1. Plus de 'with autocast()', calcul direct en FP32
            reconstructed, spatial_mask = model(cont, cat)
            raw_loss = criterion(reconstructed, cont)
            mask_expanded = spatial_mask.unsqueeze(-1)  # [B, S, 1]
            masked_loss = (raw_loss * mask_expanded).sum() / mask_expanded.sum().clamp(min=1) / raw_loss.shape[-1]
            loss = masked_loss / ACC_STEPS

            # 2. Rétropropagation directe (Plus de scaler.scale)
            loss.backward()

            if (i + 1) % ACC_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * ACC_STEPS
            if i % 10 == 0:
                progress_bar.set_postfix({
                    "loss": f"{epoch_loss/(i+1):.4f}", 
                    "MAE": f"{mae_r:.2f}",
                    "MFM": f"{mfm_r:.2f}"
                })

        # Sauvegarde complète à la fin de l'époque
        checkpoint_path = f"{CHECKPOINT_DIR}/stt_epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': epoch_loss / len(train_loader),
        }, checkpoint_path)
        
        
        print(f" Epoch {epoch+1} complete. Loss: {epoch_loss/len(train_loader):.6f} | Checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    train_foundation()
    