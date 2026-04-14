import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
import numpy as np
import json
import os

class SpatioTemporalNIDSDataset(Dataset):
    def __init__(self, arrow_dir_path, stats_path="nids_normalization_stats.json", seq_len=32):
        if not os.path.exists(arrow_dir_path):
            raise FileNotFoundError(f"Dataset introuvable à : {arrow_dir_path}")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Fichier de statistiques introuvable à : {stats_path}")

        # 1. Chargement Mmap (Zéro RAM)
        self.data = load_from_disk(arrow_dir_path)
        self.seq_len = seq_len
        
        # 2. Chargement des vecteurs de normalisation
        with open(stats_path, "r") as f:
            stats = json.load(f)
            
        self.cont_cols = stats["features"] # Doit contenir 40 features
        self.mean = torch.tensor(stats["mean"], dtype=torch.float32)
        self.std = torch.tensor(stats["std"], dtype=torch.float32)
        
        # 3. Définition stricte des 9 catégories NF-v3
        self.cat_cols = [
            'PROTOCOL', 'L7_PROTO', 'TCP_FLAGS', 'L4_SRC_PORT', 'L4_DST_PORT', 
            'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS', 'ICMP_TYPE', 'ICMP_IPV4_TYPE'
        ]
        self.label_col = 'Label'
        self.attack_col = 'Attack' # Ajout utile pour l'évaluation future

        # 4. Échantillonnage par blocs (Chunking) pour éviter l'explosion combinatoire
        self.num_sequences = len(self.data) // self.seq_len

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        # Indexation par blocs stricts (0-32, 32-64, etc.)
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len
        
        window = self.data[start_idx : end_idx]
        
        # 1. Traitement des variables continues (40)
        cont_features = np.column_stack([
            np.array(window[col], dtype=np.float64) for col in self.cont_cols
        ])
        cont_features = np.nan_to_num(cont_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        cont_tensor = torch.tensor(cont_features, dtype=torch.float32)
        normalized_cont = (cont_tensor - self.mean) / self.std
        
        # 2. Traitement des variables catégorielles (9)
        cat_features = np.column_stack([
            np.array(window[col], dtype=np.int64) for col in self.cat_cols
        ])
        cat_tensor = torch.tensor(cat_features, dtype=torch.long)
        
        # 3. Label de la séquence (On prend le label du dernier paquet pour la classification)
        target_label = window[self.label_col][-1]
        
        return {
            'continuous': normalized_cont,
            'categorical': cat_tensor,
            'label': torch.tensor(target_label, dtype=torch.long)
        }

# Test d'intégration
if __name__ == "__main__":
    TRAIN_PATH = "/home/aka/PFE-code/data/nids_transformer_split/train" # À adapter selon ton arborescence
    STATS_PATH = "nids_normalization_stats.json"
    
    try:
        train_dataset = SpatioTemporalNIDSDataset(arrow_dir_path=TRAIN_PATH, stats_path=STATS_PATH, seq_len=32)
        # pin_memory=True est crucial pour la vitesse de transfert CPU -> GPU
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
        
        print(f"Nombre total de séquences d'entraînement : {len(train_dataset)}")
        
        for batch in train_loader:
            print("\n--- Analyse du premier Batch ---")
            print(f"Continuous Tensor Shape : {batch['continuous'].shape} -> [Batch(128), SeqLen(32), Cont_Vars(40)]") 
            print(f"Categorical Tensor Shape: {batch['categorical'].shape} -> [Batch(128), SeqLen(32), Cat_Vars(9)]") 
            print(f"Label Tensor Shape      : {batch['label'].shape} -> [Batch(128)]")       
            break 
            
        print("\n✅ DataLoader opérationnel. Architecture Spatio-Temporelle validée.")
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution : {e}")