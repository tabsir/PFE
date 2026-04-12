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
            raise FileNotFoundError(f"Fichier de statistiques introuvable à : {stats_path}. Exécute compute_stats.py d'abord.")

        self.data = load_from_disk(arrow_dir_path)
        self.seq_len = seq_len
        
        # 1. Chargement des vecteurs de normalisation (Dépendance externe)
        with open(stats_path, "r") as f:
            stats = json.load(f)
            
        self.cont_cols = stats["features"]
        self.mean = torch.tensor(stats["mean"], dtype=torch.float32)
        self.std = torch.tensor(stats["std"], dtype=torch.float32)
        
        # 2. Définition stricte des autres colonnes
        self.cat_cols = ['PROTOCOL', 'L4_DST_PORT', 'TCP_FLAGS', 'L4_SRC_PORT']
        self.label_col = 'Label'

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        window = self.data[idx : idx + self.seq_len]
        
        # 1. Extraction et sécurisation des données continues
        # On force le type float64 pour gérer les éventuels 'None' qui deviennent des 'NaN'
        cont_features = np.column_stack([
            np.array(window[col], dtype=np.float64) for col in self.cont_cols
        ])
        
        # 2. Nettoyage des NaNs (remplacement par 0.0)
        cont_features = np.nan_to_num(cont_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 3. Conversion en tenseur et normalisation
        cont_tensor = torch.tensor(cont_features, dtype=torch.float32)
        normalized_cont = (cont_tensor - self.mean) / self.std
        
        # 4. Extraction et sécurisation des catégories
        # On s'assure que les catégories sont bien des entiers (long)
        cat_features = np.column_stack([
            np.array(window[col], dtype=np.int64) for col in self.cat_cols
        ])
        
        # 5. Extraction de la cible finale
        target_label = window[self.label_col][-1]
        
        return {
            'continuous': normalized_cont,
            'categorical': torch.tensor(cat_features, dtype=torch.long),
            'label': torch.tensor(target_label, dtype=torch.long)
        }

# Test d'intégration complet et sans approximation
if __name__ == "__main__":
    TRAIN_PATH = "/home/aka/PFE-code/data/nids_transformer_split/train"
    STATS_PATH = "nids_normalization_stats.json"
    
    try:
        train_dataset = SpatioTemporalNIDSDataset(arrow_dir_path=TRAIN_PATH, stats_path=STATS_PATH, seq_len=32)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        
        for batch in train_loader:
            print(f"Batch Continuous Shape : {batch['continuous'].shape} (Attendu: [32, 32, 47])") 
            print(f"Batch Categorical Shape: {batch['categorical'].shape} (Attendu: [32, 32, 4])") 
            print(f"Batch Label Shape      : {batch['label'].shape} (Attendu: [32])")       
            break 
        print("✅ DataLoader opérationnel et normalisation Z-Score appliquée.")
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution : {e}")