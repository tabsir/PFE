import numpy as np
import json
from datasets import load_from_disk
import os

def compute_global_statistics(arrow_dir_path, batch_size=500000):
    print(f"Initialisation du calcul des statistiques globales depuis {arrow_dir_path}...")
    
    if not os.path.exists(arrow_dir_path):
        raise FileNotFoundError(f"ERREUR FATALE : Le dossier {arrow_dir_path} est introuvable.")

    dataset = load_from_disk(arrow_dir_path)
    
    all_cols = dataset.column_names
    
    # --- LA CORRECTION EST ICI ---
    # Liste unique et stricte de tout ce qui NE DOIT PAS être normalisé (Z-Score)
    # On ajoute TOUTES les catégories trouvées dans ton JSON + les dates et IPs
    exclude_cols = [
        'FLOW_START_MILLISECONDS', 
        'FLOW_END_MILLISECONDS',
        'IPV4_SRC_ADDR', 
        'IPV4_DST_ADDR',
        'Label', 
        'Attack',
        'PROTOCOL', 
        'L7_PROTO',
        'TCP_FLAGS',
        'L4_SRC_PORT',
        'L4_DST_PORT', 
        'CLIENT_TCP_FLAGS',
        'SERVER_TCP_FLAGS',
        'ICMP_TYPE',
        'ICMP_IPV4_TYPE'
    ]
    
    # Déduction dynamique : on garde uniquement les variables continues
    cont_cols = [c for c in all_cols if c not in exclude_cols]
    
    n_features = len(cont_cols)
    total_count = len(dataset)
    
    print(f"Extraction des statistiques pour exactement {n_features} features continues...")
    
    # Vecteurs d'accumulation en Float64 pour éviter l'overflow
    sum_x = np.zeros(n_features, dtype=np.float64)
    sum_sq_x = np.zeros(n_features, dtype=np.float64)
    
    # Parcours par lots
    for i in range(0, total_count, batch_size):
        batch = dataset[i : i + batch_size]
        
        batch_matrix = np.column_stack([
            np.array(batch[col], dtype=np.float64) for col in cont_cols
        ])
        
        # Nettoyage des NaNs
        batch_matrix = np.nan_to_num(batch_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        sum_x += np.sum(batch_matrix, axis=0)
        sum_sq_x += np.sum(batch_matrix ** 2, axis=0)
        
        progress = min(i + batch_size, total_count)
        print(f"   -> Progression : {progress} / {total_count} lignes traitées.")
        
    mean = sum_x / total_count
    variance = (sum_sq_x / total_count) - (mean ** 2)
    std = np.sqrt(np.maximum(variance, 1e-8)) 
    
    # Sécurité anti-division par zéro
    std = np.where(std == 0, 1.0, std)
    
    stats = {
        "features": cont_cols,
        "mean": mean.tolist(),
        "std": std.tolist()
    }
    
    output_file = "nids_normalization_stats.json"
    with open(output_file, "w") as f:
        json.dump(stats, f, indent=4)
        
    print(f"✅ Statistiques globales générées et sauvegardées dans '{output_file}'.")

if __name__ == "__main__":
    TRAIN_PATH = "/home/aka/PFE-code/data/nids_transformer_split/train" # Vérifie juste que ce chemin est bon chez toi
    compute_global_statistics(TRAIN_PATH)