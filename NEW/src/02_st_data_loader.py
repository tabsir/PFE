import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
import numpy as np
import json
import os


class SpatioTemporalNIDSDataset(Dataset):
    def __init__(self, arrow_dir_path, stats_path="nids_normalization_stats.json", seq_len=32, stride=None, clip_value=5.0):
        if not os.path.exists(arrow_dir_path):
            raise FileNotFoundError(f"Dataset introuvable à : {arrow_dir_path}")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Fichier de statistiques introuvable à : {stats_path}")

        # 1. Chargement Mmap (Zéro RAM)
        self.data = load_from_disk(arrow_dir_path)
        self.seq_len = seq_len
        self.stride = stride or seq_len
        self.clip_value = clip_value
        self.sequence_cache_path = os.path.join(
            arrow_dir_path,
            f"sequence_ranges_seq{self.seq_len}_stride{self.stride}.npy"
        )
        
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
        self.group_col = 'sequence_group_id' if 'sequence_group_id' in self.data.column_names else None
        self.start_time_col = 'FLOW_START_MILLISECONDS'
        self.end_time_col = 'FLOW_END_MILLISECONDS'

        # 4. Construction de fenêtres cohérentes sans traverser les groupes temporels
        self.sequence_ranges = self._load_or_build_sequence_ranges()
        self.num_sequences = len(self.sequence_ranges)

    def _load_or_build_sequence_ranges(self):
        if os.path.exists(self.sequence_cache_path):
            return np.load(self.sequence_cache_path)

        sequence_ranges = np.asarray(self._build_sequence_ranges(), dtype=np.int64)
        np.save(self.sequence_cache_path, sequence_ranges)
        return sequence_ranges

    def _expand_group_ranges(self, group_start, group_end):
        group_length = group_end - group_start
        if group_length < self.seq_len:
            return []

        max_start = group_end - self.seq_len + 1
        return [
            (start_idx, start_idx + self.seq_len)
            for start_idx in range(group_start, max_start, self.stride)
        ]

    def _build_sequence_ranges(self, scan_batch_size=1_000_000):
        if self.group_col is None:
            return [
                (start_idx, start_idx + self.seq_len)
                for start_idx in range(0, len(self.data) - self.seq_len + 1, self.stride)
            ]

        sequence_ranges = []
        current_group_start = 0
        previous_last_group = None
        total_rows = len(self.data)
        n_batches = (total_rows + scan_batch_size - 1) // scan_batch_size
        print(f"  Building sequence ranges: {total_rows:,} rows in {n_batches} batches ...", flush=True)

        for batch_start in range(0, len(self.data), scan_batch_size):
            batch_end = min(batch_start + scan_batch_size, len(self.data))
            batch_num = batch_start // scan_batch_size + 1
            print(f"  [batch {batch_num}/{n_batches}] rows {batch_start:,}–{batch_end:,} ...", flush=True)
            group_batch = np.asarray(self.data[batch_start:batch_end][self.group_col], dtype=np.uint64)
            if group_batch.size == 0:
                continue

            if previous_last_group is not None and group_batch[0] != previous_last_group:
                sequence_ranges.extend(self._expand_group_ranges(current_group_start, batch_start))
                current_group_start = batch_start

            change_offsets = np.flatnonzero(group_batch[1:] != group_batch[:-1]) + 1
            for change_offset in change_offsets:
                group_end = batch_start + int(change_offset)
                sequence_ranges.extend(self._expand_group_ranges(current_group_start, group_end))
                current_group_start = group_end

            previous_last_group = group_batch[-1]

        if previous_last_group is not None:
            sequence_ranges.extend(self._expand_group_ranges(current_group_start, len(self.data)))

        print(f"  Sequence ranges built: {len(sequence_ranges):,} sequences. Saving cache ...", flush=True)
        return sequence_ranges

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start_idx, end_idx = self.sequence_ranges[idx].tolist()
        
        window = self.data[start_idx : end_idx]
        
        # 1. Traitement des variables continues (40)
        cont_features = np.column_stack([
            np.array(window[col], dtype=np.float64) for col in self.cont_cols
        ])
        cont_features = np.nan_to_num(cont_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Log-transform to handle heavy-tailed network flow distributions
        cont_features = np.sign(cont_features) * np.log1p(np.abs(cont_features))
        
        cont_tensor = torch.tensor(cont_features, dtype=torch.float32)
        normalized_cont = (cont_tensor - self.mean) / self.std
        normalized_cont = torch.clamp(normalized_cont, min=-self.clip_value, max=self.clip_value)
        
        # 2. Traitement des variables catégorielles (9)
        cat_features = np.column_stack([
            np.array(window[col], dtype=np.int64) for col in self.cat_cols
        ])
        cat_tensor = torch.tensor(cat_features, dtype=torch.long)
        
        # 3. Label de séquence pour le NIDS: si une attaque apparaît dans la fenêtre, la séquence est malveillante.
        window_labels = np.array(window[self.label_col], dtype=np.int64)
        target_label = int(window_labels.max())
        attack_names = list(window[self.attack_col])
        active_attacks = [name for name, label in zip(attack_names, window_labels) if label != 0 and name != 'Benign']
        target_attack = active_attacks[-1] if active_attacks else 'Benign'
        
        return {
            'continuous': normalized_cont,
            'categorical': cat_tensor,
            'label': torch.tensor(target_label, dtype=torch.long),
            'attack': target_attack,
            'start_time': torch.tensor(window[self.start_time_col][0], dtype=torch.long),
            'end_time': torch.tensor(window[self.end_time_col][-1], dtype=torch.long)
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
            print(f"Attack Example          : {batch['attack'][0]}")
            break 
            
        print("\n DataLoader opérationnel. Architecture Spatio-Temporelle validée.")
    except Exception as e:
        print(f" Erreur lors de l'exécution : {e}")