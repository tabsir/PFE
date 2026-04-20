import pandas as pd
import numpy as np
import gc
import os
import shutil
import warnings
# Added Hugging Face datasets import
from datasets import load_dataset 

warnings.filterwarnings('ignore')

BASE_DIR = '/home/aka/PFE-code'
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(DATA_DIR, 'nids_transformer_split')


def build_sequence_group_ids(df):
    """Build a direction-invariant 5-tuple style group id for coherent temporal windows."""
    src_ip = df['IPV4_SRC_ADDR'].astype(str)
    dst_ip = df['IPV4_DST_ADDR'].astype(str)
    src_port = df['L4_SRC_PORT'].astype('int64')
    dst_port = df['L4_DST_PORT'].astype('int64')

    keep_forward = (src_ip < dst_ip) | ((src_ip == dst_ip) & (src_port <= dst_port))

    canonical_fields = pd.DataFrame({
        'dataset_source': df['DATASET_SOURCE'].astype(str),
        'endpoint_a_ip': np.where(keep_forward, src_ip, dst_ip),
        'endpoint_a_port': np.where(keep_forward, src_port, dst_port),
        'endpoint_b_ip': np.where(keep_forward, dst_ip, src_ip),
        'endpoint_b_port': np.where(keep_forward, dst_port, src_port),
        'protocol': df['PROTOCOL'].astype('int64'),
        'l7_proto': df['L7_PROTO'].astype('int64'),
    })

    return pd.util.hash_pandas_object(canonical_fields, index=False).astype('uint64')


def split_groups_by_first_seen(df, train_ratio=0.70, validation_ratio=0.15):
    """Hold out future groups to approximate deployment-time validation and zero-day detection."""
    group_first_seen = (
        df.groupby('sequence_group_id', sort=False)['FLOW_START_MILLISECONDS']
        .min()
        .sort_values(kind='mergesort')
    )

    ordered_groups = group_first_seen.index.to_numpy()
    total_groups = len(ordered_groups)
    train_cutoff = max(1, int(total_groups * train_ratio))
    validation_cutoff = max(train_cutoff + 1, int(total_groups * (train_ratio + validation_ratio)))
    validation_cutoff = min(validation_cutoff, total_groups - 1)

    split_ids = {
        'train': ordered_groups[:train_cutoff],
        'validation': ordered_groups[train_cutoff:validation_cutoff],
        'test': ordered_groups[validation_cutoff:],
    }

    return {
        split_name: df[df['sequence_group_id'].isin(group_ids)].copy()
        for split_name, group_ids in split_ids.items()
    }

def load_optimized_csv(file_path):
    """Reads massive CSVs in chunks and cuts RAM usage by 50% via downcasting."""
    chunk_list = []
    # Read 500,000 rows at a time
    for chunk in pd.read_csv(file_path, chunksize=500000):
        # Downcast 64-bit floats to 32-bit
        float_cols = chunk.select_dtypes(include=['float64']).columns
        chunk[float_cols] = chunk[float_cols].astype('float32')
        
        # --- DÉBUT DE LA CORRECTION ---
        # 1. Identifier toutes les colonnes int64
        int_cols = chunk.select_dtypes(include=['int64']).columns
        
        # 2. Exclure explicitement les timestamps (pour éviter l'overflow)
        # et les colonnes catégorielles (pour ne pas casser les Embedding IDs)
        columns_to_exclude = [
            'FLOW_START_MILLISECONDS', 
            'FLOW_END_MILLISECONDS',
            'L7_PROTO',
            'CLIENT_TCP_FLAGS',
            'SERVER_TCP_FLAGS',
            'ICMP_TYPE',
            'ICMP_IPV4_TYPE'
        ]
        
        # 3. Filtrer : on garde seulement les colonnes qui ne sont PAS dans la liste d'exclusion
        cols_to_downcast = [col for col in int_cols if col not in columns_to_exclude]
        
        # 4. Appliquer le downcast uniquement sur ces colonnes sécurisées
        chunk[cols_to_downcast] = chunk[cols_to_downcast].astype('int32')
        # --- FIN DE LA CORRECTION ---
        
        chunk_list.append(chunk)
        
    return pd.concat(chunk_list, ignore_index=True)

# ==========================================
# 1. Load Metadata
# ==========================================
print("1. Loading NetFlow v3 Feature Metadata...")
metadata_path = os.path.join(BASE_DIR, 'init_datasets', 'NetFlow_v3_Features.csv')
df_features = pd.read_csv(metadata_path)
print(f"Total features defined: {df_features.shape[0]}\n")

# ==========================================
# 2. Optimized Data Ingestion
# ==========================================
print("2. Initiating Memory-Optimized Data Stream from Windows...")
cic_path = os.path.join(BASE_DIR, 'init_datasets', 'NF-CICIDS2018-v3.csv')
unsw_path = os.path.join(BASE_DIR, 'init_datasets', 'NF-UNSW-NB15-v3.csv')

print("   -> Streaming CIC-IDS2018 (Canada) in chunks...")
df_cic = load_optimized_csv(cic_path)
df_cic['DATASET_SOURCE'] = 'NF-CICIDS2018-v3'
print(f"      [CIC Loaded] Shape: {df_cic.shape}")

print("   -> Streaming UNSW-NB15 (Australia) in chunks...")
df_unsw = load_optimized_csv(unsw_path)
df_unsw['DATASET_SOURCE'] = 'NF-UNSW-NB15-v3'
print(f"      [UNSW Loaded] Shape: {df_unsw.shape}")

# ==========================================
# 3. The Mega-Merge & Immediate RAM Flush
# ==========================================
print("\n3. Executing Global Merge...")
df_global = pd.concat([df_unsw, df_cic], ignore_index=True)

del df_unsw
del df_cic
gc.collect()

print(f"✅ Global Matrix Secured in RAM. Shape: {df_global.shape}")

# ---------------------------------------------------------
# Step 4.0: Build coherent temporal ordering before splitting
# ---------------------------------------------------------
print("\n4.0 Building stable conversation groups...")
df_global['sequence_group_id'] = build_sequence_group_ids(df_global)

print("4.1 Sorting by conversation group and time...")
df_global = df_global.sort_values(
    by=['sequence_group_id', 'FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS'],
    kind='mergesort'
).reset_index(drop=True)

print("4.2 Executing chronological group holdout split...")
split_frames = split_groups_by_first_seen(df_global)

parquet_dir = os.path.join(DATA_DIR, 'parquet_splits')
os.makedirs(parquet_dir, exist_ok=True)
data_files = {}
for split_name, split_df in split_frames.items():
    split_path = os.path.join(parquet_dir, f'{split_name}.parquet')
    split_df.to_parquet(split_path, engine='pyarrow', compression='snappy', index=False)
    data_files[split_name] = split_path
    print(f"   -> {split_name}: {split_df.shape}")

# Destroy the Pandas object and force garbage collection
print("Clearing Pandas from RAM...")
del df_global 
del split_frames
gc.collect()

# ---------------------------------------------------------
# Step 4.3: Memory-Mapped Loading (Zero-RAM Overhead)
# ---------------------------------------------------------
print("\n4.3 Loading split artifacts via Apache Arrow Memory-Mapping...")
split_dataset = load_dataset("parquet", data_files=data_files)

# ---------------------------------------------------------
# Step 4.4: Save Artifacts for Transformer Ingestion
# ---------------------------------------------------------
print("4.4 Saving transformer-ready artifacts...")
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
split_dataset.save_to_disk(OUTPUT_DIR)

print(f"Train set shape: {split_dataset['train'].shape}")
print(f"Validation set shape: {split_dataset['validation'].shape}")
print(f"Test set shape: {split_dataset['test'].shape}")
print("✅ Pipeline Complete. Ready for PyTorch DataLoader.")