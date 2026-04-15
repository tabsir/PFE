import pandas as pd
import numpy as np
import gc
import warnings
# Added Hugging Face datasets import
from datasets import load_dataset 

warnings.filterwarnings('ignore')

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
metadata_path = '/home/aka/PFE-code/init_datasets/NetFlow_v3_Features.csv'
df_features = pd.read_csv(metadata_path)
print(f"Total features defined: {df_features.shape[0]}\n")

# ==========================================
# 2. Optimized Data Ingestion
# ==========================================
print("2. Initiating Memory-Optimized Data Stream from Windows...")
cic_path  ='/home/aka/PFE-code/init_datasets/NF-CICIDS2018-v3.csv'
unsw_path = '/home/aka/PFE-code/init_datasets/UNSW-NB15-v3.csv'

print("   -> Streaming CIC-IDS2018 (Canada) in chunks...")
df_cic = load_optimized_csv(cic_path)
print(f"      [CIC Loaded] Shape: {df_cic.shape}")

print("   -> Streaming UNSW-NB15 (Australia) in chunks...")
df_unsw = load_optimized_csv(unsw_path)
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
# Step 4.0: Serialize and Nuke from RAM
# ---------------------------------------------------------
print("\n4.0 Flushing global matrix to Parquet...")
# FIX: Changed 'df' to 'df_global'
df_global.to_parquet("global_nids_matrix.parquet", engine="pyarrow", compression="snappy")

# Destroy the Pandas object and force garbage collection
print("Clearing Pandas from RAM...")
# FIX: Changed 'df' to 'df_global'
del df_global 
gc.collect()

# ---------------------------------------------------------
# Step 4.1: Memory-Mapped Loading (Zero-RAM Overhead)
# ---------------------------------------------------------
print("\n4.1 Loading via Apache Arrow Memory-Mapping...")
dataset = load_dataset("parquet", data_files="global_nids_matrix.parquet", split="train")

# ---------------------------------------------------------
# Step 4.2: Execute the Hybrid Split Out-of-Core
# ---------------------------------------------------------
print("4.2 Executing Out-of-Core Split...")
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

# ---------------------------------------------------------
# Step 4.3: Save Artifacts for Transformer Ingestion
# ---------------------------------------------------------
print("4.3 Saving transformer-ready artifacts...")
split_dataset.save_to_disk("./nids_transformer_split")

print(f"Train set shape: {split_dataset['train'].shape}")
print(f"Test set shape: {split_dataset['test'].shape}")
print("✅ Pipeline Complete. Ready for PyTorch DataLoader.")