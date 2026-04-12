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
        
        # Downcast 64-bit ints to 32-bit
        int_cols = chunk.select_dtypes(include=['int64']).columns
        chunk[int_cols] = chunk[int_cols].astype('int32')
        
        chunk_list.append(chunk)
        
    return pd.concat(chunk_list, ignore_index=True)

# ==========================================
# 1. Load Metadata
# ==========================================
print("1. Loading NetFlow v3 Feature Metadata...")
metadata_path = '/mnt/c/Users/HP/Desktop/PFE/Downloads/NF-CSE-CIC-IDS-2018-v3/f78acbaa2afe1595_NFV3DATA-A11964_A11964/data/NetFlow_v3_Features.csv' 
df_features = pd.read_csv(metadata_path)
print(f"Total features defined: {df_features.shape[0]}\n")

# ==========================================
# 2. Optimized Data Ingestion
# ==========================================
print("2. Initiating Memory-Optimized Data Stream from Windows...")
cic_path  = '/mnt/c/Users/HP/Desktop/PFE/Downloads/NF-CSE-CIC-IDS-2018-v3/f78acbaa2afe1595_NFV3DATA-A11964_A11964/data/NF-CICIDS2018-v3.csv'
unsw_path = '/mnt/c/Users/HP/Desktop/PFE/Downloads/NF-UNSW-NB15-v3/f7546561558c07c5_NFV3DATA-A11964_A11964/data/NF-UNSW-NB15-v3.csv'

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

# ⚠️ Forcibly destroy the old variables and run Garbage Collection
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