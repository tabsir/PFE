"""
00_regroup_by_src_ip.py — Re-group NetFlow data by Source IP for NIDS sequences
================================================================================
The original ETL groups rows by 5-tuple flow hash.  Most flows have only 1–2
rows, so seq_len=32 windows cannot be formed.

This script re-sorts each split (train / validation / test) by
  (IPV4_SRC_ADDR, FLOW_START_MILLISECONDS)
and assigns a new sequence_group_id = hash(IPV4_SRC_ADDR) so the data loader
builds 32-flow "host behaviour" windows instead of per-flow windows.

The chronological train/val/test split boundaries are preserved — only the
internal ordering within each split changes.

Output: data/nids_src_grouped/{train,validation,test}/
"""

import os
import gc
import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_from_disk, Dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path("/home/aka/PFE-code")
INPUT_BASE  = BASE_DIR / "OLD" / "data" / "nids_transformer_split"
OUTPUT_BASE = BASE_DIR / "OLD" / "data" / "nids_src_grouped"

SPLITS = ["train", "validation", "test"]

# How many rows to process at once when sorting (tune to available RAM).
# At ~200 bytes/row for 17 M rows ≈ 3.4 GB; 4 M chunk = ~800 MB.
SORT_CHUNK_ROWS = None   # None = load entire split at once (simplest; needs ~4 GB RAM)


def ip_str_to_uint64(ip_series: pd.Series) -> pd.Series:
    """
    Convert dotted-decimal IPv4 strings to uint64 deterministically.
    Uses a stable hash so the same IP always maps to the same group id.
    """
    def _hash(ip_str: str) -> int:
        h = hashlib.md5(ip_str.encode(), usedforsecurity=False).digest()
        # Take first 8 bytes as little-endian uint64
        return int.from_bytes(h[:8], "little")

    return ip_series.map(_hash).astype(np.uint64)


def regroup_split(split_name: str):
    input_dir  = INPUT_BASE / split_name
    output_dir = OUTPUT_BASE / split_name

    if not input_dir.exists():
        print(f"[{split_name}] Input directory not found: {input_dir}. Skipping.")
        return

    if output_dir.exists():
        print(f"[{split_name}] Output already exists at {output_dir}. Removing and rebuilding.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{split_name}] Loading from {input_dir} ...", flush=True)
    hf_dataset = load_from_disk(str(input_dir))
    total_rows = len(hf_dataset)
    print(f"[{split_name}] {total_rows:,} rows loaded.", flush=True)

    print(f"[{split_name}] Converting to pandas ...", flush=True)
    df = hf_dataset.to_pandas()
    del hf_dataset
    gc.collect()

    # Ensure required columns exist
    for col in ("IPV4_SRC_ADDR", "FLOW_START_MILLISECONDS"):
        if col not in df.columns:
            raise ValueError(f"[{split_name}] Required column '{col}' not found. "
                             f"Columns present: {list(df.columns)}")

    print(f"[{split_name}] Reassigning sequence_group_id by source IP ...", flush=True)
    df["sequence_group_id"] = ip_str_to_uint64(df["IPV4_SRC_ADDR"].astype(str))

    print(f"[{split_name}] Sorting by (IPV4_SRC_ADDR, FLOW_START_MILLISECONDS) ...", flush=True)
    df.sort_values(
        ["IPV4_SRC_ADDR", "FLOW_START_MILLISECONDS"],
        ascending=[True, True],
        inplace=True,
        kind="mergesort",   # stable
    )
    df.reset_index(drop=True, inplace=True)

    # Quick sanity check: how many sequences of length 32 can be formed?
    grp_sizes = df.groupby("sequence_group_id").size()
    eligible_groups = int((grp_sizes >= 32).sum())
    eligible_rows   = int(grp_sizes[grp_sizes >= 32].sum())
    expected_seqs   = int(grp_sizes[grp_sizes >= 32].apply(lambda s: (s - 32) // 32 + 1).sum())
    print(f"[{split_name}] Unique source IPs : {len(grp_sizes):,}", flush=True)
    print(f"[{split_name}] IPs with >= 32 flows : {eligible_groups:,}", flush=True)
    print(f"[{split_name}] Rows in eligible IPs  : {eligible_rows:,}", flush=True)
    print(f"[{split_name}] Expected sequences (non-overlapping stride=32) : {expected_seqs:,}", flush=True)

    print(f"[{split_name}] Saving to {output_dir} ...", flush=True)
    hf_out = Dataset.from_pandas(df, preserve_index=False)
    hf_out.save_to_disk(str(output_dir))
    del df, hf_out
    gc.collect()

    print(f"[{split_name}] Done.", flush=True)


def main():
    print("=" * 70)
    print("Re-grouping NetFlow splits by Source IP for NIDS sequence windows")
    print("=" * 70)

    for split in SPLITS:
        regroup_split(split)

    print("\nAll splits processed.")
    print(f"New data root: {OUTPUT_BASE}")
    print("\nUpdate DEFAULT_TRAIN_DIR / DEFAULT_VALID_DIR / DEFAULT_TEST_DIR in")
    print("  src/04_train_foundation.py")
    print("  src/05_train_multitask_nids.py")
    print(f"to point at {OUTPUT_BASE}/{{train,validation,test}}")
    print("\nAlso delete any stale sequence_ranges_*.npy cache files in the new dirs.")


if __name__ == "__main__":
    main()
