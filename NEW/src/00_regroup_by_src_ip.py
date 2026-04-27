"""
00_regroup_by_src_ip.py — Rebuild host-session splits for the NIDS pipeline
=============================================================================
The previous workflow preserved the original train/validation/test boundaries
from data/nids_transformer_split and only regrouped rows by source IP inside
each split. That kept the downstream split pathology intact: validation ended
up containing only one attack family while test contained different families.

This script rebuilds the grouped NIDS corpus from the combined upstream data:

1. Concatenate the existing data/nids_transformer_split/{train,validation,test}
   splits into one planning table.
2. Reassign sequence_group_id as host sessions rather than raw source IPs.
   A session breaks when the same source IP is inactive for more than
   HOST_SESSION_GAP_MINUTES.
3. Build group-level metadata and assign new splits using group-safe logic:
   - family-aware interleaved assignment for attack-bearing host sessions
   - dedicated held-out OOD split for Botnets sessions
   - benign-group quotas for validation/test to keep ID evaluation informative
4. Materialize grouped HF datasets under data/nids_src_grouped/{...}.

Primary output splits consumed by the training code stay stable:
  train / validation / test

Additional output split:
  test_ood   -> held-out Botnets host sessions for unknown-family evaluation

This file is the ETL authority for grouped downstream data.
"""

import gc
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from datasets import Dataset, concatenate_datasets, load_from_disk
from pandas.util import hash_pandas_object


BASE_DIR = Path("/home/aka/PFE-code")
INPUT_BASE = BASE_DIR / "NEW" / "data" / "nids_transformer_split"
OUTPUT_BASE = BASE_DIR / "NEW" / "data" / "nids_src_grouped"

INPUT_SPLITS = ["train", "validation", "test"]
OUTPUT_SPLITS = ["train", "validation", "test", "test_ood"]

HOST_SESSION_GAP_MINUTES = 30
HOST_SESSION_GAP_MS = HOST_SESSION_GAP_MINUTES * 60 * 1000

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
INTERLEAVED_SPLIT_PATTERN = (
    "train",
    "train",
    "validation",
    "train",
    "test",
    "train",
    "train",
    "validation",
    "train",
    "test",
    "train",
    "train",
    "validation",
    "train",
    "test",
    "train",
    "train",
    "train",
    "train",
    "train",
)

MIN_FAMILY_GROUPS_FOR_ID_EVAL = 8
ID_EVAL_TARGET_ATTACK_RATE = 0.08
OOD_HOLDOUT_FAMILIES = {"Botnets"}

ROWS_PER_ARROW_SHARD = 2_000_000

SOURCE_SPLIT_COL = "__source_split"
SOURCE_ROW_INDEX_COL = "__source_row_idx"
SOURCE_IP_HASH_COL = "__source_ip_hash"
IS_ATTACK_COL = "__is_attack"
NEW_SPLIT_COL = "__new_split"
ALLOC_KEY_COL = "__allocation_key"

PROJECT_ATTACK_FAMILY_ORDER = [
    "DoS / DDoS",
    "Brute Force",
    "Botnets",
    "Infiltration",
    "Web Attacks",
    "Fuzzers",
    "Analysis / Backdoors",
    "Exploits / Shellcode",
    "Reconnaissance",
    "Worms / Generic",
]

ATTACK_FAMILY_MAP = {
    "Analysis": "Analysis / Backdoors",
    "Backdoor": "Analysis / Backdoors",
    "Bot": "Botnets",
    "Brute_Force_-Web": "Web Attacks",
    "Brute_Force_-XSS": "Web Attacks",
    "DDOS_attack-HOIC": "DoS / DDoS",
    "DDOS_attack-LOIC-UDP": "DoS / DDoS",
    "DDoS_attacks-LOIC-HTTP": "DoS / DDoS",
    "DoS": "DoS / DDoS",
    "DoS_attacks-GoldenEye": "DoS / DDoS",
    "DoS_attacks-Hulk": "DoS / DDoS",
    "DoS_attacks-SlowHTTPTest": "DoS / DDoS",
    "DoS_attacks-Slowloris": "DoS / DDoS",
    "Exploits": "Exploits / Shellcode",
    "FTP-BruteForce": "Brute Force",
    "Generic": "Worms / Generic",
    "Infilteration": "Infiltration",
    "Infiltration": "Infiltration",
    "Reconnaissance": "Reconnaissance",
    "SQL_Injection": "Web Attacks",
    "SSH-Bruteforce": "Brute Force",
    "Shellcode": "Exploits / Shellcode",
    "Worms": "Worms / Generic",
}


def map_attack_family(attack_name):
    attack_name = str(attack_name)
    if attack_name == "Benign":
        return "Benign"

    mapped_name = ATTACK_FAMILY_MAP.get(attack_name)
    if mapped_name is not None:
        return mapped_name

    lower_name = attack_name.lower()
    if "fuzzer" in lower_name:
        return "Fuzzers"
    if "recon" in lower_name:
        return "Reconnaissance"
    if "worm" in lower_name or "generic" in lower_name:
        return "Worms / Generic"
    if "shellcode" in lower_name or "exploit" in lower_name:
        return "Exploits / Shellcode"
    if "analysis" in lower_name or "backdoor" in lower_name:
        return "Analysis / Backdoors"
    if "sql" in lower_name or "xss" in lower_name or "web" in lower_name:
        return "Web Attacks"
    if "brute" in lower_name or "ssh" in lower_name or "ftp" in lower_name:
        return "Brute Force"
    if "dos" in lower_name or "ddos" in lower_name or "hoic" in lower_name or "loic" in lower_name:
        return "DoS / DDoS"
    if "bot" in lower_name:
        return "Botnets"
    if "infil" in lower_name:
        return "Infiltration"
    return attack_name


def family_sort_key(family_name):
    try:
        return (0, PROJECT_ATTACK_FAMILY_ORDER.index(family_name))
    except ValueError:
        return (1, family_name)


def ensure_input_splits_exist():
    missing = [split_name for split_name in INPUT_SPLITS if not (INPUT_BASE / split_name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required upstream split directories under {INPUT_BASE}: {missing}"
        )


def load_planning_frame(split_name):
    input_dir = INPUT_BASE / split_name
    print(f"[{split_name}] Loading planning columns from {input_dir} ...", flush=True)
    dataset = load_from_disk(str(input_dir))
    row_count = len(dataset)

    planning = pd.DataFrame({
        SOURCE_SPLIT_COL: pd.Series([split_name] * row_count, dtype="string"),
        SOURCE_ROW_INDEX_COL: np.arange(row_count, dtype=np.int64),
        "FLOW_START_MILLISECONDS": np.asarray(dataset["FLOW_START_MILLISECONDS"], dtype=np.int64),
        "Label": np.asarray(dataset["Label"], dtype=np.int16),
        "Attack": pd.Series(dataset["Attack"], dtype="string"),
        "IPV4_SRC_ADDR": pd.Series(dataset["IPV4_SRC_ADDR"], dtype="string"),
    })
    del dataset
    gc.collect()

    planning[SOURCE_IP_HASH_COL] = hash_pandas_object(planning["IPV4_SRC_ADDR"], index=False).astype(np.uint64)
    planning[IS_ATTACK_COL] = planning["Label"].astype(np.int64) > 0
    planning["attack_family"] = planning["Attack"].map(map_attack_family).astype("string")

    planning.drop(columns=["IPV4_SRC_ADDR", "Attack"], inplace=True)
    print(f"[{split_name}] Planning frame ready: {row_count:,} rows", flush=True)
    return planning


def build_host_sessions(planning):
    print("Planning host sessions across the combined corpus ...", flush=True)
    planning.sort_values(
        [SOURCE_IP_HASH_COL, "FLOW_START_MILLISECONDS", SOURCE_SPLIT_COL, SOURCE_ROW_INDEX_COL],
        ascending=[True, True, True, True],
        inplace=True,
        kind="mergesort",
    )
    planning.reset_index(drop=True, inplace=True)

    previous_ip = planning[SOURCE_IP_HASH_COL].shift()
    previous_start = planning["FLOW_START_MILLISECONDS"].shift()
    session_break = planning[SOURCE_IP_HASH_COL].ne(previous_ip)
    session_break |= (
        planning["FLOW_START_MILLISECONDS"] - previous_start
    ).fillna(0).gt(HOST_SESSION_GAP_MS)

    session_ordinal = session_break.groupby(planning[SOURCE_IP_HASH_COL]).cumsum().astype(np.int64) - 1
    planning["sequence_group_id"] = hash_pandas_object(
        pd.DataFrame({
            SOURCE_IP_HASH_COL: planning[SOURCE_IP_HASH_COL],
            "session_ordinal": session_ordinal,
        }),
        index=False,
    ).astype(np.uint64)
    return planning


def build_group_metadata(planning):
    print("Building host-session metadata ...", flush=True)
    group_meta = planning.groupby("sequence_group_id", sort=False).agg(
        first_seen=("FLOW_START_MILLISECONDS", "min"),
        last_seen=("FLOW_START_MILLISECONDS", "max"),
        rows=("Label", "size"),
        attack_rows=(IS_ATTACK_COL, "sum"),
    )
    group_meta["benign_rows"] = group_meta["rows"] - group_meta["attack_rows"]
    group_meta.reset_index(inplace=True)
    group_meta[NEW_SPLIT_COL] = "train"
    group_meta[ALLOC_KEY_COL] = hash_pandas_object(
        group_meta["sequence_group_id"].astype("uint64"),
        index=False,
    ).astype(np.uint64)

    attack_family_rows = planning.loc[
        planning[IS_ATTACK_COL],
        ["sequence_group_id", "attack_family"],
    ].copy()
    if attack_family_rows.empty:
        group_meta["dominant_family"] = "Benign"
        group_meta["distinct_attack_families"] = 0
        return group_meta

    family_counts = attack_family_rows.groupby(
        ["sequence_group_id", "attack_family"],
        sort=False,
        observed=True,
    ).size().reset_index(name="family_rows")
    dominant_family = family_counts.sort_values(
        ["sequence_group_id", "family_rows", "attack_family"],
        ascending=[True, False, True],
        kind="mergesort",
    ).drop_duplicates("sequence_group_id")
    distinct_family_counts = family_counts.groupby("sequence_group_id", observed=True).size().reset_index(
        name="distinct_attack_families"
    )

    group_meta = group_meta.merge(
        dominant_family[["sequence_group_id", "attack_family"]].rename(
            columns={"attack_family": "dominant_family"}
        ),
        on="sequence_group_id",
        how="left",
    )
    group_meta = group_meta.merge(distinct_family_counts, on="sequence_group_id", how="left")
    group_meta["dominant_family"] = group_meta["dominant_family"].fillna("Benign")
    group_meta["distinct_attack_families"] = group_meta["distinct_attack_families"].fillna(0).astype(np.int64)
    return group_meta


def assign_attack_groups(group_meta):
    print("Assigning attack-bearing host sessions to ID and OOD splits ...", flush=True)
    attack_group_mask = group_meta["attack_rows"] > 0
    ood_mask = attack_group_mask & group_meta["dominant_family"].isin(OOD_HOLDOUT_FAMILIES)
    group_meta.loc[ood_mask, NEW_SPLIT_COL] = "test_ood"

    assignable = group_meta.loc[
        attack_group_mask & group_meta[NEW_SPLIT_COL].eq("train")
    ].copy()
    family_counts = assignable["dominant_family"].value_counts().to_dict()
    rare_train_only_families = []

    for family_name in sorted(family_counts, key=family_sort_key):
        family_mask = assignable["dominant_family"].eq(family_name)
        family_groups = assignable.loc[family_mask].sort_values(
            ["first_seen", "sequence_group_id"],
            kind="mergesort",
        )
        family_count = len(family_groups)
        if family_count < MIN_FAMILY_GROUPS_FOR_ID_EVAL:
            rare_train_only_families.append(family_name)
            continue

        family_indices = family_groups.index.to_list()
        family_assignments = []
        for offset, group_index in enumerate(family_indices):
            family_assignments.append((group_index, INTERLEAVED_SPLIT_PATTERN[offset % len(INTERLEAVED_SPLIT_PATTERN)]))

        assigned_splits = Counter(split_name for _, split_name in family_assignments)
        if assigned_splits["validation"] == 0 and family_count >= MIN_FAMILY_GROUPS_FOR_ID_EVAL:
            median_offset = family_count // 2
            family_assignments[median_offset] = (family_assignments[median_offset][0], "validation")
        if assigned_splits["test"] == 0 and family_count >= MIN_FAMILY_GROUPS_FOR_ID_EVAL:
            family_assignments[-1] = (family_assignments[-1][0], "test")

        for group_index, split_name in family_assignments:
            group_meta.loc[group_index, NEW_SPLIT_COL] = split_name

    return group_meta, rare_train_only_families


def compute_benign_row_targets(group_meta):
    targets = {}
    for split_name in ("validation", "test"):
        split_attack_rows = int(
            group_meta.loc[group_meta[NEW_SPLIT_COL] == split_name, "attack_rows"].sum()
        )
        if split_attack_rows == 0:
            targets[split_name] = 0
            continue
        desired_total_rows = int(math.ceil(split_attack_rows / ID_EVAL_TARGET_ATTACK_RATE))
        targets[split_name] = max(desired_total_rows - split_attack_rows, 0)
    return targets


def assign_benign_groups(group_meta):
    print("Assigning benign host sessions to stabilize validation/test prevalence ...", flush=True)
    benign_mask = group_meta["attack_rows"] == 0
    benign_groups = group_meta.loc[benign_mask].sort_values(
        ["first_seen", ALLOC_KEY_COL, "sequence_group_id"],
        kind="mergesort",
    )
    benign_targets = compute_benign_row_targets(group_meta)
    remaining_rows = dict(benign_targets)

    for offset, (group_index, row_count) in enumerate(zip(benign_groups.index, benign_groups["rows"])):
        preferred_split = INTERLEAVED_SPLIT_PATTERN[offset % len(INTERLEAVED_SPLIT_PATTERN)]
        if preferred_split in remaining_rows and remaining_rows[preferred_split] > 0:
            group_meta.loc[group_index, NEW_SPLIT_COL] = preferred_split
            remaining_rows[preferred_split] = max(remaining_rows[preferred_split] - int(row_count), 0)

    for split_name in ("validation", "test"):
        if remaining_rows[split_name] <= 0:
            continue
        supplemental_candidates = group_meta.loc[
            (group_meta["attack_rows"] == 0) & group_meta[NEW_SPLIT_COL].eq("train")
        ].sort_values([ALLOC_KEY_COL, "first_seen"], kind="mergesort")
        cumulative_rows = 0
        for group_index, row_count in zip(supplemental_candidates.index, supplemental_candidates["rows"]):
            group_meta.loc[group_index, NEW_SPLIT_COL] = split_name
            cumulative_rows += int(row_count)
            if cumulative_rows >= remaining_rows[split_name]:
                break

    return group_meta, benign_targets


def build_split_summary(planning, group_meta, rare_train_only_families):
    summary = {
        "config": {
            "host_session_gap_minutes": HOST_SESSION_GAP_MINUTES,
            "train_ratio": TRAIN_RATIO,
            "validation_ratio": VALIDATION_RATIO,
            "test_ratio": TEST_RATIO,
            "id_eval_target_attack_rate": ID_EVAL_TARGET_ATTACK_RATE,
            "ood_holdout_families": sorted(OOD_HOLDOUT_FAMILIES),
            "min_family_groups_for_id_eval": MIN_FAMILY_GROUPS_FOR_ID_EVAL,
        },
        "rare_train_only_families": sorted(set(rare_train_only_families), key=family_sort_key),
        "splits": {},
    }

    for split_name in OUTPUT_SPLITS:
        split_rows = planning.loc[planning[NEW_SPLIT_COL] == split_name]
        split_groups = group_meta.loc[group_meta[NEW_SPLIT_COL] == split_name]
        attack_rows = int(split_rows[IS_ATTACK_COL].sum())
        total_rows = int(len(split_rows))
        family_counts = split_rows.loc[
            split_rows[IS_ATTACK_COL],
            "attack_family",
        ].value_counts().to_dict()

        summary["splits"][split_name] = {
            "rows_total": total_rows,
            "attack_rows": attack_rows,
            "benign_rows": total_rows - attack_rows,
            "attack_pct_of_total": round(100.0 * attack_rows / max(total_rows, 1), 4),
            "groups_total": int(len(split_groups)),
            "groups_with_attack": int((split_groups["attack_rows"] > 0).sum()),
            "attack_families": {
                str(family_name): int(count)
                for family_name, count in sorted(family_counts.items(), key=lambda item: family_sort_key(item[0]))
            },
        }

    return summary


def write_dataframe_shards(df, shard_dir, shard_prefix):
    shard_dir.mkdir(parents=True, exist_ok=True)
    total_rows = len(df)
    shard_count = max(1, (total_rows + ROWS_PER_ARROW_SHARD - 1) // ROWS_PER_ARROW_SHARD)
    for shard_index in range(shard_count):
        start_row = shard_index * ROWS_PER_ARROW_SHARD
        end_row = min(start_row + ROWS_PER_ARROW_SHARD, total_rows)
        shard_path = shard_dir / f"{shard_prefix}-{shard_index:04d}.arrow"
        table = pa.Table.from_pandas(df.iloc[start_row:end_row], preserve_index=False)
        with pa.ipc.new_stream(str(shard_path), table.schema) as writer:
            writer.write_table(table)
        del table
        gc.collect()


def finalize_split_from_shards(split_name, shard_dir):
    output_dir = OUTPUT_BASE / split_name
    if output_dir.exists():
        shutil.rmtree(output_dir)

    shard_paths = sorted(shard_dir.glob("*.arrow"))
    if not shard_paths:
        print(f"[{split_name}] No rows assigned. Skipping save.", flush=True)
        return

    shard_datasets = []
    for shard_path in shard_paths:
        with pa.ipc.open_stream(str(shard_path)) as reader:
            shard_datasets.append(Dataset(reader.read_all()))

    output_dataset = shard_datasets[0] if len(shard_datasets) == 1 else concatenate_datasets(shard_datasets)
    output_dataset.save_to_disk(str(output_dir))
    del output_dataset, shard_datasets
    gc.collect()
    print(f"[{split_name}] Saved to {output_dir}", flush=True)


def materialize_outputs(planning):
    print("Materializing grouped output splits ...", flush=True)
    temp_root = OUTPUT_BASE.parent / "_tmp_nids_src_grouped_v2_shards"
    if temp_root.exists():
        shutil.rmtree(temp_root)
    temp_root.mkdir(parents=True, exist_ok=True)

    planning.sort_values([SOURCE_SPLIT_COL, SOURCE_ROW_INDEX_COL], inplace=True, kind="mergesort")
    planning_by_source = {
        split_name: planning.loc[planning[SOURCE_SPLIT_COL] == split_name, ["sequence_group_id", NEW_SPLIT_COL]].reset_index(drop=True)
        for split_name in INPUT_SPLITS
    }

    for source_split in INPUT_SPLITS:
        source_dir = INPUT_BASE / source_split
        print(f"[{source_split}] Loading full rows from {source_dir} ...", flush=True)
        dataset = load_from_disk(str(source_dir))
        full_df = dataset.to_pandas()
        del dataset
        gc.collect()

        assignment = planning_by_source[source_split]
        if len(full_df) != len(assignment):
            raise RuntimeError(
                f"Assignment length mismatch for {source_split}: {len(full_df)} rows vs {len(assignment)} planned"
            )

        full_df["sequence_group_id"] = assignment["sequence_group_id"].to_numpy(dtype=np.uint64)
        full_df[NEW_SPLIT_COL] = assignment[NEW_SPLIT_COL].to_numpy(dtype=object)

        sort_columns = ["sequence_group_id", "FLOW_START_MILLISECONDS"]
        if "FLOW_END_MILLISECONDS" in full_df.columns:
            sort_columns.append("FLOW_END_MILLISECONDS")

        for split_name in OUTPUT_SPLITS:
            split_df = full_df.loc[full_df[NEW_SPLIT_COL] == split_name].copy()
            if split_df.empty:
                continue
            split_df.sort_values(sort_columns, ascending=True, inplace=True, kind="mergesort")
            split_df.drop(columns=[NEW_SPLIT_COL], inplace=True)
            write_dataframe_shards(split_df, temp_root / split_name, source_split)
            print(
                f"[{source_split} -> {split_name}] wrote {len(split_df):,} rows to temporary shards",
                flush=True,
            )
            del split_df
            gc.collect()

        del full_df
        gc.collect()

    if OUTPUT_BASE.exists():
        print(f"Removing previous output root: {OUTPUT_BASE}", flush=True)
        shutil.rmtree(OUTPUT_BASE)
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    for split_name in OUTPUT_SPLITS:
        finalize_split_from_shards(split_name, temp_root / split_name)

    shutil.rmtree(temp_root)


def main():
    print("=" * 78)
    print("Rebuilding host-session NIDS splits with family-aware split assignment")
    print("=" * 78)
    ensure_input_splits_exist()

    planning_frames = [load_planning_frame(split_name) for split_name in INPUT_SPLITS]
    planning = pd.concat(planning_frames, ignore_index=True)
    del planning_frames
    gc.collect()
    print(f"Combined planning frame ready: {len(planning):,} rows", flush=True)

    planning = build_host_sessions(planning)
    group_meta = build_group_metadata(planning)
    print(
        f"Host sessions: {len(group_meta):,} total | "
        f"attack-bearing sessions: {(group_meta['attack_rows'] > 0).sum():,}",
        flush=True,
    )

    group_meta, rare_train_only_families = assign_attack_groups(group_meta)
    group_meta, benign_targets = assign_benign_groups(group_meta)
    print(f"Benign row targets for ID evaluation splits: {benign_targets}", flush=True)

    split_map = group_meta.set_index("sequence_group_id")[NEW_SPLIT_COL]
    planning[NEW_SPLIT_COL] = planning["sequence_group_id"].map(split_map)
    summary = build_split_summary(planning, group_meta, rare_train_only_families)

    print("Planned split summary:", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    materialize_outputs(planning)

    summary_path = OUTPUT_BASE / "split_summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved split summary: {summary_path}", flush=True)
    print(f"New grouped data root: {OUTPUT_BASE}", flush=True)
    print("Primary training splits: train / validation / test", flush=True)
    print("Additional evaluation split: test_ood", flush=True)
    print("Delete stale sequence_ranges_*.npy and downstream_targets_*.npz caches after rerunning this ETL.", flush=True)


if __name__ == "__main__":
    main()
