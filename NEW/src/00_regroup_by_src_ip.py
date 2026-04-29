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

Use `--resume-materialization` after an interrupted run to reuse the saved
split plans and continue materializing only the missing grouped splits.
"""

import argparse
import gc
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from datasets import Dataset, Value, concatenate_datasets, load_from_disk
from pandas.util import hash_pandas_object


BASE_DIR = Path("/home/aka/PFE-code")
INPUT_BASE = BASE_DIR / "NEW" / "data" / "nids_transformer_split"
OUTPUT_BASE = BASE_DIR / "NEW" / "data" / "nids_src_grouped"
STATE_ROOT = OUTPUT_BASE.parent / "_tmp_nids_src_grouped_v2_state"
STATE_PLAN_ROOT = STATE_ROOT / "plans"
STATE_STATUS_PATH = STATE_ROOT / "materialization_status.json"
STATE_SUMMARY_PATH = STATE_ROOT / "planned_split_summary.json"

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
MIN_FAMILY_GROUPS_FOR_SPLIT_COVERAGE = 3
ID_EVAL_TARGET_ATTACK_RATE = 0.08
OOD_HOLDOUT_FAMILIES = {"Botnets"}

ROWS_PER_ARROW_SHARD = 2_000_000
MATERIALIZATION_SELECT_BATCH_ROWS = 100_000

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


def parse_args():
    parser = argparse.ArgumentParser(description="Rebuild grouped host-session NIDS splits.")
    parser.add_argument(
        "--resume-materialization",
        action="store_true",
        help="Resume grouped split materialization from the last saved split plans instead of recomputing planning.",
    )
    return parser.parse_args()


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

    # Use compact global session ids instead of hashed uint64 values. The generator-backed
    # HF writer overflows on Python ints above signed 64-bit range even for uint64 features.
    planning["sequence_group_id"] = session_break.cumsum().astype(np.uint64) - np.uint64(1)
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
        return group_meta, attack_family_rows

    attack_family_presence = attack_family_rows.drop_duplicates().sort_values(
        ["attack_family", "sequence_group_id"],
        kind="mergesort",
    )

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
    return group_meta, attack_family_presence


def assign_attack_groups(group_meta, attack_family_presence):
    print("Assigning attack-bearing host sessions to ID and OOD splits ...", flush=True)
    attack_group_mask = group_meta["attack_rows"] > 0
    ood_group_ids = set(
        attack_family_presence.loc[
            attack_family_presence["attack_family"].isin(OOD_HOLDOUT_FAMILIES),
            "sequence_group_id",
        ].tolist()
    )
    ood_mask = attack_group_mask & group_meta["sequence_group_id"].isin(ood_group_ids)
    group_meta.loc[ood_mask, NEW_SPLIT_COL] = "test_ood"

    assignable_presence = attack_family_presence.loc[
        ~attack_family_presence["attack_family"].isin(OOD_HOLDOUT_FAMILIES)
        & ~attack_family_presence["sequence_group_id"].isin(ood_group_ids)
    ].copy()

    family_group_counts = assignable_presence["attack_family"].value_counts().to_dict()
    for family_name in sorted(
        family_group_counts,
        key=lambda name: (family_group_counts[name],) + family_sort_key(name),
    ):
        if family_group_counts[family_name] < MIN_FAMILY_GROUPS_FOR_SPLIT_COVERAGE:
            continue

        family_group_ids = assignable_presence.loc[
            assignable_presence["attack_family"].eq(family_name),
            "sequence_group_id",
        ]
        family_groups = group_meta.loc[
            group_meta["sequence_group_id"].isin(family_group_ids)
        ].sort_values(["first_seen", "sequence_group_id"], kind="mergesort")
        used_group_ids = set()

        family_split_counts = family_groups[NEW_SPLIT_COL].value_counts().to_dict()
        if family_split_counts.get("validation", 0) == 0:
            validation_candidates = family_groups.loc[
                family_groups[NEW_SPLIT_COL].eq("train")
            ]
            if not validation_candidates.empty:
                validation_index = validation_candidates.index[0]
                group_meta.loc[validation_index, NEW_SPLIT_COL] = "validation"
                used_group_ids.add(int(group_meta.loc[validation_index, "sequence_group_id"]))

        family_groups = group_meta.loc[
            group_meta["sequence_group_id"].isin(family_group_ids)
        ].sort_values(["first_seen", "sequence_group_id"], kind="mergesort")
        family_split_counts = family_groups[NEW_SPLIT_COL].value_counts().to_dict()
        if family_split_counts.get("test", 0) == 0:
            test_candidates = family_groups.loc[
                family_groups[NEW_SPLIT_COL].eq("train")
                & ~family_groups["sequence_group_id"].isin(used_group_ids)
            ]
            if test_candidates.empty:
                test_candidates = family_groups.loc[
                    ~family_groups["sequence_group_id"].isin(used_group_ids)
                ]
            if not test_candidates.empty:
                test_index = test_candidates.index[-1]
                group_meta.loc[test_index, NEW_SPLIT_COL] = "test"

    assignable = group_meta.loc[
        attack_group_mask & group_meta[NEW_SPLIT_COL].eq("train")
    ].copy()
    family_counts = assignable["dominant_family"].value_counts().to_dict()

    for family_name in sorted(family_counts, key=family_sort_key):
        family_mask = assignable["dominant_family"].eq(family_name)
        family_groups = assignable.loc[family_mask].sort_values(
            ["first_seen", "sequence_group_id"],
            kind="mergesort",
        )
        family_count = len(family_groups)
        if family_count < MIN_FAMILY_GROUPS_FOR_ID_EVAL:
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

    rare_train_only_families = []
    for family_name, family_group_count in family_group_counts.items():
        if family_group_count >= MIN_FAMILY_GROUPS_FOR_SPLIT_COVERAGE:
            continue

        family_group_ids = assignable_presence.loc[
            assignable_presence["attack_family"].eq(family_name),
            "sequence_group_id",
        ]
        family_splits = set(
            group_meta.loc[
                group_meta["sequence_group_id"].isin(family_group_ids),
                NEW_SPLIT_COL,
            ]
        )
        if family_splits == {"train"}:
            rare_train_only_families.append(family_name)

    rare_train_only_families.sort(key=family_sort_key)
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
            "min_family_groups_for_split_coverage": MIN_FAMILY_GROUPS_FOR_SPLIT_COVERAGE,
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


def build_split_plan(planning, split_name):
    split_plan = planning.loc[
        planning[NEW_SPLIT_COL] == split_name,
        [SOURCE_SPLIT_COL, SOURCE_ROW_INDEX_COL, "FLOW_START_MILLISECONDS", "sequence_group_id"],
    ].copy()

    if split_plan.empty:
        return split_plan

    split_plan.sort_values(
        ["sequence_group_id", "FLOW_START_MILLISECONDS", SOURCE_SPLIT_COL, SOURCE_ROW_INDEX_COL],
        ascending=[True, True, True, True],
        inplace=True,
        kind="mergesort",
    )
    split_plan.reset_index(drop=True, inplace=True)
    return split_plan


def save_resume_state(planning, summary):
    if STATE_ROOT.exists():
        shutil.rmtree(STATE_ROOT)
    STATE_PLAN_ROOT.mkdir(parents=True, exist_ok=True)

    with open(STATE_SUMMARY_PATH, "w") as handle:
        json.dump(summary, handle, indent=2)

    for split_name in OUTPUT_SPLITS:
        split_plan = build_split_plan(planning, split_name)
        plan_path = STATE_PLAN_ROOT / f"{split_name}.parquet"
        split_plan.to_parquet(plan_path, index=False)
        print(f"[{split_name}] Saved materialization plan: {plan_path}", flush=True)
        del split_plan
        gc.collect()


def load_resume_state():
    if not STATE_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Missing saved split summary for resume: {STATE_SUMMARY_PATH}. "
            "Run the ETL once without --resume-materialization first."
        )

    with open(STATE_SUMMARY_PATH, "r") as handle:
        summary = json.load(handle)

    missing_plan_paths = [
        str(STATE_PLAN_ROOT / f"{split_name}.parquet")
        for split_name in OUTPUT_SPLITS
        if not (STATE_PLAN_ROOT / f"{split_name}.parquet").exists()
    ]
    if missing_plan_paths:
        raise FileNotFoundError(
            "Missing saved split plans required for resume: "
            f"{missing_plan_paths}"
        )

    return summary


def load_materialization_status():
    if not STATE_STATUS_PATH.exists():
        return {}

    with open(STATE_STATUS_PATH, "r") as handle:
        return json.load(handle)


def save_materialization_status(status):
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    temp_path = STATE_STATUS_PATH.with_suffix(".tmp")
    with open(temp_path, "w") as handle:
        json.dump(status, handle, indent=2)
    temp_path.replace(STATE_STATUS_PATH)


def split_output_matches_expected(output_dir, expected_rows):
    if not output_dir.exists():
        return False

    try:
        return len(load_from_disk(str(output_dir))) == int(expected_rows)
    except Exception:
        return False


def cleanup_resume_state():
    if STATE_ROOT.exists():
        shutil.rmtree(STATE_ROOT)


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


def build_output_features(source_dataset):
    features = source_dataset.features.copy()
    features["sequence_group_id"] = Value("uint64")
    return features


def iter_materialized_split_rows(split_name, split_plan, source_datasets):
    total_rows = len(split_plan)
    if total_rows == 0:
        return

    for batch_start in range(0, total_rows, MATERIALIZATION_SELECT_BATCH_ROWS):
        batch_end = min(batch_start + MATERIALIZATION_SELECT_BATCH_ROWS, total_rows)
        batch_plan = split_plan.iloc[batch_start:batch_end]
        source_splits = batch_plan[SOURCE_SPLIT_COL].to_numpy(dtype=object)

        if len(batch_plan) > 1:
            change_offsets = np.flatnonzero(source_splits[1:] != source_splits[:-1]) + 1
        else:
            change_offsets = np.array([], dtype=np.int64)

        run_starts = np.concatenate(([0], change_offsets))
        run_ends = np.concatenate((change_offsets, [len(batch_plan)]))

        for run_start, run_end in zip(run_starts, run_ends):
            run_plan = batch_plan.iloc[run_start:run_end]
            source_split = str(source_splits[run_start])
            source_indices = run_plan[SOURCE_ROW_INDEX_COL].tolist()
            source_rows = source_datasets[source_split][source_indices]
            source_rows["sequence_group_id"] = run_plan["sequence_group_id"].tolist()
            column_names = tuple(source_rows.keys())

            for row_offset in range(run_end - run_start):
                yield {column_name: source_rows[column_name][row_offset] for column_name in column_names}

            del source_rows

        print(
            f"[{split_name}] materialized {batch_end:,} / {total_rows:,} rows",
            flush=True,
        )
        del batch_plan
        gc.collect()


def materialize_outputs(summary, resume_materialization=False):
    print("Materializing grouped output splits ...", flush=True)
    cache_root = OUTPUT_BASE.parent / "_tmp_nids_src_grouped_v2_cache"
    if resume_materialization:
        cache_root.mkdir(parents=True, exist_ok=True)
        OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
        materialization_status = load_materialization_status()
    else:
        if cache_root.exists():
            shutil.rmtree(cache_root)
        cache_root.mkdir(parents=True, exist_ok=True)

        if OUTPUT_BASE.exists():
            print(f"Removing previous output root: {OUTPUT_BASE}", flush=True)
            shutil.rmtree(OUTPUT_BASE)
        OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
        materialization_status = {}

    source_datasets = {}
    for source_split in INPUT_SPLITS:
        source_dir = INPUT_BASE / source_split
        print(f"[{source_split}] Opening source dataset from {source_dir} ...", flush=True)
        source_datasets[source_split] = load_from_disk(str(source_dir))

    sample_dataset = source_datasets[INPUT_SPLITS[0]]
    output_features = build_output_features(sample_dataset)

    for split_name in OUTPUT_SPLITS:
        expected_rows = int(summary["splits"][split_name]["rows_total"])
        output_dir = OUTPUT_BASE / split_name
        split_cache_dir = cache_root / split_name

        if (
            materialization_status.get(split_name, {}).get("status") == "completed"
            and split_output_matches_expected(output_dir, expected_rows)
        ):
            print(f"[{split_name}] Already materialized. Skipping.", flush=True)
            continue

        plan_path = STATE_PLAN_ROOT / f"{split_name}.parquet"
        split_plan = pd.read_parquet(plan_path)

        if split_plan.empty:
            print(f"[{split_name}] No rows assigned. Skipping save.", flush=True)
            materialization_status[split_name] = {"status": "completed", "rows_total": 0}
            save_materialization_status(materialization_status)
            continue

        if split_cache_dir.exists():
            shutil.rmtree(split_cache_dir)
        if output_dir.exists():
            shutil.rmtree(output_dir)

        print(f"[{split_name}] Building HF dataset from {len(split_plan):,} planned rows ...", flush=True)

        def split_generator():
            yield from iter_materialized_split_rows(split_name, split_plan, source_datasets)

        output_dataset = Dataset.from_generator(
            split_generator,
            features=output_features,
            cache_dir=str(split_cache_dir),
            keep_in_memory=False,
        )
        output_dataset.save_to_disk(str(output_dir))
        print(f"[{split_name}] Saved to {output_dir}", flush=True)
        materialization_status[split_name] = {
            "status": "completed",
            "rows_total": int(len(split_plan)),
        }
        save_materialization_status(materialization_status)
        del output_dataset, split_plan
        gc.collect()

    del source_datasets
    gc.collect()

    if cache_root.exists():
        shutil.rmtree(cache_root)


def main():
    args = parse_args()
    print("=" * 78)
    print("Rebuilding host-session NIDS splits with family-aware split assignment")
    print("=" * 78)
    ensure_input_splits_exist()

    if args.resume_materialization:
        summary = load_resume_state()
        print("Loaded saved split summary for resume:", flush=True)
        print(json.dumps(summary, indent=2), flush=True)
        materialize_outputs(summary, resume_materialization=True)

        summary_path = OUTPUT_BASE / "split_summary.json"
        with open(summary_path, "w") as handle:
            json.dump(summary, handle, indent=2)
        cleanup_resume_state()
        print(f"Saved split summary: {summary_path}", flush=True)
        print(f"New grouped data root: {OUTPUT_BASE}", flush=True)
        print("Primary training splits: train / validation / test", flush=True)
        print("Additional evaluation split: test_ood", flush=True)
        print("Delete stale sequence_ranges_*.npy and downstream_targets_*.npz caches after rerunning this ETL.", flush=True)
        return

    planning_frames = [load_planning_frame(split_name) for split_name in INPUT_SPLITS]
    planning = pd.concat(planning_frames, ignore_index=True)
    del planning_frames
    gc.collect()
    print(f"Combined planning frame ready: {len(planning):,} rows", flush=True)

    planning = build_host_sessions(planning)
    group_meta, attack_family_presence = build_group_metadata(planning)
    print(
        f"Host sessions: {len(group_meta):,} total | "
        f"attack-bearing sessions: {(group_meta['attack_rows'] > 0).sum():,}",
        flush=True,
    )

    group_meta, rare_train_only_families = assign_attack_groups(group_meta, attack_family_presence)
    group_meta, benign_targets = assign_benign_groups(group_meta)
    print(f"Benign row targets for ID evaluation splits: {benign_targets}", flush=True)

    split_map = group_meta.set_index("sequence_group_id")[NEW_SPLIT_COL]
    planning[NEW_SPLIT_COL] = planning["sequence_group_id"].map(split_map)
    summary = build_split_summary(planning, group_meta, rare_train_only_families)

    print("Planned split summary:", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    save_resume_state(planning, summary)
    del planning, group_meta, split_map, attack_family_presence
    gc.collect()

    materialize_outputs(summary, resume_materialization=False)

    summary_path = OUTPUT_BASE / "split_summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    cleanup_resume_state()
    print(f"Saved split summary: {summary_path}", flush=True)
    print(f"New grouped data root: {OUTPUT_BASE}", flush=True)
    print("Primary training splits: train / validation / test", flush=True)
    print("Additional evaluation split: test_ood", flush=True)
    print("Delete stale sequence_ranges_*.npy and downstream_targets_*.npz caches after rerunning this ETL.", flush=True)


if __name__ == "__main__":
    main()
