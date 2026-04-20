import gc
import glob
import os
import shutil
import sqlite3
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

warnings.filterwarnings('ignore')

BASE_DIR = '/home/aka/PFE-code'
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(DATA_DIR, 'nids_transformer_split')
PARQUET_DIR = os.path.join(DATA_DIR, 'parquet_splits')
TEMP_DIR = os.path.join(DATA_DIR, 'etl_temp')
RAW_SHARD_DIR = os.path.join(TEMP_DIR, 'raw_shards')
BUCKET_PART_DIR = os.path.join(TEMP_DIR, 'bucket_parts')
GROUP_DB_PATH = os.path.join(TEMP_DIR, 'group_tracking.sqlite')

CSV_CHUNK_SIZE = int(os.environ.get('NIDS_ETL_CHUNK_SIZE', '100000'))
SORT_BUCKET_COUNT = int(os.environ.get('NIDS_ETL_SORT_BUCKETS', '128'))
SQLITE_BATCH_SIZE = int(os.environ.get('NIDS_ETL_SQLITE_BATCH', '5000'))
MAX_CHUNKS_PER_FILE = int(os.environ.get('NIDS_ETL_MAX_CHUNKS', '0'))


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


def ensure_clean_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def write_parquet_frame(df, output_path):
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, output_path, compression='snappy')


def downcast_chunk(chunk):
    float_cols = chunk.select_dtypes(include=['float64']).columns
    chunk[float_cols] = chunk[float_cols].astype('float32')

    int_cols = chunk.select_dtypes(include=['int64']).columns
    columns_to_exclude = [
        'FLOW_START_MILLISECONDS',
        'FLOW_END_MILLISECONDS',
        'L7_PROTO',
        'CLIENT_TCP_FLAGS',
        'SERVER_TCP_FLAGS',
        'ICMP_TYPE',
        'ICMP_IPV4_TYPE',
    ]
    cols_to_downcast = [col for col in int_cols if col not in columns_to_exclude]
    chunk[cols_to_downcast] = chunk[cols_to_downcast].astype('int32')
    return chunk


def format_group_id_key(group_id_value):
    return f'{int(group_id_value):016x}'


def create_group_tracking_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    conn.execute(
        '''
        CREATE TABLE IF NOT EXISTS group_first_seen (
            sequence_group_id TEXT PRIMARY KEY,
            first_seen INTEGER NOT NULL
        )
        '''
    )
    conn.execute(
        '''
        CREATE TABLE IF NOT EXISTS group_splits (
            sequence_group_id TEXT PRIMARY KEY,
            split_name TEXT NOT NULL
        )
        '''
    )
    conn.commit()
    return conn


def update_group_first_seen(conn, chunk):
    chunk_group_mins = pd.DataFrame({
        'sequence_group_id': [format_group_id_key(value) for value in chunk['sequence_group_id'].to_numpy(dtype=np.uint64)],
        'first_seen': chunk['FLOW_START_MILLISECONDS'].to_numpy(dtype=np.int64),
    })
    chunk_group_mins = (
        chunk_group_mins.groupby('sequence_group_id', sort=False)['first_seen']
        .min()
        .reset_index()
    )

    conn.executemany(
        '''
        INSERT INTO group_first_seen (sequence_group_id, first_seen)
        VALUES (?, ?)
        ON CONFLICT(sequence_group_id) DO UPDATE SET
            first_seen = MIN(group_first_seen.first_seen, excluded.first_seen)
        ''',
        list(chunk_group_mins.itertuples(index=False, name=None)),
    )
    conn.commit()


def compute_split_boundaries(total_groups, train_ratio=0.70, validation_ratio=0.15):
    train_groups = int(total_groups * train_ratio)
    validation_groups = int(total_groups * validation_ratio)

    train_groups = min(max(train_groups, 1), total_groups - 2)
    validation_groups = min(max(validation_groups, 1), total_groups - train_groups - 1)

    train_cutoff = train_groups
    validation_cutoff = train_groups + validation_groups
    return train_cutoff, validation_cutoff


def assign_group_splits(conn, train_ratio=0.70, validation_ratio=0.15):
    total_groups = conn.execute('SELECT COUNT(*) FROM group_first_seen').fetchone()[0]
    if total_groups < 3:
        raise RuntimeError('At least three sequence groups are required to build train/validation/test splits.')

    train_cutoff, validation_cutoff = compute_split_boundaries(total_groups, train_ratio, validation_ratio)
    conn.execute('DELETE FROM group_splits')

    insert_batch = []
    split_counts = {'train': 0, 'validation': 0, 'test': 0}
    ordered_groups = conn.execute(
        '''
        SELECT sequence_group_id
        FROM group_first_seen
        ORDER BY first_seen, sequence_group_id
        '''
    )

    for group_index, (group_id_key,) in enumerate(ordered_groups):
        if group_index < train_cutoff:
            split_name = 'train'
        elif group_index < validation_cutoff:
            split_name = 'validation'
        else:
            split_name = 'test'

        insert_batch.append((group_id_key, split_name))
        split_counts[split_name] += 1

        if len(insert_batch) >= SQLITE_BATCH_SIZE:
            conn.executemany(
                'INSERT INTO group_splits (sequence_group_id, split_name) VALUES (?, ?)',
                insert_batch,
            )
            conn.commit()
            insert_batch.clear()

    if insert_batch:
        conn.executemany(
            'INSERT INTO group_splits (sequence_group_id, split_name) VALUES (?, ?)',
            insert_batch,
        )
        conn.commit()

    print(f"   -> group split counts: {split_counts}")


def fetch_group_split_mapping(conn, unique_group_ids):
    group_id_keys = [format_group_id_key(value) for value in unique_group_ids]
    mapping = {}

    for start in range(0, len(group_id_keys), SQLITE_BATCH_SIZE):
        batch = group_id_keys[start:start + SQLITE_BATCH_SIZE]
        placeholders = ','.join('?' for _ in batch)
        query = (
            'SELECT sequence_group_id, split_name '
            f'FROM group_splits WHERE sequence_group_id IN ({placeholders})'
        )
        mapping.update(conn.execute(query, batch).fetchall())

    return pd.DataFrame({
        'sequence_group_id': np.array([np.uint64(int(key, 16)) for key in mapping], dtype=np.uint64),
        'split_name': list(mapping.values()),
    })


def process_source_csv(file_path, dataset_source, raw_shard_dir, conn):
    raw_shard_paths = []

    for chunk_index, chunk in enumerate(
        pd.read_csv(file_path, chunksize=CSV_CHUNK_SIZE, low_memory=False),
        start=1,
    ):
        if MAX_CHUNKS_PER_FILE and chunk_index > MAX_CHUNKS_PER_FILE:
            break

        chunk = downcast_chunk(chunk)
        chunk['DATASET_SOURCE'] = dataset_source
        chunk['sequence_group_id'] = build_sequence_group_ids(chunk).to_numpy(dtype=np.uint64)

        update_group_first_seen(conn, chunk)

        shard_path = os.path.join(raw_shard_dir, f'raw_{len(raw_shard_paths):05d}_{dataset_source}.parquet')
        write_parquet_frame(chunk, shard_path)
        raw_shard_paths.append(shard_path)

        print(
            f"      [{dataset_source}] chunk {chunk_index}: "
            f"rows={len(chunk):,} | shard={os.path.basename(shard_path)}"
        )

        del chunk
        gc.collect()

    return raw_shard_paths


def partition_raw_shards(raw_shard_paths, conn, bucket_part_dir):
    for raw_index, raw_shard_path in enumerate(raw_shard_paths, start=1):
        shard_df = pd.read_parquet(raw_shard_path)
        unique_group_ids = np.unique(shard_df['sequence_group_id'].to_numpy(dtype=np.uint64))
        split_mapping = fetch_group_split_mapping(conn, unique_group_ids)
        shard_df = shard_df.merge(split_mapping, on='sequence_group_id', how='left', copy=False)

        if shard_df['split_name'].isna().any():
            raise RuntimeError(f'Missing split assignment while processing {raw_shard_path}.')

        bucket_ids = (
            shard_df['sequence_group_id'].to_numpy(dtype=np.uint64) % np.uint64(SORT_BUCKET_COUNT)
        ).astype(np.uint16)
        shard_df['bucket_id'] = bucket_ids

        for (split_name, bucket_id), bucket_df in shard_df.groupby(['split_name', 'bucket_id'], sort=False):
            bucket_dir = os.path.join(bucket_part_dir, split_name, f'bucket_{int(bucket_id):03d}')
            os.makedirs(bucket_dir, exist_ok=True)
            output_path = os.path.join(bucket_dir, f'part_{raw_index:05d}.parquet')
            write_parquet_frame(bucket_df.drop(columns=['split_name', 'bucket_id']), output_path)

        os.remove(raw_shard_path)
        print(f"   -> partitioned raw shard {raw_index}/{len(raw_shard_paths)}")

        del shard_df
        del split_mapping
        gc.collect()


def finalize_bucketed_splits(bucket_part_dir, parquet_dir):
    ensure_clean_dir(parquet_dir)
    data_files = {'train': [], 'validation': [], 'test': []}

    for split_name in data_files:
        split_bucket_root = os.path.join(bucket_part_dir, split_name)
        if not os.path.isdir(split_bucket_root):
            continue

        split_output_dir = os.path.join(parquet_dir, split_name)
        os.makedirs(split_output_dir, exist_ok=True)

        for bucket_name in sorted(os.listdir(split_bucket_root)):
            bucket_dir = os.path.join(split_bucket_root, bucket_name)
            part_files = sorted(glob.glob(os.path.join(bucket_dir, '*.parquet')))
            if not part_files:
                continue

            bucket_df = pd.concat((pd.read_parquet(part_file) for part_file in part_files), ignore_index=True)
            bucket_df = bucket_df.sort_values(
                by=['sequence_group_id', 'FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS'],
                kind='mergesort',
            ).reset_index(drop=True)

            output_path = os.path.join(split_output_dir, f'{split_name}_{bucket_name}.parquet')
            write_parquet_frame(bucket_df, output_path)
            data_files[split_name].append(output_path)

            print(
                f"   -> finalized {split_name}/{bucket_name}: "
                f"rows={len(bucket_df):,} | files={len(part_files)}"
            )

            del bucket_df
            gc.collect()

    return data_files


def build_transformer_dataset():
    print('1. Loading NetFlow v3 Feature Metadata...')
    metadata_path = os.path.join(BASE_DIR, 'init_datasets', 'NetFlow_v3_Features.csv')
    df_features = pd.read_csv(metadata_path)
    print(f"Total features defined: {df_features.shape[0]}\n")

    print('2. Initiating Disk-Backed Data Stream...')
    ensure_clean_dir(TEMP_DIR)
    os.makedirs(RAW_SHARD_DIR, exist_ok=True)
    os.makedirs(BUCKET_PART_DIR, exist_ok=True)

    cic_path = os.path.join(BASE_DIR, 'init_datasets', 'NF-CICIDS2018-v3.csv')
    unsw_path = os.path.join(BASE_DIR, 'init_datasets', 'NF-UNSW-NB15-v3.csv')

    conn = create_group_tracking_db(GROUP_DB_PATH)
    try:
        print('   -> Streaming CIC-IDS2018 (Canada) in chunks...')
        raw_shard_paths = process_source_csv(cic_path, 'NF-CICIDS2018-v3', RAW_SHARD_DIR, conn)

        print('   -> Streaming UNSW-NB15 (Australia) in chunks...')
        raw_shard_paths.extend(process_source_csv(unsw_path, 'NF-UNSW-NB15-v3', RAW_SHARD_DIR, conn))

        print('\n3. Computing chronological group split assignments...')
        assign_group_splits(conn)

        print('\n4. Partitioning raw shards into split buckets...')
        partition_raw_shards(raw_shard_paths, conn, BUCKET_PART_DIR)
    finally:
        conn.close()

    print('\n5. Finalizing ordered split shards...')
    data_files = finalize_bucketed_splits(BUCKET_PART_DIR, PARQUET_DIR)

    print('\n6. Loading split artifacts via Apache Arrow Memory-Mapping...')
    split_dataset = load_dataset('parquet', data_files=data_files)

    print('7. Saving transformer-ready artifacts...')
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    split_dataset.save_to_disk(OUTPUT_DIR)

    shutil.rmtree(TEMP_DIR, ignore_errors=True)

    print(f"Train set shape: {split_dataset['train'].shape}")
    print(f"Validation set shape: {split_dataset['validation'].shape}")
    print(f"Test set shape: {split_dataset['test'].shape}")
    print('✅ Pipeline Complete. Ready for PyTorch DataLoader.')


if __name__ == '__main__':
    build_transformer_dataset()