import os
import re
import logging
import argparse
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

def extract_group(id_val):
    if not isinstance(id_val, str):
        return 'unknown'
    # 1. Segment-based splits
    if '_seg' in id_val:
        return id_val.split('_seg')[0]
    if '__' in id_val:
        return id_val.split('__')[0]
    # 2. Laws: gesetz_NAME_NUMBER
    match = re.match(r'^(gesetz_[a-zA-Z0-9]+)_\d+$', id_val)
    if match:
        return match.group(1)
    # 3. Europarl corpus: eupdcorp_NUMBER_NUMBER
    match = re.match(r'^(eupdcorp_\d+)_\d+$', id_val)
    if match:
        return match.group(1)
    # 4. Bundestag: NUMBER_SPEAKER_NUMBER
    match = re.match(r'^(\d+_[^_]+)_\d+$', id_val)
    if match:
        return match.group(1)
    return id_val

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset split from training CSV.")
    parser.add_argument("--input", default="Data/Final/training_pair_v5.csv", help="Input CSV path")
    parser.add_argument("--outdir", default="Data", help="Output directory for splits")
    args = parser.parse_args()

    np.random.seed(42)
    input_path = args.input
    output_dir = args.outdir
    os.makedirs(output_dir, exist_ok=True)

    log.info(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    log.info(f"Loaded {len(df):,} rows.")

    # Map source to domain for downstream callbacks/scripts
    df['source'] = df['domain']

    # Extract group IDs
    log.info("Extracting group IDs...")
    df['group_id'] = df['id'].apply(extract_group)

    # Compute group statistics
    log.info("Computing group statistics...")
    group_df = df.groupby('group_id').agg(
        domain=('domain', 'first'),
        label=('label', 'first'),
        row_count=('label', 'count')
    ).reset_index()

    # Define splitting targets
    splits = {
        'train': 0.70,
        'val': 0.10,
        'external_val': 0.10,
        'test': 0.05,
        'final_holdout': 0.05
    }

    # Initialize container for group assignment
    group_df['split'] = None

    # Perform stratified group split
    log.info("Performing stratified group splitting...")
    strata = group_df.groupby(['domain', 'label'])
    
    for (domain, label), stratum_groups in strata:
        # Shuffle groups within stratum
        shuffled_groups = stratum_groups.sample(frac=1, random_state=42).reset_index(drop=True)
        total_rows = shuffled_groups['row_count'].sum()
        
        # Calculate row counts target for each split
        targets = {name: int(total_rows * prop) for name, prop in splits.items()}
        
        # Distribute groups to splits
        current_counts = {name: 0 for name in splits}
        assigned_splits = []
        
        for idx, row in shuffled_groups.iterrows():
            g_rows = row['row_count']
            
            # Find the split that is furthest behind its target relative to current count
            chosen_split = None
            min_deficit = float('-inf')
            
            for s_name in splits:
                deficit = targets[s_name] - current_counts[s_name]
                if deficit > min_deficit:
                    min_deficit = deficit
                    chosen_split = s_name
            
            current_counts[chosen_split] += g_rows
            assigned_splits.append(chosen_split)
            
        shuffled_groups['split'] = assigned_splits
        
        # Map back to group_df
        group_df.loc[group_df['group_id'].isin(shuffled_groups['group_id']), 'split'] = group_df['group_id'].map(
            shuffled_groups.set_index('group_id')['split']
        )

    # Map split back to the main DataFrame
    df = df.merge(group_df[['group_id', 'split']], on='group_id', how='left')

    log.info("Verifying splits...")
    # Write splits
    for s_name in splits:
        split_df = df[df['split'] == s_name].drop(columns=['group_id', 'split'])
        # Shuffle rows of the split
        split_df = split_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        out_path = os.path.join(output_dir, f"{s_name}.csv")
        split_df.to_csv(out_path, index=False, encoding="utf-8")
        
        # Print breakdown
        n_rows = len(split_df)
        label_counts = split_df['label'].value_counts()
        domain_counts = split_df['domain'].value_counts()
        log.info(f"Split {s_name} written to {out_path}: {n_rows:,} rows")
        log.info(f"  Labels: AI={label_counts.get(1, 0):,}, Human={label_counts.get(0, 0):,}")
        log.info(f"  Domains: " + ", ".join([f"{k}={v:,}" for k, v in domain_counts.items()]))

    # Double check leakages
    train_groups = set(df[df['split'] == 'train']['group_id'])
    val_groups = set(df[df['split'] == 'val']['group_id'])
    ext_val_groups = set(df[df['split'] == 'external_val']['group_id'])
    test_groups = set(df[df['split'] == 'test']['group_id'])
    fh_groups = set(df[df['split'] == 'final_holdout']['group_id'])

    log.info("Checking group leakage between splits:")
    log.info(f"  Train & Val overlap: {len(train_groups & val_groups)}")
    log.info(f"  Train & Ext Val overlap: {len(train_groups & ext_val_groups)}")
    log.info(f"  Train & Test overlap: {len(train_groups & test_groups)}")
    log.info(f"  Train & Final Holdout overlap: {len(train_groups & fh_groups)}")
    log.info(f"  Val & Test overlap: {len(val_groups & test_groups)}")
    log.info(f"  Ext Val & Final Holdout overlap: {len(ext_val_groups & fh_groups)}")
    
    total_assigned_rows = sum([len(df[df['split'] == s]) for s in splits])
    log.info(f"Assigned rows: {total_assigned_rows:,} out of {len(df):,}")

if __name__ == "__main__":
    main()
