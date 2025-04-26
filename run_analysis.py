import os
import time
import pandas as pd
from pathlib import Path
from src.analysis.config import get_config
from src.analysis.diff_utils import (
    load_all_commit_jsonl, 
    parse_patch_to_dataframe, 
    enrich_dataframe_with_terraform_semantics
)

def main():
    # Get configuration
    config = get_config()
    dataset_dir = config['dataset_dir']
    output_dir = config['output_dir']
    
    # Ensure all required directories exist
    for dir_path in [
        output_dir,
        os.path.join(output_dir, 'commits'),
        os.path.join(output_dir, 'diffs'),
        os.path.join(output_dir, 'metrics'),
        os.path.join(output_dir, 'visualizations'),
        os.path.join(output_dir, 'analysis')
    ]:
        os.makedirs(dir_path, exist_ok=True)

    # Load commits
    print(f"Loading commits from {dataset_dir}...")
    start_time = time.time()
    commits = load_all_commit_jsonl(dataset_dir)
    print(f"Loaded {len(commits)} commits in {time.time() - start_time:.2f} seconds")

    # Process commits
    rows = []
    for commit in commits:
        repo = commit.get('repo')
        hash_ = commit.get('commit_hash')
        patch = commit.get('patch', '')
        
        if not patch:
            continue
            
        df = parse_patch_to_dataframe(patch)
        if df.empty:
            continue
            
        df = enrich_dataframe_with_terraform_semantics(df)
        
        # Add metadata
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            row_dict.update({
                'repo': repo,
                'commit': hash_,
                'date': commit.get('date'),
                'file_path': commit.get('file', '')
            })
            rows.append(row_dict)

    # Create DataFrame
    df_all = pd.DataFrame(rows)
    print(f"\nProcessed {len(commits)} commits, resulting in {len(df_all)} diff lines")

    # Save results
    parquet_path = os.path.join(output_dir, 'analysis', 'terraform_diffs.parquet')
    df_all.to_parquet(parquet_path, index=False)
    print(f"Data written to {parquet_path}")

    # Print some basic statistics
    print("\nBasic Statistics:")
    print(f"Number of repositories: {df_all['repo'].nunique()}")
    print(f"Number of commits: {df_all['commit'].nunique()}")
    if 'resource_type' in df_all.columns:
        print(f"Number of unique resource types: {df_all['resource_type'].nunique()}")
    print(f"Number of files modified: {df_all['file_path'].nunique()}")

if __name__ == '__main__':
    main() 