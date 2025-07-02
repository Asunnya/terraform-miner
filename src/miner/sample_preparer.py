"""
sample_preparer.py

This module provides functions to prepare samples and extract features 
from the mined commit data. It is intended to be called by the main mining
script after the initial data export (data.jsonl files).

Key functions:
- create_all_commits_csv: Aggregates data from multiple data.jsonl files into a single CSV.
- confirm_date_range: Determines the min and max commit dates from the aggregated CSV.
- prepare_sample: Creates a stratified sample of commits.
- compute_token_coverage: Calculates token coverage for the most frequent tokens.
- extract_bigrams: Extracts the most frequent bigrams from commit messages.
"""
import os
import glob
import json
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter, defaultdict
import re
import csv
from math import ceil

# --- Utility functions (adapted from nlp_analysis/phase1.py) ---
# These are included here to make sample_preparer.py self-contained or 
# until they are moved to a shared utils module.

PATCH_EXCERPT_LINES = 10
MSG_CLEAN_MAX_LEN = 150
RELEVANT_EXTENSIONS_ORDER = ['.tf', '.go', '.py', '.yaml', '.yml', '.json', '.sh', '.hcl']
EXCLUDED_DIRS_PATTERNS = [ # Not used by current functions in this script directly but kept for context if enhancing get_relevant_file
    r"vendor/", r"test/fixtures/", r"examples/", r"tests/", r"testdata/",
    r"\.github/", r"docs/"
]
MERGE_PATTERNS_FOR_INFERENCE = [ # Used for inferring 'is_merge'
    r"^Merge pull request #\d+ from .*",
    r"^Merge remote-tracking branch '.*'",
    r"^Merge branch '.*'( of .*)?( into .*)?",
    r"^\s*Merged in .*",
]


def clean_commit_message_advanced(message, max_len=MSG_CLEAN_MAX_LEN):
    if not message: return None
    msg = message.splitlines()[0]
    msg = re.sub(r"(\s|^)(#|GH-)\d+\b", " ", msg)
    msg = msg.lower()
    msg = re.sub(r"https?://\S+", " ", msg)
    msg = re.sub(r"\s+", " ", msg).strip()
    if len(msg) > max_len:
        msg = msg[:max_len].rsplit(' ', 1)[0] if ' ' in msg[:max_len] else msg[:max_len]
    return msg if msg else None

def get_relevant_file_and_patch_original(modifications_list):
    selected_file_path = None
    patch_excerpt = ""
    best_mod_info = None

    for ext in RELEVANT_EXTENSIONS_ORDER:
        for mod_info in modifications_list:
            if mod_info.get('filename') and mod_info['filename'].endswith(ext):
                best_mod_info = mod_info
                break
        if best_mod_info:
            break

    if not best_mod_info and modifications_list:
        best_mod_info = modifications_list[0] 

    if best_mod_info and best_mod_info.get('diff'):
        selected_file_path = best_mod_info['filename']
        diff_lines = best_mod_info['diff'].splitlines()
        patch_excerpt = "\n".join(diff_lines[:PATCH_EXCERPT_LINES])
    
    return selected_file_path, patch_excerpt

def get_relevant_file_and_patch_enhanced(modifications_list, analysis_modules=None):
    # For now, enhanced is the same as original. This can be expanded later.
    # analysis_modules is not used here yet.
    return get_relevant_file_and_patch_original(modifications_list)

# --- Core Functions ---

def create_all_commits_csv(json_glob_pattern, output_csv_path):
    """
    Reads all data.jsonl files matching the glob pattern, processes commit data,
    and saves it to a single CSV file.

    Each row in the output CSV represents a unique commit with its aggregated data.
    """
    print(f"Starting: Create all_commits.csv from {json_glob_pattern}")
    commits_grouped_temp_data = {}

    json_file_paths = glob.glob(json_glob_pattern)
    if not json_file_paths:
        print(f"WARNING: No .jsonl files found for pattern {json_glob_pattern}. Cannot create {output_csv_path}")
        return False

    print(f"Found {len(json_file_paths)} JSONL files to process.")

    for json_file_path in json_file_paths:
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        record = json.loads(line.strip())
                    except json.JSONDecodeError as e:
                        print(f"WARNING: JSON decode error in {json_file_path} line {line_num+1}: {e}")
                        continue

                    repo_name = record.get('repo')
                    commit_hash = record.get('commit_hash')
                    
                    if not repo_name or not commit_hash:
                        continue

                    commit_key = (repo_name, commit_hash)

                    if commit_key not in commits_grouped_temp_data:
                        original_msg = record.get('message', "")
                        is_merge_inferred = any(re.match(pattern, original_msg, re.IGNORECASE) for pattern in MERGE_PATTERNS_FOR_INFERENCE)
                        
                        commits_grouped_temp_data[commit_key] = {
                            "repo_name": repo_name,
                            "commit_hash": commit_hash,
                            "msg_original": original_msg,
                            "author_name": record.get('author'),
                            "date": record.get('date'), # Keep as string initially, pandas will parse
                            "modifications_raw": [], # Store raw modifications to select best later
                            "is_merge": is_merge_inferred,
                        }
                    
                    modification_item = {
                        "filename": record.get('file'),
                        "diff": record.get('patch'),
                    }
                    commits_grouped_temp_data[commit_key]["modifications_raw"].append(modification_item)
            
        except Exception as e:
            print(f"ERROR: Processing file {json_file_path}: {e}")

    if not commits_grouped_temp_data:
        print("WARNING: No commit data was loaded. all_commits.csv will not be created.")
        return False

    processed_commits_for_df = []
    for commit_key, data in commits_grouped_temp_data.items():
        msg_clean = clean_commit_message_advanced(data['msg_original'])
        relevant_file, patch_excerpt = get_relevant_file_and_patch_enhanced(data['modifications_raw'])

        processed_commits_for_df.append({
            'commit_hash': data['commit_hash'],
            'repo_name': data['repo_name'],
            'date': data['date'],
            'author_name': data['author_name'],
            'msg_original': data['msg_original'],
            'msg_clean': msg_clean if msg_clean else '', # Ensure empty string for CSV if None
            'relevant_file': relevant_file if relevant_file else '',
            'patch_excerpt': patch_excerpt if patch_excerpt else '',
            'is_merge': data['is_merge']
        })
    
    df_all_commits = pd.DataFrame(processed_commits_for_df)
    
    # Ensure 'date' column is proper datetime for any operations before saving if needed,
    # but for CSV, string is fine. Pandas to_csv handles datetime objects correctly.
    # df_all_commits['date'] = pd.to_datetime(df_all_commits['date'], errors='coerce')

    try:
        df_all_commits.to_csv(output_csv_path, index=False, quoting=csv.QUOTE_ALL)
        print(f"Successfully created {output_csv_path} with {len(df_all_commits)} unique commits.")
        return True
    except Exception as e:
        print(f"ERROR: Failed to save {output_csv_path}: {e}")
        return False

def confirm_date_range(all_commits_csv_path):
    """
    Confirms the minimum and maximum commit dates from the all_commits.csv file.
    """
    print(f"Starting: Confirm date range from {all_commits_csv_path}")
    if not os.path.exists(all_commits_csv_path):
        print(f"ERROR: {all_commits_csv_path} not found.")
        return None, None
    try:
        df = pd.read_csv(all_commits_csv_path, usecols=['date'], parse_dates=['date'], infer_datetime_format=True)
        if df['date'].isnull().all():
            print(f"WARNING: 'date' column in {all_commits_csv_path} has no valid dates.")
            return None, None
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        print(f"Date range confirmed: Min: {min_date}, Max: {max_date}")
        return min_date, max_date
    except Exception as e:
        print(f"ERROR: Could not confirm date range from {all_commits_csv_path}: {e}")
        return None, None

def prepare_sample(all_commits_csv_path, output_csv_path, target_n=1000):
    """
    Prepares a stratified sample from all_commits.csv.
    The stratification is by month.
    Outputs sample_messages_stratified.csv.
    """
    print(f"Starting: Prepare sample from {all_commits_csv_path}")
    if not os.path.exists(all_commits_csv_path):
        print(f"ERROR: {all_commits_csv_path} not found. Cannot prepare sample.")
        return False
        
    try:
        df = pd.read_csv(all_commits_csv_path, parse_dates=['date'], infer_datetime_format=True)
        df.dropna(subset=['date', 'msg_clean'], inplace=True) # Ensure date and msg_clean are present
        df = df[~df['is_merge']] # Exclude merge commits from sampling as per typical MSR practices

        if df.empty:
            print(f"WARNING: No valid data after loading and filtering merges from {all_commits_csv_path}. Sample not created.")
            return False

        months = df['date'].dt.to_period('M').unique()
        if not months.any(): # Check if months is empty or all NaT
             print(f"WARNING: No valid months found for stratification in {all_commits_csv_path}. Sample not created.")
             return False

        k = ceil(target_n / len(months)) if len(months) > 0 else target_n

        sample_df = (
            df.assign(month_period=df['date'].dt.to_period('M'))
            .groupby('month_period', group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), k), random_state=42) if len(g) > 0 else None) # Handle empty groups
            .reset_index(drop=True)
        )
        
        sample_df.dropna(how='all', inplace=True) # Remove rows that might have become all None if a group was empty

        if sample_df.empty:
            print(f"WARNING: Stratified sample is empty. No sample created from {all_commits_csv_path}.")
            return False

        # Select columns for the final sample CSV as per typical sample_messages.csv
        output_columns = ['commit_hash', 'msg_clean', 'relevant_file', 'patch_excerpt']
        # Ensure all expected columns are present, fill with empty string if not
        for col in output_columns:
            if col not in sample_df.columns:
                sample_df[col] = ''
        
        sample_df_output = sample_df[output_columns]

        sample_df_output.to_csv(output_csv_path, index=False, quoting=csv.QUOTE_ALL)
        print(f"Successfully created stratified sample {output_csv_path} with {len(sample_df_output)} commits.")
        print(f"Months covered: {sample_df['month_period'].nunique() if 'month_period' in sample_df else 0}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to prepare sample from {all_commits_csv_path}: {e}")
        return False

def compute_token_coverage(all_commits_csv_path, top_n=50):
    """
    Calculates the coverage of the top_n most frequent tokens from msg_clean in all_commits.csv.
    """
    print(f"Starting: Compute token coverage from {all_commits_csv_path}")
    if not os.path.exists(all_commits_csv_path):
        print(f"ERROR: {all_commits_csv_path} not found.")
        return 0.0

    try:
        df = pd.read_csv(all_commits_csv_path, usecols=['msg_clean'])
        df.dropna(subset=['msg_clean'], inplace=True)
        
        if df.empty:
            print(f"WARNING: No 'msg_clean' data in {all_commits_csv_path} for token coverage.")
            return 0.0

        tokens = [tok for msg in df['msg_clean'] for tok in str(msg).lower().split()]
        if not tokens:
            print(f"WARNING: No tokens found after processing 'msg_clean' from {all_commits_csv_path}.")
            return 0.0
            
        total_tokens = len(tokens)
        freq = Counter(tokens)
        top_n_freq = freq.most_common(top_n)
        
        coverage_count = sum(count for _, count in top_n_freq)
        coverage_percentage = (coverage_count / total_tokens) if total_tokens > 0 else 0.0
        
        print(f"Coverage for top-{top_n} tokens: {coverage_percentage:.2%}")
        return coverage_percentage
    except Exception as e:
        print(f"ERROR: Failed to compute token coverage from {all_commits_csv_path}: {e}")
        return 0.0

def extract_bigrams(all_commits_csv_path, top_k=20):
    """
    Extracts the top_k most frequent bigrams from msg_clean in all_commits.csv.
    Requires scikit-learn.
    """
    print(f"Starting: Extract bigrams from {all_commits_csv_path}")
    try:
        from sklearn.feature_extraction.text import CountVectorizer
    except ImportError:
        print("ERROR: scikit-learn is required to extract bigrams. Please install it.")
        return []

    if not os.path.exists(all_commits_csv_path):
        print(f"ERROR: {all_commits_csv_path} not found.")
        return []

    try:
        df = pd.read_csv(all_commits_csv_path, usecols=['msg_clean'])
        df.dropna(subset=['msg_clean'], inplace=True)

        if df.empty:
            print(f"WARNING: No 'msg_clean' data in {all_commits_csv_path} for bigram extraction.")
            return []

        # Ensure messages are strings
        msgs = df['msg_clean'].astype(str).tolist()
        
        if not msgs:
            print(f"WARNING: No messages available for bigram extraction after filtering.")
            return []

        try:
            # Attempt to use 'english' stopwords list; if it fails, proceed without it.
            vect = CountVectorizer(ngram_range=(2,2), stop_words='english')
            X = vect.fit_transform(msgs)
        except ValueError as ve:
            # This can happen if 'english' stopwords list is not found or if vocabulary is empty
            print(f"WARNING: CountVectorizer failed with stop_words='english' ({ve}). Retrying without stopwords.")
            vect = CountVectorizer(ngram_range=(2,2))
            X = vect.fit_transform(msgs)


        sums = X.sum(axis=0).A1
        # get_feature_names_out() is preferred over get_feature_names() in newer sklearn
        feature_names = vect.get_feature_names_out() if hasattr(vect, 'get_feature_names_out') else vect.get_feature_names()
        
        bigrams_with_counts = sorted(zip(feature_names, sums), key=lambda x: -x[1])
        
        top_bigrams_list = [[bigram, int(count)] for bigram, count in bigrams_with_counts[:top_k]] # Store count as int
        
        print(f"Top-{top_k} bigrams extracted: {top_bigrams_list}")
        return top_bigrams_list
    except Exception as e:
        print(f"ERROR: Failed to extract bigrams from {all_commits_csv_path}: {e}")
        return []

if __name__ == '__main__':
    # Example Usage (assuming script is in src/miner/ and data is in ../../data/dataset, reports in ../../reports)
    print("Running sample_preparer.py example usage...")
    
    # Create dummy project structure for testing if it doesn't exist
    # In a real scenario, these paths would be relative to the project root
    # and data/dataset would be populated by the mining process.
    # Reports directory should also exist.
    
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_dir = os.path.abspath(os.path.join(current_script_dir, '..', '..')) #terraform-miner/
    
    test_data_dir = os.path.join(project_root_dir, 'data', 'dataset', 'test_repo')
    os.makedirs(test_data_dir, exist_ok=True)
    reports_dir = os.path.join(project_root_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)

    # Create a dummy data.jsonl for testing
    dummy_jsonl_path = os.path.join(test_data_dir, 'data.jsonl')
    dummy_commits = [
        {"repo": "test_repo", "commit_hash": "abc1", "message": "Fix: resolve issue #123. Added new feature.", "author": "dev1", "date": "2023-01-15T10:00:00Z", "file": "main.tf", "patch": "diff --git a/main.tf b/main.tf\nindex ..."},
        {"repo": "test_repo", "commit_hash": "abc1", "message": "Fix: resolve issue #123. Added new feature.", "author": "dev1", "date": "2023-01-15T10:00:00Z", "file": "variables.tf", "patch": "diff --git a/variables.tf b/variables.tf\nindex ..."},
        {"repo": "test_repo", "commit_hash": "def2", "message": "Feat: add new component for terraform provider", "author": "dev2", "date": "2023-02-20T12:30:00Z", "file": "module/resource.tf", "patch": "diff --git a/module/resource.tf b/module/resource.tf\nindex ..."},
        {"repo": "test_repo", "commit_hash": "ghi3", "message": "Refactor: cleanup code and improve docs.", "author": "dev1", "date": "2023-02-22T15:00:00Z", "file": "README.md", "patch": "diff --git a/README.md b/README.md\nindex ..."},
        {"repo": "test_repo", "commit_hash": "jkl4", "message": "Merge pull request #42 from feature/new-things", "author": "dev-bot", "date": "2023-03-10T18:00:00Z", "file": "main.go", "patch": "diff --git a/main.go b/main.go\nindex ..."}
    ]
    with open(dummy_jsonl_path, 'w', encoding='utf-8') as f:
        for commit_data in dummy_commits:
            f.write(json.dumps(commit_data) + '\n')

    glob_pattern = os.path.join(project_root_dir, 'data', 'dataset', '*', 'data.jsonl')
    all_commits_csv = os.path.join(reports_dir, 'all_commits.csv')
    stratified_sample_csv = os.path.join(reports_dir, 'sample_messages_stratified.csv')

    # 1. Create all_commits.csv
    if create_all_commits_csv(glob_pattern, all_commits_csv):
        # 2. Confirm date range
        min_d, max_d = confirm_date_range(all_commits_csv)
        if min_d and max_d:
             print(f"Min date: {min_d}, Max date: {max_d}")

        # 3. Prepare sample
        prepare_sample(all_commits_csv, stratified_sample_csv, target_n=2) # Small sample for test

        # 4. Compute token coverage
        coverage = compute_token_coverage(all_commits_csv, top_n=10)
        print(f"Top-10 token coverage: {coverage:.2%}")

        # 5. Extract bigrams
        bigrams = extract_bigrams(all_commits_csv, top_k=5)
        print(f"Top-5 bigrams: {bigrams}")
        
        # Optional: Save bigrams to a JSON file like main.py would
        top_bigrams_output_path = os.path.join(reports_dir, 'top_bigrams.json')
        try:
            with open(top_bigrams_output_path, 'w', encoding='utf-8') as f_out_bigram:
                json.dump(bigrams, f_out_bigram, indent=2)
            print(f"Dummy top_bigrams.json saved to {top_bigrams_output_path}")
        except Exception as e_json_save:
            print(f"Error saving dummy top_bigrams.json: {e_json_save}")
            
    print("\nExample usage finished.")
    # Clean up dummy file (optional)
    # os.remove(dummy_jsonl_path)
    # os.remove(all_commits_csv)
    # os.remove(stratified_sample_csv)
    # os.remove(top_bigrams_output_path) 