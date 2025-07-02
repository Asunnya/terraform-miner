import os
import json
import pandas as pd

class DataExporter:
    """
    Export mined commit entries to JSONL or CSV per repo.
    """
    def __init__(self, base_path='dataset'):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def export(self, full_name, entries, append=False):
        """
        Save entries list of dicts under dataset/{owner_repo}/data.jsonl and CSV.
        Supports appending to existing files for incremental mining.
        
        Args:
            full_name (str): Repository name in format 'owner/repo'
            entries (list): List of commit entries to export
            append (bool): If True, append to existing files instead of overwriting
        """
        if not entries:
            return
            
        owner, name = full_name.split('/')
        repo_dir = os.path.join(self.base_path, f"{owner}_{name}")
        os.makedirs(repo_dir, exist_ok=True)
        
        jsonl_path = os.path.join(repo_dir, 'data.jsonl')
        csv_path = os.path.join(repo_dir, 'data.csv')
        
        # Track existing entries to avoid duplicates
        existing_entries = []
        
        if append and os.path.exists(jsonl_path):
            # Load existing data if we're appending
            try:
                with open(jsonl_path, 'r', encoding='utf-8') as f:
                    existing_entries = [json.loads(line) for line in f]
                    
                # Get set of existing commit hashes and files for duplicate detection
                existing_keys = {(e['commit_hash'], e['file']) for e in existing_entries}
                
                # Filter out entries that are already in the file
                new_entries = [
                    e for e in entries 
                    if (e['commit_hash'], e['file']) not in existing_keys
                ]
                
                # Combine existing and new entries
                all_entries = existing_entries + new_entries
            except Exception as e:
                import logging
                logging.warning(f"Error reading existing data, will overwrite: {e}")
                all_entries = entries
        else:
            all_entries = entries
            
        # Write JSONL file
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for e in all_entries:
                f.write(json.dumps(e) + '\n')
                
        # Create CSV from the full dataset
        df = pd.DataFrame(all_entries)
        df.to_csv(csv_path, index=False)
        
        import logging
        logging.info(f"Saved {len(entries)} new entries (total: {len(all_entries)}) to {repo_dir}")
