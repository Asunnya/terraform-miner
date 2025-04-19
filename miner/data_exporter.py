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

    def export(self, full_name, entries):
        """
        Save entries list of dicts under dataset/{owner_repo}/data.jsonl and CSV.
        """
        owner, name = full_name.split('/')
        repo_dir = os.path.join(self.base_path, f"{owner}_{name}")
        os.makedirs(repo_dir, exist_ok=True)
        jsonl_path = os.path.join(repo_dir, 'data.jsonl')
        with open(jsonl_path, 'w') as f:
            for e in entries:
                f.write(json.dumps(e) + '\n')
        df = pd.DataFrame(entries)
        df.to_csv(os.path.join(repo_dir, 'data.csv'), index=False)
