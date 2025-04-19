import os
import json
import pandas as pd
import hcl2
from typing import Dict, List, Any, Optional
import re
from tqdm.auto import tqdm

def load_all_commit_jsonl(dataset_dir: str) -> List[Dict[str, Any]]:
    """
    Traverse dataset_dir, find all .jsonl files and add 'repo' field.
    """
    commits = []
    for root, _, files in os.walk(dataset_dir):
        repo = os.path.basename(root)
        for fname in files:
            if fname.endswith('.jsonl'):
                path = os.path.join(root, fname)
                with open(path, encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line)
                        data['repo'] = repo
                        commits.append(data)
    return commits

def parse_patch_to_dataframe(patch: str) -> pd.DataFrame:
    """
    Transform unified patch into DataFrame with enhanced information:
    - file: filename
    - old_lineno, new_lineno: line numbers in original and new file
    - hunk_id: which hunk in the patch this line belongs to
    - change: added/removed/context/meta
    - content: the actual line content
    """
    rows = []
    current_file = None
    current_hunk_id = 0
    old_lineno = 0
    new_lineno = 0
    
    for line in patch.splitlines():
        if line.startswith('--- '):
            current_file = line[4:].strip()
            change = 'meta'
            content = line
            rows.append({
                'file': current_file,
                'old_lineno': None,
                'new_lineno': None,
                'hunk_id': None,
                'change': change,
                'content': content
            })
        elif line.startswith('@@ '):
            current_hunk_id += 1
            change = 'meta'
            content = line
            
            match = re.search(r'@@ -(\d+),\d+ \+(\d+),\d+ @@', line)
            if match:
                old_lineno = int(match.group(1))
                new_lineno = int(match.group(2))
            
            rows.append({
                'file': current_file,
                'old_lineno': None,
                'new_lineno': None,
                'hunk_id': current_hunk_id,
                'change': change,
                'content': content
            })
        elif line.startswith('+') and not line.startswith('+++'):
            change = 'added'
            content = line[1:]
            rows.append({
                'file': current_file,
                'old_lineno': None,
                'new_lineno': new_lineno,
                'hunk_id': current_hunk_id,
                'change': change,
                'content': content
            })
            new_lineno += 1
        elif line.startswith('-') and not line.startswith('---'):
            change = 'removed'
            content = line[1:]
            rows.append({
                'file': current_file,
                'old_lineno': old_lineno,
                'new_lineno': None,
                'hunk_id': current_hunk_id,
                'change': change,
                'content': content
            })
            old_lineno += 1
        else:
            change = 'context'
            content = line
            rows.append({
                'file': current_file,
                'old_lineno': old_lineno,
                'new_lineno': new_lineno,
                'hunk_id': current_hunk_id,
                'change': change,
                'content': content
            })
            old_lineno += 1
            new_lineno += 1
    
    return pd.DataFrame(rows)

def parse_hcl_snippet(txt: str) -> Dict:
    """
    Parse Terraform HCL snippet to extract semantic information.
    """
    try:
        return hcl2.loads(txt + '\n')
    except Exception:
        return {}

def is_bugfix_commit(message: str) -> bool:
    """
    Check if commit message indicates a bug fix.
    """
    pattern = r"(fix|bug|issue|error|crash|problem|fail|resolve|patch)"
    return bool(re.search(pattern, message.lower()))

def enrich_dataframe_with_terraform_semantics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Terraform semantic information to the DataFrame.
    """
    # TODO implement a better way to extract resource info to terraform
    def extract_resource_info(content: str) -> Dict:
        resource_match = re.search(r'resource\s+"([^"]+)"\s+"([^"]+)"\s+{', content)
        if resource_match:
            return {
                'resource_type': resource_match.group(1),
                'resource_name': resource_match.group(2)
            }
        
        attr_match = re.search(r'(\w+)\s+=\s+(.+)', content)
        if attr_match:
            return {
                'attr_name': attr_match.group(1),
                'attr_value': attr_match.group(2).strip()
            }
        
        return {}
    
    enriched_rows = []
    for _, row in df.iterrows():
        new_row = row.to_dict()
        if row['change'] in ['added', 'removed']:
            info = extract_resource_info(row['content'])
            new_row.update(info)
        enriched_rows.append(new_row)
    
    return pd.DataFrame(enriched_rows)