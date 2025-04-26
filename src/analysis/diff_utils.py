# %% [1] - Importações e configuração

import sys
import os
import json
import pandas as pd
import hcl2
import logging
import concurrent.futures
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import re
from tqdm.auto import tqdm
from pathlib import Path
import hashlib
from datetime import datetime

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("terraform_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("terraform_diff_utils")

# Configuração de caminhos
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from analysis.terraform_ast import TerraformAstAnalyzer
except ModuleNotFoundError:
    from terraform_ast import TerraformAstAnalyzer


# --------------------------------------------------------------------- #
# Exported API (usado por diff_stats e notebooks)                       #
# --------------------------------------------------------------------- #
CHANGE_CATEGORIES: dict[str, list[str]] = {
    "value_modification": ["=", "+=", "-="],                      # simples
    "structure_modification": ["{", "}", "[", "]"],              # estrutura
    "dependency_modification": ["depends_on", "count", "for_each"],  # deps
    "resource_definition": ["resource", "data", "module", "provider"],  # defs
}

# Add block-type categories for Terraform
BLOCK_TYPES = {
    'variable_definition': ['variable'],
    'locals_block': ['locals'],
    'output_definition': ['output'],
    'terraform_block': ['terraform'],
    'backend_definition': ['backend'],
}

__all__ = [
    "CHANGE_CATEGORIES",
    "parse_patch_to_dataframe",
    "categorize_change_line",
    "load_all_commit_jsonl",
    "summarize_patch_counts",
    "parse_hcl_snippet",
    "parse_hcl_with_ast",
    "parse_diff_fragment_with_ast",
    "is_bugfix_commit",
    "enrich_dataframe_with_terraform_semantics",
    "enrich_dataframe_with_terraform_ast",
    "analyze_terraform_changes",
    "CommitProcessor",
    "TerraformResource",
    "MutationOperator",
    "MutationOperatorDetector",
    "get_other_lines"
]

# %% [2] - Funções de carregamento e processamento básico de commits

def load_all_commit_jsonl(dataset_dir: str) -> List[Dict[str, Any]]:
    """
    Traverse dataset_dir, find all .jsonl files and add 'repo' field.
    
    Parameters
    ----------
    dataset_dir : str
        Directory containing the dataset to be analyzed.
        
    Returns
    -------
    list
        List of dictionaries containing commit information with repo field added.
    """
    commits = []
    repo_counts = {}
    
    logger.info(f"Loading commits from directory: {dataset_dir}")
    
    for root, _, files in os.walk(dataset_dir):
        repo = os.path.basename(root)
        if repo == os.path.basename(dataset_dir):  # Skip base directory
            continue
            
        for fname in files:
            if fname.endswith('.jsonl'):
                path = os.path.join(root, fname)
                try:
                    with open(path, encoding='utf-8') as f:
                        for line in f:
                            data = json.loads(line)
                            data['repo'] = repo
                            # Add unique ID for deduplication and tracking
                            unique_id = f"{repo}_{data.get('commit_hash', '')}_{data.get('file', '')}"
                            data['unique_id'] = hashlib.md5(unique_id.encode()).hexdigest()
                            commits.append(data)
                    
                    repo_counts[repo] = repo_counts.get(repo, 0) + 1
                except Exception as e:
                    logger.error(f"Error processing {path}: {str(e)}")
    
    logger.info(f"Total commits loaded: {len(commits)}")
    for repo, count in repo_counts.items():
        logger.info(f"  Repository {repo}: {count} commits")
            
    return commits


def parse_patch_to_dataframe(patch: str) -> pd.DataFrame:
    """
    Transform unified patch into DataFrame with enhanced information.
    
    Parameters
    ----------
    patch : str
        String containing the unified diff patch.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - file: filename
        - old_lineno, new_lineno: line numbers in original and new file
        - hunk_id: which hunk in the patch this line belongs to
        - change: added/removed/context/meta
        - content: the actual line content
        - category: categorization of the change (resource_definition, 
                    value_modification, etc.)
    """
    if not patch or not isinstance(patch, str):
        return pd.DataFrame(columns=['file', 'old_lineno', 'new_lineno', 'hunk_id', 
                                    'change', 'content', 'category'])
    
    rows = []
    current_file = None
    current_hunk_id = 0
    old_lineno = 0
    new_lineno = 0
    line_num = 0
    
    for line in patch.splitlines():
        category = categorize_change_line(line)
        
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
                'content': content,
                'category': category,
                'line_num': line_num
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
                'content': content,
                'category': category,
                'line_num': line_num
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
                'content': content,
                'category': category,
                'line_num': line_num
            })
            new_lineno += 1
            line_num += 1
        elif line.startswith('-') and not line.startswith('---'):
            change = 'removed'
            content = line[1:]
            rows.append({
                'file': current_file,
                'old_lineno': old_lineno,
                'new_lineno': None,
                'hunk_id': current_hunk_id,
                'change': change,
                'content': content,
                'category': category,
                'line_num': line_num
            })
            old_lineno += 1
            line_num += 1
        else:
            change = 'context'
            content = line
            rows.append({
                'file': current_file,
                'old_lineno': old_lineno,
                'new_lineno': new_lineno,
                'hunk_id': current_hunk_id,
                'change': change,
                'content': content,
                'category': category,
                'line_num': line_num
            })
            old_lineno += 1
            new_lineno += 1
            line_num += 1
    
    return pd.DataFrame(rows)


def categorize_change_line(line: str) -> str:
    """
    Categorize a change line in the patch with refined rule order and regex.
    """
    # 1. Meta diff headers
    if line.startswith('--- ') or line.startswith('+++ ') or line.startswith('@@ '):
        return 'meta'

    # 2. Strip git-change marker
    marker = line[:1]
    if marker in ['+', '-']:
        content = line[1:]
    else:
        content = line
    stripped = content.strip()

    # 3. Block headers: resource, data, module, provider
    m = re.match(r'^(resource|data|module|provider)\s+"([^"]+)"\s+"([^"]+)"', stripped)
    if m:
        kind = m.group(1)
        return 'resource_definition' if kind == 'resource' else f'{kind}_definition'

    # 4. Other Terraform blocks
    if stripped.startswith('variable '):
        return 'variable_definition'
    if stripped.startswith('output '):
        return 'output_definition'
    if stripped.startswith('locals '):
        return 'locals_block'
    if stripped.startswith('terraform '):
        return 'terraform_block'
    if stripped.startswith('backend '):
        return 'backend_definition'
    if stripped.startswith('dynamic '):
        return 'dynamic_block'

    # 5. Assignment (value modification)
    if re.match(r'^\s*[\w\.\-]+\s*=\s*.+', stripped):
        return 'value_modification'

    # 6. Dependency-related keywords
    if any(kw in stripped for kw in ['depends_on', 'count', 'for_each']):
        return 'dependency_modification'

    # 7. Structure-only lines (braces, brackets)
    if re.match(r'^\s*[{}\[\]]+\s*$', stripped):
        return 'structure_modification'

    # 8. Everything else
    return 'other'


# %% [3] - Funções de análise HCL e AST

def parse_hcl_snippet(txt: str) -> Dict:
    """
    Parse Terraform HCL snippet to extract semantic information.
    
    Parameters
    ----------
    txt : str
        HCL content as string.
        
    Returns
    -------
    dict
        Parsed HCL as dictionary.
        
    Notes
    -----
    This is a basic parser for backwards compatibility.
    For advanced parsing, use the TerraformAstAnalyzer.
    """
    if not txt or not isinstance(txt, str):
        logger.warning("Empty or non-string input provided to parse_hcl_snippet")
        return {}
        
    try:
        return hcl2.loads(txt + '\n')
    except Exception as e:
        logger.error(f"Error parsing HCL snippet: {str(e)}")
        return {}


def parse_hcl_with_ast(txt: str) -> Tuple[Dict, TerraformAstAnalyzer]:
    """
    Parse Terraform HCL snippet using the advanced AST analyzer.
    
    Parameters
    ----------
    txt : str
        HCL content as a string.
        
    Returns
    -------
    tuple
        Tuple containing the parsed AST as a dict and the analyzer instance.
    """
    analyzer = TerraformAstAnalyzer()
    ast = analyzer.parse_hcl(txt)
    return ast, analyzer


def parse_diff_fragment_with_ast(diff_fragment: str) -> Tuple[Dict, TerraformAstAnalyzer]:
    """
    Parse a Terraform diff fragment using the advanced AST analyzer.
    
    Parameters
    ----------
    diff_fragment : str
        A Terraform diff fragment from git.
        
    Returns
    -------
    tuple
        Tuple containing the parsed AST as a dict and the analyzer instance.
    """
    analyzer = TerraformAstAnalyzer()
    ast = analyzer.parse_diff_fragment(diff_fragment)
    return ast, analyzer


def is_bugfix_commit(message: str) -> bool:
    """
    Check if commit message indicates a bug fix.
    
    Parameters
    ----------
    message : str
        Commit message
        
    Returns
    -------
    bool
        True if message indicates a bugfix, False otherwise
    """
    pattern = r"(fix|bug|issue|error|crash|problem|fail|resolve|patch)"
    return bool(re.search(pattern, message.lower()))


# %% [4] - Enriquecimento de DataFrames com dados Terraform

def enrich_dataframe_with_terraform_semantics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic Terraform semantic information to the DataFrame using regex patterns for resource and assignment lines.
    """
    df = df.copy()
    # Initialize semantic columns
    df['resource_type'] = None
    df['resource_name'] = None
    df['attr_name'] = None
    df['attr_value'] = None

    # Regex patterns
    resource_re = re.compile(r'^\s*resource\s+"([^"]+)"\s+"([^"]+)"')
    assign_re = re.compile(r'^\s*(\w+)\s*=\s*(.+)')

    # Extract semantics using regex to avoid invalid HCL parsing
    for idx, row in df.iterrows():
        if row.get('change') not in ['added', 'removed']:
            continue
        line = row.get('content', '')
        # Resource block
        m = resource_re.match(line)
        if m:
            df.at[idx, 'resource_type'] = m.group(1)
            df.at[idx, 'resource_name'] = m.group(2)
            continue
        # Assignment
        m2 = assign_re.match(line)
        if m2:
            df.at[idx, 'attr_name'] = m2.group(1)
            df.at[idx, 'attr_value'] = m2.group(2).strip()

    return df


def enrich_dataframe_with_terraform_ast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Terraform semantic information to the DataFrame using the advanced AST analyzer.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing diff information.
        
    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with semantic information.
    """
    # Group by hunk to analyze complete fragments
    enriched_rows = []
    
    # First, extract old and new versions of each hunk
    hunks = {}
    for hunk_id, hunk_df in df.groupby('hunk_id'):
        if hunk_id is None:
            continue
            
        # Skip meta lines
        hunk_df = hunk_df[hunk_df['change'] != 'meta']
        
        # Get removed and added lines to create before/after versions
        old_lines = hunk_df[(hunk_df['change'] == 'context') | (hunk_df['change'] == 'removed')]['content'].tolist()
        new_lines = hunk_df[(hunk_df['change'] == 'context') | (hunk_df['change'] == 'added')]['content'].tolist()
        
        hunks[hunk_id] = {
            'file': hunk_df['file'].iloc[0] if not hunk_df['file'].empty else None,
            'old_content': '\n'.join(old_lines),
            'new_content': '\n'.join(new_lines)
        }
    
    # Analyze each hunk with the AST analyzer
    hunk_analysis = {}
    for hunk_id, hunk_data in hunks.items():
        if not hunk_data['file'] or not hunk_data['file'].endswith('.tf'):
            continue
            
        try:
            # Parse old and new content
            old_analyzer = TerraformAstAnalyzer()
            old_analyzer.parse_hcl(hunk_data['old_content'])
            
            new_analyzer = TerraformAstAnalyzer()
            new_analyzer.parse_hcl(hunk_data['new_content'])
            
            # Analyze changes
            changes = new_analyzer.analyze_changes(hunk_data['old_content'], hunk_data['new_content'])
            
            hunk_analysis[hunk_id] = {
                'old_analyzer': old_analyzer,
                'new_analyzer': new_analyzer,
                'changes': changes
            }
        except Exception as e:
            # If analysis fails, skip this hunk
            logger.error(f"Error analyzing hunk {hunk_id}: {str(e)}")
            continue
    
    # Now enrich each row with semantic information
    for _, row in df.iterrows():
        new_row = row.to_dict()
        
        # Skip if not part of a hunk or not a Terraform file
        if row['hunk_id'] is None or (row['file'] and not row['file'].endswith('.tf')):
            enriched_rows.append(new_row)
            continue
        
        # Get analysis for this hunk if available
        analysis = hunk_analysis.get(row['hunk_id'])
        if not analysis:
            enriched_rows.append(new_row)
            continue
        
        # Add semantic information based on change type
        if row['change'] == 'added':
            # For added lines, use new_analyzer
            analyzer = analysis['new_analyzer']
            
            # Try to identify the attribute path for this line
            for attr in analyzer.attributes:
                # Simple check - if the attribute value appears in the content line
                if isinstance(attr.value, str) and attr.value in row['content']:
                    new_row['attribute_path'] = attr.path
                    new_row['resource_type'] = attr.path.split('.')[1] if len(attr.path.split('.')) > 1 else None
                    new_row['resource_name'] = attr.path.split('.')[2] if len(attr.path.split('.')) > 2 else None
                    break
                    
            # Check if this line is part of any added changes
            for added_change in analysis['changes'].get('added', []):
                # Simple check - if the path or value appears in the content line
                if (added_change['path'] in row['content'] or 
                    (isinstance(added_change['value'], str) and added_change['value'] in row['content'])):
                    new_row['change_type'] = 'added'
                    new_row['attribute_path'] = added_change['path']
                    parts = added_change['path'].split('.')
                    if len(parts) > 1 and parts[0] == 'resource':
                        new_row['resource_type'] = parts[1]
                        new_row['resource_name'] = parts[2] if len(parts) > 2 else None
                    break
        
        elif row['change'] == 'removed':
            # For removed lines, use old_analyzer
            analyzer = analysis['old_analyzer']
            
            # Try to identify the attribute path for this line
            for attr in analyzer.attributes:
                # Simple check - if the attribute value appears in the content line
                if isinstance(attr.value, str) and attr.value in row['content']:
                    new_row['attribute_path'] = attr.path
                    new_row['resource_type'] = attr.path.split('.')[1] if len(attr.path.split('.')) > 1 else None
                    new_row['resource_name'] = attr.path.split('.')[2] if len(attr.path.split('.')) > 2 else None
                    break
                    
            # Check if this line is part of any removed changes
            for removed_change in analysis['changes'].get('removed', []):
                # Simple check - if the path or value appears in the content line
                if (removed_change['path'] in row['content'] or 
                    (isinstance(removed_change['value'], str) and removed_change['value'] in row['content'])):
                    new_row['change_type'] = 'removed'
                    new_row['attribute_path'] = removed_change['path']
                    parts = removed_change['path'].split('.')
                    if len(parts) > 1 and parts[0] == 'resource':
                        new_row['resource_type'] = parts[1]
                        new_row['resource_name'] = parts[2] if len(parts) > 2 else None
                    break
        
        # For modified lines, check if they match any modified attributes
        for mod_change in analysis['changes'].get('modified', []):
            old_val = str(mod_change['old_value'])
            new_val = str(mod_change['new_value'])
            
            if (row['change'] == 'removed' and old_val in row['content']) or \
               (row['change'] == 'added' and new_val in row['content']):
                new_row['change_type'] = 'modified'
                new_row['attribute_path'] = mod_change['path']
                new_row['old_value'] = old_val
                new_row['new_value'] = new_val
                parts = mod_change['path'].split('.')
                if len(parts) > 1 and parts[0] == 'resource':
                    new_row['resource_type'] = parts[1]
                    new_row['resource_name'] = parts[2] if len(parts) > 2 else None
                break
        
        enriched_rows.append(new_row)
    
    return pd.DataFrame(enriched_rows)


def analyze_terraform_changes(old_content: str, new_content: str) -> Dict:
    """
    Analyze changes between two Terraform configurations using the AST analyzer.
    
    Parameters
    ----------
    old_content : str
        The old Terraform configuration.
    new_content : str
        The new Terraform configuration.
        
    Returns
    -------
    dict
        Dict with detailed analysis of changes.
    """
    analyzer = TerraformAstAnalyzer()
    changes = analyzer.analyze_changes(old_content, new_content)
    
    # Add dependency information
    old_analyzer = TerraformAstAnalyzer()
    old_analyzer.parse_hcl(old_content)
    
    new_analyzer = TerraformAstAnalyzer()
    new_analyzer.parse_hcl(new_content)
    
    # Extract resource type information for each change
    for change_type in ['added', 'removed', 'modified']:
        for change in changes.get(change_type, []):
            path = change['path']
            parts = path.split('.')
            
            if len(parts) > 1 and parts[0] == 'resource':
                change['resource_type'] = parts[1]
                change['resource_name'] = parts[2] if len(parts) > 2 else None
    
    # Add dependency changes
    old_deps = {node: set(old_analyzer.dependency_graph.successors(node)) 
                for node in old_analyzer.dependency_graph.nodes}
    
    new_deps = {node: set(new_analyzer.dependency_graph.successors(node)) 
                for node in new_analyzer.dependency_graph.nodes}
    
    # Find added and removed dependencies
    all_resources = set(old_deps.keys()) | set(new_deps.keys())
    
    dep_changes = []
    for resource in all_resources:
        old_dependencies = old_deps.get(resource, set())
        new_dependencies = new_deps.get(resource, set())
        
        added_deps = new_dependencies - old_dependencies
        removed_deps = old_dependencies - new_dependencies
        
        if added_deps or removed_deps:
            dep_changes.append({
                'resource': resource,
                'added_dependencies': list(added_deps),
                'removed_dependencies': list(removed_deps)
            })
    
    changes['dependency_changes'] = dep_changes
    
    return changes


# %% [5] - Processamento paralelo de commits

class CommitProcessor:
    """
    Process multiple commits in parallel with checkpoint support.
    
    This class handles loading, processing and analyzing multiple commits
    with support for checkpointing and resumption.
    """
    
    def __init__(self, dataset_dir: Union[str, Path], output_dir: Union[str, Path], 
                 checkpoint_dir: Union[str, Path] = None, max_workers: int = None):
        """
        Initialize the commit processor.
        
        Parameters
        ----------
        dataset_dir : str or Path
            Directory containing the dataset to be analyzed.
        output_dir : str or Path
            Base directory to save results.
        checkpoint_dir : str or Path, optional
            Directory to save checkpoints. If None, uses output_dir/checkpoints.
        max_workers : int, optional
            Maximum number of workers for parallel processing. If None, uses CPU count.
        """
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else self.output_dir / "checkpoints"
        self.max_workers = max_workers or os.cpu_count() or 4
        
        # Create directories if they don't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata = {
            'version': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'processed_commits': 0,
            'processed_repos': set(),
            'processed_files': 0,
            'start_time': datetime.now().isoformat(),
        }
        
        logger.info(f"CommitProcessor initialized. Dataset: {self.dataset_dir}, Output: {self.output_dir}")
    
    def save_checkpoint(self, processed_ids: Set[str], current_index: int) -> Path:
        """
        Save a checkpoint of the processing to allow later resumption.
        
        Parameters
        ----------
        processed_ids : set
            Set of unique IDs of already processed commits.
        current_index : int
            Current index in the processing.
            
        Returns
        -------
        Path
            Path to the saved checkpoint file.
        """
        checkpoint_data = {
            'version': self.metadata.get('version'),
            'timestamp': datetime.now().isoformat(),
            'processed_count': len(processed_ids),
            'processed_ids': list(processed_ids),
            'current_index': current_index,
            'metadata': {k: list(v) if isinstance(v, set) else v for k, v in self.metadata.items()}
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{self.metadata['version']}_{current_index}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
            
        logger.info(f"Checkpoint saved: {checkpoint_path} ({len(processed_ids)} commits processed)")
        return checkpoint_path
    
    def load_latest_checkpoint(self) -> Tuple[Set[str], int]:
        """
        Load the most recent checkpoint to continue processing.
        
        Returns
        -------
        tuple
            Set of processed IDs and current index.
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.json"))
        if not checkpoints:
            logger.info("No checkpoint found. Starting from scratch.")
            return set(), 0
            
        latest = max(checkpoints, key=os.path.getmtime)
        logger.info(f"Loading checkpoint: {latest}")
        
        try:
            with open(latest) as f:
                data = json.load(f)
                self.metadata.update({k: set(v) if k == 'processed_repos' else v 
                                     for k, v in data.get('metadata', {}).items()})
                return set(data.get('processed_ids', [])), data.get('current_index', 0)
        except Exception as e:
            logger.error(f"Error loading checkpoint {latest}: {str(e)}")
            return set(), 0
    
    def process_commits_parallel(self, commit_analyzer_func, resume: bool = True, 
                                chunk_size: int = 100) -> pd.DataFrame:
        """
        Process commits in parallel using ThreadPoolExecutor.
        
        Parameters
        ----------
        commit_analyzer_func : callable
            Function that analyzes a single commit and returns results.
            Should accept a single dict parameter (commit) and return a dict.
        resume : bool, optional
            If True, try to resume processing from the last checkpoint.
        chunk_size : int, optional
            Number of commits to process before saving a checkpoint.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with analysis results.
        """
        # Load commits if not already loaded
        commits = load_all_commit_jsonl(str(self.dataset_dir))
            
        if not commits:
            logger.warning("No commits to process!")
            return pd.DataFrame()
            
        processed_ids = set()
        start_index = 0
        
        # Try to load the last checkpoint
        if resume:
            processed_ids, start_index = self.load_latest_checkpoint()
            logger.info(f"Resuming processing from index {start_index}, with {len(processed_ids)} commits already processed")
        
        # Initialize list to store results
        results = []
        
        # Remaining commits to process
        remaining_commits = [c for i, c in enumerate(commits) 
                            if i >= start_index and c['unique_id'] not in processed_ids]
        
        logger.info(f"Processing {len(remaining_commits)} remaining commits")
        
        # Configure executor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks to the executor
            future_to_commit = {
                executor.submit(commit_analyzer_func, commit): commit 
                for commit in remaining_commits
            }
            
            # Process results as they complete
            for i, future in enumerate(tqdm(
                concurrent.futures.as_completed(future_to_commit),
                total=len(future_to_commit),
                desc="Analyzing commits"
            )):
                commit = future_to_commit[future]
                try:
                    result = future.result()
                    results.append(result)
                    processed_ids.add(commit['unique_id'])
                    
                    # Update metadata
                    self.metadata['processed_commits'] = len(processed_ids)
                    self.metadata['processed_repos'].add(commit.get('repo', 'unknown'))
                    self.metadata['processed_files'] += 1
                    
                    # Save checkpoint every chunk_size commits
                    if (i + 1) % chunk_size == 0:
                        current_index = start_index + i + 1
                        self.save_checkpoint(processed_ids, current_index)
                        
                except Exception as e:
                    logger.error(f"Error processing commit {commit.get('repo', 'unknown')}/{commit.get('commit_hash', '')[:7]}: {str(e)}")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save consolidated results
        output_path = self.output_dir / f"commit_analysis_{self.metadata['version']}.csv"
        results_df.to_csv(output_path, index=False)
        
        # Save final checkpoint
        self.save_checkpoint(processed_ids, len(commits))
        
        # Update final metadata
        self.metadata['end_time'] = datetime.now().isoformat()
        self.metadata['total_processed'] = len(processed_ids)
        
        # Save metadata
        with open(self.output_dir / f"metadata_{self.metadata['version']}.json", 'w') as f:
            json.dump({k: list(v) if isinstance(v, set) else v for k, v in self.metadata.items()}, 
                     f, indent=2)
            
        logger.info(f"Processing complete. Results saved to {output_path}")
        logger.info(f"Total commits processed: {len(processed_ids)}")
        
        return results_df


# %% [6] - Classes para detecção de operadores de mutação

class TerraformResource:
    """
    Class representing a Terraform resource identified in a patch.
    
    Parameters
    ----------
    resource_type : str
        Type of resource (e.g. aws_instance, aws_s3_bucket)
    resource_name : str
        Resource name defined in the code
    properties : dict
        Dictionary of resource properties and values
    """
    def __init__(self, resource_type: str, resource_name: str, properties: Dict = None):
        self.type = resource_type
        self.name = resource_name
        self.properties = properties or {}
        
    def __repr__(self):
        return f"TerraformResource(type='{self.type}', name='{self.name}', props_count={len(self.properties)})"


class MutationOperator:
    """
    Class representing a mutation operator for Terraform.
    
    Parameters
    ----------
    name : str
        Name of the operator
    description : str
        Description of the operator
    category : str
        Category of the operator
    target_resources : list
        List of target resource types
    difficulty : int
        Implementation difficulty level (1-5)
    """
    def __init__(self, name: str, description: str, category: str, 
                 target_resources: List[str] = None, difficulty: int = 3):
        self.name = name
        self.description = description
        self.category = category
        self.target_resources = target_resources or []
        self.difficulty = difficulty
        self.occurrences = 0
        self.examples = []
        
    def __repr__(self):
        return f"MutationOperator(name='{self.name}', category='{self.category}', occurrences={self.occurrences})"
        
    def add_example(self, repo: str, commit_hash: str, file: str, original: str, mutated: str):
        """
        Add an example of operator application.
        
        Parameters
        ----------
        repo : str
            Source repository
        commit_hash : str
            Commit hash
        file : str
            Modified file
        original : str
            Original code
        mutated : str
            Mutated code
        """
        self.examples.append({
            'repo': repo,
            'commit_hash': commit_hash,
            'file': file,
            'original': original,
            'mutated': mutated
        })
        self.occurrences += 1


class MutationOperatorDetector:
    """
    Detector for potential mutation operators in Terraform diffs.
    """
    
    # Definition of common mutation operators
    DEFAULT_OPERATORS = [
        #TODO: add operators	
    ]
    
    def __init__(self):
        # Dictionary of operators by name
        self.operators = {op.name: op for op in self.DEFAULT_OPERATORS}
        
    def detect_potential_operators(self, diff_df: pd.DataFrame) -> Dict[str, int]:
        """
        Detect potential mutation operators in a diff.
        
        Parameters
        ----------
        diff_df : pd.DataFrame
            DataFrame containing the diff
            
        Returns
        -------
        dict
            Dictionary with count of potential operators
        """
        if diff_df.empty:
            return {}
            
        # Counters for each operator
        operator_counts = {op_name: 0 for op_name in self.operators}
        
        # Analyze added and removed lines to detect patterns
        added = diff_df[diff_df['change'] == 'added']['content'].tolist()
        removed = diff_df[diff_df['change'] == 'removed']['content'].tolist()
        
        # # Check for BooleanNegation
        # for r_line in removed:
        #     r_line = r_line.strip()
        #     if ' = true' in r_line:
        #         for a_line in added:
        #             a_line = a_line.strip()
        #             if r_line.replace('true', 'false') == a_line:
        #                 operator_counts['BooleanNegation'] += 1
        #     elif ' = false' in r_line:
        #         for a_line in added:
        #             a_line = a_line.strip()
        #             if r_line.replace('false', 'true') == a_line:
        #                 operator_counts['BooleanNegation'] += 1
        
        # # Check for NumberModification
        # import re
        # num_pattern = re.compile(r'(\w+)\s*=\s*(\d+)')
        # for r_line in removed:
        #     r_match = num_pattern.search(r_line)
        #     if r_match:
        #         attr_name, old_val = r_match.groups()
        #         for a_line in added:
        #             a_match = num_pattern.search(a_line)
        #             if a_match and a_match.group(1) == attr_name and a_match.group(2) != old_val:
        #                 operator_counts['NumberModification'] += 1
        
        # # Check for StringReplacement
        # str_pattern = re.compile(r'(\w+)\s*=\s*"([^"]*)"')
        # for r_line in removed:
        #     r_match = str_pattern.search(r_line)
        #     if r_match:
        #         attr_name, old_val = r_match.groups()
        #         for a_line in added:
        #             a_match = str_pattern.search(a_line)
        #             if a_match and a_match.group(1) == attr_name and a_match.group(2) != old_val:
        #                 operator_counts['StringReplacement'] += 1
        
        # Other operators would follow a similar pattern...
        
        # Remove operators with no occurrences
        return {k: v for k, v in operator_counts.items() if v > 0}
    


    def summarize_patch_counts(patch: str) -> dict[str, int]:
        """
        Retorna contagem de categorias de mudança delegando a diff_stats.summarize_patch.
        Import local evita dependência circular (diff_stats já importa diff_utils).
        """
        from diff_stats import summarize_patch   
        return summarize_patch(patch=patch)

def get_other_lines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract lines categorized as 'other' for further analysis.
    """
    return df[df.get('category') == 'other'].copy()
