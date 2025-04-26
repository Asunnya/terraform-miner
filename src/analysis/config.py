import os
import yaml
import pandas as pd
import json
from typing import Dict, List, Optional, Union
from pathlib import Path

# Get the project root directory (2 levels up from this file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_config():
    """Load configuration from YAML file or environment variables."""
    # Try to load from config file first
    config = {}
    config_path = os.environ.get('TF_CONFIG_PATH', os.path.join(PROJECT_ROOT, 'config.yaml'))
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError):
        pass  # No config file or invalid YAML
    
    # Override/set from environment variables with absolute paths
    config['dataset_dir'] = os.path.abspath(os.environ.get('TF_DATASET_DIR', 
                                          os.path.join(PROJECT_ROOT, 'data', 'dataset')))
    config['output_dir'] = os.path.abspath(os.environ.get('TF_OUTPUT_DIR', 
                                         os.path.join(PROJECT_ROOT, 'data', 'reports')))
    config['checkpoints_dir'] = os.path.abspath(os.environ.get('TF_CHECKPOINTS_DIR',
                                              os.path.join(PROJECT_ROOT, 'data', 'checkpoints')))
    config['repos_dir'] = os.path.abspath(os.environ.get('TF_REPOS_DIR',
                                        os.path.join(PROJECT_ROOT, 'data', 'repos')))
    
    return config

def load_commit_reports(reports_dir: str = None) -> List[Dict]:
    """
    Load commit reports from markdown files in the reports directory.
    
    Args:
        reports_dir: Directory containing commit reports. If None, uses default from config.
        
    Returns:
        List of dictionaries containing commit information
    """
    if reports_dir is None:
        config = get_config()
        reports_dir = os.path.join(config['output_dir'], 'commits')
    
    reports_path = Path(reports_dir)
    commits = []
    
    if not reports_path.exists():
        return commits
        
    for file in reports_path.glob('commit_*.md'):
        commit_data = {}
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            
            # Parse metadata from markdown
            for line in lines:
                if line.startswith('**Hash:**'):
                    commit_data['hash'] = line.split('**Hash:**')[1].strip()
                elif line.startswith('**Autor:**'):
                    commit_data['author'] = line.split('**Autor:**')[1].strip()
                elif line.startswith('**Data:**'):
                    commit_data['date'] = line.split('**Data:**')[1].strip()
                elif line.startswith('**Mensagem:**'):
                    commit_data['message'] = line.split('**Mensagem:**')[1].strip()
                elif line.startswith('**Arquivo:**'):
                    commit_data['file'] = line.split('**Arquivo:**')[1].strip()
            
            # Extract repository name from filename
            repo = file.stem.split('_')[1]  # commit_REPO_HASH.md
            commit_data['repo'] = repo
            
            commits.append(commit_data)
            
    return commits

def load_diff_data(reports_dir: str = None) -> Optional[pd.DataFrame]:
    """
    Load consolidated diff data from the all_diffs.csv file.
    
    Args:
        reports_dir: Directory containing the consolidated diff data. If None, uses default from config.
        
    Returns:
        DataFrame containing all diffs or None if file doesn't exist
    """
    if reports_dir is None:
        config = get_config()
        reports_dir = config['output_dir']
        
    diff_path = Path(reports_dir) / 'all_diffs.csv'
    
    if not diff_path.exists():
        return None
        
    return pd.read_csv(diff_path)

def get_commit_diffs(repo: str, commit_hash: str, reports_dir: str = None) -> Optional[pd.DataFrame]:
    """
    Get specific diffs for a commit from its CSV file.
    
    Args:
        repo: Repository name
        commit_hash: Commit hash
        reports_dir: Directory containing diff reports. If None, uses default from config.
        
    Returns:
        DataFrame containing diffs for the specific commit or None if not found
    """
    if reports_dir is None:
        config = get_config()
        reports_dir = config['output_dir']
        
    diff_path = Path(reports_dir) / f'diff_{repo}_{commit_hash}.csv'
    
    if not diff_path.exists():
        return None
        
    return pd.read_csv(diff_path)

def get_commit_patch_html(repo: str, commit_hash: str, reports_dir: str = None) -> Optional[str]:
    """
    Get HTML visualization of a specific commit patch.
    
    Args:
        repo: Repository name
        commit_hash: Commit hash
        reports_dir: Directory containing patch HTML files. If None, uses default from config.
        
    Returns:
        HTML content as string or None if not found
    """
    if reports_dir is None:
        config = get_config()
        reports_dir = config['output_dir']
        
    html_path = Path(reports_dir) / f'patch_{repo}_{commit_hash}.html'
    
    if not html_path.exists():
        return None
        
    with open(html_path, 'r', encoding='utf-8') as f:
        return f.read()

def get_analysis_summary(reports_dir: str = None) -> Dict[str, Union[int, List[str]]]:
    """
    Generate a summary of the analysis data available in the reports directory.
    
    Args:
        reports_dir: Directory containing reports. If None, uses default from config.
        
    Returns:
        Dictionary containing summary statistics
    """
    if reports_dir is None:
        config = get_config()
        reports_dir = config['output_dir']
        
    reports_path = Path(reports_dir)
    commits_path = reports_path / 'commits'
    
    summary = {
        'total_commits': 0,
        'total_repositories': set(),
        'total_diffs': 0,
        'available_reports': []
    }
    
    # Count commits and repositories
    if commits_path.exists():
        commit_files = list(commits_path.glob('commit_*.md'))
        summary['total_commits'] = len(commit_files)
        summary['total_repositories'] = len({f.stem.split('_')[1] for f in commit_files})
    
    # Check for consolidated reports
    if (reports_path / 'all_diffs.csv').exists():
        summary['available_reports'].append('all_diffs.csv')
        df = pd.read_csv(reports_path / 'all_diffs.csv')
        summary['total_diffs'] = len(df)
    
    if (reports_path / 'all_diffs.html').exists():
        summary['available_reports'].append('all_diffs.html')
    
    # Look for visualization files
    for img_file in reports_path.glob('*.png'):
        summary['available_reports'].append(img_file.name)
    
    return summary