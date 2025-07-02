# Terraform Repository Mining Module

## Overview
This module automates the mining of GitHub repositories containing Terraform code, collecting repository metadata, commit history, and bug fix patterns. The module implements a comprehensive pipeline for discovering, cloning, and analyzing Terraform repositories to build datasets for Infrastructure as Code (IaC) research.

## Features

- **GitHub API Integration**: Search and filter Terraform repositories
- **Intelligent Repository Selection**: Filter by stars, commits, and quality criteria
- **Commit Analysis**: Identify bug-fixing commits using keyword patterns
- **Data Export**: Export structured data in JSONL format
- **Comprehensive Logging**: Detailed logging and progress tracking
- **Statistical Reporting**: Generate summary statistics and reports

## Directory Structure
```
src/miner/
├── main.py              # Main entry point for mining pipeline
├── github_api.py        # GitHub API interface and repository search
├── repo_cloner.py       # Repository cloning and management
├── commit_miner.py      # Commit analysis and filtering
├── data_exporter.py     # Data export and formatting
├── utils.py             # Utility functions
├── config.yaml          # Mining configuration
├── README.md            # This file
├── clone_summary.json   # Generated: Mining summary (JSON)
└── clone_summary.txt    # Generated: Mining summary (text)
```

## Output Structure
The mining process creates the following directory structure (relative to project root):
```
../data/                 # Created one level above project root
├── repos/              # Cloned repositories
│   ├── repo1_hash/
│   ├── repo2_hash/
│   └── ...
├── dataset/            # Exported data
│   ├── data.jsonl      # Main dataset file
│   └── ...
└── logs/               # Logs and summaries
    ├── terraform_miner.log
    └── summary.json
```

## Installation and Setup

### Prerequisites
- Python 3.12+
- Git
- UV package manager
- GitHub token (recommended for higher rate limits)

### Environment Setup

1. **Activate the virtual environment** (from the project root):
   ```powershell
   .\.venv\Scripts\activate.ps1  # Windows
   # source .venv/bin/activate   # Linux/Mac
   ```
   > You should see `(.venv)` at the start of your terminal prompt.

2. **Install dependencies** (if not already installed):
   ```powershell
   uv pip install -r requirements.txt
   ```

3. **Navigate to the mining module directory**:
   ```powershell
   cd src/miner
   ```

## Configuration

Edit `config.yaml` to customize mining parameters:

```yaml
github:
  # GitHub API settings
  min_stars: 10
  min_commits: 50
  max_repos: 100
  
repositories:
  # Repository filtering
  keywords: ["terraform", "infrastructure"]
  exclude_forks: true
  exclude_archived: true
  
commits:
  # Commit filtering keywords
  bug_keywords: ["fix", "bug", "error", "issue", "patch"]
  exclude_merge: true
  
output:
  # Output configuration
  format: "jsonl"
  include_patches: true
```

## Usage

### Basic Mining

Run the mining pipeline with default settings:

```powershell
python main.py --config config.yaml --limit 50
```

### Advanced Options

```powershell
# Full parameter specification
python main.py \
  --config config.yaml \
  --limit 100 \
  --token YOUR_GITHUB_TOKEN \
  --stars 50 \
  --min-commits 10 \
  --keywords fix bug error issue \
  --output-dir ../data/custom \
  --verbose

# Statistics only (no cloning)
python main.py --config config.yaml --stats-only --limit 20

# Resume interrupted mining
python main.py --config config.yaml --resume --limit 100
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to configuration file | `config.yaml` |
| `--limit` | Maximum repositories to process | `50` |
| `--token` | GitHub API token | From environment |
| `--stars` | Minimum repository stars | From config |
| `--min-commits` | Minimum repository commits | From config |
| `--keywords` | Bug fix keywords | From config |
| `--output-dir` | Output directory | `../data` |
| `--stats-only` | Only collect statistics | `False` |
| `--resume` | Resume interrupted mining | `False` |
| `--verbose` | Enable verbose logging | `False` |

## Data Collection

### Repository Metadata
For each repository, the following information is collected:
- Repository name and owner
- Number of stars (`stargazers_count`)
- Number of forks (`forks_count`)
- Creation date and last update
- Primary language and size
- Total number of commits
- License information

### Commit Analysis
For each relevant commit, the module extracts:
- Commit hash and metadata
- Author information
- Commit message and timestamp
- Modified files and patches
- Diff statistics

### Bug Fix Detection
Commits are classified as bug fixes based on:
- Message keywords (`fix`, `bug`, `error`, `issue`, `patch`)
- File modification patterns
- Commit message structure analysis
- Pull request integration patterns

## Output Formats

### JSONL Dataset (`data.jsonl`)
Each line represents a file modification within a commit:
```json
{
  "repo": "owner_repository_hash",
  "commit_hash": "abc123...",
  "author": "Developer Name",
  "date": "2023-08-23T13:01:40-07:00",
  "message": "fix: resolve authentication issue",
  "file": "main.tf",
  "patch": "@@ -10,6 +10,7 @@ resource \"aws_instance\"...",
  "head_hash": "def456..."
}
```

### Summary Reports
- `clone_summary.json`: Structured summary with statistics
- `clone_summary.txt`: Human-readable summary report
- Mining logs with detailed progress information

## Testing and Debugging

### Individual Component Testing

1. **Test Repository Cloning**:
   ```powershell
   cd ..  # Navigate to src/
   python test_cloner.py owner/repository
   ```

2. **Test Search and Cloning Pipeline**:
   ```powershell
   cd ..  # Navigate to src/
   python test_mining.py [github_token]
   ```

3. **Run Simplified End-to-End Test**:
   ```powershell
   cd ..  # Navigate to src/
   python run_simple_test.py
   ```

### Debugging Workflow

1. **Check Logs**:
   - Main log: `../data/logs/terraform_miner.log`
   - Test logs: `test_cloner.log`, `test_mining_*.log`

2. **Verify Directory Structure**:
   ```
   ../data/
   ├── repos/          # Check for cloned repositories
   ├── logs/           # Check log files and summaries
   ├── dataset/        # Check exported data files
   └── test_repos/     # Test repositories (if testing)
   ```

3. **Common Issues**:
   - **Rate Limiting**: Use GitHub token for higher limits
   - **Network Issues**: Check internet connection and GitHub access
   - **Permission Errors**: Verify write permissions for output directories
   - **Disk Space**: Ensure sufficient space for repository cloning

### Test Output Verification

After running tests, verify:
- Repositories are successfully cloned to `../data/repos/`
- Log files contain detailed progress information
- Summary files are generated with correct statistics
- JSONL export contains structured commit data

## Performance Considerations

### Rate Limiting
- GitHub API: 5,000 requests/hour with token, 60 without
- Cloning: Depends on repository size and network speed
- Recommended: Use personal access token for production runs

### Resource Usage
- **Disk Space**: ~100MB-1GB per repository (varies by size)
- **Memory**: ~500MB-2GB depending on repository count
- **Network**: Significant bandwidth for repository cloning

### Optimization Tips
- Use `--stats-only` for initial surveys
- Set appropriate `--limit` values for testing
- Monitor disk space before large mining operations
- Use `--resume` to continue interrupted operations

## Integration with NLP Pipeline

The mining module's output serves as input for the NLP analysis pipeline:

1. **Mining Output**: `../data/dataset/data.jsonl`
2. **NLP Input**: Processed by `src/nlp_analysis/01_preprocess_and_embed.py`
3. **Analysis Chain**: Mining → NLP Clustering → Semantic Analysis

Ensure the mining completes successfully before running NLP analysis.

## Contributing

When contributing to the mining module:
1. Follow the existing code structure and documentation patterns
2. Add appropriate logging for new features
3. Update configuration options in `config.yaml`
4. Include tests for new functionality
5. Update this README with new features or changes

## Troubleshooting

### Common Error Messages

**"GitHub API rate limit exceeded"**
- Solution: Provide GitHub token with `--token` parameter
- Check: Current rate limit status at https://api.github.com/rate_limit

**"Permission denied when creating directory"**
- Solution: Verify write permissions for output directory
- Check: Current working directory and path permissions

**"Repository clone failed"**
- Solution: Check internet connection and repository accessibility
- Check: Repository URL and authentication requirements

**"Invalid configuration file"**
- Solution: Verify YAML syntax in `config.yaml`
- Check: Required configuration parameters are present

For additional issues, check the detailed logs in `../data/logs/terraform_miner.log`. 