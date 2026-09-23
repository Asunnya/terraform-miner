# Terraform Miner

A tool for mining and analyzing Terraform code from GitHub repositories to identify bug fixes and common issues. This project combines repository mining, commit analysis, and Natural Language Processing (NLP) to build a comprehensive dataset of Infrastructure as Code (IaC) defects and their fixes.

## Overview

Terraform Miner helps researchers and developers understand bug patterns in Infrastructure as Code (IaC) by:

1. **Repository Mining**: Search for and clone Terraform repositories from GitHub
2. **Commit Analysis**: Identify bug-fixing commits based on commit messages and keywords
3. **NLP Analysis**: Apply semantic clustering and text analysis to understand defect patterns
4. **Dataset Generation**: Build structured datasets of common bugs and their fixes

## Key Features

- **GitHub Repository Mining**: Automated search and cloning of Terraform repositories
- **Intelligent Commit Filtering**: Identify bug-fixing commits using multiple criteria
- **Semantic Clustering**: Group similar commits using advanced NLP techniques (SBERT, UMAP, HDBSCAN)
- **Comprehensive Reporting**: Generate detailed analysis reports and visualizations

## Installation

### Prerequisites

- Python 3.12+
- Git
- UV package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/terraform-miner.git
cd terraform-miner

# Install UV (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# or on Windows:
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Create and activate virtual environment
uv venv
# On Windows PowerShell:
.\.venv\Scripts\activate.ps1
# On Linux/Mac:
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

## Project Structure

```
terraform-miner/
├── src/
│   ├── miner/              # Repository mining and commit analysis
│   │   ├── main.py         # Main mining script
│   │   ├── github_api.py   # GitHub API interface
│   │   ├── repo_cloner.py  # Repository cloning
│   │   ├── commit_miner.py # Commit analysis
│   │   └── data_exporter.py # Data export utilities
│   └── nlp_analysis/       # NLP pipeline for semantic analysis
│       ├── 01_preprocess_and_embed.py  # Stage 1: Clustering
│       ├── 02_analyze_and_report_clusters.py # Stage 2: Analysis
│       ├── config.yaml     # NLP configuration
│       └── analysis_results_nlp/ # Generated results
├── reports/                # Generated reports and visualizations
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Usage

### Phase 1: Mining Terraform Repositories

Run the miner as a module **from the project root** (it uses absolute `src.` imports, so `python main.py` from inside `src/miner` fails):

```bash
# Activate virtual environment (if not already active)
.\.venv\Scripts\activate.ps1  # Windows
# source .venv/bin/activate   # Linux/Mac

# Run mining with configuration
python -m src.miner.main --config src/miner/config.yaml --limit 50

# Optional parameters:
python -m src.miner.main --config src/miner/config.yaml \
  --limit 100 \
  --token YOUR_GITHUB_TOKEN \
  --stars 50 \
  --min-commits 10 \
  --workers 4
# Also available: --keywords, --stats-only, --full-clone, --clone-depth N
```

`--config` is resolved relative to the project root first, then relative to `src/miner/`.

The mining process will:
1. Search for Terraform repositories on GitHub
2. Clone repositories to the `../data/repos/` directory
3. Mine commits containing fix-related keywords
4. Export relevant data to JSONL format
5. Generate summary files and logs

### Phase 2: NLP Analysis and Semantic Clustering

After mining data, run the NLP analysis pipeline. Run it **from `src/nlp_analysis`**: stage 1 writes `analysis_results_nlp/` relative to the current directory, and stage 2 reads it from `src/nlp_analysis/analysis_results_nlp/`.

```bash
# Navigate to NLP analysis directory
cd src/nlp_analysis

# Stage 1: Preprocessing, embedding, and clustering
python 01_preprocess_and_embed.py

# Stage 2: Cluster analysis and reporting
python 02_analyze_and_report_clusters.py
```

#### NLP Pipeline Configuration

The NLP analysis supports two clustering approaches configured in `src/nlp_analysis/config.yaml`:

**Top-Down Approach** (default):
```yaml
clustering_settings:
  use_all_commits_for_clustering: true
  remove_duplicates_for_clustering: false
```
- Clusters all commits (~7,240) without pre-filtering
- Creates a complete "map" of activities
- Isolates merge commits and noise in specific clusters

**Bottom-Up Approach**:
```yaml
clustering_settings:
  use_all_commits_for_clustering: false
  remove_duplicates_for_clustering: true
```
- Applies eligibility filters before clustering
- Results in clusters focused on bug fixes
- Produces cleaner, more targeted results (~697 commits)

## Advanced Features

### NLP Analysis Pipeline

The NLP pipeline uses state-of-the-art techniques:

- **Sentence-BERT**: `all-roberta-large-v1` for high-quality embeddings
- **UMAP**: Dimensionality reduction for visualization and clustering
- **HDBSCAN**: Density-based clustering for identifying semantic groups
- **spaCy**: Advanced text preprocessing and lemmatization

Key features:
- Semantic message extraction from pull request information
- Automatic noise filtering and deduplication
- Configurable clustering parameters
- Detailed reporting with keyword analysis

## Configuration

### Mining Configuration (`src/miner/config.yaml`)

Configure repository search and mining parameters:
- Minimum repository stars
- Keywords to identify bug fix commits
- Maximum repositories to process
- File patterns and exclusions

### NLP Configuration (`src/nlp_analysis/config.yaml`)

Configure the semantic analysis pipeline:
- SBERT model selection
- UMAP and HDBSCAN parameters
- Text processing options
- Clustering approaches and filtering

## Output and Results

### Mining Results
- `../data/repos/`: Cloned repositories
- `../data/logs/`: Mining logs and summaries
- `../data/dataset/`: Exported commit data in JSONL format

### NLP Analysis Results
- `src/nlp_analysis/analysis_results_nlp/commits_com_clusters.jsonl`: Clustered commits
- `src/nlp_analysis/analysis_results_nlp/corpus_overview.md`: Corpus statistics
- Cluster analysis reports in TXT format

### Analysis Reports
- `reports/`: Comprehensive analysis reports and visualizations
- Statistical summaries and insights

## Testing

The test suite covers the miner (`src/miner/`) and `src/analyze_null_patches.py`. It runs offline (GitHub is mocked, git repos are created in temporary directories):

```bash
uv pip install -r requirements.txt
python -m pytest
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this tool in your research, please cite: WIP

```bibtex
@misc{terraform-miner,

}
```

## Research Context

This tool was developed as part of research into mutation testing for Infrastructure as Code (IaC), specifically focusing on identifying common defect patterns in Terraform configurations to inform the development of effective mutation operators.
