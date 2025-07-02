# NLP Analysis Pipeline for Terraform Defect Classification

## Overview

This module implements a comprehensive Natural Language Processing (NLP) pipeline for analyzing and classifying Terraform bug fixes collected during the mining phase. The pipeline transforms raw commit data into semantically organized clusters, enabling researchers to understand common defect patterns in Infrastructure as Code (IaC) projects.

## Features

- **Semantic Commit Analysis**: Extract meaningful information from commit messages and pull request data
- **Advanced Text Preprocessing**: Clean and normalize commit messages for analysis
- **High-Quality Embeddings**: Generate sentence embeddings using Sentence-BERT (RoBERTa)
- **Dimensionality Reduction**: Apply UMAP for efficient clustering and visualization
- **Density-Based Clustering**: Use HDBSCAN to identify semantic clusters of related commits
- **Thematic Analysis**: Generate detailed reports with keyword extraction and examples
- **Flexible Configuration**: Support multiple clustering approaches and filtering strategies

## Pipeline Architecture

The NLP analysis pipeline consists of two main stages:

### Stage 1: Preprocessing, Embedding, and Clustering
**Script**: `01_preprocess_and_embed.py`
- Data aggregation and refinement
- Semantic message extraction from pull requests
- Sentence embedding generation using SBERT
- Dimensionality reduction with UMAP
- Density-based clustering with HDBSCAN

### Stage 2: Cluster Analysis and Reporting
**Script**: `02_analyze_and_report_clusters.py`
- Keyword extraction and frequency analysis
- Thematic interpretation of clusters
- Generation of detailed analysis reports
- Sample message extraction for manual review

## Directory Structure

```
src/nlp_analysis/
├── 01_preprocess_and_embed.py      # Stage 1: Clustering pipeline
├── 02_analyze_and_report_clusters.py # Stage 2: Analysis and reporting
├── config.yaml                     # Main pipeline configuration
├── config_02_analyze.yaml          # Analysis-specific configuration
├── analysis_results_nlp/           # Generated results directory
│   ├── commits_com_clusters.jsonl  # Clustered commits dataset
│   ├── sbert_embeddings.pkl        # Cached embeddings
│   ├── corpus_overview.md          # Corpus statistics
│   └── keywords_stop_nlp_suggestions.txt # Stopword suggestions
├── README.md                       # This file
└── cluster_analysis_reports/       # Detailed cluster reports (TXT files)
```

## Installation and Setup

### Prerequisites
- Python 3.12+
- UV package manager
- Completed mining phase (data available in `../data/dataset/data.jsonl`)

### Environment Setup

1. **Activate virtual environment** (from project root):
   ```powershell
   .\.venv\Scripts\activate.ps1  # Windows
   # source .venv/bin/activate   # Linux/Mac
   ```

2. **Install NLP dependencies** (if not already installed):
   ```powershell
   uv pip install -r requirements.txt
   ```

3. **Navigate to NLP analysis directory**:
   ```powershell
   cd src/nlp_analysis
   ```

## Configuration

### Main Configuration (`config.yaml`)

The main configuration file controls all aspects of the NLP pipeline:

#### Key Configuration Sections

**SBERT Model Settings**:
```yaml
SBERT_MODEL_NAME: 'sentence-transformers/all-roberta-large-v1'
```

**UMAP Parameters**:
```yaml
UMAP_PARAMS:
  n_neighbors: 20
  min_dist: 0.1
  n_components: 5
  metric: 'cosine'
```

**HDBSCAN Parameters**:
```yaml
HDBSCAN_PARAMS:
  min_cluster_size: 30
  min_samples: 10
  allow_single_cluster: true
  cluster_selection_epsilon: 0.5
```

**Text Processing**:
```yaml
text_processing_params:
  custom_stopwords_additions: ['git', 'github', 'gitlab', 'pull', 'request']
  excluded_dirs_patterns: ['vendor/', 'test/', 'examples/', 'docs/']
  relevant_extensions_order: ['.tf', '.go', '.py', '.yaml', '.yml']
  extract_semantic_from_merge: true
```

#### Clustering Approaches

The pipeline supports two distinct approaches:

**Top-Down Approach** (Comprehensive):
```yaml
clustering_settings:
  use_all_commits_for_clustering: true
  remove_duplicates_for_clustering: false
```
- Clusters all available commits (~7,240)
- Creates complete activity map
- Isolates merge commits and noise in specific clusters
- May reveal non-obvious bug fix groupings

**Bottom-Up Approach** (Focused):
```yaml
clustering_settings:
  use_all_commits_for_clustering: false
  remove_duplicates_for_clustering: true
```
- Applies filtering before clustering
- Results in bug fix-focused clusters
- Cleaner, more targeted results (~697 commits)
- Better for balanced repository representation

### Analysis Configuration (`config_02_analyze.yaml`)

Controls the detailed cluster analysis and reporting:

```yaml
terms_analysis:
  num_top_terms: 25                    # Top terms per cluster
  num_terms_with_context: 15           # Terms with detailed context
  examples_per_term: 10                # Examples per term
  max_example_length: 400              # Maximum example length

message_samples:
  num_sample_messages: 25              # Sample messages per cluster
  max_sample_length: 250               # Maximum sample length

report_settings:
  include_detailed_contexts: true      # Include detailed term contexts
  show_cluster_percentages: true       # Show cluster size percentages
  highlight_terms_in_examples: true    # Highlight terms in examples
```

## Usage

### Running the Complete Pipeline

Execute both stages in sequence:

```powershell
# Stage 1: Preprocessing and clustering
python 01_preprocess_and_embed.py

# Stage 2: Analysis and reporting
python 02_analyze_and_report_clusters.py
```

### Stage 1: Preprocessing and Clustering

```powershell
python 01_preprocess_and_embed.py
```

**What it does**:
1. **Data Aggregation**: Groups file modifications by commit hash
2. **Semantic Extraction**: Processes merge messages to extract meaningful content
3. **Filtering**: Removes irrelevant commits (documentation-only, vendor files)
4. **Embedding**: Generates sentence embeddings using Sentence-BERT
5. **Dimensionality Reduction**: Applies UMAP for visualization and clustering
6. **Clustering**: Uses HDBSCAN to identify semantic groups
7. **Export**: Saves clustered data to JSONL format

**Outputs**:
- `analysis_results_nlp/commits_com_clusters.jsonl`: Main clustered dataset
- `analysis_results_nlp/sbert_embeddings.pkl`: Cached embeddings
- `analysis_results_nlp/corpus_overview.md`: Corpus statistics

### Stage 2: Cluster Analysis and Reporting

```powershell
python 02_analyze_and_report_clusters.py
```

**What it does**:
1. **Term Extraction**: Identifies most frequent terms per cluster
2. **Keyword Analysis**: Analyzes term distributions and significance
3. **Context Generation**: Provides examples showing terms in context
4. **Report Generation**: Creates detailed textual reports for each cluster
5. **Thematic Interpretation**: Summarizes cluster themes and patterns

**Outputs**:
- Detailed analysis reports in `cluster_analysis_reports/`
- One TXT file per cluster with comprehensive analysis
- Keyword frequency analysis and context examples

## Data Flow and Processing

### Input Data Format
The pipeline expects JSONL data from the mining phase:
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

### Processing Steps

1. **Aggregation**: Group file modifications by `(repo, commit_hash)`
2. **Semantic Message Extraction**:
   ```python
   # Extract meaningful content from merge messages
   "Merge pull request #123 from feature/auth-fix" 
   → "feature auth fix [+ PR description if available]"
   ```
3. **Relevance Filtering**: Remove commits that only modify:
   - Documentation files
   - Vendor directories
   - Test fixtures
   - Non-code files

4. **Text Preprocessing**:
   - Tokenization and lemmatization
   - Stopword removal (standard + domain-specific)
   - Message length normalization

5. **Embedding Generation**:
   - Use Sentence-BERT to create 1024-dimensional vectors
   - Cache embeddings for reproducibility

6. **Clustering**:
   - UMAP reduces dimensionality to 5 components
   - HDBSCAN identifies dense regions as clusters
   - Assign cluster labels and membership probabilities

### Output Data Format
The clustered dataset includes additional fields:
```json
{
  "repo_name": "owner/repository",
  "commit_hash": "abc123...",
  "author_name": "Developer Name",
  "author_date": "2023-08-23T13:01:40-07:00",
  "msg_original": "fix: resolve authentication issue",
  "msg_clean_for_val": "fix resolve authentication issue",
  "hdbscan_cluster_label": 5,
  "hdbscan_cluster_probability": 0.85
}
```

## Analysis and Interpretation

### Cluster Analysis Reports

Each cluster generates a comprehensive report including:

1. **Cluster Overview**: Size, percentage of total, noise classification
2. **Top Terms**: Most frequent terms with counts and frequencies
3. **Detailed Context**: Examples showing terms in actual commit messages
4. **Sample Messages**: Representative commit messages from the cluster
5. **Manual Analysis Placeholders**: Sections for researcher annotations

### Example Report Structure
```
=== CLUSTER 5 ANALYSIS ===
Size: 156 commits (15.2% of total)
Classification: Semantic cluster (not noise)

TOP TERMS (showing top 15 of 25):
1. authentication (89 occurrences, 57.1%)
2. oauth (76 occurrences, 48.7%)
3. token (71 occurrences, 45.5%)
...

DETAILED TERM CONTEXTS:
Term: authentication (89 total occurrences)
Examples:
- "fix authentication flow for oauth providers"
- "resolve authentication timeout issues"
...

SAMPLE MESSAGES (25 examples):
1. "fix: resolve authentication issue with aws provider"
2. "authentication: handle token refresh edge cases"
...
```

### Thematic Interpretation

Common cluster themes identified include:
- **Authentication/Authorization**: OAuth, tokens, access control
- **Provider Configuration**: AWS, Azure, GCP provider issues
- **Resource Management**: Instance creation, network configuration
- **State Management**: Terraform state file issues
- **Dependency Resolution**: Module and resource dependencies

## Advanced Features

### Custom Stopwords
The pipeline supports dynamic stopword detection:
```yaml
stopwords_heuristics:
  merge_terms: ['merge', 'pull', 'branch']
  domain_common_terms: ['terraform', 'provider', 'resource']
  keep_frequent_terms: ['fix', 'error', 'bug', 'issue']  # Never filter these
```

### Semantic Message Extraction
Intelligent parsing of merge commit messages:
```python
# Input: "Merge pull request #123 from feature/auth-fix\n\nFix OAuth token refresh"
# Output: "feature auth fix Fix OAuth token refresh"
```

### Flexible Filtering
Multiple filtering strategies:
- File extension-based relevance
- Directory pattern exclusions
- Commit message quality thresholds
- Repository-specific filtering

## Performance and Scalability

### Computational Requirements
- **Memory**: 4-8GB RAM for typical datasets (~7,000 commits)
- **Processing Time**: 
  - Stage 1: 10-30 minutes (depending on dataset size)
  - Stage 2: 5-15 minutes
- **Storage**: ~100MB for embeddings and results

### Optimization Tips
- Use cached embeddings (`sbert_embeddings.pkl`) for repeated runs
- Adjust HDBSCAN parameters for dataset size
- Use sampling for initial exploration of large datasets
- Configure appropriate batch sizes for embedding generation

## Troubleshooting

### Common Issues

**Memory errors during embedding**:
- Reduce batch size in SBERT configuration
- Use smaller model (e.g., `all-MiniLM-L6-v2`)
- Process data in chunks

**Poor clustering results**:
- Adjust HDBSCAN `min_cluster_size` parameter
- Modify UMAP `n_neighbors` for different granularity
- Review stopword configuration for domain relevance

**Empty or noisy clusters**:
- Check input data quality and filtering criteria
- Adjust `cluster_selection_epsilon` parameter
- Review semantic message extraction patterns

### Debug Output
Enable verbose logging by modifying the scripts:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check intermediate results:
- Corpus statistics in `corpus_overview.md`
- Embedding dimensions and quality
- Cluster size distribution

## Integration with Research Workflow

The NLP analysis module integrates with the broader research workflow:

1. **Input**: Mining results from `src/miner/` 
2. **Processing**: Two-stage NLP pipeline
3. **Output**: Clustered datasets and thematic analysis
4. **Next Steps**: Mutation operator development based on identified patterns

### Research Applications
- **Defect Pattern Discovery**: Identify common bug types in IaC
- **Mutation Operator Design**: Inform mutation testing strategies
- **Quality Assessment**: Evaluate commit message quality and patterns
- **Repository Comparison**: Compare defect patterns across projects

## Contributing

When contributing to the NLP analysis module:

1. **Code Quality**: Follow existing patterns and documentation
2. **Configuration**: Update relevant YAML files for new features
3. **Testing**: Validate with different datasets and parameters
4. **Documentation**: Update this README and inline comments
5. **Reproducibility**: Ensure deterministic results with fixed seeds

## References and Methodology

The NLP pipeline implements state-of-the-art techniques:

- **Sentence-BERT**: Reimers & Gurevych (2019) for high-quality sentence embeddings
- **UMAP**: McInnes et al. (2018) for dimensionality reduction
- **HDBSCAN**: Campello et al. (2013) for density-based clustering
- **spaCy**: Industrial-strength NLP for text preprocessing

For detailed algorithmic specifications, see the project's appendix documentation.
