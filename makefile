# Makefile for Terraform Diff Analysis

.PHONY: all setup extract analyze report clean test

# Default target
all: setup extract analyze report

# Setup directories
setup:
	python -c "from src.analysis.config import get_config; \
	           import os; \
	           config = get_config(); \
	           dirs = [ \
	               config['dataset_dir'], \
	               config['output_dir'], \
	               config['checkpoints_dir'], \
	               config['repos_dir'], \
	               os.path.join(config['output_dir'], 'commits'), \
	               os.path.join(config['output_dir'], 'diffs'), \
	               os.path.join(config['output_dir'], 'metrics'), \
	               os.path.join(config['output_dir'], 'visualizations'), \
	               os.path.join(config['output_dir'], 'analysis') \
	           ]; \
	           [os.makedirs(d, exist_ok=True) for d in dirs]; \
	           print('Directory structure created successfully');"

# Extract data from commits
extract: setup
	python run_analysis.py

# Run analysis
analyze: extract
	python -c "import pandas as pd, os, matplotlib.pyplot as plt, seaborn as sns; \
	           from src.analysis.config import get_config; \
	           config = get_config(); \
	           df = pd.read_parquet(os.path.join(config['output_dir'], 'analysis', 'terraform_diffs.parquet')); \
	           plt.figure(figsize=(10, 6)); \
	           sns.countplot(data=df, x='change'); \
	           plt.title('Distribution of Change Types'); \
	           plt.tight_layout(); \
	           plt.savefig(os.path.join(config['output_dir'], 'visualizations', 'change_types.png')); \
	           print('Analysis complete. Visualizations saved to', os.path.join(config['output_dir'], 'visualizations'));"

# Generate report
report: analyze
	python -c "import pandas as pd, os, jinja2; \
	           from src.analysis.config import get_config; \
	           config = get_config(); \
	           df = pd.read_parquet(os.path.join(config['output_dir'], 'analysis', 'terraform_diffs.parquet')); \
	           env = jinja2.Environment(loader=jinja2.FileSystemLoader('.')); \
	           template = env.get_template('report_template.html'); \
	           summary = { \
	               'total_commits': df['commit'].nunique(), \
	               'total_repos': df['repo'].nunique(), \
	               'total_lines': len(df), \
	               'bugfix_commits': df[df['is_bugfix']]['commit'].nunique(), \
	           }; \
	           html = template.render(summary=summary); \
	           report_path = os.path.join(config['output_dir'], 'analysis', 'report.html'); \
	           with open(report_path, 'w') as f: \
	               f.write(html); \
	           print('Report generated:', report_path);"

# Run tests
test:
	python -m unittest discover

# Clean generated files
clean:
	python -c "from src.analysis.config import get_config; \
	           import shutil, os; \
	           config = get_config(); \
	           output_dir = config['output_dir']; \
	           if os.path.exists(output_dir): \
	               shutil.rmtree(output_dir); \
	           print(f'Cleaned {output_dir}');"