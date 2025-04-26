# Terraform Miner

A tool for mining and analyzing Terraform code from GitHub repositories to identify bug fixes and common issues.

## Overview

Terraform Miner helps researchers and developers understand bug patterns in Infrastructure as Code (IaC) by:

1. Mining Terraform code repositories on GitHub
2. Identifying bug-fixing commits based on commit messages
3. Extracting and analyzing changes in Terraform files
4. Building a dataset of common bugs and their fixes
5. Providing deep semantic analysis of Terraform code structures

## Key Features

- **GitHub Repository Mining**: Search for and clone Terraform repositories
- **Commit Analysis**: Find bug-fixing commits using keyword matching
- **AST Analysis**: Parse Terraform HCL into abstract syntax trees
- **Dependency Tracking**: Build graphs of resource dependencies
- **Change Analysis**: Compare versions to identify semantic changes

## Installation

### Prerequisites

- Python 3.12+
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/terraform-miner.git
cd terraform-miner

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Mining Terraform Repositories

To execute the mining process:

```bash
python main.py --config config.yaml --limit 50
```

The process will:
1. Search for Terraform repositories on GitHub
2. Clone repositories to the `repos` directory
3. Mine commits containing fix-related keywords
4. Export relevant data to the `dataset` directory
5. Generate summary files (`clone_summary.json` and `clone_summary.txt`)

### Analyzing Terraform Code

The project includes a powerful `TerraformAstAnalyzer` for deep semantic analysis:

```python
from analysis.terraform_ast import TerraformAstAnalyzer

# Parse a Terraform file
analyzer = TerraformAstAnalyzer()
ast = analyzer.parse_file("path/to/main.tf")

# Analyze resource dependencies
dependencies = analyzer.get_resource_dependencies("aws_instance.web")
dependents = analyzer.get_resource_dependents("aws_vpc.main")

# Find resources by type
instances = analyzer.find_resources_by_type("aws_instance")

# Check resource count/for_each
count = analyzer.get_resource_count("aws_instance.web")
```

### Using the Example Script

The project includes an example script demonstrating the AST analyzer:

```bash
# Analyze a Terraform file
python analysis/terraform_ast_example.py analyze path/to/main.tf

# Compare two Terraform files
python analysis/terraform_ast_example.py compare old_version.tf new_version.tf

# Find references to a specific resource
python analysis/terraform_ast_example.py references main.tf aws_vpc main
```

## Project Structure

- `/miner`: Core mining functionality modules
- `/analysis`: Analysis scripts and notebooks
- `/dataset`: Storage for mined data (created during execution)
- `/repos`: Temporary storage for cloned repositories (created during execution)
- `/reports`: Generated reports and visualizations
- `/tests`: Test cases for the mining functionality

## Advanced Features

### TerraformAstAnalyzer

The `TerraformAstAnalyzer` class provides deep semantic analysis of Terraform code:

- **Complete AST parsing**: Parses HCL into a hierarchical structure
- **Resource dependency tracking**: Builds a graph of resource dependencies
- **Reference resolution**: Resolves `${...}` references and `depends_on` declarations
- **Attribute path tracking**: Maintains full paths to attributes (e.g., "aws_instance.web.ebs_block_device[0].volume_size")
- **Terraform-specific constructs**: Handles `count`, `for_each`, and `dynamic` blocks
- **Change analysis**: Compares different versions to identify semantic changes
- **Visualization**: Generates dependency graphs from Terraform configurations

Sample usage for analyzing changes:

```python
from analysis.diff_utils import analyze_terraform_changes

# Compare two versions of Terraform code
with open('old_version.tf', 'r') as f_old, open('new_version.tf', 'r') as f_new:
    old_content = f_old.read()
    new_content = f_new.read()

changes = analyze_terraform_changes(old_content, new_content)

# Examine the changes
print(f"Added attributes: {len(changes['added'])}")
print(f"Removed attributes: {len(changes['removed'])}")
print(f"Modified attributes: {len(changes['modified'])}")
print(f"Dependency changes: {len(changes['dependency_changes'])}")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
