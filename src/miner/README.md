# Mining Module

## Overview
This module automates the mining of GitHub repositories, collecting repository metadata and statistics.

## Directory Structure
- `../data/repos/`: Stores the cloned repositories (relative to project root).
- `../data/logs/`: Stores mining logs and summary files (relative to project root).

## Collected Statistics
For each repository:
- Repository name.
- Number of stars (`stargazers_count`).
- Number of forks (`forks_count`).
- Total number of commits.

## Summary Output
- Total number of repositories mined.
- Repository with the most stars.
- Detailed statistics for each repository.

## How to Run

1. **Activate the virtual environment** (from the project root):

   ```powershell
   .\.venv\Scripts\activate.ps1
   ```
   > You should see `(.venv)` at the start of your terminal prompt, indicating the environment is active.

2. **Install dependencies** (if not already installed):

   ```powershell
   uv pip install -r requirements.txt
   ```
   > This ensures all required Python packages are available.

3. **Navigate to the mining module directory** (from the project root):

   ```powershell
   cd src/miner
   ```

4. **Run the mining script**:

   ```powershell
   python main.py --config config.yaml --limit 50
   ```
   - `--config config.yaml`: Path to your configuration file (default: `config.yaml`).
   - `--limit 50`: Maximum number of repositories to process (adjust as needed).
   - Other useful arguments:
     - `--token <YOUR_GITHUB_TOKEN>`: Provide your GitHub token (recommended for higher rate limits).
     - `--stars <MIN_STARS>`: Minimum number of stars for repositories.
     - `--min-commits <MIN_COMMITS>`: Minimum number of commits for repositories.
     - `--keywords <KEYWORD1> <KEYWORD2> ...`: Filter commits by keywords.
     - `--stats-only`: Only collect repository statistics, skip cloning and mining.

   > For a full list of options, run:
   > ```powershell
   > python main.py --help
   > ```

5. **Directory structure**:

   - The script will automatically create the `../data/` directory (one level above the project root, e.g., `terraform-miner/../data/`) if it does not exist.
   - Cloned repositories will be saved in `../data/repos/`.
   - Logs and summary files will be saved in `../data/logs/`.

6. **Check results**:

   - **Cloned repositories:** `../data/repos/`
   - **Logs and summaries:** `../data/logs/`
     - `../data/logs/terraform_miner.log`: Detailed log of the mining process.
     - `../data/logs/summary.json`: Summary of mined repositories and statistics.

## Debugging e Testes

Para depurar problemas na mineração de repositórios, foram criados scripts específicos:

### 1. Teste de Clonagem Individual

Execute o script `test_cloner.py` para testar apenas o componente de clonagem:

```powershell
cd src
python test_cloner.py owner/repositório
```

Isso tenta clonar apenas um repositório específico e gera logs detalhados em `test_cloner.log`.

### 2. Teste de Pesquisa e Clonagem

O script `test_mining.py` testa o fluxo de pesquisa e clonagem:

```powershell
cd src
python test_mining.py [seu_token_github]
```

Os resultados são salvos em:
- `../data/test_results/search_results.json`: Resultados da pesquisa
- `../data/test_results/clone_results.json`: Resultados da clonagem
- Um arquivo de log com timestamp no diretório atual

### 3. Teste Simplificado do Fluxo Completo

Para testar o fluxo completo com configurações simplificadas:

```powershell
cd src
python run_simple_test.py
```

Este script:
- Busca apenas 2 repositórios com limites menores de estrelas/commits
- Executa todo o fluxo de mineração com logging verboso
- Gera um arquivo de log com timestamp

### Verificação de Logs e Resultados

Para diagnosticar problemas:

1. Verifique os logs em:
   - `../data/logs/terraform_miner.log`: Log principal (localizado um nível acima do projeto)
   - Logs específicos de cada teste (`test_cloner.log`, `test_mining_*.log`, etc.)

2. Verifique os diretórios criados:
   - `../data/repos/`: Verifique se contém os repositórios clonados
   - `../data/logs/`: Verifique os arquivos de sumário
   - `../data/test_repos/`: Repositórios clonados pelos scripts de teste (assumindo que também estarão em `../data/`)

3. Estrutura de diretórios esperada:
   ```
   your-parent-directory/
   ├── data/             # Nova localização da pasta de dados
   │   ├── repos/        
   │   ├── logs/         
   │   ├── dataset/      
   │   └── test_repos/   
   └── terraform-miner/  # Seu diretório de projeto
       └── src/
           ├── miner/    
           ├── test_*.py 
           └── ...
   ``` 