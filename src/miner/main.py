# main.py (atualizado)
import os
import argparse
import yaml
import json
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from src.miner.github_api import GitHubAPI
from src.miner.repo_cloner import RepoCloner
from src.miner.commit_miner import CommitMiner
from src.miner.data_exporter import DataExporter
from src.miner.utils import is_real_infra_file
import logging

# --- Path setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))  

#PRINT THE PROJECT ROOT
print(f"PROJECT_ROOT: {PROJECT_ROOT}")

# --- Path setup for data directory one level above project root ---
DATA_PARENT_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, '..'))
DATA_DIR = os.path.join(DATA_PARENT_DIR, 'data')

REPOS_DIR = os.path.join(DATA_DIR, 'repos')
LOGS_DIR = os.path.join(DATA_DIR, 'logs')
DATASET_DIR = os.path.join(DATA_DIR, 'dataset')
CHECKPOINTS_DIR = os.path.join(DATA_DIR, 'checkpoints')

# Cria os diretórios necessários
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPOS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# --- Logging setup ---
DETAILED_LOG_PATH = os.path.join(LOGS_DIR, 'terraform_miner.log')
SUMMARY_PATH = os.path.join(LOGS_DIR, 'summary.json')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

# Lock for thread-safe logging and summary updates
summary_lock = threading.Lock()

def load_config(path):
    """
    Carrega configurações do arquivo YAML.
    
    Args:
        path (str): Caminho para o arquivo de configuração.
        
    Returns:
        dict: Configurações carregadas ou dict vazio se arquivo não existir.
    """
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    logging.warning(f"Arquivo de configuração não encontrado: {path}")
    return {}


def parse_args():
    """
    Configura e analisa argumentos de linha de comando.
    
    Returns:
        Namespace: Objeto com argumentos processados.
    """
    parser = argparse.ArgumentParser(description="Terraform GitHub Miner")
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--token', help='GitHub token (overrides config)')
    parser.add_argument('--stars', type=int, help='Minimum stars (overrides config)')
    parser.add_argument('--min-commits', type=int, help='Minimum commits (overrides config)')
    parser.add_argument('--keywords', nargs='+', help='Commit message keywords (overrides config)')
    parser.add_argument('--limit', type=int, help='Max repos to process (overrides config)')
    parser.add_argument('--stats-only', action='store_true', help='Only collect statistics, no cloning or mining')
    parser.add_argument('--workers', type=int, help='Number of parallel workers (default: CPU count)')
    
    # Add new clone options
    parser.add_argument('--full-clone', action='store_true', help='Use full clone instead of partial clone')
    parser.add_argument('--clone-depth', type=int, help='Depth for shallow clones (use 0 for full history)')
    
    return parser.parse_args()

def load_checkpoint(repo_name, checkpoints_dir_abs):
    """
    Load checkpoint for a repository to support resumption and idempotency.
    
    Args:
        repo_name (str): Repository name in format 'owner/repo'
        checkpoints_dir_abs (str): Absolute path to the checkpoints directory
        
    Returns:
        dict: Checkpoint data with previously processed commits
    """
    checkpoint_path = os.path.join(checkpoints_dir_abs, repo_name.replace('/', '_') + '.json')
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load checkpoint for {repo_name}: {e}")
    
    # Return empty checkpoint if file doesn't exist or loading failed
    return {
        "processed_commits": [],
        "last_head": None,
        "last_updated": None,
        "total_commits_processed": 0
    }

def save_checkpoint(repo_name, checkpoint_data, checkpoints_dir_abs):
    """
    Save checkpoint for a repository to support resumption and idempotency.
    
    Args:
        repo_name (str): Repository name in format 'owner/repo'
        checkpoint_data (dict): Checkpoint data to save
        checkpoints_dir_abs (str): Absolute path to the checkpoints directory
    """
    checkpoint_path = os.path.join(checkpoints_dir_abs, repo_name.replace('/', '_') + '.json')
    # Add timestamp to checkpoint
    from datetime import datetime
    checkpoint_data["last_updated"] = datetime.now().isoformat()
    
    with open(checkpoint_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

def process_repository(repo_info, miner, cloner, exporter, keywords, full_clone=False, clone_depth=None, checkpoints_dir_abs=None):
    """
    Process a single repository: clone, mine commits, and export data.
    This function is designed to be run in parallel.
    
    Args:
        repo_info (dict): Repository information
        miner (CommitMiner): Instance of CommitMiner
        cloner (RepoCloner): Instance of RepoCloner
        exporter (DataExporter): Instance of DataExporter
        keywords (list): List of keywords to search in commit messages
        full_clone (bool): Whether to use full clone instead of partial clone
        clone_depth (int): Depth for shallow clones (None for default)
        checkpoints_dir_abs (str): Absolute path to the checkpoints directory
        
    Returns:
        dict: Entry with processing results and status
    """
    full_name = repo_info['repo_full_name']
    entry = repo_info.copy()  # Já inclui stars, forks e commits
    thread_name = threading.current_thread().name
    
    # Load checkpoint to support resumption
    checkpoint = load_checkpoint(full_name, checkpoints_dir_abs)
    processed_commits = set(checkpoint["processed_commits"])
    
    # Print checkpoint info
    logging.info(f"[{thread_name}][{full_name}] Found checkpoint with {len(processed_commits)} processed commits")
    
    # Tenta clonar o repositório
    try:
        logging.info(f"[{thread_name}] Iniciando clonagem de {full_name}...")
        # Pass clone options to the cloner
        path = cloner.clone(
            full_name,
            depth=clone_depth,
            partial_clone=not full_clone
        )
        logging.info(f"[{thread_name}] Repositório {full_name} clonado com sucesso em {path}")
    except Exception as e:
        entry.update({'status': 'clone_failed', 'reason': str(e)})
        logging.error(f"[{thread_name}] Falha ao clonar {full_name}: {e}")
        return entry

    # Tenta minerar os commits
    try:
        logging.info(f"[{thread_name}] Iniciando mineração de commits em {full_name}...")
        commits = miner.mine(path, seen_hashes=processed_commits, last_head=checkpoint["last_head"])
        
        # Filter commits for Terraform infrastructure files
        commits = [c for c in commits if is_real_infra_file(c['file'])]
        
        if commits:
            # Export new commits
            exporter.export(full_name, commits, append=True)
            
            # Update checkpoint with newly processed commits
            new_processed = [c['commit_hash'] for c in commits]
            checkpoint["processed_commits"] = list(processed_commits.union(new_processed))
            
            # Update last_head if there were new commits
            if commits and 'head_hash' in commits[0]:
                checkpoint["last_head"] = commits[0]['head_hash']
            
            # Update total count
            checkpoint["total_commits_processed"] = len(checkpoint["processed_commits"])
            
            # Save updated checkpoint
            save_checkpoint(full_name, checkpoint, checkpoints_dir_abs)
            
            entry.update({
                'status': 'success', 
                'mined_commits': len(commits),
                'total_mined': len(checkpoint["processed_commits"])
            })
            logging.info(f"[{thread_name}] {full_name}: {len(commits)} new commits minerados, total: {len(checkpoint['processed_commits'])}")
        else:
            entry.update({
                'status': 'no_new_commits', 
                'mined_commits': 0,
                'total_mined': len(checkpoint["processed_commits"])
            })
            logging.info(f"[{thread_name}] {full_name}: Nenhum novo commit minerado, total: {len(checkpoint['processed_commits'])}")
    except Exception as e:
        entry.update({'status': 'mine_failed', 'reason': str(e)})
        logging.error(f"[{thread_name}] Falha ao minerar commits em {full_name}: {e}")
    
    return entry

def main():
    """
    Função principal que coordena a mineração de repositórios.
    """
    # Processamento de argumentos e configurações
    args = parse_args()
    config_path_from_arg = args.config
    
    # Path resolution for config file
    if os.path.isabs(config_path_from_arg):
        config_path_resolved = config_path_from_arg
    else:
        # Try relative to PROJECT_ROOT first
        path_rel_to_project_root = os.path.join(PROJECT_ROOT, config_path_from_arg)
        if os.path.exists(path_rel_to_project_root) and os.path.isfile(path_rel_to_project_root):
            config_path_resolved = path_rel_to_project_root
            logging.info(f"Configuration file resolved relative to PROJECT_ROOT: {config_path_resolved}")
        else:
            # Fallback to relative to SCRIPT_DIR (covers default 'config.yaml' next to script)
            path_rel_to_script_dir = os.path.join(SCRIPT_DIR, config_path_from_arg)
            config_path_resolved = path_rel_to_script_dir
            # If resolved to SCRIPT_DIR, and it's the default 'config.yaml', no special log needed here.
            # load_config will warn if it's ultimately not found.
            if config_path_from_arg != 'config.yaml': # Log if user provided a specific relative path that resolved to SCRIPT_DIR
                 logging.info(f"Configuration file resolved relative to SCRIPT_DIR: {config_path_resolved}")

    cfg = load_config(config_path_resolved)

    # --- Path Setup (agora que cfg está carregado) ---
    output_paths_cfg = cfg.get('output_paths', {})
    
    # Determine the base directory for dataset export.
    # Priority:
    # 1. Absolute path from config's output_paths.dataset_dir.
    # 2. Relative path from config's output_paths.dataset_dir (resolved against PROJECT_ROOT).
    # 3. Default to the globally defined DATASET_DIR.
    dataset_config_path = output_paths_cfg.get('dataset_dir')
    if dataset_config_path:
        if os.path.isabs(dataset_config_path):
            # Config provides an absolute path, use it
            base_dataset_dir_for_export = dataset_config_path
        else:
            # Config provides a relative path, resolve it against PROJECT_ROOT
            base_dataset_dir_for_export = os.path.join(PROJECT_ROOT, dataset_config_path)
    else:
        # No dataset_dir in config, use the globally defined DATASET_DIR
        # This ensures alignment with sample_preparer.py's expectations.
        base_dataset_dir_for_export = DATASET_DIR
    
    # Ensure this base directory for export exists (DataExporter will create subdirs)
    os.makedirs(base_dataset_dir_for_export, exist_ok=True)
    logging.info(f"[MainThread] Diretório base para exportação de datasets: {base_dataset_dir_for_export}")

    # Logging setup (agora que LOGS_DIR está definido e pode ser usado)

    # Inicialização dos componentes principais
    token = args.token or cfg.get('github_token')
    api = GitHubAPI(token)
    cloner = RepoCloner(REPOS_DIR) 
    
    # CommitMiner é instanciado sem caminhos, pois não lida diretamente com exportação.
    keywords_cfg = args.keywords or cfg.get('keywords', [])
    commit_miner = CommitMiner(keywords_cfg)

    # Criar uma ÚNICA instância de DataExporter com o caminho base correto
    data_exporter_instance = DataExporter(base_dataset_dir_for_export)

    # Determine stars and min_commits values, using args, then cfg, then default
    # Default values matching GitHubAPI.search_repositories if nothing else is set
    # However, your config has 500/500, so those should ideally be used if args are not present.
    # The cfg.get('stars') or cfg.get('min_commits') might return None if not in config or config not loaded.
    
    effective_stars = args.stars if args.stars is not None else cfg.get('stars', 10) # Default to 10 if not in args or cfg
    effective_min_commits = args.min_commits if args.min_commits is not None else cfg.get('min_commits', 0) # Default to 0

    # Get exclude_keywords from config
    effective_exclude_keywords = cfg.get('exclude_keywords', [])
    if effective_exclude_keywords:
        logging.info(f"Excluding repositories with keywords: {effective_exclude_keywords}")

    # Ensure they are integers if loaded from cfg (which could be None or wrong type)
    try:
        effective_stars = int(effective_stars)
    except (ValueError, TypeError):
        logging.warning(f"Invalid 'stars' value ({effective_stars}), defaulting to 10.")
        effective_stars = 10

    try:
        effective_min_commits = int(effective_min_commits)
    except (ValueError, TypeError):
        logging.warning(f"Invalid 'min_commits' value ({effective_min_commits}), defaulting to 0.")
        effective_min_commits = 0

    # Get exclude_keywords from config
    effective_exclude_keywords = cfg.get('exclude_keywords', [])
    if effective_exclude_keywords:
        logging.info(f"Excluding repositories with keywords: {effective_exclude_keywords}")

    # Fetch repositories to process
    logging.info("[MainThread] Searching for repositories...")
    repos_to_process = api.search_repositories(
        stars_filter=effective_stars, # Pass the determined stars value
        min_commits_filter=effective_min_commits, # Pass the determined min_commits value
        limit=args.limit or cfg.get('limit', 50), # Default limit to 50 if not in args or cfg
        exclude_keywords=effective_exclude_keywords
    )
    logging.info(f"[MainThread] Found {len(repos_to_process)} repositories to process.")

    # Optional: Filter repositories by min_commits here if desired
    # This section is now somewhat redundant as search_repositories handles this filtering internally.
    # However, it was filtering on repo.get('commits', 0) which could be different if search_repositories didn't fetch commits for all.
    # For now, I will comment out this redundant filtering block as the primary filtering should happen in search_repositories.
    # min_commits_filter_main = args.min_commits or cfg.get('min_commits') # This was the variable name used below
    # if min_commits_filter_main is not None:
    #     logging.info(f"[MainThread] Filtering repositories with at least {min_commits_filter_main} commits.")
    #     original_count = len(repos_to_process)
    #     repos_to_process = [repo for repo in repos_to_process if repo.get('commits', 0) >= min_commits_filter_main]
    #     logging.info(f"[MainThread] {len(repos_to_process)} repositories remaining after min_commits filter (originally {original_count}).")

    if args.stats_only:
        logging.info("[MainThread] Stats-only mode. Exibindo estatísticas e saindo...")
        # ... (potential stats display logic if needed, then exit)
        # For now, just log and prepare summary based on fetched data.
        # Update mining_summary with repository stats if needed here
        # For example:
        mining_summary = {
            'total_repositories_found': len(repos_to_process), # Reflects count after potential min_commits filter
            'repositories_details': repos_to_process # Store basic details
        }
        # Adicionar parâmetros ao sumário em modo stats_only também
        mining_summary['parameters'] = {
            'stars': effective_stars, # Log the effective value used
            'min_commits': effective_min_commits, # Log the effective value used
            'keywords': keywords_cfg,
            'full_clone': args.full_clone or cfg.get('clone_options', {}).get('full_clone', False),
            'clone_depth': args.clone_depth if args.clone_depth is not None else cfg.get('clone_options', {}).get('depth'),
            'limit': args.limit or cfg.get('limit'),
            'stats_only': args.stats_only,
            'workers': args.workers or os.cpu_count()
        }
        with open(SUMMARY_PATH, 'w', encoding='utf-8') as f_sum:
            json.dump(mining_summary, f_sum, indent=2, ensure_ascii=False)
        logging.info(f"[MainThread] Stats-only summary saved to {SUMMARY_PATH}. Exiting.")
        return # Exit if stats_only is true

    mining_summary = {
        'parameters': {
            'stars': effective_stars, # Use effective value
            'min_commits': effective_min_commits, # Use effective value
            'keywords': keywords_cfg,
            'full_clone': args.full_clone or cfg.get('clone_options', {}).get('full_clone', False),
            'clone_depth': args.clone_depth if args.clone_depth is not None else cfg.get('clone_options', {}).get('depth'),
            'limit': args.limit or cfg.get('limit'),
            'stats_only': args.stats_only,
            'workers': args.workers or os.cpu_count()
        }
    }
    # Adicionar estatísticas iniciais dos repositórios encontrados ao sumário
    mining_summary['total_repositories_found'] = len(repos_to_process) # This reflects count after potential min_commits filter
    mining_summary['top_repositories_by_stars'] = sorted(
        [{'repo': r['repo_full_name'], 'stars': r['stars']} for r in repos_to_process], # Corrected: r['stars'] instead of r['stargazers_count']
        key=lambda x: x['stars'],
        reverse=True
    )[:10] # Top 10 for brevity

    # Lista para armazenar os resultados de cada repositório processado
    processed_repo_results = []

    with ThreadPoolExecutor(max_workers=args.workers or os.cpu_count()) as executor:
        futures = {}
        for repo_info in tqdm(repos_to_process, desc="Processando repositórios (submissão)"):
            full_name = repo_info['repo_full_name']
            
            future = executor.submit(
                process_repository,
                repo_info,
                commit_miner, 
                cloner,       
                data_exporter_instance, 
                keywords_cfg, 
                args.full_clone or cfg.get('clone_options', {}).get('full_clone', False),
                args.clone_depth if args.clone_depth is not None else cfg.get('clone_options', {}).get('depth'),
                checkpoints_dir_abs=CHECKPOINTS_DIR
            )
            futures[future] = full_name
        
        # Coleta de resultados e tratamento de exceções
        logging.info(f"[MainThread] Todas as {len(futures)} tarefas de processamento de repositório foram submetidas. Aguardando conclusão...")
        for future in tqdm(as_completed(futures), total=len(futures), desc="Concluindo processamento de repositórios"):
            repo_name_for_future = futures[future]
            try:
                result = future.result()  # Aguarda o resultado desta future específica
                processed_repo_results.append(result)
                # Descomentar para log mais verboso do progresso de cada repo:
                # logging.info(f"[MainThread] Repositório {repo_name_for_future} processado com status: {result.get('status')}")
            except Exception as exc:
                logging.error(f"[MainThread] Repositório {repo_name_for_future} gerou uma exceção: {exc}")
                processed_repo_results.append({
                    'repo_full_name': repo_name_for_future,
                    'status': 'failed_exceptionally_in_future',
                    'reason': str(exc)
                })

    # Coleta de estatísticas e exportação do sumário
    mining_summary['processed_repositories_summary'] = processed_repo_results
    mining_summary['total_repositories_attempted'] = len(repos_to_process)
    mining_summary['total_repositories_successful_clone'] = len([r for r in processed_repo_results if r.get('status') not in ['clone_failed']])
    mining_summary['total_repositories_commits_mined_at_least_one'] = len([r for r in processed_repo_results if r.get('mined_commits', 0) > 0])
    # Adicionar mais estatísticas agregadas se necessário

    with open(SUMMARY_PATH, 'w', encoding='utf-8') as f_sum:
        json.dump(mining_summary, f_sum, indent=2, ensure_ascii=False)

    logging.info("Mining process completed.")
    logging.info(f"Summary saved to {SUMMARY_PATH}")
    logging.info(f"Detailed log saved to {DETAILED_LOG_PATH}")


    # --- Post-Mining Data Preparation and Analysis ---
    # Added as per planning_tarefa_X.md
    
    from src.miner.sample_preparer import (
        create_all_commits_csv,
        confirm_date_range,
        prepare_sample,
        compute_token_coverage,
        extract_bigrams
    )

    # Define paths for post-processing artifacts relative to PROJECT_ROOT
    REPORTS_DIR_POST_PROCESS = os.path.join(PROJECT_ROOT, 'reports')
    ALL_COMMITS_CSV_PATH_POST_PROCESS = os.path.join(REPORTS_DIR_POST_PROCESS, 'all_commits.csv')
    STRATIFIED_SAMPLE_CSV_PATH_POST_PROCESS = os.path.join(REPORTS_DIR_POST_PROCESS, 'sample_messages_stratified.csv')
    TOP_BIGRAMS_JSON_PATH_POST_PROCESS = os.path.join(REPORTS_DIR_POST_PROCESS, 'top_bigrams.json')

    # Ensure reports directory exists
    os.makedirs(REPORTS_DIR_POST_PROCESS, exist_ok=True)

    logging.info("--- Starting Post-Mining Data Preparation and Analysis ---")

    # Corrigido para usar a variável global DATASET_DIR que é J:/projct-tcc/terraform-miner/data/dataset
    dataset_glob_pattern = os.path.join(DATASET_DIR, '*', 'data.jsonl')
    logging.info(f"Using dataset glob pattern for preparer: {dataset_glob_pattern}")

    # Definir caminhos de saída para os artefatos do sample_preparer
    # reports_dir_abs deve ser J:/projct-tcc/terraform-miner/data/reports
    # (supondo que REPORTS_DIR global não foi definido, mas pode ser inferido de LOGS_DIR e DATASET_DIR)
    reports_dir_abs = os.path.join(DATA_DIR, 'reports')
    os.makedirs(reports_dir_abs, exist_ok=True)
    
    all_commits_csv_path = os.path.join(reports_dir_abs, 'all_commits.csv')

    # 1. Create all_commits.csv from the raw data.jsonl files
    logging.info(f"Attempting to create {ALL_COMMITS_CSV_PATH_POST_PROCESS}...")
    if create_all_commits_csv(dataset_glob_pattern, ALL_COMMITS_CSV_PATH_POST_PROCESS):
        logging.info(f"Successfully created or updated {ALL_COMMITS_CSV_PATH_POST_PROCESS}")

        # 2. Confirms dates from the generated all_commits.csv
        logging.info("Confirming date range...")
        min_date, max_date = confirm_date_range(ALL_COMMITS_CSV_PATH_POST_PROCESS)
        if min_date and max_date:
            logging.info(f"Overall commit date range from all_commits.csv: {min_date} to {max_date}")

        # 3. Prepares sample from all_commits.csv
        logging.info(f"Preparing stratified sample {STRATIFIED_SAMPLE_CSV_PATH_POST_PROCESS}...")
        prepare_sample(ALL_COMMITS_CSV_PATH_POST_PROCESS, STRATIFIED_SAMPLE_CSV_PATH_POST_PROCESS, target_n=1000)

        # 4. Check token coverage
        logging.info("Computing token coverage...")
        coverage = compute_token_coverage(ALL_COMMITS_CSV_PATH_POST_PROCESS, top_n=50) 
        logging.info(f"Top-50 token coverage on all_commits.csv: {coverage:.2%}")

        # 5. Extract bigrams
        logging.info("Extracting top bigrams...")
        top_bigrams = extract_bigrams(ALL_COMMITS_CSV_PATH_POST_PROCESS, top_k=20)
        if top_bigrams:
            logging.info(f"Top 20 bigrams extracted: {top_bigrams}")
            try:
                with open(TOP_BIGRAMS_JSON_PATH_POST_PROCESS, 'w', encoding='utf-8') as f_bigram_out:
                    json.dump(top_bigrams, f_bigram_out, indent=2)
                logging.info(f"Top bigrams saved to {TOP_BIGRAMS_JSON_PATH_POST_PROCESS}")
            except Exception as e_json:
                logging.error(f"Error saving top bigrams to JSON: {e_json}")
        else:
            logging.warning("No bigrams were extracted or an error occurred during extraction.")
    else:
        logging.critical(f"CRITICAL: Failed to create {ALL_COMMITS_CSV_PATH_POST_PROCESS}. Subsequent post-processing steps will be skipped.")

    logging.info("--- Post-Mining Data Preparation and Analysis Finished ---")


if __name__ == '__main__':
    main()