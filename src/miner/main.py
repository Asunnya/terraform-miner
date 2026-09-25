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

    parser.add_argument('--replay-from', help='Diretório de uma mineração anterior (estrutura de ../data): '
                        'minera de novo os mesmos repositórios, até o mesmo head_hash, sem buscar no GitHub')
    parser.add_argument('--replay-cutoff', help='Data ISO de corte para repos sem head_hash no replay '
                        '(padrão: data de modificação do logs/summary.json antigo)')
    parser.add_argument('--resume', action='store_true', help='Continua um --replay-from interrompido: aceita '
                        '../data/dataset e checkpoints com dados; repos com checkpoint não são minerados de novo')

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

REPO_INFO_KEYS = ('repo_full_name', 'stars', 'forks', 'commits')


def _read_head_hash(jsonl_path):
    """Retorna o primeiro head_hash de um data.jsonl, ou None."""
    if not os.path.exists(jsonl_path):
        return None
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            head = json.loads(line).get('head_hash')
            if head:
                return head
    return None


def load_replay_plan(replay_dir, cutoff=None):
    """
    Monta a lista de repositórios de uma mineração anterior para minerá-los de novo sobre o mesmo histórico.

    Lê de replay_dir (um diretório com a estrutura de ../data), nesta ordem de prioridade:
      - logs/summary.json: lista de repositórios processados e, se a mineração antiga já foi um
        replay, o replay_rev exato de cada um (vale mesmo para repos que não exportaram registros);
      - dataset/<owner>_<repo>/data.jsonl: head_hash usado na mineração original;
      - checkpoints/<owner>_<repo>.json: last_head, se o dataset não tiver head_hash.

    Args:
        replay_dir (str): Diretório da mineração antiga
        cutoff (str): Data ISO usada para repos sem head_hash. Padrão: mtime do summary.json antigo.

    Returns:
        tuple: (repos, cutoff), onde cada repo tem os campos de REPO_INFO_KEYS mais 'replay_head' (ou None)
    """
    summary_path = os.path.join(replay_dir, 'logs', 'summary.json')
    with open(summary_path, 'r', encoding='utf-8') as f:
        old_summary = json.load(f)

    if cutoff is None:
        from datetime import datetime
        cutoff = datetime.fromtimestamp(os.path.getmtime(summary_path)).astimezone().isoformat(timespec='seconds')

    repos = []
    for old in old_summary.get('processed_repositories_summary', []):
        info = {k: old[k] for k in REPO_INFO_KEYS if k in old}
        key = info['repo_full_name'].replace('/', '_')
        head = old.get('replay_rev') or _read_head_hash(os.path.join(replay_dir, 'dataset', key, 'data.jsonl'))
        if not head:
            head = load_checkpoint(info['repo_full_name'], os.path.join(replay_dir, 'checkpoints')).get('last_head')
        info['replay_head'] = head
        repos.append(info)
    return repos, cutoff


def resolve_replay_rev(repo_path, head_hash, cutoff):
    """
    Escolhe o commit final da mineração de replay.

    Usa head_hash se ele existir no clone; senão, o último commit de HEAD antes de cutoff.

    Returns:
        tuple: (hash, origem), origem é 'head_hash' ou 'cutoff_date'

    Raises:
        ValueError: se nenhum dos dois puder ser resolvido
    """
    import subprocess

    def git(*args):
        return subprocess.run(['git', '-C', repo_path, *args], capture_output=True, text=True)

    if head_hash:
        if git('cat-file', '-e', f'{head_hash}^{{commit}}').returncode == 0:
            return head_hash, 'head_hash'
        logging.warning(f"[replay] {head_hash[:8]} não existe em {repo_path}; usando a data de corte {cutoff}")

    result = git('rev-list', '-1', f'--before={cutoff}', 'HEAD')
    rev = result.stdout.strip()
    if result.returncode != 0 or not rev:
        raise ValueError(f"Nenhum commit antes de {cutoff} em {repo_path}")
    return rev, 'cutoff_date'


def _dataset_keys(dataset_dir):
    """{repo_dir: {(commit_hash, file), ...}} de todos os data.jsonl de dataset_dir."""
    keys = {}
    if not os.path.isdir(dataset_dir):
        return keys
    for repo_dir in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, repo_dir, 'data.jsonl')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                keys[repo_dir] = {(r['commit_hash'], r['file']) for r in map(json.loads, f)}
    return keys


def compare_datasets(old_dataset_dir, new_dataset_dir):
    """
    Compara os pares (commit_hash, file) da mineração antiga com os da nova.

    Returns:
        dict: totais e, por repositório, quantos pares se mantiveram, são novos ou sumiram,
        com até 20 exemplos dos que sumiram.
    """
    old, new = _dataset_keys(old_dataset_dir), _dataset_keys(new_dataset_dir)
    per_repo = {}
    for repo in sorted(old.keys() | new.keys()):
        o, n = old.get(repo, set()), new.get(repo, set())
        per_repo[repo] = {
            'old': len(o), 'new': len(n), 'kept': len(o & n),
            'only_new': len(n - o), 'only_old': len(o - n),
            'only_old_examples': sorted(o - n)[:20],
        }
    totals = {k: sum(r[k] for r in per_repo.values()) for k in ('old', 'new', 'kept', 'only_new', 'only_old')}
    return {'totals': totals, 'per_repo': per_repo}


def process_repository(repo_info, miner, cloner, exporter, keywords, full_clone=False, clone_depth=None, checkpoints_dir_abs=None, replay_cutoff=None):
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
        replay_cutoff (str): Se definido (modo --replay-from), minera só até repo_info['replay_head']
            ou, sem ele, até o último commit antes dessa data

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
        until_rev = None
        if replay_cutoff:
            until_rev, source = resolve_replay_rev(path, repo_info.get('replay_head'), replay_cutoff)
            entry.update({'replay_rev': until_rev, 'replay_source': source})
            logging.info(f"[{thread_name}] {full_name}: replay até {until_rev[:8]} ({source})")

        logging.info(f"[{thread_name}] Iniciando mineração de commits em {full_name}...")
        commits = miner.mine(path, seen_hashes=processed_commits, last_head=checkpoint["last_head"],
                             until_rev=until_rev)
        
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

FAILED_STATUSES = ('clone_failed', 'mine_failed', 'failed_exceptionally_in_future')


def miner_git_version(project_root=PROJECT_ROOT):
    """SHA do commit do minerador e se há mudanças não commitadas (None fora de um repo git)."""
    import subprocess

    def git(*args):
        return subprocess.run(['git', '-C', project_root, *args], capture_output=True, text=True)

    head = git('rev-parse', 'HEAD')
    if head.returncode != 0:
        return {'commit': None, 'dirty': None}
    return {'commit': head.stdout.strip(), 'dirty': bool(git('status', '--porcelain', '--', 'src/miner').stdout.strip())}


def dataset_counts(dataset_dir):
    """Registros (commit, arquivo) e commits únicos exportados em dataset_dir."""
    keys = _dataset_keys(dataset_dir)
    return {
        'raw_record_count': sum(len(v) for v in keys.values()),
        'unique_commit_count': sum(len({h for h, _ in v}) for v in keys.values()),
        'repositories_with_records': sum(1 for v in keys.values() if v),
    }


def resolve_github_token(cli_token, cfg, env_file=os.path.join(PROJECT_ROOT, '.env')):
    """--token, then GITHUB_TOKEN (shell or the gitignored .env), then config.yaml's github_token."""
    if os.path.isfile(env_file):
        from dotenv import load_dotenv
        load_dotenv(env_file, override=False)
    return cli_token or os.environ.get('GITHUB_TOKEN') or cfg.get('github_token')


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
    token = resolve_github_token(args.token, cfg)
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

    replay_dir = os.path.abspath(args.replay_from) if args.replay_from else None
    replay_cutoff = None
    if replay_dir:
        # O replay precisa começar do zero: checkpoints pulariam commits e o DataExporter
        # manteria registros antigos com o mesmo (commit_hash, file).
        if replay_dir == os.path.abspath(DATA_DIR):
            raise SystemExit(f"--replay-from não pode ser o próprio {DATA_DIR}; mova a mineração antiga antes.")
        for d in (base_dataset_dir_for_export, CHECKPOINTS_DIR):
            if os.listdir(d) and not args.resume:
                raise SystemExit(f"--replay-from exige {d} vazio; mova ou apague o conteúdo antes "
                                 "(ou use --resume para continuar um replay interrompido).")

        repos_to_process, replay_cutoff = load_replay_plan(replay_dir, args.replay_cutoff)
        without_head = [r['repo_full_name'] for r in repos_to_process if not r['replay_head']]
        logging.info(f"[MainThread] Replay de {replay_dir}: {len(repos_to_process)} repositórios, "
                     f"{len(without_head)} sem head_hash (usarão a data de corte {replay_cutoff}).")
    else:
        # Fetch repositories to process (the only GitHub API use; --replay-from needs no token)
        logging.info("[MainThread] Searching for repositories...")
        api = GitHubAPI(token)
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

    from datetime import datetime, timezone
    run_started = datetime.now(timezone.utc)
    mining_summary = {
        'run_id': f"mine-{run_started.strftime('%Y%m%dT%H%M%SZ')}",
        'started_at': run_started.isoformat(timespec='seconds'),
        'miner_version': miner_git_version(),
        'parameters': {
            'stars': effective_stars, # Use effective value
            'min_commits': effective_min_commits, # Use effective value
            'keywords': keywords_cfg,
            'full_clone': args.full_clone or cfg.get('clone_options', {}).get('full_clone', False),
            'clone_depth': args.clone_depth if args.clone_depth is not None else cfg.get('clone_options', {}).get('depth'),
            'limit': args.limit or cfg.get('limit'),
            'stats_only': args.stats_only,
            'workers': args.workers or os.cpu_count(),
            'replay_from': replay_dir,
            'replay_cutoff': replay_cutoff
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
                checkpoints_dir_abs=CHECKPOINTS_DIR,
                replay_cutoff=replay_cutoff
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
    mining_summary['total_repositories_failed'] = len([r for r in processed_repo_results if r.get('status') in FAILED_STATUSES])
    mining_summary['dataset_counts'] = dataset_counts(base_dataset_dir_for_export)
    mining_summary['finished_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    # Adicionar mais estatísticas agregadas se necessário

    with open(SUMMARY_PATH, 'w', encoding='utf-8') as f_sum:
        json.dump(mining_summary, f_sum, indent=2, ensure_ascii=False)

    if replay_dir:
        comparison = compare_datasets(os.path.join(replay_dir, 'dataset'), base_dataset_dir_for_export)
        comparison_path = os.path.join(LOGS_DIR, 'replay_comparison.json')
        with open(comparison_path, 'w', encoding='utf-8') as f_cmp:
            json.dump(comparison, f_cmp, indent=2, ensure_ascii=False)
        logging.info(f"[MainThread] Replay vs. mineração antiga (commit, arquivo): {comparison['totals']}")
        logging.info(f"[MainThread] Comparação por repositório salva em {comparison_path}")

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