from github import Github
import time
from datetime import datetime
import logging
import yaml

# Configure logging to file at the top of the file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('github_api.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class GitHubAPI:
    """
    Interface com GitHub API para buscar repositórios Terraform.
    Retorna lista de dicts com nome, estrelas, forks e commits totais.
    Attributes:
        client (Github): Cliente da API do GitHub inicializado com token.
    """
    def __init__(self, token):
        """
        Inicializa a interface com a API do GitHub.
        Args:
            token (str): Token de autenticação para a API do GitHub.
        """
        self.client = Github(token)
        logging.info("GitHubAPI initialized. Star and commit filters will be passed via method arguments.")

    def _check_rate_limit(self, min_remaining=10):
        """
        Verifica o limite de taxa da API e espera se necessário.
        Args:
            min_remaining (int): Número mínimo de chamadas restantes antes de pausar.
        Returns:
            bool: True se o limite foi atingido e precisou esperar, False caso contrário.
        """
        rate_limit = self.client.get_rate_limit()
        core_limit = rate_limit.core
        
        if core_limit.remaining < min_remaining:
            reset_time = core_limit.reset.timestamp()
            current_time = datetime.now().timestamp()
            sleep_time = max(1, reset_time - current_time + 5)  
            logging.info(f"API Rate Limit: {core_limit.remaining}/{core_limit.limit}. Aguardando {sleep_time:.0f}s até reset.")
            time.sleep(sleep_time)
            return True
        return False

    def get_total_commits(self, repo):
        """
        Retorna o número total de commits em um repositório.
        
        Implementa mecanismo de espera para respeitar limites de taxa.
        
        Args:
            repo (Repository): Objeto de repositório do PyGithub.
            
        Returns:
            int: Número total de commits ou 0 em caso de erro.
        """
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                total_commits = repo.get_commits().totalCount
                return total_commits
            except Exception as e:
                if "API rate limit exceeded" in str(e):
                    self._check_rate_limit()
                    retry_count += 1
                else:
                    logging.error(f"Erro ao obter commits para {repo.full_name}: {e}")
                    return 0
        
        logging.error(f"Máximo de tentativas excedido ao obter commits para {repo.full_name}")
        return 0

    def search_repositories(self, language='Terraform', limit=50, sort_by="stars", order="desc", stars_filter=10, min_commits_filter=0, exclude_keywords=None):
        """
        Busca repositórios pela API do GitHub usando diferentes estratégias de busca.
        Uses provided 'stars_filter' and 'min_commits_filter' for filtering.
        """
        if exclude_keywords is None:
            exclude_keywords = []
        # Preferred order for search_strategies, most specific ones first.
        # The star filter (stars_filter) is part of some queries for API efficiency.
        # All results are then strictly filtered by stars_filter and min_commits_filter.
        search_strategies = [
            f"language:{language} stars:>={stars_filter}", # Primary: language + configured stars
            f"terraform stars:>={stars_filter}",           # Terraform keyword + configured stars
            f"topic:terraform stars:>={stars_filter}",   # Topic terraform + configured stars
            f"filename:.tf stars:>={stars_filter}",      # Files .tf + configured stars
            f"language:HCL stars:>={stars_filter}",      # Language HCL + configured stars
            # Broader queries if the above yield too few results. 
            # These will be filtered by stars and commits post-API call.
            "terraform",
            f"language:{language}",
            "topic:terraform",
            "filename:.tf",
            "language:HCL"
        ]
        
        repos = []
        found_repo_full_names = set() # To avoid processing duplicate repositories
        
        for query_string in search_strategies: 
            if len(repos) >= limit:
                break 
                
            logging.info(f"Buscando repositórios com: {query_string}")
            
            try:
                results = self.client.search_repositories(
                    query=query_string, 
                    sort=sort_by,
                    order=order
                )
                
                if results.totalCount == 0:
                    logging.info(f"Nenhum resultado encontrado para: {query_string}")
                    continue
                
                logging.info(f"{results.totalCount} repositórios encontrados para '{query_string}'. Processando...")

                # Processa os resultados da query atual
                for repo_candidate in results:
                    if len(repos) >= limit:
                        break # Stop if we have collected enough repositories

                    if repo_candidate.full_name in found_repo_full_names:
                        continue # Skip if already processed

                    # 0. Apply exclusion filter before any other check
                    if any(keyword.lower() in repo_candidate.full_name.lower() for keyword in exclude_keywords):
                        logging.info(f"Skipping repo {repo_candidate.full_name} due to exclusion keyword.")
                        found_repo_full_names.add(repo_candidate.full_name) # Add to found to avoid re-processing from other queries
                        continue

                    # 1. Apply star filter (from argument)
                    if repo_candidate.stargazers_count < stars_filter:
                        logging.debug(f"Skipping {repo_candidate.full_name} (stars: {repo_candidate.stargazers_count} < {stars_filter})")
                        continue

                    # 2. Get total commits (only if star filter passed)
                    total_commits = self.get_total_commits(repo_candidate)

                    # 3. Apply commit filter (from argument)
                    if total_commits < min_commits_filter:
                        logging.debug(f"Skipping {repo_candidate.full_name} (commits: {total_commits} < {min_commits_filter})")
                        continue
                    
                    # If all filters pass, add to our list
                    repo_info = {
                        'repo_full_name': repo_candidate.full_name,
                        'stars': repo_candidate.stargazers_count,
                        'forks': repo_candidate.forks_count,
                        'commits': total_commits
                    }
                    
                    repos.append(repo_info)
                    found_repo_full_names.add(repo_candidate.full_name)
                    logging.debug(f"Adicionado {repo_candidate.full_name} (Stars: {repo_candidate.stargazers_count}, Commits: {total_commits})")

            except Exception as e:
                logging.error(f"Erro na busca com query '{query_string}': {e}")
                # Optionally, implement specific error handling, e.g., for rate limits if not handled by _check_rate_limit
                if "rate limit" in str(e).lower():
                    self._check_rate_limit() # Ensure we wait if it's a rate limit issue
                continue
        
        logging.info(f"Total de {len(repos)} repositórios coletados após filtros.")
        return repos[:limit] # Ensure the final list strictly respects the limit