from github import Github

class GitHubAPI:
    """
    Interface com GitHub API para buscar repositórios Terraform.
    Retorna lista de dicts com nome e estrelas.
    """
    def __init__(self, token):
        self.client = Github(token)

    def search_repositories(self, language='Terraform', stars=10, limit=50):
        query = f"language:{language} stars:>={stars}"
        results = self.client.search_repositories(query=query)
        repos = []
        for repo in results[:limit]:
            repos.append({'full_name': repo.full_name, 'stars': repo.stargazers_count})
        return repos