import os
from git import Repo, GitCommandError

class RepoCloner:
    """
    Clonagem robusta de repos, tratando nomes longos e erros.
    """
    def __init__(self, base_path='repos'):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def clone(self, full_name):
        owner, name = full_name.split('/')
        safe_name = f"{owner}_{name}"[:200] 
        dest = os.path.join(self.base_path, safe_name)
        if os.path.isdir(dest):
            try:
                repo = Repo(dest)
                repo.remotes.origin.fetch()
                return dest
            except GitCommandError as e:
                raise RuntimeError(f"Fetch failed: {e}")
        try:
            Repo.clone_from(f"https://github.com/{full_name}.git", dest)
            return dest
        except GitCommandError as e:
            raise RuntimeError(e.stderr or str(e))