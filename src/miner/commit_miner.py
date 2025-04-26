import os
from git import Repo

class CommitMiner:
    """
    Mine commits containing keywords and Terraform file changes.
    """
    def __init__(self, keywords=None):
        self.keywords = [k.lower() for k in (keywords or [])]

    def mine(self, repo_path):
        """
        Iterate commits, filter by message and .tf changes.
        Returns list of dicts.
        """
        repo = Repo(repo_path)
        entries = []
        for commit in repo.iter_commits():
            msg = commit.message.lower()
            if any(k in msg for k in self.keywords):
                diffs = commit.diff(commit.parents[0] if commit.parents else None, create_patch=True)
                tf_diffs = [d for d in diffs if d.b_path and d.b_path.endswith('.tf')]
                if tf_diffs:
                    for d in tf_diffs:
                        entries.append({
                            'repo': os.path.basename(repo_path),
                            'commit_hash': commit.hexsha,
                            'author': commit.author.name,
                            'date': commit.committed_datetime.isoformat(),
                            'message': commit.message.strip(),
                            'file': d.b_path,
                            'patch': d.diff.decode('utf-8', errors='ignore')
                        })
        return entries