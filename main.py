import os
import argparse
import yaml
import json
from tqdm import tqdm
from miner.github_api import GitHubAPI
from miner.repo_cloner import RepoCloner
from miner.commit_miner import CommitMiner
from miner.data_exporter import DataExporter
from miner.utils import is_real_infra_file


def load_config(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


def parse_args():
    parser = argparse.ArgumentParser(description="Terraform GitHub Miner")
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--token', help='GitHub token (overrides config)')
    parser.add_argument('--stars', type=int, help='Minimum stars (overrides config)')
    parser.add_argument('--keywords', nargs='+', help='Commit message keywords (overrides config)')
    parser.add_argument('--limit', type=int, help='Max repos to process (overrides config)')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    token = args.token or cfg.get('token')
    stars = args.stars or cfg.get('stars', 10)
    keywords = args.keywords or cfg.get('keywords', [])
    limit = args.limit or cfg.get('limit', 50)

    if not token:
        raise ValueError("GitHub token must be provided via --token or in config.yaml")

    os.makedirs('repos', exist_ok=True)
    os.makedirs('dataset', exist_ok=True)

    gh = GitHubAPI(token)
    cloner = RepoCloner(base_path='repos')
    miner = CommitMiner(keywords=keywords)
    exporter = DataExporter(base_path='dataset')

    repos = gh.search_repositories(language='Terraform', stars=stars, limit=limit)
    summary = []

    for repo_info in tqdm(repos, desc='Processing repos'):
        full_name = repo_info['full_name']
        repo_stars = repo_info['stars']
        entry = {'repo': full_name, 'stars': repo_stars}
        try:
            path = cloner.clone(full_name)
        except Exception as e:
            entry.update({'status': 'clone_failed', 'reason': str(e)})
            summary.append(entry)
            continue

        try:
            commits = miner.mine(path)
            # Filtrar apenas arquivos .tf válidos
            commits = [c for c in commits if is_real_infra_file(c['file'])]
            if commits:
                exporter.export(full_name, commits)
                entry.update({'status': 'success', 'commits': len(commits)})
            else:
                entry.update({'status': 'no_commits', 'commits': 0})
        except Exception as e:
            entry.update({'status': 'mine_failed', 'reason': str(e)})
        summary.append(entry)

    with open('clone_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open('clone_summary.txt', 'w', encoding='utf-8') as f:
        for s in summary:
            stars_symbol = '*'
            line = f"{s['repo']} ({s['stars']}{stars_symbol}) - {s['status']}"
            if 'reason' in s:
                line += f": {s['reason']}"
            f.write(line + "\n")

    print("\nClone & mining summary saved to clone_summary.json and clone_summary.txt")

if __name__ == '__main__':
    main()