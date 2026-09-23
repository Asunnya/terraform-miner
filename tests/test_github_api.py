from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.miner import github_api
from src.miner.github_api import GitHubAPI


class FakeResults(list):
    @property
    def totalCount(self):
        return len(self)


def fake_repo(full_name, stars=1000, commits=1000, forks=5):
    repo = MagicMock()
    repo.full_name = full_name
    repo.stargazers_count = stars
    repo.forks_count = forks
    repo.get_commits.return_value.totalCount = commits
    return repo


@pytest.fixture
def api():
    api = GitHubAPI(token=None)
    api.client = MagicMock()
    return api


def _set_search_results(api, results_by_query):
    """results_by_query: {query: [repos] | Exception}; queries ausentes retornam vazio."""
    calls = []

    def search(query, sort, order):
        calls.append(query)
        result = results_by_query.get(query, [])
        if isinstance(result, Exception):
            raise result
        return FakeResults(result)

    api.client.search_repositories.side_effect = search
    return calls


def test_search_filters_and_returns_repo_info(api):
    _set_search_results(api, {
        "language:Terraform stars:>=100": [
            fake_repo("org/good", stars=500, commits=800, forks=7),
            fake_repo("org/terraform-tutorial"),       # excluído por keyword
            fake_repo("org/few-stars", stars=50),      # abaixo do mínimo de estrelas
            fake_repo("org/few-commits", commits=10),  # abaixo do mínimo de commits
        ],
    })

    repos = api.search_repositories(limit=10, stars_filter=100, min_commits_filter=100,
                                    exclude_keywords=["Tutorial"])

    assert repos == [{"repo_full_name": "org/good", "stars": 500, "forks": 7, "commits": 800}]


def test_search_deduplicates_across_strategies_and_respects_limit(api):
    calls = _set_search_results(api, {
        "language:Terraform stars:>=10": [fake_repo("org/a")],
        "terraform stars:>=10": [fake_repo("org/a"), fake_repo("org/b"), fake_repo("org/c")],
    })

    repos = api.search_repositories(limit=2)

    assert [r["repo_full_name"] for r in repos] == ["org/a", "org/b"]
    assert len(calls) == 2  # para de buscar quando atinge o limite


def test_search_continues_after_query_error(api):
    _set_search_results(api, {
        "language:Terraform stars:>=10": RuntimeError("boom"),
        "terraform stars:>=10": [fake_repo("org/a")],
    })

    repos = api.search_repositories(limit=1)

    assert [r["repo_full_name"] for r in repos] == ["org/a"]


def test_get_total_commits_retries_on_rate_limit(api, monkeypatch):
    check = MagicMock()
    monkeypatch.setattr(api, "_check_rate_limit", check)
    repo = MagicMock()
    repo.get_commits.side_effect = [Exception("API rate limit exceeded"), SimpleNamespace(totalCount=42)]

    assert api.get_total_commits(repo) == 42
    check.assert_called_once()


def test_get_total_commits_gives_up_after_max_retries(api, monkeypatch):
    monkeypatch.setattr(api, "_check_rate_limit", MagicMock())
    repo = MagicMock()
    repo.get_commits.side_effect = Exception("API rate limit exceeded")

    assert api.get_total_commits(repo) == 0
    assert repo.get_commits.call_count == 3


def test_get_total_commits_other_error_returns_zero(api):
    repo = MagicMock()
    repo.get_commits.side_effect = Exception("404 Not Found")
    assert api.get_total_commits(repo) == 0
    assert repo.get_commits.call_count == 1


def _set_rate_limit(api, remaining):
    core = SimpleNamespace(remaining=remaining, limit=5000, reset=datetime.now() + timedelta(seconds=60))
    api.client.get_rate_limit.return_value = SimpleNamespace(core=core)


def test_check_rate_limit_sleeps_when_low(api, monkeypatch):
    sleep = MagicMock()
    monkeypatch.setattr(github_api.time, "sleep", sleep)
    _set_rate_limit(api, remaining=3)

    assert api._check_rate_limit() is True
    (seconds,), _ = sleep.call_args
    assert 55 <= seconds <= 66


def test_check_rate_limit_no_sleep_when_enough(api, monkeypatch):
    sleep = MagicMock()
    monkeypatch.setattr(github_api.time, "sleep", sleep)
    _set_rate_limit(api, remaining=4000)

    assert api._check_rate_limit() is False
    sleep.assert_not_called()
