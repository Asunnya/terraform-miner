import json
import os
from unittest.mock import MagicMock

import pytest

from src.miner import main
from src.miner.data_exporter import DataExporter
from tests.helpers import make_record, write_jsonl


def _old_mining(tmp_path):
    """Estrutura de ../data de uma mineração antiga com 3 repos."""
    old = tmp_path / "old"
    summary = {"processed_repositories_summary": [
        {"repo_full_name": "a/with-data", "stars": 10, "forks": 1, "commits": 600,
         "status": "no_new_commits", "mined_commits": 0, "total_mined": 5},
        {"repo_full_name": "b/only-checkpoint", "stars": 9, "forks": 1, "commits": 700, "status": "success"},
        {"repo_full_name": "c/nothing", "stars": 8, "forks": 1, "commits": 800, "status": "no_new_commits"},
    ]}
    (old / "logs").mkdir(parents=True)
    (old / "logs" / "summary.json").write_text(json.dumps(summary))
    write_jsonl(str(old / "dataset" / "a_with-data" / "data.jsonl"),
                [make_record(head_hash="HEAD_A"), make_record(commit_hash="b", head_hash="HEAD_A")])
    (old / "checkpoints").mkdir()
    main.save_checkpoint("b/only-checkpoint", {"processed_commits": [], "last_head": "HEAD_B"}, str(old / "checkpoints"))
    return old


def test_load_replay_plan(tmp_path):
    repos, cutoff = main.load_replay_plan(str(_old_mining(tmp_path)), cutoff="2025-05-31T00:00:00")

    assert cutoff == "2025-05-31T00:00:00"
    assert repos == [
        {"repo_full_name": "a/with-data", "stars": 10, "forks": 1, "commits": 600, "replay_head": "HEAD_A"},
        {"repo_full_name": "b/only-checkpoint", "stars": 9, "forks": 1, "commits": 700, "replay_head": "HEAD_B"},
        {"repo_full_name": "c/nothing", "stars": 8, "forks": 1, "commits": 800, "replay_head": None},
    ]


def test_load_replay_plan_default_cutoff_is_summary_mtime(tmp_path):
    old = _old_mining(tmp_path)
    os.utime(old / "logs" / "summary.json", (1748649600, 1748649600))  # 2025-05-31T00:00:00Z
    _, cutoff = main.load_replay_plan(str(old))
    assert cutoff.startswith("2025-05-3")  # 30 ou 31, conforme o fuso local


def test_resolve_replay_rev_uses_head_hash(git_repo):
    path, h = git_repo
    assert main.resolve_replay_rev(path, h["c3"], "2000-01-01") == (h["c3"], "head_hash")


@pytest.mark.parametrize("head", [None, "0" * 40])
def test_resolve_replay_rev_falls_back_to_cutoff(git_repo, head):
    path, h = git_repo
    assert main.resolve_replay_rev(path, head, "2090-01-01") == (h["c5"], "cutoff_date")


def test_resolve_replay_rev_nothing_before_cutoff(git_repo):
    path, _ = git_repo
    with pytest.raises(ValueError, match="Nenhum commit"):
        main.resolve_replay_rev(path, None, "2000-01-01")


def test_compare_datasets(tmp_path):
    write_jsonl(str(tmp_path / "old" / "o_r" / "data.jsonl"),
                [make_record(commit_hash="x", file="a.tf"), make_record(commit_hash="y", file="b.tf")])
    write_jsonl(str(tmp_path / "old" / "o_gone" / "data.jsonl"), [make_record(commit_hash="z")])
    write_jsonl(str(tmp_path / "new" / "o_r" / "data.jsonl"),
                [make_record(commit_hash="x", file="a.tf"), make_record(commit_hash="w", file="added.tf")])

    result = main.compare_datasets(str(tmp_path / "old"), str(tmp_path / "new"))

    assert result["totals"] == {"old": 3, "new": 2, "kept": 1, "only_new": 1, "only_old": 2}
    assert result["per_repo"]["o_r"]["only_old_examples"] == [("y", "b.tf")]
    assert result["per_repo"]["o_gone"]["new"] == 0


def test_process_repository_replay(git_repo, tmp_path):
    path, h = git_repo
    cloner = MagicMock()
    cloner.clone.return_value = path
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    info = {"repo_full_name": "owner/repo", "replay_head": h["c3"]}

    result = main.process_repository(info, main.CommitMiner(["fix"]), cloner,
                                     DataExporter(str(tmp_path / "dataset")), ["fix"],
                                     checkpoints_dir_abs=str(checkpoints), replay_cutoff="2090-01-01")

    assert result["status"] == "success"
    assert (result["replay_rev"], result["replay_source"]) == (h["c3"], "head_hash")
    with open(tmp_path / "dataset" / "owner_repo" / "data.jsonl") as f:
        assert [json.loads(line)["commit_hash"] for line in f] == [h["c2"]]


def test_process_repository_replay_unresolvable_is_mine_failed(git_repo, tmp_path):
    path, _ = git_repo
    cloner = MagicMock()
    cloner.clone.return_value = path
    (tmp_path / "checkpoints").mkdir()
    info = {"repo_full_name": "owner/repo", "replay_head": None}

    result = main.process_repository(info, main.CommitMiner(["fix"]), cloner,
                                     DataExporter(str(tmp_path / "dataset")), ["fix"],
                                     checkpoints_dir_abs=str(tmp_path / "checkpoints"), replay_cutoff="2000-01-01")

    assert result["status"] == "mine_failed"
    assert "Nenhum commit" in result["reason"]
