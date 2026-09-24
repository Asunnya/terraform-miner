import logging
import os

import pytest
from git import Commit, GitCommandError, Repo

from src.miner.commit_miner import CommitMiner, DiffUnavailableError, _strip_patch_header

EXPECTED_KEYS = {"repo", "commit_hash", "author", "date", "message", "file", "patch", "head_hash"}


def test_mines_only_tf_files_in_keyword_commits(git_repo):
    path, h = git_repo
    entries = CommitMiner(["fix"]).mine(path)

    assert {(e["commit_hash"], e["file"]) for e in entries} == {
        (h["c2"], "main.tf"),
        (h["c4"], "examples/demo.tf"),
    }


def test_keywords_are_case_insensitive(git_repo):
    path, h = git_repo
    entries = CommitMiner(["FIX"]).mine(path)
    # c3 ("Fix bug in docs") casa com a keyword, mas só mexe em README.md
    assert {e["commit_hash"] for e in entries} == {h["c2"], h["c4"]}


def test_entry_fields(git_repo):
    path, h = git_repo
    entries = CommitMiner(["wrong cidr"]).mine(path)

    assert len(entries) == 1
    e = entries[0]
    assert set(e) == EXPECTED_KEYS
    assert e["repo"] == os.path.basename(path)
    assert e["commit_hash"] == h["c2"]
    assert e["author"] == "Test Author"
    assert e["message"] == "fix: wrong cidr"
    assert e["head_hash"] == h["c5"]
    assert "10.0.0.0/16" in e["patch"]


def test_skips_seen_hashes(git_repo):
    path, h = git_repo
    entries = CommitMiner(["fix"]).mine(path, seen_hashes={h["c2"]})
    assert {e["commit_hash"] for e in entries} == {h["c4"]}


def test_last_head_limits_to_newer_commits(git_repo):
    path, h = git_repo
    # Depois de c3 só vêm c4 (fix) e c5 (sem keyword)
    entries = CommitMiner(["fix"]).mine(path, last_head=h["c3"])
    assert {e["commit_hash"] for e in entries} == {h["c4"]}

    assert CommitMiner(["fix"]).mine(path, last_head=h["c5"]) == []


def test_invalid_last_head_falls_back_to_all_commits(git_repo, caplog):
    path, h = git_repo
    with caplog.at_level(logging.WARNING):
        entries = CommitMiner(["fix"]).mine(path, last_head="0" * 40)
    assert {e["commit_hash"] for e in entries} == {h["c2"], h["c4"]}
    assert "falling back to all commits" in caplog.text


def test_missing_origin_logs_warning_and_continues(git_repo, caplog):
    path, _ = git_repo
    with caplog.at_level(logging.WARNING):
        entries = CommitMiner(["fix"]).mine(path)
    assert entries
    assert "Failed to fetch latest commits" in caplog.text


def test_root_commit_is_mined(git_repo):
    path, h = git_repo
    entries = CommitMiner(["initial"]).mine(path)
    assert [(e["commit_hash"], e["file"]) for e in entries] == [(h["c1"], "main.tf")]
    assert "10.0.0.0/8" in entries[0]["patch"]


def test_patch_direction_is_parent_to_commit(git_repo):
    path, _ = git_repo
    [entry] = CommitMiner(["wrong cidr"]).mine(path)
    assert '+  cidr_block = "10.0.0.0/16"' in entry["patch"]
    assert '-  cidr_block = "10.0.0.0/8"' in entry["patch"]


def test_added_file_patch_only_has_additions(git_repo):
    path, h = git_repo
    [entry] = CommitMiner(["fix: example"]).mine(path)
    assert entry["file"] == "examples/demo.tf"
    assert '+module "demo" {' in entry["patch"]
    assert "\n-" not in entry["patch"]


def test_deleted_tf_file_is_mined(git_repo):
    path, _ = git_repo
    repo = Repo(path)
    repo.index.remove(["variables.tf"], working_tree=True)
    sha = repo.index.commit("fix: drop unused variable").hexsha

    [entry] = CommitMiner(["unused variable"]).mine(path)
    assert (entry["commit_hash"], entry["file"]) == (sha, "variables.tf")
    assert '-variable "region" {}' in entry["patch"]


def test_no_keywords_mines_nothing(git_repo):
    path, _ = git_repo
    assert CommitMiner().mine(path) == []


def test_repo_without_changes_to_tf(tmp_path):
    repo = Repo.init(tmp_path / "r")
    (tmp_path / "r" / "a.py").write_text("x = 1\n")
    repo.index.add(["a.py"])
    repo.index.commit("fix: python only")
    assert CommitMiner(["fix"]).mine(repo.working_dir) == []


# --- fallback (git diff-tree via subprocess) ---

def _force_fallback(monkeypatch):
    def fail(*args, **kwargs):
        raise GitCommandError("diff", 128, stderr="forced failure")
    monkeypatch.setattr(Commit, "diff", fail)


def _add_rename_and_delete(path):
    repo = Repo(path)
    repo.index.move(["main.tf", "network.tf"])
    rename = repo.index.commit("fix: rename main").hexsha
    repo.index.remove(["variables.tf"], working_tree=True)
    delete = repo.index.commit("fix: drop variables").hexsha
    return rename, delete


def test_fallback_matches_standard_diff(git_repo, monkeypatch):
    path, _ = git_repo
    _add_rename_and_delete(path)
    keywords = ["fix", "initial"]

    expected = CommitMiner(keywords).mine(path)
    _force_fallback(monkeypatch)
    actual = CommitMiner(keywords).mine(path)

    key = lambda e: (e["commit_hash"], e["file"])
    assert sorted(actual, key=key) == sorted(expected, key=key)
    # Garante que a comparação cobre raiz, modificação, criação, rename e remoção
    assert {e["file"] for e in expected} == {"main.tf", "examples/demo.tf", "network.tf", "variables.tf"}
    assert len(expected) == 5


def test_fallback_is_used_and_logged(git_repo, monkeypatch, caplog):
    path, h = git_repo
    _force_fallback(monkeypatch)
    with caplog.at_level(logging.INFO):
        [entry] = CommitMiner(["wrong cidr"]).mine(path)
    assert "Trying fallback diff method" in caplog.text
    assert entry["patch"].startswith("@@")
    assert '+  cidr_block = "10.0.0.0/16"' in entry["patch"]


def test_fallback_failure_raises(git_repo, monkeypatch, caplog):
    # Ex.: clone parcial sem internet. O repo inteiro deve falhar (sem checkpoint) em vez de perder o commit.
    path, _ = git_repo
    _force_fallback(monkeypatch)
    monkeypatch.setattr(CommitMiner, "_git", staticmethod(lambda *a: (_ for _ in ()).throw(OSError("no git"))))
    with caplog.at_level(logging.WARNING), pytest.raises(DiffUnavailableError, match="no git"):
        CommitMiner(["fix"]).mine(path)
    assert "Fallback diff method also failed" in caplog.text


@pytest.mark.parametrize("patch, expected", [
    (b"diff --git a/x b/x\nindex 1..2 100644\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n", b"@@ -1 +1 @@\n-a\n+b\n"),
    (b"diff --git a/x b/x\nindex 1..2\nBinary files a/x and b/x differ\n", b"Binary files a/x and b/x differ\n"),
    (b"diff --git a/x b/y\nsimilarity index 100%\nrename from x\nrename to y\n", b""),
    (b"", b""),
])
def test_strip_patch_header(patch, expected):
    assert _strip_patch_header(patch) == expected


def test_until_rev_freezes_history(git_repo):
    path, h = git_repo
    entries = CommitMiner(["fix"]).mine(path, until_rev=h["c3"])
    # c4 vem depois de c3 e fica de fora; head_hash passa a ser o commit de corte
    assert {e["commit_hash"] for e in entries} == {h["c2"]}
    assert {e["head_hash"] for e in entries} == {h["c3"]}


def test_until_rev_with_last_head(git_repo):
    path, h = git_repo
    entries = CommitMiner(["fix"]).mine(path, last_head=h["c1"], until_rev=h["c3"])
    assert {e["commit_hash"] for e in entries} == {h["c2"]}


def test_unknown_until_rev_raises(git_repo):
    path, _ = git_repo
    with pytest.raises(ValueError, match="not found"):
        CommitMiner(["fix"]).mine(path, until_rev="0" * 40)
