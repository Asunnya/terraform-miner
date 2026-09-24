from pathlib import Path

from src.llm_analysis.preparation import (
    PreparationStats,
    aggregate_commits,
    find_input_files,
    iter_miner_records,
    keyword_match_fraction,
    sample_commits,
)
from tests.helpers import make_record, write_jsonl

SRC = Path("data.jsonl")


def _aggregate(records):
    stats = PreparationStats()
    return aggregate_commits(((SRC, r) for r in records), stats), stats


def test_groups_rows_of_the_same_commit():
    commits, stats = _aggregate([
        make_record(commit_hash="c1", file="variables.tf", patch="@@ -1 +1 @@\n-a\n+b\n"),
        make_record(commit_hash="c2", file="main.tf"),
        make_record(commit_hash="c1", file="main.tf", patch="@@ -2 +2 @@\n-x\n+y\n"),
    ])

    assert [c.commit_hash for c in commits] == ["c1", "c2"]
    c1 = commits[0]
    assert c1.files == ["main.tf", "variables.tf"]
    assert [d.file for d in c1.diffs] == ["main.tf", "variables.tf"]  # ordenado por caminho
    assert c1.n_records == 2
    assert (c1.repo, c1.author, c1.message, c1.head_hash) == ("owner_repo", "dev", "fix: something", "fff999")
    assert stats.commits == 2 and stats.records_read == 3


def test_same_hash_in_different_repos_is_not_merged():
    commits, _ = _aggregate([make_record(repo="a"), make_record(repo="b")])
    assert [(c.repo, c.commit_hash) for c in commits] == [("a", "aaa111"), ("b", "aaa111")]


def test_duplicate_patches_removed_and_recorded():
    patch = "@@ -1 +1 @@\n-a\n+b\n"
    commits, stats = _aggregate([
        make_record(file="modules/b/main.tf", patch=patch),
        make_record(file="modules/a/main.tf", patch=patch),
        make_record(file="modules/a/main.tf", patch=patch),  # registro repetido
        make_record(file="other.tf", patch="@@ -5 +5 @@\n-q\n+r\n"),
    ])

    commit = commits[0]
    assert len(commit.diffs) == 2
    shared = commit.diffs[0]
    assert (shared.file, shared.also_in) == ("modules/a/main.tf", ["modules/b/main.tf"])
    assert commit.duplicate_patches_removed == 1
    assert stats.duplicate_records == 1
    assert commit.files == ["modules/a/main.tf", "modules/b/main.tf", "other.tf"]


def test_empty_patches_keep_the_file_but_no_diff():
    commits, stats = _aggregate([
        make_record(file="main.tf", patch=""),
        make_record(file="vars.tf", patch=None),
        make_record(file="ok.tf", patch="@@ -1 +1 @@\n-a\n+b\n"),
        make_record(commit_hash="only_empty", file="x.tf", patch="  \n"),
    ])

    by_hash = {c.commit_hash: c for c in commits}
    commit = by_hash["aaa111"]
    assert commit.files == ["main.tf", "ok.tf", "vars.tf"]
    assert [d.file for d in commit.diffs] == ["ok.tf"]
    assert commit.files_without_patch == ["main.tf", "vars.tf"]
    assert by_hash["only_empty"].diffs == []
    assert stats.empty_patches == 3
    assert stats.commits_without_any_patch == 1


def test_skips_non_tf_and_keyless_records():
    commits, stats = _aggregate([
        make_record(file="README.md"),
        make_record(commit_hash=None),
        make_record(file="MAIN.TF"),
    ])
    assert len(commits) == 1
    assert stats.records_skipped_non_tf == 1
    assert stats.records_skipped_missing_key == 1


def test_aggregation_is_deterministic_regardless_of_record_order():
    records = [
        make_record(commit_hash="c2", file="b.tf", patch="@@ b\n+1\n"),
        make_record(commit_hash="c1", file="z.tf", patch="@@ z\n+1\n"),
        make_record(commit_hash="c1", file="a.tf", patch="@@ a\n+1\n"),
    ]
    first, _ = _aggregate(records)
    second, _ = _aggregate(list(reversed(records)))
    assert [c.to_dict() for c in first] == [c.to_dict() for c in second]
    assert first[0].content_sha256 == second[0].content_sha256


def test_content_hash_changes_with_patch():
    a, _ = _aggregate([make_record(patch="@@ -1 +1 @@\n-a\n+b\n")])
    b, _ = _aggregate([make_record(patch="@@ -1 +1 @@\n-b\n+a\n")])
    assert a[0].content_sha256 != b[0].content_sha256


def test_conflicting_commit_fields_are_counted():
    _, stats = _aggregate([make_record(file="a.tf", message="m1"), make_record(file="b.tf", message="m2")])
    assert stats.field_conflicts["message"] == 1


def test_reads_multiple_jsonl_files(tmp_path):
    write_jsonl(str(tmp_path / "r1" / "data.jsonl"), [make_record(repo="r1", file="a.tf")])
    write_jsonl(str(tmp_path / "r2" / "data.jsonl"), [make_record(repo="r2", file="a.tf")], extra_lines=["{bad"])
    files = find_input_files(tmp_path / "**" / "*.jsonl")

    commits = aggregate_commits(iter_miner_records(files))
    assert [c.repo for c in commits] == ["r1", "r2"]
    assert commits[0].source_files == [str(files[0])]


def test_sampling_is_reproducible():
    commits, _ = _aggregate([make_record(commit_hash=f"c{i:02d}") for i in range(30)])
    s1 = sample_commits(commits, 10, seed=7)
    s2 = sample_commits(commits, 10, seed=7)
    assert [c.commit_hash for c in s1] == [c.commit_hash for c in s2]
    assert len(s1) == 10 and s1 == sorted(s1, key=lambda c: c.commit_hash)
    assert sample_commits(commits, None, seed=7) == commits


def test_keyword_match_fraction():
    commits, _ = _aggregate([
        make_record(commit_hash="a", message="Fix cidr"),
        make_record(commit_hash="b", message="add module"),
    ])
    assert keyword_match_fraction(commits, ["fix", "bug"]) == 0.5
    assert keyword_match_fraction(commits, []) is None
