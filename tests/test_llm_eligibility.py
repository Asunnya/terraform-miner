"""Filtro de escopo do estágio 01 (casos da especificação)."""
from pathlib import Path

import pytest

from src.llm_analysis.eligibility import (
    DOCUMENTATION,
    EXAMPLE,
    GENERATED,
    IN_SCOPE,
    INFRASTRUCTURE,
    INSUFFICIENT_DATA,
    OUT_OF_SCOPE,
    REPOSITORY_METADATA,
    TEST,
    VENDOR,
    EligibilityConfig,
    assess_commit,
    file_scope,
    filter_commits_by_eligibility,
)
from src.llm_analysis.preparation import aggregate_commits, commit_content_hash, sample_commits
from src.llm_analysis.schemas import FileDiff
from src.llm_analysis.utils import load_config
from tests.helpers import make_record

CONFIG = EligibilityConfig.from_config(load_config()["eligibility"])
PATCH = "@@ -1 +1 @@\n-a\n+b\n"


def commit_of(files, message="fix: something", patch=PATCH, commit_hash="c1"):
    """Commit agregado; `files` = {caminho: patch} ou lista de caminhos (todos com PATCH distinto)."""
    if not isinstance(files, dict):
        files = {f: f"@@ -1 +1 @@\n-{f}\n+{f}x\n" if patch is PATCH else patch for f in files}
    records = [make_record(commit_hash=commit_hash, file=f, patch=p, message=message) for f, p in files.items()]
    [commit] = aggregate_commits((Path("data.jsonl"), r) for r in records)
    return commit


def assess(files, **kwargs):
    return assess_commit(commit_of(files, **kwargs), CONFIG)


@pytest.mark.parametrize("path, scope", [
    ("main.tf", INFRASTRUCTURE),
    ("modules/vpc/main.tf", INFRASTRUCTURE),
    ("environments/dev/main.tf", INFRASTRUCTURE),
    ("environments/test/main.tf", INFRASTRUCTURE),     # "test" genérico não é excluído
    ("tests/vpc/main.tf", TEST),
    ("tests/fixtures/vpc/main.tf", TEST),
    ("testdata/vpc/main.tf", TEST),
    ("modules/vpc/tests/main.tf", TEST),
    ("examples/basic/main.tf", EXAMPLE),
    ("example/basic/main.tf", EXAMPLE),
    ("docs/example/main.tf", DOCUMENTATION),           # o segmento mais externo decide
    ("vendor/module/main.tf", VENDOR),
    (".terraform/modules/x/main.tf", GENERATED),
    (".github/workflows/terraform/main.tf", REPOSITORY_METADATA),
    ("Examples/basic/main.tf", EXAMPLE),               # sem diferenciar maiúsculas
    ("my-tests/main.tf", INFRASTRUCTURE),              # segmento inteiro, não substring
    ("latest/main.tf", INFRASTRUCTURE),
    ("examples.tf", INFRASTRUCTURE),                   # só diretórios contam, não o nome do arquivo
])
def test_file_scope_uses_whole_path_segments(path, scope):
    assert file_scope(path, CONFIG) == scope


@pytest.mark.parametrize("files", [["main.tf"], ["modules/vpc/main.tf"], ["environments/dev/main.tf"],
                                   ["environments/test/main.tf"]])
def test_infrastructure_commits_are_in_scope(files):                     # casos 1-4
    decision = assess(files)
    assert (decision.state, decision.reason) == (IN_SCOPE, "eligible")
    assert decision.prepared.files == files


@pytest.mark.parametrize("files, reason", [
    (["tests/main.tf"], "only_test_files"),                               # 5
    (["testdata/main.tf"], "only_test_files"),                            # 6
    (["tests/fixtures/main.tf"], "only_test_files"),                      # 7
    (["examples/basic/main.tf"], "only_example_files"),                   # 8
    (["docs/example/main.tf"], "only_documentation_files"),               # 9
    (["vendor/module/main.tf"], "only_vendor_files"),                     # 10
    ([".terraform/modules/x/main.tf"], "only_generated_files"),           # 11
    ([".github/actions/x/main.tf"], "only_repository_metadata_files"),
    (["tests/main.tf", "examples/main.tf"], "only_excluded_files"),
])
def test_excluded_only_commits_are_out_of_scope(files, reason):
    decision = assess(files)
    assert (decision.state, decision.reason) == (OUT_OF_SCOPE, reason)
    assert decision.prepared is None
    assert decision.audit_record()["excluded_files"] == [{"file": f, "scope": file_scope(f, CONFIG)}
                                                         for f in sorted(files)]


def test_mixed_implementation_and_tests_keeps_only_implementation():     # 12
    decision = assess(["modules/vpc/main.tf", "tests/vpc/main.tf"])
    prepared = decision.prepared
    assert decision.state == IN_SCOPE and decision.mixed
    assert prepared.files == ["modules/vpc/main.tf"]
    assert [d.file for d in prepared.diffs] == ["modules/vpc/main.tf"]
    assert all("tests/" not in d.patch for d in prepared.diffs)
    assert prepared.excluded_files == [{"file": "tests/vpc/main.tf", "scope": TEST}]


def test_mixed_implementation_examples_docs_keeps_only_main():           # 13
    prepared = assess(["main.tf", "examples/main.tf", "docs/main.tf"]).prepared
    assert prepared.files == ["main.tf"]
    assert [d.file for d in prepared.diffs] == ["main.tf"]
    assert {e["scope"] for e in prepared.excluded_files} == {EXAMPLE, DOCUMENTATION}


def test_generic_merge_is_out_of_scope():                                # 14
    decision = assess(["modules/vpc/main.tf"], message="Merge pull request #123 from foo/bar\n\nFix subnet")
    assert (decision.state, decision.reason) == (OUT_OF_SCOPE, "generic_merge")


@pytest.mark.parametrize("message", ["Merge branch 'main' into feature", "Merge remote-tracking branch 'origin/x'",
                                     "  Merged in feature/x (pull request #7)", "merge pull request #9 from a/b"])
def test_merge_patterns(message):
    assert assess(["main.tf"], message=message).reason == "generic_merge"


@pytest.mark.parametrize("message", ["Fix subnet merge behavior", "fix: merge tags correctly",
                                     "Revert \"Merge pull request #1 from a/b\""])
def test_the_word_merge_alone_is_not_a_merge_commit(message):            # 15
    assert assess(["main.tf"], message=message).state == IN_SCOPE


def test_empty_message_does_not_affect_eligibility():                    # 16
    assert assess(["modules/vpc/main.tf"], message="").state == IN_SCOPE


def test_in_scope_files_without_patch_are_insufficient_data():           # 17
    decision = assess({"main.tf": "", "variables.tf": None})
    assert (decision.state, decision.reason) == (INSUFFICIENT_DATA, "no_in_scope_patch")
    assert decision.in_scope_files == ["main.tf", "variables.tf"]


def test_only_excluded_file_has_a_patch_is_insufficient_data():
    decision = assess({"main.tf": "", "tests/main.tf": PATCH})
    assert (decision.state, decision.reason) == (INSUFFICIENT_DATA, "no_in_scope_patch")


def test_in_scope_file_without_patch_stays_in_files_without_patch():     # 18
    prepared = assess({"main.tf": PATCH, "variables.tf": "", "tests/x.tf": ""}).prepared
    assert prepared.files == ["main.tf", "variables.tf"]
    assert prepared.files_without_patch == ["variables.tf"]


def test_shared_patch_with_excluded_canonical_path_is_reassigned():      # 19
    commit = commit_of(["main.tf"])
    commit.files = ["modules/vpc/main.tf", "tests/main.tf"]
    commit.diffs = [FileDiff(file="tests/main.tf", patch=PATCH, also_in=["modules/vpc/main.tf"])]
    [diff] = assess_commit(commit, CONFIG).prepared.diffs
    assert (diff.file, diff.patch, diff.also_in) == ("modules/vpc/main.tf", PATCH, [])


def test_shared_patch_drops_excluded_also_in_paths():                    # 20
    commit = commit_of(["main.tf"])
    commit.files = ["b/main.tf", "modules/vpc/main.tf", "tests/main.tf"]
    commit.diffs = [FileDiff(file="modules/vpc/main.tf", patch=PATCH, also_in=["b/main.tf", "tests/main.tf"])]
    [diff] = assess_commit(commit, CONFIG).prepared.diffs
    assert (diff.file, diff.also_in) == ("b/main.tf", ["modules/vpc/main.tf"])  # 1º caminho em ordem


def test_filtering_is_deterministic():                                   # 21
    files = ["z/main.tf", "tests/a.tf", "a/main.tf", "examples/b.tf", "m/main.tf"]
    first = assess(files).prepared
    second = assess(list(reversed(files))).prepared
    assert first.to_dict() == second.to_dict()
    assert first.files == ["a/main.tf", "m/main.tf", "z/main.tf"]
    assert [d.file for d in first.diffs] == first.files


def test_content_hash_is_recomputed_from_the_filtered_commit():
    commit = commit_of(["main.tf", "tests/main.tf"])
    prepared = assess_commit(commit, CONFIG).prepared
    assert prepared.content_sha256 == commit_content_hash(prepared) != commit.content_sha256
    assert assess(["main.tf"]).prepared.content_sha256 == commit_of(["main.tf"]).content_sha256  # nada removido


def test_sampling_draws_from_the_eligible_population():                  # 22
    commits = [commit_of(["main.tf"], commit_hash=f"in{i:02d}") for i in range(10)]
    commits += [commit_of(["tests/main.tf"], commit_hash=f"out{i:02d}") for i in range(10)]
    result = filter_commits_by_eligibility(commits, CONFIG)
    sample = sample_commits(result.eligible, 5, seed=1)
    assert len(sample) == 5 and all(c.commit_hash.startswith("in") for c in sample)


def test_stats_and_invariant():
    commits = [commit_of(["main.tf"], commit_hash="a"), commit_of(["main.tf", "tests/x.tf"], commit_hash="b"),
               commit_of(["tests/x.tf"], commit_hash="c"), commit_of({"main.tf": ""}, commit_hash="d"),
               commit_of(["main.tf"], commit_hash="e", message="Merge branch 'x'")]
    stats = filter_commits_by_eligibility(commits, CONFIG).stats()
    assert (stats["commits_before_eligibility"], stats["commits_in_scope"], stats["commits_out_of_scope"],
            stats["commits_insufficient_data"]) == (5, 2, 2, 1)
    assert stats["by_reason"]["only_test_files"] == 1 and stats["by_reason"]["generic_merge"] == 1
    assert stats["by_reason"]["no_in_scope_patch"] == 1
    assert stats["mixed_scope_commits"] == 1
    assert stats["retained_in_scope_file_count"] == 2


def test_disabled_filter_keeps_every_commit_unchanged():
    config = EligibilityConfig.from_config({"enabled": False})
    commit = commit_of(["tests/main.tf"])
    assert filter_commits_by_eligibility([commit], config).eligible == [commit]


def test_config_rejects_unknown_scope_and_duplicate_segments():
    with pytest.raises(ValueError, match="unknown scope"):
        EligibilityConfig.from_config({"excluded_scope_segments": {"tests_dir": ["tests"]}})
    with pytest.raises(ValueError, match="two scopes"):
        EligibilityConfig.from_config({"excluded_scope_segments": {"test": ["tests"], "example": ["Tests"]}})
