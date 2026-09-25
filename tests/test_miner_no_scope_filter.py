"""O minerador exporta todo .tf de um commit candidato; o escopo é decidido só no estágio 01 do LLM."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from git import Repo

from src.llm_analysis.eligibility import IN_SCOPE, TEST, EligibilityConfig, filter_commits_by_eligibility
from src.llm_analysis.preparation import aggregate_commits, find_input_files, iter_miner_records
from src.llm_analysis.utils import load_config
from src.miner import main
from src.miner.commit_miner import CommitMiner
from src.miner.data_exporter import DataExporter
from tests.conftest import AUTHOR, _commit

KEYWORDS = load_config("src/miner/config.yaml")["keywords"]  # lista do artigo, sem alteração
PATHS = [
    "tests/main.tf",
    "examples/basic/main.tf",
    "docs/example/main.tf",
    "environments/test/main.tf",
    "test.tf",
    "modules/vpc/main.tf",
]


@pytest.fixture
def scope_repo(tmp_path):
    repo = Repo.init(tmp_path / "owner_repo")
    with repo.config_writer() as cw:
        cw.set_value("user", "name", AUTHOR.name)
        cw.set_value("user", "email", AUTHOR.email)
        cw.set_value("commit", "gpgsign", "false")
    base = _commit(repo, {p: "# v1\n" for p in PATHS} | {"README.md": "v1\n"}, "initial")
    fix = _commit(repo, {p: "# v2\n" for p in PATHS} | {"README.md": "v2\n"}, "Fix wrong cidr in vpc")
    other = _commit(repo, {"modules/vpc/main.tf": "# v3\n"}, "Add outputs")
    return str(tmp_path / "owner_repo"), {"base": base, "fix": fix, "other": other}


def _export(path, tmp_path):
    cloner = MagicMock()
    cloner.clone.return_value = path
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    main.process_repository({"repo_full_name": "owner/repo"}, CommitMiner(KEYWORDS), cloner,
                            DataExporter(str(tmp_path / "dataset")), KEYWORDS,
                            checkpoints_dir_abs=str(checkpoints))
    with open(tmp_path / "dataset" / "owner_repo" / "data.jsonl") as f:
        return [json.loads(line) for line in f]


def test_keyword_list_is_the_article_list():
    assert KEYWORDS == ["fix", "bug", "issue", "resolve", "error", "failure", "incorrect"]


@pytest.mark.parametrize("path", PATHS)
def test_every_tf_path_survives_export(scope_repo, tmp_path, path):
    repo_path, h = scope_repo
    records = _export(repo_path, tmp_path)
    assert (h["fix"], path) in {(r["commit_hash"], r["file"]) for r in records}


def test_non_tf_files_and_non_keyword_commits_are_not_exported(scope_repo, tmp_path):
    repo_path, h = scope_repo
    records = _export(repo_path, tmp_path)
    assert all(r["file"].endswith(".tf") for r in records)                 # README.md fora
    assert {r["commit_hash"] for r in records} == {h["fix"]}              # "initial"/"Add outputs" fora
    assert sorted(r["file"] for r in records) == sorted(PATHS)


def test_miner_no_longer_imports_a_scope_filter():
    source = Path(main.__file__).read_text()
    assert "is_real_infra_file" not in source and "src.llm_analysis" not in source


def test_end_to_end_miner_preserves_eligibility_removes(tmp_path):
    """Minerador guarda modules/ e tests/; o estágio 01 mantém o commit e só envia modules/."""
    repo = Repo.init(tmp_path / "owner_repo")
    with repo.config_writer() as cw:
        cw.set_value("commit", "gpgsign", "false")
    files = {"modules/vpc/main.tf": 'cidr_block = "10.0.0.0/16"\n', "tests/vpc/main.tf": "a = 1\n"}
    _commit(repo, files, "initial")
    fix = _commit(repo, {"modules/vpc/main.tf": 'cidr_block = "10.0.0.0/24"\n', "tests/vpc/main.tf": "a = 2\n"},
                  "fix: subnet mask")

    records = _export(str(tmp_path / "owner_repo"), tmp_path)
    assert sorted(r["file"] for r in records) == ["modules/vpc/main.tf", "tests/vpc/main.tf"]

    commits = aggregate_commits(iter_miner_records(find_input_files(tmp_path / "dataset" / "**" / "*.jsonl")))
    result = filter_commits_by_eligibility(commits, EligibilityConfig.from_config(load_config()["eligibility"]))
    [decision] = result.decisions
    assert (decision.commit.commit_hash, decision.state) == (fix, IN_SCOPE)
    assert decision.prepared.files == ["modules/vpc/main.tf"]
    assert [d.file for d in decision.prepared.diffs] == ["modules/vpc/main.tf"]
    assert "a = 2" not in json.dumps(decision.prepared.to_dict()["diffs"])
    assert decision.prepared.excluded_files == [{"file": "tests/vpc/main.tf", "scope": TEST}]


def test_replay_plan_prefers_the_recorded_replay_rev(tmp_path):
    old = tmp_path / "old"
    (old / "logs").mkdir(parents=True)
    (old / "logs" / "summary.json").write_text(json.dumps({"processed_repositories_summary": [
        {"repo_full_name": "a/r", "stars": 1, "forks": 0, "commits": 1, "replay_rev": "REV_A"},
        {"repo_full_name": "b/r", "stars": 1, "forks": 0, "commits": 1},
    ]}))
    repos, _ = main.load_replay_plan(str(old), cutoff="2025-05-31T00:00:00")
    assert [r["replay_head"] for r in repos] == ["REV_A", None]


def test_dataset_counts(tmp_path):
    from tests.helpers import make_record, write_jsonl
    write_jsonl(str(tmp_path / "o_r" / "data.jsonl"), [make_record(file="a.tf"), make_record(file="b.tf"),
                                                      make_record(commit_hash="c2")])
    assert main.dataset_counts(str(tmp_path)) == {"raw_record_count": 3, "unique_commit_count": 2,
                                                  "repositories_with_records": 1}
