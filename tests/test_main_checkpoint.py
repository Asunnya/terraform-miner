import json
from unittest.mock import MagicMock

import pytest

# Importar main.py cria ../data/{repos,logs,dataset,checkpoints} (mkdir exist_ok) e imprime PROJECT_ROOT.
from src.miner import main
from src.miner.data_exporter import DataExporter
from tests.helpers import make_record

EMPTY_CHECKPOINT = {
    "processed_commits": [],
    "last_head": None,
    "last_updated": None,
    "total_commits_processed": 0,
}


# --- load_checkpoint / save_checkpoint ---

def test_load_checkpoint_missing(tmp_path):
    assert main.load_checkpoint("owner/repo", str(tmp_path)) == EMPTY_CHECKPOINT


def test_load_checkpoint_corrupt(tmp_path):
    (tmp_path / "owner_repo.json").write_text("{not json")
    assert main.load_checkpoint("owner/repo", str(tmp_path)) == EMPTY_CHECKPOINT


def test_checkpoint_round_trip(tmp_path):
    data = {"processed_commits": ["a", "b"], "last_head": "h", "total_commits_processed": 2}
    main.save_checkpoint("owner/repo", data, str(tmp_path))

    assert (tmp_path / "owner_repo.json").exists()
    loaded = main.load_checkpoint("owner/repo", str(tmp_path))
    assert loaded["processed_commits"] == ["a", "b"]
    assert loaded["last_head"] == "h"
    assert loaded["last_updated"]  # preenchido pelo save


# --- process_repository ---

REPO_INFO = {"repo_full_name": "owner/repo", "stars": 1, "forks": 0, "commits": 1}


@pytest.fixture
def env(tmp_path):
    cloner = MagicMock()
    cloner.clone.return_value = str(tmp_path / "clone")
    miner = MagicMock()
    exporter = DataExporter(str(tmp_path / "dataset"))
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()

    def run(**kwargs):
        return main.process_repository(REPO_INFO, miner, cloner, exporter, ["fix"],
                                       checkpoints_dir_abs=str(checkpoints), **kwargs)

    return {"cloner": cloner, "miner": miner, "run": run, "tmp": tmp_path, "checkpoints": checkpoints}


def test_process_repository_clone_failed(env):
    env["cloner"].clone.side_effect = RuntimeError("no network")
    result = env["run"]()
    assert result["status"] == "clone_failed"
    assert "no network" in result["reason"]
    env["miner"].mine.assert_not_called()


def test_process_repository_exports_every_tf_record(env):
    # Sem filtro de escopo no minerador: examples/ também é exportado (o escopo é decidido no llm_analysis).
    env["miner"].mine.return_value = [
        make_record(commit_hash="a", file="main.tf", head_hash="HEAD1"),
        make_record(commit_hash="b", file="examples/demo.tf", head_hash="HEAD1"),
    ]
    result = env["run"]()

    assert result["status"] == "success"
    assert result["mined_commits"] == 2
    assert result["stars"] == 1  # mantém os campos de repo_info

    with open(env["tmp"] / "dataset" / "owner_repo" / "data.jsonl") as f:
        assert [json.loads(line)["file"] for line in f] == ["main.tf", "examples/demo.tf"]

    checkpoint = json.loads((env["checkpoints"] / "owner_repo.json").read_text())
    assert sorted(checkpoint["processed_commits"]) == ["a", "b"]
    assert checkpoint["last_head"] == "HEAD1"


def test_process_repository_passes_clone_options_and_checkpoint(env):
    main.save_checkpoint("owner/repo", {"processed_commits": ["old"], "last_head": "H0"}, str(env["checkpoints"]))
    env["miner"].mine.return_value = []

    env["run"](full_clone=True, clone_depth=5)

    env["cloner"].clone.assert_called_once_with("owner/repo", depth=5, partial_clone=False)
    _, kwargs = env["miner"].mine.call_args
    assert kwargs == {"seen_hashes": {"old"}, "last_head": "H0", "until_rev": None}


def test_process_repository_no_new_commits(env):
    env["miner"].mine.return_value = []
    result = env["run"]()
    assert result["status"] == "no_new_commits"
    assert result["mined_commits"] == 0


def test_process_repository_mine_failed(env):
    env["miner"].mine.side_effect = RuntimeError("corrupt repo")
    result = env["run"]()
    assert result["status"] == "mine_failed"
    assert "corrupt repo" in result["reason"]
