import json

import pandas as pd

from src.miner.data_exporter import DataExporter
from tests.helpers import make_record


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_export_writes_jsonl_and_csv(tmp_path):
    records = [make_record(commit_hash="a"), make_record(commit_hash="b", file="vars.tf")]
    DataExporter(str(tmp_path)).export("owner/repo", records)

    repo_dir = tmp_path / "owner_repo"
    assert _read_jsonl(repo_dir / "data.jsonl") == records
    df = pd.read_csv(repo_dir / "data.csv")
    assert list(df["commit_hash"]) == ["a", "b"]


def test_export_empty_entries_creates_nothing(tmp_path):
    DataExporter(str(tmp_path)).export("owner/repo", [])
    assert not (tmp_path / "owner_repo").exists()


def test_append_deduplicates_by_commit_and_file(tmp_path):
    exporter = DataExporter(str(tmp_path))
    exporter.export("owner/repo", [make_record(commit_hash="a"), make_record(commit_hash="b")])
    exporter.export(
        "owner/repo",
        [
            make_record(commit_hash="a"),                   # duplicado
            make_record(commit_hash="a", file="other.tf"),  # mesmo commit, outro arquivo
            make_record(commit_hash="c"),
        ],
        append=True,
    )

    rows = _read_jsonl(tmp_path / "owner_repo" / "data.jsonl")
    assert [(r["commit_hash"], r["file"]) for r in rows] == [
        ("a", "main.tf"), ("b", "main.tf"), ("a", "other.tf"), ("c", "main.tf"),
    ]


def test_without_append_overwrites(tmp_path):
    exporter = DataExporter(str(tmp_path))
    exporter.export("owner/repo", [make_record(commit_hash="a")])
    exporter.export("owner/repo", [make_record(commit_hash="b")])

    rows = _read_jsonl(tmp_path / "owner_repo" / "data.jsonl")
    assert [r["commit_hash"] for r in rows] == ["b"]


def test_append_with_corrupt_jsonl_overwrites(tmp_path):
    repo_dir = tmp_path / "owner_repo"
    repo_dir.mkdir()
    (repo_dir / "data.jsonl").write_text("{not json\n")

    DataExporter(str(tmp_path)).export("owner/repo", [make_record(commit_hash="z")], append=True)

    rows = _read_jsonl(repo_dir / "data.jsonl")
    assert [r["commit_hash"] for r in rows] == ["z"]
