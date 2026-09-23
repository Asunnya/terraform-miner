import datetime
import json
import os

import pandas as pd
import pytest

from src.miner import sample_preparer as sp
from tests.helpers import make_record, write_jsonl


# --- clean_commit_message_advanced ---

@pytest.mark.parametrize("msg, expected", [
    ("Fix VPC CIDR\n\nlong body", "fix vpc cidr"),
    ("fix: issue #123 in module", "fix: issue in module"),
    ("GH-45 fix thing", "fix thing"),
    ("see https://github.com/x/y for details", "see for details"),
    ("  lots   of\tspaces  ", "lots of spaces"),
    ("", None),
    (None, None),
    ("#12", None),
])
def test_clean_commit_message(msg, expected):
    assert sp.clean_commit_message_advanced(msg) == expected


def test_clean_commit_message_truncates_at_word_boundary():
    msg = "word " * 10  # 50 caracteres
    assert sp.clean_commit_message_advanced(msg, max_len=12) == "word word"


# --- get_relevant_file_and_patch_original ---

def test_relevant_file_prefers_tf():
    mods = [{"filename": "main.go", "diff": "go"}, {"filename": "main.tf", "diff": "tf"}]
    assert sp.get_relevant_file_and_patch_original(mods) == ("main.tf", "tf")


def test_relevant_file_falls_back_to_first():
    mods = [{"filename": "README.md", "diff": "a"}, {"filename": "x.txt", "diff": "b"}]
    assert sp.get_relevant_file_and_patch_original(mods) == ("README.md", "a")


def test_relevant_file_empty_diff():
    assert sp.get_relevant_file_and_patch_original([{"filename": "main.tf", "diff": ""}]) == (None, "")
    assert sp.get_relevant_file_and_patch_original([]) == (None, "")


def test_patch_excerpt_is_truncated():
    diff = "\n".join(f"line{i}" for i in range(50))
    _, excerpt = sp.get_relevant_file_and_patch_original([{"filename": "main.tf", "diff": diff}])
    assert excerpt.splitlines() == [f"line{i}" for i in range(sp.PATCH_EXCERPT_LINES)]


# --- create_all_commits_csv ---

@pytest.fixture
def dataset(tmp_path):
    """Dataset com 2 repos; 'a1' tem dois arquivos, 'm1' é merge."""
    ds = tmp_path / "dataset"
    write_jsonl(str(ds / "owner_r1" / "data.jsonl"), [
        make_record(repo="r1", commit_hash="a1", file="main.go", patch="go patch",
                    message="Fix #10 bad cidr", date="2024-01-10T00:00:00+00:00"),
        make_record(repo="r1", commit_hash="a1", file="main.tf", patch="tf patch",
                    message="Fix #10 bad cidr", date="2024-01-10T00:00:00+00:00"),
        make_record(repo="r1", commit_hash="a2", message="fix subnet mask", date="2024-02-20T00:00:00+00:00"),
    ], extra_lines=["{broken json", json.dumps({"repo": "r1"})])
    write_jsonl(str(ds / "owner_r2" / "data.jsonl"), [
        make_record(repo="r2", commit_hash="m1", message="Merge pull request #5 from x/y",
                    date="2024-03-01T00:00:00+00:00"),
        make_record(repo="r2", commit_hash="b1", message="fix subnet route", date="2024-03-05T00:00:00+00:00"),
    ])
    return ds


@pytest.fixture
def all_commits_csv(dataset, tmp_path):
    out = tmp_path / "all_commits.csv"
    assert sp.create_all_commits_csv(str(dataset / "*" / "data.jsonl"), str(out))
    return out


def test_create_all_commits_csv(all_commits_csv):
    df = pd.read_csv(all_commits_csv).set_index("commit_hash")

    assert sorted(df.index) == ["a1", "a2", "b1", "m1"]
    assert df.loc["a1", "relevant_file"] == "main.tf"
    assert df.loc["a1", "patch_excerpt"] == "tf patch"
    assert df.loc["a1", "msg_clean"] == "fix bad cidr"
    assert bool(df.loc["m1", "is_merge"]) is True
    assert bool(df.loc["a1", "is_merge"]) is False


def test_create_all_commits_csv_no_match(tmp_path):
    assert sp.create_all_commits_csv(str(tmp_path / "*" / "data.jsonl"), str(tmp_path / "o.csv")) is False
    assert not (tmp_path / "o.csv").exists()


# --- confirm_date_range / prepare_sample ---

def test_confirm_date_range(all_commits_csv):
    assert sp.confirm_date_range(str(all_commits_csv)) == (
        datetime.date(2024, 1, 10), datetime.date(2024, 3, 5),
    )


def test_date_functions_handle_mixed_utc_offsets(tmp_path):
    # Datas reais misturam offsets; o pandas 3 deixa essa coluna como str se não normalizar
    ds = tmp_path / "dataset"
    write_jsonl(str(ds / "r" / "data.jsonl"), [
        make_record(commit_hash="a", date="2024-08-14T08:14:37-07:00"),
        make_record(commit_hash="b", date="2024-01-01T10:00:00+02:00"),
    ])
    csv_path = tmp_path / "all.csv"
    sp.create_all_commits_csv(str(ds / "*" / "data.jsonl"), str(csv_path))

    assert sp.confirm_date_range(str(csv_path)) == (datetime.date(2024, 1, 1), datetime.date(2024, 8, 14))
    assert sp.prepare_sample(str(csv_path), str(tmp_path / "s.csv"), target_n=10) is True


def test_confirm_date_range_missing_file(tmp_path):
    assert sp.confirm_date_range(str(tmp_path / "nope.csv")) == (None, None)


def test_prepare_sample(all_commits_csv, tmp_path):
    out = tmp_path / "sample.csv"
    assert sp.prepare_sample(str(all_commits_csv), str(out), target_n=10) is True

    df = pd.read_csv(out)
    assert list(df.columns) == ["commit_hash", "msg_clean", "relevant_file", "patch_excerpt"]
    assert sorted(df["commit_hash"]) == ["a1", "a2", "b1"]  # merge m1 excluído


def test_prepare_sample_stratifies_by_month(tmp_path):
    ds = tmp_path / "dataset"
    records = [
        make_record(commit_hash=f"jan{i}", date=f"2024-01-{i + 1:02d}T00:00:00+00:00") for i in range(5)
    ] + [make_record(commit_hash="feb0", date="2024-02-01T00:00:00+00:00")]
    write_jsonl(str(ds / "r" / "data.jsonl"), records)
    csv_path = tmp_path / "all.csv"
    sp.create_all_commits_csv(str(ds / "*" / "data.jsonl"), str(csv_path))

    out = tmp_path / "sample.csv"
    # 2 meses, target_n=2 -> até 1 commit por mês
    assert sp.prepare_sample(str(csv_path), str(out), target_n=2)
    hashes = list(pd.read_csv(out)["commit_hash"])
    assert len(hashes) == 2
    assert "feb0" in hashes
    assert sum(h.startswith("jan") for h in hashes) == 1


def test_prepare_sample_missing_file(tmp_path):
    assert sp.prepare_sample(str(tmp_path / "nope.csv"), str(tmp_path / "o.csv")) is False


# --- compute_token_coverage / extract_bigrams ---

def _msg_csv(tmp_path, msgs):
    path = tmp_path / "msgs.csv"
    pd.DataFrame({"msg_clean": msgs}).to_csv(path, index=False)
    return str(path)


def test_compute_token_coverage(tmp_path):
    path = _msg_csv(tmp_path, ["fix fix bug", "fix typo"])  # fix=3, bug=1, typo=1
    assert sp.compute_token_coverage(path, top_n=1) == pytest.approx(3 / 5)
    assert sp.compute_token_coverage(path, top_n=10) == pytest.approx(1.0)


def test_compute_token_coverage_missing_file(tmp_path):
    assert sp.compute_token_coverage(str(tmp_path / "nope.csv")) == 0.0


def test_extract_bigrams(tmp_path):
    path = _msg_csv(tmp_path, ["security group", "security group ingress", "subnet cidr block"])
    bigrams = sp.extract_bigrams(path, top_k=2)

    assert bigrams[0] == ["security group", 2]
    assert len(bigrams) == 2
    assert all(isinstance(count, int) for _, count in bigrams)


def test_extract_bigrams_missing_file(tmp_path):
    assert sp.extract_bigrams(str(tmp_path / "nope.csv")) == []
