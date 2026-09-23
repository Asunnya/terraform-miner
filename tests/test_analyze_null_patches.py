import sys

import pytest

from src import analyze_null_patches as anp
from tests.helpers import make_record, write_jsonl


def test_analyze_file_counts(tmp_path):
    path = write_jsonl(str(tmp_path / "data.jsonl"), [
        make_record(file="main.tf", patch="@@ diff"),
        make_record(file="vars.TF", patch=""),
        make_record(file="outputs.tf", patch=None),
        make_record(file="README.md", patch=""),
        make_record(file=None),
    ], extra_lines=["", "   "])

    assert anp.analyze_file(path) == (3, 2, 2)  # (total .tf, vazios, não-.tf ignorados)


def test_main_prints_totals(tmp_path, monkeypatch, capsys):
    write_jsonl(str(tmp_path / "repo_a" / "data.jsonl"), [
        make_record(patch=""), make_record(patch="x"), make_record(patch="x"), make_record(patch="x"),
    ])
    write_jsonl(str(tmp_path / "repo_b" / "data.jsonl"), [make_record(file="a.md")])  # sem .tf: fica fora da tabela
    (tmp_path / "not_a_repo").mkdir()

    monkeypatch.setattr(sys, "argv", ["analyze_null_patches.py", str(tmp_path)])
    anp.main()
    out = capsys.readouterr().out

    assert "repo_a" in out
    assert "repo_b" not in out
    total_line = next(line for line in out.splitlines() if line.startswith("TOTAL"))
    assert total_line.split()[1:] == ["4", "1", "25.00%"]
    assert "Registros ignorados (arquivo não é .tf): 1" in out


def test_main_missing_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["analyze_null_patches.py", str(tmp_path / "nope")])
    with pytest.raises(SystemExit, match="Diretório não encontrado"):
        anp.main()
