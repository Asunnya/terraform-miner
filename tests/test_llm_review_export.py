"""Planilha de revisão (estágio 05) e leitura dela como gold labels no estágio 03."""
import json

import pytest

pytest.importorskip("openpyxl")
from openpyxl import load_workbook  # noqa: E402

from tests.helpers import make_record, write_jsonl  # noqa: E402
from tests.test_llm_pipeline import load_script, write_config  # noqa: E402


@pytest.fixture
def classified(tmp_path):
    dataset, out = tmp_path / "dataset", tmp_path / "out"
    write_jsonl(str(dataset / "owner_repo" / "data.jsonl"), [
        make_record(repo="owner_repo_1a2b3c4d", commit_hash="1" * 40, message="Fix cidr"),
        make_record(repo="owner_repo_1a2b3c4d", commit_hash="2" * 40, message="Add module"),
        make_record(repo="owner_repo_1a2b3c4d", commit_hash="3" * 40, file="tests/x.tf"),  # fora do escopo
    ])
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "summary.json").write_text(json.dumps({"processed_repositories_summary": [{"repo_full_name": "owner/repo"}]}))
    config = write_config(tmp_path, dataset, out)
    assert load_script("01_prepare_commits.py").main(["--config", str(config)]) == 0
    assert load_script("02_classify_commits.py").main(["--config", str(config)]) == 0
    return config, out


def test_review_workbook(classified):
    config, out = classified
    xlsx = out / "review.xlsx"
    assert load_script("05_export_review.py").main(["--config", str(config), "--out", str(xlsx)]) == 0

    wb = load_workbook(xlsx)
    assert wb.sheetnames == ["Leia-me", "Revisao", "Resumo"]
    assert wb.calculation.fullCalcOnLoad
    ws = wb["Revisao"]
    header = [c.value for c in ws[1]]
    rows = [dict(zip(header, r)) for r in ws.iter_rows(min_row=2, values_only=True)]
    assert len(rows) == 2                                               # só os elegíveis
    assert {"commit_hash", "gold_label", "revisor", "observacoes", "rotulo_msg_diff"} <= set(header)
    assert "label" not in header                                        # não confunde o leitor de gold labels
    assert rows[0]["commit_url"] == f"https://github.com/owner/repo/commit/{'1' * 40}"
    assert ws.cell(row=2, column=header.index("commit_url") + 1).hyperlink.target == rows[0]["commit_url"]
    assert all(r["rotulo_msg"] in ("BUG_FIX", "NON_BUG_FIX") for r in rows)
    assert sorted(r["ordem_aleatoria"] for r in rows) == [1, 2]
    assert all(r["gold_label"] is None for r in rows)
    assert any(dv.formula1 == '"BUG_FIX,NON_BUG_FIX,MAYBE"' for dv in ws.data_validations.dataValidation)
    formulas = [c.value for row in wb["Resumo"].iter_rows() for c in row if str(c.value).startswith("=")]
    assert any("COUNTIF(Revisao!" in f for f in formulas)


def test_filled_review_is_read_back_as_gold_labels(classified, tmp_path):
    config, out = classified
    xlsx = out / "review.xlsx"
    load_script("05_export_review.py").main(["--config", str(config), "--out", str(xlsx)])
    wb = load_workbook(xlsx)
    ws = wb["Revisao"]
    header = [c.value for c in ws[1]]
    ws.cell(row=2, column=header.index("gold_label") + 1, value="BUG_FIX")
    ws.cell(row=3, column=header.index("gold_label") + 1, value="MAYBE")
    wb.save(xlsx)

    assert load_script("03_summarize_results.py").main(["--config", str(config), "--gold-labels", str(xlsx)]) == 0
    [report] = list((out / "reports").iterdir())
    gold = json.loads((report / "summary.json").read_text())["gold_evaluation"]
    assert gold["gold_commits_matched"] == 1                            # MAYBE e em branco ficam de fora
    assert gold["gold_file_stats"]["usable_labels"] == 1
