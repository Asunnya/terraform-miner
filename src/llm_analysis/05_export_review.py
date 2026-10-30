#!/usr/bin/env python3
"""Stage 05: one review spreadsheet (.xlsx) per experiment, one row per commit. Details: docs/pipeline-llm.md."""
from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import date
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openpyxl import Workbook  # noqa: E402
from openpyxl.formatting.rule import CellIsRule  # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill  # noqa: E402
from openpyxl.utils import get_column_letter  # noqa: E402
from openpyxl.worksheet.datavalidation import DataValidation  # noqa: E402

from src.llm_analysis.classification import experiment_from_config  # noqa: E402
from src.llm_analysis.evaluation import load_experiment_results, outcome  # noqa: E402
from src.llm_analysis.llm_client import LLMConfigError  # noqa: E402
from src.llm_analysis.schemas import INPUT_MODES, LABELS  # noqa: E402
from src.llm_analysis.utils import cli_or_config_path, load_config, read_jsonl, resolve_path, setup_logging  # noqa: E402

logger = logging.getLogger("llm_analysis.review")

FONT = "Arial"
REVIEW_OPTIONS = ("BUG_FIX", "NON_BUG_FIX", "MAYBE")
SHUFFLE_SEED = 20261003
MAX_CELL_TEXT = 32_000  # Excel limit is 32,767 characters per cell
SHORT_MODE = {"message_only": "msg", "diff_only": "diff", "message_and_diff": "msg_diff"}
INPUT_FILL = PatternFill("solid", start_color="FFFF00")
HEADER_FILL = PatternFill("solid", start_color="D9D9D9")
BUG_FILL = PatternFill("solid", start_color="F8CBAD")
NON_FILL = PatternFill("solid", start_color="C6E0B4")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Config file (default: src/llm_analysis/config.yaml)")
    parser.add_argument("--output-dir", help="Override output.directory")
    parser.add_argument("--experiment-id", help="Export this experiment instead of the configured one")
    parser.add_argument("--model", help="Override llm.model when deriving the experiment id")
    parser.add_argument("--out", help="Output .xlsx (default: reports/<experiment_id>/revisao_<experiment_id>.xlsx)")
    return parser.parse_args(argv)


def repo_names(summary_path: Path) -> list[str]:
    if not summary_path.exists():
        return []
    with open(summary_path, encoding="utf-8") as f:
        return [r["repo_full_name"] for r in json.load(f).get("processed_repositories_summary", [])]


def full_repo_name(clone_dir: str, names: list[str]) -> str:
    """Clone dir "owner_repo_1a2b3c4d" -> "owner/repo" (longest matching name wins)."""
    matches = [n for n in names if clone_dir.startswith(n.replace("/", "_") + "_")]
    return max(matches, key=len) if matches else clone_dir


def clip(text: str | None) -> str:
    text = text or ""
    return text if len(text) <= MAX_CELL_TEXT else text[:MAX_CELL_TEXT] + " [...]"


def build_rows(commits: list[dict], results: dict, names: list[str]) -> list[dict]:
    order = list(range(1, len(commits) + 1))
    random.Random(SHUFFLE_SEED).shuffle(order)
    rows = []
    for commit, rank in zip(commits, order):
        key = (commit["repo"], commit["commit_hash"])
        repo = full_repo_name(commit["repo"], names)
        row = {
            "ordem_aleatoria": rank,
            "repo": repo,
            "commit_hash": commit["commit_hash"],
            "commit_url": f"https://github.com/{repo}/commit/{commit['commit_hash']}",
            "data": (commit.get("date") or "")[:10],
            "mensagem_1a_linha": (commit["message"].strip().splitlines() or [""])[0],
            "mensagem_completa": clip(commit["message"]),
            "n_arquivos": len(commit["files"]),
            "arquivos": clip("\n".join(commit["files"])),
            "arquivos_excluidos": len(commit.get("excluded_files") or []),
        }
        labels = {}
        for mode in INPUT_MODES:
            rec = results.get(mode, {}).get(key)
            short = SHORT_MODE[mode]
            labels[mode] = outcome(rec) if rec else "NOT_RUN"
            row[f"rotulo_{short}"] = labels[mode]
            row[f"confianca_{short}"] = rec.get("confidence") if rec else None
            row[f"intencao_{short}"] = clip(rec.get("primary_intent")) if rec else ""
            row[f"justificativa_{short}"] = clip(rec.get("reason")) if rec else ""
            if mode == "message_and_diff":
                row["diff_cortado"] = "sim" if rec and rec.get("diff_truncated") else "não"
        valid = [v for v in labels.values() if v in LABELS]
        row["condicoes_concordam"] = "sim" if len(valid) == len(INPUT_MODES) and len(set(valid)) == 1 else "não"
        row.update(gold_label=None, revisor=None, observacoes=None)
        rows.append(row)
    return rows


COLUMNS = [  # (key, header width, wrap)
    ("ordem_aleatoria", 9, False), ("repo", 26, False), ("commit_hash", 12, False), ("commit_url", 14, False),
    ("data", 11, False), ("mensagem_1a_linha", 45, True), ("mensagem_completa", 45, True),
    ("n_arquivos", 8, False), ("arquivos", 40, True), ("arquivos_excluidos", 9, False),
    ("rotulo_msg", 13, False), ("confianca_msg", 9, False),
    ("rotulo_diff", 13, False), ("confianca_diff", 9, False),
    ("rotulo_msg_diff", 13, False), ("confianca_msg_diff", 9, False),
    ("condicoes_concordam", 10, False), ("diff_cortado", 8, False),
    ("intencao_msg_diff", 35, True), ("justificativa_msg_diff", 55, True),
    ("justificativa_msg", 45, True), ("justificativa_diff", 45, True),
    ("gold_label", 14, False), ("revisor", 12, False), ("observacoes", 40, True),
]
INPUT_COLUMNS = ("gold_label", "revisor", "observacoes")


def write_review_sheet(ws, rows: list[dict]) -> dict[str, str]:
    letters = {key: get_column_letter(i) for i, (key, _, _) in enumerate(COLUMNS, start=1)}
    for i, (key, width, _) in enumerate(COLUMNS, start=1):
        cell = ws.cell(row=1, column=i, value=key)
        cell.font = Font(name=FONT, bold=True)
        cell.fill = INPUT_FILL if key in INPUT_COLUMNS else HEADER_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        ws.column_dimensions[get_column_letter(i)].width = width
    body_font, link_font = Font(name=FONT, size=9), Font(name=FONT, size=9, color="0563C1", underline="single")
    for r, row in enumerate(rows, start=2):
        for i, (key, _, wrap) in enumerate(COLUMNS, start=1):
            cell = ws.cell(row=r, column=i, value=row[key])
            cell.font = body_font
            cell.alignment = Alignment(wrap_text=wrap, vertical="top")
            if key == "commit_url":
                cell.hyperlink, cell.font = row[key], link_font
            elif key.startswith("confianca_"):
                cell.number_format = "0.00"
    last = len(rows) + 1
    ws.freeze_panes = "G2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(COLUMNS))}{last}"
    validation = DataValidation(type="list", formula1='"' + ",".join(REVIEW_OPTIONS) + '"', allow_blank=True)
    validation.error = "Use BUG_FIX, NON_BUG_FIX ou MAYBE"
    ws.add_data_validation(validation)
    validation.add(f"{letters['gold_label']}2:{letters['gold_label']}{last}")
    for key in ("rotulo_msg", "rotulo_diff", "rotulo_msg_diff", "gold_label"):
        rng = f"{letters[key]}2:{letters[key]}{last}"
        ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=['"BUG_FIX"'], fill=BUG_FILL))
        ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=['"NON_BUG_FIX"'], fill=NON_FILL))
    return letters


def write_summary_sheet(ws, letters: dict[str, str], n: int, meta: dict) -> None:
    rv = "Revisao"
    last = n + 1

    def col(key):
        return f"{rv}!${letters[key]}$2:${letters[key]}${last}"

    bold = Font(name=FONT, bold=True)
    ws["A1"], ws["A1"].font = "Resumo dos resultados do LLM", Font(name=FONT, bold=True, size=13)
    ws["A2"] = f"Experimento {meta['experiment_id']} · {meta['model']} · prompt {meta['prompt_version']} · " \
               f"{n} commits elegíveis. Valores calculados por fórmula a partir da aba Revisao."
    ws["A2"].font = Font(name=FONT, italic=True, size=9)

    headers = ["Condição", "BUG_FIX", "NON_BUG_FIX", "Falha", "Taxa BUG_FIX", "Confiança média"]
    for i, h in enumerate(headers, start=1):
        c = ws.cell(row=4, column=i, value=h)
        c.font, c.fill = bold, HEADER_FILL
    for r, (name, key) in enumerate((("message_only", "rotulo_msg"), ("diff_only", "rotulo_diff"),
                                     ("message_and_diff", "rotulo_msg_diff")), start=5):
        conf = key.replace("rotulo", "confianca")
        ws.cell(row=r, column=1, value=name)
        ws.cell(row=r, column=2, value=f'=COUNTIF({col(key)},"BUG_FIX")')
        ws.cell(row=r, column=3, value=f'=COUNTIF({col(key)},"NON_BUG_FIX")')
        ws.cell(row=r, column=4, value=f"=ROWS({col(key)})-B{r}-C{r}")
        ws.cell(row=r, column=5, value=f"=IFERROR(B{r}/(B{r}+C{r}),0)").number_format = "0.0%"
        ws.cell(row=r, column=6, value=f"=IFERROR(AVERAGE({col(conf)}),0)").number_format = "0.00"
    ws["A8"] = "Commits em que as 3 condições concordam"
    ws["B8"] = f'=COUNTIF({col("condicoes_concordam")},"sim")'
    ws["E8"] = f"=B8/{n}"
    ws["E8"].number_format = "0.0%"

    ws["A10"], ws["A10"].font = "Progresso da revisão", bold
    ws["A11"], ws["B11"] = "Commits revisados (gold_label preenchido)", f"=COUNTA({col('gold_label')})"
    ws["A12"], ws["B12"] = "  BUG_FIX", f'=COUNTIF({col("gold_label")},"BUG_FIX")'
    ws["A13"], ws["B13"] = "  NON_BUG_FIX", f'=COUNTIF({col("gold_label")},"NON_BUG_FIX")'
    ws["A14"], ws["B14"] = "  MAYBE", f'=COUNTIF({col("gold_label")},"MAYBE")'

    ws["A16"], ws["A16"].font = "Concordância LLM × revisão (só commits com BUG_FIX/NON_BUG_FIX na revisão)", bold
    for i, h in enumerate(["Condição", "Iguais", "Comparados", "Concordância"], start=1):
        c = ws.cell(row=17, column=i, value=h)
        c.font, c.fill = bold, HEADER_FILL
    gold = col("gold_label")
    for r, (name, key) in enumerate((("message_only", "rotulo_msg"), ("diff_only", "rotulo_diff"),
                                     ("message_and_diff", "rotulo_msg_diff")), start=18):
        ws.cell(row=r, column=1, value=name)
        ws.cell(row=r, column=2, value=f'=SUMPRODUCT(({col(key)}={gold})*(({gold}="BUG_FIX")+({gold}="NON_BUG_FIX")))')
        ws.cell(row=r, column=3, value="=B12+B13")
        ws.cell(row=r, column=4, value=f"=IFERROR(B{r}/C{r},0)").number_format = "0.0%"

    ws["A22"], ws["A22"].font = "Custo da execução (valores fixos, de 04_cost_report.py)", bold
    for r, (label, value) in enumerate(meta["cost_rows"], start=23):
        ws.cell(row=r, column=1, value=label)
        ws.cell(row=r, column=2, value=value).font = Font(name=FONT, color="0000FF")
    ws.cell(row=23 + len(meta["cost_rows"]), column=1,
            value=f"Fonte: {meta['cost_source']}").font = Font(name=FONT, italic=True, size=9)
    for row in ws.iter_rows():
        for cell in row:
            if cell.font.name != FONT:
                cell.font = Font(name=FONT, bold=cell.font.bold, italic=cell.font.italic, size=cell.font.size,
                                 color=cell.font.color)
    ws.column_dimensions["A"].width = 58
    for c in "BCDEF":
        ws.column_dimensions[c].width = 15


def write_readme_sheet(ws, meta: dict, system_prompt: str) -> None:
    bold = Font(name=FONT, bold=True)
    lines = [
        ("Revisão dos resultados da classificação por LLM", Font(name=FONT, bold=True, size=13)),
        (f"Experimento {meta['experiment_id']} · modelo {meta['model']} · prompt {meta['prompt_version']} · "
         f"gerado em {meta['created_at']}", None),
        ("", None),
        ("COMO REVISAR", bold),
        ("1. Na aba Revisao, ordene por ordem_aleatoria (crescente) para revisar uma amostra sorteada, ou filtre "
         "condicoes_concordam = não para ver os casos em que as condições discordam.", None),
        ("2. Abra o commit pelo link da coluna commit_url e leia a mensagem e o diff no GitHub.", None),
        ("3. Preencha SÓ as colunas amarelas: gold_label (BUG_FIX, NON_BUG_FIX ou MAYBE), revisor (seu nome) e "
         "observacoes. Não altere as outras colunas.", None),
        ("4. Decida pela finalidade principal do commit, com a definição abaixo, sem olhar só as palavras da mensagem.",
         None),
        ("", None),
        ("DEFINIÇÃO (resumo do prompt de sistema)", bold),
        ("BUG_FIX: a finalidade principal é corrigir comportamento, configuração ou implementação existente que estava "
         "defeituoso, errado, não intencional, inseguro ou inválido (restaura algo que já deveria funcionar).", None),
        ("NON_BUG_FIX: qualquer outra finalidade: funcionalidade ou recurso novo, refatoração, limpeza, formatação, "
         "renomeação, documentação, exemplos, testes, atualização de versão sem defeito demonstrado, merge, revert.",
         None),
        ("MAYBE: a evidência não permite decidir. Explique em observacoes. Esses commits não entram nas métricas.",
         None),
        ("", None),
        ("EXEMPLO DE LINHA PREENCHIDA (ilustrativo, não está na aba Revisao)", bold),
        ("mensagem: 'fix: ensure test/setup/outputs.folder_id returns proper ID' · rotulo_msg_diff: BUG_FIX · "
         "gold_label: BUG_FIX · revisor: Isadora · observacoes: troca .id por .folder_id; atributo errado.", None),
        ("", None),
        ("COLUNAS", bold),
        ("ordem_aleatoria: posição num sorteio fixo (semente 20261003), para amostrar. "
         "rotulo_* / confianca_*: resposta do modelo em cada condição (msg = só mensagem; diff = só diff; msg_diff = "
         "os dois). PARSE_FAILURE = resposta inválida. A confiança é declarada pelo modelo, não é probabilidade.", None),
        ("condicoes_concordam: 'sim' se as 3 condições deram o mesmo rótulo. diff_cortado: o diff enviado ao modelo "
         "foi cortado por tamanho. arquivos_excluidos: arquivos de teste/exemplo/etc. removidos antes do LLM.", None),
        ("intencao_msg_diff / justificativa_*: texto do modelo. Use como apoio, não como verdade.", None),
        ("", None),
        ("USO DA REVISÃO", bold),
        ("As colunas commit_hash e gold_label permitem calcular precision/recall/F1/MCC por condição: salve esta aba "
         "como CSV e rode 03_summarize_results.py --gold-labels <arquivo.csv>. A aba Resumo já mostra a "
         "concordância à medida que a revisão é preenchida.", None),
        ("", None),
        ("PROMPT DE SISTEMA COMPLETO", bold),
        (clip(system_prompt), Font(name=FONT, size=8)),
    ]
    for r, (text, font) in enumerate(lines, start=1):
        cell = ws.cell(row=r, column=1, value=text)
        cell.font = font or Font(name=FONT)
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.column_dimensions["A"].width = 140


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    output_dir = cli_or_config_path(args.output_dir, cfg["output"]["directory"])
    setup_logging(output_dir / "pipeline.log")
    try:
        experiment, settings, template = experiment_from_config(cfg, model=args.model)
    except (LLMConfigError, ValueError, KeyError) as e:
        logger.error("Invalid configuration: %s", e)
        return 2
    experiment_id = args.experiment_id or experiment.experiment_id
    results = load_experiment_results(output_dir, experiment_id)
    if not any(results.values()):
        logger.error("No classifications of experiment %s in %s", experiment_id, output_dir)
        return 1
    commits = [r for _, r in read_jsonl(output_dir / cfg["output"]["prepared_commits_file"])]
    names = repo_names(resolve_path(cfg["input"]["dataset_glob"]).parents[2] / "logs" / "summary.json")
    rows = build_rows(commits, results, names)

    report_dir = output_dir / "reports" / experiment_id
    cost_path = report_dir / "costs_summary.json"
    cost_rows, cost_source = [], "rode 04_cost_report.py para preencher"
    if cost_path.exists():
        with open(cost_path, encoding="utf-8") as f:
            cost = json.load(f)
        total = cost["total"]
        cost_rows = [("Chamadas", total["calls"]), ("Tokens de entrada", total["input_tokens"]),
                     ("Tokens de saída", total["output_tokens"]), ("Custo (US$)", total["cost_usd"])]
        cost_source = f"{cost_path.name}, preços de {cost.get('prices_as_of')} ({cost.get('prices_source')})"
    any_rec = next(r for mode in results.values() for r in mode.values())
    meta = {"experiment_id": experiment_id, "model": any_rec["model"], "prompt_version": any_rec["prompt_version"],
            "created_at": date.today().isoformat(), "cost_rows": cost_rows,
            "cost_source": cost_source}

    wb = Workbook()
    readme = wb.active
    readme.title = "Leia-me"
    review = wb.create_sheet("Revisao")
    summary = wb.create_sheet("Resumo")
    letters = write_review_sheet(review, rows)
    write_summary_sheet(summary, letters, len(rows), meta)
    write_readme_sheet(readme, meta, template.system if any_rec["prompt_version"] == template.version else "")
    out = Path(args.out) if args.out else report_dir / f"revisao_{experiment_id}.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.calculation.fullCalcOnLoad = True  # no cached values: recompute on open
    wb.save(out)
    logger.info("Planilha de revisão: %s (%d commits)", out, len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
