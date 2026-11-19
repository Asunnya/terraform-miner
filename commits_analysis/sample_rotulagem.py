#!/usr/bin/env python3
"""Sorteia a amostra de rotulagem manual e gera as planilhas das 3 rodadas, uma por modalidade da LLM.

Amostra aleatória simples de --n commits sobre os commits elegíveis (commits_prepared.jsonl); --overlap deles
vão para dois anotadores (para o kappa), o resto é dividido igualmente. Cada anotador recebe 3 planilhas,
uma por rodada, cada uma em ordem aleatória própria e com o texto exatamente como a LLM recebe
(prompt do config.yaml, sem o SHA):
  rodada1_diff:          só o diff (diff_only)
  rodada2_mensagem:      só a mensagem (message_only)
  rodada3_mensagem_diff: mensagem + diff (message_and_diff)

Os commits comuns têm "SIM" na coluna comum (rotulados primeiro, para medir a concordância).
Saídas em commits_analysis/rotulagem/ (ou --out): <anotador>/<anotador>_rodadaN_*.xlsx e manifest.csv
(item de cada planilha -> commit; só para quem coordena). Uma célula do Excel guarda no máximo 32.767
caracteres: texto mais longo continua nas colunas evidencia_2, evidencia_3, ... (o visualizador.html junta).
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from openpyxl import Workbook  # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill  # noqa: E402
from openpyxl.worksheet.datavalidation import DataValidation  # noqa: E402

from src.llm_analysis.classification import build_prompt_input, experiment_from_config  # noqa: E402
from src.llm_analysis.schemas import PreparedCommit  # noqa: E402
from src.llm_analysis.utils import load_config, read_jsonl, resolve_path  # noqa: E402

ANNOTATORS = ["heleno", "isadora", "joao", "vania"]
ROUNDS = [  # (arquivo, input_mode, prefixo do item)
    ("rodada1_diff", "diff_only", "D"),
    ("rodada2_mensagem", "message_only", "M"),
    ("rodada3_mensagem_diff", "message_and_diff", "A"),
]
LABELS = ("BUG_FIX", "NON_BUG_FIX")
SHA_BLOCK = re.compile(r"Commit SHA:\n[0-9a-f]{40}\n\n")
OUT_DIR = HERE / "rotulagem"
CELL_CHARS = 30000  # abaixo do limite do Excel (32.767), com folga para o escape de \r (_x000D_)


def evidence(user_prompt: str) -> str:
    """O que a LLM recebe sobre o commit: do 'Repository:' em diante, sem o SHA e sem a última instrução."""
    text = user_prompt[user_prompt.index("Repository:"):]
    text = SHA_BLOCK.sub("", text)
    return text.replace("Base your classification only on the evidence above.", "").strip()


def assign(hashes: list[str], overlap: int) -> dict[str, list[str]]:
    pairs = list(combinations(ANNOTATORS, 2))
    who = {h: list(pairs[i % len(pairs)]) for i, h in enumerate(hashes[:overlap])}
    who.update({h: [ANNOTATORS[i % len(ANNOTATORS)]] for i, h in enumerate(hashes[overlap:])})
    return who


def split_cells(text: str) -> list[str]:
    """Corta em fim de linha em pedaços de até CELL_CHARS (contando o escape de \\r); juntos refazem o texto."""
    parts, current, size = [], [], 0
    for line in text.splitlines(keepends=True):
        while line:
            cost = len(line) + 6 * line.count("\r")
            if size + cost <= CELL_CHARS:
                current.append(line)
                size += cost
                break
            if current:
                parts.append("".join(current))
                current, size = [], 0
                continue
            cut = CELL_CHARS // 7  # linha sozinha maior que uma célula
            parts.append(line[:cut])
            line = line[cut:]
    if current or not parts:
        parts.append("".join(current))
    return parts


def write_sheet(path: Path, rows: list[dict]) -> None:
    """rows: dicts com item, evidencia, comum (bool) e, opcionalmente, rotulo e obs já preenchidos."""
    cells = [(r, split_cells(r["evidencia"])) for r in rows]
    extra = [f"evidencia_{i}" for i in range(2, max(len(parts) for _, parts in cells) + 1)]
    wb = Workbook()
    ws = wb.active
    ws.title = "Rotulagem"
    ws.append(["item", "comum", "evidencia", "rotulo", "obs", *extra])
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="DDDDDD")
    for r, parts in cells:
        ws.append([r["item"], "SIM" if r["comum"] else "", parts[0], r.get("rotulo", ""), r.get("obs", ""),
                   *parts[1:]])
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    common_fill = PatternFill("solid", fgColor="FFF2CC")
    for row in ws.iter_rows(min_row=2, max_col=2):
        if row[1].value:
            for cell in row:
                cell.fill = common_fill
    widths = {"A": 8, "B": 9, "C": 120, "D": 16, "E": 40}
    for i in range(len(extra)):
        widths[chr(ord("F") + i)] = 120
    for col, width in widths.items():
        ws.column_dimensions[col].width = width
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:E{len(rows) + 1}"
    validation = DataValidation(type="list", formula1=f'"{",".join(LABELS)}"', allow_blank=True)
    ws.add_data_validation(validation)
    validation.add(f"D2:D{len(rows) + 1}")
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=361, help="tamanho da amostra (95%%, ±5%%, N=5.941: 361)")
    parser.add_argument("--overlap", type=int, default=40, help="commits rotulados por dois anotadores")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="pasta de saída (padrão: %(default)s)")
    args = parser.parse_args()

    cfg = load_config(None)
    experiment, _, template = experiment_from_config(cfg)
    prepared = resolve_path(cfg["output"]["directory"]) / cfg["output"]["prepared_commits_file"]
    commits = {c.commit_hash: c for c in (PreparedCommit.from_dict(r) for _, r in read_jsonl(prepared))}

    rng = random.Random(args.seed)
    sample = rng.sample(sorted(commits), args.n)
    who = assign(sample, args.overlap)

    manifest = []
    for annotator in ANNOTATORS:
        mine = [h for h in sample if annotator in who[h]]
        for name, mode, prefix in ROUNDS:
            order = mine[:]
            rng.shuffle(order)
            rows = []
            for i, h in enumerate(order, start=1):
                item = f"{prefix}{i:03d}"
                prompt = build_prompt_input(commits[h], mode, template, experiment.limits).user_prompt
                rows.append({"item": item, "evidencia": evidence(prompt), "comum": len(who[h]) > 1})
                manifest.append({"anotador": annotator, "rodada": name, "input_mode": mode, "item": item,
                                 "repo": commits[h].repo, "commit_hash": h,
                                 "dupla": ";".join(who[h]) if len(who[h]) > 1 else ""})
            write_sheet(args.out / annotator / f"{annotator}_{name}.xlsx", rows)
        print(f"{annotator}: {len(mine)} commits x {len(ROUNDS)} rodadas")

    with open(args.out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"amostra: {len(sample)} commits, {args.overlap} com dois anotadores; manifest: {args.out / 'manifest.csv'}")


if __name__ == "__main__":
    main()
