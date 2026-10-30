#!/usr/bin/env python3
"""Sorteia a amostra de rotulagem manual e gera as planilhas das 3 rodadas, uma por modalidade da LLM.

Amostra aleatória simples de --n commits sobre os commits elegíveis (commits_prepared.jsonl); --overlap deles
vão para dois anotadores (para o kappa), o resto é dividido igualmente. Cada anotador recebe 3 planilhas,
uma por rodada, cada uma em ordem aleatória própria e com o texto exatamente como a LLM recebe
(prompt do config.yaml, sem o SHA):
  rodada1_diff:          só o diff (diff_only)
  rodada2_mensagem:      só a mensagem (message_only)
  rodada3_mensagem_diff: mensagem + diff (message_and_diff)

Saídas em commits_analysis/rotulagem/: <anotador>/<anotador>_rodadaN_*.xlsx e manifest.csv
(item de cada planilha -> commit; só para quem coordena).
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


def write_sheet(path: Path, rows: list[tuple[str, str]]) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Rotulagem"
    ws.append(["item", "evidencia", "rotulo", "obs"])
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="DDDDDD")
    for item, text in rows:
        ws.append([item, text, "", ""])
    for row in ws.iter_rows(min_row=2):
        row[1].alignment = Alignment(wrap_text=True, vertical="top")
        row[0].alignment = row[2].alignment = row[3].alignment = Alignment(vertical="top", wrap_text=True)
    ws.column_dimensions["A"].width = 8
    ws.column_dimensions["B"].width = 120
    ws.column_dimensions["C"].width = 16
    ws.column_dimensions["D"].width = 40
    ws.freeze_panes = "A2"
    validation = DataValidation(type="list", formula1=f'"{",".join(LABELS)}"', allow_blank=True)
    ws.add_data_validation(validation)
    validation.add(f"C2:C{len(rows) + 1}")
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=361, help="tamanho da amostra (95%%, ±5%%, N=5.941: 361)")
    parser.add_argument("--overlap", type=int, default=40, help="commits rotulados por dois anotadores")
    parser.add_argument("--seed", type=int, default=42)
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
                rows.append((item, evidence(prompt)))
                manifest.append({"anotador": annotator, "rodada": name, "input_mode": mode, "item": item,
                                 "repo": commits[h].repo, "commit_hash": h,
                                 "dupla": ";".join(who[h]) if len(who[h]) > 1 else ""})
            write_sheet(OUT_DIR / annotator / f"{annotator}_{name}.xlsx", rows)
        print(f"{annotator}: {len(mine)} commits x {len(ROUNDS)} rodadas")

    with open(OUT_DIR / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"amostra: {len(sample)} commits, {args.overlap} com dois anotadores; manifest: {OUT_DIR / 'manifest.csv'}")


if __name__ == "__main__":
    main()
