#!/usr/bin/env python3
"""Etapa de calibração da rotulagem: os 40 commits comuns (rotulados por dois anotadores) vêm primeiro.

  marcar    põe a coluna comum (SIM) nas planilhas já geradas, sem mudar itens, ordem nem rótulos.
  calcular  lê as planilhas preenchidas de rotulagem/<nome>/ e, por rodada, mede a
            concordância nos commits comuns: % de acordo, kappa de Fleiss (pares de anotadores variam),
            kappa de Cohen (comparável com a literatura) e IC de 95% por bootstrap. Grava as divergências
            em rotulagem/concordancia_divergencias.csv para a reunião de discussão.
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from openpyxl import load_workbook

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sample_rotulagem import ANNOTATORS, LABELS, OUT_DIR, ROUNDS, write_sheet  # noqa: E402

ROUND_RE = re.compile(r"(rodada\d_(?:mensagem_diff|mensagem|diff))\.xlsx$")
EVIDENCE_RE = re.compile(r"evidencia(_\d+)?$")


def read_sheet(path: Path) -> list[dict]:
    ws = load_workbook(path).active
    rows = ws.iter_rows(values_only=True)
    header = [str(h or "") for h in next(rows)]
    ev_cols = sorted((i for i, h in enumerate(header) if EVIDENCE_RE.match(h)),
                     key=lambda i: int(header[i].split("_")[1]) if "_" in header[i] else 1)
    out = []
    for row in rows:
        if not row or not row[0]:
            continue
        rec = dict(zip(header, row))
        out.append({"item": str(rec["item"]), "rotulo": str(rec.get("rotulo") or "").strip().upper(),
                    "obs": str(rec.get("obs") or "").strip(),
                    "evidencia": "".join(str(row[i] or "") for i in ev_cols)})
    return out


def load_manifest(base: Path) -> list[dict]:
    with open(base / "manifest.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def marcar(base: Path, annotators: list[str]) -> None:
    """Põe a coluna comum (SIM nos commits com dupla) nas planilhas, mantendo itens, ordem, rótulos e obs."""
    common = {(m["anotador"], m["rodada"], m["item"]) for m in load_manifest(base) if m["dupla"]}
    for annotator in annotators:
        for name, _, _ in ROUNDS:
            path = base / annotator / f"{annotator}_{name}.xlsx"
            rows = read_sheet(path)
            for r in rows:
                r["comum"] = (annotator, name, r["item"]) in common
            write_sheet(path, rows)
            print(f"{path.name}: {sum(r['comum'] for r in rows)} comuns de {len(rows)}")


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in LABELS)
    return None if pe == 1 else (po - pe) / (1 - pe)


def fleiss_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Dois rótulos por item, anotadores diferentes em cada item (equivale ao pi de Scott)."""
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    flat = [label for pair in pairs for label in pair]
    pe = sum((flat.count(c) / len(flat)) ** 2 for c in LABELS)
    return None if pe == 1 else (po - pe) / (1 - pe)


def bootstrap_ci(pairs: list[tuple[str, str]], reps: int = 2000, seed: int = 42) -> tuple[float, float] | None:
    rng = random.Random(seed)
    values = sorted(k for k in (fleiss_kappa([rng.choice(pairs) for _ in pairs]) for _ in range(reps))
                    if k is not None)
    if len(values) < reps * 0.9:
        return None
    return values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]


def fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f}"


def calcular(base: Path, out_csv: Path) -> None:
    manifest = {(m["anotador"], m["rodada"], m["item"]): m for m in load_manifest(base)}
    labels: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)  # (rodada, commit) -> anotador -> linha
    for annotator in ANNOTATORS:
        for path in sorted((base / annotator).glob("*.xlsx")):
            match = ROUND_RE.search(path.name)
            if not match:
                continue
            rodada = match.group(1)
            for r in read_sheet(path):
                m = manifest.get((annotator, rodada, r["item"]))
                if m is None:
                    sys.exit(f"{path.name}: item {r['item']} não está no manifest.csv")
                if m["dupla"] and r["rotulo"] in LABELS:
                    labels[(rodada, m["commit_hash"])][annotator] = {**r, "repo": m["repo"]}

    common = {(m["rodada"], m["commit_hash"]): m["dupla"].split(";") for m in manifest.values() if m["dupla"]}
    divergences = []
    print(f"{'rodada':<24}{'n':>4}{'acordo':>8}{'Fleiss':>8}{'IC95%':>14}{'Cohen':>7}  pendentes")
    for name, _, _ in ROUNDS:
        keys = sorted(k for k in common if k[0] == name)
        pairs, by_pair, pending = [], defaultdict(lambda: [0, 0]), defaultdict(int)
        for key in keys:
            a, b = sorted(common[key])
            got = labels.get(key, {})
            for who in (a, b):
                if who not in got:
                    pending[who] += 1
            if a not in got or b not in got:
                continue
            la, lb = got[a]["rotulo"], got[b]["rotulo"]
            pairs.append((la, lb))
            by_pair[f"{a}-{b}"][0] += la == lb
            by_pair[f"{a}-{b}"][1] += 1
            if la != lb:
                divergences.append({"rodada": name, "repo": got[a]["repo"], "commit_hash": key[1],
                                    "anotador_1": a, "item_1": got[a]["item"], "rotulo_1": la, "obs_1": got[a]["obs"],
                                    "anotador_2": b, "item_2": got[b]["item"], "rotulo_2": lb, "obs_2": got[b]["obs"]})
        pend = ", ".join(f"{w} {c}" for w, c in sorted(pending.items())) or "-"
        if not pairs:
            print(f"{name:<24}{0:>4}{'':>8}{'':>8}{'':>14}{'':>7}  {pend}")
            continue
        agree = sum(x == y for x, y in pairs) / len(pairs)
        ci = bootstrap_ci(pairs)
        ci_txt = f"[{fmt(ci[0])}, {fmt(ci[1])}]" if ci else "-"
        cohen = cohen_kappa([x for x, _ in pairs], [y for _, y in pairs])
        print(f"{name:<24}{len(pairs):>4}{agree:>8.0%}{fmt(fleiss_kappa(pairs)):>8}{ci_txt:>14}{fmt(cohen):>7}  {pend}")
        print("    por dupla: " + ", ".join(f"{p} {ok}/{n}" for p, (ok, n) in sorted(by_pair.items())))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fields = ["rodada", "repo", "commit_hash", "anotador_1", "item_1", "rotulo_1", "obs_1",
                  "anotador_2", "item_2", "rotulo_2", "obs_2"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(divergences)
    print(f"{len(divergences)} divergências em {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=OUT_DIR, help="pasta da rotulagem (padrão: %(default)s)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    mark = sub.add_parser("marcar", help="põe a coluna comum nas planilhas")
    mark.add_argument("anotadores", nargs="*", default=ANNOTATORS)
    calc = sub.add_parser("calcular", help="concordância nos commits comuns")
    calc.add_argument("--out", type=Path, help="CSV de divergências (padrão: <dir>/concordancia_divergencias.csv)")
    args = parser.parse_args()
    if args.cmd == "marcar":
        marcar(args.dir, args.anotadores)
    else:
        calcular(args.dir, args.out or args.dir / "concordancia_divergencias.csv")


if __name__ == "__main__":
    main()
