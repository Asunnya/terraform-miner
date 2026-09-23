#!/usr/bin/env python3
"""Analisa a proporção de registros com campo 'patch' nulo/vazio no dataset minerado."""
import argparse
import json
import os


def is_terraform_file(file_path):
    return os.path.splitext(file_path or "")[1].lower() == ".tf"


def analyze_file(path):
    total = 0
    empty = 0
    skipped_non_tf = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not is_terraform_file(record.get("file")):
                skipped_non_tf += 1
                continue
            total += 1
            if not record.get("patch"):
                empty += 1
    return total, empty, skipped_non_tf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "dataset"),
        help="Diretório contendo as subpastas de repositórios com data.jsonl (padrão: msr/data/dataset)",
    )
    args = parser.parse_args()
    dataset_dir = os.path.abspath(args.dataset_dir)

    if not os.path.isdir(dataset_dir):
        raise SystemExit(f"Diretório não encontrado: {dataset_dir}")

    results = []
    grand_total = 0
    grand_empty = 0
    grand_skipped = 0

    for repo_name in sorted(os.listdir(dataset_dir)):
        repo_path = os.path.join(dataset_dir, repo_name)
        jsonl_path = os.path.join(repo_path, "data.jsonl")
        if not os.path.isfile(jsonl_path):
            continue

        total, empty, skipped_non_tf = analyze_file(jsonl_path)
        grand_skipped += skipped_non_tf
        if total == 0:
            continue

        pct = (empty / total) * 100
        results.append((repo_name, total, empty, pct))
        grand_total += total
        grand_empty += empty

    results.sort(key=lambda r: r[3], reverse=True)

    print(f"{'Repositório':<55} {'Total .tf':>10} {'Vazios':>8} {'%':>7}")
    print("-" * 82)
    for repo_name, total, empty, pct in results:
        print(f"{repo_name:<55} {total:>10} {empty:>8} {pct:>6.2f}%")

    print("-" * 82)
    grand_pct = (grand_empty / grand_total) * 100 if grand_total else 0
    print(f"{'TOTAL':<55} {grand_total:>10} {grand_empty:>8} {grand_pct:>6.2f}%")
    if grand_skipped:
        print(f"\nRegistros ignorados (arquivo não é .tf): {grand_skipped}")


if __name__ == "__main__":
    main()
