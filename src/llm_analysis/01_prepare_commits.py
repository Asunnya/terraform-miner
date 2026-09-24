#!/usr/bin/env python3
"""
Stage 01: aggregate the miner's per-file JSONL records into one record per commit.

Writes <output.directory>/commits_prepared.jsonl and prepare_stats.json. The miner files are only read.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.llm_analysis.preparation import (  # noqa: E402
    PreparationStats,
    aggregate_commits,
    find_input_files,
    iter_miner_records,
    keyword_match_fraction,
    sample_commits,
)
from src.llm_analysis.schemas import EXPERIMENT_MODES, FULL_CLASSIFICATION  # noqa: E402
from src.llm_analysis.utils import (  # noqa: E402
    cli_or_config_path,
    load_config,
    resolve_path,
    setup_logging,
    sha256_file,
    utc_now_iso,
    write_json,
    write_jsonl,
)

logger = logging.getLogger("llm_analysis.prepare")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Config file (default: src/llm_analysis/config.yaml)")
    parser.add_argument("--input-glob", help="Override input.dataset_glob")
    parser.add_argument("--output-dir", help="Override output.directory")
    parser.add_argument("--sample-size", type=int, help="Override preparation.sample_size")
    parser.add_argument("--seed", type=int, help="Override preparation.seed")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    output_dir = cli_or_config_path(args.output_dir, cfg["output"]["directory"])
    setup_logging(output_dir / "pipeline.log")

    experiment_mode = cfg["experiment"]["mode"]
    if experiment_mode not in EXPERIMENT_MODES:
        logger.error("experiment.mode must be one of %s", EXPERIMENT_MODES)
        return 2

    pattern = cli_or_config_path(args.input_glob, cfg["input"]["dataset_glob"])
    input_files = [p for p in find_input_files(pattern) if output_dir not in p.parents]
    if not input_files:
        logger.error("No input files match %s", pattern)
        return 1
    logger.info("Reading %d miner files matching %s", len(input_files), pattern)

    stats = PreparationStats(input_files=len(input_files))
    commits = aggregate_commits(iter_miner_records(input_files), stats)

    prep_cfg = cfg.get("preparation", {})
    sample_size = args.sample_size if args.sample_size is not None else prep_cfg.get("sample_size")
    seed = args.seed if args.seed is not None else prep_cfg.get("seed", 0)
    selected = sample_commits(commits, sample_size, seed)

    miner_keywords: list[str] = []
    miner_config = cfg["input"].get("miner_config")
    if miner_config:
        try:
            miner_keywords = load_config(resolve_path(miner_config)).get("keywords", [])
        except FileNotFoundError:
            logger.warning("Miner config %s not found; keyword statistics skipped", miner_config)
    kw_fraction = keyword_match_fraction(selected, miner_keywords)
    threshold = cfg["experiment"].get("keyword_prefilter_warning_threshold", 1.0)
    if experiment_mode == FULL_CLASSIFICATION and kw_fraction is not None and kw_fraction >= threshold:
        logger.warning(
            "experiment.mode is full_classification, but %.1f%% of the commit messages contain a miner "
            "bug-fix keyword: this corpus looks keyword-prefiltered. Use candidate_validation for it, or "
            "supply an unfiltered commit collection.", 100 * kw_fraction)

    prepared_path = output_dir / cfg["output"]["prepared_commits_file"]
    write_jsonl(prepared_path, (c.to_dict() for c in selected))

    write_json(output_dir / "prepare_stats.json", {
        "created_at": utc_now_iso(),
        "experiment_mode": experiment_mode,
        "input_glob": str(pattern),
        "input_files": [{"path": str(p), "sha256": sha256_file(p)} for p in input_files],
        "aggregation": stats.to_dict(),
        "sample_size": sample_size,
        "seed": seed,
        "commits_written": len(selected),
        "commits_without_any_patch": sum(1 for c in selected if not c.diffs),
        "multi_file_commits": sum(1 for c in selected if len(c.files) > 1),
        "miner_keywords": miner_keywords,
        "share_of_messages_matching_miner_keywords": kw_fraction,
        "output": str(prepared_path),
    })
    logger.info("Wrote %d commits (from %d records) to %s", len(selected), stats.records_read, prepared_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
