#!/usr/bin/env python3
"""
Stage 02: classify each prepared commit as BUG_FIX / NON_BUG_FIX, once per input mode.

Results are appended to <output.directory>/classifications_<input_mode>.jsonl as soon as each request
finishes. Rerunning resumes: requests already completed for the same experiment are skipped.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.llm_analysis.classification import (  # noqa: E402
    Experiment,
    ResultConflictError,
    build_prompt_input,
    experiment_from_config,
    run_classification,
)
from src.llm_analysis.evaluation import extract_commit_hash  # noqa: E402
from src.llm_analysis.llm_client import LLMConfigError, create_client  # noqa: E402
from src.llm_analysis.prompts import PromptTemplate  # noqa: E402
from src.llm_analysis.schemas import INPUT_MODES, PreparedCommit  # noqa: E402
from src.llm_analysis.utils import (  # noqa: E402
    cli_or_config_path,
    load_config,
    load_env_file,
    read_jsonl,
    setup_logging,
    utc_now_iso,
    write_json,
)

logger = logging.getLogger("llm_analysis.classify")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Config file (default: src/llm_analysis/config.yaml)")
    parser.add_argument("--output-dir", help="Override output.directory")
    parser.add_argument("--input-modes", nargs="+", choices=INPUT_MODES, help="Override experiment.input_modes")
    parser.add_argument("--provider", help="Override llm.provider (e.g. mock for an offline dry run)")
    parser.add_argument("--model", help="Override llm.model")
    parser.add_argument("--workers", type=int, help="Override llm.workers")
    parser.add_argument("--limit", type=int, help="Only the first N prepared commits (in file order)")
    parser.add_argument("--commits-file",
                        help="Only commits listed in this file (one SHA or GitHub commit URL per line, or a "
                             "CSV whose first column holds them), e.g. the commits of a gold-label sample")
    parser.add_argument("--force", action="store_true", help="Reclassify commits that already have results")
    parser.add_argument("--retry-parse-failures", action="store_true",
                        help="Also redo requests whose response could not be parsed")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build the prompts and print truncation statistics, without calling the LLM")
    return parser.parse_args(argv)


def load_commit_filter(path: Path) -> set[str]:
    hashes = set()
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            first_cell = line.split(",")[0].strip().strip('"')
            commit_hash = extract_commit_hash(line) or extract_commit_hash(first_cell)
            if commit_hash:
                hashes.add(commit_hash)
    return hashes


def select_commits(commits: list[PreparedCommit], args: argparse.Namespace) -> list[PreparedCommit]:
    if args.commits_file:
        wanted = load_commit_filter(cli_or_config_path(args.commits_file, None))
        commits = [c for c in commits
                   if any(c.commit_hash.lower().startswith(h) for h in wanted)]
        logger.info("%d prepared commits match %s (%d hashes)", len(commits), args.commits_file, len(wanted))
    if args.limit is not None:
        commits = commits[:args.limit]
    return commits


def dry_run(commits: list[PreparedCommit], input_modes: list[str], experiment: Experiment,
            template: PromptTemplate) -> None:
    for mode in input_modes:
        truncated = 0
        for commit in commits:
            prompt = build_prompt_input(commit, mode, template, experiment.limits)
            truncated += bool(prompt.rendered and prompt.rendered.diff_truncated)
        logger.info("[dry run] %s: %d prompts, %d with truncated diffs", mode, len(commits), truncated)
    if commits:
        sample = build_prompt_input(commits[0], input_modes[-1], template, experiment.limits)
        print(f"===== SYSTEM PROMPT ({template.version}) =====\n{template.system}")
        print(f"===== USER PROMPT ({input_modes[-1]}, {commits[0].repo} {commits[0].commit_hash}) =====")
        print(sample.user_prompt)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    output_dir = cli_or_config_path(args.output_dir, cfg["output"]["directory"])
    setup_logging(output_dir / "pipeline.log")

    workers = args.workers if args.workers is not None else int(cfg["llm"].get("workers", 1))
    input_modes = args.input_modes or list(cfg["experiment"]["input_modes"])
    try:
        experiment, settings, template = experiment_from_config(cfg, args.provider, args.model)
    except (LLMConfigError, ValueError, KeyError) as e:
        logger.error("Invalid configuration: %s", e)
        return 2

    prepared_path = output_dir / cfg["output"]["prepared_commits_file"]
    if not prepared_path.exists():
        logger.error("%s not found: run 01_prepare_commits.py first", prepared_path)
        return 1
    commits = select_commits([PreparedCommit.from_dict(r) for _, r in read_jsonl(prepared_path)], args)
    logger.info("Experiment %s (%s, %s/%s, prompt %s): %d commits x %s", experiment.experiment_id,
                experiment.experiment_mode, settings.provider, settings.model, template.version,
                len(commits), input_modes)

    if args.dry_run:
        dry_run(commits, input_modes, experiment, template)
        return 0
    if settings.provider == "mock":
        logger.warning("Provider 'mock' returns FAKE labels: use it only to test the pipeline.")

    load_env_file()
    try:
        client = create_client(settings)
    except LLMConfigError as e:
        logger.error("%s", e)
        return 2

    write_json(output_dir / "experiments" / f"{experiment.experiment_id}.json", {
        "experiment_id": experiment.experiment_id,
        "experiment": experiment.to_dict(),
        "system_prompt": template.system,
        "prompt_template": {k: v for k, v in template.__dict__.items() if k != "system"},
        "prepared_commits_file": str(prepared_path),
    })
    started = utc_now_iso()
    try:
        stats = run_classification(
            client, commits, input_modes, experiment, template, output_dir,
            force=args.force, retry_parse_failures=args.retry_parse_failures, workers=workers,
            store_prompts=bool(cfg["output"].get("store_prompts", True)),
        )
    except ResultConflictError as e:
        logger.error("%s", e)
        return 1

    run_log = {
        "started_at": started,
        "finished_at": utc_now_iso(),
        "experiment_id": experiment.experiment_id,
        "argv": sys.argv[1:] if argv is None else argv,
        "input_modes": input_modes,
        "commits_selected": len(commits),
        "python": platform.python_version(),
        **stats.to_dict(),
    }
    with open(output_dir / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(run_log) + "\n")
    logger.info("Done: %s", stats.to_dict())
    if stats.api_failures:
        logger.warning("%d requests failed; rerun the same command to retry them.", stats.api_failures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
