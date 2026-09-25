#!/usr/bin/env python3
"""Stage 03: summarize one experiment."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.llm_analysis.classification import experiment_from_config  # noqa: E402
from src.llm_analysis.evaluation import (  # noqa: E402
    BUGFIX_RATE,
    BUGFIX_RATE_DEFINITION,
    DISAGREEMENT_FIELDS,
    SCOPE_NOTES,
    SUMMARY_CSV_FIELDS,
    classification_count_rows,
    cross_mode_comparison,
    disagreement_rows,
    error_records,
    evaluate_against_gold,
    load_experiment_results,
    load_gold_labels,
    match_gold,
    summarize_mode,
    summary_csv_rows,
)
from src.llm_analysis.llm_client import LLMConfigError  # noqa: E402
from src.llm_analysis.schemas import CANDIDATE_VALIDATION, INPUT_MODES  # noqa: E402
from src.llm_analysis.utils import (  # noqa: E402
    cli_or_config_path,
    load_config,
    read_jsonl,
    setup_logging,
    utc_now_iso,
    write_csv,
    write_json,
    write_jsonl,
)

logger = logging.getLogger("llm_analysis.summarize")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Config file (default: src/llm_analysis/config.yaml)")
    parser.add_argument("--output-dir", help="Override output.directory")
    parser.add_argument("--experiment-id", help="Summarize this experiment instead of the configured one")
    parser.add_argument("--provider", help="Override llm.provider when deriving the experiment id")
    parser.add_argument("--model", help="Override llm.model when deriving the experiment id")
    parser.add_argument("--gold-labels", help="Override evaluation.gold_labels")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    output_dir = cli_or_config_path(args.output_dir, cfg["output"]["directory"])
    setup_logging(output_dir / "pipeline.log")

    experiment_mode = cfg["experiment"]["mode"]
    experiment_id = args.experiment_id
    if experiment_id is None:
        try:
            experiment, _, _ = experiment_from_config(cfg, args.provider, args.model)
        except (LLMConfigError, ValueError, KeyError) as e:
            logger.error("Invalid configuration: %s", e)
            return 2
        experiment_id = experiment.experiment_id

    results = load_experiment_results(output_dir, experiment_id)
    records = [r for mode in results.values() for r in mode.values()]
    if not records:
        logger.error("No classifications of experiment %s in %s", experiment_id, output_dir)
        return 1
    experiment_mode = records[0].get("experiment_mode", experiment_mode)
    providers = sorted({f"{r['provider']}/{r['model']}" for r in records})
    if any(r["provider"] == "mock" for r in records):
        logger.warning("These results come from the MOCK provider: the labels are fake.")

    messages = {}
    prepared_path = output_dir / cfg["output"]["prepared_commits_file"]
    if prepared_path.exists():
        messages = {(r["repo"], r["commit_hash"]): r.get("message", "") for _, r in read_jsonl(prepared_path)}

    by_mode = {mode: summarize_mode(results[mode]) for mode in INPUT_MODES if results[mode]}
    summary = {
        "created_at": utc_now_iso(),
        "experiment_id": experiment_id,
        "experiment_mode": experiment_mode,
        "scope_note": SCOPE_NOTES.get(experiment_mode, ""),
        "provider_model": providers,
        "prompt_versions": sorted({r["prompt_version"] for r in records}),
        "unique_commits": len({(r["repo"], r["commit_hash"]) for r in records}),
        "by_input_mode": by_mode,
        "cross_mode": cross_mode_comparison(results),
        "confidence_note": "confidence is self-reported by the model; it is not a calibrated probability.",
    }
    if experiment_mode == CANDIDATE_VALIDATION:
        summary[BUGFIX_RATE] = {
            "definition": BUGFIX_RATE_DEFINITION,
            "by_input_mode": {mode: data[BUGFIX_RATE] for mode, data in by_mode.items()},
        }

    gold_setting = args.gold_labels or cfg.get("evaluation", {}).get("gold_labels")
    gold = None
    if gold_setting:
        gold_path = cli_or_config_path(args.gold_labels, gold_setting)
        eval_cfg = cfg.get("evaluation", {})
        gold = load_gold_labels(gold_path, eval_cfg.get("gold_label_column"), eval_cfg.get("gold_commit_column"))
        summary["gold_labels_file"] = str(gold_path)
        summary["gold_evaluation"] = evaluate_against_gold(results, gold, experiment_mode)
        if gold.conflicting_commits:
            logger.warning("%d commits have contradictory gold labels and were excluded",
                           len(gold.conflicting_commits))
    else:
        summary["gold_evaluation"] = None
        summary["gold_evaluation_note"] = "No gold labels supplied: supervised metrics were not computed."

    report_dir = output_dir / "reports" / experiment_id
    write_json(report_dir / "summary.json", summary)
    write_csv(report_dir / "summary.csv", summary_csv_rows(summary), SUMMARY_CSV_FIELDS)
    write_csv(report_dir / "classification_counts.csv", classification_count_rows(results),
              ["input_mode", "label", "count", "fraction"])
    n_errors = write_jsonl(report_dir / "errors.jsonl", error_records(results))
    gold_by_hash = match_gold(gold.labels, {k[1] for m in results.values() for k in m}) if gold else None
    disagreements = disagreement_rows(results, messages, gold_by_hash)
    write_csv(report_dir / "disagreements.csv", disagreements, DISAGREEMENT_FIELDS)

    logger.info("Experiment %s: %d commits, %d failures, %d disagreements. Reports in %s", experiment_id,
                summary["unique_commits"], n_errors, len(disagreements), report_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
