#!/usr/bin/env python3
"""Stage 04: tokens and cost per call and in total (sync vs. batch), actual or estimated."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.llm_analysis import batch  # noqa: E402
from src.llm_analysis.classification import (  # noqa: E402
    ResultConflictError,
    RunStats,
    experiment_from_config,
    new_record,
    plan_tasks,
)
from src.llm_analysis.costs import (  # noqa: E402
    COST_CSV_FIELDS,
    actual_cost_rows,
    estimate_cost_rows,
    load_pricing,
    summarize,
)
from src.llm_analysis.evaluation import load_experiment_results  # noqa: E402
from src.llm_analysis.llm_client import LLMConfigError  # noqa: E402
from src.llm_analysis.schemas import INPUT_MODES, PreparedCommit  # noqa: E402
from src.llm_analysis.utils import (  # noqa: E402
    cli_or_config_path,
    load_config,
    read_jsonl,
    setup_logging,
    utc_now_iso,
    write_csv,
    write_json,
)

logger = logging.getLogger("llm_analysis.costs")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Config file (default: src/llm_analysis/config.yaml)")
    parser.add_argument("--output-dir", help="Override output.directory")
    parser.add_argument("--experiment-id", help="Actual costs of this experiment instead of the configured one")
    parser.add_argument("--model", help="Override llm.model (e.g. openai/gpt-4.1-mini, to estimate another model)")
    parser.add_argument("--estimate", action="store_true",
                        help="Estimate the calls still to make (input tokens counted locally)")
    parser.add_argument("--all", action="store_true", help="With --estimate: every prepared commit, not only pending")
    parser.add_argument("--limit", type=int, help="With --estimate: only the first N prepared commits")
    parser.add_argument("--output-tokens", type=int,
                        help="With --estimate: output tokens per call (default: mean of calls already made)")
    return parser.parse_args(argv)


def print_summary(title: str, summary: dict) -> None:
    print(f"\n{title}")
    print(f"{'':20} {'chamadas':>9} {'tokens entrada':>15} {'tokens saída':>13} {'custo (US$)':>12} "
          f"{'se síncrono':>12} {'se batch':>10}")
    lines = [("TOTAL", summary["total"])] + sorted(summary["by_input_mode"].items()) + \
        [(f"via {k}", v) for k, v in sorted(summary["by_via"].items())]
    for name, s in lines:
        print(f"{name:20} {s['calls']:9} {s['input_tokens']:15,} {s['output_tokens']:13,} {s['cost_usd']:12.4f} "
              f"{s['cost_if_sync_usd']:12.4f} {s['cost_if_batch_usd']:10.4f}")


def mean_output_tokens(output_dir: Path) -> float | None:
    values = [r["output_tokens"] for mode in INPUT_MODES
              for path in [output_dir / f"classifications_{mode}.jsonl"] if path.exists()
              for _, r in read_jsonl(path) if r.get("output_tokens")]
    return sum(values) / len(values) if values else None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    output_dir = cli_or_config_path(args.output_dir, cfg["output"]["directory"])
    setup_logging(output_dir / "pipeline.log")
    pricing = load_pricing(cfg.get("pricing") or {})

    try:
        experiment, settings, template = experiment_from_config(cfg, model=args.model)
    except (LLMConfigError, ValueError, KeyError) as e:
        logger.error("Invalid configuration: %s", e)
        return 2
    experiment_id = args.experiment_id or experiment.experiment_id
    report_dir = output_dir / "reports" / experiment_id

    try:
        if args.estimate:
            return estimate(args, cfg, output_dir, report_dir, experiment, settings, template, pricing)
        results = load_experiment_results(output_dir, experiment_id)
        rows = actual_cost_rows((r for mode in results.values() for r in mode.values()), pricing)
    except KeyError as e:
        logger.error("%s", e.args[0])
        return 2
    if not rows:
        logger.error("No calls of experiment %s in %s", experiment_id, output_dir)
        return 1
    summary = {"created_at": utc_now_iso(), "experiment_id": experiment_id, "kind": "actual",
               "prices_as_of": cfg["pricing"].get("as_of"), "prices_source": cfg["pricing"].get("source"), **summarize(rows)}
    write_csv(report_dir / "costs_per_call.csv", rows, COST_CSV_FIELDS)
    write_json(report_dir / "costs_summary.json", summary)
    print_summary(f"Custo real, experimento {experiment_id} ({rows[0]['model']})", summary)
    print(f"\nPor chamada: {report_dir / 'costs_per_call.csv'}")
    return 0


def estimate(args, cfg, output_dir, report_dir, experiment, settings, template, pricing) -> int:
    import litellm

    prepared_path = output_dir / cfg["output"]["prepared_commits_file"]
    commits = [PreparedCommit.from_dict(r) for _, r in read_jsonl(prepared_path)]
    if args.limit is not None:
        commits = commits[:args.limit]
    modes = list(cfg["experiment"]["input_modes"])
    if args.all:
        tasks = [(c, m) for c in commits for m in modes]
    else:
        pending = batch.pending_record_ids(output_dir, experiment.experiment_id)
        try:
            tasks = plan_tasks(commits, modes, experiment, output_dir, RunStats(), exclude_record_ids=pending)
        except ResultConflictError as e:
            logger.error("%s", e)
            return 1

    known = mean_output_tokens(output_dir)
    output_tokens = args.output_tokens or round(known or cfg["pricing"].get("estimate_output_tokens", 150))
    model_name = settings.model.split("/", 1)[-1]

    def requests():
        for commit, mode in tasks:
            record, user_prompt = new_record(commit, mode, experiment, template, store_prompt=False)
            base = {k: record[k] for k in ("record_id", "repo", "commit_hash", "input_mode")} | {"timestamp": None}
            yield base, [{"role": "system", "content": template.system}, {"role": "user", "content": user_prompt}]

    rows = estimate_cost_rows(requests(), settings.model, pricing, output_tokens,
                              lambda messages: litellm.token_counter(model=model_name, messages=messages))
    summary = {
        "created_at": utc_now_iso(), "experiment_id": experiment.experiment_id, "kind": "estimate",
        "model": settings.model, "scope": "all prepared commits" if args.all else "calls not made yet",
        "assumed_output_tokens_per_call": output_tokens,
        "output_tokens_source": "--output-tokens" if args.output_tokens else (
            "mean of calls already made" if known else "pricing.estimate_output_tokens"),
        "note": "Input tokens counted locally (LiteLLM token_counter, ~0.5% off the API count). For reasoning "
                "models the real output (with reasoning tokens) can be much larger.",
        "prices_as_of": cfg["pricing"].get("as_of"), "prices_source": cfg["pricing"].get("source"),
        **summarize(rows),
    }
    write_csv(report_dir / "cost_estimate_per_call.csv", rows, COST_CSV_FIELDS)
    write_json(report_dir / "cost_estimate.json", summary)
    print_summary(f"Estimativa ({summary['scope']}), {settings.model}, {output_tokens} tokens de saída por chamada",
                  summary)
    print(f"\nPor chamada: {report_dir / 'cost_estimate_per_call.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
