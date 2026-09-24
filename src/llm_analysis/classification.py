"""
Stage 02 logic: classify prepared commits, persisting each result immediately and resuming safely.

Identity
- experiment_id: hash of everything that can change a model output other than the commit itself
  (experiment mode, provider, model, sampling settings, prompt version + template hash, context
  limits, diff format version). The three input modes share one experiment_id.
- record_id: hash of (experiment_id, repo, commit_hash, input_mode). A record_id is classified once;
  reruns skip it unless --force.

Files are append-only: a retry or --force appends a new line and the LAST line of a record_id wins.
Records whose API call failed (`error` set) are retried on the next run; parse failures are kept
(the model did answer, its raw output is stored) unless --retry-parse-failures.
"""
from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from src.llm_analysis.diff_context import DIFF_FORMAT_VERSION, ContextLimits, RenderedDiff, render_diffs
from src.llm_analysis.llm_client import LLMClient, LLMError, LLMSettings
from src.llm_analysis.prompts import PromptTemplate, build_user_prompt, get_prompt
from src.llm_analysis.schemas import (
    EXPERIMENT_MODES,
    INPUT_MODES,
    MESSAGE_ONLY,
    RESULT_FIELDS,
    PreparedCommit,
    parse_llm_response,
)
from src.llm_analysis.utils import read_jsonl, sha256_text, stable_hash, utc_now_iso

logger = logging.getLogger(__name__)

EXPERIMENT_ID_LENGTH = 16
RECORD_ID_LENGTH = 24


@dataclass(frozen=True)
class Experiment:
    experiment_mode: str
    provider: str
    model: str
    temperature: float | None
    max_output_tokens: int
    seed: int | None
    json_mode: bool
    extra_params: str
    prompt_version: str
    prompt_sha256: str
    limits: ContextLimits
    diff_format_version: str = DIFF_FORMAT_VERSION

    @classmethod
    def build(cls, experiment_mode: str, settings: LLMSettings, template: PromptTemplate,
              limits: ContextLimits) -> Experiment:
        if experiment_mode not in EXPERIMENT_MODES:
            raise ValueError(f"experiment.mode must be one of {EXPERIMENT_MODES}, got {experiment_mode!r}")
        return cls(
            experiment_mode=experiment_mode,
            provider=settings.provider,
            model=settings.model,
            temperature=settings.temperature,
            max_output_tokens=settings.max_output_tokens,
            seed=settings.seed,
            json_mode=settings.json_mode,
            extra_params=json.dumps(settings.extra_params, sort_keys=True),
            prompt_version=template.version,
            prompt_sha256=template.sha256,
            limits=limits,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def experiment_id(self) -> str:
        return stable_hash(self.to_dict())[:EXPERIMENT_ID_LENGTH]


def experiment_from_config(cfg: dict[str, Any], provider: str | None = None,
                           model: str | None = None) -> tuple[Experiment, LLMSettings, PromptTemplate]:
    """Build the experiment described by config.yaml (optionally overriding provider/model)."""
    llm_cfg = dict(cfg["llm"])
    if provider:
        llm_cfg["provider"] = provider
    if model:
        llm_cfg["model"] = model
    settings = LLMSettings.from_config(llm_cfg)
    template = get_prompt(cfg["classification"]["prompt_version"])
    limits = ContextLimits.from_config(cfg["context"])
    return Experiment.build(cfg["experiment"]["mode"], settings, template, limits), settings, template


def record_id(experiment_id: str, repo: str, commit_hash: str, input_mode: str) -> str:
    return stable_hash([experiment_id, repo, commit_hash, input_mode])[:RECORD_ID_LENGTH]


def result_path(output_dir: Path, input_mode: str) -> Path:
    return output_dir / f"classifications_{input_mode}.jsonl"


@dataclass
class PromptInput:
    user_prompt: str
    rendered: RenderedDiff | None
    original_diff_chars: int


def build_prompt_input(commit: PreparedCommit, input_mode: str, template: PromptTemplate,
                       limits: ContextLimits) -> PromptInput:
    rendered = render_diffs(commit.diffs, commit.files_without_patch, limits)
    submitted = None if input_mode == MESSAGE_ONLY else rendered
    return PromptInput(
        user_prompt=build_user_prompt(template, input_mode, commit.message, submitted),
        rendered=submitted,
        original_diff_chars=rendered.original_diff_chars,
    )


def classify_commit(client: LLMClient, commit: PreparedCommit, input_mode: str, experiment: Experiment,
                    template: PromptTemplate, store_prompt: bool = True) -> dict[str, Any]:
    """Classify one (commit, input_mode). Never raises for API/parse problems: they go in the record."""
    prompt = build_prompt_input(commit, input_mode, template, experiment.limits)
    rendered = prompt.rendered
    record: dict[str, Any] = dict.fromkeys(RESULT_FIELDS)
    record.update(
        record_id=record_id(experiment.experiment_id, commit.repo, commit.commit_hash, input_mode),
        experiment_id=experiment.experiment_id,
        experiment_mode=experiment.experiment_mode,
        repo=commit.repo,
        commit_hash=commit.commit_hash,
        input_mode=input_mode,
        provider=experiment.provider,
        model=experiment.model,
        temperature=experiment.temperature,
        max_output_tokens=experiment.max_output_tokens,
        prompt_version=experiment.prompt_version,
        prompt_sha256=experiment.prompt_sha256,
        diff_truncated=rendered.diff_truncated if rendered else False,
        original_diff_chars=prompt.original_diff_chars,
        submitted_diff_chars=rendered.submitted_diff_chars if rendered else 0,
        files_submitted=rendered.files_submitted if rendered else 0,
        files_omitted=len(rendered.files_omitted) if rendered else 0,
        prepared_commit_sha256=commit.content_sha256,
        user_prompt_sha256=sha256_text(prompt.user_prompt),
        user_prompt=prompt.user_prompt if store_prompt else None,
        parse_success=False,
    )

    try:
        response = client.classify(template.system, prompt.user_prompt)
    except LLMError as e:
        record.update(timestamp=utc_now_iso(), error=str(e), error_type=e.error_type, attempts=e.attempts)
        return record

    parsed = parse_llm_response(response.text)
    record.update(
        timestamp=utc_now_iso(),
        response_model=response.model,
        raw_response=response.text,
        attempts=response.attempts,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        stop_reason=response.stop_reason,
        latency_seconds=response.latency_seconds,
        parse_success=parsed.parse_success,
        parse_error=parsed.parse_error,
        parse_warnings=parsed.warnings,
    )
    if parsed.parse_success:
        record.update(
            label=parsed.label,
            confidence=parsed.confidence,
            reason=parsed.reason,
            message_evidence=parsed.message_evidence,
            diff_evidence=parsed.diff_evidence,
            uncertainty=parsed.uncertainty,
        )
    return record


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    """Latest record per record_id (later lines override earlier ones)."""
    if not path.exists():
        return {}
    return {r["record_id"]: r for _, r in read_jsonl(path) if r.get("record_id")}


def is_complete(record: dict[str, Any], retry_parse_failures: bool) -> bool:
    if record.get("error"):
        return False
    return bool(record.get("parse_success")) or not retry_parse_failures


class ResultConflictError(RuntimeError):
    """Existing results of this experiment were produced from different inputs or prompt text."""


def check_prompt_unchanged(existing: dict[str, dict[str, Any]], experiment: Experiment) -> None:
    edited = {r.get("prompt_sha256") for r in existing.values()
              if r.get("prompt_version") == experiment.prompt_version} - {experiment.prompt_sha256}
    if edited:
        raise ResultConflictError(
            f"Prompt {experiment.prompt_version!r} was edited after results were produced with it "
            f"(stored hash {sorted(edited)[0][:12]} != current {experiment.prompt_sha256[:12]}). "
            "Restore it and add the new text as a new prompt version in prompts.py."
        )


def check_not_stale(existing: dict[str, dict[str, Any]], commits: dict[tuple[str, str], PreparedCommit],
                    experiment: Experiment) -> None:
    stale = []
    for rec in existing.values():
        if rec.get("experiment_id") != experiment.experiment_id:
            continue
        commit = commits.get((rec.get("repo"), rec.get("commit_hash")))
        if commit is not None and rec.get("prepared_commit_sha256") != commit.content_sha256:
            stale.append(rec["record_id"])
    if stale:
        raise ResultConflictError(
            f"{len(stale)} existing classifications of experiment {experiment.experiment_id} were made from a "
            "different version of commits_prepared.jsonl (e.g. the dataset was re-mined). Use a new output "
            "directory, or --force to reclassify them."
        )


@dataclass
class RunStats:
    planned: int = 0
    skipped_existing: int = 0
    classified: int = 0
    parse_failures: int = 0
    api_failures: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def run_classification(
    client: LLMClient,
    commits: list[PreparedCommit],
    input_modes: Iterable[str],
    experiment: Experiment,
    template: PromptTemplate,
    output_dir: Path,
    force: bool = False,
    retry_parse_failures: bool = False,
    workers: int = 1,
    store_prompts: bool = True,
) -> RunStats:
    """
    Classify every (commit, input_mode) not yet completed. Tasks are ordered commit by commit (all modes
    of a commit before the next commit), so a partial run still covers the same commits in every mode.
    """
    input_modes = list(input_modes)
    for mode in input_modes:
        if mode not in INPUT_MODES:
            raise ValueError(f"Unknown input mode {mode!r}; valid: {INPUT_MODES}")
    output_dir.mkdir(parents=True, exist_ok=True)

    by_key = {(c.repo, c.commit_hash): c for c in commits}
    existing = {mode: load_results(result_path(output_dir, mode)) for mode in input_modes}
    for mode in input_modes:
        check_prompt_unchanged(existing[mode], experiment)
        if not force:
            check_not_stale(existing[mode], by_key, experiment)

    stats = RunStats()
    tasks: list[tuple[PreparedCommit, str]] = []
    for commit in commits:
        for mode in input_modes:
            rid = record_id(experiment.experiment_id, commit.repo, commit.commit_hash, mode)
            previous = existing[mode].get(rid)
            if previous is not None and not force and is_complete(previous, retry_parse_failures):
                stats.skipped_existing += 1
                continue
            tasks.append((commit, mode))
    stats.planned = len(tasks)
    logger.info("Experiment %s: %d requests to make, %d already done", experiment.experiment_id,
                stats.planned, stats.skipped_existing)

    handles = {mode: open(result_path(output_dir, mode), "a", encoding="utf-8") for mode in input_modes}
    lock = threading.Lock()

    def persist(record: dict[str, Any]) -> None:
        with lock:
            handle = handles[record["input_mode"]]
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            stats.classified += 1
            if record.get("error"):
                stats.api_failures += 1
                logger.warning("API failure %s %s [%s]: %s", record["repo"], record["commit_hash"][:10],
                               record["input_mode"], record["error"])
            elif not record.get("parse_success"):
                stats.parse_failures += 1
                logger.warning("Unparseable response %s %s [%s]: %s", record["repo"], record["commit_hash"][:10],
                               record["input_mode"], record["parse_error"])
            if stats.classified % 50 == 0 or stats.classified == stats.planned:
                logger.info("Progress: %d/%d", stats.classified, stats.planned)

    try:
        if workers <= 1:
            for commit, mode in tasks:
                persist(classify_commit(client, commit, mode, experiment, template, store_prompts))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(classify_commit, client, c, m, experiment, template, store_prompts)
                           for c, m in tasks]
                try:
                    for future in as_completed(futures):
                        persist(future.result())
                except BaseException:
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise
    finally:
        for handle in handles.values():
            handle.close()
    return stats
