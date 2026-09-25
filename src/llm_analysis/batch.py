"""Stage 02 through LiteLLM's batch API: submit, status, collect, cancel."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.llm_analysis.classification import (
    Experiment,
    ResultConflictError,
    RunStats,
    append_records,
    new_record,
    plan_tasks,
    record_with_error,
    record_with_response,
)
from src.llm_analysis.llm_client import LiteLLMClient, chat_completion_to_response
from src.llm_analysis.prompts import PromptTemplate
from src.llm_analysis.schemas import PreparedCommit
from src.llm_analysis.utils import read_jsonl, utc_now_iso, write_json

logger = logging.getLogger(__name__)

BATCH_ENDPOINT = "/v1/chat/completions"
COMPLETION_WINDOW = "24h"
TERMINAL_STATUSES = frozenset({"completed", "expired", "failed", "cancelled"})
HTTP_OK = 200
BYTES_PER_MB = 1_000_000


class BatchError(RuntimeError):
    """The batch path cannot be used (wrong provider) or a batch operation failed."""


@dataclass(frozen=True)
class BatchLimits:
    max_requests_per_batch: int
    max_file_bytes: int
    poll_interval_seconds: float

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> BatchLimits:
        return cls(
            max_requests_per_batch=int(cfg.get("max_requests_per_batch", 50_000)),
            max_file_bytes=int(float(cfg.get("max_file_mb", 190)) * BYTES_PER_MB),
            poll_interval_seconds=float(cfg.get("poll_interval_seconds", 60)),
        )


def batch_dir(output_dir: Path) -> Path:
    return output_dir / "batches"


def load_states(output_dir: Path, experiment_id: str | None = None) -> list[dict[str, Any]]:
    folder = batch_dir(output_dir)
    if not folder.is_dir():
        return []
    states = []
    for path in sorted(folder.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        if experiment_id is None or state.get("experiment_id") == experiment_id:
            states.append(state)
    return sorted(states, key=lambda s: s["created_at"])


def save_state(output_dir: Path, state: dict[str, Any]) -> None:
    write_json(batch_dir(output_dir) / f"{state['batch_id']}.json", state)


def pending_record_ids(output_dir: Path, experiment_id: str) -> set[str]:
    """record_ids submitted in batches whose results have not been collected yet."""
    return {rid for s in load_states(output_dir, experiment_id) if not s.get("collected") for rid in s["requests"]}


# Vendors whose batch creation LiteLLM supports (anthropic: pending LiteLLM PR #42576).
BATCH_VENDORS = frozenset({"openai", "azure", "vertex_ai", "bedrock", "hosted_vllm"})


class BatchAPI:
    """LiteLLM files/batches calls for one vendor."""

    def __init__(self, vendor: str):
        import litellm

        self._litellm = litellm
        self.vendor = vendor

    def upload(self, path: Path) -> str:
        with open(path, "rb") as f:
            return self._litellm.create_file(file=f, purpose="batch", custom_llm_provider=self.vendor).id

    def create(self, input_file_id: str, metadata: dict[str, str]) -> Any:
        return self._litellm.create_batch(completion_window=COMPLETION_WINDOW, endpoint=BATCH_ENDPOINT,
                                          input_file_id=input_file_id, custom_llm_provider=self.vendor,
                                          metadata=metadata)

    def retrieve(self, batch_id: str) -> Any:
        return self._litellm.retrieve_batch(batch_id=batch_id, custom_llm_provider=self.vendor)

    def content(self, file_id: str) -> str:
        response = self._litellm.file_content(file_id=file_id, custom_llm_provider=self.vendor)
        return response.text if hasattr(response, "text") else response.content.decode("utf-8")

    def cancel(self, batch_id: str) -> None:
        self._litellm.cancel_batch(batch_id=batch_id, custom_llm_provider=self.vendor)


def batch_api_for(client: Any) -> BatchAPI:
    """BatchAPI for the configured model; fails if its vendor has no LiteLLM batch support."""
    if not isinstance(client, LiteLLMClient):
        raise BatchError(f"--batch needs llm.provider: litellm (configured: {client.settings.provider!r})")
    if client.vendor not in BATCH_VENDORS:
        raise BatchError(f"LiteLLM has no batch support for {client.vendor!r} (supported: {sorted(BATCH_VENDORS)})")
    client.check_supported_params()
    return BatchAPI(client.vendor)


def _chunks(lines: list[tuple[str, str, dict[str, Any]]], limits: BatchLimits) -> list[list]:
    """Split requests to respect the batch count and size limits."""
    chunks: list[list] = []
    current: list = []
    size = 0
    for item in lines:
        line_bytes = len(item[1].encode("utf-8")) + 1
        if current and (len(current) >= limits.max_requests_per_batch or size + line_bytes > limits.max_file_bytes):
            chunks.append(current)
            current, size = [], 0
        current.append(item)
        size += line_bytes
    if current:
        chunks.append(current)
    return chunks


def submit(
    client: LiteLLMClient,
    api: BatchAPI,
    commits: list[PreparedCommit],
    input_modes: list[str],
    experiment: Experiment,
    template: PromptTemplate,
    output_dir: Path,
    limits: BatchLimits,
    force: bool = False,
    retry_parse_failures: bool = False,
) -> tuple[RunStats, list[dict[str, Any]]]:
    """Submit every request not yet done or pending."""
    stats = RunStats()
    pending = pending_record_ids(output_dir, experiment.experiment_id)
    tasks = plan_tasks(commits, input_modes, experiment, output_dir, stats, force, retry_parse_failures, pending)
    logger.info("Experiment %s: %d requests to submit, %d already done, %d pending in batches",
                experiment.experiment_id, stats.planned, stats.skipped_existing, stats.skipped_pending_batch)

    lines = []
    for commit, mode in tasks:
        record, user_prompt = new_record(commit, mode, experiment, template, store_prompt=False)
        request = {
            "custom_id": record["record_id"],
            "method": "POST",
            "url": BATCH_ENDPOINT,
            "body": client.request_body(template.system, user_prompt),
        }
        meta = {k: record[k] for k in ("repo", "commit_hash", "input_mode", "prepared_commit_sha256",
                                       "user_prompt_sha256")}
        lines.append((record["record_id"], json.dumps(request, ensure_ascii=False), meta))

    folder = batch_dir(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    states = []
    for n, chunk in enumerate(_chunks(lines, limits), start=1):
        staging = folder / f"upload_{int(time.time())}_{n}.jsonl"
        staging.write_text("".join(line + "\n" for _, line, _ in chunk), encoding="utf-8")
        input_file_id = api.upload(staging)
        batch = api.create(input_file_id, {"experiment_id": experiment.experiment_id, "part": str(n)})
        input_path = folder / f"{batch.id}_input.jsonl"
        staging.rename(input_path)
        state = {
            "batch_id": batch.id,
            "experiment_id": experiment.experiment_id,
            "provider": experiment.provider,
            "model": experiment.model,
            "created_at": utc_now_iso(),
            "vendor": api.vendor,
            "input_file_id": input_file_id,
            "input_file": str(input_path),
            "n_requests": len(chunk),
            "status": batch.status,
            "request_counts": None,
            "output_file_id": None,
            "error_file_id": None,
            "errors": None,
            "collected": False,
            "collected_at": None,
            "results_written": 0,
            "requests": {rid: meta for rid, _, meta in chunk},
        }
        save_state(output_dir, state)
        states.append(state)
        logger.info("Submitted batch %s with %d requests", batch.id, len(chunk))
    return stats, states


def refresh(api: BatchAPI, output_dir: Path, state: dict[str, Any]) -> dict[str, Any]:
    batch = api.retrieve(state["batch_id"])
    counts = getattr(batch, "request_counts", None)
    errors = getattr(batch, "errors", None)
    state.update(
        status=batch.status,
        request_counts=counts.model_dump() if hasattr(counts, "model_dump") else counts,
        output_file_id=batch.output_file_id,
        error_file_id=batch.error_file_id,
        errors=errors.model_dump() if hasattr(errors, "model_dump") else errors,
    )
    save_state(output_dir, state)
    return state


def status(api: BatchAPI, output_dir: Path, experiment_id: str) -> list[dict[str, Any]]:
    states = load_states(output_dir, experiment_id)
    for state in states:
        if not state.get("collected"):
            refresh(api, output_dir, state)
    return states


def cancel(api: BatchAPI, output_dir: Path, experiment_id: str) -> list[str]:
    cancelled = []
    for state in load_states(output_dir, experiment_id):
        if state.get("collected") or state.get("status") in TERMINAL_STATUSES:
            continue
        api.cancel(state["batch_id"])
        refresh(api, output_dir, state)
        cancelled.append(state["batch_id"])
    return cancelled


def _download(api: BatchAPI, file_id: str | None, path: Path) -> list[dict[str, Any]]:
    if not file_id:
        return []
    path.write_text(api.content(file_id), encoding="utf-8")
    return [line for _, line in read_jsonl(path)]


def _line_to_record(line: dict[str, Any], state: dict[str, Any], commits: dict[tuple[str, str], PreparedCommit],
                    experiment: Experiment, template: PromptTemplate, store_prompts: bool) -> dict[str, Any] | None:
    meta = state["requests"].get(line.get("custom_id"))
    if meta is None:
        logger.warning("Batch %s returned unknown custom_id %r", state["batch_id"], line.get("custom_id"))
        return None
    commit = commits.get((meta["repo"], meta["commit_hash"]))
    if commit is None or commit.content_sha256 != meta["prepared_commit_sha256"]:
        raise ResultConflictError(
            f"Batch {state['batch_id']} was built from a different commits_prepared.jsonl (commit "
            f"{meta['commit_hash'][:10]} missing or changed). Restore the prepared file used at submission.")
    record, user_prompt = new_record(commit, meta["input_mode"], experiment, template, store_prompts)
    if record["user_prompt_sha256"] != meta["user_prompt_sha256"]:
        raise ResultConflictError(f"The prompt of {meta['commit_hash'][:10]} [{meta['input_mode']}] no longer "
                                  "matches what was submitted in the batch.")
    record["batch_id"] = state["batch_id"]

    response = line.get("response") or {}
    status_code = response.get("status_code")
    if status_code == HTTP_OK:
        result = chat_completion_to_response(response.get("body") or {})
        result.attempts = 1
        return record_with_response(record, result)

    error = line.get("error") or (response.get("body") or {}).get("error") or {}
    message = error.get("message") if isinstance(error, dict) else str(error)
    error_type = f"HTTP {status_code}" if status_code else (error.get("code") if isinstance(error, dict) else None)
    return record_with_error(record, message or "batch request failed", error_type or "BatchRequestError", 1)


def collect(
    api: BatchAPI,
    commits: list[PreparedCommit],
    experiment: Experiment,
    template: PromptTemplate,
    output_dir: Path,
    store_prompts: bool = True,
    wait: bool = False,
    poll_interval_seconds: float = 60,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, int]:
    """Save the results of finished batches."""
    by_key = {(c.repo, c.commit_hash): c for c in commits}
    totals = {"batches_collected": 0, "records_written": 0, "api_failures": 0, "still_running": 0}
    while True:
        open_states = [s for s in load_states(output_dir, experiment.experiment_id) if not s.get("collected")]
        running = 0
        for state in open_states:
            refresh(api, output_dir, state)
            if state["status"] not in TERMINAL_STATUSES:
                running += 1
                continue
            folder = batch_dir(output_dir)
            lines = (_download(api, state["output_file_id"], folder / f"{state['batch_id']}_output.jsonl")
                     + _download(api, state["error_file_id"], folder / f"{state['batch_id']}_errors.jsonl"))
            records = [r for r in (_line_to_record(line, state, by_key, experiment, template, store_prompts)
                                   for line in lines) if r is not None]
            written = append_records(output_dir, records)
            failures = sum(1 for r in records if r.get("error"))
            missing = state["n_requests"] - len(records)
            if state["status"] != "completed" or missing:
                logger.warning("Batch %s ended as %s: %d results, %d requests without a result (resubmit them). %s",
                               state["batch_id"], state["status"], len(records), missing, state.get("errors") or "")
            state.update(collected=True, collected_at=utc_now_iso(), results_written=written)
            save_state(output_dir, state)
            totals["batches_collected"] += 1
            totals["records_written"] += written
            totals["api_failures"] += failures
            logger.info("Collected batch %s: %d records (%d failed requests)", state["batch_id"], written, failures)
        totals["still_running"] = running
        if not wait or running == 0:
            return totals
        logger.info("%d batch(es) still running; checking again in %ss", running, poll_interval_seconds)
        sleep(poll_interval_seconds)


def describe(state: dict[str, Any]) -> str:
    counts = state.get("request_counts") or {}
    progress = f"{counts.get('completed', 0)}/{counts.get('total', state['n_requests'])} done" if counts else ""
    collected = "collected" if state.get("collected") else "not collected"
    return f"{state['batch_id']}  {state['status']:<11} {state['n_requests']:>6} requests  {progress}  ({collected})"

