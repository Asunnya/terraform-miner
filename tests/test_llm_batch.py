"""Batch via LiteLLM, com uma BatchAPI falsa (sem rede)."""
import json
from types import SimpleNamespace

import pytest

from src.llm_analysis import batch
from src.llm_analysis.classification import Experiment, ResultConflictError, load_results, result_path, run_classification
from src.llm_analysis.diff_context import ContextLimits
from src.llm_analysis.llm_client import LiteLLMClient, LLMConfigError, LLMSettings
from src.llm_analysis.prompts import get_prompt
from tests.test_llm_classification import VALID, FakeClient, make_commit

PROMPT = get_prompt("v1")
SETTINGS = LLMSettings(provider="litellm", model="openai/gpt-4.1-nano", temperature=0, seed=7, json_mode=True)
LIMITS = ContextLimits(max_diff_chars=1000, max_chars_per_file=500, max_files_per_commit=5, max_listed_file_names=10)
BATCH_LIMITS = batch.BatchLimits(max_requests_per_batch=50_000, max_file_bytes=10**9, poll_interval_seconds=0)
MODES = ["message_only", "diff_only"]


class FakeBatchAPI:
    """Imita batch.BatchAPI; `finish` simula o fornecedor processando um batch."""

    vendor = "openai"

    def __init__(self):
        self.uploads, self.batches, self.contents = {}, {}, {}

    def upload(self, path):
        file_id = f"file-{len(self.uploads)}"
        self.uploads[file_id] = [json.loads(line) for line in path.read_text().splitlines()]
        return file_id

    def create(self, input_file_id, metadata):
        batch_id = f"batch_{len(self.batches)}"
        self.batches[batch_id] = SimpleNamespace(id=batch_id, status="validating", input_file_id=input_file_id,
                                                 output_file_id=None, error_file_id=None, errors=None,
                                                 request_counts=None, metadata=metadata)
        return self.batches[batch_id]

    def retrieve(self, batch_id):
        return self.batches[batch_id]

    def cancel(self, batch_id):
        self.batches[batch_id].status = "cancelled"

    def content(self, file_id):
        return self.contents[file_id]

    def requests(self, batch_id):
        return self.uploads[self.batches[batch_id].input_file_id]

    def finish(self, batch_id, status="completed", fail=(), skip=(), text=VALID):
        """Responde cada pedido (200), falha os custom_ids em `fail`, omite os de `skip` (ex.: expirado)."""
        out, err = [], []
        for req in self.requests(batch_id):
            cid = req["custom_id"]
            if cid in skip:
                continue
            if cid in fail:
                err.append({"id": "r", "custom_id": cid, "error": None, "response": {
                    "status_code": 400, "body": {"error": {"message": "bad request", "code": "invalid"}}}})
            else:
                out.append({"id": "r", "custom_id": cid, "error": None, "response": {"status_code": 200, "body": {
                    "model": "gpt-test-2026", "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20}}}})
        b = self.batches[batch_id]
        b.status = status
        b.request_counts = SimpleNamespace(model_dump=lambda: {"total": len(self.requests(batch_id)),
                                                               "completed": len(out), "failed": len(err)})
        for kind, lines in (("output", out), ("error", err)):
            if lines:
                file_id = f"{kind}-{batch_id}"
                self.contents[file_id] = "".join(json.dumps(line) + "\n" for line in lines)
                setattr(b, f"{kind}_file_id", file_id)


@pytest.fixture
def openai_client():
    pytest.importorskip("litellm")
    client = LiteLLMClient(SETTINGS)
    client.batch_api = FakeBatchAPI()
    return client, client.batch_api


def experiment():
    return Experiment.build("candidate_validation", SETTINGS, PROMPT, LIMITS)


def submit(client, commits, tmp_path, limits=BATCH_LIMITS, **kwargs):
    return batch.submit(client, client.batch_api, commits, MODES, experiment(), PROMPT, tmp_path, limits, **kwargs)


def collect(client, commits, tmp_path, **kwargs):
    return batch.collect(client.batch_api, commits, experiment(), PROMPT, tmp_path, **kwargs)


def test_submit_builds_the_same_requests_as_the_synchronous_path(openai_client, tmp_path):
    client, fake = openai_client
    commits = [make_commit("c1"), make_commit("c2")]
    stats, states = submit(client, commits, tmp_path)

    assert stats.planned == 4 and len(states) == 1
    state = states[0]
    requests = fake.requests(state["batch_id"])
    assert len(requests) == 4 and set(state["requests"]) == {r["custom_id"] for r in requests}
    body = requests[0]["body"]
    assert body["model"] == "gpt-4.1-nano" and body["temperature"] == 0 and body["seed"] == 7
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][0] == {"role": "system", "content": PROMPT.system}
    assert (tmp_path / "batches" / f"{state['batch_id']}_input.jsonl").exists()
    assert (tmp_path / "batches" / f"{state['batch_id']}.json").exists()


def test_pending_requests_are_not_sent_twice(openai_client, tmp_path):
    client, _ = openai_client
    commits = [make_commit("c1")]
    submit(client, commits, tmp_path)

    stats, states = submit(client, commits, tmp_path)
    assert states == [] and stats.skipped_pending_batch == 2

    sync_client = FakeClient()
    pending = batch.pending_record_ids(tmp_path, experiment().experiment_id)
    sync_stats = run_classification(sync_client, commits, MODES, experiment(), PROMPT, tmp_path,
                                    exclude_record_ids=pending)
    assert sync_stats.classified == 0 and sync_client.calls == []


def test_collect_waits_for_running_batches(openai_client, tmp_path):
    client, _ = openai_client
    submit(client, [make_commit()], tmp_path)
    totals = collect(client, [make_commit()], tmp_path)
    assert totals == {"batches_collected": 0, "records_written": 0, "api_failures": 0, "still_running": 1}
    assert not result_path(tmp_path, "diff_only").exists()


def test_collect_writes_records_and_failed_requests_are_resubmitted(openai_client, tmp_path):
    client, fake = openai_client
    commits = [make_commit("c1"), make_commit("c2")]
    _, [state] = submit(client, commits, tmp_path)
    failed_id = next(iter(state["requests"]))
    fake.finish(state["batch_id"], fail={failed_id})

    totals = collect(client, commits, tmp_path)
    assert totals["records_written"] == 4 and totals["api_failures"] == 1

    records = {**load_results(result_path(tmp_path, "message_only")),
               **load_results(result_path(tmp_path, "diff_only"))}
    ok = next(r for rid, r in records.items() if rid != failed_id)
    assert ok["label"] == "BUG_FIX" and ok["parse_success"] and ok["batch_id"] == state["batch_id"]
    assert ok["response_model"] == "gpt-test-2026" and ok["input_tokens"] == 100
    assert ok["user_prompt"] and ok["raw_response"] == VALID
    assert records[failed_id]["error"] == "bad request" and records[failed_id]["error_type"] == "HTTP 400"

    assert batch.pending_record_ids(tmp_path, experiment().experiment_id) == set()
    stats, [retry] = submit(client, commits, tmp_path)
    assert stats.skipped_existing == 3 and list(retry["requests"]) == [failed_id]
    assert collect(client, commits, tmp_path)["batches_collected"] == 0  # já coletado não é coletado de novo


def test_expired_batch_keeps_missing_requests_for_resubmission(openai_client, tmp_path):
    client, fake = openai_client
    commits = [make_commit("c1")]
    _, [state] = submit(client, commits, tmp_path)
    missing = sorted(state["requests"])[0]
    fake.finish(state["batch_id"], status="expired", skip={missing})

    assert collect(client, commits, tmp_path)["records_written"] == 1
    _, [again] = submit(client, commits, tmp_path)
    assert list(again["requests"]) == [missing]


def test_collect_wait_polls_until_done(openai_client, tmp_path):
    client, fake = openai_client
    _, [state] = submit(client, [make_commit()], tmp_path)
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        fake.finish(state["batch_id"])

    totals = collect(client, [make_commit()], tmp_path, wait=True, poll_interval_seconds=5, sleep=sleep)
    assert sleeps == [5] and totals["records_written"] == 2 and totals["still_running"] == 0


def test_split_into_several_batches(openai_client, tmp_path):
    client, _ = openai_client
    limits = batch.BatchLimits(max_requests_per_batch=3, max_file_bytes=10**9, poll_interval_seconds=0)
    _, states = submit(client, [make_commit(f"c{i}") for i in range(4)], tmp_path, limits=limits)
    assert [s["n_requests"] for s in states] == [3, 3, 2]


def test_status_and_cancel(openai_client, tmp_path):
    client, _ = openai_client
    submit(client, [make_commit()], tmp_path)
    [state] = batch.status(client.batch_api, tmp_path, experiment().experiment_id)
    assert "validating" in batch.describe(state)

    assert batch.cancel(client.batch_api, tmp_path, experiment().experiment_id) == [state["batch_id"]]
    assert collect(client, [make_commit()], tmp_path)["batches_collected"] == 1
    assert batch.pending_record_ids(tmp_path, experiment().experiment_id) == set()


def test_collect_rejects_changed_prepared_commits(openai_client, tmp_path):
    client, fake = openai_client
    _, [state] = submit(client, [make_commit(patch="@@ old\n")], tmp_path)
    fake.finish(state["batch_id"])
    with pytest.raises(ResultConflictError):
        collect(client, [make_commit(patch="@@ new\n")], tmp_path)


def test_batch_requires_litellm_and_a_batch_vendor():
    pytest.importorskip("litellm")
    with pytest.raises(batch.BatchError, match="provider: litellm"):
        batch.batch_api_for(FakeClient())
    with pytest.raises(batch.BatchError, match="no batch support for 'anthropic'"):
        batch.batch_api_for(LiteLLMClient(LLMSettings(model="anthropic/claude-sonnet-5")))
    assert batch.batch_api_for(LiteLLMClient(SETTINGS)).vendor == "openai"


def test_batch_rejects_parameters_the_vendor_does_not_accept():
    pytest.importorskip("litellm")
    bedrock = LiteLLMClient(LLMSettings(model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0", seed=1))
    with pytest.raises(LLMConfigError, match="seed"):  # o Bedrock não aceita seed
        batch.batch_api_for(bedrock)
