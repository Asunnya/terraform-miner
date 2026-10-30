"""Batch via LiteLLM e via SDK da Anthropic, com APIs falsas (sem rede)."""
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
    assert totals == {"batches_collected": 0, "records_written": 0, "api_failures": 0, "still_running": 1,
                      "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
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
    with pytest.raises(batch.BatchError, match="No batch support for 'gemini'"):
        batch.batch_api_for(LiteLLMClient(LLMSettings(model="gemini/gemini-2.5-flash", seed=None)))
    assert batch.batch_api_for(LiteLLMClient(SETTINGS)).vendor == "openai"


def test_batch_rejects_parameters_the_vendor_does_not_accept():
    pytest.importorskip("litellm")
    bedrock = LiteLLMClient(LLMSettings(model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0", seed=1))
    with pytest.raises(LLMConfigError, match="seed"):  # o Bedrock não aceita seed
        batch.batch_api_for(bedrock)


def test_batch_records_keep_api_usage_and_batch_price(openai_client, tmp_path):
    from src.llm_analysis.costs import load_pricing
    pricing = load_pricing({"models": {"gpt-4.1-nano": {"sync": {"input": 0.10, "output": 0.40},
                                                        "batch": {"input": 0.05, "output": 0.20}}}})
    client, fake = openai_client
    _, [state] = submit(client, [make_commit()], tmp_path)
    fake.finish(state["batch_id"])
    totals = collect(client, [make_commit()], tmp_path, pricing=pricing)

    record = next(iter(load_results(result_path(tmp_path, "diff_only")).values()))
    assert record["usage"] == {"prompt_tokens": 100, "completion_tokens": 20}   # como veio da API
    assert record["cost_usd"] == pytest.approx((100 * 0.05 + 20 * 0.20) / 1e6)  # preço de batch
    assert totals["input_tokens"] == 200 and totals["output_tokens"] == 40
    assert totals["cost_usd"] == pytest.approx(2 * record["cost_usd"])


ANTHROPIC_SETTINGS = LLMSettings(provider="litellm", model="anthropic/claude-haiku-5-5", temperature=None, seed=None,
                                 json_mode=True, max_output_tokens=256)


class Dumpable(SimpleNamespace):
    """Imita um objeto pydantic do SDK (só model_dump)."""

    def model_dump(self, mode="python"):
        return dict(self.data)


class FakeAnthropicBatches:
    """Imita anthropic.Anthropic().messages.batches; `finish` simula a Anthropic processando o batch."""

    def __init__(self):
        self.created, self.state, self.out = {}, {}, {}

    def _batch(self, batch_id):
        return Dumpable(data=self.state[batch_id])

    def create(self, requests):
        batch_id = f"msgbatch_{len(self.created)}"
        self.created[batch_id] = requests
        self.state[batch_id] = {"id": batch_id, "processing_status": "in_progress", "cancel_initiated_at": None,
                                "request_counts": {"processing": len(requests), "succeeded": 0, "errored": 0,
                                                   "canceled": 0, "expired": 0}}
        return self._batch(batch_id)

    def retrieve(self, batch_id):
        return self._batch(batch_id)

    def cancel(self, batch_id):
        self.state[batch_id].update(processing_status="ended", cancel_initiated_at="2026-10-30T00:00:00Z")

    def results(self, batch_id):
        return [Dumpable(data=entry) for entry in self.out[batch_id]]

    def finish(self, batch_id, fail=(), expire=(), text=VALID):
        entries = []
        for req in self.created[batch_id]:
            cid = req["custom_id"]
            if cid in fail:
                result = {"type": "errored", "error": {"type": "error", "error": {
                    "type": "invalid_request_error", "message": "temperature: must be 1"}}}
            elif cid in expire:
                result = {"type": "expired"}
            else:
                result = {"type": "succeeded", "message": {
                    "id": "msg_1", "model": "claude-haiku-5-5", "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": text}],
                    "usage": {"input_tokens": 90, "cache_read_input_tokens": 10, "output_tokens": 20}}}
            entries.append({"custom_id": cid, "result": result})
        self.out[batch_id] = entries
        types = [e["result"]["type"] for e in entries]
        self.state[batch_id].update(processing_status="ended", request_counts={
            "processing": 0, "succeeded": types.count("succeeded"), "errored": types.count("errored"),
            "canceled": 0, "expired": types.count("expired")})


@pytest.fixture
def anthropic_client():
    pytest.importorskip("litellm")
    client = LiteLLMClient(ANTHROPIC_SETTINGS)
    fake = FakeAnthropicBatches()
    client.batch_api = batch.AnthropicBatchAPI(sdk=SimpleNamespace(messages=SimpleNamespace(batches=fake)))
    return client, fake


def anthropic_experiment():
    return Experiment.build("candidate_validation", ANTHROPIC_SETTINGS, PROMPT, LIMITS)


def test_anthropic_submit_translates_the_synchronous_request(anthropic_client, tmp_path):
    client, fake = anthropic_client
    _, [state] = batch.submit(client, client.batch_api, [make_commit("c1")], MODES, anthropic_experiment(), PROMPT,
                              tmp_path, BATCH_LIMITS)

    requests = fake.created[state["batch_id"]]
    assert {r["custom_id"] for r in requests} == set(state["requests"]) and state["vendor"] == "anthropic"
    params = requests[0]["params"]
    assert params["model"] == "claude-haiku-5-5" and params["max_tokens"] == 256
    assert params["system"] == PROMPT.system
    assert [m["role"] for m in params["messages"]] == ["user"]
    assert "response_format" not in params and "temperature" not in params and "seed" not in params
    assert batch.describe(batch.status(client.batch_api, tmp_path, anthropic_experiment().experiment_id)[0]).split()[1] \
        == "in_progress"


def test_anthropic_collect_writes_records_and_resubmits_failed_and_expired(anthropic_client, tmp_path):
    client, fake = anthropic_client
    commits = [make_commit("c1"), make_commit("c2")]
    exp = anthropic_experiment()
    _, [state] = batch.submit(client, client.batch_api, commits, MODES, exp, PROMPT, tmp_path, BATCH_LIMITS)
    failed, expired = sorted(state["requests"])[:2]
    fake.finish(state["batch_id"], fail={failed}, expire={expired})

    totals = batch.collect(client.batch_api, commits, exp, PROMPT, tmp_path)
    assert totals["records_written"] == 3 and totals["api_failures"] == 1
    assert totals["input_tokens"] == 200 and totals["output_tokens"] == 40  # 90 + 10 lidos do cache

    records = {**load_results(result_path(tmp_path, "message_only")),
               **load_results(result_path(tmp_path, "diff_only"))}
    ok = next(r for rid, r in records.items() if rid != failed)
    assert ok["label"] == "BUG_FIX" and ok["batch_id"] == state["batch_id"] and ok["stop_reason"] == "stop"
    assert ok["response_model"] == "claude-haiku-5-5" and ok["usage"]["cache_read_input_tokens"] == 10
    assert records[failed]["error"] == "temperature: must be 1"
    assert records[failed]["error_type"] == "invalid_request_error"
    assert expired not in records

    stats, [retry] = batch.submit(client, client.batch_api, commits, MODES, exp, PROMPT, tmp_path, BATCH_LIMITS)
    assert set(retry["requests"]) == {failed, expired} and stats.skipped_existing == 2


def test_anthropic_cancel(anthropic_client, tmp_path):
    client, fake = anthropic_client
    exp = anthropic_experiment()
    batch.submit(client, client.batch_api, [make_commit()], MODES, exp, PROMPT, tmp_path, BATCH_LIMITS)
    [batch_id] = batch.cancel(client.batch_api, tmp_path, exp.experiment_id)
    fake.out[batch_id] = []
    assert batch.load_states(tmp_path, exp.experiment_id)[0]["status"] == "cancelled"
    assert batch.collect(client.batch_api, [make_commit()], exp, PROMPT, tmp_path)["batches_collected"] == 1
    assert batch.pending_record_ids(tmp_path, exp.experiment_id) == set()


def test_anthropic_batch_rejects_seed():
    pytest.importorskip("litellm")
    with pytest.raises(LLMConfigError, match="seed"):
        batch.batch_api_for(LiteLLMClient(LLMSettings(model="anthropic/claude-haiku-5-5", seed=1)))
