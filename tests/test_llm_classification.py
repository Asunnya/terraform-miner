import json
from dataclasses import replace

import pytest

from src.llm_analysis.classification import (
    Experiment,
    ResultConflictError,
    classify_commit,
    load_results,
    result_path,
    run_classification,
)
from src.llm_analysis.diff_context import ContextLimits
from src.llm_analysis.llm_client import LLMClient, LLMError, LLMResponse, LLMSettings
from src.llm_analysis.prompts import PROMPT_V1
from src.llm_analysis.schemas import BUG_FIX, INPUT_MODES, NOT_PROVIDED, RESULT_FIELDS, FileDiff, PreparedCommit

SETTINGS = LLMSettings(provider="fake", model="fake-model", retries=2, retry_backoff_seconds=0)
LIMITS = ContextLimits(max_diff_chars=1000, max_chars_per_file=500, max_files_per_commit=5, max_listed_file_names=10)
VALID = json.dumps({"label": "BUG_FIX", "confidence": 0.9, "message_evidence": "m", "diff_evidence": "d",
                    "reason": "r", "uncertainty": ""})


class HTTPError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class FakeClient(LLMClient):
    """Cliente falso: devolve as respostas da fila (str ou exceção) ou VALID quando ela acaba."""

    def __init__(self, responses=(), settings=SETTINGS):
        super().__init__(settings, sleep=lambda s: self.sleeps.append(s))
        self.responses = list(responses)
        self.calls = []
        self.sleeps = []

    def _complete(self, system, user):
        self.calls.append(user)
        item = self.responses.pop(0) if self.responses else VALID
        if isinstance(item, Exception):
            raise item
        return LLMResponse(text=item, model="fake-model-2026", input_tokens=10, output_tokens=5, stop_reason="end")


def make_commit(commit_hash="c1", message="Fix invalid subnet mask", patch="@@ -1 +1 @@\n-a\n+b\n"):
    return PreparedCommit(repo="owner_repo", commit_hash=commit_hash, author="dev", date="2024-01-01",
                          message=message, head_hash="h", files=["main.tf"], diffs=[FileDiff("main.tf", patch)],
                          files_without_patch=[], n_records=1, duplicate_patches_removed=0,
                          source_files=["x"], content_sha256=f"sha-{commit_hash}-{patch}")


def experiment(**overrides):
    exp = Experiment.build("candidate_validation", SETTINGS, PROMPT_V1, LIMITS)
    return replace(exp, **overrides)


def run(client, commits, tmp_path, modes=INPUT_MODES, **kwargs):
    return run_classification(client, commits, modes, kwargs.pop("exp", experiment()), PROMPT_V1, tmp_path, **kwargs)


def test_record_has_all_metadata():
    rec = classify_commit(FakeClient(), make_commit(), "message_and_diff", experiment(), PROMPT_V1)

    assert list(rec) == list(RESULT_FIELDS)
    assert rec["label"] == BUG_FIX and rec["confidence"] == 0.9 and rec["parse_success"]
    assert rec["raw_response"] == VALID
    assert rec["error"] is None
    assert rec["response_model"] == "fake-model-2026"
    assert rec["submitted_diff_chars"] == rec["original_diff_chars"] > 0
    assert "Fix invalid subnet mask" in rec["user_prompt"] and "FILE: main.tf" in rec["user_prompt"]


def test_message_only_submits_no_diff():
    client = FakeClient()
    rec = classify_commit(client, make_commit(), "message_only", experiment(), PROMPT_V1)
    assert rec["submitted_diff_chars"] == 0 and rec["original_diff_chars"] > 0 and not rec["diff_truncated"]
    assert "FILE:" not in client.calls[0] and NOT_PROVIDED in client.calls[0]


def test_parse_failure_keeps_raw_response():
    rec = classify_commit(FakeClient(["I think it is a bug fix"]), make_commit(), "diff_only", experiment(), PROMPT_V1)
    assert not rec["parse_success"] and rec["label"] is None
    assert rec["raw_response"] == "I think it is a bug fix"
    assert rec["error"] is None


def test_retries_transient_errors_then_succeeds():
    client = FakeClient([HTTPError(429), TimeoutError("slow"), VALID])
    response = client.classify("s", "u")
    assert response.attempts == 3
    assert client.sleeps == [0, 0]


def test_does_not_retry_client_errors():
    client = FakeClient([HTTPError(400)])
    with pytest.raises(LLMError) as exc:
        client.classify("s", "u")
    assert exc.value.attempts == 1 and exc.value.error_type == "HTTPError"


def test_api_failure_after_retries_is_recorded():
    client = FakeClient([HTTPError(503)] * 3)
    rec = classify_commit(client, make_commit(), "diff_only", experiment(), PROMPT_V1)
    assert rec["error"] and rec["attempts"] == 3 and rec["error_type"] == "HTTPError"
    assert not rec["parse_success"]


def test_same_commits_in_every_mode_and_immediate_persistence(tmp_path):
    commits = [make_commit("c1"), make_commit("c2")]
    stats = run(FakeClient(), commits, tmp_path)

    assert stats.classified == 6 and stats.planned == 6
    for mode in INPUT_MODES:
        records = load_results(result_path(tmp_path, mode))
        assert sorted(r["commit_hash"] for r in records.values()) == ["c1", "c2"]


def test_resume_skips_completed_requests(tmp_path):
    commits = [make_commit("c1"), make_commit("c2")]
    run(FakeClient(), commits[:1], tmp_path)

    client = FakeClient()
    stats = run(client, commits, tmp_path)
    assert stats.skipped_existing == 3 and stats.classified == 3
    assert len(client.calls) == 3

    client = FakeClient()
    assert run(client, commits, tmp_path).classified == 0 and client.calls == []


def test_resume_retries_api_failures_but_not_parse_failures(tmp_path):
    commit = make_commit()
    run(FakeClient([HTTPError(400), "garbage"]), [commit], tmp_path, modes=["message_only", "diff_only"])

    client = FakeClient()
    stats = run(client, [commit], tmp_path, modes=["message_only", "diff_only"])
    assert stats.classified == 1  # só a falha de API

    stats = run(FakeClient(), [commit], tmp_path, modes=["message_only", "diff_only"], retry_parse_failures=True)
    assert stats.classified == 1
    latest = load_results(result_path(tmp_path, "diff_only"))
    assert next(iter(latest.values()))["parse_success"]


def test_force_reclassifies_and_last_record_wins(tmp_path):
    commit = make_commit()
    run(FakeClient(), [commit], tmp_path, modes=["message_only"])
    other = json.dumps({**json.loads(VALID), "label": "NON_BUG_FIX"})
    run(FakeClient([other]), [commit], tmp_path, modes=["message_only"], force=True)

    path = result_path(tmp_path, "message_only")
    assert len(path.read_text().splitlines()) == 2
    assert next(iter(load_results(path).values()))["label"] == "NON_BUG_FIX"


def test_different_model_is_a_different_experiment(tmp_path):
    commit = make_commit()
    run(FakeClient(), [commit], tmp_path, modes=["message_only"])
    stats = run(FakeClient(), [commit], tmp_path, modes=["message_only"], exp=experiment(model="other"))
    assert stats.classified == 1


def test_stale_prepared_commit_is_rejected(tmp_path):
    run(FakeClient(), [make_commit(patch="@@ old\n")], tmp_path, modes=["diff_only"])
    with pytest.raises(ResultConflictError, match="different version"):
        run(FakeClient(), [make_commit(patch="@@ new\n")], tmp_path, modes=["diff_only"])


def test_edited_prompt_with_same_version_is_rejected(tmp_path):
    run(FakeClient(), [make_commit()], tmp_path, modes=["diff_only"])
    with pytest.raises(ResultConflictError, match="edited"):
        run(FakeClient(), [make_commit()], tmp_path, modes=["diff_only"], exp=experiment(prompt_sha256="0" * 64))


def test_truncated_line_from_interrupted_run_is_ignored(tmp_path):
    commit = make_commit()
    run(FakeClient(), [commit], tmp_path, modes=["message_only"])
    with open(result_path(tmp_path, "message_only"), "a") as f:
        f.write('{"record_id": "abc", "label"')  # escrita interrompida
    assert run(FakeClient(), [commit], tmp_path, modes=["message_only"]).classified == 0


def test_parallel_workers(tmp_path):
    commits = [make_commit(f"c{i}") for i in range(10)]
    stats = run(FakeClient(), commits, tmp_path, workers=4)
    assert stats.classified == 30
    assert len(load_results(result_path(tmp_path, "diff_only"))) == 10


def test_unknown_input_mode(tmp_path):
    with pytest.raises(ValueError):
        run(FakeClient(), [make_commit()], tmp_path, modes=["code_only"])


def test_settings_reject_unknown_keys():
    from src.llm_analysis.llm_client import LLMConfigError
    with pytest.raises(LLMConfigError):
        LLMSettings.from_config({"provider": "anthropic", "model": "m", "temprature": 0})
    settings = LLMSettings.from_config({"provider": "anthropic", "model": "m", "workers": 4,
                                        "extra_params": {"top_p": 1}})
    assert settings.extra_params == {"top_p": 1}


def test_env_file_loads_keys_without_overriding_shell(tmp_path, monkeypatch):
    from src.llm_analysis.utils import load_env_file
    env = tmp_path / ".env"
    env.write_text("LLM_TEST_KEY_A=from_file\nLLM_TEST_KEY_B=from_file\n")
    monkeypatch.delenv("LLM_TEST_KEY_A", raising=False)
    monkeypatch.setenv("LLM_TEST_KEY_B", "from_shell")

    assert load_env_file(env)
    import os
    assert os.environ["LLM_TEST_KEY_A"] == "from_file"
    assert os.environ["LLM_TEST_KEY_B"] == "from_shell"
    monkeypatch.delenv("LLM_TEST_KEY_A")
    assert not load_env_file(tmp_path / "missing.env")



def test_litellm_provider_offline():
    """Exercita o caminho real do LiteLLM com mock_response (sem rede)."""
    pytest.importorskip("litellm")
    from src.llm_analysis.llm_client import LiteLLMClient, create_client
    settings = LLMSettings(provider="litellm", model="gemini/gemini-test", extra_params={"mock_response": VALID})
    client = create_client(settings)
    assert isinstance(client, LiteLLMClient)

    rec = classify_commit(client, make_commit(), "message_and_diff", experiment(), PROMPT_V1)
    assert rec["parse_success"] and rec["label"] == BUG_FIX and rec["error"] is None
    assert rec["stop_reason"] == "stop" and rec["input_tokens"] is not None


def test_litellm_rejects_unsupported_params_without_retrying():
    """Gemini não aceita seed: o LiteLLM deve falhar em vez de descartar o parâmetro em silêncio."""
    pytest.importorskip("litellm")
    from src.llm_analysis.llm_client import create_client
    settings = LLMSettings(provider="litellm", model="gemini/gemini-test", seed=1, retry_backoff_seconds=0,
                           extra_params={"mock_response": VALID})
    rec = classify_commit(create_client(settings), make_commit(), "message_only", experiment(), PROMPT_V1)
    assert rec["error"] and "seed" in rec["error"]
    assert rec["attempts"] == 1
