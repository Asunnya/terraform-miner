"""
Provider-independent LLM access.

`LLMClient.classify(system, user)` sends one classification request and returns the raw text plus
usage metadata; parsing/validation happens in schemas.parse_llm_response so the raw output is always
preserved. Retries (with exponential backoff) are done here, not by the vendor SDKs, so the attempt
count is recorded and the behavior does not depend on SDK defaults.

To add a provider: subclass LLMClient, implement `_complete`, and register it with
`register_provider("name", MyClient)` (or add it to PROVIDERS). See README.md.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from src.llm_analysis.schemas import LABELS

RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})
SERVER_ERROR_STATUS = 500


class LLMConfigError(RuntimeError):
    """Invalid provider configuration (unknown provider, missing credentials, missing SDK)."""


class LLMError(RuntimeError):
    """A request failed after all attempts."""

    def __init__(self, message: str, attempts: int, cause: BaseException | None = None):
        super().__init__(message)
        self.attempts = attempts
        self.error_type = type(cause).__name__ if cause else type(self).__name__


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    temperature: float | None = 0.0
    max_output_tokens: int = 1024
    timeout_seconds: float = 120.0
    retries: int = 3
    retry_backoff_seconds: float = 2.0
    api_key_env: str | None = None
    base_url: str | None = None
    seed: int | None = None
    json_mode: bool = False
    max_tokens_param: str = "max_tokens"
    # Additional provider request parameters (e.g. top_p), passed through verbatim.
    extra_params: dict[str, Any] = field(default_factory=dict)

    # Keys of the `llm` config section that are not request settings.
    NON_REQUEST_KEYS = frozenset({"workers"})

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> LLMSettings:
        unknown = set(cfg) - set(cls.__dataclass_fields__) - cls.NON_REQUEST_KEYS
        if unknown:
            raise LLMConfigError(f"Unknown llm settings: {sorted(unknown)} (use llm.extra_params for request parameters)")
        values = {k: v for k, v in cfg.items() if k not in cls.NON_REQUEST_KEYS}
        values["extra_params"] = dict(values.get("extra_params") or {})
        if not values.get("provider") or not values.get("model"):
            raise LLMConfigError("llm.provider and llm.model are required")
        return cls(**values)


@dataclass
class LLMResponse:
    text: str
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    stop_reason: str | None = None
    attempts: int = 1
    latency_seconds: float | None = None


class LLMClient(ABC):
    def __init__(self, settings: LLMSettings, sleep: Callable[[float], None] = time.sleep):
        self.settings = settings
        self._sleep = sleep

    @abstractmethod
    def _complete(self, system: str, user: str) -> LLMResponse:
        """Send one request, without retrying."""

    def is_retryable(self, error: BaseException) -> bool:
        """Retry timeouts/connection errors (no HTTP status), 408/409/429 and 5xx; not 4xx client errors."""
        status = getattr(error, "status_code", None)
        return status is None or status in RETRYABLE_STATUS_CODES or status >= SERVER_ERROR_STATUS

    def classify(self, system: str, user: str) -> LLMResponse:
        max_attempts = max(1, int(self.settings.retries) + 1)
        for attempt in range(1, max_attempts + 1):
            started = time.monotonic()
            try:
                response = self._complete(system, user)
            except Exception as e:  # noqa: BLE001 - vendor SDKs raise many exception types
                if attempt == max_attempts or not self.is_retryable(e):
                    raise LLMError(f"{type(e).__name__}: {e}", attempts=attempt, cause=e) from e
                self._sleep(self.settings.retry_backoff_seconds * 2 ** (attempt - 1))
                continue
            response.attempts = attempt
            response.latency_seconds = round(time.monotonic() - started, 3)
            return response
        raise AssertionError("unreachable")

    def _api_key(self, default_env: str, required: bool = True) -> str | None:
        env_name = self.settings.api_key_env or default_env
        key = os.environ.get(env_name)
        if required and not key:
            raise LLMConfigError(f"Environment variable {env_name} is not set (API key for {self.settings.provider})")
        return key


class AnthropicClient(LLMClient):
    DEFAULT_API_KEY_ENV = "ANTHROPIC_API_KEY"

    def __init__(self, settings: LLMSettings, sleep: Callable[[float], None] = time.sleep):
        super().__init__(settings, sleep)
        try:
            import anthropic
        except ImportError as e:
            raise LLMConfigError("The 'anthropic' package is required: uv pip install -r requirements.txt") from e
        kwargs: dict[str, Any] = {
            "api_key": self._api_key(self.DEFAULT_API_KEY_ENV),
            "timeout": settings.timeout_seconds,
            "max_retries": 0,
        }
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        self._client = anthropic.Anthropic(**kwargs)

    def _complete(self, system: str, user: str) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_output_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            **self.settings.extra_params,
        }
        if self.settings.temperature is not None:
            kwargs["temperature"] = self.settings.temperature
        response = self._client.messages.create(**kwargs)
        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        return LLMResponse(
            text=text,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
        )


class OpenAICompatibleClient(LLMClient):
    """OpenAI Chat Completions API; also covers compatible servers (Ollama, vLLM, LM Studio) via base_url."""
    DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
    # Local OpenAI-compatible servers usually ignore the key, but the SDK requires a non-empty one.
    PLACEHOLDER_API_KEY = "not-needed"

    def __init__(self, settings: LLMSettings, sleep: Callable[[float], None] = time.sleep):
        super().__init__(settings, sleep)
        try:
            import openai
        except ImportError as e:
            raise LLMConfigError("The 'openai' package is required: uv pip install -r requirements.txt") from e
        key = self._api_key(self.DEFAULT_API_KEY_ENV, required=not settings.base_url)
        self._client = openai.OpenAI(
            api_key=key or self.PLACEHOLDER_API_KEY,
            base_url=settings.base_url or None,
            timeout=settings.timeout_seconds,
            max_retries=0,
        )

    def _complete(self, system: str, user: str) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            self.settings.max_tokens_param: self.settings.max_output_tokens,
            **self.settings.extra_params,
        }
        if self.settings.temperature is not None:
            kwargs["temperature"] = self.settings.temperature
        if self.settings.seed is not None:
            kwargs["seed"] = self.settings.seed
        if self.settings.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        usage = response.usage
        return LLMResponse(
            text=choice.message.content or "",
            model=response.model,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            stop_reason=choice.finish_reason,
        )


class LiteLLMClient(LLMClient):
    """
    Any vendor supported by LiteLLM (Gemini, Bedrock, Azure, Mistral, Groq, ...), with the model given as
    "<vendor>/<model>", e.g. "gemini/gemini-2.5-pro". LiteLLM reads each vendor's usual environment
    variable (GEMINI_API_KEY, ...) itself; set llm.api_key_env to pass a specific one explicitly.

    Unsupported parameters raise an error instead of being silently dropped (LiteLLM's drop_params
    stays off), so the stored settings are the ones actually sent.
    """

    def __init__(self, settings: LLMSettings, sleep: Callable[[float], None] = time.sleep):
        super().__init__(settings, sleep)
        try:
            import litellm
        except ImportError as e:
            raise LLMConfigError("The 'litellm' package is required: uv pip install -r requirements.txt") from e
        litellm.telemetry = False
        litellm.suppress_debug_info = True
        self._litellm = litellm
        self._api_key_value = self._api_key("", required=True) if settings.api_key_env else None

    def _complete(self, system: str, user: str) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": self.settings.max_output_tokens,
            "timeout": self.settings.timeout_seconds,
            "num_retries": 0,
            **self.settings.extra_params,
        }
        if self.settings.temperature is not None:
            kwargs["temperature"] = self.settings.temperature
        if self.settings.seed is not None:
            kwargs["seed"] = self.settings.seed
        if self.settings.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.settings.base_url:
            kwargs["base_url"] = self.settings.base_url
        if self._api_key_value:
            kwargs["api_key"] = self._api_key_value
        response = self._litellm.completion(**kwargs)
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return LLMResponse(
            text=choice.message.content or "",
            model=response.model,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            stop_reason=choice.finish_reason,
        )


class MockLLMClient(LLMClient):
    """
    Offline stand-in for pipeline dry runs and tests. Its labels are a hash of the prompt, NOT a
    classification: results produced with provider "mock" must never be analyzed.
    """

    def _complete(self, system: str, user: str) -> LLMResponse:
        digest = hashlib.sha256((system + user).encode("utf-8")).digest()
        payload = {
            "label": LABELS[digest[0] % len(LABELS)],
            "confidence": round(digest[1] / 255, 2),
            "message_evidence": "",
            "diff_evidence": "",
            "reason": "Mock provider: deterministic placeholder, not a real classification.",
            "uncertainty": "Mock provider output.",
        }
        return LLMResponse(text=json.dumps(payload), model=f"mock:{self.settings.model}",
                           input_tokens=0, output_tokens=0, stop_reason="mock")


PROVIDERS: dict[str, type[LLMClient]] = {
    "anthropic": AnthropicClient,
    "openai": OpenAICompatibleClient,
    "litellm": LiteLLMClient,
    "mock": MockLLMClient,
}


def register_provider(name: str, client_class: type[LLMClient]) -> None:
    PROVIDERS[name] = client_class


def create_client(settings: LLMSettings) -> LLMClient:
    try:
        client_class = PROVIDERS[settings.provider]
    except KeyError:
        raise LLMConfigError(f"Unknown llm.provider {settings.provider!r}; available: {sorted(PROVIDERS)}") from None
    return client_class(settings)
