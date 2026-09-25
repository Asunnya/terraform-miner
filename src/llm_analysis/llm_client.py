"""LLM access through LiteLLM (plus an offline mock). Retries are done here."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from src.llm_analysis.schemas import LABELS

LITELLM_LOGGERS = ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy")
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})
SERVER_ERROR_STATUS = 500


class LLMConfigError(RuntimeError):
    """Bad provider setup: unknown provider, missing key or SDK."""


class LLMError(RuntimeError):
    """A request failed after all attempts."""

    def __init__(self, message: str, attempts: int, cause: BaseException | None = None):
        super().__init__(message)
        self.attempts = attempts
        self.error_type = type(cause).__name__ if cause else type(self).__name__


@dataclass(frozen=True)
class LLMSettings:
    model: str
    provider: str = "litellm"
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
    # Extra request parameters, sent as-is.
    extra_params: dict[str, Any] = field(default_factory=dict)

    # `llm` config keys that are not request settings.
    NON_REQUEST_KEYS = frozenset({"workers"})

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> LLMSettings:
        unknown = set(cfg) - set(cls.__dataclass_fields__) - cls.NON_REQUEST_KEYS
        if unknown:
            raise LLMConfigError(f"Unknown llm settings: {sorted(unknown)} (use llm.extra_params for request parameters)")
        values = {k: v for k, v in cfg.items() if k not in cls.NON_REQUEST_KEYS}
        values["extra_params"] = dict(values.get("extra_params") or {})
        if not values.get("model"):
            raise LLMConfigError("llm.model is required")
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
        """Retry timeouts, connection errors, 408/409/429 and 5xx."""
        status = getattr(error, "status_code", None)
        return status is None or status in RETRYABLE_STATUS_CODES or status >= SERVER_ERROR_STATUS

    def classify(self, system: str, user: str) -> LLMResponse:
        max_attempts = max(1, int(self.settings.retries) + 1)
        for attempt in range(1, max_attempts + 1):
            started = time.monotonic()
            try:
                response = self._complete(system, user)
            except Exception as e:  # noqa: BLE001 - SDK errors vary
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


def chat_completion_to_response(body: dict[str, Any]) -> LLMResponse:
    """Chat Completions response dict -> LLMResponse."""
    choice = (body.get("choices") or [{}])[0]
    usage = body.get("usage") or {}
    return LLMResponse(
        text=(choice.get("message") or {}).get("content") or "",
        model=body.get("model"),
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        stop_reason=choice.get("finish_reason"),
    )


class LiteLLMClient(LLMClient):
    """Any LiteLLM model, named "<vendor>/<model>" (e.g. openai/gpt-4.1-nano)."""

    def __init__(self, settings: LLMSettings, sleep: Callable[[float], None] = time.sleep):
        super().__init__(settings, sleep)
        try:
            import litellm
        except ImportError as e:
            raise LLMConfigError("The 'litellm' package is required: uv pip install -r requirements.txt") from e
        litellm.telemetry = False
        litellm.suppress_debug_info = True
        for name in LITELLM_LOGGERS:  # one INFO line per call otherwise
            logging.getLogger(name).setLevel(logging.WARNING)
        self._litellm = litellm
        try:
            self.model_name, self.vendor, _, _ = litellm.get_llm_provider(settings.model)
        except Exception as e:  # noqa: BLE001 - LiteLLM raises several types
            raise LLMConfigError(f"LiteLLM does not recognize llm.model {settings.model!r}: {e}") from e
        self._api_key_value = self._api_key("", required=True) if settings.api_key_env else None

    def request_params(self) -> dict[str, Any]:
        """Sampling and format parameters, shared by the sync and batch paths."""
        params: dict[str, Any] = {self.settings.max_tokens_param: self.settings.max_output_tokens}
        if self.settings.temperature is not None:
            params["temperature"] = self.settings.temperature
        if self.settings.seed is not None:
            params["seed"] = self.settings.seed
        if self.settings.json_mode:
            params["response_format"] = {"type": "json_object"}
        return {**params, **self.settings.extra_params}

    def check_supported_params(self) -> None:
        """Fail if the vendor does not accept a configured parameter."""
        supported = self._litellm.get_supported_openai_params(model=self.model_name, custom_llm_provider=self.vendor)
        if supported is None:
            return
        configured = {self.settings.max_tokens_param, "temperature", "seed", "response_format"}
        rejected = sorted(k for k in configured & set(self.request_params()) if k not in supported)
        if rejected:
            raise LLMConfigError(f"{self.settings.model} does not accept {rejected}; set them to null/false in config.yaml")

    def request_body(self, system: str, user: str) -> dict[str, Any]:
        """Raw Chat Completions body for a batch input line."""
        return {
            "model": self.model_name,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            **self.request_params(),
        }

    def _complete(self, system: str, user: str) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "timeout": self.settings.timeout_seconds,
            "num_retries": 0,
            **self.request_params(),
        }
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
    """Offline client with fake, hash-based labels. Tests only."""

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
