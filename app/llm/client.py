"""Thin async client for the OpenAI-compatible LLM endpoint (Ollama in dev, vLLM in prod).

Retries with backoff come from the openai SDK (``max_retries``). A small circuit breaker
stops hammering the server when it is down, so callers fail fast with a clear message.
"""

import re
import time
from functools import lru_cache

from loguru import logger
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from app.config import get_settings

_THINK_RE = re.compile(r"<(think|thought)>.*?</\1>", re.DOTALL | re.IGNORECASE)


class LLMUnavailable(Exception):
    """The LLM server is unreachable or failing."""


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, reset_after_secs: float = 30.0):
        self._threshold = failure_threshold
        self._reset_after = reset_after_secs
        self._failures = 0
        self._opened_at: float | None = None

    def allow(self) -> bool:
        if self._opened_at is None:
            return True
        if time.monotonic() - self._opened_at >= self._reset_after:
            return True  # half-open: let one request probe the server
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold:
            self._opened_at = time.monotonic()


class LLMClient:
    def __init__(self):
        s = get_settings()
        self.model = s.llm_model
        self._extra_body = s.llm_extra_body
        self._client = AsyncOpenAI(
            base_url=s.llm_base_url,
            api_key=s.llm_api_key,
            timeout=s.llm_timeout_secs,
            max_retries=s.llm_max_retries,
        )
        self._breaker = CircuitBreaker()

    async def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        json_schema: dict | None = None,
    ) -> str:
        if not self._breaker.allow():
            raise LLMUnavailable("LLM circuit open")
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "extra_body": self._extra_body,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if json_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": json_schema},
            }
        started = time.perf_counter()
        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except (APIConnectionError, APITimeoutError, APIStatusError) as e:
            self._breaker.record_failure()
            raise LLMUnavailable(str(e)) from e
        self._breaker.record_success()
        content = resp.choices[0].message.content or ""
        usage = resp.usage
        logger.debug(
            f"llm {time.perf_counter() - started:.2f}s "
            f"in={getattr(usage, 'prompt_tokens', '?')} out={getattr(usage, 'completion_tokens', '?')}"
        )
        return _THINK_RE.sub("", content).strip()


@lru_cache
def get_llm() -> LLMClient:
    return LLMClient()
