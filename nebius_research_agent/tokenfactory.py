"""OpenAI-compatible chat client for Nebius Token Factory + resilient failover.

Nebius Token Factory serves NVIDIA open models (Nemotron 3 family) behind an
OpenAI-compatible chat-completions API::

    POST {NEBIUS_BASE_URL}/chat/completions
    Authorization: Bearer $NEBIUS_API_KEY

Base URL and model names are env-configurable::

    NEBIUS_API_KEY   required — without it every call raises
                     NebiusNotConfiguredError (no fake responses, ever)
    NEBIUS_BASE_URL  default https://api.tokenfactory.nebius.com/v1
    NEBIUS_MODEL     single model override
    NEBIUS_MODELS    comma-separated failover chain (overrides NEBIUS_MODEL)

``ResilientChatClient`` walks an ordered model chain: each rail gets bounded
retries with exponential backoff + jitter for transient failures (timeouts,
5xx, 429); a non-retryable response (4xx other) rotates immediately to the
next rail. Every hop is logged. If the whole chain fails the call raises
``ChainExhaustedError`` — it never fabricates an answer.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1"
# NVIDIA open models currently served on Token Factory
# (see https://github.com/nebius/token-factory-cookbook/tree/main/models/nemotron).
DEFAULT_MODEL_CHAIN = (
    "nvidia/Nemotron-3_5-Lightning",      # fast MoE execution layer for agents
    "nvidia/Nemotron-3-Super-120B-A12B",  # deep-reasoning fallback rail
)

RETRYABLE_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class NebiusNotConfiguredError(RuntimeError):
    """NEBIUS_API_KEY is missing; the client refuses to fake a response."""


class ChainExhaustedError(RuntimeError):
    """Every model rail in the failover chain failed for this call."""

    def __init__(self, attempts: list[tuple[str, str]]) -> None:
        self.attempts = attempts
        detail = "; ".join(f"{model}: {err}" for model, err in attempts)
        super().__init__(f"all model rails exhausted — {detail}")


@dataclass
class ChatResult:
    content: str
    model: str
    usage: dict = field(default_factory=dict)
    rail_index: int = 0
    attempts: int = 1


class ResilientChatClient:
    """Ordered failover across Token Factory model rails.

    ``http_client`` and ``sleep_fn`` are injectable so tests can run fully
    offline with deterministic backoff.
    """

    def __init__(
        self,
        models: list[str] | tuple[str, ...],
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        *,
        rail_tries: int = 2,
        backoff_base: float = 1.0,
        backoff_max: float = 30.0,
        timeout: float = 60.0,
        http_client: httpx.Client | None = None,
        sleep_fn=time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        if not models:
            raise ValueError("failover chain needs at least one model")
        if not api_key:
            raise NebiusNotConfiguredError(
                "NEBIUS_API_KEY is not set. Get a key from "
                "tokenfactory.nebius.com and export it; without it this "
                "client raises instead of pretending to answer."
            )
        self.models = list(models)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.rail_tries = max(1, rail_tries)
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self._http = http_client or httpx.Client(timeout=timeout)
        self._sleep = sleep_fn
        self._rng = rng or random.Random()
        self.health = {
            m: {"consecutive_failures": 0, "calls": 0, "failures": 0, "last_latency": None}
            for m in self.models
        }

    @classmethod
    def from_env(cls, **kwargs) -> "ResilientChatClient":
        api_key = os.environ.get("NEBIUS_API_KEY", "").strip()
        if not api_key:
            raise NebiusNotConfiguredError(
                "NEBIUS_API_KEY is not set. Get a key from "
                "tokenfactory.nebius.com and export it; without it this "
                "client raises instead of pretending to answer."
            )
        base_url = os.environ.get("NEBIUS_BASE_URL", DEFAULT_BASE_URL)
        chain = os.environ.get("NEBIUS_MODELS", "").strip()
        if chain:
            models = [m.strip() for m in chain.split(",") if m.strip()]
        else:
            single = os.environ.get("NEBIUS_MODEL", "").strip()
            models = [single] if single else list(DEFAULT_MODEL_CHAIN)
        return cls(models, api_key, base_url, **kwargs)

    def close(self) -> None:
        self._http.close()

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> ChatResult:
        """Walk the model chain until one rail answers or all fail."""
        failures: list[tuple[str, str]] = []
        for rail_index, model in enumerate(self.models):
            try:
                return self._call_rail(
                    rail_index, model, messages,
                    temperature=temperature, max_tokens=max_tokens,
                )
            except Exception as exc:  # noqa: BLE001 — failover must catch all
                failures.append((model, str(exc)))
                logger.warning("rail %d (%s) exhausted: %s — rotating",
                               rail_index, model, exc)
        raise ChainExhaustedError(failures)

    def _call_rail(
        self,
        rail_index: int,
        model: str,
        messages: list[dict],
        *,
        temperature: float,
        max_tokens: int,
    ) -> ChatResult:
        h = self.health[model]
        last_exc: Exception | None = None
        for attempt in range(1, self.rail_tries + 1):
            h["calls"] += 1
            started = time.monotonic()
            try:
                result = self._request(model, messages, temperature, max_tokens)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                h["failures"] += 1
                h["consecutive_failures"] += 1
                retryable = isinstance(exc, httpx.TransportError) or (
                    isinstance(exc, RailHTTPError) and exc.retryable
                )
                logger.info("rail %d (%s) attempt %d/%d failed (%s)%s",
                            rail_index, model, attempt, self.rail_tries,
                            type(exc).__name__,
                            "" if retryable else " — non-retryable, leaving rail")
                if not retryable or attempt == self.rail_tries:
                    break
                delay = min(
                    self.backoff_max,
                    self.backoff_base * (2 ** (attempt - 1)),
                ) * self._rng.uniform(0.5, 1.5)
                logger.info("rail %d (%s) backing off %.2fs", rail_index, model, delay)
                self._sleep(delay)
                continue
            h["consecutive_failures"] = 0
            h["last_latency"] = time.monotonic() - started
            result.rail_index = rail_index
            result.attempts = attempt
            logger.info("rail %d (%s) answered in %.2fs on attempt %d",
                        rail_index, model, h["last_latency"], attempt)
            return result
        raise RailExhaustedError(model, last_exc)

    def _request(
        self,
        model: str,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> ChatResult:
        resp = self._http.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        if resp.status_code != 200:
            raise RailHTTPError(resp.status_code, resp.text[:500])
        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RailHTTPError(
                200, f"malformed chat-completion body: {json.dumps(data)[:300]}"
            ) from exc
        return ChatResult(content=content, model=model, usage=data.get("usage") or {})


class RailHTTPError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self.retryable = status in RETRYABLE_STATUSES or status == 200
        super().__init__(f"HTTP {status}: {body}")


class RailExhaustedError(RuntimeError):
    def __init__(self, model: str, cause: Exception | None) -> None:
        self.model = model
        self.cause = cause
        super().__init__(f"rail {model} exhausted: {cause}")
