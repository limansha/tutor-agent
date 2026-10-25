"""Cross-provider failover with rate-limit-aware rotation.

Failover wraps the whole *operation* (a structured-output call, a plain LLM
call), not a single returned model: a structured call needs its schema bound to
one provider's endpoint, and LangChain's RunnableWithFallbacks cannot carry that
binding across providers. Hence a runner, not a fused model.

A provider that returns 429 is benched in a small cache until its next-available
time (its Retry-After, else _DEFAULT_COOLDOWN). The next call promotes the next
provider to primary; when several are available the canonical order
    (Groq -> OpenRouter -> Gemini) holds. A benched provider is still tried last, so
a wrongly-guessed cooldown can't lock it out for good — and a success clears it.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

from openai import APIError

from app.llm.providers import LLMSpec, provider_specs

logger = logging.getLogger(__name__)

_402_RETRIES = 2  # in-flight budget is account-wide; retry same model, not a fallback
_DEFAULT_COOLDOWN = 60.0  # seconds to bench a provider that 429s without Retry-After

# ponytail: in-process cache only — each gunicorn worker rotates independently;
# move to Redis/Postgres if the fleet needs one shared view of rate limits.
_cooldown_until: dict[tuple[str, str], float] = {}
_lock = threading.Lock()


def _key(spec: LLMSpec) -> tuple[str, str]:
    return (spec.model, spec.base_url)


def _retry_after_seconds(exc: APIError) -> int:
    resp = getattr(exc, "response", None)
    if resp is None:
        return 0
    try:
        return int(resp.headers.get("retry-after", "0"))
    except (TypeError, ValueError):
        return 0


def _available(spec: LLMSpec, now: float) -> bool:
    with _lock:
        return _cooldown_until.get(_key(spec), 0.0) <= now


def _bench(spec: LLMSpec, seconds: float) -> None:
    with _lock:
        _cooldown_until[_key(spec)] = time.monotonic() + seconds


def _clear(spec: LLMSpec) -> None:
    with _lock:
        _cooldown_until.pop(_key(spec), None)


def _ordered_chain(chain_fn: Callable[[], list[LLMSpec]]) -> list[LLMSpec]:
    """Available providers first (canonical order), then the benched ones."""
    now = time.monotonic()
    chain = chain_fn()
    return [s for s in chain if _available(s, now)] + [
        s for s in chain if not _available(s, now)
    ]


def with_llm_fallback(
    call: Callable[[LLMSpec], Any],
    retry_on: tuple[type[BaseException], ...] = (),
    chain_fn: Callable[[], list[LLMSpec]] = provider_specs,
) -> Any:
    """Run `call(spec)` down the rotation until one succeeds.

    `call` receives an LLMSpec and raises openai.APIError on provider failure;
    `retry_on` adds exception types that also mean "this provider's output was
    unusable, try the next" (e.g. OutputParserException from a structured call
    that returned junk). A 429 benches that provider for its Retry-After (or the
    default); if every hop fails, the last error is re-raised.
    """
    retriable = (APIError, *retry_on)
    last_error: BaseException | None = None
    chain = _ordered_chain(chain_fn)
    if not chain:
        raise RuntimeError(
            "no LLM configured — set a provider key in .env"
        )
    for spec in chain:
        attempts = 1 + (_402_RETRIES if spec == chain[0] else 0)
        for _ in range(attempts):
            try:
                result = call(spec)
                _clear(spec)
                return result
            except retriable as exc:
                last_error = exc
                status = getattr(exc, "status_code", None)
                if status == 402:
                    wait = _retry_after_seconds(exc) or 5
                    logger.warning(
                        "OpenRouter in-flight budget exhausted on %s; retry in %ss",
                        spec.model,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                if status == 429:
                    wait = _retry_after_seconds(exc) or _DEFAULT_COOLDOWN
                    _bench(spec, wait)
                    logger.warning(
                        "429 from %s (%s); benching %ss, trying next",
                        spec.model,
                        spec.base_url,
                        wait,
                    )
                break
        if last_error is not None and getattr(last_error, "status_code", None) not in (
            402,
            429,
        ):
            logger.warning(
                "LLM call failed on %s (%s): %s — trying next",
                spec.model,
                spec.base_url,
                last_error,
            )
    if last_error is not None:
        raise last_error
    raise RuntimeError("no LLM configured")
