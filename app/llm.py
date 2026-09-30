"""OpenRouter LLM factory + model-chain failover routing (shared by all LLM flows).

The routing lives here so every caller (agent MCQ, question ingestion, ...) gets
the same behaviour: try OPENROUTER_MODEL, fall back through
OPENROUTER_FALLBACK_MODELS on OpenRouter APIError (credit/rate/provider), and
raise the last APIError if every model fails.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from langchain_openai import ChatOpenAI
from openai import APIError

from app.config import config

logger = logging.getLogger(__name__)

_402_RETRIES = 2  # in-flight budget is account-wide; retry same model, not a fallback


def _retry_after_seconds(exc: APIError) -> int:
    resp = getattr(exc, "response", None)
    if resp is None:
        return 0
    try:
        return int(resp.headers.get("retry-after", "0"))
    except (TypeError, ValueError):
        return 0


def get_llm(model: str | None = None) -> ChatOpenAI:
    """Return a configured ChatOpenAI instance pointing at OpenRouter.

    `model` selects the specific model; defaults to config.OPENROUTER_MODEL.
    """
    if not config.OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to .env to use /api/mcq-answer."
        )
    return ChatOpenAI(
        model=model or config.OPENROUTER_MODEL,
        temperature=config.OPENROUTER_TEMPERATURE,
        max_tokens=config.OPENROUTER_MAX_TOKENS,
        api_key=config.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )


def model_chain() -> list[str]:
    """Ordered, de-duplicated list: primary model first, then fallbacks."""
    return list(dict.fromkeys(
        m for m in [config.OPENROUTER_MODEL, *config.OPENROUTER_FALLBACK_MODELS] if m
    ))


def with_llm_fallback(call: Callable[[str], Any]) -> Any:
    """Run `call(model)` across model_chain(); return first success.

    `call` receives a model name and must raise openai.APIError on provider
    failure. If every model raises APIError, the last error is re-raised.
    """
    last_error: APIError | None = None
    chain = model_chain()
    for model in chain:
        attempts = 1 + (_402_RETRIES if model == chain[0] else 0)
        for _ in range(attempts):
            try:
                return call(model)
            except APIError as exc:
                last_error = exc
                if getattr(exc, "status_code", None) == 402:
                    wait = _retry_after_seconds(exc) or 5
                    logger.warning("OpenRouter in-flight budget exhausted on %s; retry in %ss", model, wait)
                    time.sleep(wait)
                    continue
                break
        if last_error is not None and getattr(last_error, "status_code", None) != 402:
            logger.warning("OpenRouter call failed on %s: %s — trying next model", model, last_error)
    if last_error is not None:
        raise last_error
    raise RuntimeError("no LLM model configured in OPENROUTER_MODEL / OPENROUTER_FALLBACK_MODELS")