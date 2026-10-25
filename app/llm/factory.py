"""Model construction for the app's OpenAI-compatible providers."""

from __future__ import annotations

from langchain.chat_models import init_chat_model

from app.config import config
from app.llm.providers import LLMSpec, OPENROUTER_BASE_URL


def get_llm(
    spec: LLMSpec | str | None = None,
    *,
    purpose: str = "mcq",
):
    """Return a model-agnostic LangChain chat model for an OpenAI-compatible API.

    An LLMSpec (from app.llm.providers.provider_specs) targets that provider. A
    bare string or None means OpenRouter with that model / config.OPENROUTER_MODEL.
    """
    if purpose == "ingest":
        temperature = config.INGEST_TEMPERATURE
        max_tokens = config.INGEST_MAX_TOKENS
        timeout = config.INGEST_REQUEST_TIMEOUT
    elif purpose == "mcq":
        temperature = config.OPENROUTER_TEMPERATURE
        max_tokens = config.OPENROUTER_MAX_TOKENS
        timeout = None
    else:
        raise ValueError(f"unknown LLM purpose: {purpose}")

    if isinstance(spec, LLMSpec):
        return init_chat_model(
            model=spec.model,
            model_provider="openai",
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=spec.api_key,
            base_url=spec.base_url,
            **({"timeout": timeout} if timeout is not None else {}),
            **({"max_retries": 0} if purpose == "ingest" else {}),
        )
    if not config.OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to .env to use /api/mcq-answer."
        )
    return init_chat_model(
        model=spec or config.OPENROUTER_MODEL,
        model_provider="openai",
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=config.OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        **({"timeout": timeout} if timeout is not None else {}),
        **({"max_retries": 0} if purpose == "ingest" else {}),
    )
