"""Model construction for the app's OpenAI-compatible providers."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.config import config
from app.llm.providers import LLMSpec, openrouter_spec


def get_llm(
    spec: LLMSpec | str | None = None,
    *,
    purpose: str = "mcq",
):
    """Return a model-agnostic LangChain chat model for an OpenAI-compatible API.

    If omitted, the configured OpenRouter model is used.
    """
    if purpose == "ingest":
        temperature = config.INGEST_TEMPERATURE
        max_tokens = config.INGEST_MAX_TOKENS
        timeout = config.INGEST_REQUEST_TIMEOUT
    elif purpose == "mcq":
        temperature = config.MCQ_TEMPERATURE
        max_tokens = config.MCQ_MAX_TOKENS
        timeout = None
    else:
        raise ValueError(f"unknown LLM purpose: {purpose}")

    spec = spec if isinstance(spec, LLMSpec) else openrouter_spec()
    return ChatOpenAI(
        model=spec.model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=spec.api_key,
        base_url=spec.base_url,
        **({"timeout": timeout} if timeout is not None else {}),
            **({"max_retries": 0} if purpose == "ingest" else {}),
    )
