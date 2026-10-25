"""Model construction for the app's OpenAI-compatible providers."""

from __future__ import annotations

from langchain.chat_models import init_chat_model

from app.config import config
from app.llm.providers import LLMSpec, provider_specs


def get_llm(
    spec: LLMSpec | str | None = None,
    *,
    purpose: str = "mcq",
):
    """Return a model-agnostic LangChain chat model for an OpenAI-compatible API.

    An LLMSpec (from app.llm.providers.provider_specs) targets that provider. If
    omitted, the first configured provider is used.
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
    specs = provider_specs()
    if not specs:
        raise RuntimeError("no LLM provider is configured")
    spec = specs[0]
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
