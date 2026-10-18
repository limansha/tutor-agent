"""Model construction: turn an LLMSpec (or a bare OpenRouter model name) into a
ready ChatOpenAI bound to the right endpoint.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.config import config
from app.llm.providers import LLMSpec, OPENROUTER_BASE_URL


def get_llm(spec: LLMSpec | str | None = None) -> ChatOpenAI:
    """Return a ChatOpenAI bound to `spec`'s OpenAI-compatible endpoint.

    An LLMSpec (from app.llm.providers.provider_specs) targets that provider. A
    bare string or None means OpenRouter with that model / config.OPENROUTER_MODEL.
    """
    if isinstance(spec, LLMSpec):
        return ChatOpenAI(
            model=spec.model,
            temperature=config.OPENROUTER_TEMPERATURE,
            max_tokens=config.OPENROUTER_MAX_TOKENS,
            api_key=spec.api_key,
            base_url=spec.base_url,
        )
    if not config.OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to .env to use /api/mcq-answer."
        )
    return ChatOpenAI(
        model=spec or config.OPENROUTER_MODEL,
        temperature=config.OPENROUTER_TEMPERATURE,
        max_tokens=config.OPENROUTER_MAX_TOKENS,
        api_key=config.OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )
