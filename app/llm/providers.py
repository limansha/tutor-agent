"""OpenRouter LLM configuration."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import config

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class LLMSpec:
    """OpenAI-compatible model connection details."""

    model: str
    base_url: str
    api_key: str


def openrouter_spec() -> LLMSpec:
    if not config.OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return LLMSpec(
        config.OPENROUTER_MODEL, OPENROUTER_BASE_URL, config.OPENROUTER_API_KEY
    )
