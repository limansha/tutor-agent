"""Gemini LLM configuration."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import config

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


@dataclass(frozen=True)
class LLMSpec:
    """Gemini model connection details."""

    model: str
    base_url: str
    api_key: str


def gemini_spec() -> LLMSpec:
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return LLMSpec(config.GEMINI_MODEL, GEMINI_BASE_URL, config.GEMINI_API_KEY)
