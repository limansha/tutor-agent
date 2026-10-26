"""OpenRouter LLM access."""

from app.llm.factory import get_llm
from app.llm.providers import LLMSpec, openrouter_spec

__all__ = [
    "get_llm",
    "LLMSpec",
    "openrouter_spec",
]
