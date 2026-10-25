"""Provider registry: the OpenAI-compatible endpoints we can reach.

Every provider here speaks the OpenAI API, so ChatOpenAI reaches all of them with
just a base_url + key — no provider-specific SDK. Swap a slug in .env if a
provider retires a model; the slug must support structured output
(response_format json_schema — the MCQ answer path uses it).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import config

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


@dataclass(frozen=True)
class LLMSpec:
    """One reachable model: its OpenAI-compatible endpoint + credentials."""

    model: str
    base_url: str
    api_key: str


_PROVIDER_ORDER = ("groq", "openrouter", "gemini")


def _ordered_specs(
    order: tuple[str, ...], model_overrides: dict[str, str] | None = None
) -> list[LLMSpec]:
    providers = {
        "openrouter": (OPENROUTER_BASE_URL, config.OPENROUTER_API_KEY, config.OPENROUTER_MODEL),
        "groq": (GROQ_BASE_URL, config.GROQ_API_KEY, config.GROQ_MODEL),
        "gemini": (GEMINI_BASE_URL, config.GEMINI_API_KEY, config.GEMINI_MODEL),
    }
    model_overrides = model_overrides or {}
    specs = []
    for name in order:
        base, key, model = providers[name]
        model = model_overrides.get(name, model)
        if key and model:
            specs.append(LLMSpec(model, base, key))
    return list(dict.fromkeys(specs))


def provider_specs() -> list[LLMSpec]:
    """Common ordered chain: Groq -> OpenRouter -> Gemini.

    A provider with a blank key is skipped.
    """
    return _ordered_specs(_PROVIDER_ORDER)


def ingest_provider_specs() -> list[LLMSpec]:
    """Use the common provider chain with an optional ingestion Groq model override."""
    overrides = {"groq": config.INGEST_MODEL} if config.INGEST_MODEL else None
    return _ordered_specs(_PROVIDER_ORDER, overrides)
