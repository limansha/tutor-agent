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


def provider_specs() -> list[LLMSpec]:
    """Ordered, de-duplicated chain: OpenRouter primary, then Groq -> Gemini.

    A provider with a blank key is skipped, so the chain degrades to whatever is
    configured. The cross-provider hops have independent quotas, which is what
    lets the app survive OpenRouter's account-wide free-model daily cap (429).
    """
    specs: list[LLMSpec] = []
    if config.OPENROUTER_API_KEY:
        specs.append(
            LLMSpec(config.OPENROUTER_MODEL, OPENROUTER_BASE_URL, config.OPENROUTER_API_KEY)
        )
    for key, model, base in (
        (config.GROQ_API_KEY, config.GROQ_MODEL, GROQ_BASE_URL),
        (config.GEMINI_API_KEY, config.GEMINI_MODEL, GEMINI_BASE_URL),
    ):
        if key and model:
            specs.append(LLMSpec(model, base, key))
    return list(dict.fromkeys(specs))
