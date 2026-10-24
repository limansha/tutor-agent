"""LLM access — provider registry + cross-provider failover.

Public surface:
    get_llm(spec=None)       -> a provider-agnostic chat model
    with_llm_fallback(call)  -> run `call(spec)` across the provider chain

`with_llm_fallback` is why this is not a single method: a structured-output call
must bind its response schema to one provider's endpoint, so failover wraps the
whole operation rather than a single returned model.
"""

from app.llm.factory import get_llm
from app.llm.providers import LLMSpec, ingest_provider_specs, provider_specs
from app.llm.routing import with_llm_fallback

__all__ = [
    "get_llm",
    "with_llm_fallback",
    "LLMSpec",
    "provider_specs",
    "ingest_provider_specs",
]
