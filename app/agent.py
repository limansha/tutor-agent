"""Agentic MCQ chat: OpenRouter agent that produces a structured MCQAnswer.

Two evidence sources, selected by config.ANSWER_MODE:
  * 'web' (default) — pull results from a web-search backend and answer from them.
    The backend is config.WEB_SEARCH_PROVIDER: 'duckduckgo' (free, keyless; see
    app/websearch.py) or 'openrouter' (OpenRouter's paid `web` plugin; citations
    read from the raw completion because ChatOpenAI drops the annotations).
  * 'books' — an agent calls the RAG retrieval tool (same code path as
    POST /api/retrieve), receives the extracted chunk TEXT, and answers from it.

Model routing (primary -> fallbacks across OPENROUTER_API on APIError) is shared
in app/llm.with_llm_fallback.
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.factory import ToolStrategy
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.config import config
from app.llm import get_llm, get_openai_client, with_llm_fallback
from app.prompts import render
from app.retrieval.service import retrieve
from app.schemas import ChatFilters, ChunkMetadata, MCQAnswer, SourceChunk
from app import websearch

_SYSTEM_PROMPT = render("mcq_system.j2")
_WEB_SYSTEM_PROMPT = render("mcq_web.j2")

# json_schema response_format for web mode (strict, so the model must fill every
# field) — built once from the pydantic model so the two stay in sync.
_MCQ_JSON_SCHEMA = {
    "name": "MCQAnswer",
    "strict": True,
    "schema": convert_to_openai_tool(MCQAnswer, strict=True)["function"]["parameters"],
}


def _answer_once(
    model: str,
    question: str,
    options: dict[str, str],
    filters: ChatFilters | None = None,
    top_k: int = 8,
    parent_expansion: bool = True,
) -> tuple[MCQAnswer, list[SourceChunk]]:
    """Run one agent build+invoke on a single model, returning (answer, sources)."""
    used_sources: list[SourceChunk] = []

    @tool
    def search_study_notes(query: str, top_k: int = top_k) -> str:
        """Search the FRM study notes for context on `query`. Returns the relevant
        excerpts as plain text (no metadata). Call this before answering."""
        sources = retrieve(
            query,
            filters=filters,
            top_k=top_k,
            parent_expansion=parent_expansion,
        )
        used_sources.extend(sources)
        texts = [s.text.strip() for s in sources if s.text and s.text.strip()]
        return "\n\n".join(texts) if texts else "No relevant context found."

    llm = get_llm(model)
    agent = create_agent(
        model=llm,
        tools=[search_study_notes],
        system_prompt=_SYSTEM_PROMPT,
        response_format=ToolStrategy(MCQAnswer),
    )

    option_block = "\n".join(f"{label}. {text}" for label, text in options.items())
    result = agent.invoke(
        {"messages": [HumanMessage(content=f"{question}\n\nOptions:\n{option_block}")]}
    )

    structured = result.get("structured_response")
    if not isinstance(structured, MCQAnswer):
        raise RuntimeError(
            "Agent did not produce a structured response. Check OPENROUTER_MODEL "
            "supports tool calling + structured output."
        )
    return structured, used_sources


_MAX_CITATION_CHARS = 1200


def _citations_to_sources(annotations: object) -> list[SourceChunk]:
    """Map OpenRouter web-search `url_citation` annotations to SourceChunks.

    `annotations` is the raw SDK list (or None); each item may carry a
    `url_citation` with `url`/`title`/`content`. Duplicate URLs are collapsed.
    """
    sources: list[SourceChunk] = []
    seen: set[str] = set()
    for ann in annotations or []:
        citation = getattr(ann, "url_citation", None)
        url = getattr(citation, "url", None)
        if not url or url in seen:
            continue
        seen.add(url)
        excerpt = (getattr(citation, "content", None) or url)[:_MAX_CITATION_CHARS]
        sources.append(
            SourceChunk(
                role="web",
                text=excerpt,
                metadata=ChunkMetadata(
                    doc_url=url,
                    reading_title=getattr(citation, "title", None) or url,
                ),
            )
        )
    return sources


def _chat_json(
    model: str,
    messages: list[dict],
    extra_body: dict | None = None,
):
    """Structured web-mode completion; returns (MCQAnswer, raw message)."""
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": config.OPENROUTER_TEMPERATURE,
        "max_tokens": config.OPENROUTER_MAX_TOKENS,
        "response_format": {"type": "json_schema", "json_schema": _MCQ_JSON_SCHEMA},
    }
    if extra_body:
        kwargs["extra_body"] = extra_body
    completion = get_openai_client().chat.completions.create(**kwargs)
    message = completion.choices[0].message
    if not message.content:
        raise RuntimeError("Web-mode model returned no structured content.")
    return MCQAnswer.model_validate_json(message.content), message


def _answer_web_duckduckgo(
    model: str,
    question: str,
    options: dict[str, str],
) -> tuple[MCQAnswer, list[SourceChunk]]:
    """Free path: fetch DDG results, feed them in, answer from them."""
    results = websearch.search(
        question,
        max_results=config.WEB_SEARCH_MAX_RESULTS,
        exclude_domains=config.WEB_SEARCH_EXCLUDE_DOMAINS,
    )
    context = "\n\n".join(
        f"[{i}] {r['title']}\n{r['url']}\n{r['snippet']}"
        for i, r in enumerate(results, 1)
    ) or "(no web results found)"
    option_block = "\n".join(f"{label}. {text}" for label, text in options.items())
    answer, _ = _chat_json(
        model,
        [
            {"role": "system", "content": _WEB_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{question}\n\nOptions:\n{option_block}\n\n"
                    f"Web search results:\n{context}"
                ),
            },
        ],
    )
    sources = [
        SourceChunk(
            role="web",
            text=(r["snippet"] or r["url"])[:_MAX_CITATION_CHARS],
            metadata=ChunkMetadata(
                doc_url=r["url"], reading_title=r["title"] or r["url"]
            ),
        )
        for r in results
    ]
    return answer, sources


def _answer_web_openrouter(
    model: str,
    question: str,
    options: dict[str, str],
) -> tuple[MCQAnswer, list[SourceChunk]]:
    """Paid path: OpenRouter `web` plugin; sources from url_citation annotations."""
    option_block = "\n".join(f"{label}. {text}" for label, text in options.items())
    answer, message = _chat_json(
        model,
        [
            {"role": "system", "content": _WEB_SYSTEM_PROMPT},
            {"role": "user", "content": f"{question}\n\nOptions:\n{option_block}"},
        ],
        extra_body={
            "plugins": [
                {
                    "id": "web",
                    "max_results": config.WEB_SEARCH_MAX_RESULTS,
                    "exclude_domains": config.WEB_SEARCH_EXCLUDE_DOMAINS,
                }
            ]
        },
    )
    return answer, _citations_to_sources(getattr(message, "annotations", None))


def _answer_web_once(
    model: str,
    question: str,
    options: dict[str, str],
) -> tuple[MCQAnswer, list[SourceChunk]]:
    """One web-mode call on a single model, backend chosen by config."""
    if config.WEB_SEARCH_PROVIDER == "openrouter":
        return _answer_web_openrouter(model, question, options)
    return _answer_web_duckduckgo(model, question, options)


def answer_mcq(
    question: str,
    options: dict[str, str],
    filters: ChatFilters | None = None,
    top_k: int = 8,
    parent_expansion: bool = True,
) -> tuple[MCQAnswer, list[SourceChunk]]:
    """Run the agent and return (structured MCQAnswer, retrieved sources used)."""
    if not options:
        raise ValueError("options must not be empty")

    if config.ANSWER_MODE == "web":
        return with_llm_fallback(lambda model: _answer_web_once(model, question, options))

    return with_llm_fallback(
        lambda model: _answer_once(
            model, question, options, filters, top_k, parent_expansion
        )
    )