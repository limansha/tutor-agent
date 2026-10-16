"""Agentic MCQ chat: OpenRouter agent that produces a structured MCQAnswer.

The agent has one tool, `web_search` (DuckDuckGo Lite scrape, free and keyless —
see app/websearch.py). It decides what to search, calls the tool, and answers
from the returned excerpts. The RAG retrieval code (app/retrieval/*) still exists
and is served by POST /api/retrieve, but the MCQ path does not use it: books are
not being ingested right now (memory limits), so the tool is web-based.

Model routing (primary -> fallbacks across OPENROUTER_API on APIError) is shared
in app/llm.with_llm_fallback.
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.factory import ToolStrategy
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from app.config import config
from app.llm import get_llm, with_llm_fallback
from app.prompts import render
from app.schemas import ChunkMetadata, MCQAnswer, SourceChunk
from app import websearch

_SYSTEM_PROMPT = render("mcq_web.j2")

_MAX_SOURCE_CHARS = 1200


def _result_sources(
    results: list[dict[str, str]], seen_urls: set[str]
) -> list[SourceChunk]:
    """Map web_search results to de-duplicated `role="web"` SourceChunks."""
    out: list[SourceChunk] = []
    for r in results:
        url = r["url"]
        if url in seen_urls:
            continue
        seen_urls.add(url)
        out.append(
            SourceChunk(
                role="web",
                text=(r["snippet"] or url)[:_MAX_SOURCE_CHARS],
                metadata=ChunkMetadata(doc_url=url, reading_title=r["title"] or url),
            )
        )
    return out


def _answer_once(
    model: str,
    question: str,
    options: dict[str, str],
) -> tuple[MCQAnswer, list[SourceChunk]]:
    """Run one agent build+invoke on a single model, returning (answer, sources)."""
    used_sources: list[SourceChunk] = []
    seen_urls: set[str] = set()

    @tool
    def web_search(query: str) -> str:
        """Search the web for context on `query`. Returns title/url/snippet
        excerpts as plain text. Call this before answering."""
        results = websearch.search(
            query,
            max_results=config.WEB_SEARCH_MAX_RESULTS,
            exclude_domains=config.WEB_SEARCH_EXCLUDE_DOMAINS,
        )
        if not results:
            return "No web results found."
        used_sources.extend(_result_sources(results, seen_urls))
        return "\n\n".join(
            f"[{i}] {r['title']}\n{r['url']}\n{r['snippet']}"
            for i, r in enumerate(results, 1)
        )

    llm = get_llm(model)
    agent = create_agent(
        model=llm,
        tools=[web_search],
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


def answer_mcq(
    question: str,
    options: dict[str, str],
) -> tuple[MCQAnswer, list[SourceChunk]]:
    """Run the agent and return (structured MCQAnswer, web sources used)."""
    if not options:
        raise ValueError("options must not be empty")
    return with_llm_fallback(lambda model: _answer_once(model, question, options))
