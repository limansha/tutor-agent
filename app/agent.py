"""MCQ answer: one DuckDuckGo search + one structured LLM call (no agent loop).

The search runs once per request (keyless DDG-Lite scrape — see app/websearch.py),
its excerpts are pasted into the prompt, and a single `with_structured_output`
call returns an MCQAnswer. Dropping `create_agent`/tools removes every
tool-calling failure mode seen when switching models: Gemini's thought_signature
400 on the tool round-trip, gpt-oss mis-calling tool args, and the
response_format + tool-calling conflict behind the old ToolStrategy workaround.
A parse failure on one provider now fails over to the next, like an API error.

Model routing (OpenRouter primary -> Groq -> Gemini) is shared in
app/llm.with_llm_fallback."""

from __future__ import annotations

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import config
from app.llm import LLMSpec, get_llm, with_llm_fallback
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


def _search(question: str) -> tuple[str, list[SourceChunk]]:
    """One DDG search for `question` -> (excerpt block for the prompt, sources)."""
    results = websearch.search(
        question,
        max_results=config.WEB_SEARCH_MAX_RESULTS,
        exclude_domains=config.WEB_SEARCH_EXCLUDE_DOMAINS,
    )
    if not results:
        return "No web excerpts found.", []
    block = "\n\n".join(
        f"[{i}] {r['title']}\n{r['url']}\n{r['snippet']}"
        for i, r in enumerate(results, 1)
    )
    return block, _result_sources(results, set())


def _answer_once(
    spec: LLMSpec,
    question: str,
    options: dict[str, str],
    excerpts: str,
) -> MCQAnswer:
    """One structured call on one model; raises APIError/OutputParserException."""
    option_block = "\n".join(f"{label}. {text}" for label, text in options.items())
    prompt = (
        f"Web search excerpts:\n{excerpts}\n\n"
        f"{question}\n\nOptions:\n{option_block}"
    )
    structured = get_llm(spec).with_structured_output(MCQAnswer)
    return structured.invoke(
        [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )


def answer_mcq(
    question: str,
    options: dict[str, str],
) -> tuple[MCQAnswer, list[SourceChunk], LLMSpec]:
    """Search once, then answer; returns (MCQAnswer, web sources, winning spec)."""
    if not options:
        raise ValueError("options must not be empty")
    excerpts, sources = _search(question)
    winner: list[LLMSpec] = []

    def _run(spec: LLMSpec) -> MCQAnswer:
        winner.append(spec)
        return _answer_once(spec, question, options, excerpts)

    answer = with_llm_fallback(_run, retry_on=(OutputParserException,))
    return answer, sources, winner[-1]
