"""MCQ answer: one DuckDuckGo search + one plain-text LLM call (no agent loop).

The search runs once per request (keyless DDG-Lite scrape — see app/websearch.py),
its excerpts are pasted into the prompt, and Python builds the MCQAnswer from
a small plain-text response contract. Dropping `create_agent`/tools removes every
tool-calling failure modes seen with provider-specific APIs.
400 on the tool round-trip, gpt-oss mis-calling tool args, and the
response_format + tool-calling conflict behind the old ToolStrategy workaround.
The configured OpenRouter model handles the single LLM call."""

from __future__ import annotations

import logging
import re
import time

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import config
from app.llm import LLMSpec, get_llm, openrouter_spec
from app.prompts import render
from app.schemas import ChunkMetadata, MCQAnswer, SourceChunk
from app import websearch

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = render("mcq_web.j2")

_MAX_SOURCE_CHARS = 1200
_MAX_EXPLANATION_WORDS = 200


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
    """Call one model and build the response schema in Python."""
    option_block = "\n".join(f"{label}. {text}" for label, text in options.items())
    prompt = (
        f"Web search excerpts:\n{excerpts}\n\n"
        f"{question}\n\nOptions:\n{option_block}"
    )
    response = get_llm(spec).invoke(
        [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    content = response.content
    if not isinstance(content, str):
        raise OutputParserException("LLM response was not plain text")

    answer_match = re.search(
        r"^ANSWER:\s*([A-Za-z0-9]+)\s*$", content, re.MULTILINE | re.IGNORECASE
    )
    explanation_match = re.search(
        r"^EXPLANATION:\s*(.*?)(?=^REJECTIONS:\s*$|\Z)",
        content,
        re.MULTILINE | re.IGNORECASE | re.DOTALL,
    )
    rejections_match = re.search(
        r"^REJECTIONS:\s*$\n?(.*)\Z", content, re.MULTILINE | re.IGNORECASE | re.DOTALL
    )
    if not (answer_match and explanation_match and rejections_match):
        raise OutputParserException("LLM response did not follow the plain-text answer format")

    label = answer_match.group(1).upper()
    normalized_options = {key.upper(): value for key, value in options.items()}
    if label not in normalized_options:
        raise OutputParserException(f"LLM selected unknown option: {label}")

    explanation = " ".join(explanation_match.group(1).split()[:_MAX_EXPLANATION_WORDS])
    reasons: dict[str, str] = {}
    for match in re.finditer(
        r"^([A-Za-z0-9]+):\s*(.+?)(?=^\s*[A-Za-z0-9]+:\s*|\Z)",
        rejections_match.group(1),
        re.MULTILINE | re.DOTALL,
    ):
        reasons[match.group(1).upper()] = " ".join(match.group(2).split())
    missing = set(normalized_options) - {label} - set(reasons)
    if missing or not explanation:
        raise OutputParserException("LLM response omitted an option rejection or explanation")

    return MCQAnswer(
        selected_option_label=label,
        selected_option_text=normalized_options[label],
        explanation=explanation,
        rejections=[
            {"option_label": key, "option_text": text, "reason": reasons[key]}
            for key, text in normalized_options.items()
            if key != label
        ],
    )


def answer_mcq(
    question: str,
    options: dict[str, str],
) -> tuple[MCQAnswer, list[SourceChunk], LLMSpec]:
    """Search once, then answer; returns (MCQAnswer, web sources, winning spec)."""
    if not options:
        raise ValueError("options must not be empty")
    t_total = time.monotonic()
    t_search = time.monotonic()
    excerpts, sources = _search(question)
    logger.info(
        "mcq-answer: websearch took %.3fs, sources=%d",
        time.monotonic() - t_search,
        len(sources),
    )
    spec = openrouter_spec()
    t_llm = time.monotonic()
    answer = _answer_once(spec, question, options, excerpts)
    logger.info("mcq-answer: OpenRouter call took %.3fs", time.monotonic() - t_llm)
    logger.info(
        "mcq-answer: total took %.3fs, model=%s",
        time.monotonic() - t_total,
        spec.model,
    )
    return answer, sources, spec
