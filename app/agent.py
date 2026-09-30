"""Agentic MCQ chat: OpenRouter agent that uses the RAG retrieval service as a tool.

Flow: question + multiple-choice options in -> the agent decides a search query,
calls the retrieval tool (same code path as POST /api/retrieve), receives only the
extracted chunk TEXT, and produces a structured MCQAnswer (correct option + why
each other option is wrong).

Model routing (primary -> fallbacks across OPENROUTER_API on APIError) is shared
in app/llm.with_llm_fallback.
"""

from __future__ import annotations

from app.llm import get_llm, with_llm_fallback
from app.prompts import render
from app.retrieval.service import retrieve
from app.schemas import ChatFilters, MCQAnswer, SourceChunk
from langchain.agents import create_agent
from langchain.agents.factory import ToolStrategy
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

_SYSTEM_PROMPT = render("mcq_system.j2")


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

    return with_llm_fallback(
        lambda model: _answer_once(
            model, question, options, filters, top_k, parent_expansion
        )
    )