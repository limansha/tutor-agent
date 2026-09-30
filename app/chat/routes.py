"""POST /api/mcq-answer — agentic MCQ answer via OpenRouter + RAG retrieval tool."""

from __future__ import annotations

import time

from openai import APIError
from flask import Blueprint, jsonify, request
from pydantic import ValidationError

from app.agent import answer_mcq
from app.config import config
from app.schemas import MCQChatRequest, MCQChatResponse, ServiceUnavailable

bp = Blueprint("chat", __name__)


@bp.post("/api/mcq-answer")
def chat():
    try:
        body = MCQChatRequest.model_validate(request.get_json(force=True))
    except ValidationError as exc:
        return jsonify({"error": "invalid request", "detail": exc.errors()}), 400
    try:
        start = time.monotonic()
        answer, sources = answer_mcq(
            body.question,
            body.options,
            filters=body.filters,
            top_k=body.top_k,
            parent_expansion=body.parent_expansion,
        )
    except APIError as exc:
        return jsonify(
            ServiceUnavailable(error=str(exc), hint="OpenRouter provider error; retry shortly.").model_dump()
        ), 503
    except (RuntimeError, ValueError) as exc:
        return jsonify(
            ServiceUnavailable(error=str(exc), hint="Set OPENROUTER_API_KEY in .env and restart.").model_dump()
        ), 503

    elapsed_ms = int((time.monotonic() - start) * 1000)
    response = MCQChatResponse(
        question=body.question,
        answer=answer,
        sources=sources if body.include_sources else None,
        model=config.OPENROUTER_MODEL,
        elapsed_ms=elapsed_ms,
    )
    return jsonify(response.model_dump(exclude_none=True)), 200
