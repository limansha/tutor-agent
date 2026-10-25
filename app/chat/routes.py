"""POST /api/mcq-answer — web search + one structured OpenRouter call."""

from __future__ import annotations

import logging
import time

from openai import APIError
from flask import Blueprint, jsonify, request
from pydantic import ValidationError

from app.agent import answer_mcq
from app.schemas import MCQChatRequest, MCQChatResponse, ServiceUnavailable

logger = logging.getLogger(__name__)

bp = Blueprint("chat", __name__)


@bp.post("/api/mcq-answer")
def chat():
    try:
        body = MCQChatRequest.model_validate(request.get_json(force=True))
    except ValidationError as exc:
        return jsonify({"error": "invalid request", "detail": exc.errors()}), 400
    try:
        start = time.monotonic()
        answer, sources, spec = answer_mcq(body.question, body.options)
    except APIError as exc:
        logger.exception("mcq-answer: provider error (route elapsed %.3fs)", time.monotonic() - start)
        return jsonify(
            ServiceUnavailable(error=str(exc), hint="OpenRouter provider error; retry shortly.").model_dump()
        ), 503
    except (RuntimeError, ValueError) as exc:
        logger.exception("mcq-answer: no usable answer (route elapsed %.3fs)", time.monotonic() - start)
        return jsonify(
            ServiceUnavailable(
                error=str(exc),
                hint="Set a Groq, OpenRouter, or Gemini provider key, then restart.",
            ).model_dump()
        ), 503

    elapsed_ms = int((time.monotonic() - start) * 1000)
    logger.info("mcq-answer: route total %.3fs (model=%s)", elapsed_ms / 1000.0, spec.model)
    response = MCQChatResponse(
        question=body.question,
        answer=answer,
        sources=sources if body.include_sources else None,
        model=spec.model,
        elapsed_ms=elapsed_ms,
    )
    return jsonify(response.model_dump(exclude_none=True)), 200
