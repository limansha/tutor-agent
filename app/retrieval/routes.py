"""POST /api/retrieve + GET /docs/<file> — RAG chunk retrieval and source links."""

from __future__ import annotations

import os
import time
from urllib.parse import unquote

from flask import Blueprint, jsonify, request, send_from_directory
from pydantic import ValidationError

from app import db
from app.config import config
from app.retrieval.service import retrieve
from app.schemas import RetrieveRequest, RetrieveResponse, ServiceUnavailable

bp = Blueprint("retrieval", __name__)


@bp.post("/api/retrieve")
def retrieve_endpoint():
    try:
        body = RetrieveRequest.model_validate(request.get_json(force=True))
    except ValidationError as exc:
        return jsonify({"error": "invalid request", "detail": exc.errors()}), 400
    try:
        start = time.monotonic()
        sources = retrieve(
            body.query,
            filters=body.filters,
            top_k=body.top_k,
            parent_expansion=body.parent_expansion,
            section_expansion=body.section_expansion,
        )
    except db.DatabaseNotReady as exc:
        return jsonify(
            ServiceUnavailable(error=str(exc), hint=db.readiness_hint()).model_dump()
        ), 503

    elapsed_ms = int((time.monotonic() - start) * 1000)
    response = RetrieveResponse(
        query=body.query,
        sources=sources,
        elapsed_ms=elapsed_ms,
    )
    return jsonify(response.model_dump()), 200


@bp.get("/docs/<path:filename>")
def docs(filename: str):
    """Serve the original PDF so doc_url references open the right page (#page=N)."""
    safe = unquote(filename)
    if not os.path.exists(os.path.join(config.BOOKS_DIR, safe)):
        return jsonify({"error": "file not found"}), 404
    return send_from_directory(config.BOOKS_DIR, safe)


@bp.get("/api/health")
def health():
    ready = db.is_ready()
    return jsonify(
        {
            "status": "ok" if ready else "degraded",
            "database_ready": ready,
            "database_url_set": bool(config.DATABASE_URL),
            "openrouter_key_set": bool(config.OPENROUTER_API_KEY),
            "embedding_model": config.EMBEDDING_MODEL,
            "embedding_dim": config.EMBEDDING_DIM,
            "books_dir": config.BOOKS_DIR,
            "hint": None if ready else db.readiness_hint(),
        }
    ), 200