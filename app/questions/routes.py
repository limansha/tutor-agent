"""POST /api/questions — fetch practice questions by requested topics.

`topics` are matched exactly against the stored question_topics array (GIN);
`search_topics` are matched semantically (cosine over frm_question_topics
embeddings) and only run when provided. Results are merged, deduped by qid and
returned in batches of page_size. Requested topics with no hit are reported in
uncovered_topics.
"""

from __future__ import annotations

import time

from flask import Blueprint, jsonify, request
from pydantic import ValidationError

from app import db
from app.config import config
from app.embeddings import embed_documents
from app.schemas import (
    AnswerCheckRequest,
    AnswerCheckResponse,
    AnswerCheckResult,
    QuestionsRequest,
    QuestionsResponse,
    ServiceUnavailable,
    TopicMatch,
    UncoveredTopic,
)

bp = Blueprint("questions", __name__)

_Q_COLS = (
    "qid", "question_text", "options", "answer", "has_answer_in_doc",
    "question_topics", "prerequisites", "pdf_name", "source_path", "page_number",
)

_EXACT_SQL = f"""
    SELECT {", ".join(_Q_COLS)}
    FROM frm_questions
    WHERE question_topics && %s::text[]
    ORDER BY qid
"""

_SEMANTIC_SQL = f"""
    SELECT {", ".join(_Q_COLS)}, 1 - (qt.embedding <=> %s::vector) AS score
    FROM frm_question_topics qt
    JOIN frm_questions q USING (qid)
    ORDER BY qt.embedding <=> %s::vector
    LIMIT %s
"""

_ALL_SQL = f"""
    SELECT {", ".join(_Q_COLS)}
    FROM frm_questions
    ORDER BY qid
    LIMIT %s OFFSET %s
"""

_COUNT_SQL = "SELECT count(*) FROM frm_questions"


def _emb_sql(vec: list[float]) -> str:
    return f"[{','.join(f'{x:.6f}' for x in vec)}]"


def _exact_rows(cur, topics: list[str]) -> tuple[list[dict], set[str]]:
    """Rows whose question_topics literally contain a requested topic (GIN)."""
    if not topics:
        return [], set()
    cur.execute(_EXACT_SQL, (topics,))
    rows = [dict(zip(_Q_COLS, row)) for row in cur.fetchall()]
    matched = {t for row in rows for t in row["question_topics"] if t in topics}
    return rows, matched


def _semantic_rows(cur, topics: list[str], limit: int) -> tuple[list[dict], set[str]]:
    """Rows whose stored topic embedding is near a requested search topic."""
    if not topics:
        return [], set()
    rows: list[dict] = []
    matched: set[str] = set()
    for topic, vec in zip(topics, embed_documents(topics)):
        cur.execute(_SEMANTIC_SQL, (_emb_sql(vec), _emb_sql(vec), limit))
        hits = []
        for row in cur.fetchall():
            d = dict(zip(_Q_COLS + ("score",), row))
            if d.pop("score") >= config.TOPIC_MATCH_THRESHOLD:
                hits.append(d)
        if hits:
            rows.extend(hits)
            matched.add(topic)
    return rows, matched


def _collect(cur, topics: list[str], search_topics: list[str], limit: int):
    """Merge exact + semantic matches, dedup by qid, report matched/uncovered topics."""
    exact_rows, exact_matched = _exact_rows(cur, topics)
    sem_rows, sem_matched = _semantic_rows(cur, search_topics, limit)

    seen: set[int] = set()
    merged: list[dict] = []
    for row in exact_rows + sem_rows:
        if row["qid"] not in seen:
            seen.add(row["qid"])
            merged.append(row)

    match_type = {t: "exact" for t in exact_matched}
    match_type.update({t: "semantic" for t in sem_matched})
    uncovered = [t for t in dict.fromkeys([*topics, *search_topics]) if t not in match_type]
    return merged, match_type, uncovered


@bp.post("/api/questions")
def questions():
    try:
        body = QuestionsRequest.model_validate(request.get_json(force=True))
    except ValidationError as exc:
        return jsonify({"error": str(exc)}), 400

    if not db.is_ready():
        return jsonify(
            ServiceUnavailable(
                error="database is not ready", hint=db.readiness_hint()
            ).model_dump()
        ), 503

    offset = (body.page_number - 1) * body.page_size
    topic_mode = bool(body.topics or body.search_topics)
    try:
        start = time.monotonic()
        with db.connect() as conn:
            with conn.cursor() as cur:
                if topic_mode:
                    merged, match_type, uncovered = _collect(
                        cur, body.topics, body.search_topics, body.page_size
                    )
                    total = len(merged)
                    selected = merged[offset : offset + body.page_size]
                    # ponytail: semantic rows are capped at page_size per search
                    # topic, so semantic matches beyond that cap never appear;
                    # switch to ORDER BY score OFFSET if deeper paging matters.
                else:  # no topics = every question in the DB, SQL-paginated
                    cur.execute(_COUNT_SQL)
                    total = cur.fetchone()[0]
                    cur.execute(_ALL_SQL, (body.page_size, offset))
                    selected = [dict(zip(_Q_COLS, row)) for row in cur.fetchall()]
                    match_type, uncovered = {}, []
    except db.DatabaseNotReady as exc:
        return jsonify(
            ServiceUnavailable(error=str(exc), hint=db.readiness_hint()).model_dump()
        ), 503
    except RuntimeError as exc:
        return jsonify(
            ServiceUnavailable(
                error=str(exc),
                hint="Embedding model unavailable; run `make deps` and restart.",
            ).model_dump()
        ), 503

    has_more = offset + len(selected) < total

    requested = dict.fromkeys([*body.topics, *body.search_topics])
    elapsed_ms = int((time.monotonic() - start) * 1000)
    response = QuestionsResponse(
        topics=[
            TopicMatch(topic=t, match_type=match_type[t])
            for t in requested
            if t in match_type
        ],
        total=total,
        has_more=has_more,
        uncovered_topics=[UncoveredTopic(topic=t) for t in uncovered],
        questions=selected,
        elapsed_ms=elapsed_ms,
    )
    return jsonify(response.model_dump()), 200


def _normalize(label: str) -> str:
    return label.strip().upper()


def _is_correct(chosen: str | None, correct: str | None) -> bool:
    return (
        chosen is not None
        and correct is not None
        and _normalize(correct) == _normalize(chosen)
    )


@bp.post("/api/check-answer")
def check_answer():
    try:
        body = AnswerCheckRequest.model_validate(request.get_json(force=True))
    except ValidationError as exc:
        return jsonify({"error": str(exc)}), 400

    if not db.is_ready():
        return jsonify(
            ServiceUnavailable(
                error="database is not ready", hint=db.readiness_hint()
            ).model_dump()
        ), 503

    qids = [item.qid for item in body.answers]
    try:
        start = time.monotonic()
        with db.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT qid, answer FROM frm_questions WHERE qid = ANY(%s)",
                    (qids,),
                )
                correct_by_qid = {qid: ans for qid, ans in cur.fetchall()}
    except db.DatabaseNotReady as exc:
        return jsonify(
            ServiceUnavailable(error=str(exc), hint=db.readiness_hint()).model_dump()
        ), 503

    results = []
    for item in body.answers:
        correct = correct_by_qid.get(item.qid)
        results.append(
            AnswerCheckResult(
                qid=item.qid,
                chosen_answer=item.answer,
                correct_answer=correct,
                is_correct=_is_correct(item.answer, correct),
            )
        )

    elapsed_ms = int((time.monotonic() - start) * 1000)
    response = AnswerCheckResponse(
        results=results,
        total=len(results),
        correct=sum(r.is_correct for r in results),
        elapsed_ms=elapsed_ms,
    )
    return jsonify(response.model_dump()), 200