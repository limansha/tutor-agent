"""POST /api/questions — fetch practice questions by requested topics.

Exact topic match first (GIN on question_topics), then semantic fallback
(HNSW cosine over frm_question_topics embeddings) for topics with no literal
hit. Topics with no questions at all are reported in uncovered_topics and the
response covers the rest. Guarantees at least one question per matched topic.
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
    SELECT {", ".join(_Q_COLS)}, t AS matched_topic
    FROM frm_questions q
    CROSS JOIN LATERAL unnest(q.question_topics) AS t
    WHERE q.question_topics && %s::text[]
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


def _exact_pools(cur, topics: list[str]) -> dict[str, list[dict]]:
    """topic -> rows whose question_topics literally contain it (GIN)."""
    cur.execute(_EXACT_SQL, (topics,))
    pools: dict[str, list[dict]] = {}
    for row in cur.fetchall():
        d = dict(zip(_Q_COLS + ("matched_topic",), row))
        pools.setdefault(d.pop("matched_topic"), []).append(d)
    return pools


def _semantic_pools(cur, topics: list[str], limit: int) -> dict[str, list[dict]]:
    """topic -> rows whose stored topic embedding is near the requested topic."""
    if not topics:
        return {}
    pools: dict[str, list[dict]] = {}
    for topic, vec in zip(topics, embed_documents(topics)):
        cur.execute(_SEMANTIC_SQL, (_emb_sql(vec), _emb_sql(vec), limit))
        rows = []
        for row in cur.fetchall():
            d = dict(zip(_Q_COLS + ("score",), row))
            if d.pop("score") >= config.TOPIC_MATCH_THRESHOLD:
                rows.append(d)
        if rows:
            pools[topic] = rows
    return pools


def _match_questions(cur, topics: list[str], limit: int = 50) -> tuple[dict[str, list[dict]], dict[str, str], list[str]]:
    exact = _exact_pools(cur, topics)
    missing = [t for t in topics if not exact.get(t)]
    semantic = _semantic_pools(cur, missing, limit)

    pools: dict[str, list[dict]] = {}
    match_type: dict[str, str] = {}
    uncovered: list[str] = []
    for t in topics:
        rows = exact.get(t) or semantic.get(t)
        if rows:
            pools[t] = rows
            match_type[t] = "exact" if t in exact else "semantic"
        else:
            uncovered.append(t)
    return pools, match_type, uncovered


def _allocate(pools: dict[str, list[dict]], page_size: int) -> list[dict]:
    """Round-robin selection guaranteeing ≥1 question per matched topic (dedup by qid)."""
    covered = [t for t in pools if pools[t]]
    if page_size < len(covered):
        raise ValueError(
            f"page_size ({page_size}) must be >= number of topics with matches ({len(covered)})"
        )

    selected: list[dict] = []
    seen: set[int] = set()

    def take(topic: str) -> dict | None:
        for q in pools[topic]:
            if q["qid"] not in seen:
                seen.add(q["qid"])
                return q
        return None

    for t in covered:  # guarantee one per matched topic
        if (q := take(t)) is not None:
            selected.append(q)

    while len(selected) < page_size:  # fill remaining slots round-robin
        added = False
        for t in covered:
            if len(selected) >= page_size:
                break
            if (q := take(t)) is not None:
                selected.append(q)
                added = True
        if not added:
            break
    return selected


@bp.post("/api/questions")
def questions():
    try:
        body = QuestionsRequest.model_validate(request.get_json(force=True))
    except ValidationError as exc:
        return jsonify({"error": str(exc)}), 400

    offset = (body.page - 1) * body.page_size
    try:
        start = time.monotonic()
        with db.connect() as conn:
            with conn.cursor() as cur:
                if body.topics:
                    pools, match_type, uncovered = _match_questions(
                        cur, body.topics, body.page_size
                    )
                    all_mode = False
                else:  # no topics = every question in the DB, SQL-paginated
                    cur.execute(_COUNT_SQL)
                    total = cur.fetchone()[0]
                    cur.execute(_ALL_SQL, (body.page_size, offset))
                    selected = [dict(zip(_Q_COLS, row)) for row in cur.fetchall()]
                    match_type, uncovered = {}, []
                    all_mode = True
    except db.DatabaseNotReady as exc:
        return jsonify(
            ServiceUnavailable(error=str(exc), hint=db.readiness_hint()).model_dump()
        ), 503
    except (RuntimeError, ValueError) as exc:
        return jsonify(
            ServiceUnavailable(
                error=str(exc),
                hint="Embedding model unavailable; run `make deps` and restart.",
            ).model_dump()
        ), 503

    if not all_mode:
        try:
            selected = _allocate(pools, body.page_size)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        total = len(selected)
        has_more = False
    else:
        has_more = offset + len(selected) < total

    elapsed_ms = int((time.monotonic() - start) * 1000)
    response = QuestionsResponse(
        topics=[
            TopicMatch(topic=t, match_type=match_type[t])
            for t in body.topics
            if t in match_type
        ],
        page=body.page,
        page_size=body.page_size,
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