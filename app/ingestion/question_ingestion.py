"""Question ingestion pipeline: extract questions from PDFs → LLM → JSON dump.

Logs per-file progress and timing. DB loading is a separate step
(store_questions_from_json / make ingest-questions-store).
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from langchain_openai import ChatOpenAI
from app import db
from app.config import config
from app.embeddings import embed_documents
from app.llm import get_llm, with_llm_fallback
from app.prompts import render
from pypdf import PdfReader

logger = logging.getLogger(__name__)

_PADDING = re.compile(r"[ \t ]{2,}")


def _clean_text(text: str) -> str:
    """Flatten layout-mode column padding so the LLM sees normal prose.

    layout extraction pads columns with tabs/spaces; normalizing them keeps the
    prompt readable. Do NOT run this through MarkdownHeaderTextSplitter — it
    strips whitespace entirely ("JohnWick,FRM,manages...").
    """
    return _PADDING.sub(" ", text.replace("\t", " ")).strip()


def _extract_questions(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    questions: list[dict] = []
    pdf_name = pdf_path.name
    logger.info(f"  Pages: {len(reader.pages)}")

    for page_num, page in enumerate(reader.pages, start=1):
        text = _clean_text(page.extract_text(extraction_mode="layout") or "")

        if not text:
            logger.debug(f"  Page {page_num}: empty text, skipping")
            continue

        question_count = len(re.findall(r"^Q[.\s]*\d+", text, re.MULTILINE))
        logger.info(f"  Page {page_num}: {len(text)} chars, ~{question_count} question(s)")
        prompt = render("question_extract.j2", page_text=text)
        response = with_llm_fallback(
            lambda model: get_llm(model).invoke([{"role": "user", "content": prompt}])
        )
        response_text = response.content if hasattr(response, "content") else str(response)
        # Strip markdown code block markers
        response_text = re.sub(r"^```(?:json)?\s*", "", response_text.strip())
        response_text = re.sub(r"\s*```$", "", response_text.strip())
        logger.info(f"  Page {page_num}: LLM response ({len(response_text)} chars)")
        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, list):
                for item in parsed:
                    item.update({
                        "pdf_name": pdf_name,
                        "source_path": str(pdf_path),
                        "page_number": page_num,
                        "llm_model_used": config.OPENROUTER_MODEL,
                    })
                    questions.append(item)
                logger.info(f"  Page {page_num}: {len(parsed)} question(s) parsed")
            else:
                logger.warning(f"  Page {page_num}: LLM returned non-list: {type(parsed)}")
        except (json.JSONDecodeError, TypeError) as exc:
            logger.error(f"  Page {page_num}: Failed to parse LLM response ({exc}): {response_text[:200]}")
            continue

    logger.info(f"  Total questions from {pdf_name}: {len(questions)}")
    return questions


def _dump_questions(questions: list[dict], output_dir: str) -> None:
    if not questions:
        return
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_name = questions[0].get("pdf_name", "questions")
    filepath = out_dir / f"questions_{Path(pdf_name).stem}.json"
    with open(filepath, "w") as f:
        json.dump(questions, f, indent=2)


def _topic_rows(qid: int, topics: list[str]) -> list[tuple]:
    vecs = embed_documents(topics) if topics else []
    return [
        (qid, topic, f"[{','.join(f'{x:.6f}' for x in vec)}]")
        for topic, vec in zip(topics, vecs)
    ]


def _store_questions(questions: list[dict]) -> None:
    if not questions:
        return
    logger.info(f"  Storing {len(questions)} question(s) to DB...")
    conn = db.connect()
    try:
        with conn.cursor() as cur:
            topic_rows: list[tuple] = []
            for q in questions:
                opts = q.get("options", {})
                cur.execute(
                    """
                    INSERT INTO frm_questions
                        (question_text, options, answer, has_answer_in_doc,
                         question_topics, prerequisites, pdf_name, source_path,
                         page_number, llm_model_used)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING qid
                    """,
                    (
                        q["question_text"], json.dumps(opts), q["answer"],
                        q.get("has_answer_in_doc", False), q.get("question_topics", []),
                        json.dumps(q.get("prerequisites", {})), q["pdf_name"],
                        q.get("source_path", ""), q["page_number"],
                        q.get("llm_model_used", config.OPENROUTER_MODEL),
                    ),
                )
                topic_rows += _topic_rows(cur.fetchone()[0], q.get("question_topics", []))
            if topic_rows:
                cur.executemany(
                    """
                    INSERT INTO frm_question_topics (qid, topic, embedding)
                    VALUES (%s, %s, %s::vector)
                    """,
                    topic_rows,
                )
        conn.commit()
        logger.info(f"  DB commit OK: {len(questions)} question(s) persisted")
    finally:
        conn.close()


def _pdf_already_stored(pdf_name: str) -> bool:
    conn = db.connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT EXISTS(SELECT 1 FROM frm_questions WHERE pdf_name = %s)", (pdf_name,))
            return cur.fetchone()[0]
    finally:
        conn.close()


def _delete_all_questions() -> None:
    conn = db.connect()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM frm_questions")
        conn.commit()
    finally:
        conn.close()


def ingest_questions(books_dir: str | None = None) -> dict:
    logger.info("=" * 60)
    logger.info("Starting Phase 1 question ingestion")
    logger.info("=" * 60)

    if not db.is_ready():
        raise db.DatabaseNotReady(db.readiness_hint())

    directory = Path(books_dir) if books_dir else Path(config.QUESTIONS_DIR)
    logger.info(f"Source directory: {directory}")
    pdf_files = sorted(p for p in directory.glob("*.pdf") if p.is_file())

    for f in directory.glob("*"):
        if f.is_file() and f.suffix.lower() != ".pdf":
            logger.info(f"Skipping non-PDF file: {f.name} (extension: {f.suffix.lower()})")

    if not pdf_files:
        logger.warning(f"No PDF files found in {directory}")
        return {"status": "error", "detail": f"No PDFs found in {directory}"}

    logger.info(f"Found {len(pdf_files)} PDF file(s) to process")

    overall_start = time.monotonic()
    processed = failed = total_questions = 0

    for pdf_path in pdf_files:
        file_start = time.monotonic()
        logger.info(f"Processing: {pdf_path.name}")
        try:
            questions = _extract_questions(pdf_path)
            elapsed = time.monotonic() - file_start
            processed += 1
            total_questions += len(questions)
            if questions:
                _dump_questions(questions, config.QUESTIONS_JSON_DIR)
            logger.info(
                f"  ✓ {pdf_path.name}: {len(questions)} question(s) extracted "
                f"in {elapsed:.1f}s — dumped to JSON"
            )
        except Exception as exc:
            elapsed = time.monotonic() - file_start
            failed += 1
            logger.error(f"  ✗ {pdf_path.name}: FAILED after {elapsed:.1f}s — {exc}")

    total_elapsed = time.monotonic() - overall_start
    logger.info("-" * 60)
    logger.info(
        f"Ingestion complete: {processed} processed, {failed} failed, "
        f"{total_questions} total questions in {total_elapsed:.1f}s"
    )
    logger.info("=" * 60)

    return {
        "status": "ok", "total_files": len(pdf_files),
        "processed": processed, "failed": failed,
        "total_questions": total_questions,
        "total_time_seconds": round(total_elapsed, 1),
        "message": "Questions extracted to JSON. Run `make ingest-questions-store` to store them in the DB.",
    }


def store_questions_from_json(json_dir: str | None = None, force_reingest: bool = False) -> dict:
    logger.info("=" * 60)
    logger.info("Storing questions from JSON to DB")
    logger.info("=" * 60)

    if not db.is_ready():
        raise db.DatabaseNotReady(db.readiness_hint())

    directory = Path(json_dir) if json_dir else Path(config.QUESTIONS_JSON_DIR)
    json_files = sorted(directory.glob("questions_*.json"))
    if not json_files:
        logger.warning(f"No questions_*.json files found in {directory}")
        return {"status": "error", "detail": f"No JSON files found in {directory}"}

    if force_reingest:
        logger.info("Force reingest: deleting existing questions")
        _delete_all_questions()

    logger.info(f"Found {len(json_files)} JSON file(s) to store")
    processed = failed = skipped = total_questions = 0

    for json_path in json_files:
        file_start = time.monotonic()
        logger.info(f"Storing: {json_path.name}")
        try:
            with open(json_path) as f:
                questions = json.load(f)
            if not questions:
                logger.info(f"  - {json_path.name}: empty, skipped")
                skipped += 1
                continue
            pdf_name = questions[0].get("pdf_name", "")
            if pdf_name and _pdf_already_stored(pdf_name):
                elapsed = time.monotonic() - file_start
                skipped += 1
                logger.info(f"  - {json_path.name}: {pdf_name} already in DB, skipped "
                            f"in {elapsed:.1f}s")
                continue
            _store_questions(questions)
            elapsed = time.monotonic() - file_start
            processed += 1
            total_questions += len(questions)
            logger.info(
                f"  ✓ {json_path.name}: {len(questions)} question(s) stored "
                f"in {elapsed:.1f}s"
            )
        except Exception as exc:
            elapsed = time.monotonic() - file_start
            failed += 1
            logger.error(f"  ✗ {json_path.name}: FAILED after {elapsed:.1f}s — {exc}")

    logger.info("-" * 60)
    logger.info(
        f"Store complete: {processed} processed, {skipped} skipped, {failed} failed, "
        f"{total_questions} total questions"
    )
    logger.info("=" * 60)

    return {
        "status": "ok", "total_files": len(json_files),
        "processed": processed, "skipped": skipped, "failed": failed,
        "total_questions": total_questions,
        "message": "Questions stored in DB."
    }
