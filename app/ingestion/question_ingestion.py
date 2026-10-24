import json
import logging
import re
import time
from functools import lru_cache
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pypdf import PdfReader

from app import db
from app.config import config
from app.embeddings import embed_documents
from app.llm import LLMSpec, get_llm, ingest_provider_specs
from app.prompts import render

logger = logging.getLogger(__name__)

_PADDING = re.compile(r"[ \t]{2,}")

_SYSTEM_PROMPT = render(
    "question_batch_system.j2", explanation_words=config.INGEST_EXPLANATION_WORDS
)


@lru_cache(maxsize=8)
def _get_ingest_llm(spec: LLMSpec):
    """Cached client per provider so batches reuse connections/TLS."""
    return get_llm(spec, purpose="ingest")


def _clean_text(text: str) -> str:
    return _PADDING.sub(" ", text.replace("\t", " ")).strip()


def pdf_to_parsed_text(pdf_path: Path) -> Path:
    reader = PdfReader(str(pdf_path))
    pdf_name = pdf_path.name
    parsed_dir = Path(config.QUESTIONS_PARSED_DIR)
    parsed_dir.mkdir(parents=True, exist_ok=True)
    out_path = parsed_dir / f"{pdf_path.stem}.txt"
    pages_out = []
    logger.info(f"[pdf_to_parsed_text] {pdf_name}: {len(reader.pages)} pages")
    for page_num, page in enumerate(reader.pages, start=1):
        text = _clean_text(page.extract_text(extraction_mode="layout") or "")
        if not text:
            continue
        pages_out.append((page_num, text))
    with open(out_path, "w", encoding="utf-8") as f:
        for page_num, text in pages_out:
            f.write(f"=== Page {page_num} ===\n{text}\n\n")
    logger.info(f"[pdf_to_parsed_text] wrote {out_path} ({len(pages_out)} pages)")
    return out_path


_QUESTION_START = re.compile(
    r"^\s*(?:Q\.?\s*\d+|Question\s+\d+|Q\d+[\.\):]|\d{1,3}\.\s+(?![A-Za-z0-9]+\.\s*$))",
    re.I,
)
_OPTION_LINE = re.compile(
    r"^\s*\(?([A-Za-z]|\d{1,3})\)?\s*[\.\)\:\-]\s*(.+?)\s*$"
)
_PAGE_MARKER = re.compile(r"^=== Page (\d+) ===$")
_FOOTER = re.compile(r"AnalystPrep|©|protected by International copyright", re.I)
_STANDALONE_NUM = re.compile(r"^\s*\d{1,3}\s*$")
_SECTION_BREAK = re.compile(
    r"^\s*(ANSWER|ANSWERS|EXPLANATION|EXPLANATIONS|SOLUTION|SOLUTIONS|KEY|ANSWER KEY)\b",
    re.I,
)
_ANNOTATION_SEP = "\x1e"  # ASCII record separator: split("\x1e") -> blocks, O(total chars)
_PAGE_SEP = "\x1f"  # record/page separator: page\x1ftext
_ANSWER_LINE = re.compile(r"^\s*ANSWER\s*:\s*([A-Za-z0-9]+)\b", re.I | re.M)
_EXPLANATION_BLOCK = re.compile(
    r"^\s*EXPLANATION\s*:\s*(.*?)(?=^\s*TOPICS\s*:|\Z)", re.I | re.M | re.S
)
_TOPICS_LINE = re.compile(r"^\s*TOPICS\s*:\s*(.+?)\s*$", re.I | re.M)


def _parse_plain_response(text: str) -> tuple[str, str, list[str]]:
    answer = _ANSWER_LINE.search(text)
    explanation = _EXPLANATION_BLOCK.search(text)
    topics = _TOPICS_LINE.search(text)
    if not answer or not explanation or not topics:
        raise ValueError("LLM response must contain ANSWER, EXPLANATION, and TOPICS lines")

    explanation_text = " ".join(explanation.group(1).split())
    words = explanation_text.split()
    explanation_text = " ".join(words[: config.INGEST_EXPLANATION_WORDS])
    topic_values = [topic.strip() for topic in re.split(r"[,;]", topics.group(1)) if topic.strip()]
    if not topic_values:
        raise ValueError("LLM response must contain at least one topic")
    return answer.group(1).upper(), explanation_text, topic_values[:3]


def _clean_block_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    option_started = False
    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        if _PAGE_MARKER.match(s) or _FOOTER.search(s) or _STANDALONE_NUM.match(s):
            continue
        m = _OPTION_LINE.match(s)
        if m and (m.group(1).isalpha() or option_started):
            option_started = True
            out.append(f"{m.group(1).upper()}. {m.group(2).strip()}")
        else:
            out.append(_PADDING.sub(" ", s))
    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()
    return out


def annotate_parsed_text(parsed_path: Path) -> Path:
    annotated_dir = Path(config.QUESTIONS_ANNOTATED_DIR)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = annotated_dir / f"{parsed_path.stem}.txt"
    with open(parsed_path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    # track page number across the parsed file (markers may appear before questions)
    page_at: list[int] = [0] * len(lines)
    current_page = 0
    for i, line in enumerate(lines):
        m = _PAGE_MARKER.match(line.strip())
        if m:
            current_page = int(m.group(1))
        page_at[i] = current_page

    starts = [i for i, line in enumerate(lines) if _QUESTION_START.match(line)]
    first_start = starts[0] if starts else len(lines)
    answer_key = next(
        (
            i
            for i in range(first_start, len(lines))
            if re.search(r"Answer Key\s*$", lines[i], re.I)
        ),
        len(lines),
    )
    starts = [i for i in starts if i < answer_key]
    logger.info(f"[annotate_parsed_text] found {len(starts)} question starts")

    blocks: list[tuple[int, str]] = []
    for idx, s_i in enumerate(starts):
        e_i = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        if idx + 1 == len(starts):
            for j in range(s_i + 1, len(lines)):
                if _SECTION_BREAK.match(lines[j]):
                    e_i = j
                    break
        # use first page marker seen from start of block onward (question start page)
        page_num = 0
        for j in range(s_i, e_i):
            if page_at[j]:
                page_num = page_at[j]
                break
        cleaned = _clean_block_lines(lines[s_i:e_i])
        option_count = sum(
            bool((m := _OPTION_LINE.match(line)) and m.group(1).isalpha())
            for line in cleaned
        )
        if cleaned and option_count >= 2:
            blocks.append((page_num, "\n".join(cleaned)))

    with open(annotated_path, "w", encoding="utf-8") as f:
        f.write(
            ("\n" + _ANNOTATION_SEP + "\n").join(
                f"{p}{_PAGE_SEP}{t}" for p, t in blocks
            )
            + "\n"
        )
    logger.info(f"[annotate_parsed_text] wrote {len(blocks)} blocks to {annotated_path}")
    return annotated_path


def _blocks_from_annotated(annotated_path: Path) -> list[dict]:
    with open(annotated_path, encoding="utf-8") as f:
        content = f.read()
    blocks = []
    for i, raw in enumerate(content.split(_ANNOTATION_SEP), start=1):
        text = raw.strip()
        if not text:
            continue
        page_num = 0
        if _PAGE_SEP in text:
            p, t = text.split(_PAGE_SEP, 1)
            try:
                page_num = int(p.strip())
            except ValueError:
                page_num = 0
            text = t.strip()
        blocks.append({"id": i, "text": text, "page_number": page_num})
    return blocks


def _split_question_block(text: str) -> tuple[str, dict[str, str]]:
    """Split a clean block into (question_text, {label: option_text})."""
    question_lines: list[str] = []
    options: dict[str, str] = {}
    current: str | None = None
    option_started = False
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        m = _OPTION_LINE.match(s)
        if m and (m.group(1).isalpha() or option_started):
            option_started = True
            current = m.group(1).upper()
            options[current] = m.group(2).strip()
        elif current is not None:
            options[current] = (options[current] + " " + s).strip()
        else:
            question_lines.append(s)
    question_text = _PADDING.sub(" ", " ".join(question_lines)).strip()
    return question_text, options


def _load_json(path: Path) -> list:
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _write_json_atomic(path: Path, data: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def _process_batch(
    pdf_path: Path,
    stem: str,
    batch: list[dict],
    batch_num: int,
    start_idx: int,
    end_idx: int,
    label: str,
    spec: LLMSpec,
) -> tuple[list[dict] | None, dict, str]:
    """Run one batch through the LLM. Returns (items, bench_record, raw_resp)."""
    logger.info(
        "[document] %s batch %d START (questions %d-%d)",
        pdf_path.name,
        batch_num,
        start_idx,
        end_idx,
    )
    prompt = render(
        "question_batch_extract.j2",
        pdf_name=pdf_path.name,
        batch_size=len(batch),
        start_index=start_idx,
        questions=[b["text"] for b in batch],
    )
    resp_text = ""
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stem": stem,
        "batch_num": batch_num,
        "batch_size": len(batch),
        "question_ids": [b["id"] for b in batch],
    }
    try:
        response = _get_ingest_llm(spec).invoke(
            [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        record["model"] = f"{spec.model}@{spec.base_url}"
        resp_text = response.content if hasattr(response, "content") else str(response)
        answer, explanation, topics = _parse_plain_response(resp_text)
        items: list[dict] = []
        block = batch[0]
        question_text, options = _split_question_block(block["text"])
        if answer not in options:
            raise ValueError(f"LLM answer {answer!r} is not an option in question {block['id']}")
        items.append({
            "question_text": question_text,
            "options": options,
            "answer": answer,
            "explanation": explanation,
            "question_topics": topics,
            "pdf_name": pdf_path.name,
            "source_path": str(pdf_path),
            "page_number": block.get("page_number", 0),
            "llm_model_used": record["model"],
        })
        record["status"] = "ok"
        return items, record, resp_text
    except Exception:
        record["model"] = f"{spec.model}@{spec.base_url}"
        record["status"] = "failed"
        question_text, options = _split_question_block(batch[0]["text"])
        return [{
            "question_text": question_text,
            "options": options,
            "answer": "",
            "explanation": "",
            "question_topics": [],
            "pdf_name": pdf_path.name,
            "source_path": str(pdf_path),
            "page_number": batch[0].get("page_number", 0),
            "llm_model_used": record["model"],
        }], record, resp_text


def process_one_pdf(
    pdf_path: Path,
    batch_size: int = 1,
    force: bool = False,
    label: str = "",
    max_batches: int | None = None,
) -> dict:
    pdf_path = Path(pdf_path)
    stem = pdf_path.stem
    document_start = time.monotonic()
    logger.info(
        "[document] START %s (batch_size=%d, force=%s)",
        pdf_path.name,
        batch_size,
        force,
    )
    parsed_dir = Path(config.QUESTIONS_PARSED_DIR)
    annotated_dir = Path(config.QUESTIONS_ANNOTATED_DIR)
    json_dir = Path(config.QUESTIONS_JSON_FILES_DIR)
    json_dir.mkdir(parents=True, exist_ok=True)
    parsed_path = parsed_dir / f"{stem}.txt"
    annotated_path = annotated_dir / f"{stem}.txt"
    json_path = json_dir / f"{stem}.json"

    if not parsed_path.exists() or force:
        parsed_path = pdf_to_parsed_text(pdf_path)
    if not annotated_path.exists() or force:
        annotated_path = annotate_parsed_text(parsed_path)

    blocks = _blocks_from_annotated(annotated_path)
    total = len(blocks)
    if batch_size != 1:
        raise ValueError("question ingestion requires batch_size=1")
    specs = ingest_provider_specs()
    if not specs:
        raise RuntimeError("no ingestion LLM configured")
    logger.info(
        "[document] %s prepared: %d question blocks, %d batches, output=%s",
        pdf_path.name,
        total,
        (total + batch_size - 1) // batch_size if batch_size else 0,
        json_path,
    )

    batches: list[tuple[int, int, int, list[dict]]] = []
    next_index = 0
    batch_num = 1
    while next_index < total:
        if max_batches is not None and batch_num > max_batches:
            logger.info(f"Reached max_batches={max_batches}, stopping")
            break
        batch = blocks[next_index : next_index + batch_size]
        if not batch:
            break
        batches.append((batch_num, next_index + 1, next_index + len(batch), batch))
        next_index += len(batch)
        batch_num += 1

    results: dict[int, list[dict] | None] = {}

    def _commit(bn: int, items: list[dict] | None) -> None:
        results[bn] = items
        merged: list[dict] = []
        for key in sorted(results):
            if results[key]:
                merged.extend(results[key])
        _write_json_atomic(json_path, merged)
        logger.info(
            "[document] %s batch %d/%d committed (%d questions)",
            pdf_path.name,
            bn,
            len(batches),
            len(items or []),
        )

    provider_index = 0
    for bn, s, e, batch in batches:
        spec = specs[provider_index]
        items, record, _ = _process_batch(
            pdf_path, stem, batch, bn, s, e, label, spec
        )
        _commit(bn, items)
        if items is None or record.get("status") == "failed":
            provider_index = (provider_index + 1) % len(specs)

    processed = next_index if max_batches is None else min(next_index, batch_size * max_batches)
    logger.info(
        "[document] COMPLETE %s: %d/%d batches, %d questions, %.1fs, output=%s",
        pdf_path.name,
        len(results),
        len(batches),
        sum(len(items) for items in results.values() if items),
        time.monotonic() - document_start,
        json_path,
    )
    return {
        "status": "ok",
        "stem": stem,
        "done": processed,
        "total": total,
        "questions": sum(len(items) for items in results.values() if items),
        "json_path": str(json_path),
        "batches": len(batches),
    }


def _dump_path(pdf_name: str, output_dir: str) -> Path:
    return Path(output_dir) / f"{Path(pdf_name).stem}.json"


def _topic_rows(qid: int, topics: list[str]) -> list[tuple]:
    vecs = embed_documents(topics) if topics else []
    return [
        (qid, topic, f"[{','.join(f'{x:.6f}' for x in vec)}]")
        for topic, vec in zip(topics, vecs)
    ]


def _store_questions(questions: list[dict]) -> None:
    if not questions:
        return
    total = len(questions)
    logger.info(f"  Storing {total} question(s) to DB...")
    conn = db.connect()
    try:
        with conn.cursor() as cur:
            topic_rows: list[tuple] = []
            for i, q in enumerate(questions, start=1):
                opts = q.get("options", {})
                cur.execute(
                    """
                    INSERT INTO frm_questions
                        (question_text, options, answer, explanation,
                         question_topics, pdf_name, source_path, page_number,
                         llm_model_used)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING qid
                    """,
                    (
                        q["question_text"], json.dumps(opts), q["answer"],
                        q.get("explanation", ""), q.get("question_topics", []),
                        q["pdf_name"], q.get("source_path", ""),
                        q.get("page_number", 0),
                        q.get("llm_model_used", config.OPENROUTER_MODEL),
                    ),
                )
                topic_rows += _topic_rows(cur.fetchone()[0], q.get("question_topics", []))
                if i % 25 == 0 or i == total:
                    logger.info(f"    {i}/{total} question(s) inserted")
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


def ingest_questions(
    books_dir: str | None = None,
    batch_size: int = 1,
) -> dict:
    logger.info("=" * 60)
    logger.info("Starting Phase 1 question ingestion")
    logger.info("=" * 60)

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
    processed = failed = skipped = total_questions = 0

    for document_num, pdf_path in enumerate(pdf_files, start=1):
        if _dump_path(pdf_path.name, config.QUESTIONS_JSON_FILES_DIR).exists():
            skipped += 1
            logger.info(
                "[document %d/%d] SKIP %s: JSON already exists",
                document_num,
                len(pdf_files),
                pdf_path.name,
            )
            continue
        file_start = time.monotonic()
        logger.info(
            "[document %d/%d] START %s",
            document_num,
            len(pdf_files),
            pdf_path.name,
        )
        try:
            result = process_one_pdf(
                pdf_path,
                batch_size=batch_size,
                force=False,
            )
            elapsed = time.monotonic() - file_start
            processed += 1
            total_questions += result["questions"]
            logger.info(
                "[document %d/%d] COMPLETE %s: %d questions in %.1fs, output=%s",
                document_num,
                len(pdf_files),
                pdf_path.name,
                result["questions"],
                elapsed,
                result["json_path"],
            )
        except Exception as exc:
            elapsed = time.monotonic() - file_start
            failed += 1
            logger.error(
                "[document %d/%d] FAILED %s after %.1fs: %s",
                document_num,
                len(pdf_files),
                pdf_path.name,
                elapsed,
                exc,
            )

    total_elapsed = time.monotonic() - overall_start
    logger.info("-" * 60)
    logger.info(
        f"Ingestion complete: {processed} processed, {skipped} skipped, "
        f"{failed} failed, {total_questions} total questions in {total_elapsed:.1f}s"
    )
    logger.info("=" * 60)

    return {
        "status": "ok", "total_files": len(pdf_files),
        "processed": processed, "skipped": skipped, "failed": failed,
        "total_questions": total_questions,
        "total_time_seconds": round(total_elapsed, 1),
        "message": "Questions extracted to JSON. Run `make ingest-questions-store` to store them in the DB.",
    }


def store_questions_from_json(json_dir: str | None = None) -> dict:
    logger.info("=" * 60)
    logger.info("Storing questions from JSON to DB")
    logger.info("=" * 60)

    if not db.is_ready():
        raise db.DatabaseNotReady(db.readiness_hint())

    directory = Path(json_dir) if json_dir else Path(config.QUESTIONS_JSON_FILES_DIR)
    json_files = sorted(directory.glob("*.json"))
    if not json_files:
        directory = Path(config.QUESTORS_JSON_DIR) if hasattr(config, 'QUESTORS_JSON_DIR') else Path(config.QUESTIONS_JSON_DIR)
        json_files = sorted(directory.glob("*.json"))

    logger.info(f"Found {len(json_files)} JSON file(s) to store")
    processed = failed = skipped = total_questions = 0

    for json_path in json_files:
        file_start = time.monotonic()
        logger.info(f"Storing: {json_path.name}")
        try:
            with open(json_path) as f:
                questions = json.load(f)
            if isinstance(questions, dict):
                questions = [questions]
            if not isinstance(questions, list):
                raise ValueError("question JSON must be an object or an array of objects")
            if not questions:
                logger.info(f"  - {json_path.name}: empty, skipped")
                skipped += 1
                continue
            valid = [
                q for q in questions
                if isinstance(q, dict) and (q.get("question_text") or "").strip()
            ]
            dropped = len(questions) - len(valid)
            if dropped:
                logger.warning(
                    f"  {json_path.name}: dropping {dropped} question(s) with null/empty question_text"
                )
            pdf_name = valid[0].get("pdf_name", "") if valid else ""
            if pdf_name and _pdf_already_stored(pdf_name):
                elapsed = time.monotonic() - file_start
                skipped += 1
                logger.info(f"  - {json_path.name}: {pdf_name} already in DB, skipped "
                            f"in {elapsed:.1f}s")
                continue
            _store_questions(valid)
            elapsed = time.monotonic() - file_start
            processed += 1
            total_questions += len(valid)
            logger.info(
                f"  ✓ {json_path.name}: {len(valid)} question(s) stored "
                f"in {elapsed:.1f}s — running total {total_questions}"
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


def _parse_questions_from_text_old(text: str) -> list[str]:
    if not text:
        return []
    blocks = []
    current = []
    for line in text.splitlines():
        m = re.match(r"^\s*(?:Q\.?\s*\d+|Question\s+\d+|Q\d+[\.\):])", line, re.I)
        if m and current:
            blocks.append("\n".join(current))
            current = [line]
        elif m and not current:
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return [_clean_text(b) for b in blocks if _clean_text(b)]


def process_parsed_text_to_json(parsed_path: Path, batch_size: int = 1) -> dict:
    document_start = time.monotonic()
    logger.info("[document] START parsed input %s", parsed_path.name)
    logger.info("=" * 60)
    logger.info("Processing parsed text -> batched LLM -> JSON")
    logger.info("=" * 60)
    with open(parsed_path, encoding="utf-8") as f:
        text = f.read()
    questions = _parse_questions_from_text_old(text)
    pdf_name = parsed_path.stem
    total = len(questions)
    logger.info(f"Found {total} question(s) in {parsed_path.name}")
    if not questions:
        logger.warning("[document] COMPLETE %s: no questions found", parsed_path.name)
        return {"status": "error", "detail": "No questions found"}

    out_dir = Path(config.QUESTIONS_JSON_FILES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pdf_name}.json"

    if batch_size != 1:
        raise ValueError("question ingestion requires batch_size=1")
    specs = ingest_provider_specs()
    if not specs:
        raise RuntimeError("no ingestion LLM configured")

    all_results = []
    provider_index = 0
    for i in range(0, total, batch_size):
        batch = questions[i : i + batch_size]
        logger.info(f"  Batch {i//batch_size + 1}: questions {i+1}-{min(i+batch_size,total)} ({len(batch)})")
        prompt = render(
            "question_batch_extract.j2",
            pdf_name=pdf_name,
            batch_size=len(batch),
            start_index=i + 1,
            questions=batch,
        )
        try:
            spec = specs[provider_index]
            response = _get_ingest_llm(spec).invoke(
                [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=prompt)]
            )
            resp_text = response.content if hasattr(response, "content") else str(response)
            answer, explanation, topics = _parse_plain_response(resp_text)
            all_results.append({
                "question_text": batch[0],
                "options": {},
                "answer": answer,
                "explanation": explanation,
                "question_topics": topics,
                "pdf_name": pdf_name,
                "source_path": "",
                "page_number": 0,
                "llm_model_used": f"{spec.model}@{spec.base_url}",
            })
            logger.info("    -> parsed 1 question")
        except Exception:
            all_results.append({
                "question_text": batch[0],
                "options": {},
                "answer": "",
                "explanation": "",
                "question_topics": [],
                "pdf_name": pdf_name,
                "source_path": "",
                "page_number": 0,
                "llm_model_used": f"{spec.model}@{spec.base_url}",
            })
            provider_index = (provider_index + 1) % len(specs)
            continue

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info(
        "[document] COMPLETE %s: %d questions in %.1fs, output=%s",
        parsed_path.name,
        len(all_results),
        time.monotonic() - document_start,
        out_path,
    )
    return {"status": "ok", "total_questions": len(all_results), "file": str(out_path)}
