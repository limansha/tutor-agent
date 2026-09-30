-- RAG tutor-agent schema: pgvector extension + frm_books table + HNSW index.
-- Applied via: make db-setup  (scripts/db_setup.py)

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS frm_books (
    id            BIGSERIAL PRIMARY KEY,
    role          TEXT NOT NULL CHECK (role IN ('parent', 'child')),
    parent_id     BIGINT REFERENCES frm_books(id) ON DELETE CASCADE,
    book_no       INT,
    book_title    TEXT,
    source_path   TEXT,
    year          INT,
    reading_no    INT,
    reading_title TEXT,
    module_no     TEXT,
    module_title  TEXT,
    lo_code       TEXT,
    section       TEXT,
    page_number   INT,
    chunk_index   INT,
    token_count   INT,
    doc_url       TEXT,
    text          TEXT NOT NULL,
    embedding     vector(384)
);

-- HNSW index for fast approximate cosine similarity (<=>) over child chunks only.
-- Parents are NOT embedded (their text is pulled via parent_id for context).
-- DROP INDEX IF EXISTS frm_books_embedding_hnsw;
-- CREATE INDEX frm_books_embedding_hnsw
--     ON frm_books USING hnsw (embedding vector_cosine_ops)
--     WHERE role = 'child';

-- Add the section column and relax embedding NOT NULL on databases created
-- before this update (parents are no longer embedded).
ALTER TABLE frm_books ADD COLUMN IF NOT EXISTS section TEXT;
ALTER TABLE frm_books ALTER COLUMN embedding DROP NOT NULL;

-- Lookups used by parent expansion and metadata filtering.
CREATE INDEX IF NOT EXISTS frm_books_parent_id_idx ON frm_books (parent_id);
CREATE INDEX IF NOT EXISTS frm_books_meta_idx
    ON frm_books (book_no, year, reading_no, module_no, lo_code);
CREATE INDEX IF NOT EXISTS frm_books_section_idx ON frm_books (section);

-- FRM Questions table: one-time ingestion of practice exam questions.
DROP TABLE IF EXISTS frm_questions CASCADE;

CREATE TABLE IF NOT EXISTS frm_questions (
    qid                     BIGSERIAL PRIMARY KEY,
    question_text           TEXT NOT NULL,
    options                 JSONB NOT NULL DEFAULT '{}',
    answer                  TEXT NOT NULL,
    has_answer_in_doc       BOOLEAN NOT NULL DEFAULT false,
    question_topics         TEXT[] NOT NULL,
    prerequisites           JSONB NOT NULL DEFAULT '{}',
    pdf_name                TEXT NOT NULL,
    source_path             TEXT NOT NULL,
    page_number             INT NOT NULL,
    ingested_at             TIMESTAMP DEFAULT NOW(),
    llm_model_used          TEXT
);

-- Per-topic embeddings: one row per (question, topic) so each topic can be
-- filtered independently (pgvector cannot index/filter a vector[] column).
CREATE TABLE IF NOT EXISTS frm_question_topics (
    qid       BIGINT REFERENCES frm_questions(qid) ON DELETE CASCADE,
    topic     TEXT NOT NULL,
    embedding vector(384) NOT NULL,
    PRIMARY KEY (qid, topic)
);

CREATE INDEX IF NOT EXISTS frm_questions_pdf_name_idx ON frm_questions (pdf_name);
CREATE INDEX IF NOT EXISTS frm_questions_source_path_idx ON frm_questions (source_path);
CREATE INDEX IF NOT EXISTS frm_questions_question_topics_idx ON frm_questions USING GIN (question_topics);
CREATE INDEX frm_question_topics_emb_hnsw
    ON frm_question_topics USING hnsw (embedding vector_cosine_ops);