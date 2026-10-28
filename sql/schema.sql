-- RAG tutor-agent schema: pgvector extension + frm_books table.
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

-- Parents are NOT embedded (embedding stays NULL; their text is pulled via
-- parent_id for context), so only child rows carry a vector. Similarity is an
-- exact sequential scan over those child vectors — no vector index at this size.

-- Add the section column and relax embedding NOT NULL on databases created
-- before this update (parents are no longer embedded).
ALTER TABLE frm_books ADD COLUMN IF NOT EXISTS section TEXT;
ALTER TABLE frm_books ALTER COLUMN embedding DROP NOT NULL;

-- Lookups used by parent expansion and metadata filtering.
CREATE INDEX IF NOT EXISTS frm_books_parent_id_idx ON frm_books (parent_id);
CREATE INDEX IF NOT EXISTS frm_books_meta_idx
    ON frm_books (book_no, year, reading_no, module_no, lo_code);
CREATE INDEX IF NOT EXISTS frm_books_section_idx ON frm_books (section);

-- Similarity is an exact sequential scan now; clear any HNSW index an earlier
-- schema created so the planner stops using it.
DROP INDEX IF EXISTS frm_books_embedding_hnsw;
DROP INDEX IF EXISTS frm_question_topics_emb_hnsw;

-- FRM Questions table. Columns mirror the question JSON files.
CREATE TABLE IF NOT EXISTS frm_questions (
    qid                     BIGSERIAL PRIMARY KEY,
    question_text           TEXT NOT NULL,
    options                 JSONB NOT NULL DEFAULT '{}',
    answer                  TEXT NOT NULL,
    explanation             TEXT NOT NULL DEFAULT '',
    question_topics         TEXT[] NOT NULL,
    pdf_name                TEXT NOT NULL,
    source_path             TEXT NOT NULL,
    page_number             INT NOT NULL,
    llm_model_used          TEXT
);

-- Migrate databases created with the previous question shape without losing rows.
ALTER TABLE frm_questions ADD COLUMN IF NOT EXISTS explanation TEXT NOT NULL DEFAULT '';
ALTER TABLE frm_questions DROP COLUMN IF EXISTS has_answer_in_doc;
ALTER TABLE frm_questions DROP COLUMN IF EXISTS prerequisites;
ALTER TABLE frm_questions DROP COLUMN IF EXISTS ingested_at;

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
CREATE UNIQUE INDEX IF NOT EXISTS frm_questions_question_text_uidx
    ON frm_questions (question_text);

-- Single-user question progress. A row means the question was attempted.
CREATE TABLE IF NOT EXISTS frm_question_progress (
    qid                 BIGINT PRIMARY KEY REFERENCES frm_questions(qid) ON DELETE CASCADE,
    attempt_count       INT NOT NULL DEFAULT 1,
    last_answer         TEXT NOT NULL,
    last_is_correct     BOOLEAN NOT NULL,
    first_attempted_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_attempted_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Single-user UI preference. The frontend never provides a user identity.
CREATE TABLE IF NOT EXISTS frm_question_preferences (
    preference_key TEXT PRIMARY KEY,
    show_attempted BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO frm_question_preferences (preference_key)
VALUES ('student-1')
ON CONFLICT (preference_key) DO NOTHING;
