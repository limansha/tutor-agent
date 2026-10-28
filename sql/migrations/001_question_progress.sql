-- Single-user question progress. A row means the question was attempted.
CREATE TABLE IF NOT EXISTS frm_question_progress (
    qid                 BIGINT PRIMARY KEY REFERENCES frm_questions(qid) ON DELETE CASCADE,
    attempt_count       INT NOT NULL DEFAULT 1,
    last_answer         TEXT NOT NULL,
    last_is_correct     BOOLEAN NOT NULL,
    first_attempted_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_attempted_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
