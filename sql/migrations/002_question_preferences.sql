-- Single-user UI preference. The frontend never provides a user identity.
CREATE TABLE IF NOT EXISTS frm_question_preferences (
    preference_key TEXT PRIMARY KEY,
    show_attempted BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO frm_question_preferences (preference_key)
VALUES ('student-1')
ON CONFLICT (preference_key) DO NOTHING;
