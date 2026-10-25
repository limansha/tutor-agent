import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Config:
    """Runtime configuration loaded from .env (documented in .env.example)."""

    DATABASE_URL: str = os.getenv("DATABASE_URL", "").strip()
    MCQ_TEMPERATURE: float = float(os.getenv("MCQ_TEMPERATURE", "0.2"))
    MCQ_MAX_TOKENS: int = int(os.getenv("MCQ_MAX_TOKENS", "4000"))

    # Cross-provider fallbacks (all OpenAI-compatible; a provider with a blank
    # key is skipped). Independent quotas keep the answer path available.
    # Best model per provider (quota is account-based, not model-wise, so use the
    # strongest). Slugs drift — providers retire models (e.g. Groq killed
    # llama-3.3-70b-versatile in Aug 2026); swap in .env when one 404s.
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()

    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    ).strip()
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "384"))
    EMBEDDING_CACHE_DIR: str = os.getenv("EMBEDDING_CACHE_DIR", "").strip()

    BOOKS_DIR: str = os.getenv("BOOKS_DIR", "").strip()
    QUESTIONS_DIR: str | None = os.getenv("QUESTIONS_DIR")
    if QUESTIONS_DIR is None:
        raise RuntimeError("QUESTIONS_DIR is not set. Add it to .env.")
    QUESTIONS_DIR: str = QUESTIONS_DIR.strip()

    QUESTIONS_JSON_DIR: str = os.getenv(
        "QUESTIONS_JSON_DIR", str(BASE_DIR / "questions_dump")
    ).strip()
    QUESTIONS_PARSED_DIR: str = os.getenv(
        "QUESTIONS_PARSED_DIR", str(BASE_DIR / "questions_dump" / "parsed_text_files")
    ).strip()
    QUESTIONS_ANNOTATED_DIR: str = os.getenv(
        "QUESTIONS_ANNOTATED_DIR", str(BASE_DIR / "questions_dump" / "annotated_text_files")
    ).strip()
    QUESTIONS_JSON_FILES_DIR: str = os.getenv(
        "QUESTIONS_JSON_FILES_DIR", str(BASE_DIR / "questions_dump" / "json_files")
    ).strip()
    # Question-ingestion LLM settings (kept separate from the MCQ answer path).
    INGEST_MODEL: str = os.getenv("INGEST_MODEL", "").strip()
    INGEST_TEMPERATURE: float = float(os.getenv("INGEST_TEMPERATURE", "0"))
    INGEST_MAX_TOKENS: int = int(os.getenv("INGEST_MAX_TOKENS", "1500"))
    INGEST_EXPLANATION_WORDS: int = min(
        int(os.getenv("INGEST_EXPLANATION_WORDS", "150")), 150
    )
    INGEST_REQUEST_TIMEOUT: int = int(os.getenv("INGEST_REQUEST_TIMEOUT", "60"))

    CHILD_WINDOW_SENTENCES: int = int(os.getenv("CHILD_WINDOW_SENTENCES", "4"))
    CHILD_SLIDE_SENTENCES: int = int(os.getenv("CHILD_SLIDE_SENTENCES", "2"))
    CHILD_MAX_TOKENS: int = int(os.getenv("CHILD_MAX_TOKENS", "400"))
    PARENT_MAX_TOKENS: int = int(os.getenv("PARENT_MAX_TOKENS", "800"))
    MIN_CHUNK_TOKENS: int = int(os.getenv("MIN_CHUNK_TOKENS", "200"))

    TOPIC_MATCH_THRESHOLD: float = float(os.getenv("TOPIC_MATCH_THRESHOLD", "0.7"))

    # --- Web search (app/agent.py `_search`) ---
    WEB_SEARCH_MAX_RESULTS: int = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "3"))
    # Domains to keep out of web-mode results: Q&A sites, forums and answer mills
    # that show the question but not a reliable, official explanation.
    WEB_SEARCH_EXCLUDE_DOMAINS: list[str] = [
        d.strip()
        for d in os.getenv(
            "WEB_SEARCH_EXCLUDE_DOMAINS",
            "quora.com,reddit.com,facebook.com,twitter.com,x.com,instagram.com,"
            "tiktok.com,pinterest.com,stackexchange.com,stackoverflow.com,"
            "chegg.com,coursehero.com,studocu.com,scribd.com,brainly.com,"
            "quizlet.com,numerade.com,bartleby.com,answers.yahoo.com",
        ).split(",")
        if d.strip()
    ]

    YEAR: int = int(os.getenv("DOC_YEAR", "2026"))
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "5000"))
    DEBUG: bool = _as_bool(os.getenv("DEBUG"), False)

    SCHEMA_SQL: Path = BASE_DIR / "sql" / "schema.sql"


config = Config()
