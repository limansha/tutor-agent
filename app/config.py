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
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "").strip()
    # Free by default so a no-credit account works; set a paid slug (e.g.
    # openai/gpt-oss-20b) once you buy credits.
    OPENROUTER_MODEL: str = os.getenv(
        "OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"
    ).strip()
    OPENROUTER_FALLBACK_MODELS: list[str] = [
        m.strip()
        for m in os.getenv(
            "OPENROUTER_FALLBACK_MODELS",
            "dots-studio/dots-3-note-preview:free,"
            "google/gemma-4-31b-it:free,"
            "liquid/lfm-2.5-2.6b:free",
        ).split(",")
        if m.strip()
    ]
    OPENROUTER_TEMPERATURE: float = float(os.getenv("OPENROUTER_TEMPERATURE", "0.2"))
    OPENROUTER_MAX_TOKENS: int = int(os.getenv("OPENROUTER_MAX_TOKENS", "4000"))

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

    CHILD_WINDOW_SENTENCES: int = int(os.getenv("CHILD_WINDOW_SENTENCES", "4"))
    CHILD_SLIDE_SENTENCES: int = int(os.getenv("CHILD_SLIDE_SENTENCES", "2"))
    CHILD_MAX_TOKENS: int = int(os.getenv("CHILD_MAX_TOKENS", "400"))
    PARENT_MAX_TOKENS: int = int(os.getenv("PARENT_MAX_TOKENS", "800"))
    MIN_CHUNK_TOKENS: int = int(os.getenv("MIN_CHUNK_TOKENS", "200"))

    TOPIC_MATCH_THRESHOLD: float = float(os.getenv("TOPIC_MATCH_THRESHOLD", "0.7"))

    # --- MCQ answer source toggle (app/agent.py) ---
    # 'web' (default) = answer from web-search results; 'books' = RAG over
    # frm_books via the agent tool.
    ANSWER_MODE: str = os.getenv("ANSWER_MODE", "web").strip().lower()
    # Web search backend for ANSWER_MODE=web:
    #   duckduckgo = free, keyless scrape (app/websearch.py) — works on the free tier.
    #   openrouter = OpenRouter's `web` plugin — richer, but needs purchased credits.
    WEB_SEARCH_PROVIDER: str = os.getenv(
        "WEB_SEARCH_PROVIDER", "duckduckgo"
    ).strip().lower()
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