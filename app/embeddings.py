from __future__ import annotations

from fastembed import TextEmbedding

from app.config import config


_client: TextEmbedding | None = None


def _get_client() -> TextEmbedding:
    global _client
    if _client is None:
        _client = TextEmbedding(
            model_name=config.EMBEDDING_MODEL,
            cache_dir=config.EMBEDDING_CACHE_DIR or None,
        )
    return _client


def embed_documents(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return [vec.tolist() for vec in _get_client().embed(texts)]


def embed_query(text: str) -> list[float]:
    return next(iter(_get_client().query_embed(text))).tolist()


def count_tokens(text: str) -> int:
    """Token count using the embedding model's own tokenizer."""
    return _get_client().token_count(text)


BACKEND = "fastembed/onnxruntime"


def status() -> dict:
    """Embedding backend status for /api/health.

    Loads the model if it is not already loaded (the singleton caches it after
    the first call), so a missing model or cache reports ready=False instead of
    taking the whole endpoint down.
    """
    ready = _client is not None
    if not ready:
        try:
            _get_client()
            ready = True
        except Exception:
            ready = False
    return {
        "backend": BACKEND,
        "cache_dir": config.EMBEDDING_CACHE_DIR or None,
        "ready": ready,
    }
