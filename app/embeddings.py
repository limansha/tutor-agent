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


def is_loaded() -> bool:
    """Whether the model singleton is already in memory (never loads it)."""
    return _client is not None
