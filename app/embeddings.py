from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import config


_client: HuggingFaceEmbeddings | None = None


def _get_client() -> HuggingFaceEmbeddings:
    global _client
    if _client is None:
        _client = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return _client


def embed_documents(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return _get_client().embed_documents(texts)


def embed_query(text: str) -> list[float]:
    return _get_client().embed_query(text)
