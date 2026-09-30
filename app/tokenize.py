from __future__ import annotations

from app.config import config

_tokenizer = None


def _load_tokenizer():
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer
    try:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)
    except Exception:
        _tokenizer = None
    return _tokenizer


def count_tokens(text: str) -> int:
    """Token count using the embedding model's tokenizer (whitespace fallback)."""
    tok = _load_tokenizer()
    if tok is not None:
        return len(tok.encode(text, add_special_tokens=False))
    return max(1, len(text.split()))
