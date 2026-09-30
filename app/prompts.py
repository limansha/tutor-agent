"""Prompt template loader: renders prompts from the prompts/ directory.

Templates are Jinja2 files under prompts/. Each is loaded once via
langchain_core.prompts.PromptTemplate.from_file and rendered with kwargs.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.prompts import PromptTemplate

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

_templates: dict[str, PromptTemplate] = {}


def load(name: str) -> PromptTemplate:
    """Load (and cache) a PromptTemplate from prompts/<name>."""
    if name not in _templates:
        _templates[name] = PromptTemplate.from_file(
            PROMPTS_DIR / name,
            template_format="jinja2",
        )
    return _templates[name]


def render(name: str, **kwargs) -> str:
    """Render a prompt template to a string with the given variables."""
    return load(name).format(**kwargs)