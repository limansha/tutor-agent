"""Free web search for web-mode MCQ answers (DuckDuckGo Lite, no API key).

This is the free-tier alternative to OpenRouter's paid `web` plugin (which 402s
without purchased credits). It scrapes the DDG-Lite HTML with only
`requests` + stdlib `html.parser`; on any failure it returns no results and the
model falls back to its own knowledge.

ponytail: best-effort single-page scrape — swap WEB_SEARCH_PROVIDER for a real
search API if DDG starts blocking or you need reliability.
"""

from __future__ import annotations

from html.parser import HTMLParser
from urllib.parse import parse_qs, urlparse

import requests

_DDG_LITE = "https://lite.duckduckgo.com/lite/"
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


class _LiteParser(HTMLParser):
    """Collect ('link', url, title) and ('snippet', '', text) in document order."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.events: list[tuple[str, str, str]] = []
        self._mode: str | None = None
        self._href = ""
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        cls = a.get("class") or ""
        if tag == "a" and "result-link" in cls:
            self._mode, self._href, self._buf = "link", a.get("href") or "", []
        elif tag == "td" and "result-snippet" in cls:
            self._mode, self._buf = "snippet", []

    def handle_endtag(self, tag: str) -> None:
        if self._mode == "link" and tag == "a":
            self.events.append(("link", self._href, "".join(self._buf).strip()))
            self._mode = None
        elif self._mode == "snippet" and tag == "td":
            self.events.append(("snippet", "", "".join(self._buf).strip()))
            self._mode = None

    def handle_data(self, data: str) -> None:
        if self._mode:
            self._buf.append(data)


def _unwrap(url: str) -> str:
    """DDG-Lite sometimes wraps links as //duckduckgo.com/l/?uddg=<real-url>."""
    if "duckduckgo.com/l/" in url:
        q = parse_qs(urlparse(url).query).get("uddg")
        if q:
            return q[0]
    return url


def _excluded(url: str, domains: tuple[str, ...] | list[str]) -> bool:
    host = urlparse(url).netloc.lower().split(":")[0]
    return any(host == d or host.endswith("." + d) for d in domains)


def _pair_events(events: list[tuple[str, str, str]]) -> list[dict[str, str]]:
    """Turn parser events into {url, title, snippet} dicts (link then snippet)."""
    results: list[dict[str, str]] = []
    pending: dict[str, str] | None = None
    for kind, url, text in events:
        if kind == "link":
            if pending:
                results.append(pending)
            pending = {"url": _unwrap(url), "title": text, "snippet": ""}
        elif kind == "snippet" and pending and not pending["snippet"]:
            pending["snippet"] = text
    if pending:
        results.append(pending)
    return results


def search(
    query: str,
    max_results: int = 3,
    exclude_domains: tuple[str, ...] | list[str] = (),
) -> list[dict[str, str]]:
    """Return up to `max_results` {url, title, snippet} dicts; [] on any failure."""
    try:
        resp = requests.post(
            _DDG_LITE, data={"q": query}, headers={"User-Agent": _UA}, timeout=15
        )
        resp.raise_for_status()
    except requests.RequestException:
        return []

    parser = _LiteParser()
    parser.feed(resp.text)
    results = _pair_events(parser.events)

    return [
        r for r in results if r["url"] and not _excluded(r["url"], exclude_domains)
    ][:max_results]
