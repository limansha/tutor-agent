"""Self-check for the web-search result -> SourceChunk mapper + DDG-Lite parser.

Run:  python test_web_sources.py
No framework: a few asserts over fake search-result dicts and a fake DDG page.
Requires the app deps (imports app.agent), so run inside the pipenv venv.
"""

from app.agent import _MAX_SOURCE_CHARS, _result_sources
from app.websearch import _LiteParser, _excluded, _pair_events, _unwrap

_FIXTURE = """
<table>
<tr><td>1.</td><td><a rel="nofollow" href="https://www.bis.org/a" class='result-link'>BIS A</a></td></tr>
<tr><td></td><td class='result-snippet'>snippet A</td></tr>
<tr><td>2.</td><td><a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.garp.org%2Fb" class='result-link'>GARP B</a></td></tr>
<tr><td></td><td class='result-snippet'>snippet B</td></tr>
</table>
"""


def _test_ddg_parser() -> None:
    parser = _LiteParser()
    parser.feed(_FIXTURE)
    results = _pair_events(parser.events)

    assert len(results) == 2, results
    assert results[0]["url"] == "https://www.bis.org/a"
    assert results[0]["title"] == "BIS A" and results[0]["snippet"] == "snippet A"
    assert results[1]["url"] == "https://www.garp.org/b"  # unwrapped
    assert _unwrap("https://x.org/y") == "https://x.org/y"  # untouched

    assert _excluded("https://www.quora.com/q", ["quora.com"])
    assert _excluded("https://m.reddit.com/x", ["reddit.com"])
    assert not _excluded("https://www.bis.org/x", ["quora.com", "reddit.com"])


def _test_result_sources() -> None:
    seen: set[str] = set()
    results = [
        {"url": "https://www.garp.org/a", "title": "GARP", "snippet": "x" * 5000},
        {"url": "https://www.garp.org/a", "title": "duplicate url", "snippet": "dropped"},
        {"url": "https://www.bis.org/b", "title": "", "snippet": ""},
    ]
    out = _result_sources(results, seen)

    assert len(out) == 2, out
    assert all(s.role == "web" for s in out)
    assert out[0].metadata.doc_url == "https://www.garp.org/a"
    assert out[0].metadata.reading_title == "GARP"
    assert len(out[0].text) == _MAX_SOURCE_CHARS  # excerpt trimmed
    assert out[1].metadata.reading_title == "https://www.bis.org/b"  # title -> url
    assert out[1].text == "https://www.bis.org/b"  # snippet -> url

    # shared `seen` set de-dupes across separate tool calls
    assert _result_sources(results, seen) == []


def main() -> None:
    _test_result_sources()
    _test_ddg_parser()
    print("test_web_sources: OK")


if __name__ == "__main__":
    main()
