"""Self-check for the web-mode citation -> SourceChunk mapper.

Run:  python test_web_sources.py
No framework: a few asserts over fake OpenRouter annotation objects.
"""

from types import SimpleNamespace

from app.agent import _MAX_CITATION_CHARS, _citations_to_sources
from app.websearch import _LiteParser, _excluded, _pair_events, _unwrap


def _cite(url, title, content):
    return SimpleNamespace(
        url_citation=SimpleNamespace(url=url, title=title, content=content)
    )


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


def main() -> None:
    annotations = [
        _cite("https://www.garp.org/a", "GARP", "x" * 5000),
        _cite("https://www.garp.org/a", "duplicate url", "dropped"),
        SimpleNamespace(url_citation=None),
        _cite("https://www.bis.org/b", "", ""),
    ]
    out = _citations_to_sources(annotations)

    assert len(out) == 2, out
    assert all(s.role == "web" for s in out)
    assert out[0].metadata.doc_url == "https://www.garp.org/a"
    assert out[0].metadata.reading_title == "GARP"
    assert len(out[0].text) == _MAX_CITATION_CHARS  # excerpt trimmed
    assert out[1].metadata.reading_title == "https://www.bis.org/b"  # title -> url
    assert out[1].text == "https://www.bis.org/b"  # content -> url
    assert _citations_to_sources(None) == []
    _test_ddg_parser()
    print("test_web_sources: OK")


if __name__ == "__main__":
    main()
