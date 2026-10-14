"""Self-check for the /api/check-answer label comparison.

Run:  python test_check_answer.py
No framework: a few asserts over the pure helper.
"""

from app.questions.routes import _is_correct, _normalize


def main() -> None:
    assert _normalize(" a ") == "A"
    assert _is_correct("A", "A")
    assert _is_correct("a", " A ")
    assert not _is_correct("B", "A")
    assert not _is_correct("A", None)  # unknown qid -> not correct
    assert not _is_correct(None, "A")  # unanswered -> not correct
    assert not _is_correct(None, None)
    print("test_check_answer: OK")


if __name__ == "__main__":
    main()
