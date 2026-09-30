"""Streamlit UI: topic-scoped practice questions with on-demand MCQ answers.

Talks to the Flask API (POST /api/questions, POST /api/mcq-answer).
Run: make ui  (backend must be up first: make up)
"""

from __future__ import annotations

import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:5000").rstrip("/")
TOPICS = ["Bond Pricing", "Modern portfolio theory", "Term structure theory"]
PAGE_SIZE = 2  # questions rendered per page

st.set_page_config(page_title="FRM MCQ Practice", layout="wide")


def call(path: str, *, timeout: int = 180, **payload) -> dict:
    resp = requests.post(f"{API_BASE}{path}", json=payload, timeout=timeout)
    body = resp.json()
    if resp.status_code != 200:
        raise RuntimeError(body.get("error") or body.get("hint") or resp.text)
    return body


def render_question(q: dict) -> None:
    qid = q["qid"]
    with st.expander(q["question_text"], expanded=True):
        for label, text in q["options"].items():
            st.markdown(f"**{label}.** {text}")

        if st.button("Answer", key=f"ans_{qid}", use_container_width=True):
            with st.spinner("Retrieving notes and answering..."):
                try:
                    st.session_state.answers[qid] = call(
                        "/api/mcq-answer",
                        question=q["question_text"],
                        options=q["options"],
                        include_sources=True,
                    )
                except Exception as exc:
                    st.session_state.answers[qid] = {"error": str(exc)}

        ans = st.session_state.answers.get(qid)
        if not ans:
            return
        if "error" in ans:
            st.error(ans["error"])
            return

        a = ans["answer"]
        st.success(f"**Selected: {a['selected_option_label']}. {a['selected_option_text']}**")
        st.markdown(f"**Why:** {a['explanation']}")
        for r in a.get("rejections", []):
            st.markdown(
                f"- ~~{r['option_label']}. {r['option_text']}~~ — {r['reason']}"
            )
        with st.expander(f"Sources ({len(ans.get('sources', []))})"):
            seen_urls: set[str] = set()
            for src in ans.get("sources", []):
                meta = src.get("metadata") or {}
                url = meta.get("doc_url")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    title = (
                        meta.get("reading_title")
                        or meta.get("module_title")
                        or "chunk"
                    )
                    page = meta.get("page_number")
                    label = f"{title} — page {page}" if page else title
                    st.markdown(f"- [{label}]({url})")


def main() -> None:
    st.title("FRM MCQ Practice")
    st.caption(f"Backend: {API_BASE}")

    with st.sidebar:
        st.header("Topics")
        checked = [t for t in TOPICS if st.checkbox(t, key=f"topic_{t}", value=True)]
        page_size = st.slider("Questions per fetch", min_value=3, max_value=50, value=20)
        fetch_btn = st.button("Fetch questions", type="primary", use_container_width=True)

    if fetch_btn:
        if not checked:
            st.error("Select at least one topic.")
        else:
            try:
                data = call("/api/questions", topics=checked, page_size=page_size)
            except Exception as exc:
                st.error(f"Failed to fetch questions: {exc}")
            else:
                st.session_state.questions = data["questions"]
                st.session_state.page = 0
                st.session_state.answers = {}
                if data.get("uncovered_topics"):
                    st.warning(
                        "No questions found for: "
                        + ", ".join(u["topic"] for u in data["uncovered_topics"])
                    )

    questions = st.session_state.get("questions")
    if questions is None:
        st.info("Pick topics and click *Fetch questions* to begin.")
        return

    if not questions:
        st.warning("No questions returned for the selected topics.")
        return

    page_count = max(1, (len(questions) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = min(st.session_state.get("page", 0), page_count - 1)

    c1, c2, c3 = st.columns([1, 3, 1])
    if c1.button("← Prev", disabled=page == 0, use_container_width=True):
        st.session_state.page -= 1
        st.rerun()
    c2.markdown(f"**Page {page + 1} / {page_count}** — {len(questions)} questions")
    if c3.button("Next →", disabled=page >= page_count - 1, use_container_width=True):
        st.session_state.page += 1
        st.rerun()

    st.divider()
    for q in questions[page * PAGE_SIZE : (page + 1) * PAGE_SIZE]:
        render_question(q)


if __name__ == "__main__":
    main()