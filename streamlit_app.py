"""Streamlit UI: topic-scoped practice questions with stored explanations.

Talks to the Flask API (GET /api/questions-metadata, POST /api/questions,
POST /api/check-answer).
Run: make ui  (backend must be running first: make run)
"""

from __future__ import annotations

import os
import json

import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:5000").rstrip("/")
TOPICS = [
    "Bond Pricing",
    "Modern portfolio theory",
    "Term structure theory",
    "Skewness",
    "Risk-Neutral Valuation",
    "Stress Testing",
    "Basic Indicator Approach",
    "Accounting",
    "Risk Measures",
    "Probability",
    "Statistical Distributions",
    "Option Strategies",
    "Risk Governance",
    "Quantitative Analysis",
    "Monthly Compounding",
    "Yield Curve Construction",
    "Risk Management",
    "Insurance",
    "Bootstrapping",
    "Value at Risk",
    "Greeks",
    "Market Risk",
    "Currency Risk",
    "Banking and Credit Risk",
    "Equity Derivatives",
    "Duration",
    "Interest Rate Risk",
    "Interest Rates",
    "Bond Valuation",
    "OTC derivatives",
    "Fixed Income - Swaps",
    "Futures",
    "Option Payoffs",
    "Risk Management Framework",
    "Option Pricing Models",
    "Volatility Estimation",
    "Volatility Modeling",
    "Commodity Futures",
    "Derivatives",
    "Credit Risk",
    "Probability Theory",
    "Independence",
    "Basel Accord",
    "Options and Futures",
    "Trading and Market Risk",
    "Liquidity Risk",
    "Discounting Cash Flows",
    "Duration and Convexity",
    "Liquidity Risk Management",
    "Margin and Leverage",
    "Regression Analysis",
    "Risk Identification",
    "Equity Strategies",
    "Statistical Inference",
    "Life Insurance",
    "Investment Management",
    "Statistical Concepts and Market Risk",
    "Cost of Carry",
    "Fixed Income - Yield Spreads",
    "Equity Market Neutral",
    "Bayes Theorem",
    "European Options",
    "Credit Rating Agencies",
    "Capital Adequacy",
    "Put-Call parity",
    "Operational Risk",
    "Compounding",
    "Fixed Income Derivatives",
    "Option Sensitivity",
    "Options Strategies",
    "Dividend adjustments",
    "Ethics and Professional Conduct",
    "Interest Rate Swaps",
    "Risk-Adjusted Performance Measures",
    "Credit Risk Modeling",
    "Hedging",
    "Sortino Ratio",
    "Corporate Governance",
    "Volatility and Correlation",
    "Immunization",
    "Expected Loss",
    "Counterparty Risk",
    "Compounding Interest Rates",
    "Futures and Options",
    "Options",
    "Coefficient of Determination",
    "Cash Carry",
    "Interest Rate Calculations",
    "Option pricing",
    "Statistics",
    "Credit Risk Management",
    "Supervisory Review",
    "Least Squares",
    "Multicollinearity",
    "Statistical Methods",
    "Futures and Forward Pricing",
    "Resampling Methods",
    "Delivery Options",
    "Investment Performance Measurement",
    "Fixed Income",
    "Recovery Rates",
    "Ethics and Professionalism",
    "Hedge Fund Fees and Performance",
    "Time Series Analysis",
    "Correlation and Covariance",
    "Hypothesis Testing",
    "Risk Measurement",
    "Treasury Bonds",
    "American Options",
    "Value at Risk (VaR)",
    "Performance Measurement",
    "CAPM",
    "Insurance and Actuarial Science",
    "Statistical Analysis",
    "Option Sensitivities",
    "Accrued Interest",
    "Fixed Income - Bonds",
    "Nominal vs Effective Rates",
    "Bond Duration",
    "Arbitrage",
    "Incentive Compensation",
    "Capital Asset Pricing Model",
    "Variance Inflation Factor",
    "Continuous Compounding",
    "Credit Ratings",
    "Yield Curve",
    "Repo",
    "Bond Pricing and Yield",
    "Business Risk",
    "GARCH",
    "Present Value",
    "Liquidity",
    "Portfolio Theory",
    "Hedging Strategies",
    "Binomial Models",
    "Portfolio Management",
    "Asset Pricing Models",
    "Corporate Risk Management",
    "Descriptive Statistics",
    "Sovereign Risk",
    "Regulatory Risk",
    "Conditional Probability",
    "Index Futures",
    "Continuous Distributions",
    "Risk-Adjusted Return",
    "Volatility",
    "Legal Risk",
    "Derivatives - Option Pricing",
    "Ethics and Professional Standards",
]
# One-click topic groups, named after the FRM syllabus papers/chapters. Every
# TOPICS entry must appear in exactly one group (asserted at import below).
TOPIC_GROUPS: dict[str, list[str]] = {
    "Foundations & Ethics": [
        "Risk Management", "Risk Management Framework", "Risk Governance",
        "Risk Identification", "Corporate Risk Management", "Business Risk",
        "Corporate Governance", "Accounting", "Ethics and Professional Conduct",
        "Ethics and Professionalism", "Ethics and Professional Standards",
        "Insurance", "Life Insurance", "Insurance and Actuarial Science",
    ],
    "Quantitative Analysis": [
        "Quantitative Analysis", "Probability", "Probability Theory",
        "Conditional Probability", "Independence", "Statistical Distributions",
        "Continuous Distributions", "Statistics", "Descriptive Statistics",
        "Statistical Analysis", "Statistical Methods", "Statistical Inference",
        "Statistical Concepts and Market Risk", "Skewness",
        "Correlation and Covariance", "Regression Analysis", "Least Squares",
        "Multicollinearity", "Variance Inflation Factor",
        "Coefficient of Determination", "Hypothesis Testing", "Bayes Theorem",
        "Resampling Methods", "Time Series Analysis", "GARCH",
    ],
    "Markets, Derivatives & Hedging": [
        "Derivatives", "Futures", "OTC derivatives", "Repo",
        "Cost of Carry", "Cash Carry", "Delivery Options", "Index Futures",
        "Commodity Futures", "Margin and Leverage", "Currency Risk",
        "Market Risk", "Trading and Market Risk", "Interest Rates",
        "Futures and Options", "Options and Futures",
        "Futures and Forward Pricing", "Hedging", "Hedging Strategies",
        "Arbitrage",
    ],
    "Fixed Income — Bonds & Yields": [
        "Bond Pricing", "Bond Valuation", "Bond Pricing and Yield",
        "Bond Duration", "Duration", "Duration and Convexity", "Fixed Income",
        "Fixed Income - Bonds", "Fixed Income - Yield Spreads", "Yield Curve",
        "Yield Curve Construction", "Term structure theory", "Bootstrapping",
        "Accrued Interest", "Treasury Bonds", "Immunization",
    ],
    "Fixed Income — Rates & Swaps": [
        "Fixed Income - Swaps", "Fixed Income Derivatives",
        "Interest Rate Calculations", "Interest Rate Risk",
        "Interest Rate Swaps", "Present Value", "Discounting Cash Flows",
        "Compounding", "Continuous Compounding", "Monthly Compounding",
        "Compounding Interest Rates", "Nominal vs Effective Rates",
    ],
    "Equity & Portfolio Management": [
        "Equity Derivatives", "Equity Strategies", "Equity Market Neutral",
        "CAPM", "Capital Asset Pricing Model", "Asset Pricing Models",
        "Portfolio Theory", "Modern portfolio theory", "Portfolio Management",
        "Investment Management", "Investment Performance Measurement",
        "Performance Measurement", "Risk-Adjusted Performance Measures",
        "Risk-Adjusted Return", "Sortino Ratio",
        "Hedge Fund Fees and Performance", "Incentive Compensation",
    ],
    "Options & Volatility": [
        "Options", "Option Payoffs", "Option Strategies", "Options Strategies",
        "Option pricing", "Option Pricing Models",
        "Derivatives - Option Pricing", "Option Sensitivity",
        "Option Sensitivities", "Greeks", "Binomial Models",
        "European Options", "American Options", "Put-Call parity",
        "Risk-Neutral Valuation", "Dividend adjustments", "Volatility",
        "Volatility Estimation", "Volatility Modeling",
        "Volatility and Correlation",
    ],
    "VaR & Risk Models": [
        "Value at Risk", "Value at Risk (VaR)", "Stress Testing",
        "Risk Measures", "Risk Measurement", "Liquidity",
    ],
    "Credit Risk": [
        "Credit Risk", "Credit Risk Modeling", "Credit Risk Management",
        "Banking and Credit Risk", "Expected Loss", "Recovery Rates",
        "Credit Ratings", "Credit Rating Agencies", "Counterparty Risk",
        "Basel Accord", "Capital Adequacy", "Supervisory Review",
    ],
    "Liquidity, Operational & Other Risks": [
        "Liquidity Risk", "Liquidity Risk Management", "Operational Risk",
        "Basic Indicator Approach", "Legal Risk", "Sovereign Risk",
        "Regulatory Risk",
    ],
}
assert sorted(t for g in TOPIC_GROUPS.values() for t in g) == sorted(TOPICS), (
    "TOPIC_GROUPS must cover every TOPICS entry exactly once"
)
PAGE_SIZE = 2  # questions rendered per page

st.set_page_config(page_title="FRM MCQ Practice", layout="wide")


def call(path: str, *, timeout: int = 180, **payload) -> dict:
    resp = requests.post(f"{API_BASE}{path}", json=payload, timeout=timeout)
    try:
        body = resp.json()
    except ValueError:
        raise RuntimeError(f"{resp.status_code} {resp.reason}: {resp.text[:300]}")
    if resp.status_code != 200:
        raise RuntimeError(body.get("error") or body.get("hint") or resp.text)
    return body


def call_get(path: str, *, timeout: int = 30, **params) -> dict:
    resp = requests.get(f"{API_BASE}{path}", params=params, timeout=timeout)
    try:
        body = resp.json()
    except ValueError:
        raise RuntimeError(f"{resp.status_code} {resp.reason}: {resp.text[:300]}")
    if resp.status_code != 200:
        raise RuntimeError(body.get("error") or body.get("hint") or resp.text)
    return body


def call_put(path: str, *, timeout: int = 30, **payload) -> dict:
    resp = requests.put(f"{API_BASE}{path}", json=payload, timeout=timeout)
    try:
        body = resp.json()
    except ValueError:
        raise RuntimeError(f"{resp.status_code} {resp.reason}: {resp.text[:300]}")
    if resp.status_code != 200:
        raise RuntimeError(body.get("error") or body.get("hint") or resp.text)
    return body


def remember_answer(qid: int) -> None:
    st.session_state.answers[qid] = st.session_state.get(f"choice_{qid}")


def copy_question_widget(q: dict) -> None:
    text = f"{q['question_text']}\n\nOptions:\n" + "\n".join(
        f"{label}. {option}" for label, option in q["options"].items()
    )
    safe_text = json.dumps(text).replace("<", "\\u003c").replace(">", "\\u003e")
    components.html(
        f"""
        <button id="copy" title="Copy question and options" aria-label="Copy question and options">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <rect x="8" y="8" width="11" height="12" rx="1.5"></rect>
            <path d="M16 8V5.5A1.5 1.5 0 0 0 14.5 4h-9A1.5 1.5 0 0 0 4 5.5v10A1.5 1.5 0 0 0 5.5 17H8"></path>
          </svg>
        </button>
        <style>
          body {{ margin: 0; display: flex; justify-content: flex-end; }}
          button {{ border: 0; background: transparent; color: #64748b; cursor: pointer; padding: 5px; }}
          button:hover {{ color: #0f172a; }}
          svg {{ width: 20px; height: 20px; fill: none; stroke: currentColor; stroke-width: 1.8; }}
        </style>
        <script>
          const button = document.getElementById("copy");
          const icon = button.innerHTML;
          button.onclick = async () => {{
            try {{
              await navigator.clipboard.writeText({safe_text});
              button.textContent = "✓";
              setTimeout(() => button.innerHTML = icon, 1200);
            }} catch {{
              button.textContent = "!";
            }}
          }};
        </script>
        """,
        height=32,
        scrolling=False,
    )


def render_question(q: dict) -> None:
    qid = q["qid"]
    with st.expander(f"Question {qid}", expanded=True):
        question_col, copy_col = st.columns([0.94, 0.06], vertical_alignment="top")
        question_col.markdown(f"**{q['question_text']}**")
        with copy_col:
            copy_question_widget(q)
        st.session_state.setdefault("answers", {})
        st.session_state.setdefault(f"choice_{qid}", st.session_state.answers.get(qid))
        st.radio(
            "Choose an option",
            options=list(q["options"].keys()),
            format_func=lambda label: f"{label}. {q['options'][label]}",
            key=f"choice_{qid}",
            index=None,
            on_change=remember_answer,
            args=(qid,),
            label_visibility="collapsed",
        )

        checked = st.session_state.checks.get(qid)
        selected = st.session_state.answers.get(qid)
        if checked and checked["submitted"] == selected:
            correct_answer = checked.get("correct_answer") or ""
            explanation = checked.get("explanation") or ""
            if not correct_answer and not explanation:
                st.info("The answer and explanation are not available for this question yet.")
            elif not correct_answer:
                st.info("The correct answer is not available for this question yet.")
            elif checked["chosen"] is None:
                st.info(f"**Not answered** — correct answer: {correct_answer}")
            elif checked["is_correct"]:
                st.success("**Correct**")
            else:
                st.error(f"**Incorrect** — correct answer: {correct_answer}")
            if st.button("Explanation", key=f"explanation_{qid}", use_container_width=True):
                st.session_state.explanations[qid] = True
            if st.session_state.explanations.get(qid):
                st.markdown(
                    f"**Explanation:** {explanation or 'An explanation is not available for this question yet.'}"
                )


def batch_as_markdown(questions: list[dict], checks: dict) -> str:
    lines = ["# FRM Practice Questions", ""]
    for number, question in enumerate(questions, start=1):
        qid = question["qid"]
        checked = checks.get(qid, {})
        selected = st.session_state.answers.get(qid)
        correct = checked.get("correct_answer") or question.get("answer", "")
        result = (
            "Correct"
            if checked.get("is_correct")
            else "Incorrect"
            if checked
            else "Not checked"
        )
        explanation = checked.get("explanation") or question.get("explanation", "")

        lines.extend([f"## {number}. {question['question_text']}", "", "### Options", ""])
        lines.extend(
            f"- **{label}.** {text}" for label, text in question["options"].items()
        )
        lines.extend(
            [
                "",
                f"**Selected answer:** {selected or 'Not answered'}",
                f"**Correct answer:** {correct or 'Unknown'}",
                f"**Result:** {result}",
                "",
                "### Explanation",
                "",
                explanation or "An explanation is not available for this question yet.",
                "",
            ]
        )
    return "\n".join(lines)


@st.dialog("Load more questions?")
def confirm_load_next_batch() -> None:
    st.write(
        "Your current batch will be replaced by the next batch. "
        "Any selections, answer checks, and explanations from this batch "
        "will be cleared, and you will not be able to go back to it."
    )
    st.download_button(
        "Export current batch",
        data=batch_as_markdown(
            st.session_state.get("questions", []),
            st.session_state.get("checks", {}),
        ),
        file_name=f"frm_questions_batch_{st.session_state.get('server_page', 1)}.md",
        mime="text/markdown",
        use_container_width=True,
    )
    if st.button("Yes, load next batch", type="primary", use_container_width=True):
        st.session_state.load_next_batch_confirmed = True
        st.rerun()


def main() -> None:
    st.title("FRM MCQ Practice")
    st.session_state.setdefault("explanations", {})
    st.session_state.setdefault("answers", {})

    with st.sidebar:
        if "show_attempted" not in st.session_state:
            try:
                st.session_state.show_attempted = call_get(
                    "/api/question-preferences"
                ).get("show_attempted", False)
                st.session_state.question_preference_error = None
            except Exception as exc:
                st.session_state.show_attempted = False
                st.session_state.question_preference_error = str(exc)
            st.session_state.saved_show_attempted = st.session_state.show_attempted
        st.session_state.setdefault(
            "saved_show_attempted", st.session_state.show_attempted
        )

        show_attempted = st.toggle(
            "Show attempted questions",
            key="show_attempted",
            help="Off shows unattempted questions; on shows attempted questions.",
        )
        if show_attempted != st.session_state.saved_show_attempted:
            try:
                call_put(
                    "/api/question-preferences",
                    show_attempted=show_attempted,
                )
                st.session_state.saved_show_attempted = show_attempted
                st.session_state.question_preference_error = None
            except Exception as exc:
                st.session_state.question_preference_error = str(exc)
        if st.session_state.get("question_preference_error"):
            st.warning(
                "Preference could not be saved: "
                + st.session_state.question_preference_error
            )

        st.header("PDF sources")
        refresh_pdf_names = st.button("Refresh PDF list", use_container_width=True)
        if "pdf_names" not in st.session_state or refresh_pdf_names:
            try:
                st.session_state.pdf_names = call_get(
                    "/api/questions-metadata", filter="pdfs"
                ).get("pdf_names", [])
                st.session_state.pdf_names_error = None
            except Exception as exc:
                st.session_state.pdf_names = []
                st.session_state.pdf_names_error = str(exc)

        pdf_names = st.session_state.get("pdf_names", [])
        pdf_names_error = st.session_state.get("pdf_names_error")
        selected_pdf_names = []
        if pdf_names_error:
            st.error(f"Failed to load PDF names: {pdf_names_error}")
        else:
            st.caption(f"{len(pdf_names)} PDF source(s) available")
            with st.expander("Show PDF names", expanded=False):
                if pdf_names:
                    for pdf_name in pdf_names:
                        st.caption(pdf_name)
                else:
                    st.info("No PDF names found.")
            selected_pdf_names = st.multiselect(
                "PDF sources for questions",
                options=pdf_names,
                help="Leave empty to search all PDF sources.",
            )

        st.header("Topics")
        all_topics = st.checkbox("All topics", key="topic_all", value=False)
        st.caption("One click adds a whole group (click again to add more):")
        group_cols = st.columns(2)
        for i, (gname, gtopics) in enumerate(TOPIC_GROUPS.items()):
            if group_cols[i % 2].button(
                f"{gname} ({len(gtopics)})",
                key=f"group_{i}",
                disabled=all_topics,
                use_container_width=True,
            ):
                cur = st.session_state.get("topic_pick") or []
                st.session_state.topic_pick = list(dict.fromkeys(cur + gtopics))
        if st.button("Clear selection", key="group_clear", disabled=all_topics,
                     use_container_width=True):
            st.session_state.topic_pick = []
        picked = st.multiselect(
            "Topics (type to filter)",
            options=TOPICS,
            key="topic_pick",
            disabled=all_topics,
        )
        checked = [] if all_topics else picked
        page_size = st.slider("Questions per fetch", min_value=3, max_value=100, value=100)
        fetch_btn = st.button("Fetch questions", type="primary", use_container_width=True)

    if fetch_btn:
        if not all_topics and not checked and not selected_pdf_names:
            st.error("Select at least one topic or PDF source.")
        else:
            try:
                data = call(
                    "/api/questions",
                    pdf_names=selected_pdf_names,
                    topics=checked,
                    attempted=show_attempted,
                    page_number=1,
                    page_size=page_size,
                )
            except Exception as exc:
                st.error(f"Failed to fetch questions: {exc}")
            else:
                st.session_state.questions = data["questions"]
                st.session_state.server_page = 1
                st.session_state.has_more = data.get("has_more", False)
                st.session_state.fetch_topics = checked
                st.session_state.fetch_pdf_names = selected_pdf_names
                st.session_state.fetch_attempted = show_attempted
                st.session_state.page = 0
                st.session_state.checks = {}
                st.session_state.explanations = {}
                st.session_state.answers = {}
                if data.get("uncovered_topics"):
                    st.warning(
                        "No questions found for: "
                        + ", ".join(u["topic"] for u in data["uncovered_topics"])
                    )

    questions = st.session_state.get("questions")
    if questions is None:
        st.info("Pick topics from right side menu and click *Fetch questions* to begin.")
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
    page_qs = questions[page * PAGE_SIZE : (page + 1) * PAGE_SIZE]
    for q in page_qs:
        render_question(q)

    if page >= page_count - 1 and st.session_state.get("has_more"):
        if st.button("Load more questions", use_container_width=True):
            confirm_load_next_batch()

    if st.session_state.pop("load_next_batch_confirmed", False):
        try:
            with st.spinner("Fetching more questions..."):
                data = call(
                    "/api/questions",
                    pdf_names=st.session_state.get("fetch_pdf_names", []),
                    topics=st.session_state.get("fetch_topics", []),
                    attempted=st.session_state.get("fetch_attempted", False),
                    page_number=st.session_state.server_page + 1,
                    page_size=page_size,
                )
        except Exception as exc:
            st.error(f"Failed to fetch more questions: {exc}")
        else:
            st.session_state.questions = data["questions"]
            st.session_state.server_page += 1
            st.session_state.has_more = data.get("has_more", False)
            st.session_state.page = 0
            st.session_state.checks = {}
            st.session_state.explanations = {}
            st.session_state.answers = {}
            st.rerun()

    if st.button("Check Answers", type="primary", use_container_width=True):
        payload = [
            {"qid": q["qid"], "answer": st.session_state.answers.get(q["qid"])}
            for q in page_qs
        ]
        try:
            data = call("/api/check-answer", answers=payload)
        except Exception as exc:
            st.error(f"Failed to check answers: {exc}")
        else:
            st.session_state.explanations = {}
            st.session_state.checks.update(
                {
                    r["qid"]: {
                        "submitted": st.session_state.answers.get(r["qid"]),
                        "chosen": r["chosen_answer"],
                        "is_correct": r["is_correct"],
                        "correct_answer": r["correct_answer"],
                        "explanation": r.get("explanation", ""),
                    }
                    for r in data["results"]
                }
            )
            st.rerun()


if __name__ == "__main__":
    main()
