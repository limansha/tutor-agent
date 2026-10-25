# RAG tutor-agent commands.
# Usage: make env && make deps && make db-setup && make ingest && make run

SHELL := /bin/bash
QUERY ?= What is credit risk?
QUESTION ?= Which option best describes credit risk?
OPTIONS ?= {"A":"Risk that a counterparty fails to meet its contractual obligations","B":"Risk of losses from changes in market prices","C":"Risk of losses from system failures","D":"Risk of regulatory penalties"}
BOOKS_DIR ?=
QUESTIONS_DIR ?=
FORCE_REINGEST ?=
TOPICS ?= ["Futures and Forwards", "Credit Risk", "Option Pricing"]
SEARCH_TOPICS ?= []
PAGE ?= 1
PAGE_SIZE ?= 10
BACKFILL_LIMIT ?= 3
-include .env
.PHONY: env deps db-setup reset ingest ingest-questions-json ingest-questions-store backfill-question-metadata schedule-backfill run stop mcq-answer retrieve questions health ui

# Copy .env.example -> .env (never overwrite an existing .env)
env:
	@test -f .env || cp .env.example .env
	@echo ".env ready (edit it and set DATABASE_URL / GEMINI_API_KEY)"

s:
	pipenv shell

r:
	pipenv install --verbose
	pipenv lock
	pipenv requirements > requirements.txt

# Create the pgvector schema (frm_books + question tables) using DATABASE_URL
db-setup:
	pipenv run python scripts/db_setup.py

# Drop the frm_books table and re-apply the schema (start fresh)
reset:
	pipenv run python scripts/reset.py

# Parse -> structure-aware chunk -> embed -> store into pgvector
ingest:
	pipenv run python scripts/ingest.py --books-dir "$(BOOKS_DIR)" $(FORCE_REINGEST)

# Explicit name for PDF directory -> JSON-only ingestion.
# Usage: make ingest-questions-json QUESTIONS_DIR=/path/to/pdf-directory
ingest-questions-json:
	@test -d "$(QUESTIONS_DIR)" || { echo "ERROR: QUESTIONS_DIR must be an existing PDF directory"; exit 1; }
	pipenv run python scripts/ingest_questions.py --phase 1 --books-dir "$(QUESTIONS_DIR)"

# Independently load successful question JSON files into frm_questions
ingest-questions-store:
	pipenv run python scripts/store_questions.py

# Retry only rows whose answer, explanation, or topics are still empty.
# Usage: make backfill-question-metadata [BACKFILL_LIMIT=3]
backfill-question-metadata:
	pipenv run python scripts/backfill_question_metadata.py --limit $(BACKFILL_LIMIT)

# Run the backfill every three minutes, three questions per cycle, using Gemini.
schedule-backfill:
	pipenv run python scripts/schedule_backfill.py

# Start the Flask server (foreground; stops any stale server first)
run: stop
	pipenv run python run.py

# Stop the running Flask server
stop:
	@pkill -f "[r]un\.py" || echo "no server running"
	@sleep 1

# Agentic MCQ chat against a running server (override QUESTION="..." OPTIONS='{"A":"...","B":"..."}')
mcq-answer:
	@curl -s -X POST http://127.0.0.1:5000/api/mcq-answer \
		-H 'Content-Type: application/json' \
		-d '{"question": "$(QUESTION)", "options": $(OPTIONS)}' | pipenv run python -m json.tool

# RAG chunk retrieval against a running server (override: QUERY="...")
retrieve:
	curl -s -X POST http://127.0.0.1:5000/api/retrieve \
		-H 'Content-Type: application/json' \
		-d "{\"query\": \"$(QUERY)\"}" | pipenv run python -m json.tool

# Topic-scoped questions against a running server
# (override: TOPICS='["Futures","VaR"]' SEARCH_TOPICS='["duration"]' PAGE=2 PAGE_SIZE=15)
questions:
	@curl -s -X POST http://127.0.0.1:5000/api/questions \
		-H 'Content-Type: application/json' \
		-d '{"topics": $(TOPICS), "search_topics": $(SEARCH_TOPICS), "page_number": $(PAGE), "page_size": $(PAGE_SIZE)}' | pipenv run python -m json.tool

# Readiness check
health:
	curl -s http://127.0.0.1:5000/api/health | pipenv run python -m json.tool

# Streamlit UI (backend must already be running via `make run`)
ui:
	pipenv run streamlit run streamlit_app.py
