# RAG tutor-agent — commands to run the DB operations after code generation.
# Usage: make env && make deps && make db-setup && make ingest && make run
# Reset: make reset && make ingest

SHELL := /bin/bash
QUERY ?= What is credit risk?
QUESTION ?= Which option best describes credit risk?
OPTIONS ?= {"A":"Risk that a counterparty fails to meet its contractual obligations","B":"Risk of losses from changes in market prices","C":"Risk of losses from system failures","D":"Risk of regulatory penalties"}
BOOKS_DIR ?=
QUESTIONS_DIR ?=
FORCE_REINGEST ?=
TOPICS ?= ["Futures and Forwards", "Credit Risk", "Option Pricing"]
PAGE_SIZE ?= 10
-include .env
.PHONY: env deps db-setup reset ingest ingest-questions ingest-questions-store run stop mcq-answer retrieve questions health ui

s:
	pipenv shell

r:
	pipenv install --verbose
	pipenv lock
	pipenv requirements > requirements.txt

# Copy .env.example -> .env (never overwrite an existing .env)
env:
	@test -f .env || cp .env.example .env
	@echo ".env ready (edit it and set DATABASE_URL / OPENROUTER_API_KEY)"

# Install Python dependencies via pipenv
deps:
	pipenv install --verbose

# Create the pgvector schema (frm_books table + HNSW index) using DATABASE_URL
db-setup:
	pipenv run python scripts/db_setup.py

# Drop the frm_books table and re-apply the schema (start fresh)
reset:
	pipenv run python scripts/reset.py

# Parse -> structure-aware chunk -> embed -> store into pgvector
ingest:
	pipenv run python scripts/ingest.py --books-dir "$(BOOKS_DIR)" $(FORCE_REINGEST)

# Parse -> extract questions -> LLM -> store in frm_questions
ingest-questions:
	pipenv run python scripts/ingest_questions.py --books-dir "$(QUESTIONS_DIR)"

# Load questions_*.json into frm_questions
ingest-questions-store:
	pipenv run python scripts/ingest_questions.py --phase store $(FORCE_REINGEST)

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

# Topic-scoped questions against a running server (override: TOPICS='["Futures","VaR"]' PAGE_SIZE=15)
questions:
	@curl -s -X POST http://127.0.0.1:5000/api/questions \
		-H 'Content-Type: application/json' \
		-d '{"topics": $(TOPICS), "page_size": $(PAGE_SIZE)}' | pipenv run python -m json.tool

# Readiness check
health:
	curl -s http://127.0.0.1:5000/api/health | pipenv run python -m json.tool

# Streamlit UI (backend must already be running via `make run`)
ui:
	pipenv run streamlit run streamlit_app.py

