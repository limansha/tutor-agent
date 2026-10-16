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
DUMP ?= frm_data.dump
# Prefix for Render targets: forces the run-time DATABASE_URL (external) over the
# local one in .env. The PIPENV_DONT_LOAD_ENV is REQUIRED — pipenv otherwise loads
# .env and wins. Never stores the URL anywhere.
TARGET_DB = PIPENV_DONT_LOAD_ENV=1 DATABASE_URL="$(DATABASE_URL)"
TOPICS ?= ["Futures and Forwards", "Credit Risk", "Option Pricing"]
SEARCH_TOPICS ?= []
PAGE ?= 1
PAGE_SIZE ?= 10
-include .env
.PHONY: env deps db-setup reset ingest ingest-questions ingest-questions-store ingest-questions-render restore-render _require-db-url dump restore run stop mcq-answer retrieve questions health ui

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


# Create the pgvector schema (frm_books + question tables) using DATABASE_URL
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

# Extract questions from QUESTIONS_DIR into questions_dump/*.json AND load them
# into DATABASE_URL. For Render, pass its EXTERNAL url at run time (never commit):
#   make ingest-questions-render DATABASE_URL="postgresql://...external..."
ingest-questions-render: _require-db-url
	$(TARGET_DB) pipenv run python scripts/ingest_questions.py --books-dir "$(QUESTIONS_DIR)"
	$(TARGET_DB) pipenv run python scripts/ingest_questions.py --phase store $(FORCE_REINGEST)

# One command to rebuild a fresh Render DB after the free 30-day expiry:
# applies the schema (drops+recreates frm_questions) then loads questions_dump/*.json.
#   make restore-render DATABASE_URL="postgresql://...external..."
restore-render: _require-db-url
	$(TARGET_DB) pipenv run python scripts/db_setup.py
	$(TARGET_DB) pipenv run python scripts/ingest_questions.py --phase store

# Guard: refuse to run Render targets unless DATABASE_URL came from the command
# line, so the local URL in .env is never silently targeted.
_require-db-url:
	@test "$(origin DATABASE_URL)" = "command line" || { \
		echo "ERROR: pass DATABASE_URL=<render external url> on the command line (do NOT commit it)."; \
		exit 1; }

# Back up the DB to a custom-format pg_dump file. Point at Render's EXTERNAL URL:
#   make dump DATABASE_URL="postgresql://...external..."
# Command-line DATABASE_URL overrides `-include .env` (no PIPENV_DONT_LOAD_ENV needed;
# pg_dump is not run through pipenv).
dump:
	pg_dump "$(DATABASE_URL)" -Fc -f "$(DUMP)"
	@echo "wrote $(DUMP)"

# Restore a dump into DATABASE_URL. Run `make db-setup` first (dump is data-only).
#   make restore DATABASE_URL="..." DUMP=frm_data.dump
restore:
	pg_restore --no-owner --data-only -d "$(DATABASE_URL)" "$(DUMP)"

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

