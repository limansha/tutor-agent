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
.PHONY: env deps s r db-setup reset ingest ingest-questions ingest-questions-store run stop up restart chat retrieve questions health mcq-answer ui ui-up ui-stop

# Copy .env.example -> .env (never overwrite an existing .env)
env:
	@test -f .env || cp .env.example .env
	@echo ".env ready (edit it and set DATABASE_URL / OPENROUTER_API_KEY)"

# Install Python dependencies via pipenv

deps:
	pipenv install --verbose

s:
	pipenv shell

r:
	pipenv install --verbose
	pipenv lock
	pipenv requirements > requirements.txt

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

# Start the Flask server in the background (detached, survives shell exit)
up:
	@setsid nohup pipenv run python run.py > /tmp/opencode/flask.log 2>&1 < /dev/null & disown
	@sleep 4
	@echo "server up -> http://127.0.0.1:5000 (log: /tmp/opencode/flask.log)"

# Stop then start the server in the background
restart: stop up

# Agentic MCQ chat against a running server (override QUESTION="..." OPTIONS='{"A":"...","B":"..."}')
mcq-answer:
	@printf '{"question":"%s","options":%s}' "$(QUESTION)" '$(OPTIONS)' > /tmp/opencode/chat_payload.json
	@curl -s -X POST http://127.0.0.1:5000/api/mcq-answer \
		-H 'Content-Type: application/json' \
		--data @/tmp/opencode/chat_payload.json | pipenv run python -m json.tool

# RAG chunk retrieval against a running server (override: QUERY="...")
retrieve:
	curl -s -X POST http://127.0.0.1:5000/api/retrieve \
		-H 'Content-Type: application/json' \
		-d "{\"query\": \"$(QUERY)\"}" | pipenv run python -m json.tool

# Topic-scoped questions against a running server (override: TOPICS='["Futures","VaR"]' PAGE_SIZE=15)
questions:
	@printf '{"topics": %s, "page_size": %s}' '$(TOPICS)' '$(PAGE_SIZE)' > /tmp/opencode/questions_payload.json
	@curl -s -X POST http://127.0.0.1:5000/api/questions \
		-H 'Content-Type: application/json' \
		--data @/tmp/opencode/questions_payload.json | pipenv run python -m json.tool

# Readiness check
health:
	curl -s http://127.0.0.1:5000/api/health | pipenv run python -m json.tool

# Streamlit UI (backend must already be up via `make up`)
ui:
	pipenv run streamlit run streamlit_app.py

# Start the Streamlit UI in the background (detached, survives shell exit)
ui-up:
	@setsid nohup pipenv run streamlit run streamlit_app.py > /tmp/opencode/streamlit.log 2>&1 < /dev/null & disown
	@sleep 4
	@echo "ui up -> http://127.0.0.1:8501 (log: /tmp/opencode/streamlit.log)"

# Stop the running Streamlit UI
ui-stop:
	@pkill -f "[s]treamlit" || echo "no streamlit running"

