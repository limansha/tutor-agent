# tutor-agent

A retrieval-augmented **study agent**. It ingests a folder of source-document PDFs into
Postgres + pgvector, then answers questions against that corpus — either by retrieving the
relevant passages, or by running an agent that reasons over them to pick and explain a
multiple-choice answer.

The source corpus is **not** in this repo. Point `BOOKS_DIR` /
`QUESTIONS_DIR` at your own copy.

---

## How it works

```
PDFs ──► parse ──► chunk ──► embed ──► pgvector
         │          │                     │
     structure   parent +              cosine
      markers    child chunks            │
         │          │                    ▼
         └──────► metadata         query embed
                  (section,         + filters
                   page, doc)             │
                                          ▼
                              child hits ──► parent expansion ──► LLM ──► answer + sources
```

**1. Parse.** `pypdf` layout extraction, then the text is split into blocks at structural
markers it recognises (reading/module/LO headings, plus section headings like exam focus,
key concepts, formulas). Front matter — welcome, copyright, TOC — is dropped.

**2. Chunk.** A **parent-child** scheme, so context is never lost to a small embedding window:

| | what it is | embedded? |
|---|---|---|
| **parent** | a whole module/reading section, normalized to a token budget | no (`NULL`) |
| **child** | sliding sentence windows within that section | yes |

Only children are embedded and indexed (HNSW is partial, `WHERE role='child'`). At query
time we match children on cosine similarity and then fetch their **parents** for the real
text — so the LLM sees the full section, not a 400-token fragment. Metadata filters
(section, module, reading, year…) narrow the search; `section_expansion` pulls in every
sibling chunk of the matched module.

**3. Embed.** `sentence-transformers/all-MiniLM-L6-v2` → 384-dim vectors, cosine distance
(`<=>`). The dimension is baked into `sql/schema.sql`, so changing `EMBEDDING_DIM` means
changing the schema too.

**4. Answer.** Two modes:
- **Retrieve** — pure RAG, no LLM needed. Returns ranked chunks with scores and source links.
- **Agentic MCQ** — a LangChain agent calls retrieval as a *tool*, then returns the selected
  option, an explanation, and why the other options were rejected.

There is also a separate **question bank** path: PDFs are parsed, questions are extracted via
LLM into JSON, and loaded into `frm_questions` with a topic index. `POST /api/questions`
returns practice questions for given topics — exact topic match first, semantic fallback over
topic embeddings.

---

## Quickstart

Requires Python 3.12, [pipenv](https://pipenv.pypa.io/), and a running Postgres with the
[pgvector](https://github.com/pgvector/pgvector) extension.

```bash
make env        # copy .env.example -> .env
$EDITOR .env    # set DATABASE_URL, OPENROUTER_API_KEY, BOOKS_DIR
make deps       # pipenv install
make db-setup   # create schema + HNSW index
make ingest     # parse -> chunk -> embed -> store  (slow first run: downloads the model)
make run        # http://127.0.0.1:5000
```

Then, against a running server:

```bash
make health                                        # readiness
make retrieve QUERY="what is duration risk?"
make mcq-answer QUESTION="..." OPTIONS='{"A":"...","B":"..."}'
make questions TOPICS='["Credit Risk","VaR"]' PAGE_SIZE=10
```

A Streamlit UI is included: run `make run` (backend, in another terminal), then `make ui`
→ http://127.0.0.1:8501.

Re-ingesting: `make reset && make ingest`, or `make ingest FORCE_REINGEST=--force`.

---

## Configuration

Everything is env-driven; see `.env.example` for the full annotated list. The ones that
matter most:

| var | purpose |
|---|---|
| `DATABASE_URL` | Postgres + pgvector connection string (**required**) |
| `OPENROUTER_API_KEY` | LLM access — required for `/api/mcq-answer` and question extraction; `/api/retrieve` works without it |
| `OPENROUTER_MODEL` / `_FALLBACK_MODELS` | primary model, then a comma-separated fallback list tried in order if it errors |
| `BOOKS_DIR` / `QUESTIONS_DIR` | where the source PDFs live (**not** in the repo) |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | must match `vector(384)` in the schema |
| `MIN_CHUNK_TOKENS` / `PARENT_MAX_TOKENS` / `CHILD_*` | chunking budgets |
| `TOPIC_MATCH_THRESHOLD` | minimum cosine similarity for the semantic topic fallback |

`HOST` / `PORT` / `DEBUG` control the Flask server; `API_BASE_URL` points the Streamlit UI
at the backend.

---

## API

| endpoint | what it does |
|---|---|
| `POST /api/retrieve` | `{query, filters?, top_k?, parent_expansion?, section_expansion?}` → ranked chunks with scores, metadata and source links |
| `POST /api/mcq-answer` | `{question, options, filters?, top_k?}` → `{selected_option, explanation, rejections[]}` + `sources[]` |
| `POST /api/questions` | `{topics[], page_size?}` → practice questions, plus which topics matched exactly vs semantically and which had no match |
| `GET /docs/<file>` | serves a source PDF (`doc_url` deep-links the exact page) |
| `GET /api/health` | readiness check |

DB-backed routes return **503 with a hint** until `make db-setup && make ingest` have been run — they are real code paths, not stubs.

---

## Layout

```
run.py                  entrypoint (app factory -> flask)
streamlit_app.py        Streamlit UI
app/
  config.py             env config
  db.py                 connect / readiness / schema init
  embeddings.py         embedding model singleton
  llm.py                OpenRouter client with model fallback
  agent.py              LangChain agent: retrieval as a tool
  prompts.py            loads prompts/*.j2
  tokenize.py           token counting
  chat/routes.py        /api/mcq-answer
  ingestion/
    parser.py           PDF -> blocks (structure markers)
    chunker.py          parent-child + sliding window
    books_pipeline.py   parse -> chunk -> embed -> store
    question_ingestion.py  PDF -> LLM -> questions JSON -> store
  retrieval/
    vectorstore.py      pgvector search, filters, parent expansion
    service.py          embed query -> children -> parents
    routes.py           /api/retrieve, /docs/<file>, /api/health
  questions/routes.py   /api/questions
prompts/                Jinja2 prompt templates
sql/schema.sql          pgvector tables + indexes
scripts/                CLI wrappers behind the make targets
```

Tables: `frm_books` (chunks, self-referencing `parent_id`), `frm_questions`,
`frm_question_topics` (topic embeddings).

---

## Notes

- No tests, no CI, no lint config yet.
- The makefile targets do real DB and network work — read before running.
- `.env`, `logs/`, `questions_dump/` and source PDFs are gitignored. Generated JSON dumps and
  logs are build artifacts, not source.
- First ingest downloads the embedding model and is slow.
