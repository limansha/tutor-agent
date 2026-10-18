FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EMBEDDING_CACHE_DIR=/opt/models \
    HOST=0.0.0.0 \
    PORT=5000 \
    QUESTIONS_DIR=/books

WORKDIR /app

# libgomp1 is onnxruntime's OpenMP runtime. Embeddings run on ONNX Runtime via
# fastembed, so there is no torch/transformers in the image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt gunicorn

COPY run.py streamlit_app.py ./
COPY app ./app
COPY prompts ./prompts
COPY sql ./sql
COPY scripts ./scripts

# Bake the ONNX embedding model and its tokenizer into the image, so runtime
# needs no HuggingFace egress. fastembed stores both under EMBEDDING_CACHE_DIR
# (/opt/models); count_tokens reuses that same model tokenizer for chunk budgets.
# This RUN must stay *above* the HF_HUB_OFFLINE=1 below, or the bake itself would
# run offline and fail. The assert is model-agnostic: a subword tokenizer always
# splits a long word into >1 token, so a missing tokenizer cache fails the build
# instead of silently shrinking chunk budgets at ingest time.
RUN python -c "from app.embeddings import count_tokens; assert count_tokens('internationalization') > 1" \
    && useradd --create-home --uid 1000 tutor \
    && chown -R tutor /app /opt/models

# Runtime: resolve everything from the baked caches only, never the hub.
ENV HF_HUB_OFFLINE=1

USER tutor
EXPOSE 5000

# $PORT: Render injects PORT=10000 and waits on it; locally it falls back to 5000.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os, urllib.request as u; p=os.environ.get('PORT','5000'); h='/_stcore/health' if os.environ.get('RUN_UI')=='1' else '/api/health'; u.urlopen('http://127.0.0.1:%s%s' % (p, h), timeout=4)" || exit 1

# ponytail: one image, two roles — RUN_UI=1 runs Streamlit, else gunicorn serves the
# API. (Compose overrides `command:` per service, so this only matters for Render /
# bare `docker run`.) 1 worker x 4 threads: the ~90MB embedding model is loaded
# lazily on the first embed, never at boot — the hosted web-only MCQ path never
# embeds, so on Render it stays unloaded and foots no RAM. --timeout 120 gives
# slow requests (first embed, web-search MCQ) room so the worker isn't killed into
# a 502. --access-logfile - prints one line per request (method, path, status) to
# stdout so the Render Logs tab shows traffic at all. Raise --workers and add RAM
# per worker, or move embeddings to a separate service, when traffic justifies it.
CMD sh -c 'if [ "$RUN_UI" = "1" ]; then exec streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port "${PORT:-8501}" --server.headless true; else exec gunicorn -k gthread -w 1 --threads 4 --timeout 120 --access-logfile - -b 0.0.0.0:${PORT:-5000} run:app; fi'
