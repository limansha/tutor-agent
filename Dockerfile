FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/models \
    HOST=0.0.0.0 \
    PORT=5000 \
    QUESTIONS_DIR=/books

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# ponytail: CPU-only torch. requirements.txt pins the CUDA wheels (nvidia-*, cuda-*, triton ~6GB),
# which a CPU EC2 box never uses. Swap in a GPU index + runner if you need GPU embeddings.
RUN pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
    && grep -vE '^(nvidia-|cuda-|triton)' requirements.txt > /tmp/req.txt \
    && pip install -r /tmp/req.txt gunicorn

COPY run.py streamlit_app.py ./
COPY app ./app
COPY prompts ./prompts
COPY sql ./sql
COPY scripts ./scripts

# Bake the embedding model into the image so runtime needs no HuggingFace egress.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
    && useradd --create-home --uid 1000 tutor \
    && chown -R tutor /app /opt/models

USER tutor
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:5000/api/health', timeout=4)" || exit 1

# ponytail: 1 worker x 4 threads: the ~90MB embedding model is loaded per worker. Raise --workers
# and add RAM per worker, or move embeddings to a separate service, when traffic justifies it.
CMD ["gunicorn", "-k", "gthread", "-w", "1", "--threads", "4", "-b", "0.0.0.0:5000", "run:app"]
