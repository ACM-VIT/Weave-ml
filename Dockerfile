# Keep the human-readable Python line and immutable manifest digest together.
# Renovation must update both stages in one reviewed change.
FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de AS builder

WORKDIR /app

ARG UV_VERSION=0.11.29
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARG EMBEDDING_MODEL_REVISION=c9745ed1d9f207416be6d2e6f8de32d1f16199bf
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}
ENV EMBEDDING_MODEL_REVISION=${EMBEDDING_MODEL_REVISION}
ENV HF_HOME=/app/.cache/huggingface

# Pin the build tool so a new upstream uv release cannot change an otherwise
# immutable build. Runtime dependencies remain locked by uv.lock.
RUN pip install --no-cache-dir "uv==${UV_VERSION}"

# Copy dependency configuration
COPY pyproject.toml uv.lock ./

# Install dependencies using uv (creates .venv)
RUN UV_PROJECT_ENVIRONMENT=.venv uv sync --locked --no-dev

# Keep inference startup deterministic and independent of runtime network
# access by baking the configured sentence-transformers model into the image.
RUN .venv/bin/python -c "import os; from sentence_transformers import SentenceTransformer; SentenceTransformer(os.environ['EMBEDDING_MODEL'], revision=os.environ['EMBEDDING_MODEL_REVISION'])"

# Runner stage
FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de AS runner

WORKDIR /app

ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARG EMBEDDING_MODEL_REVISION=c9745ed1d9f207416be6d2e6f8de32d1f16199bf
ARG ML_RELEASE_ID=development
LABEL org.opencontainers.image.revision=${ML_RELEASE_ID}
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}
ENV REPOSITORY_EMBEDDING_MODEL=${EMBEDDING_MODEL}
ENV EMBEDDING_MODEL_REVISION=${EMBEDDING_MODEL_REVISION}
ENV BAKED_EMBEDDING_MODEL=${EMBEDDING_MODEL}
ENV BAKED_EMBEDDING_MODEL_REVISION=${EMBEDDING_MODEL_REVISION}
ENV ML_RELEASE_ID=${ML_RELEASE_ID}
ENV BAKED_ML_RELEASE_ID=${ML_RELEASE_ID}
ENV HOME=/home/gh-social
ENV HF_HOME=/app/.cache/huggingface
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Conservative defaults keep one embedding job from oversubscribing a small
# production host. Deployments may raise the application-level concurrency only
# after measuring recommendation latency and memory use.
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false
ENV MALLOC_ARENA_MAX=2
ENV EMBEDDING_CPU_THREADS=1
ENV EMBEDDING_MAX_CONCURRENCY=1
ENV EMBEDDING_EXECUTOR_WORKERS=1
ENV EMBEDDING_MAX_OUTSTANDING_JOBS=4
ENV V2_RECOMMENDATION_TIMEOUT_SECONDS=12
ENV V2_HEALTH_TIMEOUT_SECONDS=5
ENV V2_HEALTH_REFRESH_SECONDS=15
ENV V2_HEALTH_MAX_AGE_SECONDS=45
ENV V2_RECOMMENDATION_EXECUTOR_WORKERS=4
ENV V2_RECOMMENDATION_MAX_OUTSTANDING=8
ENV V2_FEEDBACK_EXECUTOR_WORKERS=2
ENV V2_FEEDBACK_MAX_OUTSTANDING=8
ENV V2_FEEDBACK_TIMEOUT_SECONDS=8
ENV V2_REFRESH_EXECUTOR_WORKERS=2
ENV V2_REFRESH_MAX_OUTSTANDING=4
ENV V2_REFRESH_TIMEOUT_SECONDS=45
ENV V2_HEALTH_EXECUTOR_WORKERS=2
ENV V2_HEALTH_MAX_OUTSTANDING=4
ENV EMBEDDING_WARMUP_ON_STARTUP=true
ENV INFERENCE_WARMUP_TIMEOUT_SECONDS=120

# Create the runtime identity before copying layers so ownership changes do not
# duplicate the baked model whenever application source changes.
RUN addgroup --system gh-social \
    && adduser --system --ingroup gh-social --home /home/gh-social gh-social \
    && mkdir -p /home/gh-social /app/.cache \
    && chown -R gh-social:gh-social /home/gh-social /app/.cache

# Keep the large, stable dependency and model layers independent from source
# code. A normal application-only release can then reuse both layers instead
# of re-uploading a monolithic /app layer.
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder --chown=gh-social:gh-social /app/.cache /app/.cache
COPY . .

USER gh-social

# Update PATH to prioritize virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

EXPOSE 8000

# The check reads the secret from the running container environment; no secret
# is embedded in an image layer or command. The feedback systemd unit disables
# this API-specific image healthcheck.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD ["python", "-m", "scripts.production_smoke", "--readiness-only", "--timeout", "8"]

# Start the ML API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
