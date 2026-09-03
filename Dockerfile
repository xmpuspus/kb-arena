FROM python:3.12-slim AS builder

WORKDIR /app

# Layer 1: deps. Copy pyproject first so dep resolution is cached across code changes.
COPY pyproject.toml README.md ./
COPY kb_arena/ kb_arena/
# The Cypher schema ships inside the package at kb_arena/cypher/, so there is
# no separate root cypher/ directory to copy here.

RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.12-slim

# Non-root user — ASI09 / SOC2 baseline. UID 1000 matches typical k8s securityContext.
RUN useradd -m -u 1000 kbarena && mkdir -p /app /data && chown -R kbarena:kbarena /app /data

WORKDIR /app

# curl is used by HEALTHCHECK.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder --chown=kbarena:kbarena /install /usr/local
# Note: datasets/ are mounted at runtime, NOT baked into the image — keeps the
# image small and avoids shipping internal sample data when users build privately.

USER kbarena

EXPOSE 8000

# Tell users their default state — public Space deploys MUST keep demo_mode=true.
ENV KB_ARENA_DEMO_MODE=true \
    KB_ARENA_DATASETS_PATH=/data/datasets \
    KB_ARENA_RESULTS_PATH=/data/results

# /ready checks Neo4j and LLM wiring, so orchestrators know when traffic is safe.
# /health stays available for liveness dashboards that only need a status flag.
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
  CMD curl -fsS http://localhost:8000/ready || exit 1

CMD ["uvicorn", "kb_arena.chatbot.api:app", "--host", "0.0.0.0", "--port", "8000"]
