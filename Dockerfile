FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/klei30/tuner-mcp" \
      org.opencontainers.image.description="Agent-native MCP control plane for Tinker post-training" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GIT_TERMINAL_PROMPT=0 \
    TUNER_TRANSPORT=http \
    TUNER_HOST=0.0.0.0 \
    TUNER_PORT=8000 \
    TUNER_STATE_DIR=/var/lib/tuner

# HTTP startup fails closed unless TUNER_AUTH_TOKEN is supplied at runtime.

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv==0.5.18
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
# Cookbook brings CPU PyTorch for local renderers, tensor preparation, and
# recipe-specific losses. Tinker continues to execute GPU work remotely.
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_HTTP_TIMEOUT=120 UV_HTTP_RETRIES=5 \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

RUN useradd --create-home --uid 10001 tuner \
    && install -d -o tuner -g tuner /var/lib/tuner /home/tuner/.cache/harbor \
       /home/tuner/.cache/huggingface \
       /app/.venv/lib/python3.11/site-packages/ifbench/.nltk_data
USER tuner

EXPOSE 8000
CMD ["/app/.venv/bin/tuner", "--transport", "http", "--host", "0.0.0.0", "--port", "8000"]
