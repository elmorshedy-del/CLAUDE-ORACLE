# Dockerfile — alternative to Nixpacks.
# Railway will use this if you set "Builder: Dockerfile" in service settings.
# Also useful for local `docker run` testing before deploy.

FROM python:3.11-slim

# System deps for scipy/numpy wheels (blas/lapack); gcc for any fallbacks.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    JSON_LOGS=1

WORKDIR /app

# Install deps first (better Docker-layer caching).
COPY pyproject.toml README.md ./
# Copy package source so editable install picks it up.
COPY poly_paper ./poly_paper
COPY scripts ./scripts
COPY tests ./tests

RUN pip install -e .

# Railway provides PORT at runtime; default 8080 for local.
ENV PORT=8080
EXPOSE 8080

# No HEALTHCHECK directive — Railway does its own via /healthz.

CMD ["python", "-m", "scripts.run_paper"]
