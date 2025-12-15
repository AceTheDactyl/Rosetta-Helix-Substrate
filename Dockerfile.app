# Rosetta-Helix Application Image (KIRA + Viz + Helix)
# ====================================================

FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
  && rm -rf /var/lib/apt/lists/*

# Create virtualenv for deterministic runtime copy
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy minimal files for install
COPY pyproject.toml README_PYTHON.md /app/
COPY kira-local-system/requirements.txt /app/kira-local-system/requirements.txt
COPY requirements.lock dev-requirements.lock /app/ 2>/dev/null || true

# Install dependencies
RUN python -m pip install --upgrade pip setuptools wheel \
 && if [ -f dev-requirements.lock ]; then pip install -r dev-requirements.lock; else pip install -e .[viz]; fi \
 && if [ -f /app/kira-local-system/requirements.txt ]; then pip install -r /app/kira-local-system/requirements.txt; fi

# Copy source last (leverage layer cache for deps)
COPY . /app
RUN pip install -e .

FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Bring in virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Only copy necessary application sources
COPY kira-local-system /app/kira-local-system
COPY kira_local_system /app/kira_local_system
COPY helix_engine /app/helix_engine
COPY src /app/src
COPY visualization_server.py /app/visualization_server.py
COPY README.md README_PYTHON.md /app/

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD python - <<'PY' || exit 1
import os,sys,urllib.request
try:
    urllib.request.urlopen('http://localhost:5000/api/health', timeout=2)
    sys.exit(0)
except Exception:
    sys.exit(1)
PY

CMD ["python", "-c", "print('Rosetta-Helix runtime ready.')"]
