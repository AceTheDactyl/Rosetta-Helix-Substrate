# Rosetta-Helix Application Image (KIRA + Viz + Helix)
# ====================================================

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
  && rm -rf /var/lib/apt/lists/*

# Copy repository
COPY . /app

# Install Python deps
RUN python -m pip install --upgrade pip setuptools wheel \
 && if [ -f requirements.txt ]; then pip install -r requirements.txt; fi \
 && pip install -e . \
 && if [ -f kira-local-system/requirements.txt ]; then pip install -r kira-local-system/requirements.txt; fi \
 && if [ -f requirements.spinner.txt ]; then pip install -r requirements.spinner.txt || true; fi

# Default command prints help; actual commands set by docker-compose
CMD ["python", "-c", "print('Rosetta-Helix container ready. Use docker-compose services.')"]

