# Helix Training Engine Docker Image
# ===================================
# Multi-stage build for minimal production image

# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY pyproject.toml setup.py* ./
COPY src/ ./src/
COPY helix_engine/ ./helix_engine/
COPY training/ ./training/
COPY core/ ./core/
COPY configs/ ./configs/

# Install package
RUN pip install --no-cache-dir --user -e .

# Stage 2: Runtime
FROM python:3.11-slim as runtime

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app /app

# Make sure scripts are in PATH
ENV PATH=/root/.local/bin:$PATH

# Create directories for runs and models
RUN mkdir -p /app/runs /app/models

# Set default environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command: show help
CMD ["helix", "--help"]

# Example usage:
# docker build -t helix-engine:latest .
# docker run helix-engine:latest helix train --config configs/smoke.yaml
# docker run -v $(pwd)/runs:/app/runs helix-engine:latest helix train --config configs/full.yaml
