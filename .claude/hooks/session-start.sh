#!/bin/bash
set -euo pipefail

# Only run on Claude Code remote environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

# Install Python dependencies with dev extras (pytest, ruff, black, mypy)
pip install -e ".[dev]"

# Install Node.js dependencies (ajv for schema validation tests)
npm install
