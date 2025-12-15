.PHONY: help venv install reinstall clean kira kira-server kira-ui kira-stop \
        train train-nightly train-wumbo-apl train-wumbo-n0 spinner \
        physics-verify apl-run apl-test viz-server tests lint fmt node-deps ci ci-menu smoke kira-health \
        ci-local-python ci-local-python-full ci-act-python ci-act-helix \
        docker-build docker-up docker-down docker-logs

# Local venv paths
VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip
HELIX := $(VENV)/bin/helix
PYTEST := $(VENV)/bin/pytest

help:
	@echo "Rosetta-Helix-Substrate CLI (via make)"
	@echo
	@echo "Setup"
	@echo "  make venv            # Create local .venv"
	@echo "  make install         # Install Python deps into .venv"
	@echo
	@echo "KIRA"
		@echo "  make kira-server     # Start KIRA Flask server (port 5000)"
	@echo "  make kira            # Start wrapper (prints URLs)"
		@echo "  make kira-ui         # Print UI path to open in browser"
		@echo "  make kira-stop       # Kill KIRA server"
	@echo
	@echo "Training"
	@echo "  make train           # Helix engine training (via helix CLI)"
		@echo "  make train-nightly   # Nightly training runner"
		@echo "  make train-wumbo-apl # WUMBO APL integrated run"
		@echo "  make train-wumbo-n0  # WUMBO N0 integrated run"
	@echo
	@echo "Physics / APL / Spinner"
		@echo "  make physics-verify  # Cross-module physics checks"
		@echo "  make apl-run         # Run Quantum APL sample"
		@echo "  make apl-test        # Run Quantum APL tests (Node)"
	@echo "  make spinner         # Nuclear spinner simulation"
	@echo
	@echo "Visualization / Dev"
		@echo "  make viz-server      # Start visualization HTTP server (8765)"
	@echo "  make tests           # Run Python tests"
	@echo "  make lint            # Ruff + flake8 (if present)"
	@echo "  make fmt             # Black format"
		@echo "  make node-deps       # Install Node dev deps for APL tests"
	@echo "  make ci              # Lint, tests, physics verify"
	@echo "  make smoke           # Fast cross-module smoke tests"

venv:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@if [ -f kira-local-system/requirements.txt ]; then \
		$(PIP) install -r kira-local-system/requirements.txt; \
	fi
	$(PIP) install -r requirements.spinner.txt

reinstall:
	rm -rf $(VENV)
	$(MAKE) install

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .coverage htmlcov

# --- KIRA ---
kira:
	$(PY) start_kira.py --no-browser

kira-server:
		cd kira-local-system && $(PY) kira_server.py

kira-ui:
		@echo "Open UI in a browser: docs/kira/index.html"
		@echo "If running server in sub-shell: make kira-server"

kira-stop:
		pkill -f kira_server.py || true

kira-health:
		curl -s http://localhost:5000/api/health | jq .status || true

# --- Training ---
train:
	$(HELIX) train --config configs/full.yaml || $(PY) train_helix.py

train-nightly:
		$(PY) nightly_training_runner.py

train-wumbo-apl:
		$(PY) run_wumbo_apl_integrated.py

train-wumbo-n0:
		$(PY) run_wumbo_n0_integrated.py

# --- Physics / APL / Spinner ---
physics-verify:
		$(PY) verify_physics.py

apl-run:
		$(VENV)/bin/qapl-run || true

apl-test: node-deps
		npm test

spinner:
	$(PY) scripts/nuclear_spinner.py

# --- Visualization / Dev ---
viz-server:
		$(PY) visualization_server.py --port 8765

tests:
	$(PYTEST) -q

lint:
	$(VENV)/bin/ruff check . || true
	$(VENV)/bin/flake8 || true

fmt:
	$(VENV)/bin/black .

node-deps:
		npm install

ci: lint tests physics-verify
	@echo "CI checks completed."

ci-menu:
    $(PY) scripts/ci_menu.py

smoke:
    $(PY) -m pytest -q tests/smoke

# --- Local CI helpers ---
ci-local-python:
	$(PIP) install -e .[viz,dev]
	@if [ -f kira-local-system/requirements.txt ]; then \
		$(PIP) install -r kira-local-system/requirements.txt; \
	fi
	$(PY) -m pytest -q tests/smoke tests/api

ci-local-python-full:
	$(PIP) install -e .[viz,dev]
	@if [ -f kira-local-system/requirements.txt ]; then \
		$(PIP) install -r kira-local-system/requirements.txt; \
	fi
	$(PY) -m pytest -q tests/smoke tests/api
	$(PY) -m pytest -q tests/test_hex_prism.py
	node -v >/dev/null 2>&1 && qapl-run --steps 3 --mode unified --output analyzer_test.json || true
	node -v >/dev/null 2>&1 && qapl-analyze analyzer_test.json || true

ci-act-python:
	@if ! command -v act >/dev/null 2>&1; then echo "act not installed" && exit 1; fi
	act -W .github/workflows/python-tests.yml -j smoke-api-matrix --secret-file .secrets

ci-act-helix:
	@if ! command -v act >/dev/null 2>&1; then echo "act not installed" && exit 1; fi
	act -W .github/workflows/helix-ci.yml -j test --secret-file .secrets

# --- Docker helpers ---
docker-build:
	docker compose build

docker-up:
	docker compose up -d kira viz

docker-down:
	docker compose down

docker-logs:
    docker compose logs -f --tail=200

# --- npm token helper ---
npm-token:
	bash scripts/generate_npm_token.sh
