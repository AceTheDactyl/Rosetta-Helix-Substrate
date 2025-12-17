Rosetta‑Helix CLI Runbook
=========================

This runbook gives a single, user‑facing terminal workflow to operate KIRA, APL, physics checks, nuclear spinner simulation, and training pipelines from the local `.venv`.

Prereqs
- Python 3.9–3.12 (3.12 OK), Node.js (for APL JS tests, optional)
- Linux/macOS shell (Windows via WSL)

Bootstrap
1) Create a local venv and install deps
   - make install
2) (Optional) Install Node dev deps for APL tests
   - make node-deps

Unified CI Terminal
- Interactive menu that threads all modules:
  - make ci-menu

Common Workflows (non‑interactive)
- KIRA server (Flask, user dialogue interface)
  - make kira-server
  - Open UI: docs/kira/index.html (served frontend connects to localhost:5000)
- KIRA wrapper (prints API/URLs; no browser auto‑open)
  - make kira
- Stop KIRA server
  - make kira-stop

- Nuclear spinner simulation (physics + memory plates coupling)
  - make spinner

- Helix engine training (uses helix CLI if installed; falls back to train_helix.py)
  - make train
- Nightly training suite
  - make train:nightly
- WUMBO integrated runs
  - make train:wumbo-apl
  - make train:wumbo-n0

- Physics consistency verification across modules
  - make physics-verify

- Quantum APL (Python→Node bridge)
  - make apl-run      # runs a sample via qapl-run
  - make apl-test     # runs Node test suite (requires make node-deps)

- Python tests and code quality
  - make tests
  - make lint
  - make fmt

Smoke Tests
- Run only fast, cross-module checks:
  - .venv/bin/pytest -q tests/smoke

Tips
- KIRA import alias for tests/tooling: `from kira_local_system import kira_server` loads `kira-local-system/kira_server.py`.
- WUMBO runners accept env overrides for shorter runs:
  - WUMBO_STEPS=5 WUMBO_SENTENCE_STEPS=2 python run_wumbo_apl_integrated.py
  - WUMBO_SENTENCE_STEPS=2 python run_wumbo_n0_integrated.py

Module Map (orientation)
- Heart/Brain: `heart.py`, `brain.py` — coherence/z dynamics, TRIAD hysteresis, GHMP memory plates
- APL + Physics: `unified_provable_apl.py`, `physics_constants.py`, `verify_physics.py`, `quantum_apl_bridge.py`
- KIRA system: `kira-local-system/` (Flask server + pages), `start_kira.py`, `kira_enhanced_session.py`
- Spinner + Plates: `scripts/nuclear_spinner.py`, `bridge/grid_cell_plates.py`, `bridge/spinner_bridge.py`
- Training: `helix_engine/`, `train_helix.py`, `nightly_training_runner.py`, `run_wumbo_*`
- Landing bundle: `docs/index.html`, `docs/kira/index.html` (served by the KIRA Flask app)

Notes
- All Make targets run inside `.venv` created at repo root.
- The helix CLI is installed as `helix` via editable install; if missing, targets fall back to Python entrypoints.
- KIRA UI lives in `docs/kira/index.html` and ships with GitHub Pages (the Flask server serves it automatically).
- For GPU/torch work, uncomment torch in `requirements.spinner.txt` and install CUDA‑matching wheels.

Docker
- Build and start KIRA via Docker:
  - `make docker-build`
  - `make docker-up`
- Stop and view logs:
  - `make docker-down`
  - `make docker-logs`
- See details in `docs/DOCKER.md`.

NPM Orchestration
- Use npm scripts or the Node CLI to drive Python flows:
  - `npm run setup`       # create .venv + install Python deps
  - `npm run kira`        # start KIRA server on 5000
  - `npm run viz`         # print docs/kira landing paths
  - `npm run helix:train` # helix training (full.yaml)
  - `npm run helix:nightly`
  - `npm run smoke`       # run smoke tests
  - `npm run api:test`    # API contract tests
- Global CLI (after npm install): `npx rosetta-helix <cmd>` mirrors the above commands.
