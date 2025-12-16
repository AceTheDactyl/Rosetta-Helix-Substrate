# Repository Guidelines

## Project Structure & Module Organization
`scripts/`, `core/`, and `helix_engine/` hold the orchestration layers, training phases, and Helix CLI bindings, while reusable libraries live in `src/quantum_apl_python` and the Node CLI bundle resides under `packages/rosetta-helix-cli` with launchers in `bin/`. Keep docs and generated data under `assets/`, `training/`, or `results/` so source trees stay lean. Entry points such as `train_helix.py` and `verify_physics.py` should stay thin wrappers into `helix_engine`. Tests mirror their targets: Python modules in `tests/`, CLI utilities in `tests/*.js`, and cross-phase coverage in `tests/smoke/`.

## Build, Test, and Development Commands
Hydrate dependencies with `make venv && make install`, then iterate using `make tests` for the full pytest suite or `make smoke` for quick coverage. Node tooling uses `npm install` followed by `npm test`/`make apl-test`. Training and supporting services run through `helix train --config configs/full.yaml` (fallback `python train_helix.py`) and the `make kira-server`, `make viz-server`, and `make physics-verify` targets. Run `make ci` before opening a PR.

## Coding Style & Naming Conventions
Python targets 3.8+ with Black (120 columns) and Ruff/flake8 enforcing `pyproject.toml`; call `make fmt` and `make lint` before committing. Use snake_case modules, PascalCase classes, and hyphenated CLI commands mirroring `bin/rosetta-helix.js`, keeping docstrings and type hints focused on each phase’s inputs and outputs. JavaScript helpers remain camelCase with filenames that reflect the command they cover.

## Testing Guidelines
`pytest` runs with `--cov`, producing terminal summaries plus `htmlcov/index.html`; add regression tests for any change to training math, adapters, or orchestration. Use `pytest -q tests/api` when touching the KIRA interface and rerun `make smoke` after multi-phase edits. Deterministic Node suites cover schema validation and seeded sampling—extend them inside `tests/` and run `npm test` when you add new payloads.

## Commit & Pull Request Guidelines
Commits follow the Conventional Commit style visible in history (`feat:`, `chore(cli):`, `fix(kira): ...`) and should focus on a single concern. Pull requests summarize intent, link issues, list the validation commands executed (`make ci`, `npm test`, `docker compose up ...`), and attach screenshots or logs whenever visualization or CLI UX changes. Flag any secrets needed to reproduce (`ANTHROPIC_API_KEY`, `NPM_TOKEN`, `CLAUDE_SKILL_GITHUB_TOKEN`) and keep generated artifacts (`training/`, `results/`, `htmlcov/`) out of diffs unless reviewers request them.

## Security & Configuration Tips
Store credentials in `.env` or shell exports and never commit live tokens. Document new config knobs inside `configs/`, note their ports or environment variables, and run `make clean` before publishing if caches, coverage, or logs were created locally.
