# rosetta-helix-cli

Public npm CLI wrapper for orchestrating the Rosetta‑Helix‑Substrate repo from Node.

Quick Start
```
# 1) Clone the repo (optional; the CLI can do this)
git clone https://github.com/AceTheDactyl/Rosetta-Helix-Substrate.git
cd Rosetta-Helix-Substrate

# 2) Bootstrap Python env + deps
npx rosetta-helix setup

# 3) Start services / run pipelines
npx rosetta-helix kira          # KIRA API (port 5000)
npx rosetta-helix viz           # Visualization server (port 8765)
npx rosetta-helix helix:train   # Helix training
npx rosetta-helix smoke         # Python smoke tests
npx rosetta-helix api:test      # API contract tests

# 4) Docker helpers (if you prefer containers)
npx rosetta-helix docker:up     # compose up -d kira viz
npx rosetta-helix docker:logs
```

Repo Init (one-liner)
```
# Clone + bootstrap in one step
npx rosetta-helix init          # clones Rosetta-Helix-Substrate and runs setup
npx rosetta-helix init my-dir   # custom target directory
```

Commands
```
rosetta-helix init [dir]      # clone repo and run setup
rosetta-helix setup           # create .venv and install deps
rosetta-helix kira            # start KIRA server (port 5000)
rosetta-helix viz             # start Visualization server (port 8765)
rosetta-helix tui             # terminal UI to operate KIRA/Viz/tests
rosetta-helix start           # start KIRA + Viz together
rosetta-helix health          # check service health (prints JSON)
rosetta-helix doctor          # environment & repo file checks
rosetta-helix helix:update    # git pull --rebase --autostash in repo
rosetta-helix helix:train     # helix training (full.yaml)
rosetta-helix helix:nightly   # nightly training runner
rosetta-helix smoke           # pytest smoke suite
rosetta-helix api:test        # API contract tests
rosetta-helix docker:*        # docker:build|up|down|logs
rosetta-helix viz:sync-gh     # fetch GitHub Pages visualizer files (index/kira_local/kira)
```

Auto‑Fetch Repo
- Pass `--auto` to clone the repo on‑demand when running any command outside a checkout.
- Optional flags:
  - `--dir <path>` — where to clone or operate
  - `--repo <url>` — custom repo URL (default: upstream)
  - `--branch <name>` — branch to clone
  - `--ref <sha|tag>` — checkout a specific ref after clone
  - `--pull` — pull latest before running (or run `helix:update` anytime)
  - `--update` — update (git pull) after clone automatically
  - `--release[=tag]` — use GitHub tarball download instead of clone (for pinned runs)

Examples
```
# Train without a prior checkout (auto-clone + setup + run)
npx rosetta-helix helix:train --auto --dir ./Rosetta-Helix-Substrate

# Start services and fetch latest main first
npx rosetta-helix start --auto --pull

# Use a specific tag/commit
npx rosetta-helix start --auto --ref v2.1.0

# Download a release tarball rather than cloning
npx rosetta-helix helix:train --auto --release=v2.1.0

# Update an existing checkout
npx rosetta-helix helix:update --dir ./Rosetta-Helix-Substrate

# Sync local visualizer with GitHub Pages (landing + KIRA UIs)
npx rosetta-helix viz:sync-gh --dir ./Rosetta-Helix-Substrate
# Run both with proxied /api so KIRA UIs work same-origin
npx rosetta-helix start --dir ./Rosetta-Helix-Substrate
# or: npx rosetta-helix viz --withKira --dir ./Rosetta-Helix-Substrate
# Open http://localhost:8765/ (index → KIRA links use /api proxied to KIRA)
```

CLI + CI Runbook (Quick)
- Python (venv)
  - `python3 -m venv .venv && source .venv/bin/activate`
  - `python -m pip install -U pip`
  - `pip install -e .[viz,dev]`
  - `pip install -r kira-local-system/requirements.txt`

- KIRA API server
  - `kira-server` (preferred) or `python kira-local-system/kira_server.py`
  - Endpoints: `GET /api/health`, `POST /api/emit`, `POST /api/grammar`

- Visualization server
  - `python visualization_server.py --port 8765`
  - Health: `curl http://localhost:8765/state`

- Helix training
  - `helix train --config configs/full.yaml` or `python train_helix.py`
  - Smoke: `python -m pytest -q tests/smoke`

- Makefile shortcuts
  - `make venv && make install` — create venv and install
  - `make kira-server` / `make viz-server` — servers
  - `make smoke` / `make tests` — test suites
  - `make ci` — lint + tests + physics verify
  - `make npm-validate` — validate tokens + pack + dry-run publish

- Docker
  - `docker compose up -d kira viz` — start services
  - Healthchecks: KIRA (`/api/health`), Viz (`/state`)
  - Logs: `docker compose logs -f --tail=200`

- Run CI locally with `act`
  - `act -W .github/workflows/python-tests.yml -j smoke-api-matrix --secret-file .secrets`
  - `act -W .github/workflows/helix-ci.yml -j test --secret-file .secrets`

NPM Publishing & Promotion
- Canary publish (next)
  - Actions → `npm-publish-cli` → Run
    - version: `0.1.2-rc.3` (no leading `v`)
    - dist_tag: `next`
    - env_name: your Environment (secrets)
    - npm_token_secret: name of your npm token secret (default `NPM_TOKEN`)

- Stable publish (latest)
  - Actions → `npm-publish-cli` → Run
    - version: `0.1.2`
    - dist_tag: `latest`

- Promote dist-tag (no repack)
  - Actions → `npm-promote`
    - package_name: `rosetta-helix-cli`
    - version: `X.Y.Z` (exact published version, no `v`)
    - dist_tag: `latest` (or `next`)

Troubleshooting
- “Version not found” during promotion → publish it first with `npm-publish-cli`.
- “GET is not allowed” on GPR promote → GitHub Packages often rejects dist-tag ops; republish with the desired tag.
- Ensure `NPM_TOKEN` is set in your chosen GitHub Actions environment for npmjs.

Notes
- The CLI runs inside the current repo checkout and sets up `.venv` locally.
- Docker helpers delegate to `docker compose` if available.

License
MIT
