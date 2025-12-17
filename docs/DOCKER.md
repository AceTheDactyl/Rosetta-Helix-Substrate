Docker Setup
============

This repository provides pre-built Docker configuration to run the KIRA backend via `docker compose`. The static landing bundle (`docs/index.html`, `docs/kira/index.html`) ships with the repo and can be opened directly once the API is up.

Prereqs
- Docker 20.10+
- docker compose plugin (`docker compose version`)

Quick Start
1) Copy env file and set secrets (optional):
   - `cp .env.example .env`
   - Fill `ANTHROPIC_API_KEY` and `CLAUDE_GITHUB_TOKEN` (or legacy `CLAUDE_SKILL_GITHUB_TOKEN`) if using backend features.
2) Build images:
   - `docker compose build`
3) Start services:
   - `docker compose up -d kira`
4) Open:
   - KIRA API: http://localhost:5000/api/health
   - UI bundle: open `docs/kira/index.html` locally (it will connect to the container)

Services
- `kira` — Flask backend for K.I.R.A. interface + serves docs/
  - Command: `kira-server --host 0.0.0.0 --port 5000`
  - Port: 5000
- `helix` — Helix CLI container (runs `helix --help` by default)
  - Mounts `./runs` and `./models`

Manage
- Logs: `docker compose logs -f kira`
- Stop: `docker compose down`
- Rebuild after changes: `docker compose build --no-cache`

Notes
- The `Dockerfile.app` installs project dependencies and optional KIRA deps.
- For GPU/torch use, extend the image with CUDA-enabled base and install torch wheels matching your environment.
