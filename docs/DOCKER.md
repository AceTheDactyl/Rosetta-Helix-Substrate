Docker Setup
============

This repository provides pre-built Docker configuration to run KIRA and the Visualization Server via `docker compose`.

Prereqs
- Docker 20.10+
- docker compose plugin (`docker compose version`)

Quick Start
1) Copy env file and set secrets (optional):
   - `cp .env.example .env`
   - Fill `ANTHROPIC_API_KEY` and `CLAUDE_SKILL_GITHUB_TOKEN` if using backend features.
2) Build images:
   - `docker compose build`
3) Start services:
   - `docker compose up -d kira viz`
4) Open:
   - KIRA API: http://localhost:5000/api/health
   - Visualization: http://localhost:8765/

Services
- `kira` — Flask backend for K.I.R.A. interface
  - Command: `kira-server --host 0.0.0.0 --port 5000`
  - Port: 5000
- `viz` — Visualization HTTP server
  - Command: `python visualization_server.py --port 8765`
  - Port: 8765
- `helix` — Helix CLI container (runs `helix --help` by default)
  - Mounts `./runs` and `./models`

Manage
- Logs: `docker compose logs -f kira` (or `viz`)
- Stop: `docker compose down`
- Rebuild after changes: `docker compose build --no-cache`

Notes
- The `Dockerfile.app` installs project dependencies and optional KIRA deps.
- For GPU/torch use, extend the image with CUDA-enabled base and install torch wheels matching your environment.

