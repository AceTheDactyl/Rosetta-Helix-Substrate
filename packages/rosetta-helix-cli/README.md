# rosetta-helix-cli

Public npm CLI wrapper that orchestrates the Rosetta‑Helix‑Substrate repository from Node.

Usage
```
npx rosetta-helix setup       # create .venv + install deps
npx rosetta-helix kira        # start KIRA server on 5000
npx rosetta-helix viz         # start Visualization server on 8765
npx rosetta-helix helix:train # run helix training
npx rosetta-helix smoke       # run smoke tests
npx rosetta-helix api:test    # run API contract tests
```

Notes
- Run commands from within a Rosetta‑Helix‑Substrate checkout — the CLI operates against the current working directory.
- The CLI sets up a local Python virtual environment (`.venv`) and installs repository requirements.
- For Docker helpers: `docker:build|up|down|logs` commands delegate to `docker compose` if available.

Publishing
- `npm publish --access public` (requires NPM_TOKEN configured in your environment or CI).

