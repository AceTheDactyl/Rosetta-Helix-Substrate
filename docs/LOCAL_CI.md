Local CI (Run GitHub Actions Locally)
====================================

This repo supports running the main GitHub Actions workflows locally via `act`.

Prereqs
- Docker installed and running
- `act` installed: https://github.com/nektos/act

Setup
1) Provide a secrets file (optional, for backend flows):
   - Copy `.secrets.example` to `.secrets` and fill in values.
2) Ensure platform mapping is set (provided in `.actrc`).

Examples
- Run smoke/API matrix (default job selection may run all matrix entries):
  - `act -W .github/workflows/python-tests.yml -j smoke-api-matrix --secret-file .secrets`
- Run the main pytest job:
  - `act -W .github/workflows/python-tests.yml -j pytest --secret-file .secrets`
- Run Helix CI unit tests (includes API contract tests):
  - `act -W .github/workflows/helix-ci.yml -j test --secret-file .secrets`

Generate npm token for publishing
- Use the helper to create a token and save it to a file:
  - `make npm-token` or `npm run token`
- The token will be saved to `.npm_token.txt`. Copy its contents and add it to your repository secrets as `NPM_TOKEN`.
- If you have 2FA enabled on npm, have your OTP ready — the script supports interactive fallback.

Notes
- The workflows install system packages via `apt-get` and should run under the `catthehacker` act images configured in `.actrc`.
- If you see network restrictions, use `--container-architecture linux/amd64` as needed or set `ACT_DEFAULT_PLATFORM`.
