#!/usr/bin/env bash
set -euo pipefail

echo "== Rosetta-Helix: Generate npm Token =="

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is not installed or not on PATH." >&2
  exit 1
fi

echo "This script will create an npm auth token using 'npm token create'."
echo "Prerequisites: you are already logged in (npm login), and have your OTP ready if 2FA is enabled."
echo

TOKEN=""

# Try JSON mode first (non-interactive if session is valid and no 2FA prompt)
echo "Attempting non-interactive creation (JSON mode)..."
if OUT=$(npm token create --json 2>/dev/null); then
  # Parse JSON {token: "..."} or an array of objects
  TOKEN=$(printf '%s' "$OUT" | node -e '
    let s="";process.stdin.on("data",d=>s+=d).on("end",()=>{
      try{
        let j=JSON.parse(s);
        if(Array.isArray(j)) j=j[0];
        if(j && j.token){ console.log(j.token); return; }
      }catch(e){}
      process.exit(1);
    })' 2>/dev/null || true)
fi

if [[ -z "$TOKEN" ]]; then
  echo "Falling back to interactive mode. You may be prompted for OTP if 2FA is enabled."
  echo "Note: When the token is printed, copy it."
  npm token create || true
  echo
  read -r -p "Paste the new npm token to save it locally: " TOKEN
fi

if [[ -z "${TOKEN:-}" ]]; then
  echo "No token captured. Aborting." >&2
  exit 1
fi

OUTFILE=.npm_token.txt
printf '%s' "$TOKEN" > "$OUTFILE"
chmod 600 "$OUTFILE" || true
echo "Saved token to $OUTFILE"
echo "Upload this token to GitHub repository secrets as NPM_TOKEN."

