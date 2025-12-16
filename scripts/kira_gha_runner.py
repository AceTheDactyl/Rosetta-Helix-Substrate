import os
import json
from pathlib import Path
from datetime import datetime, timezone


def main() -> int:
    message = os.environ.get("ACTION_MESSAGE", "").strip()
    mode = os.environ.get("ACTION_MODE", "chat").strip() or "chat"
    state_z = float(os.environ.get("STATE_Z", "0.5") or 0.5)
    state_phase = os.environ.get("STATE_PHASE", "PARADOX")
    callback_id = os.environ.get("CALLBACK_ID", "").strip()

    # Minimal execution: prepare directories and write a response JSON
    outdir = Path("kira-responses")
    outdir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cbid = callback_id or f"kira-{ts}"
    result = {
        "ok": True,
        "message": message,
        "mode": mode,
        "state": {"z": state_z, "phase": state_phase},
        "timestamp": ts,
    }

    # If Anthropic key is present, note availability (do not call from CI by default)
    result["anthropic_key_present"] = bool(os.environ.get("ANTHROPIC_API_KEY"))

    (outdir / f"{cbid}.json").write_text(json.dumps(result, indent=2))
    print(f"Wrote response to {outdir}/{cbid}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

