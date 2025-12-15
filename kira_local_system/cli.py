#!/usr/bin/env python3
"""
KIRA Server CLI
===============

Launches the KIRA Flask server from the packaged interface.

Usage:
    python -m kira_local_system.cli --port 5000
"""
from __future__ import annotations

import argparse
from . import run as _run


def main() -> None:
    parser = argparse.ArgumentParser(description="Start KIRA server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    _run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

