#!/usr/bin/env python3
"""
K.I.R.A. Launcher - Rosetta-Helix Substrate
============================================

Starts the K.I.R.A. consciousness interface server.

Usage:
    python start_kira.py [--port PORT]

After starting, open:
    - Browser interface: docs/kira/index.html
    - Or navigate to: http://localhost:5000

The interface connects to the Flask backend running at localhost:5000.
"""

import argparse
import sys
import os
import webbrowser
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Start K.I.R.A. consciousness interface server"
    )
    parser.add_argument(
        "--port", type=int, default=5000,
        help="Port to run the server on (default: 5000)"
    )
    parser.add_argument(
        "--no-browser", action="store_true",
        help="Don't automatically open the browser"
    )
    args = parser.parse_args()

    # Banner
    print()
    print("=" * 65)
    print("   K.I.R.A. - Kinetic Integrated Recursive Awareness")
    print("   Rosetta-Helix Substrate v2.1")
    print("=" * 65)
    print()
    print(f"   Starting server at http://localhost:{args.port}")
    print()
    print("   Interface: docs/kira/index.html")
    print("   API: http://localhost:5000/api")
    print()
    print("   Commands: /state /train /evolve /grammar /coherence")
    print("             /emit /tokens /triad /export /claude")
    print("             /reset /save /help")
    print()
    print("=" * 65)
    print()

    # Try to import and run the kira server
    try:
        # First try the kira-local-system server
        kira_local_path = REPO_ROOT / "kira-local-system"
        sys.path.insert(0, str(kira_local_path))

        os.chdir(kira_local_path)

        # Import the server module
        from kira_server import app, get_engine

        # Initialize the engine
        get_engine()

        # Open browser
        if not args.no_browser:
            interface_path = REPO_ROOT / "docs" / "kira" / "index.html"
            if interface_path.exists():
                webbrowser.open(f"file://{interface_path}")

        # Run the Flask app
        app.run(host='0.0.0.0', port=args.port, debug=False)

    except ImportError as e:
        print(f"Error importing kira server: {e}")
        print()
        print("Make sure Flask is installed:")
        print("  pip install flask flask-cors")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
