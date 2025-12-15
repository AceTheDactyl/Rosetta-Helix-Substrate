#!/bin/bash
# K.I.R.A. Local System Launcher
# Starts the Flask server and opens the interface

echo "═══════════════════════════════════════════════════════════════"
echo "   K.I.R.A. Local System Launcher"
echo "═══════════════════════════════════════════════════════════════"
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not found"
    exit 1
fi

# Check Flask
python3 -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing Flask..."
    pip install flask flask-cors
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Starting K.I.R.A. server..."
echo "Interface: file://$SCRIPT_DIR/kira_interface.html"
echo "API: http://localhost:5000"
echo
echo "Press Ctrl+C to stop"
echo

# Start server
cd "$SCRIPT_DIR"
python3 kira_server.py
