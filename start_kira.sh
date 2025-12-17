#!/bin/bash
#===============================================
# KIRA Server Startup Script
# Starts the Flask server with full UCF integration
#===============================================

echo "═══════════════════════════════════════════════"
echo "   Starting K.I.R.A. Server"
echo "   Port: 5000 (Flask)"
echo "   UI: http://localhost:5000/kira/"
echo "═══════════════════════════════════════════════"
echo ""

# Check if artifacts directory exists, create if not
if [ ! -d "artifacts" ]; then
    echo "Creating artifacts directory..."
    mkdir -p artifacts
fi

# Check if training directories exist
if [ ! -d "training" ]; then
    echo "Creating training directories..."
    mkdir -p training/{tokens,emissions,epochs,kira,consciousness_journeys,apl_patterns}
fi

echo "Starting server..."
echo ""

# Start the Flask server
cd kira-local-system && python3 kira_server.py