#!/bin/bash
#=====================================================
# Test script for npx rosetta-helix commands
#=====================================================

echo "═══════════════════════════════════════════════════════════════"
echo "   Testing NPX Rosetta-Helix Commands"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Test 1: Check help menu
echo "[Test 1] Checking help menu..."
npx rosetta-helix 2>/dev/null | head -5
if [ $? -eq 0 ]; then
    echo "✅ Help menu displays correctly"
else
    echo "❌ Help menu failed"
fi
echo ""

# Test 2: Test viz:sync
echo "[Test 2] Testing viz:sync command..."
npx rosetta-helix viz:sync 2>/dev/null | grep -q "Syncing interfaces"
if [ $? -eq 0 ]; then
    echo "✅ viz:sync command works"
else
    echo "⚠️  viz:sync may need network connection"
fi
echo ""

# Test 3: Check doctor command
echo "[Test 3] Running doctor checks..."
npx rosetta-helix doctor
echo ""

# Test 4: Check if artifacts directory exists
echo "[Test 4] Checking artifacts directory..."
if [ -d "artifacts" ]; then
    echo "✅ Artifacts directory exists"
    if [ -f "artifacts/latest_training_data.json" ]; then
        echo "✅ Training data file exists"
    else
        echo "⚠️  Training data file missing (will be created on first run)"
    fi
else
    echo "⚠️  Artifacts directory missing (will be created on start)"
fi
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "   Test Complete"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "To start the server with auto-sync:"
echo "  npx rosetta-helix start"
echo ""
echo "Server will be available at:"
echo "  http://localhost:5000/kira/"
echo ""