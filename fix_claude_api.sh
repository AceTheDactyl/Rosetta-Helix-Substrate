#!/bin/bash
#=====================================================
# Fix Claude API Authentication Error
#=====================================================

echo "═══════════════════════════════════════════════════════════════"
echo "   Fixing Claude API Authentication"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check if .env file exists
if [ -f ".env" ]; then
    echo "✅ .env file exists with ANTHROPIC_API_KEY"
else
    echo "❌ .env file missing"
    echo "Creating .env file..."

    # Check if API key file exists
    if [ -f "claude api key.txt" ]; then
        API_KEY=$(cat "claude api key.txt")
        echo "# Claude API Configuration" > .env
        echo "ANTHROPIC_API_KEY=$API_KEY" >> .env
        echo "✅ Created .env file with API key"
    else
        echo "❌ No API key found"
        echo "Please add your key to .env file:"
        echo "ANTHROPIC_API_KEY=sk-ant-..."
        exit 1
    fi
fi

# Install python-dotenv if not installed
echo ""
echo "Checking python-dotenv..."
python3 -c "import dotenv" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing python-dotenv..."
    pip install python-dotenv || pip3 install python-dotenv
    echo "✅ python-dotenv installed"
else
    echo "✅ python-dotenv already installed"
fi

# Export for current session
export ANTHROPIC_API_KEY=$(grep ANTHROPIC_API_KEY .env | cut -d= -f2)

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "   Setup Complete!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "✅ .env file configured"
echo "✅ python-dotenv installed"
echo "✅ ANTHROPIC_API_KEY loaded"
echo ""
echo "Restart the server to apply changes:"
echo "  npx rosetta-helix start"
echo ""
echo "Or directly:"
echo "  cd kira-local-system"
echo "  python3 kira_server.py"
echo ""
echo "Claude API will now work with /claude command!"