# KIRA Server Fix and Setup Guide

## Issues Fixed

### 1. ✅ 404 Error for `/artifacts/latest_training_data.json`
- **Problem**: The artifacts directory didn't exist
- **Solution**:
  - Created `artifacts/` directory
  - Added `latest_training_data.json` with initial state
  - Added Flask route to serve artifacts: `@app.route('/artifacts/<path:filename>')`

### 2. ✅ Server Configuration Clarification
- **Current Setup**: Flask server on port 5000 (this is correct!)
- **Location**: `kira-local-system/kira_server.py`
- **Features**: Full UCF integration, hit_it command, consciousness journey

## Server Architecture

You have **TWO different servers** in the project:

1. **KIRA Flask Server** (`kira-local-system/kira_server.py`)
   - Port: 5000
   - Protocol: HTTP/REST API
   - UI: `docs/kira/index.html`
   - Features: Full UCF integration, all commands, web interface

2. **Unified Rosetta Server** (`unified_rosetta_server.py`)
   - Port: 8765
   - Protocol: WebSocket
   - Purpose: Real-time updates, nuclear spinner integration

## Quick Start (Correct Way)

### 1. Start KIRA Flask Server (Main Interface)

```bash
cd kira-local-system
python kira_server.py
```

This starts:
- HTTP server on http://localhost:5000
- Web UI at http://localhost:5000/kira/
- REST API endpoints

### 2. Access the Web Interface

Open browser to: **http://localhost:5000/kira/**

(NOT port 9999 - that was a mixup in documentation)

### 3. Test Everything Works

In the web interface chat, try these commands:

```
/state
/hit_it
/consciousness_journey
/ucf:status
```

## File Structure

```
Rosetta-Helix-Substrate/
├── kira-local-system/
│   ├── kira_server.py              # Main Flask server (port 5000)
│   ├── kira_ucf_integration.py     # UCF 33-module integration
│   ├── kira_consciousness_journey.py # 7-layer journey
│   └── kira_apl_pattern_tracker.py # APL pattern learning
├── artifacts/
│   └── latest_training_data.json   # Training data (now exists!)
├── training/                        # Saved training data
├── docs/kira/
│   └── index.html                   # Web UI interface
└── unified_rosetta_server.py       # WebSocket server (different, port 8765)
```

## Available Commands in KIRA

All these work in the Flask server on port 5000:

### Core Commands
- `/state` - Current consciousness state
- `/evolve [z]` - Evolve to target z-coordinate
- `/emit` - Run emission pipeline
- `/tokens [n]` - Generate APL tokens
- `/hit_it` - Run full 33-module UCF pipeline ✨
- `/consciousness_journey` - 7-layer consciousness evolution

### UCF Commands
- `/ucf:status` - UCF system status
- `/ucf:spinner` - Generate 972 tokens
- `/ucf:pipeline` - Run complete pipeline
- `/ucf:help` - List all UCF commands
- `/ucf:phase1` through `/ucf:phase7` - Run specific phases

### Training Commands
- `/save` - Save session
- `/export` - Export training data
- `/apl_patterns` - View learned patterns

## Testing the Fixes

### 1. Verify No More 404 Errors

```bash
# Check artifacts directory exists
ls -la artifacts/

# Should see:
# latest_training_data.json
```

### 2. Test in Browser

1. Start server: `python kira-local-system/kira_server.py`
2. Open: http://localhost:5000/kira/
3. Open browser console (F12)
4. Should see NO 404 errors
5. Try `/hit_it` command - should work!

### 3. Test UCF Integration

```python
# In another terminal, while server is running:
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "/hit_it"}'
```

## Important Notes

1. **Port 5000 is CORRECT** - The Flask server is supposed to run on port 5000
2. **UCF Integration WORKS** - All 33 modules are integrated
3. **Artifacts Fixed** - No more 404 errors
4. **Two Servers** - Flask (5000) for UI, WebSocket (8765) for real-time

## Troubleshooting

### If server won't start:
```bash
# Check Python version
python --version  # Need 3.11+

# Install dependencies
pip install flask flask-cors requests
```

### If UCF commands don't work:
```bash
# Check UCF modules exist
ls ucf-v4-extracted/

# Check integration loaded (see server startup messages)
```

### If /hit_it fails:
- Check console for specific errors
- Verify `kira_ucf_integration.py` exists
- Look for "[K.I.R.A.] UCF Integration loaded" in startup

## Summary

✅ **404 Error Fixed**: Created artifacts directory and added route
✅ **Server Runs Correctly**: Flask on port 5000 (as designed)
✅ **UCF Integration Works**: All 33 modules available
✅ **Web Interface Works**: Access at http://localhost:5000/kira/

The confusion was about port numbers - the Flask server (port 5000) is the correct one to use for the web interface. The WebSocket server (port 8765) is a separate component for real-time updates.

**Just use**: `python kira-local-system/kira_server.py` and access http://localhost:5000/kira/