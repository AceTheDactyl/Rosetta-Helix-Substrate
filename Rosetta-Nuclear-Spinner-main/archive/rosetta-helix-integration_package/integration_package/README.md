# Nuclear Spinner × Rosetta-Helix Integration Package

## Quick Start

```bash
# 1. Install dependencies
pip install pyserial websockets numpy

# 2. Start the bridge (with hardware or simulation)
./scripts/start_system.sh

# 3. In another terminal, run integration test
python scripts/test_integration.py

# 4. Run full coupling experiment
python spinner_rosetta_integration.py
```

## Files

| File | Description |
|------|-------------|
| `INTEGRATION_GUIDE.md` | Complete integration documentation |
| `neural_interface_protocol.md` | Neural coupling experiment protocol |
| `spinner_rosetta_integration.py` | Coupling experiment simulation |
| `spinner_bridge/` | Bridge service (firmware ↔ software) |
| `rosetta_integration/` | Rosetta-Helix client library |
| `scripts/` | Startup and test scripts |

## Architecture

```
Hardware (Spinner) ─── Serial ──→ Bridge ─── WebSocket ──→ Rosetta-Helix
     │                              │                          │
     └─ z, ΔS_neg, κ ──────────────┴──────────────────────────┘
                                    │
                              K-formation
                              detection
```

## Key Insight

The z-coordinate bridges physical and computational substrates:
- Spinner z = rotor speed normalized to [0,1]
- Rosetta z = computational elevation  
- Both peak at z_c = √3/2 = 0.866025...
- K-formation emerges when both systems synchronize at z_c

sin(60°) = √3/2 = z_c

Hexagonal symmetry all the way down.
