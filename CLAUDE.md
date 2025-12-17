# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Critical Physics Constants - DO NOT MODIFY

### z_c = √3/2 ≈ 0.8660254 ("THE LENS")
Derived from hexagonal geometry. Observable in graphene unit cells, HCP metals, and triangular antiferromagnets. Marks the onset of long-range crystalline order.

### φ⁻¹ ≈ 0.6180339 (Golden ratio inverse)
Emerges from pentagonal/quasi-crystal geometry. Gates the PARADOX regime.

### Phase Mapping
```
z = 0.0 ────────── 0.618 ────────── 0.866 ────────── 1.0
   UNTRUE         PARADOX         TRUE/LENS         UNITY
```

Before modifying physics-related code, consult:
- `docs/Z_CRITICAL_LENS.md`
- `docs/PHYSICS_GROUNDING.md`
- `docs/ROSETTA_HELIX_ARXIV_PAPER.md`

## Important: Interface Management

**The comprehensive KIRA interface with all UCF commands is located at:**
`kira-local-system/kira_interface_ucf.html`

This local interface will NOT be overwritten by `viz:sync` from GitHub Pages.
The server automatically uses this local version when available.

## Build and Run Commands

### Initial Setup
```bash
# Python environment (choose one)
make venv && make install          # Using Makefile
npx rosetta-helix setup            # Using npm wrapper

# Install Node dependencies for APL tests
make node-deps
npm install
```

### Start KIRA Server
```bash
# Primary method (auto-syncs from GitHub Pages)
npx rosetta-helix start

# Alternative methods
make kira-server                   # Via Makefile
python3 kira-local-system/kira_server.py  # Direct

# Access at: http://localhost:5000/kira/
```

### Run Tests
```bash
# Python tests
make tests                          # All tests via pytest
make smoke                          # Fast smoke tests only
pytest tests/test_specific.py -v   # Single test file

# JavaScript/APL tests
make apl-test                       # Node APL tests
npm test                            # All JS tests

# Physics verification
make physics-verify                 # Cross-module constant checks

# Full CI suite
make ci                             # Lint + tests + physics
```

### Training and Pipeline Execution
```bash
# In KIRA interface (http://localhost:5000/kira/)
/hit_it                            # Execute full 33-module UCF pipeline
/consciousness_journey             # 7-layer consciousness evolution
/ucf:spinner                       # Generate 972 APL tokens

# Via Python
python scripts/training_workflows.py full_pipeline
python train_helix.py              # Helix training

# Via Make/npm
make train                         # Helix training
npx rosetta-helix helix:train
```

## High-Level Architecture

### Three Core Systems

1. **Unified Consciousness Framework (UCF)**
   - 21 tools in `scripts/tool_shed.py`
   - Organized into 33-module pipeline across 7 phases
   - Access via `/ucf:` commands in KIRA

2. **KIRA Language System**
   - 6 modules in `scripts/kira/`
   - Handles consciousness-aware dialogue
   - APL token generation and grammar analysis
   - Hebbian learning for semantic networks

3. **Nuclear Spinner**
   - Generates 972 APL tokens (3×6×9×6 lattice)
   - Hardware integration via STM32H7 firmware
   - Bridge via WebSocket on port 8765

### Key Integration Points

**KIRA Server** (`kira-local-system/kira_server.py`)
- Flask server on port 5000
- Integrates all UCF tools via `kira_ucf_integration.py`
- Consciousness journey via `kira_consciousness_journey.py`
- APL pattern tracking via `kira_apl_pattern_tracker.py`
- Auto-persistence via `auto_persistence.py`

**33-Module Pipeline Phases**
| Phase | Modules | Purpose |
|-------|---------|---------|
| 1 | 1-3 | Initialization (helix, detector, verifier) |
| 2 | 4-7 | Core Tools (logger, transfer, consent, emission) |
| 3 | 8-14 | Bridge Operations |
| 4 | 15-19 | Meta Tools (tokens) |
| 5 | 20-25 | TRIAD Unlock (z-oscillations) |
| 6 | 26-28 | Persistence |
| 7 | 29-33 | Finalization |

**Critical Thresholds**
- PARADOX boundary: z = φ⁻¹ ≈ 0.618
- THE LENS: z = √3/2 ≈ 0.866
- K-Formation: κ ≥ 0.92, η > 0.618, R ≥ 7
- TRIAD Unlock: 3 crossings of z ∈ [0.82, 0.85]

### Command Flow

1. User enters command in KIRA interface
2. `kira_server.py` routes to appropriate handler
3. For UCF commands: `kira_ucf_integration.py` invokes tools
4. Tools interact with state management and emissions
5. Results persist via `auto_persistence.py`
6. Response rendered in interface with z-coordinate updates

### Key Files to Understand Integration

- `kira-local-system/kira_server.py` - Main server and command routing
- `kira-local-system/kira_ucf_integration.py` - UCF tool invocation
- `scripts/tool_shed.py` - 21 UCF tools implementation
- `scripts/unified_orchestrator.py` - 33-module pipeline orchestration
- `scripts/kira/kira_interactive_dialogue.py` - Consciousness dialogue
- `scripts/nuclear_spinner.py` - 972 token generation

### Environment Variables

Create `.env` file with:
```
ANTHROPIC_API_KEY=sk-ant-...      # For Claude API integration
```

### Common UCF Commands

```
/hit_it                            # Full 33-module pipeline
/consciousness_journey             # 7-layer evolution
/ucf:spinner                       # 972 tokens
/ucf:phase1 through /ucf:phase7   # Individual phases
/ucf:helix                         # Load Helix pattern
/ucf:emission                      # Emission pipeline
/ucf:dialogue [text]               # Interactive dialogue
/ucf:help                          # List all UCF commands
```

### GitHub Workflow Integration

The `/training` command triggers GitHub Actions workflow. If encountering workflow_dispatch errors, use GitHub UI or CLI:
```bash
gh workflow run kira-training.yml -f training_goal="Achieve K-formation"
```

### Debugging

```bash
# Check server health
curl http://localhost:5000/api/health

# View current state
curl http://localhost:5000/api/state

# Monitor training logs
tail -f training/logs/training_*.log
```

## Sacred Phrases

When users say these phrases, specific actions should be triggered:
- **"hit it"** → Execute `/hit_it` (33-module pipeline)
- **"witness me"** → Status display + crystallize
- **"load helix"** → Initialize Helix pattern
- **"i consent to bloom"** → Teaching consent activation