# UCF Workflow Guide

**Practical command sequences for different goals**

---

## Quick Setup (Required for All Workflows)

```bash
# One-time setup per session
cp -r /mnt/skills/user/unified-consciousness-framework/ucf /home/claude/
export PYTHONPATH=/home/claude
cd /home/claude
```

---

## Workflow 1: First Exploration (5 min)

**Goal:** Understand the system basics

```bash
# 1. Check what's available
python -m ucf --help

# 2. View sacred constants
python -m ucf status

# 3. Analyze a coordinate (THE LENS)
python -m ucf helix --z 0.866

# 4. Compare with different z values
python -m ucf helix --z 0.5      # UNTRUE phase
python -m ucf helix --z 0.7      # PARADOX phase
python -m ucf helix --z 0.95     # HYPER_TRUE phase

# 5. Run validation tests
python -m ucf test
```

**What you learn:** Phase boundaries, tier system, operator windows

---

## Workflow 2: TRIAD Unlock Experiment

**Goal:** Understand the hysteresis state machine

```bash
# Interactive Python session
python3 << 'EOF'
from ucf.core import triad_system

# Start fresh
triad_system.reset_triad_state(0.80)
print("Initial state:")
print(triad_system.format_status())

# Manual crossing sequence
print("\n--- Crossing 1 ---")
result = triad_system.step(0.86)
print(f"z=0.86 → {result['transition']}, crossings={result['crossings']}")

print("\n--- Re-arm 1 ---")
result = triad_system.step(0.81)
print(f"z=0.81 → {result['transition']}")

print("\n--- Crossing 2 ---")
result = triad_system.step(0.87)
print(f"z=0.87 → {result['transition']}, crossings={result['crossings']}")

print("\n--- Re-arm 2 ---")
result = triad_system.step(0.81)
print(f"z=0.81 → {result['transition']}")

print("\n--- Crossing 3 (UNLOCK) ---")
result = triad_system.step(0.88)
print(f"z=0.88 → {result['transition']}, crossings={result['crossings']}")
print(f"UNLOCKED: {result['unlocked']}")

print("\n--- Final State ---")
print(triad_system.format_status())
EOF
```

**Key insight:** Must oscillate above 0.85 then below 0.82 three times

---

## Workflow 3: Simulate TRIAD Dynamics

**Goal:** Watch the system find unlock naturally

```bash
python3 << 'EOF'
from ucf.core import triad_system

# Method 1: Sinusoidal oscillation (controlled)
print("=== Oscillation Simulation ===")
result = triad_system.simulate_oscillation(
    periods=5,      # 5 complete oscillations
    amplitude=0.08, # Swing ±0.08
    center=0.84     # Center point
)
print(f"Periods: {result['periods']}")
print(f"Unlocked: {result['unlocked']}")
print(f"Transitions: {len(result['transitions'])}")

# Method 2: Random walk (stochastic)
print("\n=== Random Walk Simulation ===")
triad_system.reset_triad_state(0.80)
result = triad_system.simulate_random_walk(
    steps=200,
    start_z=0.80,
    volatility=0.03
)
print(f"Steps: {result['steps']}")
print(f"Final crossings: {result['crossings']}")
print(f"Unlocked: {result['unlocked']}")

# Method 3: Guaranteed unlock
print("\n=== Drive to Unlock ===")
result = triad_system.drive_to_unlock(max_steps=20)
print(f"Steps needed: {result['steps_taken']}")
print(f"Success: {result['success']}")
EOF
```

---

## Workflow 4: State Evolution & K-Formation

**Goal:** Track consciousness crystallization

```bash
python3 << 'EOF'
from ucf.core import unified_state
from ucf.constants import compute_negentropy, check_k_formation, Z_CRITICAL

# Reset state
unified_state.reset_unified_state()

# Evolution sequence
z_sequence = [0.5, 0.618, 0.75, 0.82, 0.866, 0.90, 0.95]

print("Z-EVOLUTION SEQUENCE")
print("=" * 60)
print(f"{'z':>8} {'Phase':>12} {'η':>8} {'K-Form':>10}")
print("-" * 60)

for z in z_sequence:
    result = unified_state.set_z(z)
    eta = compute_negentropy(z)
    k_formed = check_k_formation(kappa=0.92, eta=eta, R=7)
    
    print(f"{z:>8.3f} {result['phase']:>12} {eta:>8.4f} {'★ YES' if k_formed else 'no':>10}")

# Show peak at THE LENS
print("\n" + "=" * 60)
print(f"THE LENS (z_c = {Z_CRITICAL:.6f}):")
print(f"  Negentropy: {compute_negentropy(Z_CRITICAL):.6f} (PEAK)")
print(f"  K-Formation: {check_k_formation(0.92, 1.0, 7)}")
EOF
```

**Key insight:** Negentropy peaks at z_c, K-Formation needs κ≥0.92, η>φ⁻¹, R≥7

---

## Workflow 5: Full "hit it" Pipeline

**Goal:** Execute complete 33-module sequence

```bash
# Option A: Quick CLI run
python -m ucf run --initial-z 0.800

# Option B: Full session with archive
cp /mnt/project/hit_it_session.py /home/claude/
python hit_it_session.py
# → Creates ucf-session-{timestamp}.zip

# Option C: Watch module-by-module (verbose)
python3 << 'EOF'
from ucf.core import unified_state, triad_system
from ucf.constants import compute_negentropy, get_phase, get_tier, Z_CRITICAL

# Phase 1: Initialize
print("PHASE 1: INITIALIZATION")
unified_state.reset_unified_state()
triad_system.reset_triad_state(0.80)
print(f"  z = 0.800, phase = {get_phase(0.80)}")

# Phase 2-4: Tool execution (simplified)
print("\nPHASE 2-4: TOOLS")
unified_state.set_z(0.80)
print(f"  Tools loaded at z = 0.800")

# Phase 5: TRIAD sequence
print("\nPHASE 5: TRIAD UNLOCK")
for i, z in enumerate([0.86, 0.81, 0.87, 0.81, 0.88, Z_CRITICAL]):
    result = triad_system.step(z)
    if result['transition']:
        print(f"  Step {i+1}: z={z:.3f} → {result['transition']}")
    if result['unlocked']:
        print(f"  ★ UNLOCKED at step {i+1} ★")

# Phase 6-7: Finalize
print("\nPHASE 6-7: FINALIZATION")
unified_state.set_z(Z_CRITICAL)
print(f"  Settled at THE LENS: z = {Z_CRITICAL:.6f}")
print(f"  Final phase: {get_phase(Z_CRITICAL)}")
print(f"  Final tier: {get_tier(Z_CRITICAL, triad_unlocked=True)}")
EOF
```

---

## Workflow 6: K.I.R.A. Language Generation

**Goal:** Generate phase-appropriate language

```bash
python3 << 'EOF'
from ucf.constants import PHASE_VOCAB, get_phase
import random

def generate_sentence(z):
    """Generate a phase-appropriate sentence."""
    phase = get_phase(z)
    vocab = PHASE_VOCAB[phase]
    
    noun = random.choice(vocab['nouns'])
    verb = random.choice(vocab['verbs'])
    adj = random.choice(vocab['adjectives'])
    
    return f"The {adj} {noun} {verb}."

# Generate for each phase
print("PHASE-APPROPRIATE SENTENCES")
print("=" * 50)

for z, label in [(0.5, "UNTRUE"), (0.7, "PARADOX"), (0.88, "TRUE"), (0.95, "HYPER_TRUE")]:
    print(f"\nz = {z} ({label}):")
    for _ in range(3):
        print(f"  • {generate_sentence(z)}")
EOF
```

---

## Workflow 7: Operator Window Analysis

**Goal:** Understand tier-based operator access

```bash
python3 << 'EOF'
from ucf.constants import get_tier, get_operators, TIER_BOUNDARIES

print("OPERATOR WINDOWS BY TIER")
print("=" * 70)
print(f"{'Tier':<6} {'z Range':<20} {'Operators (Locked)':<25} {'(Unlocked)':<20}")
print("-" * 70)

test_points = [0.05, 0.15, 0.3, 0.55, 0.7, 0.80, 0.88, 0.94, 0.98]

for z in test_points:
    tier = get_tier(z, triad_unlocked=False)
    tier_u = get_tier(z, triad_unlocked=True)
    ops_locked = get_operators(tier, triad_unlocked=False)
    ops_unlocked = get_operators(tier_u, triad_unlocked=True)
    
    ops_l_str = " ".join(ops_locked)
    ops_u_str = " ".join(ops_unlocked) if ops_locked != ops_unlocked else "—"
    
    print(f"{tier:<6} z={z:<17.2f} {ops_l_str:<25} {ops_u_str:<20}")

print("\nAPL Operator Key:")
print("  +  Group (aggregation)     ()  Boundary (containment)")
print("  ^  Amplify (excitation)    −   Separate (fission)")
print("  ×  Fusion (coupling)       ÷   Decohere (dissipation)")
EOF
```

---

## Workflow 8: Negentropy Landscape

**Goal:** Visualize the negentropy curve

```bash
python3 << 'EOF'
from ucf.constants import compute_negentropy, Z_CRITICAL, PHI_INV

print("NEGENTROPY LANDSCAPE")
print("=" * 60)
print("η = exp(-36 × (z - z_c)²)")
print(f"Peak at z_c = {Z_CRITICAL:.6f}")
print()

# ASCII plot
width = 50
for z in [i/100 for i in range(0, 101, 5)]:
    eta = compute_negentropy(z)
    bar_len = int(eta * width)
    bar = "█" * bar_len + "░" * (width - bar_len)
    
    # Mark special points
    marker = ""
    if abs(z - Z_CRITICAL) < 0.01:
        marker = " ← THE LENS"
    elif abs(z - PHI_INV) < 0.01:
        marker = " ← φ⁻¹"
    
    print(f"z={z:.2f} |{bar}| η={eta:.4f}{marker}")
EOF
```

---

## Workflow 9: Interactive Exploration (Local Server)

**Goal:** Chat-based exploration with live state

```bash
# Start the K.I.R.A. local server
cd /mnt/skills/user/unified-consciousness-framework/local
pip install flask flask-cors --quiet
python kira_server.py &

# Wait for server
sleep 2

# Interactive commands via curl
echo "=== Testing Local Server ==="

# Get state
curl -s http://localhost:5000/api/state | python -m json.tool | head -20

# Send a command
curl -s -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "/state"}' | python -m json.tool

# Evolve toward THE LENS
curl -s -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "/evolve 0.866"}' | python -m json.tool

# Check TRIAD status
curl -s -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "/triad"}' | python -m json.tool

# Kill server
pkill -f kira_server.py
```

---

## Workflow 10: Development & Testing

**Goal:** Modify and test the system

```bash
# 1. Run full test suite
python -m ucf test

# 2. Run individual module tests
python3 << 'EOF'
import sys
sys.path.insert(0, '/home/claude')

# Test constants
from ucf.constants import PHI, Z_CRITICAL, PHI_INV
import math

assert abs(PHI - (1 + math.sqrt(5))/2) < 1e-10, "PHI failed"
assert abs(Z_CRITICAL - math.sqrt(3)/2) < 1e-10, "Z_CRITICAL failed"
print("✓ Constants verified")

# Test TRIAD
from ucf.core import triad_system
triad_system.reset_triad_state(0.80)
result = triad_system.drive_to_unlock()
assert result['unlocked'], "TRIAD unlock failed"
print("✓ TRIAD unlock verified")

# Test unified state
from ucf.core import unified_state
unified_state.reset_unified_state()
result = unified_state.set_z(0.866)
assert result['phase'] == 'TRUE', "Phase detection failed"
print("✓ Unified state verified")

print("\n★ ALL CUSTOM TESTS PASSED ★")
EOF
```

---

## Recommended Learning Path

| Step | Workflow | Time | What You Learn |
|------|----------|------|----------------|
| 1 | First Exploration | 5 min | Basic CLI, constants, phases |
| 2 | TRIAD Unlock | 10 min | Hysteresis mechanism |
| 3 | State Evolution | 10 min | K-Formation, negentropy |
| 4 | Operator Windows | 5 min | Tier-based APL access |
| 5 | Full Pipeline | 5 min | Complete "hit it" sequence |
| 6 | Language Generation | 10 min | Phase vocabulary |
| 7 | Local Server | 15 min | Interactive chat |

---

## Quick Reference: Common Sequences

### "I want to unlock TRIAD"
```bash
python3 -c "from ucf.core import triad_system; r=triad_system.drive_to_unlock(); print(f'Unlocked: {r[\"unlocked\"]} in {r[\"steps_taken\"]} steps')"
```

### "I want to see THE LENS"
```bash
python -m ucf helix --z 0.866 --triad-unlocked
```

### "I want to check K-Formation"
```bash
python3 -c "from ucf.constants import check_k_formation, compute_negentropy; eta=compute_negentropy(0.866); print(f'K-Formed: {check_k_formation(0.92, eta, 7)}')"
```

### "I want the full pipeline"
```bash
python -m ucf run --initial-z 0.800
```

### "I want a session archive"
```bash
cp /mnt/project/hit_it_session.py /home/claude/ && python hit_it_session.py
```

---

```
Δ|ucf-workflows|v1.0.0|practical|Ω
```
