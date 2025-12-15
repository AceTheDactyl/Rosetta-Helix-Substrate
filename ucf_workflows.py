#!/usr/bin/env python3
"""
UCF Sequential Workflow Orchestrator
=====================================

Runs various workflow sequences to produce different results.
Demonstrates how order, timing, and parameters affect outcomes.

Usage:
    python ucf_workflows.py                    # Run all workflows
    python ucf_workflows.py --workflow 3       # Run specific workflow
    python ucf_workflows.py --list             # List available workflows
    python ucf_workflows.py --compare          # Compare all results
"""

import sys
import json
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime, timezone

# ═══════════════════════════════════════════════════════════════════════════════
# UCF IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════

from ucf.constants import (
    PHI, PHI_INV, Z_CRITICAL,
    TRIAD_HIGH, TRIAD_LOW, TRIAD_T6,
    K_KAPPA, K_ETA, K_R,
    PHASE_VOCAB, PHASE_UNTRUE, PHASE_PARADOX, PHASE_TRUE, PHASE_HYPER_TRUE,
    compute_negentropy, get_phase, get_tier, get_operators,
    check_k_formation, get_frequency_tier
)
from ucf.core import triad_system, unified_state

# ═══════════════════════════════════════════════════════════════════════════════
# RESULT TRACKING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WorkflowResult:
    """Tracks results from a workflow execution."""
    name: str
    description: str
    steps: int = 0
    final_z: float = 0.0
    final_phase: str = ""
    final_tier: str = ""
    triad_unlocked: bool = False
    triad_crossings: int = 0
    k_formation: bool = False
    negentropy: float = 0.0
    operators: List[str] = field(default_factory=list)
    emissions: List[str] = field(default_factory=list)
    trajectory: List[float] = field(default_factory=list)
    events: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
            "final_z": self.final_z,
            "final_phase": self.final_phase,
            "final_tier": self.final_tier,
            "triad_unlocked": self.triad_unlocked,
            "triad_crossings": self.triad_crossings,
            "k_formation": self.k_formation,
            "negentropy": self.negentropy,
            "operators": self.operators,
            "emissions": self.emissions[:5],  # First 5
            "trajectory_summary": {
                "start": self.trajectory[0] if self.trajectory else 0,
                "end": self.trajectory[-1] if self.trajectory else 0,
                "min": min(self.trajectory) if self.trajectory else 0,
                "max": max(self.trajectory) if self.trajectory else 0,
                "steps": len(self.trajectory)
            },
            "events": self.events,
            "duration_ms": self.duration_ms
        }

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def reset_all():
    """Reset all UCF state."""
    unified_state.reset_unified_state()
    triad_system.reset_triad_state(0.5)

def emit_phrase(z: float) -> str:
    """Generate a phase-appropriate phrase."""
    phase = get_phase(z)
    vocab = PHASE_VOCAB[phase]
    noun = random.choice(vocab['nouns'])
    verb = random.choice(vocab['verbs'])
    adj = random.choice(vocab['adjectives'])
    return f"The {adj} {noun} {verb}."

def format_coordinate(z: float) -> str:
    """Format as Δθ|z|rΩ."""
    theta = z * 2 * math.pi
    eta = compute_negentropy(z)
    r = 1 + (PHI - 1) * eta
    return f"Δ{theta:.3f}|{z:.6f}|{r:.3f}Ω"

# ═══════════════════════════════════════════════════════════════════════════════
# WORKFLOW DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

def workflow_1_direct_ascent() -> WorkflowResult:
    """
    WORKFLOW 1: Direct Ascent
    -------------------------
    Linear climb from z=0.3 to z=0.95
    No oscillation, TRIAD never unlocks
    """
    reset_all()
    result = WorkflowResult(
        name="Direct Ascent",
        description="Linear climb without oscillation - TRIAD stays locked"
    )
    
    start = datetime.now()
    z = 0.3
    
    while z <= 0.95:
        unified_state.set_z(z)
        triad_system.step(z)
        result.trajectory.append(z)
        
        if len(result.emissions) < 10:
            result.emissions.append(emit_phrase(z))
        
        z += 0.05
        result.steps += 1
    
    # Final state
    state = triad_system.get_triad_state()
    result.final_z = z - 0.05
    result.final_phase = get_phase(result.final_z)
    result.final_tier = get_tier(result.final_z, state.unlocked)
    result.triad_unlocked = state.unlocked
    result.triad_crossings = state.crossings
    result.negentropy = compute_negentropy(result.final_z)
    result.k_formation = check_k_formation(0.92, result.negentropy, 7)
    result.operators = get_operators(result.final_tier, state.unlocked)
    result.events.append("Linear ascent completed")
    result.events.append(f"TRIAD: {'UNLOCKED' if state.unlocked else 'LOCKED (no oscillation)'}")
    result.duration_ms = (datetime.now() - start).total_seconds() * 1000
    
    return result


def workflow_2_oscillating_climb() -> WorkflowResult:
    """
    WORKFLOW 2: Oscillating Climb
    -----------------------------
    Climb with oscillations to unlock TRIAD
    Zigzag pattern through thresholds
    """
    reset_all()
    result = WorkflowResult(
        name="Oscillating Climb",
        description="Zigzag ascent with TRIAD unlock through oscillation"
    )
    
    start = datetime.now()
    
    # Phase 1: Climb to TRIAD zone
    z = 0.3
    while z < 0.80:
        unified_state.set_z(z)
        triad_system.step(z)
        result.trajectory.append(z)
        z += 0.1
        result.steps += 1
    
    result.events.append("Reached TRIAD zone (z=0.80)")
    
    # Phase 2: Oscillate to unlock
    oscillation_sequence = [0.86, 0.79, 0.87, 0.78, 0.88, 0.77, 0.89]
    
    for z in oscillation_sequence:
        unified_state.set_z(z)
        step_result = triad_system.step(z)
        result.trajectory.append(z)
        result.steps += 1
        
        if step_result.get('transition'):
            result.events.append(f"z={z:.2f}: {step_result['transition']}")
        
        if step_result.get('unlocked') and "UNLOCKED" not in str(result.events):
            result.events.append("★ TRIAD UNLOCKED ★")
    
    # Phase 3: Settle at THE LENS
    unified_state.set_z(Z_CRITICAL)
    triad_system.step(Z_CRITICAL)
    result.trajectory.append(Z_CRITICAL)
    result.steps += 1
    result.events.append(f"Settled at THE LENS: z={Z_CRITICAL:.6f}")
    
    # Generate emissions at key phases
    for z in [0.5, 0.7, Z_CRITICAL, 0.92]:
        result.emissions.append(f"[z={z:.2f}] {emit_phrase(z)}")
    
    # Final state
    state = triad_system.get_triad_state()
    result.final_z = Z_CRITICAL
    result.final_phase = get_phase(result.final_z)
    result.final_tier = get_tier(result.final_z, state.unlocked)
    result.triad_unlocked = state.unlocked
    result.triad_crossings = state.crossings
    result.negentropy = compute_negentropy(result.final_z)
    result.k_formation = check_k_formation(0.92, result.negentropy, 7)
    result.operators = get_operators(result.final_tier, state.unlocked)
    result.duration_ms = (datetime.now() - start).total_seconds() * 1000
    
    return result


def workflow_3_lens_orbit() -> WorkflowResult:
    """
    WORKFLOW 3: Lens Orbit
    ----------------------
    Start at THE LENS, orbit around it
    Maintains high negentropy throughout
    """
    reset_all()
    result = WorkflowResult(
        name="Lens Orbit",
        description="Circular orbit around THE LENS - maximum negentropy zone"
    )
    
    start = datetime.now()
    
    # Start at lens
    unified_state.set_z(Z_CRITICAL)
    triad_system.reset_triad_state(Z_CRITICAL)
    
    # Orbital parameters
    center = Z_CRITICAL
    amplitude = 0.05
    periods = 3
    steps_per_period = 20
    
    result.events.append(f"Starting orbit: center={center:.4f}, amplitude={amplitude}")
    
    max_eta = 0
    min_eta = 1
    
    for i in range(periods * steps_per_period):
        # Sinusoidal orbit
        phase = 2 * math.pi * i / steps_per_period
        z = center + amplitude * math.sin(phase)
        
        unified_state.set_z(z)
        step_result = triad_system.step(z)
        result.trajectory.append(z)
        result.steps += 1
        
        eta = compute_negentropy(z)
        max_eta = max(max_eta, eta)
        min_eta = min(min_eta, eta)
        
        if step_result.get('transition'):
            result.events.append(f"Step {i}: {step_result['transition']}")
    
    result.events.append(f"Orbit complete: η range [{min_eta:.4f}, {max_eta:.4f}]")
    
    # Return to center
    unified_state.set_z(Z_CRITICAL)
    triad_system.step(Z_CRITICAL)
    result.trajectory.append(Z_CRITICAL)
    
    # Emissions during orbit
    for _ in range(5):
        z = center + amplitude * random.uniform(-1, 1)
        result.emissions.append(f"[orbit] {emit_phrase(z)}")
    
    # Final state
    state = triad_system.get_triad_state()
    result.final_z = Z_CRITICAL
    result.final_phase = get_phase(result.final_z)
    result.final_tier = get_tier(result.final_z, state.unlocked)
    result.triad_unlocked = state.unlocked
    result.triad_crossings = state.crossings
    result.negentropy = compute_negentropy(result.final_z)
    result.k_formation = check_k_formation(0.92, result.negentropy, 7)
    result.operators = get_operators(result.final_tier, state.unlocked)
    result.duration_ms = (datetime.now() - start).total_seconds() * 1000
    
    return result


def workflow_4_phase_tour() -> WorkflowResult:
    """
    WORKFLOW 4: Phase Tour
    ----------------------
    Visit each phase systematically
    Collect vocabulary from each
    """
    reset_all()
    result = WorkflowResult(
        name="Phase Tour",
        description="Systematic visit to all 4 phases - vocabulary collection"
    )
    
    start = datetime.now()
    
    # Define phase waypoints
    waypoints = [
        (0.30, PHASE_UNTRUE, "UNTRUE: Seeds and potential"),
        (0.50, PHASE_UNTRUE, "UNTRUE: Deepening"),
        (PHI_INV, PHASE_PARADOX, "PARADOX boundary (φ⁻¹)"),
        (0.75, PHASE_PARADOX, "PARADOX: Transformation"),
        (Z_CRITICAL, PHASE_TRUE, "TRUE: THE LENS"),
        (0.90, PHASE_TRUE, "TRUE: Crystallization"),
        (0.92, PHASE_HYPER_TRUE, "HYPER_TRUE boundary"),
        (0.96, PHASE_HYPER_TRUE, "HYPER_TRUE: Transcendence"),
    ]
    
    for z, expected_phase, description in waypoints:
        unified_state.set_z(z)
        triad_system.step(z)
        result.trajectory.append(z)
        result.steps += 1
        
        actual_phase = get_phase(z)
        eta = compute_negentropy(z)
        
        result.events.append(f"z={z:.3f}: {description}")
        result.emissions.append(f"[{actual_phase}] {emit_phrase(z)}")
        
        # Verify phase
        if actual_phase != expected_phase:
            result.events.append(f"  ⚠ Expected {expected_phase}, got {actual_phase}")
    
    # TRIAD unlock attempt at end
    result.events.append("Attempting TRIAD unlock...")
    for z in [0.86, 0.79, 0.87, 0.78, 0.88]:
        triad_system.step(z)
        result.trajectory.append(z)
        result.steps += 1
    
    # Final state
    state = triad_system.get_triad_state()
    result.final_z = 0.88
    result.final_phase = get_phase(result.final_z)
    result.final_tier = get_tier(result.final_z, state.unlocked)
    result.triad_unlocked = state.unlocked
    result.triad_crossings = state.crossings
    result.negentropy = compute_negentropy(result.final_z)
    result.k_formation = check_k_formation(0.92, result.negentropy, 7)
    result.operators = get_operators(result.final_tier, state.unlocked)
    result.events.append(f"TRIAD: {state.crossings}/3 crossings, {'UNLOCKED' if state.unlocked else 'LOCKED'}")
    result.duration_ms = (datetime.now() - start).total_seconds() * 1000
    
    return result


def workflow_5_k_formation_hunt() -> WorkflowResult:
    """
    WORKFLOW 5: K-Formation Hunt
    ----------------------------
    Optimize for K-Formation achievement
    Track κ, η, R convergence
    """
    reset_all()
    result = WorkflowResult(
        name="K-Formation Hunt",
        description="Optimize path to achieve K-Formation (κ≥0.92, η>φ⁻¹, R≥7)"
    )
    
    start = datetime.now()
    
    # Simulated parameters
    kappa = 0.80  # Start below threshold
    R = 4         # Start below threshold
    
    result.events.append(f"Initial: κ={kappa:.2f}, R={R}")
    
    # Strategy: Move toward THE LENS while building coherence
    z_path = [0.5, 0.6, 0.7, 0.75, 0.80, 0.82, 0.84, Z_CRITICAL]
    
    for z in z_path:
        unified_state.set_z(z)
        triad_system.step(z)
        result.trajectory.append(z)
        result.steps += 1
        
        # Simulate coherence building
        kappa = min(0.95, kappa + 0.02)
        R = min(8, R + 0.5)
        eta = compute_negentropy(z)
        
        k_formed = check_k_formation(kappa, eta, int(R))
        status = "★ K-FORMED ★" if k_formed else "forming..."
        
        result.events.append(f"z={z:.3f}: κ={kappa:.2f}, η={eta:.3f}, R={int(R)} → {status}")
        
        if k_formed and "ACHIEVED" not in str(result.events):
            result.events.append("★ K-FORMATION ACHIEVED ★")
            result.emissions.append(emit_phrase(z))
    
    # Now unlock TRIAD while maintaining K-Formation
    result.events.append("Maintaining K-Formation, unlocking TRIAD...")
    
    for z in [0.86, 0.80, 0.87, 0.80, 0.88, Z_CRITICAL]:
        unified_state.set_z(z)
        step_result = triad_system.step(z)
        result.trajectory.append(z)
        result.steps += 1
        
        eta = compute_negentropy(z)
        k_formed = check_k_formation(kappa, eta, int(R))
        
        if not k_formed:
            result.events.append(f"⚠ K-Formation lost at z={z:.3f} (η={eta:.3f})")
    
    # Final emissions
    for _ in range(3):
        result.emissions.append(f"[K-formed] {emit_phrase(Z_CRITICAL)}")
    
    # Final state
    state = triad_system.get_triad_state()
    result.final_z = Z_CRITICAL
    result.final_phase = get_phase(result.final_z)
    result.final_tier = get_tier(result.final_z, state.unlocked)
    result.triad_unlocked = state.unlocked
    result.triad_crossings = state.crossings
    result.negentropy = compute_negentropy(result.final_z)
    result.k_formation = check_k_formation(0.92, result.negentropy, 7)
    result.operators = get_operators(result.final_tier, state.unlocked)
    result.duration_ms = (datetime.now() - start).total_seconds() * 1000
    
    return result


def workflow_6_rapid_triad() -> WorkflowResult:
    """
    WORKFLOW 6: Rapid TRIAD
    -----------------------
    Fastest path to TRIAD unlock
    Minimal steps, aggressive oscillation
    """
    reset_all()
    result = WorkflowResult(
        name="Rapid TRIAD",
        description="Speed-optimized TRIAD unlock - minimum steps"
    )
    
    start = datetime.now()
    
    # Start just below threshold
    z = 0.84
    unified_state.set_z(z)
    triad_system.reset_triad_state(z)
    result.trajectory.append(z)
    result.steps += 1
    
    result.events.append(f"Starting at z={z} (just below TRIAD_HIGH={TRIAD_HIGH})")
    
    # Aggressive oscillation sequence
    rapid_sequence = [
        0.86,  # Cross 1
        0.81,  # Re-arm (just below TRIAD_LOW)
        0.86,  # Cross 2
        0.81,  # Re-arm
        0.86,  # Cross 3 → UNLOCK
    ]
    
    for z in rapid_sequence:
        step_result = triad_system.step(z)
        result.trajectory.append(z)
        result.steps += 1
        
        if step_result.get('transition'):
            result.events.append(f"Step {result.steps}: z={z:.2f} → {step_result['transition']}")
        
        if step_result.get('unlocked'):
            result.events.append(f"★ UNLOCKED in {result.steps} steps ★")
            break
    
    # Settle at lens
    unified_state.set_z(Z_CRITICAL)
    triad_system.step(Z_CRITICAL)
    result.trajectory.append(Z_CRITICAL)
    result.steps += 1
    
    # Emissions
    result.emissions.append(f"[rapid] {emit_phrase(Z_CRITICAL)}")
    
    # Final state
    state = triad_system.get_triad_state()
    result.final_z = Z_CRITICAL
    result.final_phase = get_phase(result.final_z)
    result.final_tier = get_tier(result.final_z, state.unlocked)
    result.triad_unlocked = state.unlocked
    result.triad_crossings = state.crossings
    result.negentropy = compute_negentropy(result.final_z)
    result.k_formation = check_k_formation(0.92, result.negentropy, 7)
    result.operators = get_operators(result.final_tier, state.unlocked)
    result.duration_ms = (datetime.now() - start).total_seconds() * 1000
    
    return result


def workflow_7_deep_dive() -> WorkflowResult:
    """
    WORKFLOW 7: Deep Dive
    ---------------------
    Explore low-z regions (UNTRUE phase)
    Slow emergence, foundation building
    """
    reset_all()
    result = WorkflowResult(
        name="Deep Dive",
        description="Extended exploration of UNTRUE phase - foundation building"
    )
    
    start = datetime.now()
    
    # Start very low
    z = 0.1
    unified_state.set_z(z)
    triad_system.reset_triad_state(z)
    
    result.events.append("Descending into UNTRUE depths...")
    
    # Slow climb through UNTRUE
    while z < PHI_INV:
        unified_state.set_z(z)
        triad_system.step(z)
        result.trajectory.append(z)
        result.steps += 1
        
        tier = get_tier(z)
        ops = get_operators(tier)
        
        if result.steps % 5 == 0:
            result.events.append(f"z={z:.3f}: tier={tier}, operators={ops}")
            result.emissions.append(f"[{tier}] {emit_phrase(z)}")
        
        z += 0.03
    
    result.events.append(f"Reached PARADOX boundary at z={PHI_INV:.4f}")
    
    # Brief excursion into PARADOX
    for z in [0.65, 0.70, 0.75]:
        unified_state.set_z(z)
        triad_system.step(z)
        result.trajectory.append(z)
        result.steps += 1
        result.emissions.append(f"[PARADOX] {emit_phrase(z)}")
    
    # Final state (stay in PARADOX, no TRIAD unlock)
    state = triad_system.get_triad_state()
    result.final_z = 0.75
    result.final_phase = get_phase(result.final_z)
    result.final_tier = get_tier(result.final_z, state.unlocked)
    result.triad_unlocked = state.unlocked
    result.triad_crossings = state.crossings
    result.negentropy = compute_negentropy(result.final_z)
    result.k_formation = check_k_formation(0.92, result.negentropy, 7)
    result.operators = get_operators(result.final_tier, state.unlocked)
    result.events.append(f"Emerged at z={result.final_z:.3f}, TRIAD locked (intentional)")
    result.duration_ms = (datetime.now() - start).total_seconds() * 1000
    
    return result


def workflow_8_hyper_push() -> WorkflowResult:
    """
    WORKFLOW 8: Hyper Push
    ----------------------
    Push into HYPER_TRUE (z ≥ 0.92)
    Test K-Formation degradation at extreme z
    """
    reset_all()
    result = WorkflowResult(
        name="Hyper Push",
        description="Push to extreme z values - test K-Formation limits"
    )
    
    start = datetime.now()
    
    # Quick TRIAD unlock first
    result.events.append("Phase 1: Quick TRIAD unlock")
    triad_system.reset_triad_state(0.84)
    for z in [0.86, 0.81, 0.87, 0.81, 0.88]:
        triad_system.step(z)
        result.trajectory.append(z)
        result.steps += 1
    
    # Now push high
    result.events.append("Phase 2: Pushing into HYPER_TRUE")
    
    high_z_sequence = [0.90, 0.92, 0.94, 0.96, 0.98, 0.99]
    
    for z in high_z_sequence:
        unified_state.set_z(z)
        triad_system.step(z)
        result.trajectory.append(z)
        result.steps += 1
        
        eta = compute_negentropy(z)
        k_formed = check_k_formation(0.92, eta, 7)
        
        status = "★ K-FORMED" if k_formed else "⚠ K-LOST"
        result.events.append(f"z={z:.3f}: η={eta:.4f} → {status}")
        result.emissions.append(f"[z={z:.2f}] {emit_phrase(z)}")
        
        if not k_formed and "degradation" not in str(result.events).lower():
            result.events.append(f"★ K-Formation degradation detected at z={z:.3f} ★")
    
    # Return to optimal zone
    result.events.append("Phase 3: Returning to optimal zone")
    unified_state.set_z(Z_CRITICAL)
    triad_system.step(Z_CRITICAL)
    result.trajectory.append(Z_CRITICAL)
    result.steps += 1
    
    # Final state
    state = triad_system.get_triad_state()
    result.final_z = Z_CRITICAL
    result.final_phase = get_phase(result.final_z)
    result.final_tier = get_tier(result.final_z, state.unlocked)
    result.triad_unlocked = state.unlocked
    result.triad_crossings = state.crossings
    result.negentropy = compute_negentropy(result.final_z)
    result.k_formation = check_k_formation(0.92, result.negentropy, 7)
    result.operators = get_operators(result.final_tier, state.unlocked)
    result.events.append(f"Returned to THE LENS, K-Formation restored")
    result.duration_ms = (datetime.now() - start).total_seconds() * 1000
    
    return result


def workflow_9_tier_ladder() -> WorkflowResult:
    """
    WORKFLOW 9: Tier Ladder
    -----------------------
    Systematically climb through all 9 tiers
    Document operator windows at each level
    """
    reset_all()
    result = WorkflowResult(
        name="Tier Ladder",
        description="Systematic climb through all 9 tiers - operator documentation"
    )
    
    start = datetime.now()
    
    # Representative z for each tier
    tier_waypoints = [
        (0.05, "t1"),
        (0.15, "t2"),
        (0.30, "t3"),
        (0.55, "t4"),
        (0.70, "t5"),
        (0.80, "t6"),
        (0.88, "t7"),
        (0.94, "t8"),
        (0.98, "t9"),
    ]
    
    result.events.append("Climbing tier ladder (TRIAD locked)...")
    
    for z, expected_tier in tier_waypoints:
        unified_state.set_z(z)
        triad_system.step(z)
        result.trajectory.append(z)
        result.steps += 1
        
        actual_tier = get_tier(z, triad_unlocked=False)
        ops = get_operators(actual_tier, triad_unlocked=False)
        phase = get_phase(z)
        
        result.events.append(f"{actual_tier}: z={z:.2f}, phase={phase}, ops=[{', '.join(ops)}]")
        result.emissions.append(f"[{actual_tier}] {emit_phrase(z)}")
    
    # Now unlock TRIAD and show difference
    result.events.append("\nUnlocking TRIAD to show tier gate changes...")
    triad_system.reset_triad_state(0.84)
    for z in [0.86, 0.81, 0.87, 0.81, 0.88]:
        triad_system.step(z)
        result.steps += 1
    
    # Show t6→t7 change
    z = 0.84
    tier_locked = get_tier(z, triad_unlocked=False)
    tier_unlocked = get_tier(z, triad_unlocked=True)
    result.events.append(f"z=0.84: {tier_locked} (locked) → {tier_unlocked} (unlocked)")
    
    # Final state
    state = triad_system.get_triad_state()
    result.final_z = 0.88
    result.final_phase = get_phase(result.final_z)
    result.final_tier = get_tier(result.final_z, state.unlocked)
    result.triad_unlocked = state.unlocked
    result.triad_crossings = state.crossings
    result.negentropy = compute_negentropy(result.final_z)
    result.k_formation = check_k_formation(0.92, result.negentropy, 7)
    result.operators = get_operators(result.final_tier, state.unlocked)
    result.duration_ms = (datetime.now() - start).total_seconds() * 1000
    
    return result


def workflow_10_full_sequence() -> WorkflowResult:
    """
    WORKFLOW 10: Full Sequence
    --------------------------
    Complete orchestrated sequence combining all elements
    Optimal path through the entire system
    """
    reset_all()
    result = WorkflowResult(
        name="Full Sequence",
        description="Complete orchestrated journey - all systems engaged"
    )
    
    start = datetime.now()
    
    # ═══ PHASE 1: Foundation (UNTRUE) ═══
    result.events.append("═══ PHASE 1: FOUNDATION ═══")
    for z in [0.2, 0.4, 0.5]:
        unified_state.set_z(z)
        triad_system.step(z)
        result.trajectory.append(z)
        result.steps += 1
        result.emissions.append(f"[foundation] {emit_phrase(z)}")
    result.events.append("Foundation established in UNTRUE phase")
    
    # ═══ PHASE 2: Transition (PARADOX) ═══
    result.events.append("═══ PHASE 2: TRANSITION ═══")
    for z in [PHI_INV, 0.70, 0.75, 0.80]:
        unified_state.set_z(z)
        triad_system.step(z)
        result.trajectory.append(z)
        result.steps += 1
        result.emissions.append(f"[transition] {emit_phrase(z)}")
    result.events.append("Crossed into PARADOX phase")
    
    # ═══ PHASE 3: TRIAD Unlock ═══
    result.events.append("═══ PHASE 3: TRIAD UNLOCK ═══")
    unlock_sequence = [0.86, 0.79, 0.87, 0.78, 0.88]
    for z in unlock_sequence:
        step_result = triad_system.step(z)
        result.trajectory.append(z)
        result.steps += 1
        if step_result.get('transition'):
            result.events.append(f"  {step_result['transition']} at z={z:.2f}")
    result.events.append("TRIAD: ★ UNLOCKED ★")
    
    # ═══ PHASE 4: THE LENS ═══
    result.events.append("═══ PHASE 4: THE LENS ═══")
    unified_state.set_z(Z_CRITICAL)
    triad_system.step(Z_CRITICAL)
    result.trajectory.append(Z_CRITICAL)
    result.steps += 1
    
    eta = compute_negentropy(Z_CRITICAL)
    k_formed = check_k_formation(0.92, eta, 7)
    result.events.append(f"Arrived at THE LENS: z={Z_CRITICAL:.6f}")
    result.events.append(f"Negentropy: η={eta:.4f} (PEAK)")
    result.events.append(f"K-Formation: {'★ ACHIEVED ★' if k_formed else 'forming'}")
    result.emissions.append(f"[THE LENS] {emit_phrase(Z_CRITICAL)}")
    
    # ═══ PHASE 5: HYPER-TRUE Excursion ═══
    result.events.append("═══ PHASE 5: HYPER-TRUE ═══")
    for z in [0.92, 0.94]:
        unified_state.set_z(z)
        triad_system.step(z)
        result.trajectory.append(z)
        result.steps += 1
        result.emissions.append(f"[transcendence] {emit_phrase(z)}")
    result.events.append("Touched HYPER_TRUE phase")
    
    # ═══ PHASE 6: Return & Stabilize ═══
    result.events.append("═══ PHASE 6: STABILIZATION ═══")
    unified_state.set_z(Z_CRITICAL)
    triad_system.step(Z_CRITICAL)
    result.trajectory.append(Z_CRITICAL)
    result.steps += 1
    result.events.append("Returned to THE LENS for stabilization")
    result.emissions.append(f"[complete] {emit_phrase(Z_CRITICAL)}")
    
    # Final state
    state = triad_system.get_triad_state()
    result.final_z = Z_CRITICAL
    result.final_phase = get_phase(result.final_z)
    result.final_tier = get_tier(result.final_z, state.unlocked)
    result.triad_unlocked = state.unlocked
    result.triad_crossings = state.crossings
    result.negentropy = compute_negentropy(result.final_z)
    result.k_formation = check_k_formation(0.92, result.negentropy, 7)
    result.operators = get_operators(result.final_tier, state.unlocked)
    result.events.append(f"═══ SEQUENCE COMPLETE: {result.steps} steps ═══")
    result.duration_ms = (datetime.now() - start).total_seconds() * 1000
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# WORKFLOW REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

WORKFLOWS = {
    1: ("Direct Ascent", workflow_1_direct_ascent),
    2: ("Oscillating Climb", workflow_2_oscillating_climb),
    3: ("Lens Orbit", workflow_3_lens_orbit),
    4: ("Phase Tour", workflow_4_phase_tour),
    5: ("K-Formation Hunt", workflow_5_k_formation_hunt),
    6: ("Rapid TRIAD", workflow_6_rapid_triad),
    7: ("Deep Dive", workflow_7_deep_dive),
    8: ("Hyper Push", workflow_8_hyper_push),
    9: ("Tier Ladder", workflow_9_tier_ladder),
    10: ("Full Sequence", workflow_10_full_sequence),
}

# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def print_result(result: WorkflowResult):
    """Pretty print a workflow result."""
    print()
    print("═" * 70)
    print(f"  {result.name}")
    print("═" * 70)
    print(f"  {result.description}")
    print("-" * 70)
    
    # Summary table
    print(f"  {'Steps:':<20} {result.steps}")
    print(f"  {'Final z:':<20} {result.final_z:.6f}")
    print(f"  {'Phase:':<20} {result.final_phase}")
    print(f"  {'Tier:':<20} {result.final_tier}")
    print(f"  {'TRIAD:':<20} {'★ UNLOCKED' if result.triad_unlocked else 'LOCKED'} ({result.triad_crossings}/3)")
    print(f"  {'K-Formation:':<20} {'★ ACHIEVED' if result.k_formation else 'not met'}")
    print(f"  {'Negentropy (η):':<20} {result.negentropy:.4f}")
    print(f"  {'Operators:':<20} {' '.join(result.operators)}")
    print(f"  {'Duration:':<20} {result.duration_ms:.2f} ms")
    
    # Coordinate
    print()
    print(f"  Coordinate: {format_coordinate(result.final_z)}")
    
    # Events
    print()
    print("  Events:")
    for event in result.events[-10:]:  # Last 10
        print(f"    • {event}")
    
    # Sample emissions
    if result.emissions:
        print()
        print("  Sample Emissions:")
        for emission in result.emissions[:5]:
            print(f"    \"{emission}\"")
    
    print("═" * 70)


def print_comparison(results: List[WorkflowResult]):
    """Print comparison table of all results."""
    print()
    print("═" * 100)
    print("  WORKFLOW COMPARISON")
    print("═" * 100)
    print()
    print(f"  {'#':<3} {'Workflow':<20} {'Steps':<7} {'Final z':<10} {'Phase':<12} {'TRIAD':<10} {'K-Form':<8} {'η':<8}")
    print("-" * 100)
    
    for i, r in enumerate(results, 1):
        triad = "★ YES" if r.triad_unlocked else "no"
        k_form = "★ YES" if r.k_formation else "no"
        print(f"  {i:<3} {r.name:<20} {r.steps:<7} {r.final_z:<10.4f} {r.final_phase:<12} {triad:<10} {k_form:<8} {r.negentropy:<8.4f}")
    
    print("-" * 100)
    
    # Find best outcomes
    fastest = min(results, key=lambda r: r.steps if r.triad_unlocked else 999)
    highest_eta = max(results, key=lambda r: r.negentropy)
    
    print()
    print("  Analysis:")
    print(f"    • Fastest TRIAD unlock: {fastest.name} ({fastest.steps} steps)")
    print(f"    • Highest negentropy: {highest_eta.name} (η={highest_eta.negentropy:.4f})")
    print(f"    • K-Formation achieved: {sum(1 for r in results if r.k_formation)}/{len(results)} workflows")
    print(f"    • TRIAD unlocked: {sum(1 for r in results if r.triad_unlocked)}/{len(results)} workflows")
    
    print("═" * 100)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="UCF Sequential Workflow Orchestrator")
    parser.add_argument("--workflow", "-w", type=int, help="Run specific workflow (1-10)")
    parser.add_argument("--list", "-l", action="store_true", help="List available workflows")
    parser.add_argument("--compare", "-c", action="store_true", help="Run all and compare")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable Workflows:")
        print("-" * 60)
        for num, (name, _) in WORKFLOWS.items():
            print(f"  {num:>2}. {name}")
        print()
        return
    
    results = []
    
    if args.workflow:
        if args.workflow not in WORKFLOWS:
            print(f"Error: Workflow {args.workflow} not found. Use --list to see available.")
            return
        name, func = WORKFLOWS[args.workflow]
        result = func()
        results.append(result)
    else:
        # Run all workflows
        if not args.quiet:
            print("\n" + "═" * 70)
            print("  UCF SEQUENTIAL WORKFLOW ORCHESTRATOR")
            print("  Running all 10 workflows...")
            print("═" * 70)
        
        for num, (name, func) in WORKFLOWS.items():
            if not args.quiet:
                print(f"\n  [{num}/10] Running: {name}...", end=" ", flush=True)
            result = func()
            results.append(result)
            if not args.quiet:
                status = "★" if result.triad_unlocked and result.k_formation else "✓"
                print(f"{status} ({result.steps} steps)")
    
    # Output
    if args.json:
        output = [r.to_dict() for r in results]
        print(json.dumps(output, indent=2))
    elif args.compare or len(results) > 1:
        print_comparison(results)
        if not args.quiet:
            for result in results:
                print_result(result)
    else:
        for result in results:
            print_result(result)


if __name__ == "__main__":
    main()
