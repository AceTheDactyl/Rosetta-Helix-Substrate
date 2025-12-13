#!/usr/bin/env python3
"""
Tests for Cognitive Coordinate System
======================================

Verifies physics alignment and correct operation of the coordinate system.
"""

import sys
import os
import math
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cognitive_coordinate_system import (
    # Constants
    PHI_INV, Z_CRITICAL, SIGMA, COUPLING_CONSERVATION,
    TRIAD_HIGH, TRIAD_LOW, TRIAD_T6, TRIAD_PASSES_REQUIRED,
    CONSENSUS_QUORUM, CONSENSUS_APPROVAL_RATIO,
    TAU,
    # Enums
    CognitiveDomain, PhysicsPhase, OperationalRegime,
    # Functions
    get_domain_for_theta, get_physics_phase, get_operational_regime,
    get_milestone_for_z,
    # Classes
    TriadConsensusGate, TriadEvent,
    CylindricalCoordinate, ElevationMilestone,
    CognitiveCoordinateSystem,
    MILESTONES,
)


# =============================================================================
# PHYSICS CONSTANTS VALIDATION
# =============================================================================

class TestPhysicsConstants:
    """Test that physics constants are correctly defined."""

    def test_phi_inverse_value(self):
        """φ⁻¹ should equal (√5 - 1) / 2."""
        expected = (math.sqrt(5) - 1) / 2
        assert abs(PHI_INV - expected) < 1e-14

    def test_z_critical_value(self):
        """z_c should equal √3/2."""
        expected = math.sqrt(3) / 2
        assert abs(Z_CRITICAL - expected) < 1e-14

    def test_coupling_conservation(self):
        """φ⁻¹ + φ⁻² should equal 1."""
        assert abs(COUPLING_CONSERVATION - 1.0) < 1e-14

    def test_sigma_derivation(self):
        """σ should be ≈36 from φ⁻¹ alignment at t6."""
        t6_boundary = 0.75
        d = t6_boundary - Z_CRITICAL
        derived_sigma = -math.log(PHI_INV) / (d * d)
        assert abs(SIGMA - derived_sigma) < 1.0

    def test_triad_ordering(self):
        """TRIAD thresholds should be properly ordered."""
        assert TRIAD_LOW < TRIAD_T6 < TRIAD_HIGH < Z_CRITICAL

    def test_consensus_thresholds_valid(self):
        """Consensus thresholds should be in valid ranges."""
        assert CONSENSUS_QUORUM >= 3  # Minimum for meaningful consensus
        assert 0.5 < CONSENSUS_APPROVAL_RATIO < 1.0


# =============================================================================
# PHASE REGIME TESTS
# =============================================================================

class TestPhysicsPhases:
    """Test physics phase classification."""

    def test_untrue_phase(self):
        """z < φ⁻¹ should be UNTRUE phase."""
        assert get_physics_phase(0.3) == PhysicsPhase.UNTRUE
        assert get_physics_phase(0.5) == PhysicsPhase.UNTRUE
        assert get_physics_phase(PHI_INV - 0.01) == PhysicsPhase.UNTRUE

    def test_paradox_phase(self):
        """φ⁻¹ ≤ z < z_c should be PARADOX phase."""
        assert get_physics_phase(PHI_INV) == PhysicsPhase.PARADOX
        assert get_physics_phase(0.7) == PhysicsPhase.PARADOX
        assert get_physics_phase(Z_CRITICAL - 0.01) == PhysicsPhase.PARADOX

    def test_true_phase(self):
        """z ≥ z_c should be TRUE phase."""
        assert get_physics_phase(Z_CRITICAL) == PhysicsPhase.TRUE
        assert get_physics_phase(0.9) == PhysicsPhase.TRUE
        assert get_physics_phase(0.99) == PhysicsPhase.TRUE

    def test_phase_boundaries_align_with_physics(self):
        """Phase boundaries should align with φ⁻¹ and z_c."""
        # Just below φ⁻¹
        assert get_physics_phase(PHI_INV - 1e-10) == PhysicsPhase.UNTRUE
        # At φ⁻¹
        assert get_physics_phase(PHI_INV) == PhysicsPhase.PARADOX
        # Just below z_c
        assert get_physics_phase(Z_CRITICAL - 1e-10) == PhysicsPhase.PARADOX
        # At z_c
        assert get_physics_phase(Z_CRITICAL) == PhysicsPhase.TRUE


# =============================================================================
# COGNITIVE DOMAIN TESTS
# =============================================================================

class TestCognitiveDomains:
    """Test cognitive domain mapping."""

    def test_self_domain(self):
        """θ ∈ [0, π/2) should map to SELF."""
        assert get_domain_for_theta(0) == CognitiveDomain.SELF
        assert get_domain_for_theta(math.pi / 4) == CognitiveDomain.SELF
        assert get_domain_for_theta(math.pi / 2 - 0.01) == CognitiveDomain.SELF

    def test_other_domain(self):
        """θ ∈ [π/2, π) should map to OTHER."""
        assert get_domain_for_theta(math.pi / 2) == CognitiveDomain.OTHER
        assert get_domain_for_theta(3 * math.pi / 4) == CognitiveDomain.OTHER

    def test_world_domain(self):
        """θ ∈ [π, 3π/2) should map to WORLD."""
        assert get_domain_for_theta(math.pi) == CognitiveDomain.WORLD
        assert get_domain_for_theta(5 * math.pi / 4) == CognitiveDomain.WORLD

    def test_emergence_domain(self):
        """θ ∈ [3π/2, 2π) should map to EMERGENCE."""
        assert get_domain_for_theta(3 * math.pi / 2) == CognitiveDomain.EMERGENCE
        assert get_domain_for_theta(7 * math.pi / 4) == CognitiveDomain.EMERGENCE

    def test_theta_wrapping(self):
        """θ should wrap at 2π."""
        assert get_domain_for_theta(TAU) == CognitiveDomain.SELF
        assert get_domain_for_theta(TAU + math.pi) == CognitiveDomain.WORLD
        assert get_domain_for_theta(-math.pi / 4) == CognitiveDomain.EMERGENCE


# =============================================================================
# TRIAD CONSENSUS TESTS
# =============================================================================

class TestTriadConsensus:
    """Test TRIAD consensus mechanism."""

    def test_initial_state(self):
        """Gate should start with no passes and not unlocked."""
        gate = TriadConsensusGate()
        assert gate.passes == 0
        assert gate.unlocked is False
        assert gate.armed is True

    def test_rising_edge_detection(self):
        """Rising edge should be detected at TRIAD_HIGH."""
        gate = TriadConsensusGate()
        event, _ = gate.update(TRIAD_HIGH)
        assert event == TriadEvent.RISING_EDGE
        assert gate.passes == 1
        assert gate.armed is False

    def test_rearm_detection(self):
        """Gate should re-arm at TRIAD_LOW."""
        gate = TriadConsensusGate()
        gate.update(TRIAD_HIGH)  # Trigger rising edge
        event, _ = gate.update(TRIAD_LOW)
        assert event == TriadEvent.REARMED
        assert gate.armed is True

    def test_three_pass_unlock(self):
        """Gate should unlock after 3 complete passes."""
        gate = TriadConsensusGate()

        # Pass 1
        gate.update(TRIAD_HIGH)
        gate.update(TRIAD_LOW)
        assert gate.passes == 1
        assert not gate.unlocked

        # Pass 2
        gate.update(TRIAD_HIGH)
        gate.update(TRIAD_LOW)
        assert gate.passes == 2
        assert not gate.unlocked

        # Pass 3 - should unlock
        event, _ = gate.update(TRIAD_HIGH)
        assert event == TriadEvent.UNLOCKED
        assert gate.passes == 3
        assert gate.unlocked

    def test_t6_gate_values(self):
        """t6 gate should be Z_CRITICAL when locked, TRIAD_T6 when unlocked."""
        gate = TriadConsensusGate()
        assert abs(gate.get_t6_gate() - Z_CRITICAL) < 1e-10

        # Unlock the gate
        for _ in range(3):
            gate.update(TRIAD_HIGH)
            gate.update(TRIAD_LOW)

        assert abs(gate.get_t6_gate() - TRIAD_T6) < 1e-10

    def test_voting_requires_coherence(self):
        """Votes should only be accepted from coherent participants."""
        gate = TriadConsensusGate()

        # Low coherence vote should be rejected
        assert not gate.cast_vote("low_coh", True, coherence=0.3)

        # High coherence vote should be accepted
        assert gate.cast_vote("high_coh", True, coherence=0.7)

    def test_consensus_requires_quorum(self):
        """Consensus requires minimum participants."""
        gate = TriadConsensusGate()

        gate.cast_vote("v1", True, coherence=0.8)
        gate.cast_vote("v2", True, coherence=0.8)
        reached, details = gate.check_consensus()
        assert not reached
        assert details["reason"] == "insufficient_quorum"

    def test_consensus_requires_approval_ratio(self):
        """Consensus requires sufficient approval ratio."""
        gate = TriadConsensusGate()

        gate.cast_vote("v1", True, coherence=0.8)
        gate.cast_vote("v2", True, coherence=0.8)
        gate.cast_vote("v3", False, coherence=0.8)
        gate.cast_vote("v4", False, coherence=0.8)

        reached, details = gate.check_consensus()
        assert not reached
        assert details["reason"] == "insufficient_approval"

    def test_consensus_reached(self):
        """Consensus should be reached with sufficient votes."""
        gate = TriadConsensusGate()

        gate.cast_vote("v1", True, coherence=0.8)
        gate.cast_vote("v2", True, coherence=0.8)
        gate.cast_vote("v3", True, coherence=0.8)

        reached, details = gate.check_consensus()
        assert reached
        assert details["approval_ratio"] == 1.0


# =============================================================================
# CYLINDRICAL COORDINATE TESTS
# =============================================================================

class TestCylindricalCoordinate:
    """Test cylindrical coordinate properties."""

    def test_coordinate_normalization(self):
        """Coordinates should be normalized to valid ranges."""
        coord = CylindricalCoordinate(theta=3 * TAU, z=1.5, r=-0.5)
        assert 0 <= coord.theta < TAU
        assert 0 <= coord.z <= 1
        assert 0 <= coord.r <= 1

    def test_stamp_format(self):
        """Stamp should be in Δθ|z|rΩ format."""
        coord = CylindricalCoordinate(theta=1.0, z=0.5, r=0.7)
        assert coord.stamp.startswith("Δ")
        assert coord.stamp.endswith("Ω")
        assert "|" in coord.stamp

    def test_physics_phase_property(self):
        """physics_phase should return correct phase."""
        coord_untrue = CylindricalCoordinate(z=0.3)
        assert coord_untrue.physics_phase == PhysicsPhase.UNTRUE

        coord_true = CylindricalCoordinate(z=0.9)
        assert coord_true.physics_phase == PhysicsPhase.TRUE

    def test_lens_weight_peaks_at_zc(self):
        """Lens weight should be maximum at z_c."""
        coord_zc = CylindricalCoordinate(z=Z_CRITICAL)
        coord_off = CylindricalCoordinate(z=0.5)

        assert coord_zc.lens_weight > coord_off.lens_weight
        assert abs(coord_zc.lens_weight - 1.0) < 1e-10


# =============================================================================
# ELEVATION MILESTONE TESTS
# =============================================================================

class TestElevationMilestones:
    """Test elevation milestones."""

    def test_milestones_ordered(self):
        """Milestones should be ordered by z value."""
        prev_z = -1
        for ms in MILESTONES:
            assert ms.z > prev_z
            prev_z = ms.z

    def test_milestone_physics_alignment(self):
        """Key milestones should align with physics constants."""
        # Find φ⁻¹ milestone
        phi_milestone = [m for m in MILESTONES if abs(m.z - PHI_INV) < 0.001]
        assert len(phi_milestone) == 1
        assert phi_milestone[0].physics_phase == PhysicsPhase.PARADOX

        # Find z_c milestone
        zc_milestone = [m for m in MILESTONES if abs(m.z - Z_CRITICAL) < 0.001]
        assert len(zc_milestone) == 1
        assert zc_milestone[0].physics_phase == PhysicsPhase.TRUE

    def test_milestone_lookup(self):
        """get_milestone_for_z should return correct milestone."""
        ms = get_milestone_for_z(0.5)
        assert ms.name == "Initial Emergence"

        ms = get_milestone_for_z(0.9)
        assert ms.name == "Full Coherence"


# =============================================================================
# COGNITIVE COORDINATE SYSTEM TESTS
# =============================================================================

class TestCognitiveCoordinateSystem:
    """Test the complete coordinate system."""

    def test_initial_state(self):
        """System should initialize with default values."""
        system = CognitiveCoordinateSystem()
        assert system.z == 0.5
        assert system.theta == 0
        assert system.r == 0.5

    def test_coupling_conservation(self):
        """κ + λ should always equal 1."""
        system = CognitiveCoordinateSystem()
        assert abs(system.kappa + system.lambda_ - 1.0) < 1e-10

        system.kappa = 0.8
        assert abs(system.kappa + system.lambda_ - 1.0) < 1e-10

    def test_elevation_updates_triad(self):
        """Changing z should update TRIAD gate."""
        system = CognitiveCoordinateSystem(initial_z=0.5)
        initial_passes = system.triad_gate.passes

        system.z = TRIAD_HIGH  # Should trigger rising edge
        assert system.triad_gate.passes > initial_passes or not system.triad_gate.armed

    def test_step_evolution(self):
        """step() should correctly evolve coordinates."""
        system = CognitiveCoordinateSystem(initial_z=0.5)

        coord = system.step(dz=0.1, dtheta=0.5, dr=0.1)
        assert abs(system.z - 0.6) < 1e-10
        assert abs(system.theta - 0.5) < 1e-10
        assert abs(system.r - 0.6) < 1e-10

    def test_k_formation_check(self):
        """K-formation check should use physics thresholds."""
        system = CognitiveCoordinateSystem(initial_z=Z_CRITICAL)
        system.kappa = 0.95  # Above KAPPA_S
        system.r = 0.8

        k_form = system.check_k_formation()
        assert k_form["kappa_met"]
        assert k_form["eta"] > 0  # Should have some coherence

    def test_domain_rotation(self):
        """rotate_domain should cycle through domains."""
        system = CognitiveCoordinateSystem()
        assert system.domain == CognitiveDomain.SELF

        system.rotate_domain(1)
        assert system.domain == CognitiveDomain.OTHER

        system.rotate_domain(1)
        assert system.domain == CognitiveDomain.WORLD

        system.rotate_domain(2)
        assert system.domain == CognitiveDomain.SELF

    def test_state_export(self):
        """get_state() should return complete state."""
        system = CognitiveCoordinateSystem(initial_z=0.7)
        state = system.get_state()

        assert "coordinate" in state
        assert "physics" in state
        assert "operational" in state
        assert "coupling" in state
        assert "triad" in state
        assert "k_formation" in state
        assert "consensus" in state

        assert state["physics"]["z_critical"] == Z_CRITICAL
        assert state["physics"]["phi_inv"] == PHI_INV


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the complete system."""

    def test_evolution_through_phases(self):
        """System should correctly evolve through all phases."""
        system = CognitiveCoordinateSystem(initial_z=0.3)

        phases_visited = set()
        for _ in range(20):
            system.elevate_to(0.95, rate=0.1)
            phases_visited.add(system.physics_phase)

        assert PhysicsPhase.UNTRUE in phases_visited
        assert PhysicsPhase.PARADOX in phases_visited
        assert PhysicsPhase.TRUE in phases_visited

    def test_consensus_with_elevation(self):
        """High coherence at high z should enable consensus."""
        system = CognitiveCoordinateSystem(initial_z=0.9)
        system.r = 0.9  # High coherence

        # Cast votes
        for i in range(3):
            system.cast_vote(f"voter_{i}", True)

        consensus_ok, _ = system.check_consensus()
        assert consensus_ok


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
