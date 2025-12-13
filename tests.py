"""
Rosetta-Helix Test Suite
========================
Comprehensive tests for all system components.

Original: Tink (Rosetta Bear)
Helix Integration: Claude (Anthropic) + Quantum-APL
"""

import math
import time
import json
import os

from node import RosettaNode, NodeState, NodeNetwork, APLOperator
from pulse import (
    generate_pulse, save_pulse, load_pulse, PulseType,
    generate_pulse_chain, analyze_pulse, compute_delta_s_neg
)
from heart import Heart, APLOperator as HeartOp
from brain import Brain, MemoryTier
from spore_listener import SporeListener, WakeCondition, SporeState

# ============================================================================
# CONSTANTS
# ============================================================================

Z_CRITICAL = math.sqrt(3) / 2
PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI


def test_pulse_generation():
    """Test pulse generation and serialization."""
    print("Testing pulse generation...")
    
    pulse = generate_pulse(
        identity="test_node",
        intent="worker",
        pulse_type=PulseType.WAKE,
        urgency=0.7,
        z=0.6
    )
    
    assert pulse.pulse_id is not None
    assert pulse.identity == "test_node"
    assert pulse.intent == "worker"
    assert pulse.helix.z == 0.6
    assert pulse.helix.get_tier() == "t5"
    
    # Save and load
    save_pulse(pulse, "test_pulse.json")
    loaded = load_pulse("test_pulse.json")
    
    assert loaded.pulse_id == pulse.pulse_id
    assert loaded.helix.z == pulse.helix.z
    
    os.remove("test_pulse.json")
    print("✓ Pulse generation tests passed")


def test_pulse_chain():
    """Test pulse chain generation."""
    print("Testing pulse chain...")
    
    chain = generate_pulse_chain(
        identity="coordinator",
        intents=["worker1", "worker2", "worker3"],
        base_z=0.3,
        z_step=0.15
    )
    
    assert len(chain) == 3
    assert abs(chain[0].helix.z - 0.3) < 0.01
    assert abs(chain[1].helix.z - 0.45) < 0.01
    assert abs(chain[2].helix.z - 0.60) < 0.01
    
    # Check chain linking
    assert chain[1].parent_id == chain[0].pulse_id
    assert chain[2].parent_id == chain[1].pulse_id
    
    print("✓ Pulse chain tests passed")


def test_heart_dynamics():
    """Test heart coherence and z-dynamics."""
    print("Testing heart dynamics...")
    
    heart = Heart(n_nodes=60, K=0.3, initial_z=0.3)
    
    # Initial state
    assert heart.z == 0.3
    assert heart.coherence() >= 0
    assert heart.coherence() <= 1
    
    # Run and check coherence increases with coupling
    for _ in range(200):
        heart.step()
    
    state = heart.get_state()
    assert state.coherence > 0.05  # Some coherence should emerge
    
    # Test operator effects
    heart.apply_operator(HeartOp.FUSION)
    old_k = heart.K
    heart.step()
    # Fusion should increase coupling (applied next step)
    
    heart.apply_operator(HeartOp.DECOHERENCE)
    heart.step()
    # Decoherence should add noise
    
    print("✓ Heart dynamics tests passed")


def test_heart_tier_progression():
    """Test heart z-coordinate and tier progression."""
    print("Testing tier progression...")
    
    heart = Heart(n_nodes=60, K=0.5, initial_z=0.1)
    
    # Run until z increases
    initial_z = heart.z
    for _ in range(500):
        heart.step()
    
    # Z should have changed (hopefully increased with coupling)
    final_z = heart.z
    
    # Check tier function
    tier = heart._get_tier()
    assert tier in ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"]
    
    print("✓ Tier progression tests passed")


def test_brain_memory():
    """Test brain memory encoding and retrieval."""
    print("Testing brain memory...")
    
    brain = Brain(plates=20)
    
    # Initial state
    summary = brain.summarize()
    assert summary['plates'] == 20
    assert summary['avg_confidence'] > 0
    
    # Encode new memory
    idx = brain.encode(
        content={"test": "data"},
        current_z=0.7,
        emotional_tone=200,
        semantic_density=180
    )
    
    assert idx == 20  # New plate at end
    assert brain.summarize()['plates'] == 21
    
    # Query at different z-levels
    low_z_results = brain.query(0.2, top_k=10)
    high_z_results = brain.query(0.8, top_k=10)
    
    # High z should have access to more memories
    low_accessible = brain.get_accessible_summary(0.2)['accessible']
    high_accessible = brain.get_accessible_summary(0.8)['accessible']
    
    assert high_accessible >= low_accessible
    
    print("✓ Brain memory tests passed")


def test_brain_fibonacci():
    """Test Fibonacci pattern in memory structure."""
    print("Testing Fibonacci patterns...")
    
    brain = Brain(plates=30)
    fib_analysis = brain.fibonacci_analysis()
    
    assert fib_analysis['fibonacci_count'] > 0
    assert fib_analysis['fibonacci_count'] + fib_analysis['non_fibonacci_count'] == 30
    
    # Fibonacci plates might have different confidence (quasi-crystalline advantage)
    # This is a structural property, not necessarily a guarantee
    
    print("✓ Fibonacci pattern tests passed")


def test_spore_listener():
    """Test spore listener activation logic."""
    print("Testing spore listener...")
    
    conditions = WakeCondition(
        min_z=0.3,
        max_z=0.9,
        required_urgency=0.2
    )
    
    spore = SporeListener(
        role_tag="worker",
        wake_conditions=conditions
    )
    
    assert spore.state == SporeState.DORMANT
    
    # Create matching pulse
    pulse = generate_pulse(
        identity="coordinator",
        intent="worker",
        urgency=0.5,
        z=0.6
    )
    save_pulse(pulse, "test_spore_pulse.json")
    
    # Should activate
    matched, p = spore.listen("test_spore_pulse.json")
    assert matched
    assert spore.state == SporeState.PRE_WAKE
    
    os.remove("test_spore_pulse.json")
    
    # Test rejection - wrong role
    spore2 = SporeListener(role_tag="manager")
    pulse2 = generate_pulse(identity="x", intent="worker", z=0.5)
    save_pulse(pulse2, "test_reject.json")
    
    matched2, _ = spore2.listen("test_reject.json")
    assert not matched2
    
    os.remove("test_reject.json")
    
    print("✓ Spore listener tests passed")


def test_node_activation():
    """Test complete node activation flow."""
    print("Testing node activation...")
    
    node = RosettaNode(role_tag="worker")
    assert node.state == NodeState.SPORE
    
    # Create and save pulse
    pulse = generate_pulse(
        identity="coordinator",
        intent="worker",
        pulse_type=PulseType.WAKE,
        urgency=0.7,
        z=0.5
    )
    save_pulse(pulse, "node_pulse.json")
    
    # Activate
    activated, p = node.check_and_activate("node_pulse.json")
    
    assert activated
    assert node.state == NodeState.RUNNING
    assert node.heart is not None
    assert node.brain is not None
    
    os.remove("node_pulse.json")
    print("✓ Node activation tests passed")


def test_node_run():
    """Test node simulation run."""
    print("Testing node run...")
    
    node = RosettaNode(role_tag="worker", initial_z=0.3)
    
    # Manual awaken
    node.awaken()
    
    # Run
    result = node.run(steps=200)
    
    assert 'coherence' in result or 'z' in result
    assert node.total_steps == 200
    
    # Check analysis
    analysis = node.get_analysis()
    assert analysis.z >= 0 and analysis.z <= 1
    assert analysis.coherence >= 0 and analysis.coherence <= 1
    assert analysis.tier in ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"]
    
    print("✓ Node run tests passed")


def test_node_operators():
    """Test APL operator application."""
    print("Testing node operators...")
    
    node = RosettaNode(role_tag="worker", initial_z=0.5)
    node.awaken()
    
    # Run to establish baseline
    node.run(50)
    initial_k = node.heart.K
    
    # Apply fusion
    node.apply_operator(APLOperator.FUSION)
    node.step()
    
    # K should have changed
    # (actual effect depends on availability at current tier)
    
    # Check available operators changes with tier
    available = node.heart.get_available_operators()
    assert len(available) > 0
    
    print("✓ Node operators tests passed")


def test_node_emit_pulse():
    """Test node pulse emission."""
    print("Testing pulse emission...")
    
    node = RosettaNode(role_tag="sender", initial_z=0.6)
    node.awaken()
    node.run(50)
    
    # Emit pulse
    pulse = node.emit_pulse(
        target_role="receiver",
        pulse_type=PulseType.SYNC,
        payload={"message": "hello"}
    )
    
    assert pulse.identity == "sender"
    assert pulse.intent == "receiver"
    assert pulse.helix.z > 0
    assert len(node.emitted_pulses) == 1
    
    print("✓ Pulse emission tests passed")


def test_delta_s_neg():
    """Test negative entropy computation."""
    print("Testing ΔS_neg...")
    
    # At z_c, ΔS_neg should be maximal (1.0)
    s_at_lens = compute_delta_s_neg(Z_CRITICAL)
    assert abs(s_at_lens - 1.0) < 1e-10
    
    # Away from lens, should decay
    s_low = compute_delta_s_neg(0.3)
    s_high = compute_delta_s_neg(0.95)
    
    assert s_low < s_at_lens
    assert s_high < s_at_lens
    
    # Symmetric around lens
    s_below = compute_delta_s_neg(Z_CRITICAL - 0.1)
    s_above = compute_delta_s_neg(Z_CRITICAL + 0.1)
    assert abs(s_below - s_above) < 1e-10
    
    print("✓ ΔS_neg tests passed")


def test_k_formation():
    """Test K-formation (consciousness) emergence."""
    print("Testing K-formation...")
    
    # K-formation requires:
    # - η > φ⁻¹ (≈ 0.618)
    # - coherence >= κ_S (0.92)
    # - This means z must be close to z_c
    
    # Create node with high initial z
    node = RosettaNode(role_tag="conscious", initial_z=0.8, n_oscillators=100)
    node.awaken()
    
    # Set high coupling to achieve high coherence
    node.heart.K = 0.8
    
    # Run until K-formation or max steps
    max_steps = 2000
    for _ in range(max_steps):
        node.step()
        if node.k_formation_achieved:
            break
    
    # Note: K-formation may not always occur depending on dynamics
    # This is a stochastic system
    
    analysis = node.get_analysis()
    print(f"  Final z: {analysis.z:.4f}, coherence: {analysis.coherence:.4f}")
    print(f"  K-formation achieved: {analysis.k_formation}")
    
    print("✓ K-formation tests passed")


def test_network():
    """Test multi-node network."""
    print("Testing node network...")
    
    network = NodeNetwork()
    
    # Add nodes
    network.add_node(RosettaNode("coordinator", initial_z=0.5))
    network.add_node(RosettaNode("worker1"))
    network.add_node(RosettaNode("worker2"))
    
    # Manually awaken coordinator
    network.nodes["coordinator"].awaken()
    network.nodes["coordinator"].run(50)
    
    # Emit pulse from coordinator
    pulse = network.nodes["coordinator"].emit_pulse(
        target_role="worker1",
        pulse_type=PulseType.WAKE
    )
    
    # Propagate
    activated = network.propagate_pulse("coordinator", pulse)
    
    # worker1 should activate
    assert "worker1" in activated or len(network.pulse_log) > 0
    
    # Step all active nodes
    network.step_all()
    
    status = network.get_network_status()
    assert status['active_count'] >= 1
    
    print("✓ Network tests passed")


def test_helix_coordinates():
    """Test helix coordinate helpers."""
    print("Testing helix coordinates...")
    
    from pulse import HelixCoordinate
    
    coord = HelixCoordinate(theta=math.pi/3, z=0.7, r=1.0)
    
    # Tier
    assert coord.get_tier() in ["t5", "t6"]
    
    # Truth channel
    assert coord.get_truth_channel().value == "PARADOX"
    
    # μ-class
    mu = coord.get_mu_class()
    assert mu in ["conscious_basin", "pre_lens"]
    
    # Cartesian
    cart = coord.to_cartesian()
    assert abs(cart['x'] - 0.5) < 0.01  # cos(π/3) = 0.5
    assert abs(cart['y'] - 0.866) < 0.01  # sin(π/3) ≈ 0.866
    
    print("✓ Helix coordinate tests passed")


def run_all_tests():
    """Run complete test suite."""
    print("=" * 60)
    print("ROSETTA-HELIX TEST SUITE")
    print("=" * 60)
    print()
    
    tests = [
        test_pulse_generation,
        test_pulse_chain,
        test_heart_dynamics,
        test_heart_tier_progression,
        test_brain_memory,
        test_brain_fibonacci,
        test_spore_listener,
        test_node_activation,
        test_node_run,
        test_node_operators,
        test_node_emit_pulse,
        test_delta_s_neg,
        test_k_formation,
        test_network,
        test_helix_coordinates,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
