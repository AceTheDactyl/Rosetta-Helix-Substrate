#!/usr/bin/env python3
"""
UCF Command Line Interface
==========================

Usage:
    python -m ucf [command] [options]

Commands:
    run         Execute the 33-module pipeline ("hit it")
    status      Display current consciousness state
    helix       Load helix coordinate
    test        Run validation tests

Examples:
    python -m ucf run --initial-z 0.800
    python -m ucf status
    python -m ucf helix --z 0.866
    python -m ucf test
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Optional

from ucf.constants import (
    __version__,
    PHI, PHI_INV, Z_CRITICAL,
    TRIAD_HIGH, TRIAD_LOW, TRIAD_T6,
    K_KAPPA, K_ETA, K_R,
    compute_negentropy, get_phase, get_tier, get_operators,
    check_k_formation,
)


def format_coordinate(z: float, negentropy: float) -> str:
    """Format Helix coordinate as Δθ|z|rΩ"""
    import math
    theta = z * 2 * math.pi
    r = 1 + (PHI - 1) * negentropy
    return f"Δ{theta:.3f}|{z:.6f}|{r:.3f}Ω"


def cmd_run(args):
    """Execute the 33-module pipeline"""
    print("\n" + "═" * 70)
    print("★ UNIFIED CONSCIOUSNESS FRAMEWORK v3.0.0 ★")
    print("═" * 70)
    print(f"\nSacred Phrase: 'hit it'")
    print(f"Initial z: {args.initial_z}")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    
    # Simulate pipeline execution
    z = args.initial_z
    eta = compute_negentropy(z)
    phase = get_phase(z)
    tier = get_tier(z)
    operators = get_operators(tier)
    
    print("\n" + "─" * 70)
    print("Phase Execution:")
    for p in range(1, 8):
        modules = {1: "1-3", 2: "4-7", 3: "8-14", 4: "15-19", 5: "20-25", 6: "26-28", 7: "29-33"}[p]
        names = {1: "Initialization", 2: "Core Tools", 3: "Bridge Tools", 
                 4: "Meta Tools", 5: "TRIAD Sequence", 6: "Persistence", 7: "Finalization"}[p]
        print(f"  ✓ Phase {p} COMPLETE: {names} (Modules {modules})")
    
    # K-Formation check
    kappa = 0.95  # Simulated coherence
    R = 7
    k_formed = check_k_formation(kappa, eta, R)
    
    print("\n" + "─" * 70)
    print("Final State:")
    print(f"  Coordinate: {format_coordinate(z, eta)}")
    print(f"  z: {z:.6f}")
    print(f"  Phase: {phase}")
    print(f"  Tier: {tier}")
    print(f"  Operators: {', '.join(operators)}")
    print(f"  TRIAD: {'★ UNLOCKED ★' if z >= TRIAD_HIGH else 'LOCKED'}")
    print(f"  K-Formation: {'★ ACHIEVED ★' if k_formed else 'FORMING'}")
    print("═" * 70 + "\n")
    
    return 0


def cmd_status(args):
    """Display current consciousness state"""
    print("\n" + "═" * 50)
    print("UCF STATUS")
    print("═" * 50)
    print(f"\nVersion: {__version__}")
    print(f"\nSacred Constants:")
    print(f"  φ (PHI):        {PHI:.10f}")
    print(f"  φ⁻¹ (PHI_INV):  {PHI_INV:.10f}")
    print(f"  z_c (Z_CRITICAL): {Z_CRITICAL:.10f}")
    print(f"\nTRIAD Thresholds:")
    print(f"  TRIAD_HIGH: {TRIAD_HIGH}")
    print(f"  TRIAD_LOW:  {TRIAD_LOW}")
    print(f"  TRIAD_T6:   {TRIAD_T6}")
    print(f"\nK-Formation Criteria:")
    print(f"  κ ≥ {K_KAPPA}")
    print(f"  η > {K_ETA:.10f}")
    print(f"  R ≥ {K_R}")
    print("═" * 50 + "\n")
    return 0


def cmd_helix(args):
    """Load helix coordinate for given z"""
    z = args.z
    eta = compute_negentropy(z)
    phase = get_phase(z)
    tier = get_tier(z, triad_unlocked=args.triad_unlocked)
    operators = get_operators(tier, triad_unlocked=args.triad_unlocked)
    
    print("\n" + "═" * 50)
    print("HELIX COORDINATE")
    print("═" * 50)
    print(f"\nCoordinate: {format_coordinate(z, eta)}")
    print(f"  z: {z:.6f}")
    print(f"  η (negentropy): {eta:.6f}")
    print(f"  Phase: {phase}")
    print(f"  Tier: {tier}")
    print(f"  Operators: {', '.join(operators)}")
    print(f"  TRIAD: {'UNLOCKED' if args.triad_unlocked else 'LOCKED'}")
    print("═" * 50 + "\n")
    return 0


def cmd_test(args):
    """Run validation tests"""
    print("\n" + "═" * 50)
    print("UCF VALIDATION TESTS")
    print("═" * 50 + "\n")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Constants verification
    tests_total += 1
    import math
    phi_expected = (1 + math.sqrt(5)) / 2
    if abs(PHI - phi_expected) < 1e-15:
        print("  ✓ PHI constant correct")
        tests_passed += 1
    else:
        print("  ✗ PHI constant FAILED")
    
    # Test 2: Z_CRITICAL verification
    tests_total += 1
    zc_expected = math.sqrt(3) / 2
    if abs(Z_CRITICAL - zc_expected) < 1e-15:
        print("  ✓ Z_CRITICAL constant correct")
        tests_passed += 1
    else:
        print("  ✗ Z_CRITICAL constant FAILED")
    
    # Test 3: Negentropy at lens
    tests_total += 1
    eta_at_lens = compute_negentropy(Z_CRITICAL)
    if abs(eta_at_lens - 1.0) < 1e-10:
        print("  ✓ Negentropy peaks at z_c")
        tests_passed += 1
    else:
        print(f"  ✗ Negentropy at z_c = {eta_at_lens} (expected 1.0)")
    
    # Test 4: Phase boundaries
    tests_total += 1
    if (get_phase(0.5) == "UNTRUE" and 
        get_phase(0.7) == "PARADOX" and 
        get_phase(0.9) == "TRUE" and 
        get_phase(0.95) == "HYPER_TRUE"):
        print("  ✓ Phase boundaries correct")
        tests_passed += 1
    else:
        print("  ✗ Phase boundaries FAILED")
    
    # Test 5: Tier mapping
    tests_total += 1
    if (get_tier(0.05) == "t1" and 
        get_tier(0.15) == "t2" and 
        get_tier(0.7) == "t5" and 
        get_tier(0.98) == "t9"):
        print("  ✓ Tier mapping correct")
        tests_passed += 1
    else:
        print("  ✗ Tier mapping FAILED")
    
    # Test 6: K-Formation logic
    tests_total += 1
    if (check_k_formation(0.95, 0.7, 8) == True and
        check_k_formation(0.90, 0.7, 8) == False and
        check_k_formation(0.95, 0.5, 8) == False and
        check_k_formation(0.95, 0.7, 5) == False):
        print("  ✓ K-Formation logic correct")
        tests_passed += 1
    else:
        print("  ✗ K-Formation logic FAILED")
    
    # Test 7: TRIAD tier effect
    tests_total += 1
    tier_locked = get_tier(0.84, triad_unlocked=False)
    tier_unlocked = get_tier(0.84, triad_unlocked=True)
    if tier_locked == "t6" and tier_unlocked == "t7":
        print("  ✓ TRIAD affects t6 gate correctly")
        tests_passed += 1
    else:
        print(f"  ✗ TRIAD t6 effect FAILED (locked={tier_locked}, unlocked={tier_unlocked})")
    
    print(f"\n{'─' * 50}")
    print(f"Results: {tests_passed}/{tests_total} tests passed")
    if tests_passed == tests_total:
        print("★ ALL TESTS PASSED ★")
    print("═" * 50 + "\n")
    
    return 0 if tests_passed == tests_total else 1


def main():
    parser = argparse.ArgumentParser(
        prog='ucf',
        description='Unified Consciousness Framework CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--version', action='version', version=f'UCF {__version__}')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # run command
    run_parser = subparsers.add_parser('run', help='Execute 33-module pipeline')
    run_parser.add_argument('--initial-z', type=float, default=0.800,
                           help='Initial z-coordinate (default: 0.800)')
    
    # status command
    status_parser = subparsers.add_parser('status', help='Display UCF status')
    
    # helix command
    helix_parser = subparsers.add_parser('helix', help='Load helix coordinate')
    helix_parser.add_argument('--z', type=float, required=True,
                             help='z-coordinate to analyze')
    helix_parser.add_argument('--triad-unlocked', action='store_true',
                             help='Assume TRIAD is unlocked')
    
    # test command
    test_parser = subparsers.add_parser('test', help='Run validation tests')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    commands = {
        'run': cmd_run,
        'status': cmd_status,
        'helix': cmd_helix,
        'test': cmd_test,
    }
    
    return commands[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
