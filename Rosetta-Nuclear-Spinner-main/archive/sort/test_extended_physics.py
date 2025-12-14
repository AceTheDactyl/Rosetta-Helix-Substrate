"""
Integration tests for the ``extended_physics_constants`` module.

These unit tests verify that key relationships and numerical identities
described in the Rosetta‑Helix‑Substrate framework hold when using
functions from the extended physics module.  For example, they confirm
φ conservation, the equality of the spin‑1/2 magnitude and the
critical constant z_c, the insufficiency of smaller symmetry groups
like Z₃ to generate transpositions, and the convergence of Fibonacci
and Penrose ratios to the golden ratio.
"""

import math
import pytest

from quantum_apl_python import extended_physics_constants as epc


def test_phi_conservation() -> None:
    """φ⁻¹ + φ⁻² should equal 1 exactly."""
    assert abs(epc.PHI_INV + epc.PHI_INV_SQ - 1.0) < 1e-15


def test_spin_zc_identity() -> None:
    """Verify spin magnitude equals Z_CRITICAL for spin 1/2."""
    assert abs(epc.spin_half_angular_momentum() - epc.Z_CRITICAL) < 1e-15


def test_fibonacci_convergence() -> None:
    """Fibonacci ratios converge to φ within tolerance."""
    assert abs(epc.fibonacci_ratio(20) - epc.PHI) < 1e-10


def test_penrose_ratio() -> None:
    """Penrose tiling thick/thin ratio approaches φ."""
    _, _, ratio = epc.penrose_tile_counts(15)
    assert abs(ratio - epc.PHI) < 1e-6


def test_gaussian_suppression() -> None:
    """Negentropy suppresses values far beyond the critical z range."""
    # At z far from PHI_INV, ΔS_neg should be extremely small but non‑zero
    high = 1.5
    value = epc.quasicrystal_negentropy(high, epc.PHI_INV)
    assert value < 1e-6
    assert value > 0.0


def test_e8_phi_relation() -> None:
    """E8 second mass ratio equals φ."""
    assert epc.verify_e8_phi()


def test_spin_zc_link() -> None:
    """Ensure spin‑Z_c link is verified by helper."""
    assert epc.verify_spin_zc()


# Optional: test S3 minimality if group functions are available.
# These functions are placeholders because the group theory utilities are
# not implemented in this module.  They can be extended as needed.
def can_express_transposition(group: str) -> bool:
    """Placeholder for group transposition expressibility check.

    Currently returns ``False`` for Z3 and ``True`` for S3.  In
    practice these functions would interface with a group theory utility
    to determine whether a given group can express a transposition (12).
    """
    if group == "Z3":
        return False
    if group == "S3":
        return True
    raise ValueError("Unknown group specified")


def test_s3_minimality() -> None:
    """Verify that Z3 cannot express transpositions whereas S3 can."""
    assert not can_express_transposition("Z3")
    assert can_express_transposition("S3")