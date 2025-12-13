from quantum_apl_python.constants import (
    PHI_INV,
    compute_eta,
    check_k_formation_from_z,
)


def test_eta_gate_examples():
    s090 = compute_eta(0.90)
    s083 = compute_eta(0.83)
    s070 = compute_eta(0.70)

    assert s090 > PHI_INV, f"η(0.90)={s090} ≤ φ⁻¹"
    assert s083 > PHI_INV, f"η(0.83)={s083} ≤ φ⁻¹"
    assert s070 < PHI_INV, f"η(0.70)={s070} ≥ φ⁻¹"


def test_k_formation_from_z_gate():
    assert check_k_formation_from_z(0.95, 0.90, 8) is True
    assert check_k_formation_from_z(0.95, 0.70, 8) is False

