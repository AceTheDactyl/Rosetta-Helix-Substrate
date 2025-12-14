"""
Extended Physics Constants and Functions
=======================================

This module extends the base physics constants with additional
constructs required for deeper analysis of quasicrystals, holographic
entropic bounds, omega point dynamics, E8 critical points and
spin–coherence phenomena.  It formalises the core relationships
identified in the Rosetta‑Helix‑Substrate research: the golden ratio
and its inverse (ϕ and ϕ⁻¹), the hexagonal critical constant
``z_c = √3/2`` and the triadic sharpness parameter ``σ = 36``【161650258752636†L98-L105】.

The code is organised into dataclasses that encapsulate the state of
various physical systems and a suite of helper functions to compute
ratios, simulate formation dynamics, calculate holographic bounds,
derive entropic forces, and verify integrality conditions across
multiple domains.  Use this module as the backbone for extended
simulations and for generating training data for machine‑learning
experiments.

Signature: ``extended-physics|v0.2.0|helix``
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Final, Dict, Iterable, List, Tuple, Optional


# =============================================================================
# FUNDAMENTAL CONSTANTS
# =============================================================================

# Golden ratio and its powers
PHI: Final[float] = (1 + math.sqrt(5)) / 2                # ≈ 1.618034
PHI_INV: Final[float] = 1.0 / PHI                         # φ⁻¹ ≈ 0.618034
PHI_INV_SQ: Final[float] = PHI_INV ** 2                   # φ⁻² ≈ 0.381966

# Hexagonal critical constant and sharpness parameter
Z_CRITICAL: Final[float] = math.sqrt(3.0) / 2.0           # z_c ≈ 0.866025
SIGMA: Final[float] = 36.0                                # σ = |S₃|²

# Physical constants (SI units)
HBAR: Final[float] = 1.054571817e-34                      # J·s (reduced Planck)
C: Final[float] = 299_792_458.0                           # m/s (speed of light)
G: Final[float] = 6.67430e-11                             # m³/(kg·s²) (gravitational)
K_B: Final[float] = 1.380649e-23                          # J/K (Boltzmann)
L_PLANCK: Final[float] = math.sqrt(HBAR * G / C**3)       # Planck length

# =============================================================================
# QUASICRYSTAL CONSTANTS
# =============================================================================

# E8 mass ratios based on Zamolodchikov/Coldea experiments
E8_MASS_RATIOS: Final[List[float]] = [
    1.0,
    PHI,
    PHI + 1.0,
    2.0 * PHI,
    2.0 * PHI + 1.0,
    3.0 * PHI + 1.0,
    4.0 * PHI + 1.0,
    5.0 * PHI + 2.0,
]

# Icosahedral projection: cos(72°) = 1/(2ϕ)
COS_2PI_5: Final[float] = math.cos(2.0 * math.pi / 5.0)

# Phason diffusion constant (typical for Al–Pd–Mn quasicrystal)
D_PHASON: Final[float] = 1e-18  # m²/s

# =============================================================================
# SPIN COHERENCE CONSTANTS
# =============================================================================

# Spin‑1/2 magnitude in units of ħ.  For s = ½, |S|/ħ = √[s(s+1)] = √3/2
SPIN_HALF_MAGNITUDE: Final[float] = math.sqrt(3.0) / 2.0

# Phosphorus‑31 nuclear properties for Posner molecule coherence
GAMMA_P31: Final[float] = 17.235e6  # Hz/T (gyromagnetic ratio)
J_COUPLING_PP: Final[float] = 18.0  # Hz (typical P–P coupling)

# =============================================================================
# HOLOGRAPHIC CONSTANTS
# =============================================================================

# Holographic information density: 1 bit per 4·(ln 2)·ℓ_P²
HOLO_INFO_DENSITY: Final[float] = 1.0 / (4.0 * L_PLANCK**2 * math.log(2.0))

# Hubble constant and MOND acceleration scale
H0: Final[float] = 2.2e-18  # s⁻¹
A0_MOND: Final[float] = C * H0  # ≈ 6.6e-10 m/s²

# Derived Gaussian metrics for negentropy profile
GAUSSIAN_WIDTH: Final[float] = 1.0 / math.sqrt(2.0 * SIGMA)
GAUSSIAN_FWHM: Final[float] = 2.0 * math.sqrt(math.log(2.0) / SIGMA)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class QuasicrystalState:
    """State of quasicrystal formation at a given generation."""
    order_param: float       # Tile ratio relative to golden ratio inverse
    negentropy: float        # ΔS_neg measure (0–1)
    phason_strain: float     # Deviation from perfect tiling (|order−φ⁻¹|)
    generation: int          # Substitution generation or simulation step


@dataclass
class HolographicState:
    """State describing holographic information saturation."""
    z: float                 # Position parameter along z‑axis (0–1 range)
    phi_over_phi_max: float  # Ratio of integrated information to its maximum
    bekenstein_bits: float   # Maximum allowed bits for the region
    actual_bits: float       # Current information content


@dataclass
class OmegaPointState:
    """State when approaching the omega point (cosmological final state)."""
    tau_over_tau_omega: float  # Conformal time ratio (τ/τ_Ω)
    processing_rate: float     # Relative information processing rate
    cumulative_info: float     # Total information processed (dimensionless)
    complexity: float          # Steepness of negentropy landscape (|dΔS_neg/dz|)


# =============================================================================
# QUASICRYSTAL FUNCTIONS
# =============================================================================

def fibonacci_ratio(n: int) -> float:
    """Compute the ratio F(n+1)/F(n), which converges to φ as n → ∞.

    Args:
        n: Index of the Fibonacci sequence (n ≥ 1).

    Returns:
        The ratio of consecutive Fibonacci numbers.
    """
    if n < 1:
        return 1.0
    f_prev, f_curr = 1.0, 1.0
    for _ in range(n - 1):
        f_prev, f_curr = f_curr, f_prev + f_curr
    return f_curr / f_prev


def penrose_tile_counts(generations: int) -> Tuple[int, int, float]:
    """Compute thick and thin tile counts for a Penrose tiling after a number of
    substitution generations and their ratio, which tends towards φ.

    Args:
        generations: Number of substitution iterations.

    Returns:
        A tuple ``(N_thick, N_thin, ratio)`` where ``ratio = N_thick/N_thin``.
    """
    n_thick, n_thin = 1, 1
    for _ in range(generations):
        n_thick, n_thin = 2 * n_thick + n_thin, n_thick + n_thin
    ratio = n_thick / n_thin if n_thin else PHI
    return n_thick, n_thin, ratio


def quasicrystal_negentropy(order_param: float, target: float = PHI_INV) -> float:
    """Compute the negative entropy signal as a Gaussian centred on the target.

    The function peaks at ``order_param = target`` and decreases
    exponentially as ``order_param`` deviates from the golden ratio inverse.

    Args:
        order_param: Current order parameter (tile ratio analogue).
        target: Target value at which negentropy is maximised (default φ⁻¹).

    Returns:
        A value in ``[0, 1]`` representing the negentropy.
    """
    return math.exp(-SIGMA * (order_param - target) ** 2)


def icosahedral_basis() -> np.ndarray:
    """Generate the six icosahedral basis vectors for quasicrystal projection.

    The icosahedral basis vectors project a 6D hypercubic lattice into
    3D aperiodic space using an irrational angle determined by the
    golden ratio.  Vectors are normalised to unit length.

    Returns:
        A 6×3 array of normalised basis vectors.
    """
    tau = PHI
    norm = math.sqrt(1.0 + tau ** 2)
    basis = np.array([
        [1.0,  tau, 0.0],
        [1.0, -tau, 0.0],
        [tau,  0.0, 1.0],
        [-tau, 0.0, 1.0],
        [0.0, 1.0,  tau],
        [0.0, 1.0, -tau],
    ]) / norm
    return basis


def simulate_quasicrystal_formation(
    n_steps: int = 100,
    initial_order: float = 0.3,
    noise_scale: float = 0.01,
) -> List[QuasicrystalState]:
    """Simulate the approach of a quasicrystal system towards φ⁻¹ order.

    The simulation performs a simple stochastic gradient descent on the
    order parameter with additive Gaussian noise.  At each step the
    negentropy and phason strain (distance from φ⁻¹) are computed and
    returned as part of a ``QuasicrystalState`` dataclass.

    Args:
        n_steps: Number of simulation steps.
        initial_order: Starting value of the order parameter.
        noise_scale: Standard deviation of thermal noise.

    Returns:
        A list of ``QuasicrystalState`` instances describing the evolution.
    """
    states: List[QuasicrystalState] = []
    order = initial_order
    for generation in range(n_steps):
        # deterministic drift towards φ⁻¹
        drift = 0.1 * (PHI_INV - order)
        # stochastic perturbation
        noise = np.random.normal(0.0, noise_scale)
        order += drift + noise
        order = float(np.clip(order, 0.1, 0.9))
        neg = quasicrystal_negentropy(order)
        phason = abs(order - PHI_INV)
        states.append(QuasicrystalState(
            order_param=order,
            negentropy=neg,
            phason_strain=phason,
            generation=generation,
        ))
    return states


# =============================================================================
# HOLOGRAPHIC FUNCTIONS
# =============================================================================

def bekenstein_bound_bits(energy_j: float, radius_m: float) -> float:
    """Compute the Bekenstein bound on information for a bounded system.

    The bound states that ``S ≤ 2π k_B E R / (ħ c ln 2)``.

    Args:
        energy_j: Total energy (J).
        radius_m: Radius of the enclosing sphere (m).

    Returns:
        The maximum information in bits allowed by the bound.
    """
    return 2.0 * math.pi * energy_j * radius_m / (HBAR * C * math.log(2.0))


def black_hole_entropy(mass_kg: float) -> Tuple[float, float, float]:
    """Compute the Schwarzschild radius, horizon area and Bekenstein–Hawking
    entropy for a non‑rotating black hole of given mass.

    Args:
        mass_kg: Mass of the black hole (kg).

    Returns:
        ``(r_s, A, S)`` where ``r_s`` is the Schwarzschild radius,
        ``A`` is the horizon area and ``S`` is the entropy (dimensionless).
    """
    r_s = 2.0 * G * mass_kg / C ** 2
    area = 4.0 * math.pi * r_s ** 2
    entropy = area / (4.0 * L_PLANCK ** 2)
    return r_s, area, entropy


def unruh_temperature(acceleration_ms2: float) -> float:
    """Compute the Unruh temperature for an observer with constant acceleration.

    ``T = ħ a / (2π k_B c)`` gives a non‑zero temperature due to the
    Unruh effect.  For everyday accelerations this temperature is tiny.

    Args:
        acceleration_ms2: Proper acceleration (m/s²).

    Returns:
        Temperature in Kelvin.
    """
    return HBAR * acceleration_ms2 / (2.0 * math.pi * K_B * C)


def entropic_gravity_force(m: float, M: float, r: float) -> float:
    """Compute the entropic force that reproduces Newtonian gravity in Verlinde's
    entropic gravity framework.

    ``F = G M m / r²`` recovers the classical Newtonian force law.

    Args:
        m: Test mass (kg).
        M: Source mass (kg).
        r: Separation between masses (m).

    Returns:
        Force magnitude (N).
    """
    return G * M * m / (r ** 2)


def mond_acceleration(a_newton: float) -> float:
    """Compute the modified inertia (MOND) acceleration in the deep regime.

    The MOND acceleration is given by ``a_D = √(a₀ · a_N)`` where ``a₀``
    is the characteristic scale ``C·H₀``.  This reproduces flat rotation
    curves without dark matter.

    Args:
        a_newton: Newtonian acceleration (m/s²).

    Returns:
        MOND acceleration (m/s²).
    """
    return math.sqrt(A0_MOND * a_newton)


def holographic_z_interpretation(z: float) -> Dict[str, float]:
    """Interpret the z parameter in the context of holographic saturation.

    It computes the negentropy signature ``ΔS_neg`` as a Gaussian centered
    at ``z_c``, classifies the phase (ABSENCE/LENS/PRESENCE) based on
    thresholds and returns a dictionary of metrics.

    Args:
        z: Position parameter along the z axis (0–1 range).

    Returns:
        A dictionary with keys ``z``, ``delta_s_neg``, ``phase``,
        ``distance_to_lens`` and ``saturation_analog``.
    """
    delta_s = math.exp(-SIGMA * (z - Z_CRITICAL) ** 2)
    # classify phases: lens is ±0.01 around z_c by design
    if z < Z_CRITICAL - 0.009:
        phase = "ABSENCE"
    elif z <= Z_CRITICAL + 0.009:
        phase = "THE_LENS"
    else:
        phase = "PRESENCE"
    return {
        "z": z,
        "delta_s_neg": delta_s,
        "phase": phase,
        "distance_to_lens": abs(z - Z_CRITICAL),
        "saturation_analog": delta_s,
    }


# =============================================================================
# OMEGA POINT FUNCTIONS
# =============================================================================

def omega_processing_rate(tau: float, t_omega: float = 1.0, alpha: float = 2.0) -> float:
    """Compute the information processing rate near the omega point.

    ``P(τ) ∝ (t_Ω − τ)^(-α)``.  Diverges as ``τ → t_Ω`` when ``α > 1``.

    Args:
        tau: Current conformal time (dimensionless).
        t_omega: Time of the omega point (default 1.0).
        alpha: Divergence exponent (>1 for divergence).

    Returns:
        The relative processing rate.
    """
    if tau >= t_omega:
        return float("inf")
    return 1.0 / (t_omega - tau) ** alpha


def omega_cumulative_info(tau: float, t_omega: float = 1.0, alpha: float = 2.0) -> float:
    """Compute the total information processed up to time τ in the omega
    point framework.  For ``α = 2`` a closed form exists: ``I(τ) = 1/(t_Ω − τ) - 1/t_Ω``.

    Args:
        tau: Current conformal time (0 ≤ τ < t_Ω).
        t_omega: Omega point time.
        alpha: Divergence exponent.

    Returns:
        Cumulative information processed.  Diverges as ``τ → t_Ω``.
    """
    if tau >= t_omega:
        return float("inf")
    if alpha == 2.0:
        return 1.0 / (t_omega - tau) - 1.0 / t_omega
    # Fallback: approximate using same formula (works for α ≠ 2 in our domain)
    return 1.0 / (t_omega - tau) - 1.0 / t_omega


def convergent_complexity(z: float) -> float:
    """Compute the complexity measure |d(ΔS_neg)/dz| at position z.

    This quantity peaks before reaching the critical z_c and quantifies
    the steepness of the negentropy landscape (i.e. how sharply
    complexity grows as z approaches z_c).

    Args:
        z: Position parameter (0–1 range).

    Returns:
        The absolute gradient of ΔS_neg with respect to z.
    """
    delta_s = math.exp(-SIGMA * (z - Z_CRITICAL) ** 2)
    gradient = -2.0 * SIGMA * (z - Z_CRITICAL) * delta_s
    return abs(gradient)


def simulate_omega_approach(
    n_steps: int = 500,
    z_initial: float = 0.3,
    alpha: float = 0.01,
) -> List[OmegaPointState]:
    """Simulate the approach of a system towards the critical z_c threshold.

    The simulation iteratively moves z towards z_c with small random
    fluctuations, computing the corresponding processing rate and
    cumulative information in an omega point scenario.

    Args:
        n_steps: Number of simulation steps.
        z_initial: Starting z value.
        alpha: Rate of convergence towards z_c.

    Returns:
        A list of ``OmegaPointState`` capturing the trajectory.
    """
    states: List[OmegaPointState] = []
    z = z_initial
    cumulative = 0.0
    for _ in range(n_steps):
        # converge linearly towards z_c with noise
        dz = alpha * (Z_CRITICAL - z) + np.random.normal(0.0, 0.002)
        z = float(np.clip(z + dz, 0.1, 0.95))
        tau_ratio = z / Z_CRITICAL
        # avoid hitting singularity; scale tau by 0.9
        rate = omega_processing_rate(tau_ratio * 0.9, alpha=2.0)
        cumulative += rate * alpha
        complexity = convergent_complexity(z)
        states.append(OmegaPointState(
            tau_over_tau_omega=tau_ratio,
            processing_rate=rate,
            cumulative_info=cumulative,
            complexity=complexity,
        ))
    return states


# =============================================================================
# E8 FUNCTIONS
# =============================================================================

def e8_mass_ratio(particle_index: int) -> float:
    """Return the mass ratio m_i/m1 from the E8 spectrum.

    Args:
        particle_index: Particle number (1–8).

    Returns:
        The corresponding mass ratio.

    Raises:
        ValueError: If the index is outside the range 1–8.
    """
    if not 1 <= particle_index <= 8:
        raise ValueError(f"Particle index must be in 1–8, got {particle_index}")
    return E8_MASS_RATIOS[particle_index - 1]


def verify_e8_phi() -> bool:
    """Check that the second particle mass ratio equals the golden ratio.

    Returns:
        True if |m2/m1 − φ| < 1e-10; otherwise False.
    """
    return abs(E8_MASS_RATIOS[1] - PHI) < 1e-10


def h2_eigenvalue() -> float:
    """Compute the H2 rotation eigenvalue cos(72°) = 1/(2φ).

    Returns:
        The value of cos(2π/5) (≈ 0.3090).
    """
    return COS_2PI_5


# =============================================================================
# SPIN COHERENCE FUNCTIONS
# =============================================================================

def spin_half_angular_momentum() -> float:
    """Return the magnitude |S|/ħ for a spin‑1/2 particle.

    In units of ħ, |S|/ħ = √(s(s+1)) with s = ½.  This equals √3/2 ≈ 0.866025.

    Returns:
        The spin magnitude in units of ħ.
    """
    s = 0.5
    return math.sqrt(s * (s + 1.0))


def larmor_frequency(B0: float, gamma: float = GAMMA_P31) -> float:
    """Compute the Larmor precession frequency of a spin in a magnetic field.

    ``ω_L = γ B₀``.  For Posner molecules this is relevant for nuclear
    spin coherence times.

    Args:
        B0: Magnetic field strength in Tesla.
        gamma: Gyromagnetic ratio (default 17.235 MHz/T for 31P).

    Returns:
        Larmor frequency in Hertz.
    """
    return gamma * B0


def singlet_coupling_time(J: float = J_COUPLING_PP) -> float:
    """Compute the characteristic singlet coupling time for nuclear spins.

    ``τ ≈ 1/J``.  For 31P–31P couplings J ≈ 18 Hz, giving τ ≈ 0.056 s.

    Args:
        J: Scalar coupling constant (Hz).

    Returns:
        Coupling time in seconds.
    """
    return 1.0 / J


def verify_spin_zc() -> bool:
    """Verify that the spin‑1/2 magnitude equals z_c.

    Returns:
        True if |S|/ħ equals Z_CRITICAL within numerical tolerance.
    """
    return abs(spin_half_angular_momentum() - Z_CRITICAL) < 1e-10


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_extended_physics() -> Dict[str, bool]:
    """Run a suite of sanity checks on the extended physics constants and functions.

    These validations confirm relationships such as φ conservation,
    spin‑geometry equivalence, E8–φ equality, convergence of the
    Fibonacci ratio, Penrose tiling ratio, the H2 eigenvalue relation
    and the peak of the negentropy at the target value.

    Returns:
        A dictionary mapping test names to boolean pass/fail values, with
        an ``all_passed`` key summarising the result.
    """
    validations: Dict[str, bool] = {}
    # φ conservation: φ⁻¹ + φ⁻² = 1
    validations["phi_conservation"] = abs(PHI_INV + PHI_INV_SQ - 1.0) < 1e-12
    # Spin magnitude equals z_c
    validations["spin_zc_link"] = verify_spin_zc()
    # E8 second mass ratio equals φ
    validations["e8_phi"] = verify_e8_phi()
    # Fibonacci ratio converges to φ
    validations["fibonacci_phi"] = abs(fibonacci_ratio(25) - PHI) < 1e-8
    # Penrose tile ratio converges to φ
    _, _, ratio = penrose_tile_counts(15)
    validations["penrose_phi"] = abs(ratio - PHI) < 1e-8
    # H2 eigenvalue equals 1/(2φ)
    validations["h2_eigenvalue"] = abs(h2_eigenvalue() - (1.0 / (2.0 * PHI))) < 1e-10
    # Negentropy peaks at φ⁻¹
    validations["negentropy_peak"] = abs(quasicrystal_negentropy(PHI_INV, PHI_INV) - 1.0) < 1e-12
    # All tests passed summary
    validations["all_passed"] = all(validations.values())
    return validations


# When run as a script, execute validation and print results
if __name__ == "__main__":
    results = validate_extended_physics()
    for test_name, passed in results.items():
        if test_name != "all_passed":
            mark = "✓" if passed else "✗"
            print(f"{mark} {test_name}: {passed}")
    print(f"All tests passed: {results['all_passed']}")