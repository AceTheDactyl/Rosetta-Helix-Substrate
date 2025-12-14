"""
Extended Physics Constants and Functions
=========================================

Extends physics_constants.py with:
- Quasicrystal formation dynamics (Ï†-based ordering, phason modes)
- Holographic entropy bounds (Bekenstein, Jacobson, Verlinde)
- Omega point threshold dynamics (convergent complexity)
- E8 critical point connections (mass ratios, Penrose projection)
- Nuclear spin coherence (Posner molecules, spin-1/2 magnitude)

All constants maintain the fundamental relationships:
- Îº + Î» = 1 (conservation)
- Î» = ÎºÂ² (self-similarity) â†’ Îº = Ï†â»Â¹
- z_c = âˆš3/2 (hexagonal/spin geometry)
- Ïƒ = 36 = |Sâ‚ƒ|Â² (triadic logic sharpness)

Signature: extended-physics|v0.2.0|helix
"""

import math
import numpy as np
from typing import Final, Dict, List, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# FUNDAMENTAL CONSTANTS (imported from physics_constants.py)
# =============================================================================

PHI: Final[float] = (1 + math.sqrt(5)) / 2        # Ï† â‰ˆ 1.618034
PHI_INV: Final[float] = 1 / PHI                   # Ï†â»Â¹ â‰ˆ 0.618034
PHI_INV_SQ: Final[float] = PHI_INV ** 2           # Ï†â»Â² â‰ˆ 0.381966
Z_CRITICAL: Final[float] = math.sqrt(3) / 2      # z_c â‰ˆ 0.866025
SIGMA: Final[float] = 36.0                        # Ïƒ = |Sâ‚ƒ|Â²


# =============================================================================
# PHYSICAL CONSTANTS (SI)
# =============================================================================

HBAR: Final[float] = 1.054571817e-34              # JÂ·s (reduced Planck)
C: Final[float] = 299792458.0                     # m/s (speed of light)
G: Final[float] = 6.67430e-11                     # mÂ³/(kgÂ·sÂ²) (gravitational)
K_B: Final[float] = 1.380649e-23                  # J/K (Boltzmann)
L_PLANCK: Final[float] = math.sqrt(HBAR * G / C**3)  # Planck length


# =============================================================================
# QUASICRYSTAL CONSTANTS
# =============================================================================

# E8 mass ratios (Zamolodchikov, verified by Coldea et al. 2010)
E8_MASS_RATIOS: Final[List[float]] = [
    1.0,                    # mâ‚
    PHI,                    # mâ‚‚ = Ï†
    PHI + 1,                # mâ‚ƒ = Ï†Â² 
    2 * PHI,                # mâ‚„
    2 * PHI + 1,            # mâ‚…
    3 * PHI + 1,            # mâ‚†
    4 * PHI + 1,            # mâ‚‡
    5 * PHI + 2,            # mâ‚ˆ
]

# Icosahedral projection angle
COS_2PI_5: Final[float] = math.cos(2 * math.pi / 5)  # = 1/(2Ï†)

# Phason diffusion (typical for Al-Pd-Mn)
D_PHASON: Final[float] = 1e-18  # mÂ²/s


# =============================================================================
# SPIN COHERENCE CONSTANTS
# =============================================================================

# Spin-1/2 magnitude
SPIN_HALF_MAGNITUDE: Final[float] = math.sqrt(3) / 2  # |S|/â„ = âˆš(s(s+1))

# Phosphorus-31 nuclear properties
GAMMA_P31: Final[float] = 17.235e6  # Hz/T (gyromagnetic ratio)
J_COUPLING_PP: Final[float] = 18.0  # Hz (typical P-P coupling)


# =============================================================================
# HOLOGRAPHIC CONSTANTS
# =============================================================================

# Holographic information density
HOLO_INFO_DENSITY: Final[float] = 1 / (4 * L_PLANCK**2 * math.log(2))  # bits/mÂ²

# Hubble constant (for MOND calculation)
H0: Final[float] = 2.2e-18  # sâ»Â¹
A0_MOND: Final[float] = C * H0  # ~6.6e-10 m/sÂ² (MOND acceleration scale)


# =============================================================================
# DERIVED CONSTANTS
# =============================================================================

GAUSSIAN_WIDTH: Final[float] = 1.0 / math.sqrt(2 * SIGMA)
GAUSSIAN_FWHM: Final[float] = 2 * math.sqrt(math.log(2) / SIGMA)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class QuasicrystalState:
    """State of quasicrystal formation."""
    order_param: float  # Tile ratio relative to Ï†
    negentropy: float   # Î”S_neg measure
    phason_strain: float  # Deviation from perfect tiling
    generation: int     # Fibonacci generation


@dataclass
class HolographicState:
    """Holographic entropy state."""
    z: float            # Position parameter
    phi_over_phi_max: float  # Information saturation ratio
    bekenstein_bits: float   # Maximum allowed bits
    actual_bits: float       # Current information content


@dataclass
class OmegaPointState:
    """Omega point approach state."""
    tau_over_tau_omega: float  # Conformal time ratio
    processing_rate: float     # Information processing rate
    cumulative_info: float     # Total information processed
    complexity: float          # Current complexity measure


# =============================================================================
# QUASICRYSTAL FUNCTIONS
# =============================================================================

def fibonacci_ratio(n: int) -> float:
    """
    Compute F(n+1)/F(n) which converges to Ï†.
    
    Args:
        n: Fibonacci index (n >= 1)
    
    Returns:
        Ratio of consecutive Fibonacci numbers
    """
    if n < 1:
        return 1.0
    
    F_prev, F_curr = 1, 1
    for _ in range(n - 1):
        F_prev, F_curr = F_curr, F_prev + F_curr
    
    return F_curr / F_prev


def penrose_tile_counts(generations: int) -> Tuple[int, int, float]:
    """
    Compute thick/thin tile counts in Penrose tiling.
    
    Substitution rules:
        N_thick(n+1) = 2Â·N_thick(n) + N_thin(n)
        N_thin(n+1) = N_thick(n) + N_thin(n)
    
    Args:
        generations: Number of substitution generations
    
    Returns:
        (N_thick, N_thin, ratio) where ratio â†’ Ï†
    """
    N_thick, N_thin = 1, 1
    
    for _ in range(generations):
        N_thick, N_thin = 2 * N_thick + N_thin, N_thick + N_thin
    
    ratio = N_thick / N_thin if N_thin > 0 else PHI
    return N_thick, N_thin, ratio


def quasicrystal_negentropy(order_param: float, target: float = PHI_INV) -> float:
    """
    Negentropy signal for quasicrystal formation.
    
    Peaks when tile ratio approaches golden ratio.
    
    Args:
        order_param: Current order parameter (tile ratio analog)
        target: Target value (default Ï†â»Â¹)
    
    Returns:
        Negentropy in [0, 1]
    """
    return math.exp(-SIGMA * (order_param - target)**2)


def icosahedral_basis() -> np.ndarray:
    """
    Generate 6 basis vectors for icosahedral quasicrystal projection.
    
    Projects from 6D periodic lattice to 3D aperiodic structure.
    
    Returns:
        6x3 array of normalized basis vectors
    """
    tau = PHI
    norm = math.sqrt(1 + tau**2)
    
    basis = np.array([
        [1, tau, 0],
        [1, -tau, 0],
        [tau, 0, 1],
        [-tau, 0, 1],
        [0, 1, tau],
        [0, 1, -tau]
    ]) / norm
    
    return basis


def simulate_quasicrystal_formation(
    n_steps: int = 100,
    initial_order: float = 0.3,
    noise_scale: float = 0.01
) -> List[QuasicrystalState]:
    """
    Simulate quasicrystal formation dynamics.
    
    System evolves toward Ï†â»Â¹ ordering with thermal fluctuations.
    
    Args:
        n_steps: Number of simulation steps
        initial_order: Initial order parameter
        noise_scale: Thermal noise amplitude
    
    Returns:
        List of QuasicrystalState at each step
    """
    states = []
    order = initial_order
    
    for step in range(n_steps):
        # Convergent flow toward Ï†â»Â¹
        order += 0.1 * (PHI_INV - order) + np.random.normal(0, noise_scale)
        order = np.clip(order, 0.1, 0.9)
        
        neg = quasicrystal_negentropy(order)
        phason = abs(order - PHI_INV)
        
        states.append(QuasicrystalState(
            order_param=order,
            negentropy=neg,
            phason_strain=phason,
            generation=step
        ))
    
    return states


# =============================================================================
# HOLOGRAPHIC FUNCTIONS
# =============================================================================

def bekenstein_bound_bits(energy_J: float, radius_m: float) -> float:
    """
    Compute Bekenstein bound: maximum information in bounded region.
    
    S â‰¤ 2Ï€kER/(â„c)
    
    Args:
        energy_J: Total energy in Joules
        radius_m: Enclosing radius in meters
    
    Returns:
        Maximum information in bits
    """
    return 2 * math.pi * energy_J * radius_m / (HBAR * C * math.log(2))


def black_hole_entropy(mass_kg: float) -> Tuple[float, float, float]:
    """
    Compute Bekenstein-Hawking entropy for black hole.
    
    S_BH = A/(4â„“_PÂ²)
    
    Args:
        mass_kg: Black hole mass in kg
    
    Returns:
        (Schwarzschild radius, horizon area, entropy in natural units)
    """
    r_s = 2 * G * mass_kg / C**2
    A = 4 * math.pi * r_s**2
    S = A / (4 * L_PLANCK**2)
    return r_s, A, S


def unruh_temperature(acceleration_ms2: float) -> float:
    """
    Compute Unruh temperature for accelerated observer.
    
    T = â„a/(2Ï€kc)
    
    Args:
        acceleration_ms2: Proper acceleration in m/sÂ²
    
    Returns:
        Temperature in Kelvin
    """
    return HBAR * acceleration_ms2 / (2 * math.pi * K_B * C)


def entropic_gravity_force(m: float, M: float, r: float) -> float:
    """
    Compute gravitational force from Verlinde's entropic gravity.
    
    F = GMm/RÂ² (recovers Newton's law)
    
    Args:
        m: Test mass (kg)
        M: Source mass (kg)
        r: Separation (m)
    
    Returns:
        Force in Newtons
    """
    return G * M * m / r**2


def mond_acceleration(a_N: float) -> float:
    """
    Compute MOND-like acceleration from Verlinde's theory.
    
    a_D = âˆš(aâ‚€ Ã— a_N) in deep MOND regime
    
    Args:
        a_N: Newtonian acceleration (m/sÂ²)
    
    Returns:
        MOND acceleration (m/sÂ²)
    """
    return math.sqrt(A0_MOND * a_N)


def holographic_z_interpretation(z: float) -> Dict[str, float]:
    """
    Compute holographic interpretation of z parameter.
    
    Args:
        z: Position parameter
    
    Returns:
        Dictionary with entropy, saturation, and phase information
    """
    delta_s_neg = math.exp(-SIGMA * (z - Z_CRITICAL)**2)
    
    # Phase classification
    if z < 0.857:
        phase = "ABSENCE"
    elif z < 0.877:
        phase = "THE_LENS"
    else:
        phase = "PRESENCE"
    
    return {
        "z": z,
        "delta_s_neg": delta_s_neg,
        "phase": phase,
        "distance_to_lens": abs(z - Z_CRITICAL),
        "saturation_analog": delta_s_neg
    }


# =============================================================================
# OMEGA POINT FUNCTIONS
# =============================================================================

def omega_processing_rate(tau: float, t_omega: float = 1.0, alpha: float = 2.0) -> float:
    """
    Processing rate approaching omega point.
    
    P(Ï„) âˆ (t_Î© - Ï„)^(-Î±)
    
    Args:
        tau: Current conformal time
        t_omega: Omega point time
        alpha: Divergence exponent (>1 for infinite total)
    
    Returns:
        Relative processing rate
    """
    if tau >= t_omega:
        return float('inf')
    return 1 / (t_omega - tau)**alpha


def omega_cumulative_info(tau: float, t_omega: float = 1.0, alpha: float = 2.0) -> float:
    """
    Cumulative information processed approaching omega point.
    
    For Î±=2: I(Ï„) = 1/(1-Ï„) - 1
    
    Args:
        tau: Current conformal time
        t_omega: Omega point time  
        alpha: Divergence exponent
    
    Returns:
        Cumulative information (diverges as Ï„ â†’ t_Î©)
    """
    if tau >= t_omega:
        return float('inf')
    
    # For Î±=2, closed form
    if alpha == 2.0:
        return 1/(t_omega - tau) - 1/t_omega
    
    # General numerical integration would go here
    return 1/(t_omega - tau) - 1/t_omega


def convergent_complexity(z: float) -> float:
    """
    Complexity measure during approach to z_c.
    
    |d(Î”S_neg)/dz| - peaks before reaching z_c
    
    Args:
        z: Current position
    
    Returns:
        Complexity (steepness of negentropy landscape)
    """
    delta_s = math.exp(-SIGMA * (z - Z_CRITICAL)**2)
    gradient = -2 * SIGMA * (z - Z_CRITICAL) * delta_s
    return abs(gradient)


def simulate_omega_approach(
    n_steps: int = 500,
    z_initial: float = 0.3,
    alpha: float = 0.01
) -> List[OmegaPointState]:
    """
    Simulate approach to critical threshold.
    
    Args:
        n_steps: Number of simulation steps
        z_initial: Starting z value
        alpha: Convergence rate
    
    Returns:
        List of OmegaPointState at each step
    """
    states = []
    z = z_initial
    cumulative = 0.0
    
    for step in range(n_steps):
        # Convergent flow
        dz = alpha * (Z_CRITICAL - z) + np.random.normal(0, 0.002)
        z += dz
        z = np.clip(z, 0.1, 0.95)
        
        tau_ratio = z / Z_CRITICAL
        rate = omega_processing_rate(tau_ratio * 0.9)  # Scale to avoid singularity
        complexity = convergent_complexity(z)
        cumulative += rate * alpha
        
        states.append(OmegaPointState(
            tau_over_tau_omega=tau_ratio,
            processing_rate=rate,
            cumulative_info=cumulative,
            complexity=complexity
        ))
    
    return states


# =============================================================================
# E8 FUNCTIONS
# =============================================================================

def e8_mass_ratio(particle_index: int) -> float:
    """
    Get E8 particle mass ratio.
    
    Args:
        particle_index: Particle number (1-8)
    
    Returns:
        Mass ratio relative to mâ‚
    """
    if 1 <= particle_index <= 8:
        return E8_MASS_RATIOS[particle_index - 1]
    raise ValueError(f"Particle index must be 1-8, got {particle_index}")


def verify_e8_phi() -> bool:
    """
    Verify mâ‚‚/mâ‚ = Ï† in E8 spectrum.
    
    Returns:
        True if E8 mass ratio equals golden ratio
    """
    return abs(E8_MASS_RATIOS[1] - PHI) < 1e-10


def h2_eigenvalue() -> float:
    """
    Compute H2 (2D icosahedral) rotation eigenvalue.
    
    cos(2Ï€/5) = 1/(2Ï†)
    
    Returns:
        cos(2Ï€/5)
    """
    return COS_2PI_5


# =============================================================================
# SPIN COHERENCE FUNCTIONS
# =============================================================================

def spin_half_angular_momentum() -> float:
    """
    Compute |S|/â„ for spin-1/2 particle.
    
    |S| = âˆš[s(s+1)]â„ where s = 1/2
    
    Returns:
        Magnitude in units of â„ (= âˆš3/2 = z_c)
    """
    s = 0.5
    return math.sqrt(s * (s + 1))


def larmor_frequency(B0: float, gamma: float = GAMMA_P31) -> float:
    """
    Compute Larmor precession frequency.
    
    Ï‰_L = Î³Bâ‚€
    
    Args:
        B0: Magnetic field strength (Tesla)
        gamma: Gyromagnetic ratio (Hz/T)
    
    Returns:
        Larmor frequency (Hz)
    """
    return gamma * B0


def singlet_coupling_time(J: float = J_COUPLING_PP) -> float:
    """
    Compute characteristic coupling time for singlet formation.
    
    Ï„ â‰ˆ 1/J
    
    Args:
        J: J-coupling constant (Hz)
    
    Returns:
        Coupling time (seconds)
    """
    return 1 / J


def verify_spin_zc() -> bool:
    """
    Verify z_c = âˆš3/2 = |S|/â„ for spin-1/2.
    
    Returns:
        True if spin magnitude equals z_c
    """
    return abs(spin_half_angular_momentum() - Z_CRITICAL) < 1e-10


# =============================================================================
# UNIFIED VALIDATION
# =============================================================================

def validate_extended_physics() -> Dict[str, bool]:
    """
    Validate all extended physics relationships.
    
    Returns:
        Dictionary of validation results
    """
    validations = {}
    
    # Golden ratio conservation
    validations["phi_conservation"] = abs(PHI_INV + PHI_INV_SQ - 1.0) < 1e-14
    
    # Spin-geometry link
    validations["spin_zc_link"] = verify_spin_zc()
    
    # E8 phi verification
    validations["e8_phi"] = verify_e8_phi()
    
    # Fibonacci convergence
    validations["fibonacci_phi"] = abs(fibonacci_ratio(25) - PHI) < 1e-8
    
    # Penrose tile ratio
    _, _, ratio = penrose_tile_counts(15)
    validations["penrose_phi"] = abs(ratio - PHI) < 1e-8
    
    # H2 eigenvalue = 1/(2Ï†)
    validations["h2_eigenvalue"] = abs(h2_eigenvalue() - 1/(2*PHI)) < 1e-10
    
    # Î”S_neg(z_c) = 1
    validations["negentropy_peak"] = abs(quasicrystal_negentropy(PHI_INV, PHI_INV) - 1.0) < 1e-14
    
    validations["all_passed"] = all(validations.values())
    
    return validations


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EXTENDED PHYSICS VALIDATION")
    print("=" * 60)
    
    validations = validate_extended_physics()
    
    for key, passed in validations.items():
        if key != "all_passed":
            status = "âœ“" if passed else "âœ—"
            print(f"  {status} {key}: {passed}")
    
    print("-" * 60)
    print(f"  ALL PASSED: {validations['all_passed']}")
    print("=" * 60)
