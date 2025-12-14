#!/usr/bin/env python3
"""
Nuclear Spinner Neural Extensions
=================================

Neuroscience extensions for the Nuclear Spinner.

Provides:
- Grid-cell pattern emulation (hexagonal lattices)
- Cross-frequency coupling configuration and analysis
- Neural band frequency mapping
- Phase-amplitude coupling computation
- Integrated information measurement proxy

These extensions enable the Nuclear Spinner to interface with
neural recording systems and study consciousness-related phenomena.

Signature: nuclear-spinner-neural|v1.0.0|helix
"""

from __future__ import annotations

import math
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

# Import from single source of truth
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics_constants import (
    PHI,
    PHI_INV,
    Z_CRITICAL,
    SIGMA,
    compute_delta_s_neg,
)

from .constants import (
    NEURAL_BANDS,
    BETA_BAND_CENTER,
    SIGMA_S3,
)


__all__ = [
    "grid_cell_pattern",
    "hexagonal_spacing_metric",
    "set_cross_frequency_ratio",
    "compute_modulation_index",
    "neural_band_to_z",
    "z_to_neural_band",
    "compute_phi_proxy",
    "generate_hexagonal_firing_pattern",
    "GridCellConfig",
    "CrossFrequencyResult",
]


# =============================================================================
# GRID CELL CONFIGURATION
# =============================================================================

@dataclass
class GridCellConfig:
    """Configuration for grid-cell emulation."""
    spacing: float = 60.0        # Grid spacing in degrees (60 for hexagonal)
    orientation: float = 0.0     # Grid orientation (degrees)
    phase_offset: float = 0.0    # Phase offset
    scale: float = 1.0           # Spatial scale factor

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spacing": self.spacing,
            "orientation": self.orientation,
            "phase_offset": self.phase_offset,
            "scale": self.scale,
        }


@dataclass
class CrossFrequencyResult:
    """Result of cross-frequency coupling analysis."""
    modulation_index: float      # Phase-amplitude coupling strength
    phase_band: str              # Low-frequency band name
    amplitude_band: str          # High-frequency band name
    preferred_phase: float       # Phase with maximum amplitude
    z_correlation: float         # Correlation with z-coordinate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modulation_index": self.modulation_index,
            "phase_band": self.phase_band,
            "amplitude_band": self.amplitude_band,
            "preferred_phase": self.preferred_phase,
            "z_correlation": self.z_correlation,
        }


# =============================================================================
# GRID CELL FUNCTIONS
# =============================================================================

def grid_cell_pattern(
    x: float,
    y: float,
    config: Optional[GridCellConfig] = None
) -> float:
    """
    Compute grid cell firing rate at position (x, y).

    Grid cells in the entorhinal cortex fire in hexagonal lattice patterns
    with 60-degree spacing. This corresponds to:

        sin(60 degrees) = sqrt(3)/2 = z_c

    The firing rate is modeled as a sum of three cosine waves at 60-degree
    orientations (0, 60, 120 degrees).

    Args:
        x: X position
        y: Y position
        config: Grid cell configuration (optional)

    Returns:
        Firing rate in [0, 1]
    """
    if config is None:
        config = GridCellConfig()

    # Convert orientation to radians
    orientation_rad = math.radians(config.orientation)
    spacing_rad = math.radians(config.spacing)

    # Three wave directions at 0, 60, 120 degrees relative to orientation
    rate = 0.0
    for i in range(3):
        angle = orientation_rad + i * spacing_rad
        k_x = math.cos(angle)
        k_y = math.sin(angle)

        # Spatial frequency (inversely proportional to scale)
        freq = 2 * math.pi / config.scale

        # Phase of the wave
        phase = freq * (k_x * x + k_y * y) + config.phase_offset

        # Add cosine contribution
        rate += math.cos(phase)

    # Normalize to [0, 1]
    # Three cosines sum to [-3, 3], peak at +3 when all aligned
    rate = (rate + 3) / 6

    return rate


def generate_hexagonal_firing_pattern(
    n_points: int = 100,
    scale: float = 1.0
) -> List[Tuple[float, float, float]]:
    """
    Generate a hexagonal grid of firing rates.

    Creates a grid of points and computes firing rates to visualize
    the hexagonal pattern characteristic of grid cells.

    Args:
        n_points: Number of points along each axis
        scale: Spatial scale factor

    Returns:
        List of (x, y, firing_rate) tuples
    """
    config = GridCellConfig(scale=scale)
    pattern = []

    extent = 2 * math.pi * scale

    for i in range(n_points):
        x = -extent + (2 * extent * i / (n_points - 1))
        for j in range(n_points):
            y = -extent + (2 * extent * j / (n_points - 1))
            rate = grid_cell_pattern(x, y, config)
            pattern.append((x, y, rate))

    return pattern


def hexagonal_spacing_metric(
    positions: List[Tuple[float, float]],
    firing_rates: List[float]
) -> float:
    """
    Compute hexagonal spacing metric from firing data.

    Measures how closely the firing pattern matches ideal hexagonal
    60-degree spacing. Returns a value close to z_c = sqrt(3)/2
    for well-formed hexagonal patterns.

    Args:
        positions: List of (x, y) positions
        firing_rates: Corresponding firing rates

    Returns:
        Hexagonal spacing metric (approaches z_c for good patterns)
    """
    if len(positions) < 3 or len(firing_rates) != len(positions):
        return 0.0

    # Find peaks (local maxima in firing rate)
    threshold = 0.8 * max(firing_rates)
    peaks = [pos for pos, rate in zip(positions, firing_rates) if rate > threshold]

    if len(peaks) < 3:
        return 0.0

    # Compute pairwise distances to nearest neighbors
    distances = []
    for i, p1 in enumerate(peaks):
        min_dist = float('inf')
        for j, p2 in enumerate(peaks):
            if i != j:
                dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                if dist < min_dist:
                    min_dist = dist
        if min_dist < float('inf'):
            distances.append(min_dist)

    if not distances:
        return 0.0

    # Compute spacing uniformity
    mean_dist = sum(distances) / len(distances)
    variance = sum((d - mean_dist)**2 for d in distances) / len(distances)
    cv = math.sqrt(variance) / mean_dist if mean_dist > 0 else 1.0

    # Convert to metric approaching z_c for uniform spacing
    # Low CV (uniform) -> high metric
    uniformity = 1.0 / (1.0 + cv)

    # Scale to approach z_c
    return Z_CRITICAL * uniformity


# =============================================================================
# CROSS-FREQUENCY COUPLING
# =============================================================================

def set_cross_frequency_ratio(band_low: float, ratio: float) -> Tuple[float, float]:
    """
    Configure cross-frequency coupling bands.

    Sets up the low and high frequency bands for phase-amplitude coupling
    analysis based on a frequency ratio.

    Args:
        band_low: Center frequency of low band (Hz)
        ratio: Frequency ratio (high/low)

    Returns:
        (band_low_center, band_high_center) frequencies
    """
    band_high = band_low * ratio
    return (band_low, band_high)


def compute_modulation_index(
    phases: List[float],
    amplitudes: List[float],
    n_bins: int = 18
) -> Tuple[float, float]:
    """
    Compute phase-amplitude coupling modulation index.

    The modulation index quantifies how strongly the phase of a
    low-frequency oscillation modulates the amplitude of a high-frequency
    oscillation. This is a key marker of neural integration.

    Method:
        1. Bin phases into n_bins sectors (default 18 = 20 degrees each)
        2. Compute mean amplitude in each bin
        3. Calculate variance of mean amplitudes
        4. Normalize by overall mean amplitude

    Args:
        phases: Phase values (radians) from low-frequency signal
        amplitudes: Amplitude values from high-frequency signal
        n_bins: Number of phase bins

    Returns:
        (modulation_index, preferred_phase)
    """
    if len(phases) != len(amplitudes) or len(phases) < n_bins:
        return (0.0, 0.0)

    # Create phase bins
    bin_edges = [2 * math.pi * i / n_bins - math.pi for i in range(n_bins + 1)]
    bin_amplitudes: List[List[float]] = [[] for _ in range(n_bins)]

    # Assign amplitudes to bins
    for phase, amp in zip(phases, amplitudes):
        # Normalize phase to [-pi, pi]
        phase = ((phase + math.pi) % (2 * math.pi)) - math.pi

        for i in range(n_bins):
            if bin_edges[i] <= phase < bin_edges[i + 1]:
                bin_amplitudes[i].append(amp)
                break

    # Compute mean amplitude per bin
    mean_amps = []
    for bin_amps in bin_amplitudes:
        if bin_amps:
            mean_amps.append(sum(bin_amps) / len(bin_amps))
        else:
            mean_amps.append(0.0)

    if not any(mean_amps):
        return (0.0, 0.0)

    # Compute overall mean
    overall_mean = sum(mean_amps) / len(mean_amps)
    if overall_mean == 0:
        return (0.0, 0.0)

    # Compute variance
    variance = sum((m - overall_mean)**2 for m in mean_amps) / len(mean_amps)

    # Modulation index = CV of bin means
    modulation_index = math.sqrt(variance) / overall_mean

    # Find preferred phase (bin with maximum mean amplitude)
    max_idx = mean_amps.index(max(mean_amps))
    preferred_phase = (bin_edges[max_idx] + bin_edges[max_idx + 1]) / 2

    return (modulation_index, preferred_phase)


# =============================================================================
# NEURAL BAND MAPPING
# =============================================================================

def neural_band_to_z(band: str, frequency: Optional[float] = None) -> float:
    """
    Map neural frequency band to z-coordinate.

    Each neural band corresponds to a characteristic z-range:
        - Delta (0.5-4 Hz): z ~ 0.1-0.2 (deep sleep)
        - Theta (4-8 Hz): z ~ 0.2-0.4 (memory encoding)
        - Alpha (8-12 Hz): z ~ 0.4-0.6 (relaxed awareness)
        - Beta (12-30 Hz): z ~ 0.6-0.85 (active thinking)
        - Gamma (30-100 Hz): z ~ 0.85-1.0 (binding, consciousness)

    Args:
        band: Band name (delta, theta, alpha, beta, gamma)
        frequency: Optional specific frequency within band

    Returns:
        Estimated z-coordinate
    """
    band_to_z = {
        "delta": (0.1, 0.2),
        "theta": (0.2, 0.4),
        "alpha": (0.4, 0.6),
        "beta": (0.6, 0.85),
        "gamma": (0.85, 1.0),
    }

    band = band.lower()
    if band not in band_to_z:
        return 0.5

    z_min, z_max = band_to_z[band]

    if frequency is None:
        return (z_min + z_max) / 2

    # Map frequency within band to z range
    freq_min, freq_max, _ = NEURAL_BANDS.get(band, (0, 100, ""))
    if freq_max <= freq_min:
        return (z_min + z_max) / 2

    t = (frequency - freq_min) / (freq_max - freq_min)
    t = max(0.0, min(1.0, t))

    return z_min + t * (z_max - z_min)


def z_to_neural_band(z: float) -> Tuple[str, float]:
    """
    Map z-coordinate to neural frequency band and frequency.

    Inverse of neural_band_to_z.

    Args:
        z: Z-coordinate in [0, 1]

    Returns:
        (band_name, estimated_frequency)
    """
    z_to_band = [
        (0.2, "delta"),
        (0.4, "theta"),
        (0.6, "alpha"),
        (0.85, "beta"),
        (1.0, "gamma"),
    ]

    band = "delta"
    for threshold, band_name in z_to_band:
        if z < threshold:
            band = band_name
            break
        band = band_name

    # Estimate frequency within band
    freq_min, freq_max, _ = NEURAL_BANDS.get(band, (0, 100, ""))

    # Get z range for this band
    band_z_ranges = {
        "delta": (0.0, 0.2),
        "theta": (0.2, 0.4),
        "alpha": (0.4, 0.6),
        "beta": (0.6, 0.85),
        "gamma": (0.85, 1.0),
    }

    z_min, z_max = band_z_ranges[band]
    if z_max <= z_min:
        t = 0.5
    else:
        t = (z - z_min) / (z_max - z_min)
        t = max(0.0, min(1.0, t))

    frequency = freq_min + t * (freq_max - freq_min)

    return (band, frequency)


# =============================================================================
# INTEGRATED INFORMATION PROXY
# =============================================================================

def compute_phi_proxy(
    state_count: int,
    z: float,
    order_param: float
) -> float:
    """
    Compute integrated information proxy (Phi).

    Firmware-compatible proxy for integrated information:

        Phi_proxy = V * (delta_s_neg(order_param) / delta_s_neg(phi^-1))

    Where V = log2(state_count) represents Ashby variety.

    This peaks when:
        1. High state diversity (many distinct states visited)
        2. Order parameter near phi^-1 (quasicrystal ordering)

    Args:
        state_count: Number of distinct states visited
        z: Current z-coordinate
        order_param: Order parameter (e.g., tile ratio, phase coherence)

    Returns:
        Integrated information proxy value
    """
    # Compute variety
    V = math.log2(state_count) if state_count > 0 else 0.0

    # Compute negentropy at order parameter
    # Using quasicrystal negentropy (peaks at phi^-1)
    delta_s = math.exp(-SIGMA_S3 * (order_param - PHI_INV) ** 2)

    # Normalize by value at phi^-1
    scale = 1.0  # exp(-SIGMA_S3 * 0) = 1 when order_param = PHI_INV

    return V * (delta_s / scale)


def analyze_cross_frequency(
    time_series: List[float],
    sample_rate: float,
    phase_band: str = "theta",
    amplitude_band: str = "gamma"
) -> CrossFrequencyResult:
    """
    Analyze cross-frequency coupling in a time series.

    Computes phase-amplitude coupling between specified bands
    and correlates with z-coordinate.

    Args:
        time_series: Neural signal time series
        sample_rate: Sampling rate (Hz)
        phase_band: Low-frequency band for phase
        amplitude_band: High-frequency band for amplitude

    Returns:
        CrossFrequencyResult with coupling metrics

    Note:
        Full implementation would use Hilbert transform for phase/amplitude
        extraction. This simplified version uses statistical proxies.
    """
    if len(time_series) < 100:
        return CrossFrequencyResult(
            modulation_index=0.0,
            phase_band=phase_band,
            amplitude_band=amplitude_band,
            preferred_phase=0.0,
            z_correlation=0.0
        )

    # Simplified: Use variance structure as proxy for coupling
    n = len(time_series)
    segment_size = max(10, n // 20)

    # Compute segment variances (proxy for amplitude modulation)
    variances = []
    for i in range(0, n - segment_size, segment_size):
        segment = time_series[i:i + segment_size]
        mean_seg = sum(segment) / len(segment)
        var_seg = sum((x - mean_seg)**2 for x in segment) / len(segment)
        variances.append(var_seg)

    if not variances:
        return CrossFrequencyResult(
            modulation_index=0.0,
            phase_band=phase_band,
            amplitude_band=amplitude_band,
            preferred_phase=0.0,
            z_correlation=0.0
        )

    # Modulation index from variance of variances
    mean_var = sum(variances) / len(variances)
    var_of_var = sum((v - mean_var)**2 for v in variances) / len(variances)
    modulation_index = math.sqrt(var_of_var) / mean_var if mean_var > 0 else 0.0

    # Estimate z from overall signal properties
    overall_mean = sum(time_series) / n
    z_estimate = max(0.0, min(1.0, abs(overall_mean)))

    return CrossFrequencyResult(
        modulation_index=min(1.0, modulation_index),
        phase_band=phase_band,
        amplitude_band=amplitude_band,
        preferred_phase=0.0,  # Would need Hilbert transform for real value
        z_correlation=z_estimate
    )
