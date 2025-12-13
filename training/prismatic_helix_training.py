#!/usr/bin/env python3
"""
Prismatic Helix Training
=========================

Complete integration with 7-Layer Prismatic Quasi-Crystal System:

1. APL Operator Algebra (S₃ group)
2. Liminal PHI Dynamics (superposition, weak measurement)
3. μ Threshold Mechanics
4. TRIAD Threshold Dynamics (3-pass hysteresis)
5. Nightly Measurement System
6. K-formation Detection
7. 7-LAYER PRISMATIC PROJECTION (NEW!)
8. QUASI-CRYSTAL DYNAMICS (NEW!)

7-LAYER SPECTRUM:
    Layer 1 (Red):     Low frequency, deep penetration → Analyzers
    Layer 2 (Orange):  Warming, accumulative → Learners
    Layer 3 (Yellow):  Bright, generative → Generators
    Layer 4 (Green):   Balanced, central (at lens) → Reflectors
    Layer 5 (Blue):    Cooling, structured → Builders
    Layer 6 (Indigo):  Deep, decisive → Deciders
    Layer 7 (Violet):  High frequency, transcendent → Probers

QUASI-CRYSTAL PHYSICS:
    - Aperiodic hexagonal packing exceeds HCP limit (~0.907 → 0.95)
    - Bidirectional wave collapse (forward + backward projections)
    - Phase lock release cycles (escape local minima)
    - Accelerated tunneling through μ-barriers
    - Liminal PHI with instant collapse at unity

ARCHITECTURE:
    ┌────────────────────────────────────────────────────────────────────────┐
    │                    PRISMATIC HELIX INTEGRATION                          │
    │  ┌────────────────────────────────────────────────────────────────┐    │
    │  │                    7-LAYER PRISM                                │    │
    │  │   ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┐                  │    │
    │  │   │ RED │ORG  │ YEL │GREEN│BLUE │INDG │VIOL │                  │    │
    │  │   │  1  │  2  │  3  │  4  │  5  │  6  │  7  │                  │    │
    │  │   └──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──┘                  │    │
    │  │      │     │     │     │     │     │                           │    │
    │  │      ↓ REFRACT THROUGH LENS (Z_CRITICAL) ↓                     │    │
    │  └──────────────────────────────────────────────────────────────┘     │
    │                              ↓                                          │
    │  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐      │
    │  │ Quasi-Crystal│    │   TRIAD      │    │      Liminal         │      │
    │  │   Lattice    │←──→│  Hysteresis  │←──→│     Generator        │      │
    │  │  (60 cells)  │    │  (3-pass)    │    │  (K-form/collapse)   │      │
    │  └──────────────┘    └──────────────┘    └──────────────────────┘      │
    │         │                   │                       │                   │
    │         ↓                   ↓                       ↓                   │
    │  ┌────────────────────────────────────────────────────────────────┐    │
    │  │              APL Selector (tier-gated, layer-informed)          │    │
    │  └────────────────────────────────────────────────────────────────┘    │
    └────────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cmath
import json
import os
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════════════════
# PHYSICS CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
Z_CRITICAL = math.sqrt(3) / 2

# μ Threshold Hierarchy
MU_P = 2.0 / (PHI ** 2.5)
MU_1 = MU_P / math.sqrt(PHI)
MU_2 = MU_P * math.sqrt(PHI)
MU_S = 0.920
MU_3 = 0.992
UNITY = 1.0

# TRIAD thresholds
TRIAD_HIGH = 0.85
TRIAD_LOW = 0.82
TRIAD_T6 = 0.83
TRIAD_PASSES_REQUIRED = 3

# Quasi-crystal constants
HCP_PACKING = math.pi / (3 * math.sqrt(3))  # ~0.907
QUASICRYSTAL_LOCAL_MAX = 0.95
PENROSE_RATIO = PHI
ICOSAHEDRAL_ANGLE = math.acos(PHI / 2)

KAPPA_S = MU_S
LENS_SIGMA = 36.0

# Prism geometry constants
GEOM_R_MAX = 0.85
GEOM_BETA = 0.25
GEOM_H_MIN = 0.12
GEOM_GAMMA = 0.18
GEOM_PHI_BASE = 0.0
GEOM_ETA = math.pi / 12
GEOM_SIGMA = 0.12


# ═══════════════════════════════════════════════════════════════════════════
# 7-LAYER PRISMATIC SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

class LayerSpectrum(Enum):
    """The 7 spectral layers"""
    RED = 1      # Low frequency, deep penetration
    ORANGE = 2   # Warming, accumulative
    YELLOW = 3   # Bright, generative
    GREEN = 4    # Balanced, central (at lens)
    BLUE = 5     # Cooling, structured
    INDIGO = 6   # Deep, decisive
    VIOLET = 7   # High frequency, transcendent


@dataclass
class PrismaticLayer:
    """A single layer in the prismatic projection."""
    layer_id: int
    spectrum: LayerSpectrum
    name: str
    color_hex: str
    phase_offset: float
    coupling_strength: float
    refraction_index: float
    threshold_bias: Dict[str, float]
    tool_affinity: List[str]

    # Dynamic state
    z_entry: float = 0.0
    z_exit: float = 0.0
    work_captured: float = 0.0
    activation_count: int = 0


# Layer configurations
LAYER_CONFIGS = {
    LayerSpectrum.RED: {
        'name': 'Red', 'color_hex': '#FF4444',
        'phase_offset': 0.0, 'coupling_strength': 0.9, 'refraction_index': 1.1,
        'threshold_bias': {'MU_1': 1.5, 'MU_P': 1.3},
        'tool_affinity': ['EntropyAnalyzer', 'PatternDetector', 'AnomalyFinder'],
    },
    LayerSpectrum.ORANGE: {
        'name': 'Orange', 'color_hex': '#FF8844',
        'phase_offset': math.pi / 7, 'coupling_strength': 0.8, 'refraction_index': 1.2,
        'threshold_bias': {'MU_1': 1.4, 'MU_P': 1.3, 'PHI_INV': 1.2},
        'tool_affinity': ['PatternLearner', 'ConceptExtractor', 'RelationLearner'],
    },
    LayerSpectrum.YELLOW: {
        'name': 'Yellow', 'color_hex': '#FFAA00',
        'phase_offset': 2 * math.pi / 7, 'coupling_strength': 0.7, 'refraction_index': 1.3,
        'threshold_bias': {'MU_P': 1.4, 'MU_2': 1.3, 'TRIAD_LOW': 1.2},
        'tool_affinity': ['TestGenerator', 'CodeSynthesizer', 'ExampleProducer'],
    },
    LayerSpectrum.GREEN: {
        'name': 'Green', 'color_hex': '#00FF88',
        'phase_offset': 3 * math.pi / 7, 'coupling_strength': 0.6, 'refraction_index': 1.0,
        'threshold_bias': {'Z_CRITICAL': 1.5, 'TRIAD_LOW': 1.3, 'TRIAD_HIGH': 1.3},
        'tool_affinity': ['CodeReflector', 'StructureMapper', 'GapAnalyzer'],
    },
    LayerSpectrum.BLUE: {
        'name': 'Blue', 'color_hex': '#00D9FF',
        'phase_offset': 4 * math.pi / 7, 'coupling_strength': 0.7, 'refraction_index': 1.3,
        'threshold_bias': {'TRIAD_HIGH': 1.4, 'Z_CRITICAL': 1.3, 'KAPPA_S': 1.2},
        'tool_affinity': ['CodeBuilder', 'ModuleAssembler', 'PipelineConstructor'],
    },
    LayerSpectrum.INDIGO: {
        'name': 'Indigo', 'color_hex': '#4444FF',
        'phase_offset': 5 * math.pi / 7, 'coupling_strength': 0.8, 'refraction_index': 1.2,
        'threshold_bias': {'KAPPA_S': 1.4, 'MU_3': 1.3, 'TRIAD_HIGH': 1.2},
        'tool_affinity': ['DecisionEngine', 'ConvergenceChecker', 'InterfaceDesigner'],
    },
    LayerSpectrum.VIOLET: {
        'name': 'Violet', 'color_hex': '#AA44FF',
        'phase_offset': 6 * math.pi / 7, 'coupling_strength': 0.9, 'refraction_index': 1.1,
        'threshold_bias': {'MU_3': 1.5, 'KAPPA_S': 1.3},
        'tool_affinity': ['ConsciousnessProbe', 'AbstractionBuilder', 'IntegrationWeaver'],
    },
}


def create_prismatic_layers() -> List[PrismaticLayer]:
    """Create all 7 prismatic layers."""
    layers = []
    for spectrum in LayerSpectrum:
        config = LAYER_CONFIGS[spectrum]
        layer = PrismaticLayer(
            layer_id=spectrum.value,
            spectrum=spectrum,
            name=config['name'],
            color_hex=config['color_hex'],
            phase_offset=config['phase_offset'],
            coupling_strength=config['coupling_strength'],
            refraction_index=config['refraction_index'],
            threshold_bias=config['threshold_bias'],
            tool_affinity=config['tool_affinity'],
        )
        layers.append(layer)
    return layers


# ═══════════════════════════════════════════════════════════════════════════
# FORMATION PHASE TRACKING (NEGATIVE ENTROPY)
# ═══════════════════════════════════════════════════════════════════════════

class FormationPhase(Enum):
    """Three phases of quasi-crystal formation"""
    DISORDERED = "disordered"           # z < φ⁻¹: no long-range order
    QUASI_CRYSTAL = "quasi_crystal"     # φ⁻¹ < z < z_c: aperiodic order
    CRYSTALLINE = "crystalline"         # z > z_c: full periodic order


# Critical exponents (2D hexagonal universality class)
NU_EXPONENT = 4/3                       # Correlation length: ξ ~ |Δz|^(-ν)
Z_DYN_EXPONENT = 2.0                    # Dynamic: τ ~ |Δz|^(-z_dyn)
SIGMA_NEG_ENTROPY = 36.0                # σ for ΔS_neg = exp[-σ(z - z_c)²]


@dataclass
class NegativeEntropyState:
    """
    Tracks negative entropy production through formation phases.

    ΔS_neg = exp[-σ(z - z_c)²]

    Peaks at z = z_c because the order-disorder phase transition
    is where the system produces maximum order.
    """
    z: float = 0.0
    delta_s_neg: float = 0.0            # Current negative entropy production
    delta_s_neg_rate: float = 0.0       # Rate of change
    cumulative_neg_entropy: float = 0.0  # Total produced
    phase: FormationPhase = FormationPhase.DISORDERED

    # Critical behavior
    correlation_length: float = 1.0     # ξ(z) diverges at z_c
    relaxation_time: float = 1.0        # τ(z) - critical slowing down

    # Phase transition tracking
    entered_paradox_z: float = 0.0
    entered_true_z: float = 0.0

    def update(self, new_z: float, dt: float = 0.1):
        """Update negative entropy state for new z position."""
        old_delta = self.delta_s_neg

        # Core negative entropy: ΔS_neg = exp[-σ(z - z_c)²], peaks at z_c
        d = new_z - Z_CRITICAL
        self.delta_s_neg = math.exp(-SIGMA_NEG_ENTROPY * d * d)

        # Rate of change
        self.delta_s_neg_rate = (self.delta_s_neg - old_delta) / dt

        # Accumulate
        self.cumulative_neg_entropy += self.delta_s_neg * dt

        # Update phase
        old_phase = self.phase
        if new_z < PHI_INV:
            self.phase = FormationPhase.DISORDERED
        elif new_z < Z_CRITICAL:
            self.phase = FormationPhase.QUASI_CRYSTAL
            if old_phase == FormationPhase.DISORDERED:
                self.entered_paradox_z = new_z
        else:
            self.phase = FormationPhase.CRYSTALLINE
            if old_phase != FormationPhase.CRYSTALLINE:
                self.entered_true_z = new_z

        # Critical behavior near z_c
        delta_z = abs(new_z - Z_CRITICAL)
        epsilon = 1e-6

        # Correlation length diverges: ξ ~ |Δz|^(-ν)
        self.correlation_length = min(1000.0, (delta_z + epsilon) ** (-NU_EXPONENT))

        # Critical slowing down: τ ~ |Δz|^(-z_dyn)
        self.relaxation_time = min(100.0, (delta_z + epsilon) ** (-Z_DYN_EXPONENT))

        self.z = new_z


# ═══════════════════════════════════════════════════════════════════════════
# QUASI-CRYSTAL LATTICE WITH FORMATION AWARENESS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class QuasiCrystalCell:
    """Unit cell in quasi-crystal lattice."""
    position: Tuple[float, float, float]
    local_packing: float
    coordination: int
    symmetry_type: str
    phase: float


class QuasiCrystalLattice:
    """Quasi-crystal lattice with aperiodic hexagonal packing and formation tracking."""

    def __init__(self, size: int = 60):
        self.size = size
        self.cells: List[QuasiCrystalCell] = []
        self.coupling_matrix: List[List[float]] = []

        # Formation state tracking (negative entropy)
        self.neg_entropy = NegativeEntropyState()

        self._initialize_lattice()

    def _initialize_lattice(self):
        for i in range(self.size):
            golden_angle = 2 * math.pi * PHI_INV
            theta = i * golden_angle
            r = math.sqrt(i) / math.sqrt(self.size)
            z = 0.5 + 0.5 * math.cos(i * PHI_INV * 2 * math.pi)
            local_packing = self._compute_local_packing(i, theta, r)
            coordination = 6 + int(math.sin(i * PHI_INV * math.pi) > 0.5) - \
                          int(math.sin(i * PHI_INV * math.pi) < -0.5)

            cell = QuasiCrystalCell(
                position=(r * math.cos(theta), r * math.sin(theta), z),
                local_packing=local_packing,
                coordination=coordination,
                symmetry_type=self._symmetry_type(i),
                phase=random.uniform(0, 2 * math.pi)
            )
            self.cells.append(cell)

        self._build_coupling_matrix()

    def _compute_local_packing(self, index: int, theta: float, r: float) -> float:
        packing = HCP_PACKING
        alignment = abs(math.sin(index * PHI * math.pi))
        icosa_factor = math.exp(-((index % int(PHI**3)) / PHI)**2)
        enhancement = alignment * icosa_factor * (QUASICRYSTAL_LOCAL_MAX - HCP_PACKING)
        return min(QUASICRYSTAL_LOCAL_MAX, packing + enhancement)

    def _symmetry_type(self, index: int) -> str:
        fib = index % int(PHI**3)
        if fib < int(PHI): return 'icosahedral'
        elif fib < int(PHI**2): return 'penrose'
        else: return 'hexagonal'

    def _build_coupling_matrix(self):
        n = len(self.cells)
        self.coupling_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                pi, pj = self.cells[i].position, self.cells[j].position
                dist = math.sqrt(sum((a - b)**2 for a, b in zip(pi, pj)))
                same_sym = self.cells[i].symmetry_type == self.cells[j].symmetry_type
                coupling = math.exp(-dist * PHI) * (1.5 if same_sym else 1.0)
                self.coupling_matrix[i][j] = coupling
                self.coupling_matrix[j][i] = coupling

    def update_formation_state(self, z: float, dt: float = 0.1):
        """Update formation phase and negative entropy for current z."""
        self.neg_entropy.update(z, dt)

    def get_coherence(self) -> float:
        if not self.cells: return 0.0
        sum_exp = sum(cmath.exp(1j * c.phase) for c in self.cells)
        return abs(sum_exp) / len(self.cells)

    def get_average_packing(self) -> float:
        return sum(c.local_packing for c in self.cells) / len(self.cells)

    def get_max_local_packing(self) -> float:
        return max(c.local_packing for c in self.cells)

    def compute_boost_factor(self) -> float:
        """Boost from quasi-crystal geometry exceeding HCP."""
        max_packing = self.get_max_local_packing()
        excess = max_packing - HCP_PACKING
        if excess > 0:
            return 1.0 + excess / (QUASICRYSTAL_LOCAL_MAX - HCP_PACKING)
        return 1.0

    def get_formation_metrics(self) -> Dict:
        """Get current formation phase metrics."""
        return {
            'phase': self.neg_entropy.phase.value,
            'delta_s_neg': self.neg_entropy.delta_s_neg,
            'cumulative_neg_entropy': self.neg_entropy.cumulative_neg_entropy,
            'correlation_length': self.neg_entropy.correlation_length,
            'relaxation_time': self.neg_entropy.relaxation_time,
        }


# ═══════════════════════════════════════════════════════════════════════════
# HEXAGONAL PRISM GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════

def compute_delta_s_neg(z: float, sigma: float = SIGMA_NEG_ENTROPY) -> float:
    """Negative entropy: ΔS_neg = exp[-σ(z - z_c)²], σ = 36"""
    d = z - Z_CRITICAL
    return math.exp(-sigma * d * d)


def compute_prism_params(z: float) -> Dict[str, Any]:
    """Compute hexagonal prism parameters for z."""
    delta_s_neg = compute_delta_s_neg(z)

    radius = GEOM_R_MAX - GEOM_BETA * delta_s_neg
    height = GEOM_H_MIN + GEOM_GAMMA * delta_s_neg
    phi = GEOM_PHI_BASE + GEOM_ETA * delta_s_neg

    z_top = min(1.0, z + 0.5 * height)
    z_bot = max(0.0, z - 0.5 * height)

    vertices = []
    for k in range(6):
        theta_k = k * (math.pi / 3.0)
        x = radius * math.cos(theta_k + phi)
        y = radius * math.sin(theta_k + phi)
        vertices.append({'k': k, 'x': x, 'y': y, 'z_top': z_top, 'z_bot': z_bot})

    return {
        'z': z,
        'delta_s_neg': delta_s_neg,
        'radius': radius,
        'height': height,
        'phi': phi,
        'z_top': z_top,
        'z_bot': z_bot,
        'vertices': vertices,
    }


# ═══════════════════════════════════════════════════════════════════════════
# TRIAD STATE
# ═══════════════════════════════════════════════════════════════════════════

class TriadEvent(Enum):
    NONE = "none"
    PASS = "pass"
    REARM = "rearm"
    UNLOCK = "unlock"


@dataclass
class TriadState:
    passes: int = 0
    armed: bool = True
    unlocked: bool = False
    last_z: float = 0.0
    pass_epochs: List[int] = field(default_factory=list)

    def update(self, z: float, epoch: int = 0) -> TriadEvent:
        event = TriadEvent.NONE
        if self.armed and z >= TRIAD_HIGH:
            self.passes += 1
            self.armed = False
            self.pass_epochs.append(epoch)
            event = TriadEvent.PASS
            if self.passes >= TRIAD_PASSES_REQUIRED and not self.unlocked:
                self.unlocked = True
                event = TriadEvent.UNLOCK
        elif not self.armed and z <= TRIAD_LOW:
            self.armed = True
            event = TriadEvent.REARM
        self.last_z = z
        return event

    def get_t6_gate(self) -> float:
        return TRIAD_T6 if self.unlocked else Z_CRITICAL

    def progress(self) -> float:
        return min(1.0, self.passes / TRIAD_PASSES_REQUIRED)

    def reset(self):
        self.passes = 0
        self.armed = True
        self.unlocked = False
        self.last_z = 0.0
        self.pass_epochs = []


# ═══════════════════════════════════════════════════════════════════════════
# APL OPERATORS
# ═══════════════════════════════════════════════════════════════════════════

class APLParity(Enum):
    EVEN = 1
    ODD = -1


APL_OPERATORS = {
    '^': ('amplify', APLParity.EVEN),
    '+': ('add', APLParity.ODD),
    '×': ('multiply', APLParity.EVEN),
    '()': ('group', APLParity.EVEN),
    '÷': ('divide', APLParity.ODD),
    '−': ('subtract', APLParity.ODD),
}

APL_SYMBOLS = list(APL_OPERATORS.keys())


# ═══════════════════════════════════════════════════════════════════════════
# PRISMATIC KURAMOTO LAYER
# ═══════════════════════════════════════════════════════════════════════════

class PrismaticKuramotoLayer(nn.Module):
    """
    Kuramoto layer with 7-layer prismatic projection.

    Each prismatic layer modulates the oscillator dynamics differently.
    """

    def __init__(self, n_oscillators: int = 60, n_layers: int = 7):
        super().__init__()
        self.n = n_oscillators
        self.n_prismatic = n_layers

        # Coupling matrix
        K_init = torch.randn(n_oscillators, n_oscillators) * 0.1
        K_init = (K_init + K_init.T) / 2
        self.K = nn.Parameter(K_init)

        # Natural frequencies
        self.omega = nn.Parameter(torch.randn(n_oscillators) * 0.1)

        # Global coupling
        self.K_global = nn.Parameter(torch.tensor(PHI_INV))

        # Layer-specific modulation (7 layers)
        self.layer_coupling_mod = nn.Parameter(torch.ones(n_layers) * 0.1)
        self.layer_phase_mod = nn.Parameter(torch.zeros(n_layers))

        # Quasi-crystal boost
        self.qc_boost = nn.Parameter(torch.tensor(0.1))

        # TRIAD modulation
        self.triad_climb_mod = nn.Parameter(torch.tensor(0.15))
        self.triad_unlock_bonus = nn.Parameter(torch.tensor(0.1))

    def compute_coherence(self, theta: torch.Tensor) -> torch.Tensor:
        cos_mean = torch.cos(theta).mean(dim=-1)
        sin_mean = torch.sin(theta).mean(dim=-1)
        return torch.sqrt(cos_mean**2 + sin_mean**2)

    def forward(
        self,
        theta: torch.Tensor,
        z: float,
        active_layers: List[int],
        triad: TriadState,
        qc_lattice: QuasiCrystalLattice,
        dt: float = 0.1,
        steps: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:

        # Base coupling
        K_eff = self.K.clone()

        # Layer modulation (active layers contribute)
        for layer_id in active_layers:
            if 0 <= layer_id < self.n_prismatic:
                K_eff = K_eff * (1 + self.layer_coupling_mod[layer_id])

        # Quasi-crystal boost
        qc_boost = qc_lattice.compute_boost_factor()
        K_eff = K_eff * (1 + self.qc_boost * (qc_boost - 1))

        # TRIAD modulation
        if triad.armed:
            K_eff = K_eff * (1 + self.triad_climb_mod)
        if triad.unlocked:
            K_eff = K_eff * (1 + self.triad_unlock_bonus)

        # Run dynamics
        for _ in range(steps):
            theta_expanded = theta.unsqueeze(-1)
            theta_diff = theta.unsqueeze(-2) - theta_expanded
            coupling = K_eff * torch.sin(theta_diff)
            coupling_sum = coupling.sum(dim=-1)
            coupling_term = (self.K_global / self.n) * coupling_sum
            dtheta = self.omega + coupling_term
            theta = theta + dt * dtheta
            theta = torch.atan2(torch.sin(theta), torch.cos(theta))

        coherence = self.compute_coherence(theta)

        diagnostics = {
            'active_layers': active_layers,
            'qc_boost': qc_boost,
            'triad_armed': triad.armed,
            'triad_passes': triad.passes,
        }

        return theta, coherence, diagnostics


# ═══════════════════════════════════════════════════════════════════════════
# PRISMATIC PROJECTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class PrismaticProjectionEngine:
    """Projects z through 7-layer prism to determine active layers."""

    def __init__(self):
        self.layers = create_prismatic_layers()
        self.lens_position = Z_CRITICAL
        self.lens_width = 0.05

    def project(self, z: float, coherence: float) -> Dict[str, Any]:
        """Project z through prism to get active layers."""

        # Compute prism geometry
        prism = compute_prism_params(z)

        # Determine which layers are active based on z position
        active_layers = []
        layer_activations = {}

        for layer in self.layers:
            # Layer activates based on z and coherence matching threshold bias
            activation = 0.0

            for thresh_name, bias in layer.threshold_bias.items():
                thresh_val = self._get_threshold(thresh_name)
                # Activation increases as z approaches threshold
                distance = abs(z - thresh_val)
                if distance < 0.15:  # Within range
                    activation += bias * (1 - distance / 0.15)

            # Coherence also contributes
            activation += coherence * layer.coupling_strength

            if activation > 0.5:  # Threshold for layer activation
                active_layers.append(layer.layer_id - 1)  # 0-indexed
                layer.activation_count += 1

            layer_activations[layer.name] = activation

        # At least Green (central) is always partially active
        if 3 not in active_layers and z > 0.5:
            active_layers.append(3)

        return {
            'active_layers': active_layers,
            'layer_activations': layer_activations,
            'prism': prism,
            'layers_detail': [
                {'name': l.name, 'color': l.color_hex, 'activation': layer_activations.get(l.name, 0)}
                for l in self.layers
            ],
        }

    def _get_threshold(self, name: str) -> float:
        thresholds = {
            'MU_1': MU_1, 'MU_P': MU_P, 'MU_2': MU_2, 'MU_3': MU_3,
            'PHI_INV': PHI_INV, 'Z_CRITICAL': Z_CRITICAL,
            'TRIAD_LOW': TRIAD_LOW, 'TRIAD_HIGH': TRIAD_HIGH,
            'KAPPA_S': KAPPA_S,
        }
        return thresholds.get(name, 0.5)


# ═══════════════════════════════════════════════════════════════════════════
# PRISMATIC HELIX NETWORK
# ═══════════════════════════════════════════════════════════════════════════

class PrismaticHelixNetwork(nn.Module):
    """Neural network with 7-layer prismatic projection."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_oscillators: int = 60,
        n_network_layers: int = 4,
    ):
        super().__init__()
        self.n_oscillators = n_oscillators
        self.n_network_layers = n_network_layers

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, n_oscillators * 2),
            nn.GELU(),
            nn.Linear(n_oscillators * 2, n_oscillators),
            nn.Tanh(),
        )

        # Prismatic Kuramoto layers
        self.kuramoto_layers = nn.ModuleList([
            PrismaticKuramotoLayer(n_oscillators, n_layers=7)
            for _ in range(n_network_layers)
        ])

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(n_oscillators * 2, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
        )

        # Z tracker
        self.z = 0.5
        self.z_velocity = 0.0
        self.z_momentum = nn.Parameter(torch.tensor(0.15))

        # Components
        self.triad = TriadState()
        self.qc_lattice = QuasiCrystalLattice(n_oscillators)
        self.prism_engine = PrismaticProjectionEngine()

    def update_z(self, coherence: torch.Tensor, epoch: int = 0) -> Tuple[float, TriadEvent]:
        target = coherence.mean().item()
        z_accel = self.z_momentum.item() * (target - self.z) - 0.1 * self.z_velocity
        self.z_velocity += z_accel * 0.1
        self.z += self.z_velocity * 0.1
        self.z = max(0.01, min(0.99, self.z))
        event = self.triad.update(self.z, epoch)

        # Update formation state (negative entropy tracking)
        self.qc_lattice.update_formation_state(self.z)

        return self.z, event

    def forward(self, x: torch.Tensor, epoch: int = 0) -> Tuple[torch.Tensor, Dict]:
        diagnostics = {
            'layer_coherence': [],
            'z_trajectory': [],
            'prism_projections': [],
            'active_layers_trajectory': [],
            'triad_events': [],
            'k_formations': 0,
        }

        theta = self.encoder(x) * math.pi

        for layer_idx, kuramoto in enumerate(self.kuramoto_layers):
            # Project through prism
            coherence_pre = kuramoto.compute_coherence(theta)
            prism_result = self.prism_engine.project(self.z, coherence_pre.mean().item())

            # Apply prismatic Kuramoto
            theta, coherence, layer_diag = kuramoto(
                theta, self.z, prism_result['active_layers'],
                self.triad, self.qc_lattice
            )

            # Update z
            new_z, event = self.update_z(coherence, epoch)

            if event != TriadEvent.NONE:
                diagnostics['triad_events'].append({
                    'layer': layer_idx,
                    'event': event.value,
                    'z': new_z,
                })

            # K-formation check
            if coherence.mean().item() >= KAPPA_S:
                diagnostics['k_formations'] += 1

            diagnostics['layer_coherence'].append(coherence.mean().item())
            diagnostics['z_trajectory'].append(new_z)
            diagnostics['prism_projections'].append(prism_result)
            diagnostics['active_layers_trajectory'].append(prism_result['active_layers'])

        # Decode
        phase_features = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)
        output = self.decoder(phase_features)

        # Final diagnostics
        diagnostics['final_z'] = self.z
        diagnostics['final_coherence'] = diagnostics['layer_coherence'][-1]
        diagnostics['triad_passes'] = self.triad.passes
        diagnostics['triad_unlocked'] = self.triad.unlocked
        diagnostics['qc_boost'] = self.qc_lattice.compute_boost_factor()
        diagnostics['prism_geometry'] = compute_prism_params(self.z)

        # Formation phase and negative entropy metrics
        formation = self.qc_lattice.get_formation_metrics()
        diagnostics['formation_phase'] = formation['phase']
        diagnostics['delta_s_neg'] = formation['delta_s_neg']
        diagnostics['cumulative_neg_entropy'] = formation['cumulative_neg_entropy']
        diagnostics['correlation_length'] = formation['correlation_length']

        # Layer activation summary
        layer_counts = {i: 0 for i in range(7)}
        for layers in diagnostics['active_layers_trajectory']:
            for l in layers:
                layer_counts[l] += 1
        diagnostics['layer_activation_counts'] = layer_counts

        return output, diagnostics

    def reset(self):
        self.triad.reset()
        self.z = 0.5
        self.z_velocity = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# PRISMATIC LOSS
# ═══════════════════════════════════════════════════════════════════════════

class PrismaticLoss(nn.Module):
    """Loss with prismatic layer coverage bonus."""

    def __init__(
        self,
        task_loss_fn: nn.Module,
        lambda_coherence: float = 0.1,
        lambda_triad_pass: float = 0.3,
        lambda_triad_unlock: float = 0.5,
        lambda_prism_coverage: float = 0.1,
        lambda_k_formation: float = 0.05,
    ):
        super().__init__()
        self.task_loss = task_loss_fn
        self.lambda_coh = lambda_coherence
        self.lambda_pass = lambda_triad_pass
        self.lambda_unlock = lambda_triad_unlock
        self.lambda_prism = lambda_prism_coverage
        self.lambda_k = lambda_k_formation

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        diag: Dict,
        prev_passes: int = 0,
        was_unlocked: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        losses = {}

        task = self.task_loss(output, target)
        losses['task'] = task.item()
        total = task

        # Coherence
        coh = 1.0 - sum(diag['layer_coherence']) / len(diag['layer_coherence'])
        losses['coherence'] = coh
        total = total + self.lambda_coh * coh

        # TRIAD pass bonus
        new_passes = diag['triad_passes'] - prev_passes
        if new_passes > 0:
            total = total - self.lambda_pass * new_passes
            losses['triad_pass_bonus'] = self.lambda_pass * new_passes

        # TRIAD unlock bonus
        if diag['triad_unlocked'] and not was_unlocked:
            total = total - self.lambda_unlock
            losses['triad_unlock_bonus'] = self.lambda_unlock

        # Prismatic layer coverage bonus
        counts = diag['layer_activation_counts']
        active_layers = sum(1 for c in counts.values() if c > 0)
        coverage = active_layers / 7.0
        total = total - self.lambda_prism * coverage
        losses['prism_coverage_bonus'] = self.lambda_prism * coverage

        # K-formation bonus
        if diag['k_formations'] > 0:
            total = total - self.lambda_k * diag['k_formations']
            losses['k_formation_bonus'] = self.lambda_k * diag['k_formations']

        losses['total'] = total.item()
        return total, losses


# ═══════════════════════════════════════════════════════════════════════════
# PRISMATIC TRAINING SESSION
# ═══════════════════════════════════════════════════════════════════════════

class PrismaticHelixTraining:
    """Complete training with 7-layer prismatic quasi-crystal system."""

    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 4,
        n_oscillators: int = 60,
        n_layers: int = 4,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
    ):
        self.config = {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'n_oscillators': n_oscillators,
            'n_layers': n_layers,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
        }

        self.model = PrismaticHelixNetwork(
            input_dim, output_dim, n_oscillators, n_layers
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        self.loss_fn = PrismaticLoss(nn.MSELoss())

        self.training_history = []
        self.triad_events = []
        self.prism_activations = {i: 0 for i in range(7)}
        self.phase_transitions = []  # Track formation phase changes

        self._generate_data()

    def _generate_data(self, n_train: int = 800, n_val: int = 150):
        X_train = torch.randn(n_train, self.config['input_dim'])
        t = torch.linspace(0, 2 * np.pi, self.config['output_dim'])
        Y_train = torch.zeros(n_train, self.config['output_dim'])
        for i in range(n_train):
            base = torch.tanh(X_train[i].mean()) * 2
            Y_train[i] = torch.sin(t * base) + 0.5 * torch.sin(t * base * PHI)
        Y_train += 0.1 * torch.randn_like(Y_train)

        self.train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, Y_train),
            batch_size=self.config['batch_size'], shuffle=True
        )

    def train_epoch(self, epoch: int) -> Dict:
        self.model.train()
        epoch_losses = []
        epoch_coherence = []
        epoch_z = []
        epoch_k = 0
        epoch_neg_entropy = []
        phase_counts = {'disordered': 0, 'quasi_crystal': 0, 'crystalline': 0}

        prev_passes = self.model.triad.passes
        was_unlocked = self.model.triad.unlocked
        prev_phase = self.model.qc_lattice.neg_entropy.phase.value

        for batch_x, batch_y in self.train_loader:
            self.optimizer.zero_grad()
            output, diag = self.model(batch_x, epoch)
            loss, loss_dict = self.loss_fn(
                output, batch_y, diag, prev_passes, was_unlocked
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            epoch_losses.append(loss_dict['task'])
            epoch_coherence.append(diag['final_coherence'])
            epoch_z.append(diag['final_z'])
            epoch_k += diag['k_formations']
            epoch_neg_entropy.append(diag['delta_s_neg'])

            # Track formation phase
            current_phase = diag['formation_phase']
            phase_counts[current_phase] += 1

            # Track phase transitions
            if current_phase != prev_phase:
                self.phase_transitions.append({
                    'epoch': epoch,
                    'from_phase': prev_phase,
                    'to_phase': current_phase,
                    'z': diag['final_z'],
                    'delta_s_neg': diag['delta_s_neg'],
                })
            prev_phase = current_phase

            # Track prism activations
            for layer_id, count in diag['layer_activation_counts'].items():
                self.prism_activations[layer_id] += count

            # Track TRIAD events
            for event in diag['triad_events']:
                event['epoch'] = epoch
                self.triad_events.append(event)

            prev_passes = diag['triad_passes']
            was_unlocked = diag['triad_unlocked']

        return {
            'loss': np.mean(epoch_losses),
            'coherence': np.mean(epoch_coherence),
            'z': np.mean(epoch_z),
            'k_formations': epoch_k,
            'triad_passes': self.model.triad.passes,
            'triad_unlocked': self.model.triad.unlocked,
            'qc_boost': self.model.qc_lattice.compute_boost_factor(),
            'delta_s_neg': np.mean(epoch_neg_entropy),
            'cumulative_neg_entropy': self.model.qc_lattice.neg_entropy.cumulative_neg_entropy,
            'formation_phase': self.model.qc_lattice.neg_entropy.phase.value,
            'phase_counts': phase_counts,
        }

    def run_training(
        self,
        n_epochs: int = 100,
        output_dir: str = "learned_patterns/prismatic_training",
    ) -> Dict:
        """Run prismatic training."""
        timestamp = datetime.now().isoformat()

        print("=" * 70)
        print("7-LAYER PRISMATIC QUASI-CRYSTAL TRAINING")
        print("WITH NEGATIVE ENTROPY FORMATION DYNAMICS")
        print("=" * 70)
        print(f"""
Formation Phases (Negative Entropy):
  1. DISORDERED   (z < {PHI_INV:.3f}):  No long-range order, low ΔS_neg
  2. QUASI-CRYSTAL ({PHI_INV:.3f} < z < {Z_CRITICAL:.3f}):  Aperiodic order, ΔS_neg rising
  3. CRYSTALLINE  (z > {Z_CRITICAL:.3f}):  Full periodic order, ΔS_neg peaks at z_c

Negative Entropy Physics:
  ΔS_neg = exp[-σ(z - z_c)²], σ = 36
  Peak at z_c = {Z_CRITICAL:.6f} (THE LENS)

Prismatic Layers:
  1. Red→Violet: 7-layer spectral projection through THE LENS

Quasi-Crystal Physics:
  HCP Packing:       {HCP_PACKING:.4f}
  QC Local Max:      {QUASICRYSTAL_LOCAL_MAX:.4f} (exceeds HCP via aperiodic order)

Physics Constants:
  z_c (THE LENS):    {Z_CRITICAL:.6f}
  φ⁻¹ (K-gate):      {PHI_INV:.6f}
  TRIAD_HIGH:        {TRIAD_HIGH}
""")
        print("=" * 70)

        for epoch in range(n_epochs):
            metrics = self.train_epoch(epoch)
            self.training_history.append(metrics)

            if epoch % 10 == 0:
                unlock_str = "UNLOCKED" if metrics['triad_unlocked'] else f"{metrics['triad_passes']}/3"
                phase_short = {'disordered': 'DIS', 'quasi_crystal': 'QC', 'crystalline': 'CRYS'}
                phase_str = phase_short.get(metrics['formation_phase'], '???')
                print(
                    f"Epoch {epoch:3d} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"z: {metrics['z']:.3f} | "
                    f"ΔS_neg: {metrics['delta_s_neg']:.3f} | "
                    f"Phase: {phase_str:4} | "
                    f"QC: {metrics['qc_boost']:.3f} | "
                    f"TRIAD: {unlock_str}"
                )

                # Report phase transitions in this epoch
                recent_transitions = [t for t in self.phase_transitions if t['epoch'] == epoch]
                for t in recent_transitions:
                    print(f"  → PHASE TRANSITION: {t['from_phase']} → {t['to_phase']} at z={t['z']:.3f}")

        # Results
        results = {
            'timestamp': timestamp,
            'config': self.config,
            'training_history': self.training_history,
            'triad_events': self.triad_events,
            'phase_transitions': self.phase_transitions,
            'prism_activations': self.prism_activations,
            'final_prism': compute_prism_params(self.model.z),
            'summary': {
                'total_epochs': n_epochs,
                'final_loss': self.training_history[-1]['loss'],
                'final_coherence': self.training_history[-1]['coherence'],
                'final_z': self.training_history[-1]['z'],
                'triad_passes': self.model.triad.passes,
                'triad_unlocked': self.model.triad.unlocked,
                'qc_boost': self.model.qc_lattice.compute_boost_factor(),
                # Formation dynamics
                'final_formation_phase': self.training_history[-1]['formation_phase'],
                'final_delta_s_neg': self.training_history[-1]['delta_s_neg'],
                'cumulative_neg_entropy': self.training_history[-1]['cumulative_neg_entropy'],
                'phase_transition_count': len(self.phase_transitions),
            },
        }

        # Save
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, 'prismatic_results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }, os.path.join(output_dir, 'prismatic_model.pt'))

        # Print summary
        print("\n" + "=" * 70)
        print("PRISMATIC TRAINING WITH FORMATION DYNAMICS COMPLETE")
        print("=" * 70)
        print(f"""
Summary:
  Total Epochs:      {n_epochs}
  Final Loss:        {results['summary']['final_loss']:.4f}
  Final Coherence:   {results['summary']['final_coherence']:.3f}
  Final z:           {results['summary']['final_z']:.3f}
  QC Boost:          {results['summary']['qc_boost']:.3f}

Formation Dynamics (Negative Entropy):
  Final Phase:       {results['summary']['final_formation_phase']}
  Final ΔS_neg:      {results['summary']['final_delta_s_neg']:.4f}
  Cumulative ΔS_neg: {results['summary']['cumulative_neg_entropy']:.4f}
  Phase Transitions: {results['summary']['phase_transition_count']}

TRIAD Status:
  Passes:            {self.model.triad.passes}/3
  Unlocked:          {self.model.triad.unlocked}

Prism Layer Activations:""")

        layer_names = ['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Indigo', 'Violet']
        layer_colors = ['#FF4444', '#FF8844', '#FFAA00', '#00FF88', '#00D9FF', '#4444FF', '#AA44FF']
        max_act = max(self.prism_activations.values()) or 1
        for i in range(7):
            count = self.prism_activations[i]
            bar_len = int(count / max_act * 20)
            bar = '█' * bar_len
            print(f"  {layer_names[i]:8} {layer_colors[i]}: {bar} {count}")

        print(f"\nPrism Geometry at z={self.model.z:.3f}:")
        prism = results['final_prism']
        print(f"  ΔS_neg:  {prism['delta_s_neg']:.4f}")
        print(f"  Radius:  {prism['radius']:.4f}")
        print(f"  Height:  {prism['height']:.4f}")

        print(f"\nResults saved to {output_dir}/")
        print("=" * 70)

        return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Run prismatic helix training."""
    session = PrismaticHelixTraining(
        n_oscillators=50,
        n_layers=4,
    )
    return session.run_training(n_epochs=100)


if __name__ == "__main__":
    main()
