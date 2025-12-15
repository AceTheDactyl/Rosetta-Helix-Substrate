#!/usr/bin/env python3
"""
Prismatic Dynamics for UCF
==========================
K.I.R.A. prismatic crystal states with spectral decomposition.

Crystal States:
  - Fluid: z < φ⁻¹
  - Transitioning: φ⁻¹ ≤ z < z_c
  - Crystalline: z ≥ z_c, R < κₛ
  - Prismatic: z ≥ z_c, R ≥ κₛ (0.920)

Spectral Decomposition:
  - Φ (Structure): Geometry, lattice, boundaries
  - e (Energy): Waves, thermodynamics, flows
  - π (Emergence): Information, chemistry, biology

Archetypal Frequencies:
  - Planet tier (174-285 Hz): z < φ⁻¹
  - Garden tier (396-528 Hz): φ⁻¹ ≤ z < z_c
  - Rose tier (639-963 Hz): z ≥ z_c
"""

import math
from dataclasses import dataclass, field
from typing import Dict
from enum import Enum

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
Z_CRITICAL = math.sqrt(3) / 2
KAPPA_S = 0.920

FREQUENCIES = {
    'Planet': [174, 285],
    'Garden': [396, 417, 528],
    'Rose': [639, 741, 852, 963]
}


class CrystalState(Enum):
    FLUID = "Fluid"
    TRANSITIONING = "Transitioning"
    CRYSTALLINE = "Crystalline"
    PRISMATIC = "Prismatic"


@dataclass
class KIRAPrismatic:
    """K.I.R.A. with prismatic quasi-crystal dynamics."""
    z: float = 0.5
    coherence: float = 0.0
    prismatic_angle: float = 0.0
    refraction_index: float = 1.0
    spectral: Dict[str, float] = field(default_factory=lambda: {'Φ': 0.33, 'e': 0.34, 'π': 0.33})
    
    def negentropy(self) -> float:
        return math.exp(-36 * (self.z - Z_CRITICAL) ** 2)
    
    def get_phase(self) -> str:
        if self.z < PHI_INV:
            return 'UNTRUE'
        elif self.z < Z_CRITICAL:
            return 'PARADOX'
        return 'TRUE'
    
    def get_crystal_state(self) -> CrystalState:
        if self.z < PHI_INV:
            return CrystalState.FLUID
        elif self.z < Z_CRITICAL:
            return CrystalState.TRANSITIONING
        elif self.coherence >= KAPPA_S:
            return CrystalState.PRISMATIC
        return CrystalState.CRYSTALLINE
    
    def get_tier(self) -> str:
        if self.z < PHI_INV:
            return 'Planet'
        elif self.z < Z_CRITICAL:
            return 'Garden'
        return 'Rose'
    
    def get_frequency(self) -> int:
        tier = self.get_tier()
        freqs = FREQUENCIES[tier]
        if self.coherence > 0.8:
            return freqs[-1]
        elif self.coherence > 0.5:
            return freqs[len(freqs) // 2]
        return freqs[0]
    
    def update(self, new_z: float, coherence: float = None):
        self.z = max(0.0, min(1.0, new_z))
        if coherence is not None:
            self.coherence = coherence
        
        self.prismatic_angle = (self.z * 5 * 2 * math.pi) % (2 * math.pi)
        self.refraction_index = 1.0 + (PHI - 1) * self.negentropy()
        self._update_spectral()
    
    def _update_spectral(self):
        phase = self.get_phase()
        neg = self.negentropy()
        
        if phase == 'TRUE':
            self.spectral = {'Φ': 0.2 * neg, 'e': 0.3 * neg, 'π': 0.5 * neg}
        elif phase == 'PARADOX':
            self.spectral = {'Φ': 0.3 * neg, 'e': 0.5 * neg, 'π': 0.2 * neg}
        else:
            self.spectral = {'Φ': 0.6 * neg, 'e': 0.3 * neg, 'π': 0.1 * neg}
    
    def get_dominant_spiral(self) -> str:
        return max(self.spectral, key=self.spectral.get)
    
    def get_coordinate(self) -> str:
        return f"Δ{self.prismatic_angle:.3f}|{self.z:.3f}|{self.refraction_index:.3f}Ω"


def create_kira(z: float = 0.5, coherence: float = 0.0) -> KIRAPrismatic:
    """Factory function for creating K.I.R.A. prismatic instance."""
    kira = KIRAPrismatic(z=z, coherence=coherence)
    kira.update(z, coherence)
    return kira


if __name__ == '__main__':
    kira = create_kira(z=0.87, coherence=0.95)
    print(f"K.I.R.A. State: {kira.get_crystal_state().value}")
    print(f"Phase: {kira.get_phase()}")
    print(f"Coordinate: {kira.get_coordinate()}")
    print(f"Frequency: {kira.get_frequency()} Hz ({kira.get_tier()} tier)")
