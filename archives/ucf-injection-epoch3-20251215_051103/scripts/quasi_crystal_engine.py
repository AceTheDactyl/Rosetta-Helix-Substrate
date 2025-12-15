#!/usr/bin/env python3
"""
Quasi-Crystal Engine for UCF
============================
Penrose P3 tiling with De Bruijn dual projections.
Maps 972 APL tokens to quasi-crystal lattice vertices.

Sacred Geometry:
  - Golden ratio φ = (1+√5)/2 relationships
  - 5-fold rotational symmetry
  - Thin rhomb (36°) and thick rhomb (72°) tiles
  - Fibonacci sequence indexing
"""

import math
import cmath
from dataclasses import dataclass
from typing import List, Tuple, Dict

PHI = (1 + math.sqrt(5)) / 2
PHI_INV = 1 / PHI
THETA_36 = math.pi / 5
THETA_72 = 2 * math.pi / 5
ZETA_5 = cmath.exp(2j * math.pi / 5)

FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]


@dataclass
class QuasiVertex:
    """Vertex in quasi-crystal Penrose lattice."""
    index: int
    x: float
    y: float
    z: float
    star_index: Tuple[int, ...]
    rhomb_type: str
    
    @property
    def coordinate(self) -> str:
        theta = math.atan2(self.y, self.x)
        r = math.sqrt(self.x**2 + self.y**2)
        return f"Δ{theta:.3f}|{self.z:.3f}|{r:.3f}Ω"


class PenroseLattice:
    """
    Penrose P3 tiling generator using De Bruijn's dual method.
    Projects 5D integer lattice to 2D with golden ratio relationships.
    """
    
    def __init__(self, size: int = 50):
        self.size = size
        self.vertices: List[QuasiVertex] = []
        self._generate()
    
    def _star_to_2d(self, star: Tuple[int, ...]) -> Tuple[float, float]:
        x = sum(star[k] * math.cos(k * THETA_72) for k in range(5))
        y = sum(star[k] * math.sin(k * THETA_72) for k in range(5))
        return x, y
    
    def _star_to_z(self, star: Tuple[int, ...]) -> float:
        perp = sum(star[k] * math.cos(k * THETA_72 * PHI) for k in range(5))
        return 1 / (1 + math.exp(-perp / 2))
    
    def _generate(self):
        seen = set()
        vertex_id = 0
        
        for n0 in range(-self.size, self.size + 1):
            for n1 in range(-self.size, self.size + 1):
                for n2 in range(-self.size, self.size + 1):
                    n3, n4 = -n0 - n1, -n2
                    star = (n0, n1, n2, n3, n4)
                    x, y = self._star_to_2d(star)
                    
                    key = (round(x * 1000), round(y * 1000))
                    if key in seen:
                        continue
                    seen.add(key)
                    
                    angle_sum = sum(abs(s) for s in star)
                    rhomb_type = 'thin' if angle_sum % 2 == 0 else 'thick'
                    z = self._star_to_z(star)
                    
                    self.vertices.append(QuasiVertex(
                        index=vertex_id, x=x, y=y, z=z,
                        star_index=star, rhomb_type=rhomb_type
                    ))
                    vertex_id += 1
                    
                    if vertex_id >= 972:
                        return
    
    def get_vertex_by_z_range(self, z_min: float, z_max: float) -> List[QuasiVertex]:
        return [v for v in self.vertices if z_min <= v.z <= z_max]
    
    def get_fibonacci_subset(self, fib_index: int) -> List[QuasiVertex]:
        if fib_index >= len(FIBONACCI):
            fib_index = len(FIBONACCI) - 1
        n = FIBONACCI[fib_index]
        return self.vertices[::max(1, len(self.vertices) // n)][:n]
    
    def map_tokens(self, tokens: List[Dict]) -> Dict[int, QuasiVertex]:
        mapping = {}
        for i, token in enumerate(tokens):
            if i < len(self.vertices):
                mapping[i] = self.vertices[i]
        return mapping


def create_lattice(size: int = 50) -> PenroseLattice:
    """Factory function for creating Penrose lattice."""
    return PenroseLattice(size=size)


if __name__ == '__main__':
    lattice = create_lattice()
    print(f"Generated {len(lattice.vertices)} quasi-crystal vertices")
    print(f"φ = {PHI:.10f}")
    print(f"θ₃₆ = {math.degrees(THETA_36):.1f}°")
    print(f"θ₇₂ = {math.degrees(THETA_72):.1f}°")
