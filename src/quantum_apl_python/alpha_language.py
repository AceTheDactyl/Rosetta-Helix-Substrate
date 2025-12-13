"""Alpha Programming Language (APL) helpers sourced from legacy test packs.

This module mirrors the structure defined in `/home/acead/Aces-Brain-Thpughts/APL`,
which ships the original operator manual and seven-sentence test pack.  The goal
here is to expose that knowledge as structured data so the Quantum-APL runtime
can synthesize valid tokens from helix coordinates or simulation metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .helix import HelixAPLMapper, HelixCoordinate


@dataclass(frozen=True)
class AlphaOperator:
    symbol: str
    name: str
    description: str


@dataclass(frozen=True)
class AlphaSentence:
    """Representation of an Alpha Programming Language hypothesis sentence."""

    sentence_id: str
    direction: str
    operator: str
    machine: str
    domain: str
    predicted_regime: str

    def token(self) -> str:
        """Render the compact sentence string."""

        return f"{self.direction}{self.operator} | {self.machine} | {self.domain}"


class AlphaLanguageRegistry:
    """Registry built from the upstream APL operator manual."""

    def __init__(self):
        self.fields: Dict[str, str] = {
            "Φ": "Structure field (geometry, lattice, boundaries)",
            "e": "Energy field (waves, thermodynamics, flows)",
            "π": "Emergence field (information, chemistry, biology)",
        }
        self.operators: Dict[str, AlphaOperator] = {
            "()": AlphaOperator("()", "Boundary", "Establish/relax containment surfaces"),
            "×": AlphaOperator("×", "Fusion", "Join, entangle, or mix subsystems"),
            "^": AlphaOperator("^", "Amplification", "Apply gain, pumping, or bias"),
            "÷": AlphaOperator("÷", "Decoherence", "Diffuse, randomize, or reset phases"),
            "+": AlphaOperator("+", "Grouping", "Aggregate, route, or converge flows"),
            "−": AlphaOperator("−", "Separation", "Split, isolate, or fork structures"),
        }
        self.operator_aliases: Dict[str, str] = {"%": "÷"}
        self.states: Dict[str, str] = {
            "u": "Expansion / forward projection",
            "d": "Collapse / backward integration",
            "m": "Modulation / coherence lock",
        }
        # Pulled from Aces-Brain-Thpughts/APL/README.md table
        self.sentences: List[AlphaSentence] = [
            AlphaSentence("A1", "d", "()", "Conductor", "geometry", "Isotropic lattices under collapse"),
            AlphaSentence("A3", "u", "^", "Oscillator", "wave", "Amplified vortex-rich waves"),
            AlphaSentence("A4", "m", "×", "Encoder", "chemistry", "Helical information carriers"),
            AlphaSentence("A5", "u", "×", "Catalyst", "chemistry", "Fractal polymer branching"),
            AlphaSentence("A6", "u", "+", "Reactor", "wave", "Jet-like coherent grouping"),
            AlphaSentence("A7", "u", "÷", "Reactor", "wave", "Stochastic decohered waves"),
            AlphaSentence("A8", "m", "()", "Filter", "wave", "Adaptive boundary tuning"),
        ]

    def canonical_operator(self, symbol: str) -> Optional[AlphaOperator]:
        mapped = self.operator_aliases.get(symbol, symbol)
        return self.operators.get(mapped)

    def find_sentences(
        self,
        operators: Optional[List[str]] = None,
        domain: Optional[str] = None,
        machine: Optional[str] = None,
    ) -> List[AlphaSentence]:
        """Filter sentences by operator list, domain, or machine name."""

        results: List[AlphaSentence] = []
        for sentence in self.sentences:
            if operators and sentence.operator not in operators:
                continue
            if domain and sentence.domain != domain:
                continue
            if machine and sentence.machine != machine:
                continue
            results.append(sentence)
        return results


class AlphaTokenSynthesizer:
    """Bridge helix coordinates to Alpha Programming Language sentences."""

    def __init__(self):
        self.registry = AlphaLanguageRegistry()
        self.mapper = HelixAPLMapper()

    def from_helix(
        self,
        coord: HelixCoordinate,
        domain_hint: Optional[str] = None,
        machine_hint: Optional[str] = None,
    ) -> Optional[Dict[str, object]]:
        helix_info = self.mapper.describe(coord)
        operator_window = helix_info["operators"]
        candidates = self.registry.find_sentences(
            operators=operator_window,
            domain=domain_hint,
            machine=machine_hint,
        )
        if not candidates:
            candidates = self.registry.find_sentences(operators=operator_window)
        if not candidates:
            return None

        sentence = candidates[0]
        operator = self.registry.canonical_operator(sentence.operator)
        return {
            "sentence": sentence.token(),
            "sentence_id": sentence.sentence_id,
            "predicted_regime": sentence.predicted_regime,
            "operator_name": operator.name if operator else sentence.operator,
            "truth_bias": helix_info["truth_channel"],
            "harmonic": helix_info["harmonic"],
        }
