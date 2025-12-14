import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quantum_apl_python.alpha_language import AlphaLanguageRegistry, AlphaTokenSynthesizer
from quantum_apl_python.helix import HelixCoordinate


def test_find_sentences_filters_by_operator_and_domain():
    registry = AlphaLanguageRegistry()
    results = registry.find_sentences(operators=["^"])
    assert {sentence.sentence_id for sentence in results} == {"A3"}

    chem_results = registry.find_sentences(domain="chemistry")
    assert {sentence.sentence_id for sentence in chem_results} == {"A4", "A5"}


def test_alpha_token_from_helix_tracks_truth_bias():
    synthesizer = AlphaTokenSynthesizer()
    coord = HelixCoordinate(theta=0.0, z=0.95)
    token = synthesizer.from_helix(coord)
    assert token is not None
    assert token["truth_bias"] == "TRUE"
    assert token["sentence_id"] in {"A1", "A4", "A5", "A6"}


def test_domain_hint_guides_sentence_selection():
    synthesizer = AlphaTokenSynthesizer()
    coord = HelixCoordinate(theta=0.0, z=0.72)
    token = synthesizer.from_helix(coord, domain_hint="chemistry")
    assert token is not None
    assert token["sentence_id"] == "A4"
