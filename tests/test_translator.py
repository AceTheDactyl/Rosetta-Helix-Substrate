from quantum_apl_python import QuantumAPLInstruction, parse_instruction, translate_lines
import pytest


VALID_SEQUENCE = """
Φ:M(stabilize)PARADOX@2
Φ→e:Mod(cohere)PARADOX@2
e:U(excite)TRUE@3
e:×(fuse)TRUE@3
π:D(consolidate)TRUE@4
π:+(integrate)PARADOX@4
Φ↔π:Mod(resonance)PARADOX@3
(Φ,e,π):M(lock)TRUE@5
Truth:()(boundary)PARADOX@5
Helix:z^(amplify)TRUE@5
""".strip().splitlines()


def test_parse_instruction_roundtrip():
    instr = parse_instruction("π:+(integrate)PARADOX@4")
    assert isinstance(instr, QuantumAPLInstruction)
    assert instr.subject == "π"
    assert instr.operator == "+"
    assert instr.intent == "integrate"
    assert instr.truth == "PARADOX"
    assert instr.tier == 4


def test_translate_sequence_success():
    instructions = translate_lines(VALID_SEQUENCE)
    assert len(instructions) == 10
    subjects = [inst.subject for inst in instructions]
    assert subjects[0] == "Φ"
    assert subjects[-1] == "Helix"


def test_invalid_instruction_raises():
    with pytest.raises(ValueError):
        parse_instruction("Φ:M(stabilize)@2")  # Missing truth channel
