from pathlib import Path

from quantum_apl_python.helix_self_builder import HELIX_TIERS, build_report, map_instructions_to_nodes
from quantum_apl_python.translator import translate_lines

Z_SOLVE_SEQUENCE = """
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


def test_map_instructions_covers_all_nodes():
    instructions = translate_lines(Z_SOLVE_SEQUENCE)
    mapping = map_instructions_to_nodes(instructions)
    assert set(mapping.keys()) == {spec.slug for spec in HELIX_TIERS}
    # Ensure progression pushes operators up the helix
    assert len(mapping["z0p41"]) == 2
    assert mapping["z0p80"][-1].subject == "Helix"


def test_build_report_aggregates_metadata(tmp_path):
    token_file = Path(tmp_path) / "z_solve.apl"
    token_file.write_text("\n".join(Z_SOLVE_SEQUENCE), encoding="utf-8")
    instructions = translate_lines(Z_SOLVE_SEQUENCE)
    report = build_report(token_file, instructions)
    assert report["instruction_count"] == 10
    assert len(report["nodes"]) == len(HELIX_TIERS)
    final_node = report["nodes"][-1]
    assert final_node["slug"] == "z0p80"
    assert "Autonomous" in final_node["title"]
