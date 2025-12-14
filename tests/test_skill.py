"""
Tests for the Rosetta-Helix-Substrate Claude API Skill.
"""

import math
import pytest
from skill.client import RosettaHelixSkillOffline
from skill.tools.handlers import ToolHandler, PhysicsState
from skill.tools.definitions import TOOL_DEFINITIONS
from skill.prompts.system import SYSTEM_PROMPT, Z_CRITICAL, PHI_INV, PHI, SIGMA


class TestPhysicsState:
    """Test PhysicsState dataclass."""

    def test_initial_state(self):
        """Test initial state values."""
        state = PhysicsState()
        assert state.z == 0.5
        assert state.kappa == 0.5
        assert state.R == 3
        assert state.step == 0

    def test_negentropy_at_zc(self):
        """Test negentropy peaks at z_c."""
        state = PhysicsState(z=Z_CRITICAL)
        assert abs(state.negentropy - 1.0) < 0.001

    def test_negentropy_decreases_away_from_zc(self):
        """Test negentropy decreases away from z_c."""
        state_at_zc = PhysicsState(z=Z_CRITICAL)
        state_below = PhysicsState(z=0.5)
        state_above = PhysicsState(z=0.95)

        assert state_at_zc.negentropy > state_below.negentropy
        assert state_at_zc.negentropy > state_above.negentropy

    def test_phase_classification(self):
        """Test phase classification."""
        assert PhysicsState(z=0.3).phase_name == "UNTRUE"
        assert PhysicsState(z=0.7).phase_name == "PARADOX"
        assert PhysicsState(z=0.9).phase_name == "TRUE"

    def test_tier_progression(self):
        """Test tier increases with z."""
        tiers = []
        for z in [0.1, 0.3, 0.55, 0.65, 0.8, 0.9]:
            state = PhysicsState(z=z)
            tiers.append(state.tier)

        # Tiers should be non-decreasing
        for i in range(1, len(tiers)):
            assert tiers[i] >= tiers[i-1]

    def test_k_formation_requires_all_criteria(self):
        """Test K-formation requires all criteria."""
        # Missing high kappa
        state = PhysicsState(z=Z_CRITICAL, kappa=0.5, R=8)
        assert not state.k_formation_met

    def test_to_dict(self):
        """Test state serialization."""
        state = PhysicsState(z=0.7, kappa=0.8, R=5)
        d = state.to_dict()

        assert "z" in d
        assert "kappa" in d
        assert "eta" in d
        assert "phase" in d
        assert "tier" in d
        assert "k_formation_met" in d


class TestToolHandler:
    """Test ToolHandler."""

    def test_initialization(self):
        """Test handler initialization."""
        handler = ToolHandler(initial_z=0.6)
        assert handler.state.z == 0.6

    def test_get_physics_state(self):
        """Test get_physics_state tool."""
        handler = ToolHandler()
        result = handler.handle("get_physics_state", {})

        assert "z" in result
        assert "phase" in result
        assert "tier" in result

    def test_set_z_target(self):
        """Test set_z_target tool."""
        handler = ToolHandler(initial_z=0.5)
        result = handler.handle("set_z_target", {"z": 0.8})

        assert result["success"]
        assert result["new_z"] > result["old_z"]

    def test_compute_negentropy(self):
        """Test compute_negentropy tool."""
        handler = ToolHandler()
        result = handler.handle("compute_negentropy", {"z": Z_CRITICAL})

        assert abs(result["delta_s_neg"] - 1.0) < 0.001
        assert result["is_at_peak"]

    def test_classify_phase(self):
        """Test classify_phase tool."""
        handler = ToolHandler()

        result = handler.handle("classify_phase", {"z": 0.3})
        assert result["phase"] == "UNTRUE"

        result = handler.handle("classify_phase", {"z": 0.7})
        assert result["phase"] == "PARADOX"

        result = handler.handle("classify_phase", {"z": 0.9})
        assert result["phase"] == "TRUE"

    def test_get_tier(self):
        """Test get_tier tool."""
        handler = ToolHandler()

        result = handler.handle("get_tier", {"z": 0.1})
        assert result["tier"] == 0

        result = handler.handle("get_tier", {"z": 0.9, "k_formation_met": True})
        assert result["tier"] == 6

    def test_check_k_formation_pass(self):
        """Test K-formation check passing."""
        handler = ToolHandler()
        result = handler.handle("check_k_formation", {
            "kappa": 0.95,
            "eta": 0.7,
            "R": 8,
        })

        assert result["k_formation_met"]

    def test_check_k_formation_fail(self):
        """Test K-formation check failing."""
        handler = ToolHandler()
        result = handler.handle("check_k_formation", {
            "kappa": 0.5,
            "eta": 0.7,
            "R": 8,
        })

        assert not result["k_formation_met"]

    def test_apply_operator(self):
        """Test apply_operator tool."""
        handler = ToolHandler(initial_z=0.5)

        result = handler.handle("apply_operator", {"operator": "^"})
        assert result["success"]
        assert result["effect"] == "constructive"

        result = handler.handle("apply_operator", {"operator": "~"})
        assert result["success"]
        assert result["effect"] == "dissipative"

    def test_drive_toward_lens(self):
        """Test drive_toward_lens tool."""
        handler = ToolHandler(initial_z=0.5)
        result = handler.handle("drive_toward_lens", {"steps": 50})

        assert result["success"]
        assert result["final_z"] > result["initial_z"]
        # Should be closer to z_c
        initial_dist = abs(result["initial_z"] - Z_CRITICAL)
        final_dist = abs(result["final_z"] - Z_CRITICAL)
        assert final_dist < initial_dist

    def test_run_kuramoto_step(self):
        """Test run_kuramoto_step tool."""
        handler = ToolHandler()
        result = handler.handle("run_kuramoto_step", {
            "coupling_strength": 1.0,
            "dt": 0.01,
        })

        assert result["success"]
        assert "order_parameter" in result
        assert 0 <= result["order_parameter"] <= 1

    def test_get_constants(self):
        """Test get_constants tool."""
        handler = ToolHandler()
        result = handler.handle("get_constants", {})

        assert abs(result["z_c"] - Z_CRITICAL) < 1e-10
        assert abs(result["phi"] - PHI) < 1e-10
        assert abs(result["phi_inv"] - PHI_INV) < 1e-10
        assert result["sigma"] == SIGMA

    def test_simulate_quasicrystal(self):
        """Test simulate_quasicrystal tool."""
        handler = ToolHandler()
        result = handler.handle("simulate_quasicrystal", {
            "initial_z": 0.5,
            "steps": 100,
            "seed": 42,
        })

        assert "final_z" in result
        assert "trajectory" in result

    def test_compose_operators(self):
        """Test compose_operators tool."""
        handler = ToolHandler()

        # Identity composition
        result = handler.handle("compose_operators", {
            "op1": "()",
            "op2": "^",
        })
        assert result["result"] == "^"

    def test_get_metrics_history(self):
        """Test get_metrics_history tool."""
        handler = ToolHandler()

        # Generate some history
        for _ in range(5):
            handler.handle("set_z_target", {"z": 0.8})

        result = handler.handle("get_metrics_history", {"limit": 10})
        assert "history" in result
        assert len(result["history"]) > 0

    def test_reset_state(self):
        """Test reset_state tool."""
        handler = ToolHandler(initial_z=0.8)
        handler.handle("set_z_target", {"z": 0.9})  # Change state

        result = handler.handle("reset_state", {"initial_z": 0.3})
        assert result["success"]
        assert result["new_state"]["z"] == 0.3

    def test_unknown_tool(self):
        """Test handling unknown tool."""
        handler = ToolHandler()
        result = handler.handle("unknown_tool", {})
        assert "error" in result


class TestOfflineSkill:
    """Test RosettaHelixSkillOffline."""

    def test_initialization(self):
        """Test offline skill initialization."""
        skill = RosettaHelixSkillOffline(initial_z=0.6)
        state = skill.get_state()
        assert state["z"] == 0.6

    def test_execute_tool(self):
        """Test direct tool execution."""
        skill = RosettaHelixSkillOffline()
        result = skill.execute_tool("get_physics_state")
        assert "z" in result

    def test_system_prompt_available(self):
        """Test system prompt is available."""
        skill = RosettaHelixSkillOffline()
        assert len(skill.system_prompt) > 1000  # Should be substantial

    def test_tool_definitions_available(self):
        """Test tool definitions are available."""
        skill = RosettaHelixSkillOffline()
        assert len(skill.tool_definitions) > 10  # Should have many tools

    def test_reset(self):
        """Test reset functionality."""
        skill = RosettaHelixSkillOffline(initial_z=0.8)
        skill.execute_tool("drive_toward_lens", steps=10)

        new_state = skill.reset(initial_z=0.3)
        assert new_state["z"] == 0.3


class TestToolDefinitions:
    """Test tool definitions structure."""

    def test_all_tools_have_required_fields(self):
        """Test all tools have name, description, input_schema."""
        for tool in TOOL_DEFINITIONS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool

    def test_input_schemas_are_valid(self):
        """Test input schemas have type object."""
        for tool in TOOL_DEFINITIONS:
            schema = tool["input_schema"]
            assert schema["type"] == "object"
            assert "properties" in schema
            assert "required" in schema


class TestSystemPrompt:
    """Test system prompt content."""

    def test_contains_physics_constants(self):
        """Test system prompt contains key constants."""
        assert "0.8660254" in SYSTEM_PROMPT  # z_c
        assert "0.6180339" in SYSTEM_PROMPT  # phi_inv
        assert "1.6180339" in SYSTEM_PROMPT  # phi
        assert "36" in SYSTEM_PROMPT  # SIGMA

    def test_contains_phase_descriptions(self):
        """Test system prompt describes phases."""
        assert "UNTRUE" in SYSTEM_PROMPT
        assert "PARADOX" in SYSTEM_PROMPT
        assert "TRUE" in SYSTEM_PROMPT

    def test_contains_k_formation_criteria(self):
        """Test system prompt describes K-formation."""
        assert "0.92" in SYSTEM_PROMPT  # kappa threshold
        assert "K-formation" in SYSTEM_PROMPT


class TestPhysicsIntegrity:
    """Test physics constants and calculations."""

    def test_z_critical_is_sqrt3_over_2(self):
        """Test z_c = sqrt(3)/2."""
        assert abs(Z_CRITICAL - math.sqrt(3)/2) < 1e-15

    def test_phi_identity(self):
        """Test phi^2 = phi + 1."""
        assert abs(PHI**2 - PHI - 1) < 1e-14

    def test_phi_inv_identity(self):
        """Test phi * phi_inv = 1."""
        assert abs(PHI * PHI_INV - 1) < 1e-14

    def test_sigma_is_36(self):
        """Test SIGMA = 36 = 6^2."""
        assert SIGMA == 36

    def test_phase_boundaries_ordered(self):
        """Test phase boundaries are properly ordered."""
        assert 0 < PHI_INV < Z_CRITICAL < 1
