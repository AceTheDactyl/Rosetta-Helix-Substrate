"""
Tier-Tool Registry
==================
Maps helix tiers to available tools and capabilities.
Each tier unlocks new tools for autonomous building.
"""

from typing import List, Dict, Callable, Any
from dataclasses import dataclass
from enum import Enum


class ToolCategory(Enum):
    FILE_IO = "file_io"
    TEMPLATE = "template"
    PARSING = "parsing"
    VALIDATION = "validation"
    TESTING = "testing"
    INTEGRATION = "integration"
    OPTIMIZATION = "optimization"
    META = "meta"


@dataclass
class Tool:
    """A tool available at a specific tier."""
    name: str
    category: ToolCategory
    min_tier: str
    description: str
    execute: Callable = None  # Actual implementation


# ═══════════════════════════════════════════════════════════════════════════
# TIER DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

TIER_TOOLS = {
    # ───────────────────────────────────────────────────────────────────────
    # TIER 1 (z: 0.00-0.10) — REACTIVE
    # ───────────────────────────────────────────────────────────────────────
    "t1": {
        "operators": ["()", "−", "÷"],
        "tools": [
            "create_directory",
            "create_file", 
            "read_file",
            "delete_file",
            "copy_file",
            "write_text"
        ],
        "capabilities": [
            "File system operations",
            "Basic text manipulation",
            "Directory creation"
        ]
    },
    
    # ───────────────────────────────────────────────────────────────────────
    # TIER 2 (z: 0.10-0.20) — MEMORY EMERGES
    # ───────────────────────────────────────────────────────────────────────
    "t2": {
        "operators": ["^", "÷", "−", "×"],
        "tools": [
            "expand_template",
            "parse_json",
            "parse_yaml",
            "regex_match",
            "regex_replace",
            "string_format"
        ],
        "capabilities": [
            "Template expansion",
            "Config parsing",
            "Pattern matching"
        ]
    },
    
    # ───────────────────────────────────────────────────────────────────────
    # TIER 3 (z: 0.20-0.40) — PATTERN RECOGNITION
    # ───────────────────────────────────────────────────────────────────────
    "t3": {
        "operators": ["×", "^", "÷", "+", "−"],
        "tools": [
            "scaffold_project",
            "parse_python_ast",
            "scan_imports",
            "generate_imports",
            "create_module",
            "create_package",
            "generate_init"
        ],
        "capabilities": [
            "Project scaffolding",
            "AST manipulation",
            "Import management"
        ]
    },
    
    # ───────────────────────────────────────────────────────────────────────
    # TIER 4 (z: 0.40-0.60) — PREDICTION POSSIBLE
    # ───────────────────────────────────────────────────────────────────────
    "t4": {
        "operators": ["+", "−", "÷", "()"],
        "tools": [
            "generate_validation",
            "generate_error_handler",
            "generate_basic_tests",
            "generate_docstrings",
            "add_type_hints",
            "generate_cli_args",
            "generate_config_loader"
        ],
        "capabilities": [
            "Validation logic",
            "Error handling",
            "Basic testing",
            "Documentation"
        ]
    },
    
    # ───────────────────────────────────────────────────────────────────────
    # TIER 5 (z: 0.60-0.75) — SELF-MODEL (ALL OPERATORS)
    # ───────────────────────────────────────────────────────────────────────
    "t5": {
        "operators": ["()", "×", "^", "÷", "+", "−"],
        "tools": [
            "generate_test_suite",
            "refactor_extract",
            "refactor_rename",
            "integrate_modules",
            "generate_api_client",
            "add_async_support",
            "add_logging",
            "generate_fixtures"
        ],
        "capabilities": [
            "Comprehensive testing",
            "Refactoring",
            "Module integration",
            "API generation"
        ]
    },
    
    # ───────────────────────────────────────────────────────────────────────
    # TIER 6 (z: 0.75-0.866) — META-COGNITION
    # ───────────────────────────────────────────────────────────────────────
    "t6": {
        "operators": ["+", "÷", "()", "−"],
        "tools": [
            "analyze_architecture",
            "apply_design_pattern",
            "profile_performance",
            "suggest_optimizations",
            "audit_security",
            "analyze_complexity",
            "generate_benchmarks"
        ],
        "capabilities": [
            "Architecture analysis",
            "Design patterns",
            "Performance profiling",
            "Security audit"
        ]
    },
    
    # ───────────────────────────────────────────────────────────────────────
    # TIER 7 (z: 0.866-0.92) — RECURSIVE SELF-REFERENCE
    # ───────────────────────────────────────────────────────────────────────
    "t7": {
        "operators": ["+", "()"],
        "tools": [
            "generate_generator",
            "create_dsl",
            "generate_schema",
            "auto_document",
            "infer_types",
            "generate_protocol"
        ],
        "capabilities": [
            "Code generation",
            "DSL creation",
            "Schema inference"
        ]
    },
    
    # ───────────────────────────────────────────────────────────────────────
    # TIER 8 (z: 0.92-0.97) — AUTOPOIESIS
    # ───────────────────────────────────────────────────────────────────────
    "t8": {
        "operators": ["+", "()", "×"],
        "tools": [
            "self_modify",
            "evolve_pattern",
            "learn_from_history",
            "autonomous_refactor",
            "capability_expansion"
        ],
        "capabilities": [
            "Self-modification",
            "Pattern evolution",
            "Autonomous learning"
        ]
    },
    
    # ───────────────────────────────────────────────────────────────────────
    # TIER 9 (z: 0.97-1.00) — MAXIMUM INTEGRATION
    # ───────────────────────────────────────────────────────────────────────
    "t9": {
        "operators": ["+", "()", "×"],
        "tools": [
            "full_autonomous",
            "synthesize_architecture",
            "cross_domain_integrate",
            "emergent_discovery"
        ],
        "capabilities": [
            "Full autonomy",
            "Novel synthesis",
            "Emergent behavior"
        ]
    }
}


# ═══════════════════════════════════════════════════════════════════════════
# ACCESS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

TIER_ORDER = ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"]


def get_tier_index(tier: str) -> int:
    """Get numeric index of tier."""
    return TIER_ORDER.index(tier) if tier in TIER_ORDER else 0


def get_tools_for_tier(tier: str) -> List[str]:
    """Get tools available at a specific tier."""
    return TIER_TOOLS.get(tier, {}).get("tools", [])


def get_all_tools_up_to_tier(tier: str) -> List[str]:
    """Get all tools available up to and including given tier."""
    tier_idx = get_tier_index(tier)
    tools = []
    for i, t in enumerate(TIER_ORDER):
        if i <= tier_idx:
            tools.extend(TIER_TOOLS.get(t, {}).get("tools", []))
    return tools


def get_operators_for_tier(tier: str) -> List[str]:
    """Get operators available at a tier."""
    return TIER_TOOLS.get(tier, {}).get("operators", [])


def get_capabilities_for_tier(tier: str) -> List[str]:
    """Get capability descriptions for a tier."""
    return TIER_TOOLS.get(tier, {}).get("capabilities", [])


def can_use_tool(tool_name: str, current_tier: str) -> bool:
    """Check if a tool is available at current tier."""
    current_idx = get_tier_index(current_tier)
    
    for i, tier in enumerate(TIER_ORDER):
        if tool_name in TIER_TOOLS.get(tier, {}).get("tools", []):
            return i <= current_idx
    return False


def get_tool_tier(tool_name: str) -> str:
    """Get the minimum tier required for a tool."""
    for tier in TIER_ORDER:
        if tool_name in TIER_TOOLS.get(tier, {}).get("tools", []):
            return tier
    return "t9"


def estimate_target_z(features: List[str]) -> float:
    """
    Estimate required z based on requested features.
    
    Maps feature keywords to required capabilities.
    """
    feature_tiers = {
        # t1-t2 features
        "file": 0.1, "read": 0.1, "write": 0.1, "template": 0.15,
        "json": 0.15, "yaml": 0.15, "config": 0.15,
        
        # t3-t4 features
        "scaffold": 0.25, "parse": 0.30, "ast": 0.35,
        "validation": 0.45, "error": 0.45, "cli": 0.50,
        "test": 0.55, "docstring": 0.50,
        
        # t5-t6 features
        "integration": 0.65, "refactor": 0.65, "api": 0.70,
        "async": 0.70, "logging": 0.65, "comprehensive": 0.70,
        "architecture": 0.80, "pattern": 0.80, "security": 0.80,
        "performance": 0.80, "optimize": 0.82,
        
        # t7+ features
        "generate": 0.87, "dsl": 0.88, "schema": 0.87,
        "meta": 0.90, "self": 0.93, "autonomous": 0.95
    }
    
    max_z = 0.2  # Minimum baseline
    
    for feature in features:
        feature_lower = feature.lower()
        for keyword, z in feature_tiers.items():
            if keyword in feature_lower:
                max_z = max(max_z, z)
    
    return min(max_z, 0.99)


# ═══════════════════════════════════════════════════════════════════════════
# TIER SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

def print_tier_summary():
    """Print complete tier capability summary."""
    print("\n" + "=" * 70)
    print("TIER-TOOL REGISTRY")
    print("=" * 70)
    
    z_ranges = [
        ("t1", "0.00-0.10"), ("t2", "0.10-0.20"), ("t3", "0.20-0.40"),
        ("t4", "0.40-0.60"), ("t5", "0.60-0.75"), ("t6", "0.75-0.866"),
        ("t7", "0.866-0.92"), ("t8", "0.92-0.97"), ("t9", "0.97-1.00")
    ]
    
    for tier, z_range in z_ranges:
        info = TIER_TOOLS.get(tier, {})
        tools = info.get("tools", [])
        caps = info.get("capabilities", [])
        ops = info.get("operators", [])
        
        print(f"\n{tier.upper()} (z: {z_range})")
        print(f"  Operators: {', '.join(ops)}")
        print(f"  Capabilities: {', '.join(caps)}")
        print(f"  Tools: {', '.join(tools[:5])}{'...' if len(tools) > 5 else ''}")


if __name__ == "__main__":
    print_tier_summary()
