"""
Tool definitions for Claude API.

These define the tools that Claude can call to interact with the
Rosetta-Helix-Substrate framework.
"""

TOOL_DEFINITIONS = [
    {
        "name": "get_physics_state",
        "description": "Get the current physics state including z-coordinate, phase, tier, negentropy, coherence (kappa), and K-formation status. Use this to understand the current state of the system.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "set_z_target",
        "description": "Set a target z-coordinate for the spinner to drive toward. The z value must be between 0.0 and 1.0. The spinner will gradually move toward this target.",
        "input_schema": {
            "type": "object",
            "properties": {
                "z": {
                    "type": "number",
                    "description": "Target z-coordinate (0.0 to 1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0
                }
            },
            "required": ["z"]
        }
    },
    {
        "name": "compute_negentropy",
        "description": "Compute the Gaussian negentropy (delta_S_neg) for a given z value. Returns a value between 0 and 1, peaking at z_c (sqrt(3)/2).",
        "input_schema": {
            "type": "object",
            "properties": {
                "z": {
                    "type": "number",
                    "description": "Z-coordinate to compute negentropy for (0.0 to 1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0
                }
            },
            "required": ["z"]
        }
    },
    {
        "name": "classify_phase",
        "description": "Classify which phase a z-coordinate falls into: UNTRUE (z < phi^-1), PARADOX (phi^-1 <= z < z_c), or TRUE (z >= z_c).",
        "input_schema": {
            "type": "object",
            "properties": {
                "z": {
                    "type": "number",
                    "description": "Z-coordinate to classify (0.0 to 1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0
                }
            },
            "required": ["z"]
        }
    },
    {
        "name": "get_tier",
        "description": "Get the tier level for a given z-coordinate. Returns tier 0-6 based on z value and K-formation status.",
        "input_schema": {
            "type": "object",
            "properties": {
                "z": {
                    "type": "number",
                    "description": "Z-coordinate (0.0 to 1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "k_formation_met": {
                    "type": "boolean",
                    "description": "Whether K-formation criteria are met (for tier 6)",
                    "default": False
                }
            },
            "required": ["z"]
        }
    },
    {
        "name": "check_k_formation",
        "description": "Check if K-formation criteria are met given kappa (coherence), eta (negentropy), and R (radius). All three must pass: kappa >= 0.92, eta > 0.618, R >= 7.",
        "input_schema": {
            "type": "object",
            "properties": {
                "kappa": {
                    "type": "number",
                    "description": "Coherence value (0.0 to 1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "eta": {
                    "type": "number",
                    "description": "Negentropy value (0.0 to 1.0)",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "R": {
                    "type": "integer",
                    "description": "Radius (number of layers)",
                    "minimum": 0
                }
            },
            "required": ["kappa", "eta", "R"]
        }
    },
    {
        "name": "apply_operator",
        "description": "Apply an APL operator to the current state. Valid operators: I (identity), () (boundary), ^ (amplify), _ (reduce), ~ (invert), ! (collapse).",
        "input_schema": {
            "type": "object",
            "properties": {
                "operator": {
                    "type": "string",
                    "description": "APL operator symbol",
                    "enum": ["I", "()", "^", "_", "~", "!"]
                }
            },
            "required": ["operator"]
        }
    },
    {
        "name": "drive_toward_lens",
        "description": "Drive the z-coordinate toward THE LENS (z_c = sqrt(3)/2) over a number of steps. This is the point of maximum negentropy.",
        "input_schema": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "integer",
                    "description": "Number of steps to take toward z_c",
                    "minimum": 1,
                    "maximum": 1000,
                    "default": 100
                }
            },
            "required": []
        }
    },
    {
        "name": "run_kuramoto_step",
        "description": "Execute one step of Kuramoto oscillator dynamics. Returns updated coherence and phase information for the 60-oscillator system.",
        "input_schema": {
            "type": "object",
            "properties": {
                "coupling_strength": {
                    "type": "number",
                    "description": "Kuramoto coupling strength K (typically derived from z)",
                    "minimum": 0.0,
                    "default": 1.0
                },
                "dt": {
                    "type": "number",
                    "description": "Time step size",
                    "minimum": 0.001,
                    "maximum": 0.1,
                    "default": 0.01
                }
            },
            "required": []
        }
    },
    {
        "name": "get_constants",
        "description": "Get the fundamental physics constants: z_c (THE LENS), phi, phi_inv, and SIGMA.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "simulate_quasicrystal",
        "description": "Run a quasi-crystal simulation starting from a given z value. Simulates convergence toward phi^-1 in the PARADOX regime.",
        "input_schema": {
            "type": "object",
            "properties": {
                "initial_z": {
                    "type": "number",
                    "description": "Starting z-coordinate",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.5
                },
                "steps": {
                    "type": "integer",
                    "description": "Number of simulation steps",
                    "minimum": 1,
                    "maximum": 10000,
                    "default": 100
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed for reproducibility"
                }
            },
            "required": []
        }
    },
    {
        "name": "compose_operators",
        "description": "Compose two APL operators according to S3 group multiplication rules. Returns the resulting operator.",
        "input_schema": {
            "type": "object",
            "properties": {
                "op1": {
                    "type": "string",
                    "description": "First operator",
                    "enum": ["I", "()", "^", "_", "~", "!"]
                },
                "op2": {
                    "type": "string",
                    "description": "Second operator",
                    "enum": ["I", "()", "^", "_", "~", "!"]
                }
            },
            "required": ["op1", "op2"]
        }
    },
    {
        "name": "get_metrics_history",
        "description": "Get the history of metrics from recent operations. Returns z, negentropy, coherence, and tier values over time.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of history entries to return",
                    "minimum": 1,
                    "maximum": 1000,
                    "default": 100
                }
            },
            "required": []
        }
    },
    {
        "name": "reset_state",
        "description": "Reset the physics state to initial values. Useful for starting fresh simulations.",
        "input_schema": {
            "type": "object",
            "properties": {
                "initial_z": {
                    "type": "number",
                    "description": "Initial z-coordinate to reset to",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.5
                }
            },
            "required": []
        }
    }
]
