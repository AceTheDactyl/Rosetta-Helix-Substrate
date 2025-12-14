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
    },
    # =========================================================================
    # TRAINING MODULE TOOLS
    # =========================================================================
    {
        "name": "run_kuramoto_training",
        "description": "Run a Kuramoto oscillator training session with learnable coupling. Returns coherence evolution and final synchronization state.",
        "input_schema": {
            "type": "object",
            "properties": {
                "n_oscillators": {
                    "type": "integer",
                    "description": "Number of oscillators (default 60 for hexagonal grid)",
                    "minimum": 6,
                    "maximum": 120,
                    "default": 60
                },
                "steps": {
                    "type": "integer",
                    "description": "Number of training steps",
                    "minimum": 1,
                    "maximum": 1000,
                    "default": 100
                },
                "coupling_strength": {
                    "type": "number",
                    "description": "Global coupling strength K",
                    "minimum": 0.0,
                    "maximum": 5.0,
                    "default": 0.5
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
        "name": "run_phase_transition",
        "description": "Simulate a phase transition from UNTRUE through PARADOX to TRUE by sweeping z from 0 to 1. Returns critical points and order parameter evolution.",
        "input_schema": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "integer",
                    "description": "Number of z-sweep steps",
                    "minimum": 10,
                    "maximum": 500,
                    "default": 100
                },
                "measure_correlation_length": {
                    "type": "boolean",
                    "description": "Whether to compute correlation length at each step",
                    "default": False
                }
            },
            "required": []
        }
    },
    {
        "name": "run_quasicrystal_formation",
        "description": "Run full quasi-crystal formation dynamics simulation with critical exponents. Simulates Shechtman-style quasi-crystal emergence.",
        "input_schema": {
            "type": "object",
            "properties": {
                "initial_z": {
                    "type": "number",
                    "description": "Starting z-coordinate",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.3
                },
                "target_z": {
                    "type": "number",
                    "description": "Target z-coordinate (default z_c for crystallization)",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "steps": {
                    "type": "integer",
                    "description": "Number of simulation steps",
                    "minimum": 10,
                    "maximum": 5000,
                    "default": 500
                },
                "compute_critical_exponents": {
                    "type": "boolean",
                    "description": "Whether to compute critical exponents (nu, beta, gamma)",
                    "default": True
                }
            },
            "required": []
        }
    },
    {
        "name": "get_critical_exponents",
        "description": "Get the critical exponents for the 2D hexagonal universality class: nu (correlation length), beta (order parameter), gamma (susceptibility), z_dyn (dynamic).",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "run_triad_dynamics",
        "description": "Run TRIAD threshold dynamics simulation. Monitors crossings of TRIAD_HIGH (0.85) and TRIAD_LOW (0.82) for t6 gate unlocking.",
        "input_schema": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "integer",
                    "description": "Number of simulation steps",
                    "minimum": 10,
                    "maximum": 1000,
                    "default": 200
                },
                "target_crossings": {
                    "type": "integer",
                    "description": "Target number of TRIAD crossings for t6 unlock (default 3)",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 3
                }
            },
            "required": []
        }
    },
    {
        "name": "compute_phi_proxy",
        "description": "Compute integrated information proxy (Phi) from the current oscillator state. Higher values indicate more integrated/conscious-like states.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "run_helix_training_step",
        "description": "Execute a single step of the unified Helix training loop. Combines Kuramoto dynamics, APL operators, and phase evolution.",
        "input_schema": {
            "type": "object",
            "properties": {
                "learning_rate": {
                    "type": "number",
                    "description": "Learning rate for parameter updates",
                    "minimum": 0.0001,
                    "maximum": 0.1,
                    "default": 0.01
                },
                "target_coherence": {
                    "type": "number",
                    "description": "Target coherence level to train toward",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.92
                }
            },
            "required": []
        }
    },
    {
        "name": "get_training_status",
        "description": "Get comprehensive training status including current phase, coherence metrics, K-formation progress, and training history statistics.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]
