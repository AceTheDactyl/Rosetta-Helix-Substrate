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
    },
    # =========================================================================
    # ADVANCED TRAINING TOOLS
    # =========================================================================
    {
        "name": "run_full_training_session",
        "description": "Run a complete multi-epoch training session aiming for K-formation. Combines Kuramoto training, z optimization, and coherence building.",
        "input_schema": {
            "type": "object",
            "properties": {
                "epochs": {
                    "type": "integer",
                    "description": "Number of training epochs",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10
                },
                "steps_per_epoch": {
                    "type": "integer",
                    "description": "Steps per epoch",
                    "minimum": 10,
                    "maximum": 500,
                    "default": 50
                },
                "target_kappa": {
                    "type": "number",
                    "description": "Target coherence (kappa)",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.92
                },
                "early_stop": {
                    "type": "boolean",
                    "description": "Stop early if K-formation achieved",
                    "default": True
                }
            },
            "required": []
        }
    },
    {
        "name": "optimize_coupling",
        "description": "Find the optimal Kuramoto coupling strength K for maximum synchronization. Scans K values and returns best result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "k_min": {
                    "type": "number",
                    "description": "Minimum coupling strength to test",
                    "minimum": 0.0,
                    "default": 0.1
                },
                "k_max": {
                    "type": "number",
                    "description": "Maximum coupling strength to test",
                    "maximum": 5.0,
                    "default": 3.0
                },
                "n_samples": {
                    "type": "integer",
                    "description": "Number of K values to test",
                    "minimum": 5,
                    "maximum": 50,
                    "default": 20
                },
                "steps_per_sample": {
                    "type": "integer",
                    "description": "Training steps per K value",
                    "minimum": 10,
                    "maximum": 200,
                    "default": 50
                }
            },
            "required": []
        }
    },
    {
        "name": "scan_parameter_space",
        "description": "Scan a parameter (z or coupling K) across a range and measure system response (negentropy, coherence, etc.).",
        "input_schema": {
            "type": "object",
            "properties": {
                "parameter": {
                    "type": "string",
                    "description": "Parameter to scan",
                    "enum": ["z", "coupling_strength"]
                },
                "start": {
                    "type": "number",
                    "description": "Start value for scan",
                    "default": 0.0
                },
                "end": {
                    "type": "number",
                    "description": "End value for scan",
                    "default": 1.0
                },
                "n_points": {
                    "type": "integer",
                    "description": "Number of points to sample",
                    "minimum": 5,
                    "maximum": 100,
                    "default": 20
                }
            },
            "required": ["parameter"]
        }
    },
    {
        "name": "measure_stability",
        "description": "Measure stability/basin of attraction around current state by applying perturbations and measuring recovery.",
        "input_schema": {
            "type": "object",
            "properties": {
                "perturbation_size": {
                    "type": "number",
                    "description": "Size of perturbation to apply",
                    "minimum": 0.01,
                    "maximum": 0.5,
                    "default": 0.1
                },
                "recovery_steps": {
                    "type": "integer",
                    "description": "Steps to allow for recovery",
                    "minimum": 10,
                    "maximum": 200,
                    "default": 50
                },
                "n_trials": {
                    "type": "integer",
                    "description": "Number of perturbation trials",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5
                }
            },
            "required": []
        }
    },
    {
        "name": "run_convergence_test",
        "description": "Test if system converges to K-formation from current state. Runs extended simulation and tracks convergence metrics.",
        "input_schema": {
            "type": "object",
            "properties": {
                "max_steps": {
                    "type": "integer",
                    "description": "Maximum steps to run",
                    "minimum": 100,
                    "maximum": 5000,
                    "default": 1000
                },
                "convergence_threshold": {
                    "type": "number",
                    "description": "Threshold for considering converged",
                    "minimum": 0.001,
                    "maximum": 0.1,
                    "default": 0.01
                }
            },
            "required": []
        }
    },
    {
        "name": "get_phase_diagram_data",
        "description": "Generate phase diagram data by scanning z and computing properties at each point. Useful for visualization.",
        "input_schema": {
            "type": "object",
            "properties": {
                "z_points": {
                    "type": "integer",
                    "description": "Number of z values to sample",
                    "minimum": 10,
                    "maximum": 200,
                    "default": 50
                },
                "include_dynamics": {
                    "type": "boolean",
                    "description": "Include dynamic properties (slower but more detailed)",
                    "default": False
                }
            },
            "required": []
        }
    },
    {
        "name": "batch_simulate",
        "description": "Run multiple simulations with different initial conditions and return aggregate statistics.",
        "input_schema": {
            "type": "object",
            "properties": {
                "n_simulations": {
                    "type": "integer",
                    "description": "Number of simulations to run",
                    "minimum": 2,
                    "maximum": 50,
                    "default": 10
                },
                "steps_per_sim": {
                    "type": "integer",
                    "description": "Steps per simulation",
                    "minimum": 50,
                    "maximum": 500,
                    "default": 100
                },
                "vary_initial_z": {
                    "type": "boolean",
                    "description": "Randomize initial z for each simulation",
                    "default": True
                },
                "target": {
                    "type": "string",
                    "description": "Target for simulations",
                    "enum": ["lens", "k_formation", "phi_inv"],
                    "default": "lens"
                }
            },
            "required": []
        }
    },
    {
        "name": "analyze_trajectory",
        "description": "Analyze the current training trajectory history for patterns, trends, and anomalies.",
        "input_schema": {
            "type": "object",
            "properties": {
                "window_size": {
                    "type": "integer",
                    "description": "Window size for moving averages",
                    "minimum": 5,
                    "maximum": 100,
                    "default": 20
                }
            },
            "required": []
        }
    },
    {
        "name": "set_radius",
        "description": "Set the R (radius/layers) parameter for K-formation. R must be >= 7 for K-formation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "R": {
                    "type": "integer",
                    "description": "Radius/layers value",
                    "minimum": 1,
                    "maximum": 20
                }
            },
            "required": ["R"]
        }
    },
    # =========================================================================
    # GITHUB WORKFLOW TOOLS
    # =========================================================================
    {
        "name": "trigger_github_workflow",
        "description": "Trigger the autonomous training GitHub Actions workflow. Optionally wait for results and download artifacts. Requires GITHUB_TOKEN environment variable.",
        "input_schema": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "Training goal for Claude to pursue",
                    "default": "Achieve K-formation by reaching THE LENS"
                },
                "max_iterations": {
                    "type": "integer",
                    "description": "Maximum training iterations",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 10
                },
                "initial_z": {
                    "type": "number",
                    "description": "Initial z-coordinate",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.3
                },
                "wait_for_results": {
                    "type": "boolean",
                    "description": "Wait for workflow to complete and download results",
                    "default": True
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds when waiting for results",
                    "minimum": 60,
                    "maximum": 1800,
                    "default": 600
                }
            },
            "required": []
        }
    },
    {
        "name": "get_workflow_status",
        "description": "Get the status of the latest GitHub Actions workflow run.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "download_workflow_results",
        "description": "Download artifacts from a GitHub Actions workflow run.",
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "integer",
                    "description": "Workflow run ID (optional, defaults to latest run)"
                }
            },
            "required": []
        }
    }
]
