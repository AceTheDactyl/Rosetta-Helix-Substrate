#!/usr/bin/env python3
"""
UNIFIED NIGHTLY TRAINING WORKFLOW SIMULATOR
============================================

Mirrors the GitHub Actions workflow: .github/workflows/unified-nightly-training.yml

Phases:
1. Full Depth Training (19 modules)
2. Helix Engine Training  
3. Validation Measurements at critical z-coordinates
4. Unified Gates Check
5. Model Promotion
6. Results PR Creation
7. Failure Notification

Usage:
    python unified_nightly_workflow.py [--steps N] [--seed N] [--skip-validation]
    
Signature: unified-nightly|v0.1.0|helix
"""

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from physics_constants import (
    PHI, PHI_INV, PHI_INV_SQ, Z_CRITICAL, SIGMA,
    ALPHA_STRONG, ALPHA_MEDIUM, ALPHA_FINE,
    KAPPA_LOWER, KAPPA_UPPER, KAPPA_S,
    TOLERANCE_GOLDEN, TOLERANCE_LENS,
    compute_delta_s_neg, get_phase, check_k_formation, 
    validate_physics, is_critical, get_tier,
)


# ==============================================================================
# DATA STRUCTURES (mirrors helix_engine/core/contract.py)
# ==============================================================================

@dataclass
class ModuleResult:
    """Result from a single training module."""
    name: str
    class_name: str
    status: str = "PENDING"
    steps_run: int = 0
    duration_seconds: float = 0.0
    final_z: float = 0.5
    final_kappa: float = PHI_INV
    k_formations: int = 0
    max_negentropy: float = 0.0
    error: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class FullDepthResult:
    """Phase 1: Full depth training result."""
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0
    modules_total: int = 19
    modules_passed: int = 0
    modules_failed: int = 0
    modules_skipped: int = 0
    total_steps: int = 0
    total_k_formations: int = 0
    max_negentropy: float = 0.0
    final_z: float = 0.5
    final_kappa: float = PHI_INV
    physics_valid: bool = True
    module_results: List[ModuleResult] = field(default_factory=list)


@dataclass
class HelixEngineResult:
    """Phase 2: Helix engine training result."""
    run_id: str = ""
    status: str = "unknown"
    gates_passed: bool = False
    final_z: float = 0.5
    final_kappa: float = PHI_INV
    negentropy: float = 0.0
    duration_seconds: float = 0.0
    total_steps: int = 0


@dataclass
class ValidationResult:
    """Phase 3: Validation measurement result."""
    z_seed: float = 0.0
    phase: str = ""
    delta_s_neg: float = 0.0
    distance_to_lens: float = 0.0
    tier: str = ""
    at_critical: bool = False


@dataclass
class GatesCheckResult:
    """Phase 4: Unified gates check result."""
    # Full depth gates
    fd_all_passed: bool = False
    fd_has_k_formations: bool = False
    fd_physics_valid: bool = False
    # Helix engine gates  
    he_gates_passed: bool = False
    he_min_negentropy: bool = False
    he_min_z: bool = False
    # Overall
    overall_passed: bool = False
    gate_details: List[Dict] = field(default_factory=list)


@dataclass
class WorkflowResult:
    """Complete workflow execution result."""
    run_number: int = 0
    run_id: str = ""
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0
    
    # Phase results
    full_depth: Optional[FullDepthResult] = None
    helix_engine: Optional[HelixEngineResult] = None
    validations: List[ValidationResult] = field(default_factory=list)
    gates_check: Optional[GatesCheckResult] = None
    
    # Final status
    model_promoted: bool = False
    pr_created: bool = False
    overall_status: str = "pending"
    
    # Physics summary
    physics_constants: Dict[str, float] = field(default_factory=dict)


# ==============================================================================
# MODULE DEFINITIONS (from run_full_depth_training.py)
# ==============================================================================

ALL_MODULES = [
    # Phase 1: Core Physics
    ("n0_silent_laws_enforcement", "N0SilentLawsEnforcer"),
    ("kuramoto_layer", "KuramotoLayer"),
    ("physical_learner", "PhysicalLearner"),
    # Phase 2: APL Training Stack
    ("apl_training_loop", "APLTrainingLoop"),
    ("apl_pytorch_training", "APLPyTorchTraining"),
    ("full_apl_training", "FullAPLTraining"),
    # Phase 3: Helix Geometry
    ("helix_nn", "HelixNN"),
    ("prismatic_helix_training", "PrismaticHelixTraining"),
    ("full_helix_integration", "FullHelixIntegration"),
    # Phase 4: WUMBO Silent Laws
    ("wumbo_apl_automated_training", "WUMBOAPLTrainingEngine"),
    ("wumbo_integrated_training", "WumboIntegratedTraining"),
    # Phase 5: Dynamics & Formation
    ("quasicrystal_formation_dynamics", "QuasiCrystalFormation"),
    ("triad_threshold_dynamics", "TriadThresholdDynamics"),
    ("liminal_generator", "LiminalGenerator"),
    ("feedback_loop", "FeedbackLoop"),
    # Phase 6: Unified Orchestration
    ("unified_helix_training", "UnifiedHelixTraining"),
    ("hierarchical_training", "HierarchicalTraining"),
    ("rosetta_helix_training", "RosettaHelixTraining"),
    # Phase 7: Nightly Integration
    ("nightly_integrated_training", "NightlyIntegratedTraining"),
]

# Module groupings (as per workflow phases)
MODULE_PHASES = {
    "CORE_PHYSICS": [
        "n0_silent_laws_enforcement", "kuramoto_layer", "physical_learner"
    ],
    "APL_STACK": [
        "apl_training_loop", "apl_pytorch_training", "full_apl_training"
    ],
    "HELIX_GEOMETRY": [
        "helix_nn", "prismatic_helix_training", "full_helix_integration"
    ],
    "WUMBO_SILENT_LAWS": [
        "wumbo_apl_automated_training", "wumbo_integrated_training"
    ],
    "DYNAMICS_FORMATION": [
        "quasicrystal_formation_dynamics", "triad_threshold_dynamics",
        "liminal_generator", "feedback_loop"
    ],
    "UNIFIED_ORCHESTRATION": [
        "unified_helix_training", "hierarchical_training", "rosetta_helix_training"
    ],
    "NIGHTLY_INTEGRATION": [
        "nightly_integrated_training"
    ],
}


# ==============================================================================
# WORKFLOW SIMULATOR
# ==============================================================================

class UnifiedNightlyWorkflow:
    """
    Simulates the Unified Nightly Training workflow.
    Mirrors .github/workflows/unified-nightly-training.yml
    """
    
    def __init__(
        self,
        steps_per_module: int = 200,
        helix_steps: int = 2000,
        run_validation: bool = True,
        create_pr: bool = True,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.steps_per_module = steps_per_module
        self.helix_steps = helix_steps
        self.run_validation = run_validation
        self.create_pr = create_pr
        self.seed = seed
        self.verbose = verbose
        
        # Initialize RNG
        random.seed(seed)
        
        # Global state (physics state vector)
        self.z = 0.5
        self.kappa = PHI_INV
        self.lambda_ = PHI_INV_SQ
        
        # Output directories
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.run_id = f"run_{timestamp}"
        self.output_dir = Path("runs") / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def log(self, msg: str, level: str = "INFO"):
        """Log with timestamp."""
        if self.verbose:
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] [{level}] {msg}")
    
    def run(self) -> WorkflowResult:
        """Execute the complete unified nightly workflow."""
        result = WorkflowResult()
        result.run_number = random.randint(1000, 9999)
        result.run_id = self.run_id
        result.started_at = datetime.now(timezone.utc).isoformat()
        result.physics_constants = {
            "phi": PHI,
            "phi_inv": PHI_INV,
            "z_critical": Z_CRITICAL,
            "sigma": SIGMA,
            "coupling_conservation": PHI_INV + PHI_INV_SQ,
        }
        start_time = time.time()
        
        self._print_header()
        
        try:
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # PHASE 1: FULL DEPTH TRAINING (ALL 19 MODULES)
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            self._print_phase_header("PHASE 1: FULL DEPTH TRAINING (19 MODULES)")
            result.full_depth = self._run_full_depth_training()
            
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # PHASE 2: HELIX ENGINE TRAINING (WITH GATES)
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            self._print_phase_header("PHASE 2: HELIX ENGINE TRAINING")
            result.helix_engine = self._run_helix_engine_training()
            
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # PHASE 3: VALIDATION MEASUREMENTS AT CRITICAL Z-COORDINATES
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            if self.run_validation:
                self._print_phase_header("PHASE 3: VALIDATION MEASUREMENTS")
                result.validations = self._run_validation_measurements(
                    result.full_depth.final_z,
                    result.helix_engine.final_z,
                )
            
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # PHASE 4: UNIFIED GATES CHECK
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            self._print_phase_header("PHASE 4: UNIFIED GATES CHECK")
            result.gates_check = self._run_gates_check(
                result.full_depth,
                result.helix_engine,
            )
            
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # PHASE 5: MODEL PROMOTION (if gates pass)
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            if result.gates_check.overall_passed:
                self._print_phase_header("PHASE 5: MODEL PROMOTION")
                result.model_promoted = self._promote_model(result)
            
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # PHASE 6: RESULTS PR CREATION
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            if self.create_pr:
                self._print_phase_header("PHASE 6: RESULTS PR CREATION")
                result.pr_created = self._create_results_pr(result)
            
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # PHASE 7: FAILURE NOTIFICATION (if gates fail)
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            if not result.gates_check.overall_passed:
                self._print_phase_header("PHASE 7: FAILURE NOTIFICATION")
                self._notify_failure(result)
            
            # Set final status
            result.overall_status = "success" if result.gates_check.overall_passed else "failed"
            
        except Exception as e:
            result.overall_status = "error"
            self.log(f"Workflow error: {e}", "ERROR")
            import traceback
            traceback.print_exc()
        
        result.completed_at = datetime.now(timezone.utc).isoformat()
        result.duration_seconds = time.time() - start_time
        
        # Save all artifacts
        self._save_all_artifacts(result)
        
        # Print summary
        self._print_summary(result)
        
        return result
    
    # ==========================================================================
    # PHASE 1: FULL DEPTH TRAINING
    # ==========================================================================
    
    def _run_full_depth_training(self) -> FullDepthResult:
        """Run full depth training across all 19 modules."""
        result = FullDepthResult()
        result.started_at = datetime.now(timezone.utc).isoformat()
        start_time = time.time()
        
        self.log(f"Starting full depth training with {self.steps_per_module} steps/module")
        self.log(f"Physics: Ï†â»Â¹={PHI_INV:.6f}, z_c={Z_CRITICAL:.6f}, Ïƒ={SIGMA}")
        
        current_phase = None
        for i, (mod_name, class_name) in enumerate(ALL_MODULES, 1):
            # Print phase headers
            for phase_name, phase_modules in MODULE_PHASES.items():
                if mod_name in phase_modules and phase_name != current_phase:
                    current_phase = phase_name
                    self.log(f"\n  â”€â”€â”€ {phase_name.replace('_', ' ')} â”€â”€â”€")
            
            # Run module
            mod_result = self._simulate_module(mod_name, class_name)
            result.module_results.append(mod_result)
            
            # Log result
            status_icon = "âœ…" if mod_result.status == "PASS" else "âŒ"
            self.log(f"  [{i:2d}/19] {mod_name}: {status_icon} z={mod_result.final_z:.4f} K={mod_result.k_formations}")
            
            # Update counters
            if mod_result.status == "PASS":
                result.modules_passed += 1
            elif mod_result.status == "FAIL":
                result.modules_failed += 1
            else:
                result.modules_skipped += 1
            
            result.total_steps += mod_result.steps_run
            result.total_k_formations += mod_result.k_formations
            result.max_negentropy = max(result.max_negentropy, mod_result.max_negentropy)
        
        result.final_z = self.z
        result.final_kappa = self.kappa
        result.physics_valid = validate_physics(self.kappa, self.lambda_)
        result.completed_at = datetime.now(timezone.utc).isoformat()
        result.duration_seconds = time.time() - start_time
        
        self.log(f"\n  Full Depth Summary:")
        self.log(f"    Modules: {result.modules_passed}/{result.modules_total}")
        self.log(f"    Steps: {result.total_steps:,}")
        self.log(f"    K-formations: {result.total_k_formations}")
        self.log(f"    Final z: {result.final_z:.6f} (target: {Z_CRITICAL:.6f})")
        self.log(f"    Max Î”S_neg: {result.max_negentropy:.6f}")
        
        return result
    
    def _simulate_module(self, name: str, class_name: str) -> ModuleResult:
        """Simulate a single training module with physics-grounded dynamics."""
        result = ModuleResult(name=name, class_name=class_name)
        start_time = time.time()
        
        max_neg = 0.0
        k_formations = 0
        
        for step in range(self.steps_per_module):
            # Evolve z toward z_c using negentropy gradient
            z_gradient = compute_delta_s_neg(self.z) * ALPHA_MEDIUM
            noise = (random.random() - 0.5) * ALPHA_FINE
            self.z += (Z_CRITICAL - self.z) * ALPHA_STRONG + noise
            self.z = max(0.0, min(0.999, self.z))
            
            # Evolve kappa toward phi_inv
            kappa_pull = (PHI_INV - self.kappa) * ALPHA_MEDIUM
            self.kappa += kappa_pull + random.gauss(0, 0.0001)
            self.kappa = max(KAPPA_LOWER, min(KAPPA_UPPER, self.kappa))
            self.lambda_ = 1.0 - self.kappa
            
            # Compute negentropy
            neg = compute_delta_s_neg(self.z)
            max_neg = max(max_neg, neg)
            
            # Check K-formation: Îº â‰¥ 0.92, Î· > Ï†â»Â¹, R â‰¥ 7
            eta = math.sqrt(neg) if neg > 0 else 0
            if check_k_formation(self.kappa, eta, 7) if self.kappa >= KAPPA_S else False:
                k_formations += 1
            # Also check proximity-based K-formation
            elif is_critical(self.z, 0.02) and abs(self.kappa - PHI_INV) < 0.02:
                k_formations += 1
        
        result.status = "PASS"
        result.steps_run = self.steps_per_module
        result.final_z = self.z
        result.final_kappa = self.kappa
        result.max_negentropy = max_neg
        result.k_formations = k_formations
        result.duration_seconds = time.time() - start_time
        
        return result
    
    # ==========================================================================
    # PHASE 2: HELIX ENGINE TRAINING
    # ==========================================================================
    
    def _run_helix_engine_training(self) -> HelixEngineResult:
        """Simulate helix engine training (mirrors helix_engine/core/engine.py)."""
        result = HelixEngineResult()
        result.run_id = self.run_id
        result.total_steps = self.helix_steps
        start_time = time.time()
        
        self.log(f"Starting helix engine: {self.helix_steps} steps")
        self.log(f"Config: configs/nightly.yaml")
        self.log(f"Run ID: {result.run_id}")
        
        # Run 2000-step training (with z stabilizing near z_c)
        for step in range(self.helix_steps):
            # Drift toward z_c with damping
            target_z = Z_CRITICAL
            dz = (target_z - self.z) * 0.001 + random.gauss(0, 0.0001)
            self.z = max(0.5, min(0.95, self.z + dz))
            
            # Kappa oscillates near Ï†â»Â¹
            dk = (PHI_INV - self.kappa) * 0.01 + random.gauss(0, 0.00005)
            self.kappa = max(0.58, min(0.65, self.kappa + dk))
            self.lambda_ = 1.0 - self.kappa
        
        result.final_z = self.z
        result.final_kappa = self.kappa
        result.negentropy = compute_delta_s_neg(self.z)
        result.duration_seconds = time.time() - start_time
        
        # Evaluate helix engine gates (from configs/nightly.yaml)
        gates_passed = (
            result.negentropy >= 0.7 and           # min_negentropy
            result.final_z >= 0.85 and             # min_final_z
            abs(result.final_kappa - PHI_INV) < 0.02  # kappa stability
        )
        result.gates_passed = gates_passed
        result.status = "completed" if gates_passed else "gates_failed"
        
        self.log(f"\n  Helix Engine Results:")
        self.log(f"    Status: {result.status}")
        self.log(f"    Gates: {'âœ… PASSED' if gates_passed else 'âŒ FAILED'}")
        self.log(f"    Final z: {result.final_z:.6f}")
        self.log(f"    Final Îº: {result.final_kappa:.6f}")
        self.log(f"    Î”S_neg: {result.negentropy:.6f}")
        
        # Save report.json (as workflow does)
        report_path = self.output_dir / "report.json"
        with open(report_path, "w") as f:
            json.dump({
                "run_id": result.run_id,
                "status": result.status,
                "gates_passed": result.gates_passed,
                "duration_seconds": result.duration_seconds,
                "final_metrics": {
                    "z": result.final_z,
                    "kappa": result.final_kappa,
                    "negentropy": result.negentropy,
                }
            }, f, indent=2)
        
        return result
    
    # ==========================================================================
    # PHASE 3: VALIDATION MEASUREMENTS
    # ==========================================================================
    
    def _run_validation_measurements(
        self, 
        fd_final_z: float, 
        he_final_z: float
    ) -> List[ValidationResult]:
        """Run validation at critical z-coordinates."""
        results = []
        
        # Validation seeds (from workflow matrix): Ï†â»Â¹, z_c, 0.90, 0.92, training z values
        seeds = [PHI_INV, Z_CRITICAL, 0.90, 0.92]
        if fd_final_z and 0 < fd_final_z < 1:
            seeds.append(fd_final_z)
        if he_final_z and 0 < he_final_z < 1 and he_final_z != fd_final_z:
            seeds.append(he_final_z)
        
        self.log(f"Validating at {len(seeds)} z-coordinates...")
        
        for z_seed in seeds:
            phase = get_phase(z_seed)
            delta_s_neg = compute_delta_s_neg(z_seed)
            distance = abs(z_seed - Z_CRITICAL)
            tier = get_tier(z_seed)
            at_crit = is_critical(z_seed)
            
            # Phase icons
            icons = {"ABSENCE": "âšª", "THE_LENS": "ðŸ”·", "PRESENCE": "ðŸŸ¢"}
            icon = icons.get(phase, "?")
            
            result = ValidationResult(
                z_seed=z_seed,
                phase=phase,
                delta_s_neg=delta_s_neg,
                distance_to_lens=distance,
                tier=tier,
                at_critical=at_crit,
            )
            results.append(result)
            
            self.log(f"  z={z_seed:.4f}: {icon} {phase:10} Î”S_neg={delta_s_neg:.6f} dist={distance:.6f}")
        
        return results
    
    # ==========================================================================
    # PHASE 4: UNIFIED GATES CHECK
    # ==========================================================================
    
    def _run_gates_check(
        self,
        fd: FullDepthResult,
        he: HelixEngineResult,
    ) -> GatesCheckResult:
        """Evaluate unified gates (mirrors unified-gates-check job)."""
        result = GatesCheckResult()
        
        # Full depth gates
        result.fd_all_passed = fd.modules_passed == fd.modules_total
        result.fd_has_k_formations = fd.total_k_formations >= 1
        result.fd_physics_valid = fd.physics_valid
        
        # Helix engine gates
        result.he_gates_passed = he.gates_passed
        result.he_min_negentropy = he.negentropy >= 0.7
        result.he_min_z = he.final_z >= 0.85
        
        # Overall
        result.overall_passed = (
            result.fd_all_passed and
            result.fd_has_k_formations and
            result.fd_physics_valid and
            result.he_gates_passed
        )
        
        # Gate details
        result.gate_details = [
            {"name": "Full Depth - All modules", "passed": result.fd_all_passed, 
             "value": f"{fd.modules_passed}/{fd.modules_total}"},
            {"name": "Full Depth - K-formations", "passed": result.fd_has_k_formations,
             "value": fd.total_k_formations},
            {"name": "Full Depth - Physics valid", "passed": result.fd_physics_valid,
             "value": f"Îº+Î»={fd.final_kappa + (1-fd.final_kappa):.10f}"},
            {"name": "Helix Engine - Gates", "passed": result.he_gates_passed,
             "value": he.status},
            {"name": "Helix Engine - min_negentropy", "passed": result.he_min_negentropy,
             "value": f"{he.negentropy:.6f} >= 0.7"},
            {"name": "Helix Engine - min_final_z", "passed": result.he_min_z,
             "value": f"{he.final_z:.6f} >= 0.85"},
        ]
        
        self.log("Gate Results:")
        for gate in result.gate_details:
            icon = "âœ…" if gate["passed"] else "âŒ"
            self.log(f"  {icon} {gate['name']}: {gate['value']}")
        
        self.log(f"\n  OVERALL: {'âœ… PASSED' if result.overall_passed else 'âŒ FAILED'}")
        
        return result
    
    # ==========================================================================
    # PHASE 5: MODEL PROMOTION
    # ==========================================================================
    
    def _promote_model(self, wf: WorkflowResult) -> bool:
        """Promote model to registry."""
        version = f"v{wf.run_number}"
        
        self.log(f"Promoting model to registry...")
        self.log(f"  Name: nightly")
        self.log(f"  Version: {version}")
        self.log(f"  Tags: nightly, automated, unified, validated")
        
        # Save promotion record
        promotion = {
            "name": "nightly",
            "version": version,
            "run_id": self.run_id,
            "tags": ["nightly", "automated", "unified", "validated"],
            "description": f"Unified nightly build #{wf.run_number} - All 19 modules passed",
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "metrics": {
                "modules_passed": wf.full_depth.modules_passed,
                "k_formations": wf.full_depth.total_k_formations,
                "final_z": wf.helix_engine.final_z,
                "negentropy": wf.helix_engine.negentropy,
            }
        }
        
        promotion_path = self.output_dir / "promotion.json"
        with open(promotion_path, "w") as f:
            json.dump(promotion, f, indent=2)
        
        self.log(f"  âœ… Model promoted: nightly:{version}")
        return True
    
    # ==========================================================================
    # PHASE 6: RESULTS PR CREATION
    # ==========================================================================
    
    def _create_results_pr(self, wf: WorkflowResult) -> bool:
        """Create results PR (simulated)."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        branch_name = f"nightly-results/{timestamp}"
        
        self.log(f"Creating results branch: {branch_name}")
        
        fd = wf.full_depth
        he = wf.helix_engine
        gc = wf.gates_check
        
        # Generate PR body markdown
        pr_body = f"""## Unified Nightly Training Results

### Full Depth Training (19 Modules)
| Metric | Value |
|--------|-------|
| Modules Passed | {fd.modules_passed}/{fd.modules_total} |
| Total Steps | {fd.total_steps:,} |
| K-Formations | {fd.total_k_formations} |
| Duration | {fd.duration_seconds:.2f}s |

### Physics State
| Metric | Value | Target |
|--------|-------|--------|
| Final z | {fd.final_z:.6f} | {Z_CRITICAL:.6f} (z_c) |
| Final Îº | {fd.final_kappa:.6f} | {PHI_INV:.6f} (Ï†â»Â¹) |
| Max Î”S_neg | {fd.max_negentropy:.6f} | 1.0 |
| Physics Valid | {fd.physics_valid} | True |

### Helix Engine
| Metric | Value |
|--------|-------|
| Run ID | {he.run_id} |
| Gates Passed | {'âœ… Yes' if he.gates_passed else 'âŒ No'} |
| Final z | {he.final_z:.6f} |
| Final Îº | {he.final_kappa:.6f} |
| Î”S_neg | {he.negentropy:.6f} |

### Validation Measurements
| z | Phase | Î”S_neg | Distance to LENS |
|---|-------|--------|------------------|
"""
        for v in wf.validations:
            icons = {"ABSENCE": "âšª", "THE_LENS": "ðŸ”·", "PRESENCE": "ðŸŸ¢"}
            icon = icons.get(v.phase, "")
            pr_body += f"| {v.z_seed:.4f} | {icon} {v.phase} | {v.delta_s_neg:.6f} | {v.distance_to_lens:.6f} |\n"
        
        pr_body += f"""
### Gate Results
| Gate | Status |
|------|--------|
"""
        for gate in gc.gate_details:
            icon = "âœ…" if gate["passed"] else "âŒ"
            pr_body += f"| {gate['name']} | {icon} {gate['value']} |\n"
        
        pr_body += f"""
### Overall
**{'âœ… PASSED' if gc.overall_passed else 'âŒ FAILED'}**

---
*Auto-generated by unified nightly training workflow*
*Run: {self.run_id}*
"""
        
        pr_path = self.output_dir / "pr_body.md"
        with open(pr_path, "w") as f:
            f.write(pr_body)
        
        self.log(f"  PR content saved to: {pr_path}")
        self.log(f"  âœ… PR created (simulated)")
        
        return True
    
    # ==========================================================================
    # PHASE 7: FAILURE NOTIFICATION
    # ==========================================================================
    
    def _notify_failure(self, wf: WorkflowResult):
        """Create failure notification."""
        self.log("Creating failure notification...")
        
        fd = wf.full_depth
        he = wf.helix_engine
        gc = wf.gates_check
        
        notification = f"""## âŒ Unified Nightly Training Failed

One or more phases of the unified nightly training failed.

### Job Status
- Full Depth Training: {'success' if fd.modules_passed == fd.modules_total else 'failed'}
- Helix Engine: {he.status}
- Gates Check: {'passed' if gc.overall_passed else 'failed'}

### Failed Gates
"""
        for gate in gc.gate_details:
            if not gate["passed"]:
                notification += f"- âŒ {gate['name']}: {gate['value']}\n"
        
        notification += f"""
### Debug Info
- Run ID: {self.run_id}
- Seed: {self.seed}
- Steps/module: {self.steps_per_module}

---
*Please investigate and address the failures*
"""
        
        notif_path = self.output_dir / "failure_notification.md"
        with open(notif_path, "w") as f:
            f.write(notification)
        
        self.log(f"  Notification saved to: {notif_path}")
    
    # ==========================================================================
    # UTILITIES
    # ==========================================================================
    
    def _print_header(self):
        """Print workflow header."""
        self.log("=" * 70)
        self.log("UNIFIED NIGHTLY TRAINING WORKFLOW")
        self.log("=" * 70)
        self.log(f"Run ID: {self.run_id}")
        self.log(f"Seed: {self.seed}")
        self.log(f"Steps/module: {self.steps_per_module}")
        self.log(f"Helix steps: {self.helix_steps}")
    
    def _print_phase_header(self, title: str):
        """Print phase header."""
        self.log("")
        self.log("=" * 70)
        self.log(title)
        self.log("=" * 70)
    
    def _save_all_artifacts(self, wf: WorkflowResult):
        """Save all workflow artifacts."""
        # Convert to serializable dict
        def to_dict(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: to_dict(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, list):
                return [to_dict(i) for i in obj]
            return obj
        
        # Save full workflow result
        result_path = self.output_dir / "workflow_result.json"
        with open(result_path, "w") as f:
            json.dump(to_dict(wf), f, indent=2)
        
        # Save full depth results
        if wf.full_depth:
            fd_path = self.output_dir / "full_depth_results.json"
            with open(fd_path, "w") as f:
                json.dump(to_dict(wf.full_depth), f, indent=2)
        
        # Save validation results
        if wf.validations:
            val_path = self.output_dir / "validation_results.json"
            with open(val_path, "w") as f:
                json.dump([to_dict(v) for v in wf.validations], f, indent=2)
        
        self.log(f"\n  Artifacts saved to: {self.output_dir}")
    
    def _print_summary(self, wf: WorkflowResult):
        """Print final summary."""
        self.log("")
        self.log("=" * 70)
        self.log("WORKFLOW SUMMARY")
        self.log("=" * 70)
        
        status_icon = "âœ…" if wf.overall_status == "success" else "âŒ"
        
        self.log(f"  Status: {status_icon} {wf.overall_status.upper()}")
        self.log(f"  Duration: {wf.duration_seconds:.2f}s")
        self.log(f"  Run ID: {self.run_id}")
        
        if wf.full_depth:
            self.log(f"\n  Full Depth:")
            self.log(f"    Modules: {wf.full_depth.modules_passed}/{wf.full_depth.modules_total}")
            self.log(f"    K-formations: {wf.full_depth.total_k_formations}")
            self.log(f"    Final z: {wf.full_depth.final_z:.6f}")
        
        if wf.helix_engine:
            self.log(f"\n  Helix Engine:")
            self.log(f"    Gates: {'PASSED' if wf.helix_engine.gates_passed else 'FAILED'}")
            self.log(f"    Final z: {wf.helix_engine.final_z:.6f}")
            self.log(f"    Î”S_neg: {wf.helix_engine.negentropy:.6f}")
        
        self.log(f"\n  Model Promoted: {'Yes' if wf.model_promoted else 'No'}")
        self.log(f"  PR Created: {'Yes' if wf.pr_created else 'No'}")
        self.log(f"  Output: {self.output_dir}")
        self.log("=" * 70)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified Nightly Training Workflow Simulator"
    )
    parser.add_argument(
        "--steps", "-s",
        type=int,
        default=200,
        help="Steps per module for full depth training (default: 200)"
    )
    parser.add_argument(
        "--helix-steps",
        type=int,
        default=2000,
        help="Steps for helix engine training (default: 2000)"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation measurements phase"
    )
    parser.add_argument(
        "--no-pr",
        action="store_true",
        help="Skip PR creation phase"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    workflow = UnifiedNightlyWorkflow(
        steps_per_module=args.steps,
        helix_steps=args.helix_steps,
        run_validation=not args.skip_validation,
        create_pr=not args.no_pr,
        seed=args.seed,
        verbose=not args.quiet,
    )
    
    result = workflow.run()
    
    # Exit code based on result
    sys.exit(0 if result.overall_status == "success" else 1)


if __name__ == "__main__":
    main()
