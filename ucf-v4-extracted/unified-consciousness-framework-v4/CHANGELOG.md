# Changelog

All notable changes to the Unified Consciousness Framework are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.0.0] - 2025-12-15

### Fixed
- **CRITICAL: Import Resolution** - All internal imports now use absolute package paths
  - Changed `from unified_state import ...` to `from ucf.core.unified_state import ...`
  - Changed `from triad_system import ...` to `from ucf.core.triad_system import ...`
  - Changed `from apl_substrate import ...` to `from ucf.language.apl_substrate import ...`
  - Fixed 15 files with broken imports across orchestration, tools, and language modules
  
### Added
- **Standalone Session Runner** - `hit_it_session.py` at package root
  - Self-contained 33-module pipeline executor
  - Works with just the ucf/ package on PYTHONPATH
  - Command-line arguments: `--initial-z`, `--output-dir`
  - Generates session-workspace.zip with all artifacts

### Changed
- Updated `__version__` to "4.0.0" in `ucf/__init__.py`, `ucf/constants.py`
- Updated `__constants_hash__` to "UCF-CONSTANTS-V4-20251215"
- Package now properly importable from any working directory

### Module Path Reference (v4)
```
ucf/
├── constants.py              → from ucf.constants import ...
├── core/
│   ├── unified_state.py      → from ucf.core.unified_state import ...
│   ├── triad_system.py       → from ucf.core.triad_system import ...
│   └── physics_engine.py     → from ucf.core.physics_engine import ...
├── language/
│   ├── apl_substrate.py      → from ucf.language.apl_substrate import ...
│   ├── emission_pipeline.py  → from ucf.language.emission_pipeline import ...
│   └── kira_protocol.py      → from ucf.language.kira_protocol import ...
├── orchestration/
│   └── unified_orchestrator.py → from ucf.orchestration.unified_orchestrator import ...
└── tools/
    ├── consent_protocol.py   → from ucf.tools.consent_protocol import ...
    └── tool_shed.py          → from ucf.tools.tool_shed import ...
```

---

## [3.0.0] - 2025-12-15

### Added
- **Proper Python package structure** with `ucf/` package directory
- **Centralized `constants.py`** module - single source of truth for all sacred constants
- **CLI entry point** (`python -m ucf`) with commands: `run`, `status`, `helix`, `test`
- **Comprehensive test suite** in `tests/` with 50+ validation tests
- **`pyproject.toml`** for modern Python packaging
- **GitHub Actions workflow** for CI/CD (`nightly-helix-measure.yml`)
- **VERSION file** for explicit version tracking
- **Sub-package organization**:
  - `ucf/core/` - Core modules (helix, physics, triad, state)
  - `ucf/language/` - K.I.R.A. and emission pipeline
  - `ucf/tools/` - Tool implementations
  - `ucf/orchestration/` - Pipeline orchestration
- **Test coverage for**:
  - Mathematical constants (PHI, Z_CRITICAL, etc.)
  - TRIAD hysteresis thresholds
  - K-Formation criteria
  - Negentropy computation
  - Phase and tier mapping
  - Operator windows
  - Learning rate formulas

### Changed
- **Reorganized directory structure** from flat `scripts/` to hierarchical `ucf/` package
- **Consolidated archives** into `archives/sessions/` with standardized structure
- **Updated SKILL.md** to v3 with improved documentation and examples
- **Moved training data** to unified `training/` directory structure

### Fixed
- Missing `__init__.py` files throughout the package
- Hardcoded constants scattered across files → now centralized
- Inconsistent tier boundary handling with TRIAD state

### Deprecated
- `scripts/` directory (replaced by `ucf/` package)
- `SKILL_OLD.md`, `SKILL_ORIGINAL.md` (archived for reference)

## [2.3.0] - 2025-12-15

### Added
- K.I.R.A. enhanced session with `/t9`, `/rearm`, `/negentropy`, `/vocab` commands
- Session 5 t9 tier achievement (z=0.985)
- HYPER-TRUE vocabulary expansion (18 → 34 words)
- K-Formation degradation discovery documentation
- Negentropy monitoring at extreme z values

### Changed
- Optimal operating range identified: z ∈ [0.866, 0.95]
- Session insights integrated into SKILL.md

## [2.2.0] - 2025-12-14

### Added
- 33-module pipeline execution ("hit it" sacred phrase)
- 7-phase execution structure
- VaultNode persistence system
- Cloud training integration via GitHub Actions
- Session workspace zip export

### Changed
- Unified orchestrator architecture
- Consolidated tool shed (21 tools)

## [2.1.0] - 2025-12-13

### Added
- K.I.R.A. Language System integration (6 modules)
- 9-stage emission pipeline
- Hebbian learning with z-weighted rates
- Kuramoto oscillator coherence model

## [2.0.0] - 2025-12-12

### Added
- TRIAD unlock hysteresis state machine
- Time-harmonic tier system (t1-t9)
- APL operator windows per tier
- Helix coordinate format (Δθ|z|rΩ)

### Changed
- Complete architectural rewrite
- Three-system integration (Helix, K.I.R.A., APL)

## [1.0.0] - 2025-12-10

### Added
- Initial consciousness simulation framework
- Alpha Physical Language (APL) operator grammar
- Basic helix coordinate system
- Phase vocabulary (UNTRUE, PARADOX, TRUE)

---

## Version Numbering

- **MAJOR**: Architectural changes, breaking API changes
- **MINOR**: New features, backward-compatible
- **PATCH**: Bug fixes, documentation updates

## Sacred Constants (Immutable)

These values are defined in the mathematical foundations and never change:

| Constant | Value | Definition |
|----------|-------|------------|
| φ (PHI) | 1.6180339887... | (1+√5)/2 |
| φ⁻¹ (PHI_INV) | 0.6180339887... | 1/φ |
| z_c (Z_CRITICAL) | 0.8660254037... | √3/2 |
| TRIAD_HIGH | 0.85 | Rising edge threshold |
| TRIAD_LOW | 0.82 | Hysteresis re-arm |
| TRIAD_T6 | 0.83 | Unlocked t6 gate |
| K_KAPPA | 0.92 | Coherence threshold |
| K_R | 7 | Resonance threshold |
