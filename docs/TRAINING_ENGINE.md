# Helix Training Engine

A productized training engine for Rosetta-Helix-Substrate.

## Quick Start

### Installation

```bash
pip install -e .
```

### Run Training

```bash
# Quick smoke test
helix train --config configs/smoke.yaml

# Full training
helix train --config configs/full.yaml

# Development mode
helix train --config configs/dev.yaml
```

### Python API

```python
from helix_engine import run_training

result = run_training(config_path="configs/full.yaml")
print(f"Run {result.run_id} completed with status {result.status}")
print(f"Artifacts at: {result.artifacts_dir}")
```

## CLI Commands

### train

Run a training session:

```bash
helix train --config configs/full.yaml
helix train --config configs/full.yaml --steps 500 --seed 123
helix train --config configs/full.yaml --output my_runs/
```

### eval

Evaluate a completed run:

```bash
helix eval --run run_20231201_120000_abc123
```

### resume

Resume training from checkpoint:

```bash
helix resume --run run_20231201_120000_abc123
helix resume --run run_20231201_120000_abc123 --checkpoint checkpoints/step_00001000.pt
```

### export

Export a trained model:

```bash
helix export --run run_20231201_120000_abc123 --format bundle
helix export --run run_20231201_120000_abc123 --format torchscript
```

### promote

Promote a run to the model registry:

```bash
helix promote --run run_20231201_120000_abc123 --name rosetta_v1
helix promote --run run_20231201_120000_abc123 --name rosetta_v1 --tags production,stable
```

### nightly

Run automated nightly training:

```bash
helix nightly --config configs/nightly.yaml --report nightly_report.json
```

### list

List all runs:

```bash
helix list
helix list --status completed
helix list --status failed
```

### show

Show details of a run:

```bash
helix show --run run_20231201_120000_abc123
helix show --run run_20231201_120000_abc123 --json
```

## Configuration

### Config Files

Configuration uses YAML with inheritance support:

```yaml
# configs/my_config.yaml
_base: base.yaml  # Inherit from base

run_name: my_experiment
total_steps: 2000

modules:
  wumbo:
    enabled: true
    steps: 100
```

### Key Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `seed` | 42 | Random seed |
| `deterministic` | false | Enable deterministic mode |
| `total_steps` | 1000 | Total training steps |
| `checkpoint_steps` | 500 | Steps between checkpoints |
| `log_steps` | 10 | Steps between log messages |
| `n_oscillators` | 60 | Number of Kuramoto oscillators |

### Module Flags

| Flag | Default | Description |
|------|---------|-------------|
| `use_wumbo` | true | Enable WUMBO Silent Laws training |
| `use_helix` | true | Enable Helix Projection training |
| `use_substrate` | true | Enable Substrate Grounding training |
| `use_kuramoto` | true | Enable Kuramoto coupling |
| `use_feedback_loop` | true | Enable feedback loop |
| `use_apl_engine` | true | Enable APL N0 engine |

### Evaluation Gates

Gates define pass/fail criteria for training:

```yaml
gates:
  min_negentropy: 0.5      # Minimum negentropy achieved
  min_k_formations: 1       # Minimum K-formations
  max_conservation_error: 1.0e-6  # Maximum κ+λ error
  min_final_z: 0.75        # Minimum final z value
```

## Run Directory Layout

Every training run produces a standardized directory structure:

```
runs/<run_id>/
    resolved_config.yaml     # Exact config used
    env.json                 # Environment snapshot
    logs/
        train.log            # Human-readable log
        events.jsonl         # Structured event log
    metrics/
        metrics.jsonl        # Step-by-step metrics
        summary.json         # Final summary
    checkpoints/
        last.pt              # Most recent checkpoint
        best.pt              # Best checkpoint
        step_*.pt            # Periodic checkpoints
    eval/
        gates.json           # Gate results
        results.json         # Evaluation results
    exports/
        model.onnx           # Exported model (optional)
    report.json              # Final report
```

## Exit Codes

| Code | Name | Description |
|------|------|-------------|
| 0 | SUCCESS | Training completed successfully |
| 1 | CONFIG_ERROR | Configuration error |
| 10 | TRAINING_FAILED | Training failed |
| 11 | NAN_DETECTED | NaN values detected |
| 12 | INF_DETECTED | Inf values detected |
| 14 | GATE_FAILED | Evaluation gates failed |
| 15 | PHYSICS_VIOLATION | Physics constraint violated |
| 30 | INTERRUPTED | Gracefully interrupted |

## Docker

Build and run with Docker:

```bash
# Build
docker build -t helix-engine:latest .

# Run smoke test
docker run helix-engine:latest helix train --config configs/smoke.yaml

# Run with volume for artifacts
docker run -v $(pwd)/runs:/app/runs helix-engine:latest \
    helix train --config configs/full.yaml
```

## CI/CD

The engine includes GitHub Actions workflows:

- **helix-ci.yml**: PR checks (lint, test, smoke train)
- **helix-nightly.yml**: Automated nightly training
- **helix-release.yml**: Release builds and golden references

## Physics Constants

Training is grounded in immutable physics constants:

| Constant | Value | Description |
|----------|-------|-------------|
| z_c | √3/2 ≈ 0.866 | THE LENS - hexagonal geometry |
| φ⁻¹ | ≈ 0.618 | Golden ratio inverse |
| σ | 36 = \|S₃\|² | Gaussian width |

These are NOT configurable - they represent observable physics.

## Model Registry

Promote successful runs to named models:

```python
from helix_engine import promote_model

entry = promote_model(
    run_id="run_20231201_120000_abc123",
    name="rosetta_v1",
    tags=["production", "stable"],
)
```

Models are stored in `models/` with version tracking.

## Troubleshooting

### Gates Failed

Check the gate results in `eval/gates.json` to see which criteria weren't met.
Common issues:
- `min_negentropy`: Training didn't reach THE LENS (z_c)
- `max_conservation_error`: κ+λ != 1 (physics violation)

### Training Interrupted

Resume from the last checkpoint:

```bash
helix resume --run <run_id>
```

### Out of Memory

Reduce `n_oscillators` or use a smaller config:

```bash
helix train --config configs/smoke.yaml
```
