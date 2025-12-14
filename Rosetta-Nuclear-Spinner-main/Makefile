# Nuclear Spinner × Rosetta-Helix Monorepo
# Signature: monorepo-makefile|v1.0.0|helix

.PHONY: all install install-dev clean test lint format firmware bridge rosetta training run help

# Default target
all: install

# ═══════════════════════════════════════════════════════════════════════════
# INSTALLATION
# ═══════════════════════════════════════════════════════════════════════════

install:
	@echo "Installing Nuclear Spinner × Rosetta-Helix..."
	pip install -e .

install-dev:
	@echo "Installing with development dependencies..."
	pip install -e ".[dev]"

install-all:
	@echo "Installing with all optional dependencies..."
	pip install -e ".[all]"

# ═══════════════════════════════════════════════════════════════════════════
# FIRMWARE
# ═══════════════════════════════════════════════════════════════════════════

firmware:
	@echo "Building firmware..."
	$(MAKE) -C firmware

firmware-clean:
	$(MAKE) -C firmware clean

firmware-flash:
	$(MAKE) -C firmware flash

firmware-sim:
	@echo "Building firmware simulation..."
	$(MAKE) -C firmware sim
	./firmware/build/sim_test

# ═══════════════════════════════════════════════════════════════════════════
# BRIDGE SERVICE
# ═══════════════════════════════════════════════════════════════════════════

bridge:
	@echo "Starting bridge service..."
	python bridge/spinner_bridge.py

bridge-sim:
	@echo "Starting bridge in simulation mode..."
	python bridge/spinner_bridge.py --simulate

# ═══════════════════════════════════════════════════════════════════════════
# ROSETTA-HELIX
# ═══════════════════════════════════════════════════════════════════════════

rosetta:
	@echo "Starting Rosetta-Helix node..."
	python -m rosetta_helix.node

rosetta-test:
	@echo "Testing Rosetta-Helix components..."
	python -c "from rosetta_helix.physics import validate_all_constants; print(validate_all_constants())"
	python -c "from rosetta_helix.heart import test_heart; test_heart()"

# ═══════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════

training:
	@echo "Running unified nightly workflow..."
	python -m training.src.unified_workflow

training-quick:
	@echo "Running quick training (50 steps)..."
	python -m training.src.unified_workflow --steps 50 --helix-steps 500

# ═══════════════════════════════════════════════════════════════════════════
# FULL SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

run: run-sim
	@echo "System running..."

run-sim:
	@echo "Starting full system in simulation mode..."
	./scripts/start_system.sh

run-hw:
	@echo "Starting full system with hardware..."
	./scripts/start_system.sh --hardware

# ═══════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════

test:
	@echo "Running all tests..."
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-physics:
	@echo "Validating physics constants..."
	python -c "
from rosetta_helix.physics import validate_all_constants, print_all_constants
print_all_constants()
result = validate_all_constants()
print(f'\nAll valid: {result[\"all_valid\"]}')
assert result['all_valid'], 'Physics validation failed!'
"

# ═══════════════════════════════════════════════════════════════════════════
# CODE QUALITY
# ═══════════════════════════════════════════════════════════════════════════

lint:
	@echo "Linting code..."
	ruff check rosetta-helix/ bridge/ training/
	mypy rosetta-helix/src/ --ignore-missing-imports

format:
	@echo "Formatting code..."
	black rosetta-helix/ bridge/ training/
	ruff check --fix rosetta-helix/ bridge/ training/

# ═══════════════════════════════════════════════════════════════════════════
# CLEANUP
# ═══════════════════════════════════════════════════════════════════════════

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf training/runs/*
	$(MAKE) -C firmware clean

clean-all: clean
	rm -rf venv/

# ═══════════════════════════════════════════════════════════════════════════
# HELP
# ═══════════════════════════════════════════════════════════════════════════

help:
	@echo "Nuclear Spinner × Rosetta-Helix Monorepo"
	@echo ""
	@echo "Installation:"
	@echo "  make install       Install package"
	@echo "  make install-dev   Install with dev dependencies"
	@echo "  make install-all   Install with all dependencies"
	@echo ""
	@echo "Firmware:"
	@echo "  make firmware      Build firmware"
	@echo "  make firmware-sim  Build and run firmware simulation"
	@echo "  make firmware-flash Flash to STM32"
	@echo ""
	@echo "Services:"
	@echo "  make bridge        Start bridge service"
	@echo "  make bridge-sim    Start bridge in simulation mode"
	@echo "  make rosetta       Start Rosetta-Helix node"
	@echo ""
	@echo "Training:"
	@echo "  make training      Run unified nightly workflow"
	@echo "  make training-quick Run quick training"
	@echo ""
	@echo "System:"
	@echo "  make run           Start full system (simulation)"
	@echo "  make run-hw        Start full system (hardware)"
	@echo ""
	@echo "Testing:"
	@echo "  make test          Run all tests"
	@echo "  make test-physics  Validate physics constants"
	@echo ""
	@echo "Quality:"
	@echo "  make lint          Run linters"
	@echo "  make format        Format code"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         Clean build artifacts"
	@echo ""
	@echo "Physics Constants:"
	@echo "  φ = 1.618034       Golden ratio"
	@echo "  φ⁻¹ = 0.618034     κ attractor"
	@echo "  z_c = 0.866025     THE LENS (√3/2)"
	@echo "  σ = 36             Gaussian width"
