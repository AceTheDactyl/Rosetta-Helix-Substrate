# Security Instructions

This document outlines security considerations, potential misuse vectors, and safeguards for the Quantum-APL codebase.

---

## Secure Coding Practices

### Input Validation

All external inputs must be validated before processing:

```python
# GOOD: Validate z-coordinate bounds
def process_z(z: float) -> float:
    if not isinstance(z, (int, float)):
        raise TypeError("z must be numeric")
    if not 0.0 <= z <= 1.0:
        raise ValueError("z must be in [0, 1]")
    return float(z)

# BAD: No validation
def process_z(z):
    return z * 2  # Accepts anything
```

### Environment Variable Handling

Environment variables can be attack vectors if not sanitized:

```python
# GOOD: Validate and bound environment inputs
import os

def get_sigma() -> float:
    raw = os.environ.get("QAPL_LENS_SIGMA", "36.0")
    try:
        sigma = float(raw)
        if not 1.0 <= sigma <= 1000.0:
            raise ValueError
        return sigma
    except (ValueError, TypeError):
        return 36.0  # Safe default

# BAD: Direct use without validation
sigma = float(os.environ.get("QAPL_LENS_SIGMA"))  # Can crash or overflow
```

### Random Seed Security

Fixed seeds enable reproducibility but can be exploited:

```python
# GOOD: Validate seed source and range
import os
import secrets

def get_seed() -> int:
    raw = os.environ.get("QAPL_RANDOM_SEED")
    if raw is None:
        return secrets.randbelow(2**32)  # Cryptographic random
    try:
        seed = int(raw)
        if not 0 <= seed < 2**32:
            raise ValueError
        return seed
    except ValueError:
        return secrets.randbelow(2**32)

# BAD: Predictable or unbounded seeds
seed = int(os.environ.get("QAPL_RANDOM_SEED", "12345"))  # Always same default
```

---

## Injection Attack Prevention

### Command Injection

Never pass unsanitized input to shell commands:

```python
import subprocess
import shlex

# GOOD: Use subprocess with list arguments
def run_analysis(filename: str) -> str:
    # Validate filename first
    if not filename.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Invalid filename")
    result = subprocess.run(
        ["python", "-m", "analyzer", filename],
        capture_output=True,
        text=True,
        timeout=60
    )
    return result.stdout

# BAD: Shell injection vulnerability
def run_analysis(filename: str) -> str:
    import os
    os.system(f"python -m analyzer {filename}")  # VULNERABLE
```

### Path Traversal

Prevent directory traversal attacks:

```python
import os
from pathlib import Path

# GOOD: Resolve and validate paths
def safe_read(filename: str, base_dir: str = "./data") -> str:
    base = Path(base_dir).resolve()
    target = (base / filename).resolve()

    # Ensure target is within base directory
    if not str(target).startswith(str(base)):
        raise ValueError("Path traversal detected")

    if not target.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    return target.read_text()

# BAD: Direct path concatenation
def unsafe_read(filename: str) -> str:
    with open(f"./data/{filename}") as f:  # "../../../etc/passwd" works!
        return f.read()
```

### JSON/YAML Deserialization

Avoid unsafe deserialization:

```python
import json
import yaml

# GOOD: Use safe loaders
def load_config(path: str) -> dict:
    with open(path) as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            return yaml.safe_load(f)  # Safe loader
        return json.load(f)  # JSON is inherently safe

# BAD: Unsafe YAML loading allows code execution
def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.load(f, Loader=yaml.Loader)  # VULNERABLE to !!python/object
```

---

## JavaScript Security

### Prototype Pollution Prevention

```javascript
// GOOD: Use Object.create(null) for dictionaries
function createSafeDict() {
  return Object.create(null);
}

// GOOD: Validate keys before assignment
function safeAssign(obj, key, value) {
  if (key === '__proto__' || key === 'constructor' || key === 'prototype') {
    throw new Error('Invalid key');
  }
  obj[key] = value;
}

// BAD: Direct assignment allows pollution
function unsafeAssign(obj, key, value) {
  obj[key] = value;  // obj['__proto__']['polluted'] = true
}
```

### require() Security

```javascript
// GOOD: Validate module paths
const path = require('path');
const ALLOWED_MODULES = new Set(['./constants', './s3_operator_symmetry']);

function safeRequire(modulePath) {
  const normalized = path.normalize(modulePath);
  if (!ALLOWED_MODULES.has(normalized)) {
    throw new Error(`Module not allowed: ${modulePath}`);
  }
  return require(normalized);
}

// BAD: Dynamic require with user input
function unsafeRequire(userInput) {
  return require(userInput);  // Can load arbitrary modules
}
```

---

## Operator Algebra Security

### Handler Registration

The `OperatorAlgebra` class accepts arbitrary functions. Validate handlers:

```python
# GOOD: Validate handler behavior
class SecureOperatorAlgebra(OperatorAlgebra):
    def register(self, symbol: str, handler: Callable) -> None:
        # Test handler with safe values
        try:
            result = handler(1.0)
            if not isinstance(result, (int, float)):
                raise TypeError("Handler must return numeric")
        except Exception as e:
            raise ValueError(f"Invalid handler: {e}")

        super().register(symbol, handler)

# BAD: Accept any function
algebra.register("^", lambda x: os.system(f"rm -rf {x}"))  # Malicious!
```

### Sequence Execution Limits

Prevent infinite loops or resource exhaustion:

```python
# GOOD: Limit sequence length and execution time
import signal

def apply_sequence_safe(
    self,
    operators: List[str],
    value: Any,
    max_ops: int = 1000,
    timeout_sec: int = 5
) -> Any:
    if len(operators) > max_ops:
        raise ValueError(f"Sequence too long: {len(operators)} > {max_ops}")

    def handler(signum, frame):
        raise TimeoutError("Operator sequence timeout")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_sec)

    try:
        result = value
        for op in operators:
            result = self.apply(op, result)
        return result
    finally:
        signal.alarm(0)

# BAD: Unbounded execution
def apply_sequence(self, operators, value):
    for op in operators:  # Could be infinite
        value = self.apply(op, value)
    return value
```

---

## Output Integrity

### Watermarking

All generated outputs should include provenance markers:

```python
def generate_report(data: dict) -> str:
    header = """
    ╔════════════════════════════════════════════════════════════════════╗
    ║  QUANTUM-APL SIMULATION OUTPUT                                     ║
    ║  This is SIMULATED data from a classical computation.              ║
    ║  NOT from quantum hardware. NOT a consciousness measurement.       ║
    ╚════════════════════════════════════════════════════════════════════╝
    """
    return header + json.dumps(data, indent=2)
```

### Reproducibility Hashes

Include verification hashes for audit trails:

```python
import hashlib
import json

def add_integrity_hash(data: dict) -> dict:
    """Add SHA-256 hash for data integrity verification."""
    content = json.dumps(data, sort_keys=True)
    hash_value = hashlib.sha256(content.encode()).hexdigest()
    return {
        **data,
        "_integrity": {
            "algorithm": "sha256",
            "hash": hash_value,
            "timestamp": datetime.utcnow().isoformat(),
        }
    }
```

---

## Prohibited Uses

### Do NOT use this system to:

1. **Generate pseudo-scientific claims**
   - Outputs are mathematical simulations, not physical measurements
   - Do not represent as "quantum consciousness" evidence

2. **Create manipulation frameworks**
   - The PARADOX/TRUE/UNTRUE terminology is mathematical, not psychological
   - Do not use for persuasion or influence operations

3. **Bypass security controls**
   - The operator algebra is for DSL design, not exploitation
   - Do not use composition/inversion for attack primitives

4. **Credential or authority fraud**
   - Do not cite outputs as peer-reviewed research
   - Do not represent simulations as hardware measurements

---

## Vulnerability Reporting

If you discover a security vulnerability:

1. **Do NOT** open a public issue
2. **Do NOT** exploit the vulnerability
3. **Contact** maintainers via security@[project-domain] or encrypted channels
4. **Provide** detailed reproduction steps
5. **Allow** reasonable time for patching before disclosure

---

## Dependencies

### Audit Requirements

Before adding dependencies:

1. Check for known CVEs via `npm audit` or `pip-audit`
2. Review dependency tree depth
3. Verify package provenance (npm/PyPI verified publishers)
4. Pin versions in lockfiles

```bash
# Node.js
npm audit
npm audit fix

# Python
pip-audit
pip install --require-hashes -r requirements.txt
```

### Minimal Dependencies

This project intentionally minimizes dependencies:

- **Python**: numpy (numerical), PyYAML (config) - both well-audited
- **JavaScript**: ajv (JSON Schema) - OWASP recommended

---

## CI/CD Security

### Secrets Management

```yaml
# GOOD: Use GitHub secrets, never inline
env:
  API_KEY: ${{ secrets.API_KEY }}

# BAD: Hardcoded secrets
env:
  API_KEY: "sk-1234567890abcdef"  # NEVER DO THIS
```

### Workflow Permissions

```yaml
# GOOD: Minimal permissions
permissions:
  contents: read
  actions: read

# BAD: Overly permissive
permissions: write-all
```

---

## Related Documentation

- `docs/PHYSICS_GROUNDING.md` - Scientific basis for constants (z_c, φ)
- `docs/CONSTANTS_ARCHITECTURE.md` - Constant inventory and validation
- `src/quantum_apl_python/z_axis_threshold_analysis.py` - Physics verification code

---

*Last updated: 2025*
