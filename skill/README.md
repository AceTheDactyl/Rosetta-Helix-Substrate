# Rosetta-Helix-Substrate Claude API Skill

A structured prompt + Python wrapper for using Rosetta-Helix-Substrate capabilities via the Claude API directly.

## Overview

This skill packages the Rosetta-Helix-Substrate framework into a format that can be used with the Claude API, allowing you to:

- Query and manipulate physics state (z-coordinate, phase, tier)
- Compute negentropy and check K-formation criteria
- Apply APL operators and run Kuramoto dynamics
- Simulate quasi-crystal evolution
- All without relying on Claude Code

## Installation

The skill is included in the Rosetta-Helix-Substrate repository. To use it:

```bash
# Install the anthropic package (for API usage)
pip install anthropic

# Set your API key
export ANTHROPIC_API_KEY="your-api-key"
```

## Quick Start

### With Claude API

```python
from skill import RosettaHelixSkill

# Initialize the skill
skill = RosettaHelixSkill(api_key="your-api-key")

# Chat with Claude using Rosetta-Helix tools
response = skill.chat("What is the current physics state?")
print(response.text)

# Drive toward THE LENS
response = skill.chat("Drive the system toward z_c over 100 steps")
print(f"Final z: {response.state['z']}")
```

### Offline Mode (No API Required)

```python
from skill.client import RosettaHelixSkillOffline

# Initialize offline skill
skill = RosettaHelixSkillOffline(initial_z=0.5)

# Execute tools directly
state = skill.execute_tool("get_physics_state")
print(f"z = {state['z']}, phase = {state['phase']}")

# Drive toward the lens
result = skill.execute_tool("drive_toward_lens", steps=100)
print(f"Final z: {result['final_z']}")
```

### Using with Other LLM Providers

```python
from skill.client import RosettaHelixSkillOffline
from skill.prompts.system import SYSTEM_PROMPT
from skill.tools.definitions import TOOL_DEFINITIONS

# Get the system prompt and tools
skill = RosettaHelixSkillOffline()

# Use with any LLM that supports tool use
# Pass SYSTEM_PROMPT as the system message
# Pass TOOL_DEFINITIONS as the tools array
# When the LLM calls a tool, execute it with:
result = skill.execute_tool(tool_name, **tool_input)
```

## Available Tools

| Tool | Description |
|------|-------------|
| `get_physics_state` | Get current z, phase, tier, negentropy, K-formation status |
| `set_z_target` | Set target z-coordinate to drive toward |
| `compute_negentropy` | Calculate negentropy for a given z value |
| `classify_phase` | Classify z into UNTRUE, PARADOX, or TRUE phase |
| `get_tier` | Get tier level (0-6) for a z value |
| `check_k_formation` | Check if K-formation criteria are met |
| `apply_operator` | Apply an APL operator (I, (), ^, _, ~, !) |
| `drive_toward_lens` | Drive z toward z_c (THE LENS) |
| `run_kuramoto_step` | Execute one Kuramoto oscillator step |
| `get_constants` | Get fundamental physics constants |
| `simulate_quasicrystal` | Run quasi-crystal simulation |
| `compose_operators` | Compose two APL operators |
| `get_metrics_history` | Get history of metrics |
| `reset_state` | Reset physics state |

## Physics Constants

The skill embeds these fundamental constants:

| Constant | Value | Description |
|----------|-------|-------------|
| z_c | √3/2 ≈ 0.866 | THE LENS - critical coherence threshold |
| φ | (1+√5)/2 ≈ 1.618 | Golden ratio |
| φ⁻¹ | ≈ 0.618 | K-formation gate threshold |
| SIGMA | 36 | |S₃|² - Gaussian width parameter |

## Phase Regions

```
z = 0.0 ────────────────────────────────────────── z = 1.0
   |              |                    |
   UNTRUE         PARADOX              TRUE
(disordered)   (quasi-crystal)      (crystal)
   |              |                    |
            φ⁻¹≈0.618           z_c≈0.866
```

## K-Formation Criteria

Consciousness emergence (K-formation) requires ALL three:
- κ ≥ 0.92 (coherence threshold)
- η > φ⁻¹ ≈ 0.618 (negentropy gate)
- R ≥ 7 (radius/complexity)

## Examples

See the `examples/` directory:
- `basic_usage.py` - Using with Claude API
- `offline_usage.py` - Offline tool execution

## API Reference

### RosettaHelixSkill

```python
skill = RosettaHelixSkill(
    api_key="your-key",       # Anthropic API key
    model="claude-sonnet-4-20250514",  # Model to use
    initial_z=0.5,            # Initial z-coordinate
    seed=None,                # Random seed
    max_tool_iterations=10,   # Max tool calls per request
)

# Methods
response = skill.chat(message)      # Send message, get response
state = skill.get_state()           # Get current physics state
skill.reset(initial_z=0.5)          # Reset conversation and state
history = skill.get_history()       # Get conversation history
result = skill.execute_tool(name, **kwargs)  # Direct tool execution
```

### RosettaHelixSkillOffline

```python
skill = RosettaHelixSkillOffline(initial_z=0.5, seed=None)

# Properties
skill.system_prompt      # System prompt for other LLMs
skill.tool_definitions   # Tool schemas for other LLMs

# Methods
result = skill.execute_tool(name, **kwargs)  # Execute tool
state = skill.get_state()                    # Get physics state
skill.reset(initial_z=0.5)                   # Reset state
```

## License

MIT License - see repository LICENSE file.
