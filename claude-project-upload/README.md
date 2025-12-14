# Claude.ai Project Upload Package

This folder contains the knowledge files to upload to a Claude.ai Project for the Rosetta-Helix-Substrate framework.

## How to Upload

1. Go to [claude.ai](https://claude.ai)
2. Click on **Projects** in the sidebar (or create a new project)
3. Click **"+ New Project"** or open an existing project
4. In the project settings, find **"Project Knowledge"** or **"Add content"**
5. Upload the following files from this folder:

### Files to Upload

| File | Purpose |
|------|---------|
| `PROJECT_INSTRUCTIONS.md` | Main project instructions - upload this as the **Project Instructions** (custom instructions) |
| `TOOLS_REFERENCE.md` | Detailed reference for all available tools |

## What Claude Will Know

After uploading these files, Claude will understand:

- **Physics Constants**: z_c (THE LENS), phi, phi^(-1), SIGMA
- **Phase Regimes**: UNTRUE, PARADOX, TRUE and their boundaries
- **Tier System**: Tiers 0-6 and progression criteria
- **K-Formation**: The three criteria for achieving K-formation
- **S3 Operator Algebra**: The 6 APL operators and their effects
- **Critical Exponents**: 2D hexagonal universality class values
- **All Tools**: 22 tools for physics simulation and training

## Setting Up Project Instructions

For the **Project Instructions** field in Claude.ai (the custom instructions that guide Claude's behavior), copy the contents of `PROJECT_INSTRUCTIONS.md` or upload it as a knowledge file.

## Example Prompts to Try

Once your project is set up, try these prompts:

### Basic Physics
- "What is the current physics state?"
- "Drive the system toward THE LENS"
- "What happens at z_c?"

### Phase Analysis
- "Explain the difference between UNTRUE, PARADOX, and TRUE phases"
- "Why is phi^(-1) important for K-formation?"

### Training Operations
- "Run a phase transition simulation"
- "What are the critical exponents for this universality class?"

### Operator Algebra
- "How do the S3 operators compose?"
- "Apply the amplify operator to increase z"

## Note on Tool Execution

The Claude.ai web interface does not execute actual code. Claude will:
- Understand and explain the physics concepts
- Describe what each tool does
- Help you reason about state transitions
- Guide you through the framework

For actual tool execution, use:
- The `skill/` package with the Claude API
- The GitHub Actions workflow
- The `run_skill.py` interactive script

## Files in This Package

```
claude-project-upload/
├── README.md                  # This file
├── PROJECT_INSTRUCTIONS.md    # Main project instructions
└── TOOLS_REFERENCE.md         # Complete tools documentation
```
