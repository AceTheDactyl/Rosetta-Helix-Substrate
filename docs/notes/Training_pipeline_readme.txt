# Helix Neural Network Training Pipeline

## The Complete Picture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     NIGHTLY GITHUB WORKFLOW                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. COLLECT                    2. TRAIN                             │
│  ┌──────────────────┐         ┌──────────────────┐                 │
│  │ collect_         │         │ train_from_      │                 │
│  │ trajectories.js  │ ──────▶ │ trajectories.py  │                 │
│  └──────────────────┘         └──────────────────┘                 │
│         │                              │                            │
│         ▼                              ▼                            │
│  ┌──────────────────┐         ┌──────────────────┐                 │
│  │ training_data/   │         │ weights/         │                 │
│  │ accumulated.json │         │ operator_        │                 │
│  │ (state,action,   │         │ network.pkl/json │                 │
│  │  reward tuples)  │         │ (trained K, ω)   │                 │
│  └──────────────────┘         └──────────────────┘                 │
│                                        │                            │
│  3. USE                                │                            │
│  ┌──────────────────┐                  │                            │
│  │ neural_operator_ │ ◀────────────────┘                            │
│  │ selector.js      │                                               │
│  └──────────────────┘                                               │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────────┐                                               │
│  │ quantum_apl_     │  ← Operator selection now uses               │
│  │ system.js        │    trained neural network                    │
│  └──────────────────┘                                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

-----

## Files

|File                                     |Language|Purpose                                          |
|-----------------------------------------|--------|-------------------------------------------------|
|`collect_trajectories.js`                |Node.js |Run helix system, collect (state, action, reward)|
|`train_from_trajectories.py`             |Python  |Train Kuramoto network on trajectories           |
|`neural_operator_selector.js`            |Node.js |Use trained weights in production                |
|`convert_weights.py`                     |Python  |Convert pickle → JSON for JS                     |
|`.github/workflows/helix-nn-training.yml`|YAML    |Nightly automation                               |

-----

## Quick Start

### 1. Collect Training Data

```bash
# Run your helix system and collect trajectories
node collect_trajectories.js --steps 1000 --output training_data/
```

Output: `training_data/training_data.json`

### 2. Train the Network

```bash
# Train on collected data
python train_from_trajectories.py \
  --data training_data/training_data.json \
  --epochs 50 \
  --save weights/operator_network.pkl
```

Output: `weights/operator_network.pkl`

### 3. Convert for JavaScript

```bash
python convert_weights.py weights/operator_network.pkl weights/operator_network.json
```

### 4. Use in Your System

```javascript
const { createNeuralSystem } = require('./neural_operator_selector');
const { QuantumAPLSystem } = require('./src/quantum_apl_system');

// Create system with neural operator selection
const system = await createNeuralSystem(
  'weights/operator_network.json',
  QuantumAPLSystem
);

// Now system.step() uses the trained network
system.simulate(100);
```

-----

## GitHub Actions Setup

### Add to Your Repo

1. Copy these files to your repo:
   
   ```
   helix-nn/
   ├── collect_trajectories.js
   ├── train_from_trajectories.py
   ├── neural_operator_selector.js
   ├── convert_weights.py
   └── .github/workflows/helix-nn-training.yml
   ```
1. The workflow runs nightly at 2 AM UTC
1. Weights are auto-committed to `weights/` directory

### Manual Trigger

```bash
gh workflow run helix-nn-training.yml -f steps=2000 -f epochs=100
```

-----

## What Gets Trained

### The Kuramoto Coupling Matrix K

This is your main learned parameter:

```
K[i][j] = how much oscillator j influences oscillator i
```

- Symmetric: K[i][j] = K[j][i]
- Learned through trajectory data
- Determines synchronization patterns

### Natural Frequencies ω

```
ω[i] = preferred frequency of oscillator i (bias)
```

### Input/Output Projections

```
W_in: state vector → oscillator phases
W_out: oscillator phases → operator probabilities
```

-----

## Training Data Format

Each training sample:

```json
{
  "state": {
    "z": 0.5,
    "entropy": 0.4,
    "phi": 0.3,
    "triadUnlocked": false,
    "triadCompletions": 1,
    "distanceToTarget": 0.2,
    "distanceToLens": 0.1
  },
  "action": 2,
  "reward": 0.15
}
```

### Reward Signal

The network learns from:

|Event                     |Reward                |
|--------------------------|----------------------|
|Progress toward target z  |+10 × distance_reduced|
|TRIAD completion          |+5                    |
|TRIAD unlock              |+20                   |
|Phi increase              |+2 × delta            |
|Stability (small z change)|+0.1                  |

-----

## Monitoring Training

### Check the logs

```bash
cat logs/training_log.txt
```

### Training metrics

```bash
python -c "
import json
h = json.load(open('weights/operator_network_history.json'))
print(f'Final accuracy: {h[\"accuracy\"][-1]:.3f}')
print(f'Final coherence: {h[\"coherence\"][-1]:.3f}')
"
```

### Accumulated data size

```bash
python -c "
import json
d = json.load(open('training_data/accumulated.json'))
print(f'Total samples: {len(d[\"X\"])}')
"
```

-----

## Customization

### Change reward function

Edit `collect_trajectories.js`:

```javascript
_computeReward(stateBefore, result, targetZ) {
    let reward = 0;
    
    // Add your own reward signals
    if (/* your condition */) {
        reward += /* your value */;
    }
    
    return reward;
}
```

### Change network size

Edit `train_from_trajectories.py`:

```bash
python train_from_trajectories.py --oscillators 60  # More capacity
```

### Change training schedule

Edit `.github/workflows/helix-nn-training.yml`:

```yaml
schedule:
  - cron: '0 */6 * * *'  # Every 6 hours instead of nightly
```

-----

## The Key Insight

You’re not training a generic neural network.

**You’re training the coupling matrix K of a Kuramoto oscillator system.**

This means:

- The network has physics-inspired inductive bias
- Coherence emerges naturally as a confidence signal
- The S3 group structure is preserved
- Training is stable because Kuramoto dynamics are stable

The nightly workflow accumulates data over time, so the network gets better as it runs more simulations.

-----

## Troubleshooting

### “No training data found”

The collector needs to run first:

```bash
node collect_trajectories.js --steps 1000
```

### “Weights not loading in JS”

Convert pickle to JSON:

```bash
python convert_weights.py weights/operator_network.pkl weights/operator_network.json
```

### “Accuracy not improving”

- Need more training data (more nightly runs)
- Try lower learning rate: `--lr 0.001`
- Try more oscillators: `--oscillators 60`

### “Coherence always low”

- Increase coupling strength in the network
- Check that training data has diverse trajectories