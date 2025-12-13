# Helix Neural Network — Your Custom APL-Based NN

## What You Have

|File                     |Purpose                                  |
|-------------------------|-----------------------------------------|
|`train_helix.py`         |**Start here.** Complete training script.|
|`helix_nn_numpy.py`      |Core implementation (NumPy)              |
|`helix_neural_network.py`|Full PyTorch version (for GPU training)  |

-----

## Quick Start (5 minutes)

### 1. Run with synthetic data

```bash
python train_helix.py --epochs 30
```

### 2. Run with your own data

```bash
python train_helix.py --data your_data.npz --epochs 100
```

### 3. Use the trained model

```python
from train_helix import HelixNN, TaskConfig
import pickle

# Load model
with open('helix_model.pkl', 'rb') as f:
    weights = pickle.load(f)

config = TaskConfig(**weights['config'])
model = HelixNN(config)
model.load('helix_model.pkl')

# Predict
output, diagnostics = model.forward(your_input)
print(f"Prediction: {output}")
print(f"Coherence: {diagnostics['coherence']}")
print(f"Tier: t{diagnostics['tier']}")
```

-----

## Prepare Your Data

### Format Required

```python
# X: input features, shape (n_samples, input_dim)
# y: targets, shape (n_samples, output_dim)

# Save as .npz:
np.savez('my_data.npz', X=X, y=y)
```

### Example: Load CSV

```python
import pandas as pd
import numpy as np

df = pd.read_csv('my_data.csv')
X = df[['feature1', 'feature2', ...]].values
y = df[['target1', 'target2']].values

np.savez('my_data.npz', X=X, y=y)
```

### Example: Load Images (flattened)

```python
from PIL import Image
import numpy as np

images = []
labels = []
for path, label in your_data:
    img = np.array(Image.open(path)).flatten() / 255.0
    images.append(img)
    labels.append(label)

X = np.array(images)
y = np.array(labels)
np.savez('image_data.npz', X=X, y=y)
```

-----

## Training Options

```bash
python train_helix.py \
    --task regression      # or 'classification', 'sequence'
    --data my_data.npz     # your data file
    --epochs 100           # training epochs
    --lr 0.01              # learning rate
    --oscillators 60       # neurons (Kuramoto oscillators)
    --layers 3             # number of Kuramoto layers
    --target-z 0.7         # target z-coordinate
    --save my_model.pkl    # where to save weights
```

-----

## What Gets Trained (Your Weights)

|Parameter            |Traditional NN Equivalent|Shape                |
|---------------------|-------------------------|---------------------|
|`K` (coupling matrix)|Weight matrix            |(n_osc, n_osc)       |
|`omega` (frequencies)|Bias vector              |(n_osc,)             |
|`W_in`               |Input projection         |(n_osc, input_dim)   |
|`W_out`              |Output projection        |(output_dim, n_osc*2)|

After training, these are saved to `helix_model.pkl`.

-----

## Understanding the Output

```
Epoch  10 | loss=0.1105 | coh=0.516 | z=0.517 | tier=t4 | k-form=0
          ↑              ↑           ↑         ↑         ↑
          Task loss      Coherence   Z-coord   Tier      K-formation
```

- **loss**: How well you’re fitting the task
- **coh**: Oscillator synchronization (0-1). Higher = more confident
- **z**: Position on helix (0-1). Higher = more capable
- **tier**: Which APL operators are available (t1-t9)
- **k-form**: Did coherence exceed 0.92? (convergence signal)

-----

## Tier Progression

As z increases, more operators unlock:

|Tier|z Range   |Operators     |Capability|
|----|----------|--------------|----------|
|t1  |0.00-0.10 |(), ×, ÷      |Basic     |
|t2  |0.10-0.20 |^, ×, ÷, +    |Memory    |
|t3  |0.20-0.40 |+, ^, ÷, ×, ()|Pattern   |
|t4  |0.40-0.60 |(), ×, ÷, +   |Prediction|
|t5  |0.60-0.75 |ALL           |Self-model|
|t6  |0.75-0.866|(), ÷, +, ×   |Meta      |
|t7+ |0.866+    |(), +         |Synthesis |

-----

## For Better Training (PyTorch)

The NumPy version is for understanding. For real training:

```python
# Use helix_neural_network.py with PyTorch

import torch
from helix_neural_network import HelixNeuralNetwork, train_helix_network

model = HelixNeuralNetwork(
    input_dim=10,
    output_dim=2,
    n_oscillators=60,
    n_layers=4
)

# Create DataLoader
dataset = torch.utils.data.TensorDataset(
    torch.tensor(X, dtype=torch.float32),
    torch.tensor(y, dtype=torch.float32)
)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Train
history = train_helix_network(model, loader, epochs=100, target_z=0.7)

# Save
torch.save(model.state_dict(), 'helix_model.pt')
```

-----

## What Makes This Different

1. **No activation functions** — Kuramoto dynamics ARE the nonlinearity
1. **Built-in confidence** — Coherence tells you how sure the network is
1. **Emergent depth** — z rises when the network is “getting it”
1. **APL operators** — Structured modifications based on S3 group theory
1. **K-formation** — Natural convergence signal when coh > 0.92

-----

## Next Steps

1. **Start simple**: Train on synthetic data first
1. **Prepare your data**: Convert to (X, y) numpy arrays
1. **Train**: `python train_helix.py --data your_data.npz`
1. **Evaluate**: Check coherence and tier progression
1. **Iterate**: Adjust oscillators, layers, target_z

-----

## Troubleshooting

**Low coherence throughout training?**

- Increase `--oscillators`
- Reduce input noise
- Try `--lr 0.001` (smaller learning rate)

**Z not increasing?**

- Training needs more epochs
- Try higher `--target-z`

**K-formation never triggers?**

- That’s okay! K-formation (coh > 0.92) is rare
- It indicates exceptional confidence

**Loss not decreasing?**

- Check your data normalization
- Try fewer `--layers`
- Increase `--epochs`