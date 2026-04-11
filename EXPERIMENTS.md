# Running Experiments on Colab

Use `run_experiments.py` to run multiple experiments sequentially and compare results.

## Quick Start

### 1. Setup (first time only)
```python
!python train_gpt_colab.py --mode setup
```

### 2. Run predefined experiments

**Quick tests** (3 experiments, 50 steps each, ~10 minutes total):
```python
!python run_experiments.py --experiments quick
```

**Hyperparameter sweep** (6 experiments, 500 steps each, ~3 hours total):
```python
!python run_experiments.py --experiments sweep
```

**Ablation study** (5 experiments, 1000 steps each, ~5 hours total):
```python
!python run_experiments.py --experiments ablation
```

### 3. View results

Results are automatically saved to `experiment_results_YYYYMMDD_HHMMSS.json` and printed at the end.

---

## Custom Experiments

### Method 1: Edit the script

Open `run_experiments.py` and add to `QUICK_EXPERIMENTS` (around line 25):

```python
{
    "name": "my_experiment",
    "description": "What I'm testing",
    "config": {
        "RUN_ID": "exp_mine",
        "ITERATIONS": "500",
        "TRAIN_BATCH_TOKENS": "131072",
        "NUM_LAYERS": "10",      # Your changes here
        "MLP_MULT": "3",
        "MATRIX_LR": "0.05",
        "SEED": "42",
    }
},
```

Then run:
```python
!python run_experiments.py --experiments quick
```

### Method 2: JSON config file

Create `my_experiments.json`:
```json
[
  {
    "name": "test_1",
    "description": "Test wider MLP",
    "config": {
      "RUN_ID": "test1",
      "ITERATIONS": "200",
      "TRAIN_BATCH_TOKENS": "131072",
      "MLP_MULT": "3",
      "SEED": "42"
    }
  },
  {
    "name": "test_2",
    "description": "Test more layers",
    "config": {
      "RUN_ID": "test2",
      "ITERATIONS": "200",
      "TRAIN_BATCH_TOKENS": "131072",
      "NUM_LAYERS": "11",
      "SEED": "42"
    }
  }
]
```

Then run:
```python
!python run_experiments.py --config my_experiments.json
```

---

## What Gets Tested

### Quick Experiments (--experiments quick)
1. **Baseline**: 9L × 512d, MLP 2x
2. **Wider MLP**: 9L × 512d, MLP 3x
3. **More layers**: 11L × 512d, MLP 2x

⏱️ 50 steps each, ~10 min total

### Sweep Experiments (--experiments sweep)
**Learning rate sweep**:
- MATRIX_LR: 0.02, 0.04, 0.06

**MLP width sweep**:
- MLP_MULT: 2, 3, 4

⏱️ 500 steps each, ~3 hours total

### Ablation Experiments (--experiments ablation)
Tests combinations of:
- Layers: 9, 10, 11
- MLP: 2x, 3x

⏱️ 1000 steps each, ~5 hours total

---

## Available Hyperparameters

You can override any of these in the `config` section:

### Model Architecture
```python
"VOCAB_SIZE": "1024",           # Tokenizer vocab size
"NUM_LAYERS": "9",              # Number of transformer blocks
"MODEL_DIM": "512",             # Embedding dimension
"NUM_HEADS": "8",               # Attention heads
"NUM_KV_HEADS": "4",            # KV heads (for GQA)
"MLP_MULT": "2",                # MLP hidden = dim × this
"TIE_EMBEDDINGS": "1",          # 1=tied, 0=untied
"ROPE_BASE": "10000.0",         # RoPE frequency base
"LOGIT_SOFTCAP": "30.0",        # Logit capping value
```

### Training
```python
"ITERATIONS": "1000",           # Training steps
"TRAIN_BATCH_TOKENS": "131072", # Tokens per step
"TRAIN_SEQ_LEN": "1024",        # Sequence length
"MAX_WALLCLOCK_SECONDS": "3600",# Time limit (seconds)
"SEED": "42",                   # Random seed
"WARMUP_STEPS": "20",           # LR warmup steps
"WARMDOWN_ITERS": "1200",       # LR decay steps
```

### Optimizer
```python
"MATRIX_LR": "0.04",            # Muon learning rate (matrices)
"SCALAR_LR": "0.04",            # Adam learning rate (scalars)
"TIED_EMBED_LR": "0.05",        # Embedding learning rate
"MUON_MOMENTUM": "0.95",        # Muon momentum
"MUON_BACKEND_STEPS": "5",      # Newton-Schulz iterations
"MUON_MOMENTUM_WARMUP_START": "0.85",
"MUON_MOMENTUM_WARMUP_STEPS": "500",
"BETA1": "0.9",                 # Adam beta1
"BETA2": "0.95",                # Adam beta2
"GRAD_CLIP_NORM": "0.0",        # Gradient clipping (0=off)
```

### Validation
```python
"VAL_LOSS_EVERY": "0",          # Validate every N steps (0=only at end)
"VAL_BATCH_SIZE": "131072",     # Validation batch size
"TRAIN_LOG_EVERY": "50",        # Print train loss every N steps
```

---

## Output Format

Results table:
```
Name                      Status     BPB        Loss       Time (s)
--------------------------------------------------------------------------------
baseline                  SUCCESS    1.8234     3.0821     145.3
wider_mlp                 SUCCESS    1.7892     3.0234     156.8
more_layers               SUCCESS    1.8104     3.0567     172.1
```

Best result:
```
BEST RESULT
================================================================================
Name: wider_mlp
Description: 9L × 512d, MLP 3x
BPB: 1.7892
Config: {
  "RUN_ID": "exp_wider_mlp",
  "NUM_LAYERS": "9",
  "MLP_MULT": "3",
  ...
}
```

Results JSON file:
```json
[
  {
    "name": "baseline",
    "description": "9L × 512d baseline",
    "config": { ... },
    "status": "SUCCESS",
    "elapsed_seconds": 145.3,
    "val_bpb": 1.8234,
    "val_loss": 3.0821
  },
  ...
]
```

---

## Tips

### 1. Start small
Run `quick` first (50 steps) to verify everything works before long sweeps.

### 2. Use multiple seeds
For reliable comparisons, run each config with 3 different seeds:
```python
{
  "name": "exp_seed1",
  "config": { ..., "SEED": "42" }
},
{
  "name": "exp_seed2",
  "config": { ..., "SEED": "1337" }
},
{
  "name": "exp_seed3",
  "config": { ..., "SEED": "2024" }
}
```

### 3. Batch size vs GPU memory
If you get OOM:
- T4 (16GB): `"TRAIN_BATCH_TOKENS": "65536"`
- V100 (32GB): `"TRAIN_BATCH_TOKENS": "131072"`
- A100 (40GB): `"TRAIN_BATCH_TOKENS": "262144"`

### 4. Save to Drive
Mount Google Drive and copy results:
```python
from google.colab import drive
drive.mount('/content/drive')
!cp experiment_results_*.json /content/drive/MyDrive/
```

### 5. Monitor progress
Open another notebook cell and run:
```python
!tail -f logs/*.txt
```
(Press stop button to exit)

---

## Example Workflows

### Workflow 1: Find best MLP width
```python
# Create config file
experiments = []
for mlp_mult in [2, 3, 4, 5]:
    experiments.append({
        "name": f"mlp_{mlp_mult}x",
        "description": f"MLP {mlp_mult}x expansion",
        "config": {
            "RUN_ID": f"mlp{mlp_mult}x",
            "ITERATIONS": "500",
            "TRAIN_BATCH_TOKENS": "131072",
            "MLP_MULT": str(mlp_mult),
            "SEED": "42"
        }
    })

import json
with open('mlp_sweep.json', 'w') as f:
    json.dump(experiments, f, indent=2)

# Run
!python run_experiments.py --config mlp_sweep.json
```

### Workflow 2: Find best layer count
```python
# Edit run_experiments.py, add to QUICK_EXPERIMENTS:
for num_layers in [9, 10, 11, 12]:
    QUICK_EXPERIMENTS.append({
        "name": f"layers_{num_layers}",
        "description": f"{num_layers} layers",
        "config": {
            "RUN_ID": f"L{num_layers}",
            "ITERATIONS": "500",
            "TRAIN_BATCH_TOKENS": "131072",
            "NUM_LAYERS": str(num_layers),
            "SEED": "42"
        }
    })

# Run
!python run_experiments.py --experiments quick
```

### Workflow 3: Grid search
```python
# 2D grid: layers × MLP width
experiments = []
for layers in [9, 10, 11]:
    for mlp in [2, 3, 4]:
        experiments.append({
            "name": f"L{layers}_MLP{mlp}x",
            "description": f"{layers} layers, MLP {mlp}x",
            "config": {
                "RUN_ID": f"L{layers}_MLP{mlp}x",
                "ITERATIONS": "300",
                "TRAIN_BATCH_TOKENS": "131072",
                "NUM_LAYERS": str(layers),
                "MLP_MULT": str(mlp),
                "SEED": "42"
            }
        })

with open('grid_search.json', 'w') as f:
    json.dump(experiments, f, indent=2)

!python run_experiments.py --config grid_search.json
```

---

## Understanding Results

**Good BPB** (on Colab, 1 GPU, limited training):
- 50 steps: 2.0-2.5 (basically untrained)
- 500 steps: 1.6-1.9
- 1000 steps: 1.5-1.7
- 2000 steps: 1.4-1.6

**What to look for**:
- Relative differences matter more than absolute values
- If A gets 1.75 and B gets 1.65 on Colab → B is likely better on 8xH100 too
- Consistent trends across seeds = reliable signal

**When to scale up**:
- Found a config that beats baseline by >0.1 BPB on Colab
- Tested with 3 seeds, improvement is consistent
- Ready to rent 8×H100 on RunPod for real run

---

## Troubleshooting

**"Dataset not found"**
```python
!python train_gpt_colab.py --mode setup
```

**OOM errors**
Reduce batch size in config:
```python
"TRAIN_BATCH_TOKENS": "65536"  # or even "32768"
```

**Experiment fails but no error shown**
Check the log:
```python
!cat logs/YOUR_RUN_ID.txt
```

**Want to stop early**
Press the stop button in Colab, then:
```python
# Results so far are in experiment_results_*.json
!cat experiment_results_*.json
```
