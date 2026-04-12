# Colab Configuration Verification

This document verifies that `train_gpt_colab.py` and `run_experiments.py` properly configure `train_gpt.py` for single-GPU execution.

## train_gpt.py Expected Environment Variables

From `train_gpt.py` Hyperparameters class (lines 39-87):

### Data & Paths
- `DATA_PATH` (default: `"./data/datasets/fineweb10B_sp1024"`)
- `TOKENIZER_PATH` (default: `"./data/tokenizers/fineweb_1024_bpe.model"`)
- `RUN_ID` (default: random uuid)

### Training Configuration
- `SEED` (default: `1337`)
- `ITERATIONS` (default: `20000`)
- `TRAIN_BATCH_TOKENS` (default: `524288`)
- `TRAIN_SEQ_LEN` (default: `1024`)
- `MAX_WALLCLOCK_SECONDS` (default: `600.0`)
- `WARMDOWN_ITERS` (default: `1200`)
- `WARMUP_STEPS` (default: `20`)

### Validation
- `VAL_BATCH_SIZE` (default: `524288`)
- `VAL_LOSS_EVERY` (default: `1000`)
- `TRAIN_LOG_EVERY` (default: `200`)

### Model Architecture
- `VOCAB_SIZE` (default: `1024`)
- `NUM_LAYERS` (default: `9`)
- `MODEL_DIM` (default: `512`)
- `NUM_HEADS` (default: `8`)
- `NUM_KV_HEADS` (default: `4`)
- `MLP_MULT` (default: `2`)
- `TIE_EMBEDDINGS` (default: `"1"`)
- `ROPE_BASE` (default: `10000.0`)
- `LOGIT_SOFTCAP` (default: `30.0`)
- `QK_GAIN_INIT` (default: `1.5`)

### Optimizer Hyperparameters
- `EMBED_LR` (default: `0.6`)
- `HEAD_LR` (default: `0.008`)
- `TIED_EMBED_LR` (default: `0.05`)
- `TIED_EMBED_INIT_STD` (default: `0.005`)
- `MATRIX_LR` (default: `0.04`)
- `SCALAR_LR` (default: `0.04`)
- `MUON_MOMENTUM` (default: `0.95`)
- `MUON_BACKEND_STEPS` (default: `5`)
- `MUON_MOMENTUM_WARMUP_START` (default: `0.85`)
- `MUON_MOMENTUM_WARMUP_STEPS` (default: `500`)
- `BETA1` (default: `0.9`)
- `BETA2` (default: `0.95`)
- `ADAM_EPS` (default: `1e-8`)
- `GRAD_CLIP_NORM` (default: `0.0`)

### Distributed Training (MUST BE ABSENT FOR SINGLE GPU)
- `RANK` - **DO NOT SET** (auto-detects single GPU if absent)
- `WORLD_SIZE` - **DO NOT SET** (auto-detects single GPU if absent)
- `LOCAL_RANK` - **DO NOT SET** (auto-detects single GPU if absent)
- `MASTER_ADDR` - **DO NOT SET**
- `MASTER_PORT` - **DO NOT SET**

## Single-GPU Detection Logic

From `train_gpt.py` line 742:
```python
distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
```

**Critical**: If `RANK` and `WORLD_SIZE` are NOT in the environment:
- `distributed = False`
- `rank = 0` (default)
- `world_size = 1` (default)
- `local_rank = 0` (default)
- `grad_accum_steps = 8 // 1 = 8`
- **NO** `dist.init_process_group()` call
- **NO** DDP wrapper (uses `compiled_model` directly, line 844)

## train_gpt_colab.py Configuration

### Quick Mode
```python
{
    "DATA_PATH": "./data/datasets/fineweb10B_sp1024",        # ✓ Set
    "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",  # ✓ Set
    "RUN_ID": "colab_smoke",                                 # ✓ Set
    "ITERATIONS": "50",                                      # ✓ Set
    "TRAIN_BATCH_TOKENS": "65536",                           # ✓ Set
    "TRAIN_SEQ_LEN": "1024",                                 # ✓ Set
    "VAL_LOSS_EVERY": "0",                                   # ✓ Set
    "VAL_BATCH_SIZE": "65536",                               # ✓ Set
    "TRAIN_LOG_EVERY": "10",                                 # ✓ Set
    "MAX_WALLCLOCK_SECONDS": "300",                          # ✓ Set
    "SEED": "42",                                            # ✓ Set
}
```
**Omitted** (uses defaults): All model architecture params, all optimizer params

### Train Mode
```python
{
    "DATA_PATH": "./data/datasets/fineweb10B_sp1024",        # ✓ Set
    "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",  # ✓ Set
    "RUN_ID": "colab_baseline",                              # ✓ Set
    "ITERATIONS": "2000",                                    # ✓ Set
    "TRAIN_BATCH_TOKENS": batch_tokens,                      # ✓ Set (524288 or 262144)
    "TRAIN_SEQ_LEN": "1024",                                 # ✓ Set
    "VAL_LOSS_EVERY": "500",                                 # ✓ Set
    "VAL_BATCH_SIZE": batch_tokens,                          # ✓ Set
    "TRAIN_LOG_EVERY": "50",                                 # ✓ Set
    "MAX_WALLCLOCK_SECONDS": "3600",                         # ✓ Set
    "SEED": "42",                                            # ✓ Set
    "VOCAB_SIZE": "1024",                                    # ✓ Set (explicit)
    "NUM_LAYERS": "9",                                       # ✓ Set (explicit)
    "MODEL_DIM": "512",                                      # ✓ Set (explicit)
    "NUM_HEADS": "8",                                        # ✓ Set (explicit)
    "NUM_KV_HEADS": "4",                                     # ✓ Set (explicit)
    "MLP_MULT": "2",                                         # ✓ Set (explicit)
    "TIE_EMBEDDINGS": "1",                                   # ✓ Set (explicit)
}
```
**Omitted** (uses defaults): All optimizer params (fine for baseline)

### Experiment Mode
```python
{
    "DATA_PATH": "./data/datasets/fineweb10B_sp1024",        # ✓ Set
    "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",  # ✓ Set
    "RUN_ID": "colab_experiment",                            # ✓ Set
    "ITERATIONS": "1000",                                    # ✓ Set
    "TRAIN_BATCH_TOKENS": "262144",                          # ✓ Set
    "TRAIN_SEQ_LEN": "1024",                                 # ✓ Set
    "VAL_LOSS_EVERY": "200",                                 # ✓ Set
    "VAL_BATCH_SIZE": "262144",                              # ✓ Set
    "TRAIN_LOG_EVERY": "50",                                 # ✓ Set
    "MAX_WALLCLOCK_SECONDS": "1800",                         # ✓ Set
    "SEED": "1337",                                          # ✓ Set
    "NUM_LAYERS": "9",                                       # ✓ Set
    "MLP_MULT": "3",                                         # ✓ Set (modified)
}
```

### Critical Safety Check (lines 174-176)
```python
# Remove any distributed training env vars to ensure single-GPU mode
for key in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]:
    env.pop(key, None)
```
✓ **CORRECT** - Ensures no distributed vars leak into subprocess

## run_experiments.py Configuration

### Defaults Applied to All Experiments
```python
{
    "DATA_PATH": "./data/datasets/fineweb10B_sp1024",        # ✓ Set
    "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",  # ✓ Set
    "TRAIN_SEQ_LEN": "1024",                                 # ✓ Set
    "VAL_LOSS_EVERY": "0",                                   # ✓ Set (only at end)
    "VAL_BATCH_SIZE": config.get("TRAIN_BATCH_TOKENS", "131072"),  # ✓ Set (dynamic)
    "TRAIN_LOG_EVERY": "50",                                 # ✓ Set
    "MAX_WALLCLOCK_SECONDS": "3600",                         # ✓ Set
    "VOCAB_SIZE": "1024",                                    # ✓ Set
    "MODEL_DIM": "512",                                      # ✓ Set
    "NUM_HEADS": "8",                                        # ✓ Set
    "NUM_KV_HEADS": "4",                                     # ✓ Set
    "TIE_EMBEDDINGS": "1",                                   # ✓ Set
}
```

### Environment Construction (lines 243-254)
```python
# Start with clean environment (no inherited RANK/WORLD_SIZE)
env = {}
essential_keys = [
    "PATH", "HOME", "USER", "SHELL",
    "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "LD_LIBRARY_PATH",
    "PYTHONPATH", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV"
]
for key in essential_keys:
    if key in os.environ:
        env[key] = os.environ[key]
env.update(defaults)
env.update(config)
```
✓ **CORRECT** - Creates clean environment without distributed training vars

### Example: QUICK_EXPERIMENTS[0]
```python
{
    "RUN_ID": "exp_baseline",        # Overrides default
    "ITERATIONS": "50",              # Overrides default
    "TRAIN_BATCH_TOKENS": "65536",   # Overrides default
    "NUM_LAYERS": "9",               # Overrides default (redundant but explicit)
    "MLP_MULT": "2",                 # Overrides default (redundant but explicit)
    "SEED": "42",                    # Overrides default
}
```
Merged result: Defaults + this config = complete valid config for train_gpt.py

## Verification Summary

### ✓ CORRECT Configurations

1. **Data Paths**: Both scripts now set `DATA_PATH` and `TOKENIZER_PATH` ✓
2. **Single-GPU Mode**: Both scripts explicitly avoid setting `RANK`/`WORLD_SIZE`/`LOCAL_RANK` ✓
3. **Environment Cleanup**: Both scripts remove/avoid distributed vars ✓
4. **Required Params**: All critical params (RUN_ID, ITERATIONS, batch sizes) are set ✓
5. **Defaults**: Unspecified params use train_gpt.py defaults (which is correct) ✓

### Expected Behavior

**When running either script:**
1. `train_gpt.py` line 742 sees no `RANK` or `WORLD_SIZE` in environment
2. Sets `distributed = False`
3. Uses defaults: `rank=0`, `world_size=1`, `local_rank=0`
4. Calculates `grad_accum_steps = 8 // 1 = 8`
5. Skips `dist.init_process_group()` call
6. Uses `compiled_model` directly (no DDP wrapper)
7. Runs on single GPU with gradient accumulation of 8 microbatches

### Testing Commands

**Quick test via wrapper:**
```bash
python3 train_gpt_colab.py --mode quick
```

**Quick test via experiments:**
```bash
python3 run_experiments.py --experiments quick
```

**Direct test (no wrapper):**
```bash
RUN_ID=test ITERATIONS=50 TRAIN_BATCH_TOKENS=65536 python3 train_gpt.py
```

All three methods will run train_gpt.py in single-GPU mode correctly.

## Notes

- Optimizer hyperparameters (MATRIX_LR, etc.) use train_gpt.py defaults unless overridden
- This is intentional and correct - the defaults are the baseline configuration
- To experiment with optimizer params, add them to the config dicts
- Model architecture params also use defaults unless explicitly set
- For competitive submissions, you'll want to tune these, but for learning/testing, defaults are fine
