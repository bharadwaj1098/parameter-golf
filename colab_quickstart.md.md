# Parameter Golf on Google Colab - Quick Start

This guide shows how to run Parameter Golf training on Google Colab.

## Step 1: Open Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. **Runtime → Change runtime type → GPU** (T4/V100/A100)
3. Verify GPU: Run `!nvidia-smi` in a cell

## Step 2: Clone the Repo

```python
!git clone https://github.com/openai/parameter-golf.git
%cd parameter-golf
```

## Step 3: Setup (First Time Only)

This installs dependencies and downloads a small dataset (~10 train shards):

```python
!python train_gpt_colab.py --mode setup
```

⏱️ Takes ~5-10 minutes. You only need to do this once per session.

## Step 4: Quick Smoke Test

Run a 50-step training test to verify everything works:

```python
!python train_gpt_colab.py --mode quick
```

⏱️ Takes ~2-5 minutes depending on GPU.

You should see:
- Training loss decreasing
- Final BPB printed (will be terrible, only 50 steps)
- Model saved to `final_model.pt`

## Step 5: Full Training (Optional)

For a real experiment (2000 steps, ~1 hour on T4):

```python
!python train_gpt_colab.py --mode train
```

**Note**: This won't match the baseline (1.224 BPB) because:
- 1 GPU vs 8 GPUs → fewer total tokens seen
- Shorter training (2000 steps vs 13,780)
- But the code path is the same, good for learning/experiments

## Step 6: View Results

```python
# View last 20 lines of log
!tail -20 logs/*.txt

# Check model size
!ls -lh final_model*.pt*
```

## GPU Memory Issues?

If you get OOM (Out of Memory):

```python
# Edit train_gpt_colab.py, line ~95, reduce batch size:
# "TRAIN_BATCH_TOKENS": "131072",  # Half the default
```

Or upgrade to Colab Pro for better GPUs.

---

## For Experiments

Modify hyperparameters in `train_gpt_colab.py` under the `"experiment"` mode (lines 126-142):

```python
elif mode == "experiment":
    return {
        "RUN_ID": "my_experiment",
        "ITERATIONS": "1000",
        "NUM_LAYERS": "11",      # ← Try 11 layers
        "MLP_MULT": "3",         # ← Try 3x MLP
        "MATRIX_LR": "0.03",     # ← Try different LR
    }
```

Then run:

```python
!python train_gpt_colab.py --mode experiment
```

---

## Tips

1. **Mount Google Drive** to save checkpoints:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !cp final_model.pt /content/drive/MyDrive/
   ```

2. **Watch training live**:
   ```python
   !tail -f logs/*.txt
   ```
   (Press Ctrl+C to stop)

3. **Multiple runs**:
   Change `SEED` to run different random initializations

4. **Download for submission**:
   ```python
   from google.colab import files
   files.download('final_model.int8.ptz')
   ```

---

## What's Different from Real Training?

| Aspect | Colab (this script) | Real Submission |
|--------|---------------------|-----------------|
| GPUs | 1 (T4/V100/A100) | 8 × H100 |
| Steps | ~2000 | ~13,780 |
| Time | ~1 hour | 10 minutes |
| Total tokens | ~1B | ~7.2B |
| Final BPB | ~1.5-2.0 | ~1.08-1.22 |

Colab is for **learning and debugging**, not competitive submissions.

For real leaderboard runs, rent 8×H100 on RunPod or use OpenAI's compute grant.
