"""
Google Colab adapter for Parameter Golf

This script sets up and runs train_gpt_single_gpu.py on a single GPU in Google Colab.
It uses a properly adapted single-GPU version without distributed training overhead.

Usage in Colab:
    !python train_gpt_colab.py --mode setup    # First time: install deps + download data
    !python train_gpt_colab.py --mode train    # Run training
    !python train_gpt_colab.py --mode quick    # Quick smoke test (50 steps)
"""

import os
import sys
import subprocess
from pathlib import Path

def detect_colab():
    """Check if running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def setup_colab_environment():
    """Install dependencies and download data"""
    print("=" * 60)
    print("SETUP: Installing dependencies...")
    print("=" * 60)

    # Install PyTorch (Colab usually has it, but ensure latest)
    subprocess.run([
        "pip", "install", "-q",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ], check=True)

    # Install other dependencies
    subprocess.run([
        "pip", "install", "-q",
        "sentencepiece",
        "numpy",
        "huggingface-hub",
        "datasets",
        "tqdm"
    ], check=True)

    print("\n" + "=" * 60)
    print("SETUP: Downloading dataset (SP1024, 10 shards for quick testing)...")
    print("=" * 60)

    # Download small subset for testing
    subprocess.run([
        "python3", "data/cached_challenge_fineweb.py",
        "--variant", "sp1024",
        "--train-shards", "10"
    ], check=True)

    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. For quick test (50 steps): !python train_gpt_colab.py --mode quick")
    print("  2. For full training: !python train_gpt_colab.py --mode train")
    print()

def check_gpu():
    """Check GPU availability"""
    import torch
    if not torch.cuda.is_available():
        print("ERROR: No GPU detected!")
        print("In Colab: Runtime > Change runtime type > GPU")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    return gpu_name, gpu_memory

def get_colab_config(mode="train"):
    """Get hyperparameters suitable for single GPU in Colab"""

    # Check GPU specs
    import torch
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

    if mode == "quick":
        # Quick smoke test
        return {
            "RUN_ID": "colab_smoke",
            "ITERATIONS": "50",
            "TRAIN_BATCH_TOKENS": "65536",  # Small batch for any GPU
            "TRAIN_SEQ_LEN": "1024",
            "VAL_LOSS_EVERY": "0",  # Only at end
            "VAL_BATCH_SIZE": "65536",
            "TRAIN_LOG_EVERY": "10",
            "MAX_WALLCLOCK_SECONDS": "300",  # 5 min
            "SEED": "42",
        }

    elif mode == "train":
        # Adapted for single GPU
        # 8 GPUs × 524K tokens = 4.2M tokens/step
        # For 1 GPU, we use 524K tokens/step with grad_accum_steps=1
        # This matches the per-GPU load of the 8-GPU baseline

        batch_tokens = "524288" if gpu_memory > 20 else "262144"  # Reduce for small GPUs

        return {
            "RUN_ID": "colab_baseline",
            "ITERATIONS": "2000",  # Fewer steps (1 GPU is slower)
            "TRAIN_BATCH_TOKENS": batch_tokens,
            "TRAIN_SEQ_LEN": "1024",
            "VAL_LOSS_EVERY": "500",
            "VAL_BATCH_SIZE": batch_tokens,
            "TRAIN_LOG_EVERY": "50",
            "MAX_WALLCLOCK_SECONDS": "3600",  # 1 hour
            "SEED": "42",
            # Model config (same as baseline)
            "VOCAB_SIZE": "1024",
            "NUM_LAYERS": "9",
            "MODEL_DIM": "512",
            "NUM_HEADS": "8",
            "NUM_KV_HEADS": "4",
            "MLP_MULT": "2",
            "TIE_EMBEDDINGS": "1",
        }

    elif mode == "experiment":
        # For your own experiments
        return {
            "RUN_ID": "colab_experiment",
            "ITERATIONS": "1000",
            "TRAIN_BATCH_TOKENS": "262144",
            "TRAIN_SEQ_LEN": "1024",
            "VAL_LOSS_EVERY": "200",
            "VAL_BATCH_SIZE": "262144",
            "TRAIN_LOG_EVERY": "50",
            "MAX_WALLCLOCK_SECONDS": "1800",  # 30 min
            "SEED": "1337",
            # Modify these for your experiments:
            "NUM_LAYERS": "9",
            "MLP_MULT": "3",  # Try 3x MLP
        }

    else:
        raise ValueError(f"Unknown mode: {mode}")

def run_training(mode="train"):
    """Run train_gpt.py with single GPU config"""

    print("=" * 60)
    print(f"TRAINING: Mode = {mode}")
    print("=" * 60)

    # Check GPU
    gpu_name, gpu_memory = check_gpu()

    if gpu_memory < 15:
        print(f"\nWARNING: GPU has only {gpu_memory:.1f} GB memory.")
        print("This may cause OOM. Consider:")
        print("  - Using Colab Pro (A100/V100)")
        print("  - Reducing TRAIN_BATCH_TOKENS")
        print()

    # Get config
    config = get_colab_config(mode)

    # Set environment variables
    env = os.environ.copy()
    env.update(config)

    print("Configuration:")
    for key, val in config.items():
        print(f"  {key}: {val}")
    print()

    # Run training with single GPU version (no distributed setup needed)
    print("Starting training with single-GPU optimized script...")
    print("=" * 60)

    result = subprocess.run(
        ["python3", "train_gpt_single_gpu.py"],
        env=env,
        check=False  # Don't raise on non-zero exit (we'll check manually)
    )

    if result.returncode != 0:
        print("\n" + "=" * 60)
        print("TRAINING FAILED!")
        print("=" * 60)
        print("\nCommon issues:")
        print("  1. OOM (out of memory) → reduce TRAIN_BATCH_TOKENS")
        print("  2. Data not found → run: !python train_gpt_colab.py --mode setup")
        print("  3. CUDA error → restart Colab runtime")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

    # Show where logs are
    log_dir = Path("logs")
    if log_dir.exists():
        logs = list(log_dir.glob("*.txt"))
        if logs:
            latest_log = max(logs, key=lambda p: p.stat().st_mtime)
            print(f"\nLog file: {latest_log}")
            print("\nTo view last 20 lines:")
            print(f"  !tail -20 {latest_log}")

    # Show final model
    if Path("final_model.pt").exists():
        size_mb = Path("final_model.pt").stat().st_size / 1e6
        print(f"\nModel saved: final_model.pt ({size_mb:.1f} MB)")

    if Path("final_model.int8.ptz").exists():
        size_mb = Path("final_model.int8.ptz").stat().st_size / 1e6
        print(f"Compressed: final_model.int8.ptz ({size_mb:.1f} MB)")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run Parameter Golf on Google Colab")
    parser.add_argument(
        "--mode",
        choices=["setup", "train", "quick", "experiment"],
        default="train",
        help="Mode: setup (install+download), train (full), quick (smoke test), experiment (custom)"
    )

    args = parser.parse_args()

    # Check if in Colab
    in_colab = detect_colab()
    if not in_colab:
        print("NOTE: Not running in Google Colab.")
        print("This script is optimized for Colab, but will work on any single-GPU setup.")
        print()

    # Change to repo root
    repo_root = Path(__file__).parent
    os.chdir(repo_root)

    if args.mode == "setup":
        setup_colab_environment()
    else:
        # Check if data exists
        data_dir = Path("data/datasets/fineweb10B_sp1024")
        if not data_dir.exists() or not list(data_dir.glob("*.bin")):
            print("ERROR: Dataset not found!")
            print("\nRun setup first:")
            print("  !python train_gpt_colab.py --mode setup")
            sys.exit(1)

        run_training(mode=args.mode)

if __name__ == "__main__":
    main()
