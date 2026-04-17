"""
Run multiple experiments sequentially on Google Colab or any single GPU.

This script calls train_gpt.py multiple times with different configs.
train_gpt.py automatically runs in single-GPU mode when no distributed env vars are set.

Usage:
    !python run_experiments.py --experiments quick      # 3 quick tests (50 steps each)
    !python run_experiments.py --experiments sweep      # Hyperparameter sweep (500 steps each)
    !python run_experiments.py --experiments ablation   # Ablation study (1000 steps each)
    !python run_experiments.py --config my_config.json  # Custom config file
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime

# ============================================================================
# EXPERIMENT CONFIGURATIONS
# ============================================================================

QUICK_EXPERIMENTS = [
    {
        "name": "baseline",
        "description": "9L × 512d baseline",
        "config": {
            "DATA_PATH": "./data/datasets/fineweb10B_sp8192",
            "TOKENIZER_PATH": "./data/tokenizers/fineweb_8192_bpe.model",
            "RUN_ID": "exp_baseline",
            "ITERATIONS": "50",
            "TRAIN_BATCH_TOKENS": "65536",
            "NUM_LAYERS": "9",
            "MLP_MULT": "2",
            "SEED": "42",
        }
    },
    {
        "name": "wider_mlp",
        "description": "9L × 512d, MLP 3x",
        "config": {
            "DATA_PATH": "./data/datasets/fineweb10B_sp8192",
            "TOKENIZER_PATH": "./data/tokenizers/fineweb_8192_bpe.model",
            "RUN_ID": "exp_wider_mlp",
            "ITERATIONS": "50",
            "TRAIN_BATCH_TOKENS": "65536",
            "NUM_LAYERS": "9",
            "MLP_MULT": "3",
            "SEED": "42",
        }
    },
    {
        "name": "more_layers",
        "description": "11L × 512d, MLP 2x",
        "config": {
            "DATA_PATH": "./data/datasets/fineweb10B_sp8192",
            "TOKENIZER_PATH": "./data/tokenizers/fineweb_8192_bpe.model",
            "RUN_ID": "exp_more_layers",
            "ITERATIONS": "50",
            "TRAIN_BATCH_TOKENS": "65536",
            "NUM_LAYERS": "11",
            "MLP_MULT": "2",
            "SEED": "42",
        }
    },
]

SWEEP_EXPERIMENTS = [
    # Learning rate sweep
    {
        "name": "lr_low",
        "description": "Lower learning rate",
        "config": {
            "DATA_PATH": "./data/datasets/fineweb10B_sp8192",
            "TOKENIZER_PATH": "./data/tokenizers/fineweb_8192_bpe.model",
            "RUN_ID": "sweep_lr_low",
            "ITERATIONS": "500",
            "TRAIN_BATCH_TOKENS": "131072",
            "MATRIX_LR": "0.02",
            "SEED": "42",
        }
    },
    {
        "name": "lr_baseline",
        "description": "Baseline learning rate",
        "config": {
            "DATA_PATH": "./data/datasets/fineweb10B_sp8192",
            "TOKENIZER_PATH": "./data/tokenizers/fineweb_8192_bpe.model",
            "RUN_ID": "sweep_lr_baseline",
            "ITERATIONS": "500",
            "TRAIN_BATCH_TOKENS": "131072",
            "MATRIX_LR": "0.04",
            "SEED": "42",
        }
    },
    {
        "name": "lr_high",
        "description": "Higher learning rate",
        "config": {
            "DATA_PATH": "./data/datasets/fineweb10B_sp8192",
            "TOKENIZER_PATH": "./data/tokenizers/fineweb_8192_bpe.model",
            "RUN_ID": "sweep_lr_high",
            "ITERATIONS": "500",
            "TRAIN_BATCH_TOKENS": "131072",
            "MATRIX_LR": "0.06",
            "SEED": "42",
        }
    },
    # MLP width sweep
    {
        "name": "mlp_2x",
        "description": "MLP 2x (baseline)",
        "config": {
            "DATA_PATH": "./data/datasets/fineweb10B_sp8192",
            "TOKENIZER_PATH": "./data/tokenizers/fineweb_8192_bpe.model",
            "RUN_ID": "sweep_mlp_2x",
            "ITERATIONS": "500",
            "TRAIN_BATCH_TOKENS": "131072",
            "MLP_MULT": "2",
            "SEED": "42",
        }
    },
    {
        "name": "mlp_3x",
        "description": "MLP 3x",
        "config": {
            "DATA_PATH": "./data/datasets/fineweb10B_sp8192",
            "TOKENIZER_PATH": "./data/tokenizers/fineweb_8192_bpe.model",
            "RUN_ID": "sweep_mlp_3x",
            "ITERATIONS": "500",
            "TRAIN_BATCH_TOKENS": "131072",
            "MLP_MULT": "3",
            "SEED": "42",
        }
    },
    {
        "name": "mlp_4x",
        "description": "MLP 4x",
        "config": {
            "DATA_PATH": "./data/datasets/fineweb10B_sp8192",
            "TOKENIZER_PATH": "./data/tokenizers/fineweb_8192_bpe.model",
            "RUN_ID": "sweep_mlp_4x",
            "ITERATIONS": "500",
            "TRAIN_BATCH_TOKENS": "131072",
            "MLP_MULT": "4",
            "SEED": "42",
        }
    },
]

ABLATION_EXPERIMENTS = [
    # Test one technique at a time
    {
        "name": "baseline_9L",
        "description": "Baseline 9L",
        "config": {
            "DATA_PATH": "./data/datasets/fineweb10B_sp8192",
            "TOKENIZER_PATH": "./data/tokenizers/fineweb_8192_bpe.model",
            "RUN_ID": "ablation_baseline",
            "ITERATIONS": "1000",
            "TRAIN_BATCH_TOKENS": "131072",
            "NUM_LAYERS": "9",
            "MLP_MULT": "2",
            "SEED": "1337",
        }
    },
    {
        "name": "baseline_10L",
        "description": "+1 layer (10L)",
        "config": {
            "DATA_PATH": "./data/datasets/fineweb10B_sp8192",
            "TOKENIZER_PATH": "./data/tokenizers/fineweb_8192_bpe.model",
            "RUN_ID": "ablation_10L",
            "ITERATIONS": "1000",
            "TRAIN_BATCH_TOKENS": "131072",
            "NUM_LAYERS": "10",
            "MLP_MULT": "2",
            "SEED": "1337",
        }
    },
    {
        "name": "baseline_11L",
        "description": "+2 layers (11L)",
        "config": {
            "DATA_PATH": "./data/datasets/fineweb10B_sp8192",
            "TOKENIZER_PATH": "./data/tokenizers/fineweb_8192_bpe.model",
            "RUN_ID": "ablation_11L",
            "ITERATIONS": "1000",
            "TRAIN_BATCH_TOKENS": "131072",
            "NUM_LAYERS": "11",
            "MLP_MULT": "2",
            "SEED": "1337",
        }
    },
    {
        "name": "baseline_9L_mlp3x",
        "description": "9L + MLP 3x",
        "config": {
            "DATA_PATH": "./data/datasets/fineweb10B_sp8192",
            "TOKENIZER_PATH": "./data/tokenizers/fineweb_8192_bpe.model",
            "RUN_ID": "ablation_mlp3x",
            "ITERATIONS": "1000",
            "TRAIN_BATCH_TOKENS": "131072",
            "NUM_LAYERS": "9",
            "MLP_MULT": "3",
            "SEED": "1337",
        }
    },
    {
        "name": "baseline_10L_mlp3x",
        "description": "10L + MLP 3x",
        "config": {
            "DATA_PATH": "./data/datasets/fineweb10B_sp8192",
            "TOKENIZER_PATH": "./data/tokenizers/fineweb_8192_bpe.model",
            "RUN_ID": "ablation_10L_mlp3x",
            "ITERATIONS": "1000",
            "TRAIN_BATCH_TOKENS": "131072",
            "NUM_LAYERS": "10",
            "MLP_MULT": "3",
            "SEED": "1337",
        }
    },
]

# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def load_config_from_file(config_file):
    """Load experiment configs from JSON file"""
    with open(config_file, 'r') as f:
        return json.load(f)

def run_single_experiment(exp_config, experiment_number, total_experiments):
    """Run a single training experiment"""

    name = exp_config["name"]
    desc = exp_config.get("description", "")
    config = exp_config["config"]

    print("\n" + "=" * 80)
    print(f"EXPERIMENT {experiment_number}/{total_experiments}: {name}")
    print(f"Description: {desc}")
    print("=" * 80)

    # Print config
    print("\nConfiguration:")
    for key, val in config.items():
        print(f"  {key}: {val}")
    print()

    # Set default values
    # IMPORTANT: Do NOT set RANK/WORLD_SIZE/LOCAL_RANK here
    # train_gpt.py auto-detects single-GPU mode when these are absent
    defaults = {
        "DATA_PATH": "./data/datasets/fineweb10B_sp1024",
        "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
        "TRAIN_SEQ_LEN": "1024",
        "VAL_LOSS_EVERY": "0",  # Only at end for speed
        "VAL_BATCH_SIZE": config.get("TRAIN_BATCH_TOKENS", "131072"),
        "TRAIN_LOG_EVERY": "50",
        "MAX_WALLCLOCK_SECONDS": "3600",
        "VOCAB_SIZE": "1024",
        "MODEL_DIM": "512",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "4",
        "TIE_EMBEDDINGS": "1",
    }

    # Merge: defaults, then user config (user config overrides)
    # Start with clean environment (no inherited RANK/WORLD_SIZE from parent)
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

    # Record start time
    start_time = time.time()

    # Run training
    print(f"Starting training for: {name}")
    print("-" * 80)

    result = subprocess.run(
        ["python3", "train_gpt.py"],
        env=env,
        capture_output=False,  # Show output in real-time
    )

    elapsed = time.time() - start_time

    # Check result
    if result.returncode != 0:
        print("\n" + "!" * 80)
        print(f"EXPERIMENT FAILED: {name}")
        print("!" * 80)
        return {
            "name": name,
            "description": desc,
            "config": config,
            "status": "FAILED",
            "elapsed_seconds": elapsed,
            "val_bpb": None,
        }

    # Parse results from log
    run_id = config.get("RUN_ID", "unknown")
    log_file = Path(f"logs/{run_id}.txt")

    val_bpb = None
    val_loss = None

    if log_file.exists():
        with open(log_file, 'r') as f:
            log_content = f.read()

            # Find final BPB (look for "final_int8_zlib_roundtrip_exact")
            # NOTE: The log file contains the source code at the top, so we need to
            # search for the LAST occurrence (the actual output, not the code)
            lines = log_content.split('\n')
            for line in reversed(lines):
                if "final_int8_zlib_roundtrip_exact" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.startswith("val_bpb:"):
                            try:
                                val_bpb = float(part.split(':')[1])
                            except (ValueError, IndexError):
                                continue
                        if part.startswith("val_loss:"):
                            try:
                                val_loss = float(part.split(':')[1])
                            except (ValueError, IndexError):
                                continue
                    # Found the line, stop searching
                    if val_bpb is not None and val_loss is not None:
                        break

    print("\n" + "=" * 80)
    print(f"EXPERIMENT COMPLETE: {name}")
    if val_bpb:
        print(f"Final BPB: {val_bpb:.4f}")
    if val_loss:
        print(f"Final Loss: {val_loss:.4f}")
    print(f"Time: {elapsed:.1f}s")
    print("=" * 80)

    return {
        "name": name,
        "description": desc,
        "config": config,
        "status": "SUCCESS",
        "elapsed_seconds": elapsed,
        "val_bpb": val_bpb,
        "val_loss": val_loss,
    }

def run_experiments(experiments, save_results=True):
    """Run a list of experiments sequentially"""

    total = len(experiments)
    results = []

    print("\n" + "=" * 80)
    print(f"RUNNING {total} EXPERIMENTS")
    print("=" * 80)

    for i, exp in enumerate(experiments, 1):
        result = run_single_experiment(exp, i, total)
        results.append(result)

        # Small delay between experiments
        if i < total:
            print("\nWaiting 5 seconds before next experiment...")
            time.sleep(5)

    # Print summary
    print("\n\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print()

    # Table header
    print(f"{'Name':<25} {'Status':<10} {'BPB':<10} {'Loss':<10} {'Time (s)':<10}")
    print("-" * 80)

    for r in results:
        name = r["name"][:24]
        status = r["status"]
        bpb = f"{r['val_bpb']:.4f}" if r['val_bpb'] else "N/A"
        loss = f"{r['val_loss']:.4f}" if r['val_loss'] else "N/A"
        elapsed = f"{r['elapsed_seconds']:.1f}"

        print(f"{name:<25} {status:<10} {bpb:<10} {loss:<10} {elapsed:<10}")

    # Find best
    successful = [r for r in results if r['val_bpb'] is not None]
    if successful:
        best = min(successful, key=lambda r: r['val_bpb'])
        print("\n" + "=" * 80)
        print("BEST RESULT")
        print("=" * 80)
        print(f"Name: {best['name']}")
        print(f"Description: {best['description']}")
        print(f"BPB: {best['val_bpb']:.4f}")
        print(f"Config: {json.dumps(best['config'], indent=2)}")

    # Save results to JSON
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"experiment_results_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {results_file}")

    return results

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run multiple experiments sequentially")
    parser.add_argument(
        "--experiments",
        choices=["quick", "sweep", "ablation"],
        help="Predefined experiment suite"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom experiment config JSON file"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to JSON"
    )

    args = parser.parse_args()

    # Check if data exists
    data_dir = Path("data/datasets/fineweb10B_sp1024")
    if not data_dir.exists() or not list(data_dir.glob("*.bin")):
        print("ERROR: Dataset not found!")
        print("\nRun setup first:")
        print("  !python train_gpt_colab.py --mode setup")
        sys.exit(1)

    # Load experiments
    if args.config:
        print(f"Loading experiments from: {args.config}")
        experiments = load_config_from_file(args.config)
    elif args.experiments == "quick":
        print("Running QUICK experiments (50 steps each)")
        experiments = QUICK_EXPERIMENTS
    elif args.experiments == "sweep":
        print("Running SWEEP experiments (500 steps each)")
        experiments = SWEEP_EXPERIMENTS
    elif args.experiments == "ablation":
        print("Running ABLATION experiments (1000 steps each)")
        experiments = ABLATION_EXPERIMENTS
    else:
        parser.print_help()
        sys.exit(1)

    # Run experiments
    results = run_experiments(experiments, save_results=not args.no_save)

if __name__ == "__main__":
    main()
