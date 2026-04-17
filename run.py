"""
Unified script for Parameter Golf: setup, training, and experiments.

Usage:
    python run.py setup                                  # Install deps + download data
    python run.py train                                  # Full training (2000 steps)
    python run.py quick                                  # Smoke test (50 steps)
    python run.py experiment                             # Custom experiment
    python run.py sweep                                  # LR + MLP width sweep (500 steps each)
    python run.py ablation                               # Layer/MLP ablation (1000 steps each)
    python run.py --config my_experiments.json            # Custom config file
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime

# ============================================================================
# DEFAULTS
# ============================================================================

DATA_PATH = "./data/datasets/fineweb10B_sp8192"
TOKENIZER_PATH = "./data/tokenizers/fineweb_8192_bpe.model"

DEFAULTS = {
    "DATA_PATH": DATA_PATH,
    "TOKENIZER_PATH": TOKENIZER_PATH,
    "VOCAB_SIZE": "8192",
    "TRAIN_SEQ_LEN": "2048",
    "MODEL_DIM": "512",
    "NUM_HEADS": "8",
    "NUM_KV_HEADS": "4",
    "NUM_LAYERS": "9",
    "MLP_MULT": "2",
    "TIE_EMBEDDINGS": "1",
    "TRAIN_LOG_EVERY": "50",
    "MAX_WALLCLOCK_SECONDS": "3600",
    "SEED": "42",
}

# ============================================================================
# PRESET CONFIGS
# ============================================================================

PRESETS = {
    "quick_smoke": {
        "RUN_ID": "quick_smoke",
        "ITERATIONS": "50",
        "TRAIN_BATCH_TOKENS": "65536",
        "VAL_BATCH_SIZE": "65536",
        "VAL_LOSS_EVERY": "0",
        "TRAIN_LOG_EVERY": "10",
        "MAX_WALLCLOCK_SECONDS": "300",
    },
    "train": {
        "RUN_ID": "train_baseline",
        "ITERATIONS": "2000",
        "TRAIN_BATCH_TOKENS": "524288",
        "VAL_BATCH_SIZE": "524288",
        "VAL_LOSS_EVERY": "500",
    },
    "experiment": {
        "RUN_ID": "experiment",
        "ITERATIONS": "1000",
        "TRAIN_BATCH_TOKENS": "262144",
        "VAL_BATCH_SIZE": "262144",
        "VAL_LOSS_EVERY": "200",
        "MAX_WALLCLOCK_SECONDS": "1800",
        "SEED": "1337",
        "MLP_MULT": "3",
    },
}

# ============================================================================
# EXPERIMENT SUITES
# ============================================================================

QUICK_EXPERIMENTS = [
    {"name": "baseline",   "description": "9L x 512d baseline",  "config": {"RUN_ID": "exp_baseline",   "ITERATIONS": "50", "TRAIN_BATCH_TOKENS": "65536"}},
    {"name": "wider_mlp",  "description": "9L x 512d, MLP 3x",  "config": {"RUN_ID": "exp_wider_mlp",  "ITERATIONS": "50", "TRAIN_BATCH_TOKENS": "65536", "MLP_MULT": "3"}},
    {"name": "more_layers","description": "11L x 512d, MLP 2x",  "config": {"RUN_ID": "exp_more_layers","ITERATIONS": "50", "TRAIN_BATCH_TOKENS": "65536", "NUM_LAYERS": "11"}},
]

SWEEP_EXPERIMENTS = [
    {"name": "lr_low",      "description": "MATRIX_LR=0.02", "config": {"RUN_ID": "sweep_lr_low",      "ITERATIONS": "500", "TRAIN_BATCH_TOKENS": "131072", "MATRIX_LR": "0.02"}},
    {"name": "lr_baseline", "description": "MATRIX_LR=0.04", "config": {"RUN_ID": "sweep_lr_baseline", "ITERATIONS": "500", "TRAIN_BATCH_TOKENS": "131072", "MATRIX_LR": "0.04"}},
    {"name": "lr_high",     "description": "MATRIX_LR=0.06", "config": {"RUN_ID": "sweep_lr_high",     "ITERATIONS": "500", "TRAIN_BATCH_TOKENS": "131072", "MATRIX_LR": "0.06"}},
    {"name": "mlp_2x",      "description": "MLP 2x",         "config": {"RUN_ID": "sweep_mlp_2x",      "ITERATIONS": "500", "TRAIN_BATCH_TOKENS": "131072", "MLP_MULT": "2"}},
    {"name": "mlp_3x",      "description": "MLP 3x",         "config": {"RUN_ID": "sweep_mlp_3x",      "ITERATIONS": "500", "TRAIN_BATCH_TOKENS": "131072", "MLP_MULT": "3"}},
    {"name": "mlp_4x",      "description": "MLP 4x",         "config": {"RUN_ID": "sweep_mlp_4x",      "ITERATIONS": "500", "TRAIN_BATCH_TOKENS": "131072", "MLP_MULT": "4"}},
]

ABLATION_EXPERIMENTS = [
    {"name": "baseline_9L",      "description": "Baseline 9L",   "config": {"RUN_ID": "ablation_baseline",    "ITERATIONS": "1000", "TRAIN_BATCH_TOKENS": "131072", "SEED": "1337"}},
    {"name": "baseline_10L",     "description": "+1 layer (10L)","config": {"RUN_ID": "ablation_10L",         "ITERATIONS": "1000", "TRAIN_BATCH_TOKENS": "131072", "SEED": "1337", "NUM_LAYERS": "10"}},
    {"name": "baseline_11L",     "description": "+2 layers (11L)","config": {"RUN_ID": "ablation_11L",        "ITERATIONS": "1000", "TRAIN_BATCH_TOKENS": "131072", "SEED": "1337", "NUM_LAYERS": "11"}},
    {"name": "baseline_9L_mlp3x","description": "9L + MLP 3x",  "config": {"RUN_ID": "ablation_mlp3x",      "ITERATIONS": "1000", "TRAIN_BATCH_TOKENS": "131072", "SEED": "1337", "MLP_MULT": "3"}},
    {"name": "baseline_10L_mlp3x","description": "10L + MLP 3x","config": {"RUN_ID": "ablation_10L_mlp3x",  "ITERATIONS": "1000", "TRAIN_BATCH_TOKENS": "131072", "SEED": "1337", "NUM_LAYERS": "10", "MLP_MULT": "3"}},
]

SUITES = {
    "quick": QUICK_EXPERIMENTS,
    "sweep": SWEEP_EXPERIMENTS,
    "ablation": ABLATION_EXPERIMENTS,
}

# ============================================================================
# SETUP
# ============================================================================

def setup():
    print("=" * 60)
    print("SETUP: Installing dependencies...")
    print("=" * 60)

    subprocess.run(["pip", "install", "-q", "torch", "torchvision", "torchaudio",
                     "--index-url", "https://download.pytorch.org/whl/cu118"], check=True)
    subprocess.run(["pip", "install", "-q", "sentencepiece", "numpy",
                     "huggingface-hub", "datasets", "tqdm"], check=True)

    print("\n" + "=" * 60)
    print("SETUP: Downloading SP8192 dataset (80 shards)...")
    print("=" * 60)

    download_env = os.environ.copy()
    download_env["MATCHED_FINEWEB_REPO_ID"] = "kevclark/parameter-golf"
    subprocess.run(["python3", "data/cached_challenge_fineweb.py",
                     "--variant", "sp8192", "--train-shards", "80"],
                    env=download_env, check=True)

    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print("\nNext: python run.py quick")

# ============================================================================
# TRAINING RUNNER
# ============================================================================

def detect_gpu_count():
    import torch
    return torch.cuda.device_count() if torch.cuda.is_available() else 0

def check_data():
    data_dir = Path(DATA_PATH)
    if not data_dir.exists() or not list(data_dir.glob("*.bin")):
        print("ERROR: Dataset not found!")
        print("\nRun setup first: python run.py setup")
        sys.exit(1)

def build_env(config):
    env = os.environ.copy()
    env.update(DEFAULTS)
    env.update(config)
    if "VAL_BATCH_SIZE" not in config:
        env["VAL_BATCH_SIZE"] = config.get("TRAIN_BATCH_TOKENS", DEFAULTS.get("TRAIN_BATCH_TOKENS", "524288"))
    if "VAL_LOSS_EVERY" not in config:
        env["VAL_LOSS_EVERY"] = "0"
    return env

def run_torchrun(env, nproc):
    return subprocess.run(
        ["torchrun", f"--nproc_per_node={nproc}", "train_gpt.py"],
        env=env, capture_output=False,
    )

def parse_log(run_id):
    log_file = Path(f"logs/{run_id}.txt")
    val_bpb = None
    val_loss = None
    if log_file.exists():
        lines = log_file.read_text().split('\n')
        for line in reversed(lines):
            if "final_int8_zlib_roundtrip_exact" in line:
                for part in line.split():
                    if part.startswith("val_bpb:"):
                        try: val_bpb = float(part.split(':')[1])
                        except (ValueError, IndexError): pass
                    if part.startswith("val_loss:"):
                        try: val_loss = float(part.split(':')[1])
                        except (ValueError, IndexError): pass
                if val_bpb is not None and val_loss is not None:
                    break
    return val_bpb, val_loss

# ============================================================================
# SINGLE RUN
# ============================================================================

def run_single(config, label="training"):
    check_data()
    nproc = detect_gpu_count()
    if nproc == 0:
        print("ERROR: No GPU detected!")
        sys.exit(1)

    env = build_env(config)
    run_id = config.get("RUN_ID", "unknown")

    print("=" * 80)
    print(f"RUNNING: {label} (torchrun, {nproc} GPU{'s' if nproc > 1 else ''})")
    print("=" * 80)
    print("\nConfig overrides:")
    for key, val in config.items():
        print(f"  {key}: {val}")
    print()

    result = run_torchrun(env, nproc)

    if result.returncode != 0:
        print("\nTRAINING FAILED!")
        print("Common issues: OOM → reduce TRAIN_BATCH_TOKENS, CUDA error → restart runtime")
        sys.exit(1)

    val_bpb, val_loss = parse_log(run_id)
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    if val_bpb: print(f"Final BPB: {val_bpb:.4f}")
    if val_loss: print(f"Final Loss: {val_loss:.4f}")
    print(f"Log: logs/{run_id}.txt")
    print("=" * 80)

# ============================================================================
# MULTI-EXPERIMENT RUNNER
# ============================================================================

def run_experiments(experiments, save_results=True):
    check_data()
    nproc = detect_gpu_count()
    if nproc == 0:
        print("ERROR: No GPU detected!")
        sys.exit(1)

    total = len(experiments)
    results = []

    print("\n" + "=" * 80)
    print(f"RUNNING {total} EXPERIMENTS (torchrun, {nproc} GPU{'s' if nproc > 1 else ''})")
    print("=" * 80)

    for i, exp in enumerate(experiments, 1):
        name = exp["name"]
        desc = exp.get("description", "")
        config = exp["config"]

        print(f"\n{'=' * 80}")
        print(f"EXPERIMENT {i}/{total}: {name}")
        print(f"Description: {desc}")
        print("=" * 80)
        for key, val in config.items():
            print(f"  {key}: {val}")

        env = build_env(config)
        start_time = time.time()
        result = run_torchrun(env, nproc)
        elapsed = time.time() - start_time

        if result.returncode != 0:
            print(f"\nEXPERIMENT FAILED: {name}")
            results.append({"name": name, "description": desc, "config": config,
                            "status": "FAILED", "elapsed_seconds": elapsed,
                            "val_bpb": None, "val_loss": None})
        else:
            run_id = config.get("RUN_ID", "unknown")
            val_bpb, val_loss = parse_log(run_id)
            print(f"\nCOMPLETE: {name}" + (f" BPB={val_bpb:.4f}" if val_bpb else "") + f" Time={elapsed:.1f}s")
            results.append({"name": name, "description": desc, "config": config,
                            "status": "SUCCESS", "elapsed_seconds": elapsed,
                            "val_bpb": val_bpb, "val_loss": val_loss})

        if i < total:
            time.sleep(5)

    # Summary
    print(f"\n\n{'=' * 80}")
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"\n{'Name':<25} {'Status':<10} {'BPB':<10} {'Loss':<10} {'Time (s)':<10}")
    print("-" * 80)
    for r in results:
        bpb = f"{r['val_bpb']:.4f}" if r['val_bpb'] else "N/A"
        loss = f"{r['val_loss']:.4f}" if r['val_loss'] else "N/A"
        print(f"{r['name'][:24]:<25} {r['status']:<10} {bpb:<10} {loss:<10} {r['elapsed_seconds']:.1f}")

    successful = [r for r in results if r['val_bpb'] is not None]
    if successful:
        best = min(successful, key=lambda r: r['val_bpb'])
        print(f"\nBEST: {best['name']} BPB={best['val_bpb']:.4f}")
        print(f"Config: {json.dumps(best['config'], indent=2)}")

    if save_results:
        results_file = f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")

    return results

# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Parameter Golf: setup, train, and run experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python run.py setup                    Install deps + download SP8192 data
  python run.py quick_smoke              Smoke test (50 steps)
  python run.py train                    Full training (2000 steps)
  python run.py experiment               Custom experiment (edit PRESETS in run.py)
  python run.py sweep                    LR + MLP width sweep
  python run.py ablation                 Layer/MLP ablation study
  python run.py --config my_exps.json    Run custom experiment config
""")
    parser.add_argument("mode", nargs="?", default=None,
                        choices=["setup", "quick_smoke", "train", "experiment", "sweep", "ablation"],
                        help="Run mode or experiment suite")
    parser.add_argument("--config", type=str, help="Path to custom experiment JSON file")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to JSON")

    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            experiments = json.load(f)
        print(f"Loading {len(experiments)} experiments from: {args.config}")
        run_experiments(experiments, save_results=not args.no_save)
        return

    if args.mode is None:
        parser.print_help()
        sys.exit(1)

    if args.mode == "setup":
        setup()
    elif args.mode in PRESETS:
        run_single(PRESETS[args.mode], label=args.mode)
    elif args.mode in SUITES:
        run_experiments(SUITES[args.mode], save_results=not args.no_save)


if __name__ == "__main__":
    main()
