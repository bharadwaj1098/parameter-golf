"""
Unified script for Parameter Golf: setup, training, and experiments.
Default architecture: 10L x 512d, MLP 3x, SP8192.

Usage:
    python run.py setup                                  # Install deps + download data
    python run.py train                                  # Full training (2000 steps)
    python run.py quick                                  # Smoke test (50 steps)
    python run.py sweep                                  # LR + WD sweep (500 steps each)
    python run.py ablation                               # Architecture ablation (1000 steps each)
    python run.py arch                                   # Parallel-residual + recurrence ablation (500 steps each, 18 runs)
    python run.py arch_smoke                             # Smoke-test the arch code paths (4 tiny runs, ~3 min total)
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
    "NUM_LAYERS": "10",
    "MLP_MULT": "3",
    "TIE_EMBEDDINGS": "1",
    "TRAIN_LOG_EVERY": "50",
    "MAX_WALLCLOCK_SECONDS": "3600",
    "SEED": "42",
}

# ============================================================================
# PRESET CONFIGS
# ============================================================================

PRESETS = {
}

# ============================================================================
# EXPERIMENT SUITES
# ============================================================================

QUICK_EXPERIMENTS = [
    {"name": "baseline",    "description": "10L x 512d, MLP 3x",   "config": {"RUN_ID": "exp_baseline",    "ITERATIONS": "50", "TRAIN_BATCH_TOKENS": "65536"}},
    {"name": "dim576",      "description": "10L x 576d, MLP 3x",   "config": {"RUN_ID": "exp_dim576",      "ITERATIONS": "50", "TRAIN_BATCH_TOKENS": "65536", "MODEL_DIM": "576"}},
    {"name": "11L",         "description": "11L x 512d, MLP 3x",   "config": {"RUN_ID": "exp_11L",         "ITERATIONS": "50", "TRAIN_BATCH_TOKENS": "65536", "NUM_LAYERS": "11"}},
]

SWEEP_EXPERIMENTS = [
    {"name": "lr_low",      "description": "MATRIX_LR=0.02", "config": {"RUN_ID": "sweep_lr_low",      "ITERATIONS": "500", "TRAIN_BATCH_TOKENS": "131072", "MATRIX_LR": "0.02"}},
    {"name": "lr_baseline", "description": "MATRIX_LR=0.04", "config": {"RUN_ID": "sweep_lr_baseline", "ITERATIONS": "500", "TRAIN_BATCH_TOKENS": "131072", "MATRIX_LR": "0.04"}},
    {"name": "lr_high",     "description": "MATRIX_LR=0.06", "config": {"RUN_ID": "sweep_lr_high",     "ITERATIONS": "500", "TRAIN_BATCH_TOKENS": "131072", "MATRIX_LR": "0.06"}},
    {"name": "lr_higher",   "description": "MATRIX_LR=0.08", "config": {"RUN_ID": "sweep_lr_higher",   "ITERATIONS": "500", "TRAIN_BATCH_TOKENS": "131072", "MATRIX_LR": "0.08"}},
    {"name": "wd_001",      "description": "WEIGHT_DECAY=0.01","config": {"RUN_ID": "sweep_wd_001",    "ITERATIONS": "500", "TRAIN_BATCH_TOKENS": "131072", "WEIGHT_DECAY": "0.01"}},
    {"name": "wd_004",      "description": "WEIGHT_DECAY=0.04","config": {"RUN_ID": "sweep_wd_004",    "ITERATIONS": "500", "TRAIN_BATCH_TOKENS": "131072", "WEIGHT_DECAY": "0.04"}},
]

ABLATION_EXPERIMENTS = [
    {"name": "baseline",   "description": "10L + MLP 3x (default)",  "config": {"RUN_ID": "ablation_baseline",  "ITERATIONS": "1000", "TRAIN_BATCH_TOKENS": "131072", "SEED": "1337"}},
    {"name": "dim576",     "description": "dim 576",                 "config": {"RUN_ID": "ablation_d576",      "ITERATIONS": "1000", "TRAIN_BATCH_TOKENS": "131072", "SEED": "1337", "MODEL_DIM": "576"}},
    {"name": "12heads",    "description": "12 heads, 4 KV",          "config": {"RUN_ID": "ablation_h12",       "ITERATIONS": "1000", "TRAIN_BATCH_TOKENS": "131072", "SEED": "1337", "NUM_HEADS": "12", "NUM_KV_HEADS": "4"}},
    {"name": "seq4096",    "description": "seq_len 4096",            "config": {"RUN_ID": "ablation_s4096",     "ITERATIONS": "1000", "TRAIN_BATCH_TOKENS": "131072", "SEED": "1337", "TRAIN_SEQ_LEN": "4096"}},
    {"name": "untied_emb", "description": "untied embeddings",       "config": {"RUN_ID": "ablation_notie",     "ITERATIONS": "1000", "TRAIN_BATCH_TOKENS": "131072", "SEED": "1337", "TIE_EMBEDDINGS": "0"}},
    {"name": "11L",        "description": "11 layers",               "config": {"RUN_ID": "ablation_11L",       "ITERATIONS": "1000", "TRAIN_BATCH_TOKENS": "131072", "SEED": "1337", "NUM_LAYERS": "11"}},
]

# ============================================================================
# ARCHITECTURE ABLATION: parallel residuals + depth recurrence
# ============================================================================
# Drawn from leaderboard patterns observed in records/track_10min_16mb:
#   - Parallel residuals typically start layer 5-7 (of 9-11 total)
#     (msisovic PR #1204, Robby955 PR #1412, 2026-03-31 record)
#   - Depth recurrence typically repeats the middle U-Net hinge layers
#     (dexhunter PR #1331/#1437, 2026-04-03 / 2026-04-04 records)
# 10L baseline: encoder = [0..4], decoder = [5..9]. Middle = layers 4-6.
# All runs share seed + step count + 10L/MLP3x shape for a clean comparison.
_ARCH_COMMON = {
    "ITERATIONS": "1000",
    # Bigger batch (4x) for more tokens/step, better gradient SNR on short runs.
    "TRAIN_BATCH_TOKENS": "262144",
    "VAL_BATCH_SIZE": "262144",
    # Baseline defaults warmdown=1200, which is > our whole run — LR would never
    # stay at peak. Shorten to ~15% of training so we get useful learning-rate time.
    "WARMDOWN_ITERS": "150",
    "WARMUP_STEPS": "20",
    # Peak LR bumped ~25% — baseline's 0.04/0.6 are tuned for 20k steps.
    "MATRIX_LR": "0.05",
    "EMBED_LR": "0.8",
    "VAL_LOSS_EVERY": "250",
    "TRAIN_LOG_EVERY": "50",
    "MAX_WALLCLOCK_SECONDS": "3600",
    "SEED": "42",
    # Pin architecture shape explicitly so the ablation is self-documenting.
    "NUM_LAYERS": "10",
    "MLP_MULT": "3",
}

def _arch(name, desc, overrides):
    return {"name": name, "description": desc,
            "config": {"RUN_ID": f"arch_{name}", **_ARCH_COMMON, **overrides}}

ARCH_EXPERIMENTS = [
    # Single best-candidate run: parallel-from-L5 + recur middle pair.
    # Combines both techniques at leaderboard sweet spots (U-Net hinge at L4-L5
    # for a 10L model). Training knobs in _ARCH_COMMON tuned for 1000 steps.
    _arch("par_l5_recur_45",    "parallel L5 + recur 4,5",                {"PARALLEL_START_LAYER": "5", "RECUR_LAYERS": "4,5"}),
]

# Smoke test for the new arch code paths (parallel residuals + recurrence).
# 4 tiny runs (~30-60s each on 1 H100) that exercise all on/off combinations.
# Use this before running `arch` to confirm nothing crashes and BPB is finite.
_ARCH_SMOKE_COMMON = {
    "ITERATIONS": "20",
    "TRAIN_BATCH_TOKENS": "32768",
    "VAL_BATCH_SIZE": "32768",
    "VAL_LOSS_EVERY": "0",
    "TRAIN_LOG_EVERY": "5",
    "MAX_WALLCLOCK_SECONDS": "300",
    "SEED": "42",
    "NUM_LAYERS": "10",
    "MLP_MULT": "3",
}

def _smoke(name, desc, overrides):
    return {"name": name, "description": desc,
            "config": {"RUN_ID": f"smoke_{name}", **_ARCH_SMOKE_COMMON, **overrides}}

ARCH_SMOKE_EXPERIMENTS = [
    _smoke("off",      "neither parallel nor recur",  {}),
    _smoke("par",      "parallel from layer 5",       {"PARALLEL_START_LAYER": "5"}),
    _smoke("recur",    "recur layer 5",               {"RECUR_LAYERS": "5"}),
    _smoke("both",     "parallel L5 + recur 5",       {"PARALLEL_START_LAYER": "5", "RECUR_LAYERS": "5"}),
]

SUITES = {
    "quick": QUICK_EXPERIMENTS,
    "sweep": SWEEP_EXPERIMENTS,
    "ablation": ABLATION_EXPERIMENTS,
    "arch": ARCH_EXPERIMENTS,
    "arch_smoke": ARCH_SMOKE_EXPERIMENTS,
}

# ============================================================================
# SETUP
# ============================================================================

def setup():
    print("=" * 60)
    print("SETUP: Installing dependencies...")
    print("=" * 60)

    subprocess.run(["pip", "install", "-q", "torch==2.8.0", "torchvision", "torchaudio",
                     "--index-url", "https://download.pytorch.org/whl/cu128"], check=True)
    subprocess.run(["pip", "install", "-q", "sentencepiece", "numpy",
                     "huggingface-hub", "datasets", "tqdm", "zstandard", "brotli"], check=True)

    # FlashAttention 3 (Hopper-only, required by the 04-09 submission)
    subprocess.run(
        ["pip", "install", "flash_attn_3", "--no-deps", "--force-reinstall",
         "--find-links", "https://windreamer.github.io/flash-attention3-wheels/cu128_torch280/"],
        check=False,
    )

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
    submission_bytes = None
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
        # "Total submission size ...: NNN bytes" — scan whole file, keep last match
        for line in lines:
            if "Total submission size" in line and "bytes" in line:
                for tok in line.split():
                    if tok.isdigit():
                        submission_bytes = int(tok)
    return val_bpb, val_loss, submission_bytes

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

    val_bpb, val_loss, submission_bytes = parse_log(run_id)
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    if val_bpb: print(f"Final BPB: {val_bpb:.4f}")
    if val_loss: print(f"Final Loss: {val_loss:.4f}")
    if submission_bytes: print(f"Total submission size: {submission_bytes:,} bytes ({submission_bytes/1e6:.2f} MB)")
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
                            "val_bpb": None, "val_loss": None, "submission_bytes": None})
        else:
            run_id = config.get("RUN_ID", "unknown")
            val_bpb, val_loss, submission_bytes = parse_log(run_id)
            size_str = f" Size={submission_bytes/1e6:.2f}MB" if submission_bytes else ""
            print(f"\nCOMPLETE: {name}" + (f" BPB={val_bpb:.4f}" if val_bpb else "") + size_str + f" Time={elapsed:.1f}s")
            results.append({"name": name, "description": desc, "config": config,
                            "status": "SUCCESS", "elapsed_seconds": elapsed,
                            "val_bpb": val_bpb, "val_loss": val_loss,
                            "submission_bytes": submission_bytes})

        if i < total:
            time.sleep(5)

    # Summary
    print(f"\n\n{'=' * 80}")
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print(f"\n{'Name':<25} {'Status':<10} {'BPB':<10} {'Loss':<10} {'Size (MB)':<12} {'Time (s)':<10}")
    print("-" * 90)
    for r in results:
        bpb = f"{r['val_bpb']:.4f}" if r['val_bpb'] else "N/A"
        loss = f"{r['val_loss']:.4f}" if r['val_loss'] else "N/A"
        size = f"{r.get('submission_bytes')/1e6:.2f}" if r.get('submission_bytes') else "N/A"
        print(f"{r['name'][:24]:<25} {r['status']:<10} {bpb:<10} {loss:<10} {size:<12} {r['elapsed_seconds']:.1f}")

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
        description="Parameter Golf: setup, train, and run experiments (10L MLP3x default)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python run.py setup                    Install deps + download SP8192 data
  python run.py quick                    Smoke test (50 steps)
  python run.py train                    Full training (2000 steps)
  python run.py sweep                    LR + WD sweep
  python run.py ablation                 Architecture ablation study
  python run.py arch                     Parallel-residual + recurrence ablation (18 runs, 500 steps)
  python run.py arch_smoke               Smoke-test arch code paths (4 tiny runs, ~3 min total)
  python run.py --config my_exps.json    Run custom experiment config
""")
    parser.add_argument("mode", nargs="?", default=None,
                        choices=["setup", "quick", "train", "sweep", "ablation", "arch", "arch_smoke"],
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
