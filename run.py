"""
Parameter Golf: setup + architecture experiments.

Usage:
    python run.py setup        # Install deps + download SP8192 data
    python run.py arch_smoke   # Tiny pipeline smoke test (~2 min on 1 H100)
    python run.py arch         # 1×H100 iteration config (~30 min)
    python run.py arch_multi   # 8×H100 submission-shape config (~10 min)
    python run.py arch_exp     # Attention-mechanism ablation grid (9 runs × 2500 steps, 1×H100, disjoint eval)

Standard config: 10L x 512d, MLP 3x, SP8192, parallel L5 + recur layers 4,5.
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
# STANDARD CONFIG — matches the 1.1798 BPB, 14.88 MB, 8×H100 600s result.
# ============================================================================
# Every suite below inherits this. Only the per-suite dict below overrides the
# handful of knobs that truly differ by context (wallclock budget, eval stride,
# iteration counts, logging cadence).
_STANDARD = {
    # Architecture
    "NUM_LAYERS": "10",
    "MLP_MULT": "3",
    "PARALLEL_START_LAYER": "5",
    "RECUR_LAYERS": "4,5",
    # Training
    "SEED": "42",
    "TRAIN_BATCH_TOKENS": "524288",
    "VAL_BATCH_SIZE": "524288",
    "WARMDOWN_ITERS": "400",
    "WARMUP_STEPS": "20",
    "MATRIX_LR": "0.05",
    "EMBED_LR": "0.8",
    "MUON_WD": "0.025",
    # Eval
    "EVAL_STRIDE": "64",
}

# --- 8×H100 (arch_multi) — submission-shape, matches 1.1798 reference run ---
# Wallclock-governed to 600s. ITERATIONS is a safe upper bound we never hit —
# training stops on wallclock, LR warmdown projects against elapsed time.
_ARCH_MULTI_COMMON = {
    **_STANDARD,
    "MAX_WALLCLOCK_SECONDS": "600",
    "ITERATIONS": "50000",
    "VAL_LOSS_EVERY": "1000",
    "TRAIN_LOG_EVERY": "100",
}

def _arch_multi(name, desc, overrides):
    return {"name": name, "description": desc,
            "config": {"RUN_ID": f"multi_{name}", **_ARCH_MULTI_COMMON, **overrides}}

ARCH_MULTI_EXPERIMENTS = [
    _arch_multi("par_l5_recur_45", "parallel L5 + recur 4,5 (8×H100, 600s cap)", {}),
]

# --- 1×H100 (arch) — single-GPU iteration config ---------------------------
# Same model/training knobs as _STANDARD; only wallclock and eval stride differ.
# Stride 512 keeps eval ~1 min on 1 GPU (vs ~7 min at stride 64) — captures
# ~90% of the sliding-window BPB gain at 1/8 the cost.
_ARCH_COMMON = {
    **_STANDARD,
    "ITERATIONS": "2500",
    "VAL_LOSS_EVERY": "500",
    "TRAIN_LOG_EVERY": "50",
    "MAX_WALLCLOCK_SECONDS": "3600",
    "EVAL_STRIDE": "512",
}

def _arch(name, desc, overrides):
    return {"name": name, "description": desc,
            "config": {"RUN_ID": f"arch_{name}", **_ARCH_COMMON, **overrides}}

ARCH_EXPERIMENTS = [
    _arch("par_l5_recur_45", "parallel L5 + recur 4,5", {}),
]

# --- Smoke test (arch_smoke) — ~2 min on 1 H100 ----------------------------
# Shortest possible run that exercises every code path: parallel residuals,
# mini recurrence, Muon WD, sliding-window eval. Final BPB will be bad —
# this is just to verify nothing crashes before committing to a full run.
_ARCH_SMOKE_COMMON = {
    **_STANDARD,
    "ITERATIONS": "50",
    "TRAIN_BATCH_TOKENS": "32768",   # smaller to fit cold-start wallclock
    "VAL_BATCH_SIZE": "32768",
    "VAL_LOSS_EVERY": "0",           # no mid-training eval
    "TRAIN_LOG_EVERY": "10",
    "MAX_WALLCLOCK_SECONDS": "600",
    "WARMDOWN_ITERS": "10",
    "WARMUP_STEPS": "5",
    "EVAL_STRIDE": "512",            # fast final eval
}

def _smoke(name, desc, overrides):
    return {"name": name, "description": desc,
            "config": {"RUN_ID": f"smoke_{name}", **_ARCH_SMOKE_COMMON, **overrides}}

ARCH_SMOKE_EXPERIMENTS = [
    _smoke("standard", "parallel L5 + recur 4,5 (standard)", {}),
]

# --- Experimental (arch_exp) — runs train_gpt_exp.py -----------------------
# Attention-mechanism ablation study. All runs share: 10L MLP3×, parallel L5,
# recur 4,5, WD 0.025, MATRIX_LR 0.05 (from _STANDARD). Fixed 2500 steps on
# 1×H100 — same step budget across runs keeps comparisons clean regardless of
# GPU-speed variability. Disjoint eval (EVAL_STRIDE=0) for fast iteration;
# if a variant beats baseline on disjoint, it'll beat on stride=64 too.
_ARCH_EXP_COMMON = {
    **_STANDARD,
    "ITERATIONS": "2500",
    "MAX_WALLCLOCK_SECONDS": "3600",  # safety cap only; 2500 steps fits easily
    "VAL_LOSS_EVERY": "500",
    "TRAIN_LOG_EVERY": "50",
    "EVAL_STRIDE": "0",          # disjoint eval for fast iteration
}

def _arch_exp(name, desc, overrides):
    return {
        "name": name,
        "description": desc,
        "script": "train_gpt_exp.py",
        "config": {"RUN_ID": f"exp_{name}", **_ARCH_EXP_COMMON, **overrides},
    }

# Attention-only ablation grid. Every run sets USE_NEW_ATTENTION=1 so the
# CausalSelfAttentionNew path is exercised; USE_K_GAIN / USE_HEAD_GATE /
# NUM_KV_HEADS are the three knobs being swept. The first row is the baseline
# reference (USE_NEW_ATTENTION=0 — original attention class).
#
# Runs 1, 7, 9 (baseline_kv4, kv4_both, kv2_both) already completed — commented
# out so re-running `python run.py arch_exp` executes only the remaining 6.
# Completed results (post-quant BPB, step 2500):
#   baseline_kv4: 1.26969528
#   kv4_both:     1.26320724
#   kv2_both:     1.26880768
ARCH_EXP_EXPERIMENTS = [
    # Control
    # _arch_exp("baseline_kv4",     "baseline attn, kv=4",  # DONE: 1.26969528
    #           {"USE_NEW_ATTENTION": "0", "NUM_KV_HEADS": "4",
    #            "USE_K_GAIN": "0", "USE_HEAD_GATE": "0"}),
    # New-attention baseline (no feature flags) — isolates cost of always-on
    # q_norm_scale/k_norm_scale vs. the baseline class.
    _arch_exp("new_attn_kv4",     "new attn, kv=4, no extras",
              {"USE_NEW_ATTENTION": "1", "NUM_KV_HEADS": "4",
               "USE_K_GAIN": "0", "USE_HEAD_GATE": "0"}),
    # KV-head sweep
    _arch_exp("new_attn_kv2",     "new attn, kv=2",
              {"USE_NEW_ATTENTION": "1", "NUM_KV_HEADS": "2",
               "USE_K_GAIN": "0", "USE_HEAD_GATE": "0"}),
    _arch_exp("new_attn_kv1",     "new attn, kv=1 (MQA)",
              {"USE_NEW_ATTENTION": "1", "NUM_KV_HEADS": "1",
               "USE_K_GAIN": "0", "USE_HEAD_GATE": "0"}),
    # Feature ablations at kv=4
    _arch_exp("kv4_kgain",        "kv=4 + k_gain",
              {"USE_NEW_ATTENTION": "1", "NUM_KV_HEADS": "4",
               "USE_K_GAIN": "1", "USE_HEAD_GATE": "0"}),
    _arch_exp("kv4_headgate",     "kv=4 + head_gate",
              {"USE_NEW_ATTENTION": "1", "NUM_KV_HEADS": "4",
               "USE_K_GAIN": "0", "USE_HEAD_GATE": "1"}),
    # _arch_exp("kv4_both",         "kv=4 + k_gain + head_gate",  # DONE: 1.26320724
    #           {"USE_NEW_ATTENTION": "1", "NUM_KV_HEADS": "4",
    #            "USE_K_GAIN": "1", "USE_HEAD_GATE": "1"}),
    # Best-combo candidates
    _arch_exp("kv2_kgain",        "kv=2 + k_gain",
              {"USE_NEW_ATTENTION": "1", "NUM_KV_HEADS": "2",
               "USE_K_GAIN": "1", "USE_HEAD_GATE": "0"}),
    # _arch_exp("kv2_both",         "kv=2 + k_gain + head_gate",  # DONE: 1.26880768
    #           {"USE_NEW_ATTENTION": "1", "NUM_KV_HEADS": "2",
    #            "USE_K_GAIN": "1", "USE_HEAD_GATE": "1"}),
]

SUITES = {
    "arch": ARCH_EXPERIMENTS,
    "arch_multi": ARCH_MULTI_EXPERIMENTS,
    "arch_smoke": ARCH_SMOKE_EXPERIMENTS,
    "arch_exp": ARCH_EXP_EXPERIMENTS,
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
    print("\nNext: python run.py arch_smoke")

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

def run_torchrun(env, nproc, script="train_gpt.py"):
    return subprocess.run(
        ["torchrun", f"--nproc_per_node={nproc}", script],
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
            if line.startswith("final_mixed_zstd_roundtrip"):
                for part in line.split():
                    if part.startswith("val_bpb:"):
                        try: val_bpb = float(part.split(':')[1])
                        except (ValueError, IndexError): pass
                    if part.startswith("val_loss:"):
                        try: val_loss = float(part.split(':')[1])
                        except (ValueError, IndexError): pass
                if val_bpb is not None and val_loss is not None:
                    break
        for line in lines:
            if "Total submission size" in line and "bytes" in line:
                for tok in line.split():
                    if tok.isdigit():
                        submission_bytes = int(tok)
    return val_bpb, val_loss, submission_bytes

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
        # Each experiment can target a specific training script (default: train_gpt.py).
        script = exp.get("script", "train_gpt.py")
        start_time = time.time()
        result = run_torchrun(env, nproc, script=script)
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
        description="Parameter Golf: setup + architecture experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python run.py setup        Install deps + download SP8192 data
  python run.py arch_smoke   Tiny pipeline smoke test (~2 min on 1 H100)
  python run.py arch         1×H100 iteration config (~30 min)
  python run.py arch_multi   8×H100 submission-shape config (~10 min)
  python run.py arch_exp     Attention-mechanism ablation grid (9 runs × 2500 steps, 1×H100, disjoint eval)
""")
    parser.add_argument("mode", nargs="?", default=None,
                        choices=["setup", "arch", "arch_multi", "arch_smoke", "arch_exp"],
                        help="Run mode")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to JSON")

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        sys.exit(1)

    if args.mode == "setup":
        setup()
    elif args.mode in SUITES:
        run_experiments(SUITES[args.mode], save_results=not args.no_save)


if __name__ == "__main__":
    main()
