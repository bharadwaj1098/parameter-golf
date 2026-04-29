"""
Quantization sweep on a trained final_model.pt checkpoint.

One-time training is assumed; this script only varies the QUANTIZER and
re-runs the final sliding-window eval. No training happens here.

Usage (1 GPU):
    torchrun --standalone --nproc_per_node=1 quant_sweep.py

Inputs:
    - MODEL_PATH   (default: "final_model.pt")
    - Uses the same Hyperparameters / data paths / tokenizer as train_gpt.py,
      so architecture knobs (NUM_LAYERS, MLP_MULT, VOCAB_SIZE, ...) must match
      the training run that produced final_model.pt.

Sweeps (Option B: knob sweeps + SDClip):
    - percentile clip: QUANT_CLIP_PERCENTILE in {99.0, 99.9, 99.99, 99.999, 99.99984}
    - bit-widths: (matrix_bits, embed_bits, mlp_bits) variants
    - SDClip: k * std(row) with k in {8, 12, 16}
    - final eval stride: {64, 128, 256, 512}

Output:
    Prints a single sorted table at the end; no CSV.
"""
from __future__ import annotations

import io
import os
import time

import torch
import torch.distributed as dist
import zstandard as zstd
from torch import Tensor, nn

import train_gpt as T  # reuse everything


# ----------------------------------------------------------------------------
# Quantization variants
# ----------------------------------------------------------------------------

def _clip_percentile(t32: Tensor, percentile: float) -> Tensor:
    q = percentile / 100.0
    if t32.ndim == 2:
        if not t32.numel():
            return torch.empty((t32.shape[0],), dtype=torch.float32)
        return torch.quantile(t32.abs(), q, dim=1)
    return torch.tensor(float(torch.quantile(t32.abs().flatten(), q).item()) if t32.numel() else 0.0)


def _clip_sdclip(t32: Tensor, k: float) -> Tensor:
    if t32.ndim == 2:
        std = t32.std(dim=1)
        return (std * k).clamp_min(1e-12)
    std = t32.std()
    return (std * k).clamp_min(1e-12)


def quantize_tensor_generic(t: Tensor, qmax: int, *, clip_mode: str, clip_k: float, clip_percentile: float):
    """Per-row (2D) or scalar (1D) clip + round-to-nearest, returns (q_int8, scale)."""
    t32 = t.float()
    if t32.ndim == 2:
        if clip_mode == "percentile":
            clip_abs = _clip_percentile(t32, clip_percentile)
        elif clip_mode == "sdclip":
            clip_abs = _clip_sdclip(t32, clip_k)
        else:
            raise ValueError(f"unknown clip_mode: {clip_mode}")
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / qmax).clamp_min(1.0 / qmax)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -qmax, qmax).to(torch.int8).contiguous()
        return q, scale.to(dtype=T.QUANT_PER_ROW_SCALE_DTYPE).contiguous()

    if clip_mode == "percentile":
        clip_abs = float(_clip_percentile(t32, clip_percentile).item()) if t32.numel() else 0.0
    else:
        clip_abs = float(_clip_sdclip(t32, clip_k).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / qmax if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -qmax, qmax).to(torch.int8).contiguous()
    return q, scale


def quantize_state_dict_generic(
    state_dict: dict[str, Tensor],
    *,
    matrix_bits: int,
    embed_bits: int,
    mlp_bits: int,
    clip_mode: str = "percentile",
    clip_k: float = 12.0,
    clip_percentile: float = 99.99984,
):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()

        if not t.is_floating_point():
            passthrough[name] = t
            continue

        if t.numel() <= T.QUANT_KEEP_FLOAT_MAX_NUMEL:
            passthrough[name] = T.keep_float_tensor(name, t, passthrough_orig_dtypes)
            continue

        is_mlp = any(p in name for p in T.MLP_NAME_PATTERNS)
        is_embed = any(p in name for p in T.EMBED_NAME_PATTERNS)
        if is_embed:
            bits = embed_bits
        elif is_mlp:
            bits = mlp_bits
        else:
            bits = matrix_bits
        qmax = (1 << (bits - 1)) - 1

        q, s = quantize_tensor_generic(
            t, qmax=qmax,
            clip_mode=clip_mode, clip_k=clip_k, clip_percentile=clip_percentile,
        )
        meta: dict[str, object] = {"bits": bits}
        if s.ndim > 0:
            meta["scheme"] = "per_row"
            meta["axis"] = 0
        qmeta[name] = meta
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")

    obj: dict[str, object] = {
        "__quant_format__": "generic_sweep_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj


# ----------------------------------------------------------------------------
# Sweep configurations
# ----------------------------------------------------------------------------

def build_sweep():
    configs = []

    def cfg(name, **kw):
        # Defaults match current train_gpt.py settings:
        base = dict(
            matrix_bits=6, embed_bits=8, mlp_bits=5,
            clip_mode="percentile", clip_k=12.0, clip_percentile=99.99984,
            eval_stride=64,
        )
        base.update(kw)
        base["name"] = name
        configs.append(base)

    # Baseline (current production config)
    cfg("baseline (int6/8/5 + 99.99984 pct + stride64)")

    # Percentile sweep (keep bits fixed at current)
    cfg("pct_99.0",    clip_percentile=99.0)
    cfg("pct_99.9",    clip_percentile=99.9)
    cfg("pct_99.99",   clip_percentile=99.99)
    cfg("pct_99.999",  clip_percentile=99.999)

    # Embed-bits sweep (biggest lever historically)
    cfg("embed_bits=6", embed_bits=6)
    cfg("embed_bits=7", embed_bits=7)

    # Matrix-bits sweep
    cfg("matrix_bits=5", matrix_bits=5)
    cfg("matrix_bits=7", matrix_bits=7)

    # MLP-bits sweep (size tradeoff)
    cfg("mlp_bits=4", mlp_bits=4)
    cfg("mlp_bits=6", mlp_bits=6)

    # SDClip with current bits
    cfg("sdclip_k=8",  clip_mode="sdclip", clip_k=8.0)
    cfg("sdclip_k=12", clip_mode="sdclip", clip_k=12.0)
    cfg("sdclip_k=16", clip_mode="sdclip", clip_k=16.0)

    # SDClip aggressive-low-bit combos
    cfg("sdclip_k=12 + int4mlp + int7mat", clip_mode="sdclip", clip_k=12.0, mlp_bits=4, matrix_bits=7)
    cfg("sdclip_k=16 + int8emb + int6mat", clip_mode="sdclip", clip_k=16.0)

    # Eval stride (cost/quality tradeoff; doesn't affect size)
    cfg("stride=128",  eval_stride=128)
    cfg("stride=256",  eval_stride=256)
    cfg("stride=512",  eval_stride=512)

    return configs


# ----------------------------------------------------------------------------
# Load weights + eval
# ----------------------------------------------------------------------------

def setup_distributed():
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
    return distributed, rank, world_size, device


def build_model(args, device):
    model = T.GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        parallel_start_layer=args.parallel_start_layer,
        recur_layers=args.recur_layers,
        recur_extra_count=args.recur_extra_count,
    ).to(device).bfloat16()
    for module in model.modules():
        if isinstance(module, T.CastedLinear):
            module.float()
    T.restore_low_dim_params_to_fp32(model)
    return model


def eval_one(args, model, rank, world_size, device, val_tokens, luts, eval_stride: int):
    # Temporarily override args.eval_stride for this eval (eval_val reads it).
    orig_stride = args.eval_stride
    args.eval_stride = eval_stride
    try:
        loss, bpb = T.eval_val(
            args, model, rank, world_size, device,
            grad_accum_steps=1,
            val_tokens=val_tokens,
            base_bytes_lut=luts[0],
            has_leading_space_lut=luts[1],
            is_boundary_token_lut=luts[2],
            use_sliding_window=True,
        )
    finally:
        args.eval_stride = orig_stride
    return loss, bpb


def run_one_sweep(args, model_sd_cpu, model, rank, world_size, device, val_tokens, luts, cfg):
    # Quantize (CPU) → compress → decompress → dequantize → load into model → eval.
    t0 = time.perf_counter()
    quant_obj = quantize_state_dict_generic(
        model_sd_cpu,
        matrix_bits=cfg["matrix_bits"],
        embed_bits=cfg["embed_bits"],
        mlp_bits=cfg["mlp_bits"],
        clip_mode=cfg["clip_mode"],
        clip_k=cfg["clip_k"],
        clip_percentile=cfg["clip_percentile"],
    )
    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    raw = buf.getvalue()
    blob = zstd.ZstdCompressor(level=22).compress(raw)
    size_compressed = len(blob)

    # Round-trip
    decompressed = zstd.ZstdDecompressor().decompress(blob)
    loaded = torch.load(io.BytesIO(decompressed), map_location="cpu")
    restored = T.dequantize_state_dict_mixed(loaded)
    model.load_state_dict(restored, strict=True)

    t1 = time.perf_counter()
    loss, bpb = eval_one(args, model, rank, world_size, device, val_tokens, luts, cfg["eval_stride"])
    t2 = time.perf_counter()

    return {
        "name": cfg["name"],
        "matrix_bits": cfg["matrix_bits"],
        "embed_bits": cfg["embed_bits"],
        "mlp_bits": cfg["mlp_bits"],
        "clip_mode": cfg["clip_mode"],
        "clip_k": cfg["clip_k"],
        "clip_percentile": cfg["clip_percentile"],
        "eval_stride": cfg["eval_stride"],
        "size_mb": size_compressed / 1e6,
        "val_bpb": bpb,
        "val_loss": loss,
        "quant_seconds": t1 - t0,
        "eval_seconds": t2 - t1,
    }


def main():
    args = T.Hyperparameters()

    distributed, rank, world_size, device = setup_distributed()
    master = rank == 0

    def log(msg):
        if master:
            print(msg, flush=True)

    model_path = os.environ.get("MODEL_PATH", "final_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No checkpoint at {model_path} (set MODEL_PATH or cd to the right dir)")

    log(f"Loading {model_path}")
    sd = torch.load(model_path, map_location="cpu")

    # Build model skeleton + val data
    model = build_model(args, device)
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = T.load_validation_tokens(args.val_files, args.train_seq_len)
    luts = T.build_sentencepiece_luts(sp, args.vocab_size, device)
    log(f"val tokens: {val_tokens.numel() - 1}")

    configs = build_sweep()
    log(f"Running {len(configs)} quantization configs")

    results = []
    for i, cfg in enumerate(configs, 1):
        log(f"[{i}/{len(configs)}] {cfg['name']}")
        try:
            r = run_one_sweep(args, sd, model, rank, world_size, device, val_tokens, luts, cfg)
            results.append(r)
            if master:
                print(f"    BPB={r['val_bpb']:.4f}  size={r['size_mb']:.2f}MB  "
                      f"quant={r['quant_seconds']:.1f}s  eval={r['eval_seconds']:.1f}s", flush=True)
        except Exception as e:
            log(f"    FAILED: {type(e).__name__}: {e}")
            results.append({"name": cfg["name"], "val_bpb": float("nan"), "size_mb": float("nan"),
                            "val_loss": float("nan"), "quant_seconds": 0, "eval_seconds": 0,
                            "matrix_bits": cfg["matrix_bits"], "embed_bits": cfg["embed_bits"],
                            "mlp_bits": cfg["mlp_bits"], "clip_mode": cfg["clip_mode"],
                            "clip_k": cfg["clip_k"], "clip_percentile": cfg["clip_percentile"],
                            "eval_stride": cfg["eval_stride"]})

    # Final sorted summary (by BPB ascending; failed configs at bottom).
    if master:
        print("\n" + "=" * 110)
        print("QUANTIZATION SWEEP SUMMARY  (sorted by BPB ascending)")
        print("=" * 110)
        print(f"{'name':<44} {'matB':>4} {'embB':>4} {'mlpB':>4} {'clip':>10} {'k/pct':>9} "
              f"{'stride':>6} {'size_MB':>8} {'val_bpb':>8}")
        print("-" * 110)
        finite = [r for r in results if r["val_bpb"] == r["val_bpb"]]  # drop nan
        failed = [r for r in results if r["val_bpb"] != r["val_bpb"]]
        finite.sort(key=lambda r: r["val_bpb"])
        for r in finite + failed:
            pct_or_k = r["clip_k"] if r["clip_mode"] == "sdclip" else r["clip_percentile"]
            bpb_str = f"{r['val_bpb']:.4f}" if r["val_bpb"] == r["val_bpb"] else "FAILED"
            size_str = f"{r['size_mb']:.2f}" if r["size_mb"] == r["size_mb"] else "N/A"
            print(f"{r['name'][:43]:<44} {r['matrix_bits']:>4} {r['embed_bits']:>4} {r['mlp_bits']:>4} "
                  f"{r['clip_mode']:>10} {pct_or_k:>9.3f} {r['eval_stride']:>6} "
                  f"{size_str:>8} {bpb_str:>8}")
        print("=" * 110)
        print(f"Best: {finite[0]['name']} — BPB {finite[0]['val_bpb']:.4f}, {finite[0]['size_mb']:.2f} MB")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
