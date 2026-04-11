# Parameter Golf Winning Techniques

## Current SOTA: 1.0810 BPB (Apr 9, 2026)
Down from baseline 1.2244 BPB (0.1434 improvement in 3 weeks)

---

## Core Technique Stack (Must-Have)

### 1. Sliding Window Evaluation (stride=64)
**BPB Gain**: ~0.032 | **Difficulty**: Easy | **Type**: Eval-only

Instead of non-overlapping 1024-token chunks (where first token has zero context), use overlapping windows with stride 64. Each scored token gets 960+ tokens of context.

```python
# Baseline: each chunk independent
for chunk in non_overlapping_chunks(val_tokens, 1024):
    score(chunk)  # first token has 0 context

# Sliding window: overlap with stride
for window_start in range(0, len(val_tokens), 64):
    window = val_tokens[window_start:window_start+1024]
    score(window[-64:])  # only score last 64 tokens (they have 960 context)
```

**Cost**: Eval takes ~70s vs ~16s (still under 600s budget)

---

### 2. Larger Vocabulary (SP8192)
**BPB Gain**: ~0.037 | **Difficulty**: Medium | **Type**: Training change

**Key insight**: BPB = (loss / ln(2)) × (tokens/bytes). Larger vocab → fewer tokens per byte.

| Vocab | BPB | Tokens/Byte | Embedding Size |
|-------|-----|-------------|----------------|
| 1024  | 1.224 | ~0.25 | 512KB |
| 4096  | 1.098 | ~0.30 | 2MB |
| 8192  | 1.086 | ~0.33 | 4MB |

**Tradeoff**: Embedding table grows (vocab × dim), but tied embeddings + compression help.

**Implementation**: Train new SentencePiece tokenizer with higher vocab_size, retokenize dataset.

---

### 3. Depth Recurrence (Weight Sharing)
**BPB Gain**: ~0.015 | **Difficulty**: Medium | **Type**: Architecture

Reuse transformer layers multiple times. Instead of 11 unique layers, use 8 unique layers but loop some.

**Example (3-layer recurrence)**:
```
Physical layers: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Virtual execution: [0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5, 6, 7, 8, 9, 10]
                              └─────┘ └─────┘ └─────┘
                               loop 3x (activate at step 2016)
```

17 virtual layers from 11 physical → saves params for same effective depth.

**Activation timing**: Enable loops partway through training (e.g., 35% for first loop, 50% for second).

**Reference**: ALBERT, Universal Transformer

---

### 4. Parallel Residuals (GPT-J Style)
**BPB Gain**: ~0.005 | **Difficulty**: Easy | **Type**: Architecture

Attention and MLP read from same input and add in parallel (instead of sequentially).

```python
# Sequential (baseline):
h = x + attn(norm(x))
x = h + mlp(norm(h))

# Parallel (GPT-J):
x = x + attn(norm(x)) + mlp(norm(x))
```

**Why it helps**: Less interference during quantization, attention/MLP can specialize independently.

**Apply to**: Last 4-5 layers (layers 7-10 in 11L model)

---

### 5. Test-Time Training (TTT) — Legal Score-First
**BPB Gain**: ~0.015-0.025 | **Difficulty**: Hard | **Type**: Eval adaptation

Fine-tune the model on validation data **during evaluation**, but score tokens **before** updating weights.

**Legal implementation** (satisfies Issue #1017):
```python
for chunk in val_chunks(32768):  # 32K token chunks
    # Phase 1: SCORE (no gradients, no updates)
    with torch.no_grad():
        loss_sum += score_all_windows(chunk)
    
    # Phase 2: TRAIN (only on already-scored tokens)
    for epoch in range(3):
        for x, y in chunk_sequences:
            loss = model(x, y)
            loss.backward()
            optimizer.step()  # SGD, lr=0.005, momentum=0.9
```

**Rules**:
- Score before update (causality)
- No rescoring
- Each token scored exactly once
- Standard softmax (no n-gram cache, no logit bias)

**Cost**: ~290s eval time (within 600s budget)

---

## Quantization & Compression

### 6. GPTQ Quantization with SDClip
**BPB Gain**: ~0.01 | **Difficulty**: Hard | **Type**: Post-training

Full-Hessian GPTQ for principled quantization. Clip threshold based on standard deviation:

```
clip = k × std(row)
```

**Settings**:
- int6 for attention/MLP matrices (k=12.85)
- int8 for embeddings (k=20.0)
- fp16/fp32 passthrough for small tensors (<65K params)

**Why SDClip**: Tight clipping → low entropy → better compression. Standard deviation is stable across seeds.

---

### 7. Mixed Precision Quantization
**BPB Gain**: ~0.003 | **Difficulty**: Medium | **Type**: Compression

Different bitwidths for different layers:
- int5 for MLP weights (less sensitive, compress better)
- int6 for attention (more sensitive)
- fp16 for embeddings

Saves ~1.86MB vs uniform int6 → funds an extra layer.

---

### 8. Quantization-Aware Training (QAT)
**BPB Gain**: ~0.01 | **Difficulty**: Medium | **Type**: Training

Simulate quantization during training via Straight-Through Estimator (STE):

```python
def fake_quantize(w, scale):
    # Forward: quantize
    w_q = torch.clamp(torch.round(w / scale), -31, 31) * scale
    
    # Backward: straight-through (gradient flows as if no quantization)
    return w_q + (w - w.detach())  # STE trick
```

Model learns to be robust to quantization → smaller roundtrip gap.

---

### 9. Better Compression (zstd-22 / Brotli-11)
**Bytes Saved**: ~1.5MB | **Difficulty**: Easy | **Type**: Compression

Replace zlib with higher-compression algorithms:
- zstd level 22
- Brotli level 11

Same model, smaller artifact → fit more layers/width.

---

## Optimizer & Training

### 10. MuonEq-R (Row-Normalized Muon)
**BPB Gain**: ~0.005 | **Difficulty**: Medium | **Type**: Optimizer

Row-normalized variant of Muon optimizer:

```python
# Standard Muon: orthogonalize full matrix
G_ortho = newton_schulz(G)

# MuonEq-R: orthogonalize each row independently
for row in G:
    row_ortho = newton_schulz(row.unsqueeze(0))
    G[i] = row_ortho
```

Better per-row conditioning → faster convergence.

---

### 11. Weight Decay (0.085-0.095)
**BPB Gain**: ~0.005 | **Difficulty**: Easy | **Type**: Training

Decoupled weight decay on both Muon and Adam:

```python
p.mul_(1 - wd * lr)  # before gradient update
```

**Why it helps**:
- Smaller weights → tighter distribution
- Better quantization (fewer outliers)
- Better compression (lower entropy)

Optimal: WD=0.085-0.095 (higher than typical 0.01)

---

### 12. Exponential Moving Average (EMA)
**BPB Gain**: ~0.002 | **Difficulty**: Easy | **Type**: Training

Average weights across recent steps:

```python
ema_weights = 0.9965 * ema_weights + 0.0035 * current_weights
```

Use EMA weights for final eval instead of last checkpoint. Smoother convergence.

---

### 13. Stochastic Weight Averaging (SWA)
**BPB Gain**: ~0.002 | **Difficulty**: Easy | **Type**: Training

Average checkpoints from late training:

```python
# Collect checkpoints from last 40% of training
for step in range(start_frac * total_steps, total_steps):
    if step % 50 == 0:
        checkpoints.append(model.state_dict())

# Average them
final_weights = average(checkpoints)
```

**vs EMA**: SWA averages discrete checkpoints, EMA is continuous. Both help.

---

## Architecture Tweaks

### 14. MLP Expansion (3x-4x)
**BPB Gain**: ~0.01 | **Difficulty**: Easy | **Type**: Architecture

Baseline: MLP hidden = 2 × dim (512 → 1024)
SOTA: MLP hidden = 4 × dim (512 → 2048)

Wider MLPs → more capacity per layer. Funded by better compression.

---

### 15. QK-Gain Tuning (5.0-5.25)
**BPB Gain**: ~0.003 | **Difficulty**: Easy | **Type**: Training

Learnable per-head query scaling:

```python
q = q * self.q_gain[None, :, None, None]  # shape: (num_heads,)
```

Initialize higher: 4.0 → 5.0 → 5.25 (monotonic improvement)

---

### 16. Partial RoPE (16/64 dims)
**BPB Gain**: ~0.002 | **Difficulty**: Easy | **Type**: Architecture

Only apply RoPE to first 16 dims of 64-dim head (instead of all 64).

Saves computation, slight quality improvement. Used in Gemma.

---

### 17. Layerwise LN Scale
**BPB Gain**: ~0.001 | **Difficulty**: Easy | **Type**: Architecture

Per-layer learned scale on normalization output:

```python
x = rms_norm(x) * self.ln_scale[layer_idx]
```

Helps deep networks balance layer contributions.

---

### 18. LeakyReLU(0.5)² Activation
**BPB Gain**: ~0.002 | **Difficulty**: Easy | **Type**: Architecture

Replace relu² with leaky_relu(0.5)²:

```python
# Baseline:
x = relu(fc(x))²

# SOTA:
x = leaky_relu(fc(x), 0.5)²
```

Reduces dead neurons, smoother gradients.

---

## Advanced Techniques

### 19. BigramHash Embeddings
**BPB Gain**: ~0.005 | **Difficulty**: Medium | **Type**: Architecture

Hash table for token pairs:

```python
hash_idx = (prev_token * 92821 + cur_token) % 4096
bigram_emb = hash_table[hash_idx]  # (4096, 128)
projected = bigram_emb @ proj  # (128, 512)
```

Cheap bigram features (~0.5M params). Larger tables (10240) compress worse but model better.

---

### 20. SmearGate
**BPB Gain**: ~0.005 | **Difficulty**: Easy | **Type**: Architecture

Blend current token embedding with previous token:

```python
gate = sigmoid(self.gate)  # init ~0.95
emb_out = gate * cur_emb + (1 - gate) * prev_emb
```

Injects bigram context before transformer. Per-dimension learned gate.

---

### 21. XSA (Cross-Layer Self-Attention)
**BPB Gain**: ~0.003 | **Difficulty**: Medium | **Type**: Architecture

Attention across layers (not just within):

```python
# Layer L attends to hidden states from layers L-1, L-2, L-3
q = current_layer_hidden
k, v = concat([layer[L-1], layer[L-2], layer[L-3]])
```

More expensive, used selectively on deepest layers.

---

### 22. Skip Gates (U-Net with learned weights)
**BPB Gain**: ~0.002 | **Difficulty**: Easy | **Type**: Architecture

Baseline has fixed skip connections. Add learned gates:

```python
skip_out = sigmoid(gate) * encoder_hidden
x = x + skip_out
```

Model learns which skips to use.

---

### 23. Progressive Recurrence
**BPB Gain**: ~0.001 | **Difficulty**: Easy | **Type**: Training

Enable depth recurrence in phases:
- 50% training: enable first loop
- 65% training: enable second loop

Smoother adaptation than enabling all at once.

---

### 24. Hessian-Aware SDClip
**BPB Gain**: ~0.001 | **Difficulty**: Hard | **Type**: Quantization

Modulate clip threshold by row importance from Hessian:

```python
importance = hessian_diag @ weight²  # per-row
clip = k * std(row) * (1 + λ * (importance - 1))
```

λ=0.175 works best. Higher λ reduces error but hurts compression.

---

## Technique Evolution Timeline

### Mar 17-20: Compression Era
- Sliding window eval
- Int6 QAT + zstd
- 3x MLP, 10-11 layers
- BigramHash, SmearGate, SWA
- **Best**: 1.1428

### Apr 1-5: Tokenizer Breakthrough
- SP4096: 1.0979
- SP8192: 1.0856
- **Gain**: 0.057 BPB

### Apr 3-6: Architecture Stack
- Depth recurrence
- Parallel residuals
- MuonEq-R
- GPTQ embeddings
- QK-Gain tuning
- **Best**: 1.0835

### Apr 6-9: TTT Dominance
- Legal score-first TTT
- 3-layer recurrence
- Tuned hyperparams (WD=0.095, EMA=0.9965)
- **Best**: 1.0810

---

## Quick Reference: Impact vs Effort

### High Impact, Easy (Do First)
- Sliding window eval: 0.032 BPB, 30 lines of code
- Larger vocab (SP4096): 0.025 BPB, retokenize dataset
- 3x-4x MLP: 0.01 BPB, change one number
- Weight decay 0.085: 0.005 BPB, one line
- zstd-22: saves 1.5MB, swap compressor

### High Impact, Medium
- Depth recurrence: 0.015 BPB, ~100 lines
- Int6 QAT: 0.01 BPB, add STE to forward
- Parallel residuals: 0.005 BPB, ~50 lines
- GPTQ quantization: 0.01 BPB, complex

### High Impact, Hard
- TTT: 0.015-0.025 BPB, ~300 lines + careful rules
- SP8192 tokenizer: 0.012 BPB (over SP4096), retokenize everything

### Diminishing Returns
- BigramHash: 0.005 BPB, costs params
- SmearGate: 0.005 BPB, small trick
- XSA: 0.003 BPB, expensive compute
- Skip gates: 0.002 BPB

---

## Full SOTA Stack (1.0810)

```
Tokenizer:          SP8192 (8192 vocab, SentencePiece BPE)
Architecture:       11L × 512d × 8H / 4KV
MLP:                4x expansion (512 → 2048)
Activation:         LeakyReLU(0.5)²
Depth Recurrence:   3-layer (layers 3,4,5 loop 3× → 17 virtual)
Parallel Residuals: Layers 7-10 (GPT-J style)
Positional:         Partial RoPE (16/64 dims)
Embeddings:         Tied, GPTQ int8 (k=20.0)
Skip Connections:   U-Net with learned gates
Layerwise:          LN scale per layer

Optimizer:          MuonEq-R (row-normalized, 5 Newton-Schulz steps)
Matrix LR:          0.022
Weight Decay:       0.095 (decoupled)
EMA:                0.9965
Warmdown:           72% of training (linear to 0)
QK-Gain:            5.25 (learnable per-head)

Quantization:       GPTQ with SDClip
  - Matrices:       int6, k=12.85
  - Embeddings:     int8, k=20.0
  - Small tensors:  fp16/fp32 passthrough
Compression:        Brotli-11 + LZMA code wrapper

Eval:               Sliding window (stride=64)
                  + Legal Score-First TTT
                    - 3 epochs per 32K chunk
                    - SGD lr=0.005, momentum=0.9
                    - Cosine LR decay across chunks
                    - Score before update (causality)

Training:           4550 steps, 588s on 8xH100
Artifact:           ~15.99MB (99.9% of 16MB cap)
```

---

## References

- Baseline: openai/parameter-golf
- Muon: https://kellerjordan.github.io/posts/muon/
- ALBERT: Lan et al. 2019 (weight sharing)
- GPT-J: Wang & Komatsuzaki 2021 (parallel residuals)
- GPTQ: Frantar et al. 2022
- Score-First TTT: PR #549 (abaybektursun)
- Depth Recurrence: PR #1204 (msisovic), PR #1331 (dexhunter)
- SP8192 Stack: PR #1394 (clarkkev)
