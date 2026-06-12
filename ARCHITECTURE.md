# APEX Rust Architecture

### A Rust/Candle Decoder + Vision-Language Model Design

---

## Table Of Contents

1. [Overview](#1-overview)
2. [Tokenizer](#2-tokenizer)
3. [Configuration And Dimensions](#3-configuration-and-dimensions)
4. [Embedding And Tied LM Head](#4-embedding-and-tied-lm-head)
5. [Vision Input Pipeline](#5-vision-input-pipeline)
6. [RoPE And YaRN](#6-rope-and-yarn)
7. [Attention Strategy](#7-attention-strategy)
8. [Transformer Block](#8-transformer-block)
9. [FFN, MoE, Load Balancing, And Skip Gate](#9-ffn-moe-load-balancing-and-skip-gate)
10. [PEFT Adapters](#10-peft-adapters)
11. [Generation](#11-generation)
12. [Training And Alignment Helpers](#12-training-and-alignment-helpers)
13. [Checkpoints](#13-checkpoints)
14. [Module Layout](#14-module-layout)
15. [Runtime Boundaries](#15-runtime-boundaries)

---

## 1. Overview

APEX Rust is a decoder-only language model implementation with optional
vision-language input. Text is the primary sequence. Images enter as continuous
visual tokens inserted into the same context stream at `<|img|>`.

Design goals:

```txt
1. CPU-FIRST        — tiny presets and validation run locally.
2. MODULAR         — each model component lives in its own Rust module.
3. EXPLICIT SHAPES — tensor dimensions are checked at module boundaries.
4. NATIVE ARTIFACTS — safetensors + JSON/YAML sidecars.
5. EXTENSIBLE      — adapters, vision, and generation are separate layers.
```

Core capabilities:

| Area | Design |
|---|---|
| Tokenization | Hugging Face tokenizer JSON or fallback tokenizer |
| Position | RoPE with YaRN-style frequency scaling |
| Attention | Interleaved local GQA and global MLA |
| FFN | Dense SwiGLU or sparse MoE |
| Efficiency | GQA, sliding windows, MLA compression, skip gate, PEFT |
| Vision | Native patch encoder + projector + visual-token insertion |
| Adaptation | LoRA, QLoRA, DoRA, QDoRA |
| Artifacts | `model.safetensors`, optional `adapter.safetensors`, metadata |

---

## 2. Tokenizer

The tokenizer layer supports two modes:

1. Load an existing `tokenizer.json` through the `tokenizers` crate.
2. Use the fallback tokenizer for local smoke tests.

Special tokens:

```txt
<|pad|>             padding
<|begin_of_text|>   beginning of sequence
<|end_of_text|>     end of sequence
<|system|>          system message boundary
<|user|>            user message boundary
<|assistant|>       assistant message boundary
<|thinking|>        thinking span start
<|/thinking|>       thinking span end
<|img|>             image placeholder
```

Chat formatting produces a single text stream:

```txt
<|begin_of_text|><|user|>
Describe this image: <|img|>
<|assistant|>
...
```

SFT token-type masking marks assistant and thinking tokens as trainable targets
while prompt tokens are ignored in assistant-only loss.

---

## 3. Configuration And Dimensions

`ApexConfig` is the top-level runtime configuration. It is serde-backed and can
be loaded from YAML.

Preset YAML files live in `configs/`:

| File | Role |
|---|---|
| `tiny.yaml` | Minimal CPU-friendly text model |
| `small.yaml` | Small text model |
| `medium.yaml` | Medium text model |
| `large.yaml` | Large text model |
| `tiny_vision.yaml` | Minimal CPU-friendly vision-language model |
| `small_vision.yaml` | Small vision-language model |
| `medium_vision.yaml` | Medium vision-language model |
| `large_vision.yaml` | Large vision-language model |
| `tiny_{lora,qlora,dora,qdora}.yaml` | Adapter training models |
| `tiny_{lora,qlora,dora,qdora}_dpo.yaml` | Adapter-DPO alignment models |
| `tiny_{lora,qlora,dora,qdora}_inference.yaml` | Adapter inference and merge models |

Major sections:

| Section | Responsibility |
|---|---|
| `model` | Hidden size, depth, heads, vocab, RoPE, FFN dimensions |
| `attention` | Global layer cadence, local window, flash flag |
| `moe` | Expert counts, active routes, shared experts, balancer alpha |
| `skip_gate` | Conditional FFN compute gate |
| `multi_token_head` | Auxiliary speculative heads |
| `thinking` | Thinking-mode generation limits |
| `generation` | Inference sampling and speculative decoding defaults |
| `vision` | Image size, patch size, projector, visual-token count |
| `peft` | LoRA/QLoRA/DoRA/QDoRA settings |
| `training` | Batch, sequence length, LR schedule, optimizer fields |
| `grpo` | Group-relative policy optimization helpers |
| `adapter_dpo` | Preference-alignment settings for adapter workflows |

Important validation rules:

- `d_model == n_heads_q * d_head`
- `n_heads_q` must be divisible by `n_heads_kv`
- `n_layers` must align with `global_layer_freq`
- vision image size must be divisible by patch size
- PEFT rank and alpha must be positive
- adapter-DPO requires PEFT to be enabled
- generation token budget must be positive
- generation temperatures must be non-negative and `top_p` must be in `[0, 1]`

---

## 4. Embedding And Tied LM Head

Input token IDs are mapped through a learned embedding table:

```txt
token_ids [B, S]
  → embedding [B, S, d_model]
  → embedding * sqrt(d_model)
```

The output LM head is tied to the same embedding matrix:

```txt
hidden [B, S, d_model] × embedding.T [d_model, vocab]
  → logits [B, S, vocab]
```

Weight tying reduces parameters and keeps input/output token geometry aligned.

---

## 5. Vision Input Pipeline

Vision mode converts an image into a fixed number of continuous visual tokens.

```txt
Image path, HWC buffer, or CHW buffer
        │
        ▼
ImagePreprocessor
        │ convert to CHW, expand grayscale/RGB, resize, normalize
        ▼
Image tensor [B, C, H, W]
        │
        ▼
NativeVisionEncoder
        │ patch tokens [B, patches + cls, d_vision]
        ▼
VisionToTextProjector
        │ visual tokens [B, n_visual_tokens, d_model]
        ▼
Replace <|img|> token embedding
        │
        ▼
Decoder context
```

The image placeholder replacement changes sequence length:

```txt
final_seq_len = text_seq_len - img_placeholders
              + img_placeholders * n_visual_tokens
```

Text-only forward passes bypass the vision wrapper completely.

The preprocessing stage accepts CHW `ImageTensor` values directly and can
convert HWC buffers into CHW layout. It clamps byte-scaled values into `[0, 1]`,
supports grayscale-to-RGB and RGB-to-grayscale conversion when the config
requires it, resizes with CPU bilinear sampling, and normalizes with CLIP-style
channel statistics. File loading is behind the optional `image` feature and
uses the `image` crate to decode PNG/JPEG inputs before the same preprocessing
pipeline.

---

## 6. RoPE And YaRN

RoPE applies position by rotating query and key vector pairs:

```txt
x_even' = x_even * cos(pos * theta) - x_odd  * sin(pos * theta)
x_odd'  = x_even * sin(pos * theta) + x_odd  * cos(pos * theta)
```

APEX Rust precomputes cosine/sine caches for:

- standard attention head width
- MLA decoupled RoPE head width

YaRN-style scaling adjusts frequency bands when `rope_scaling > 1.0`, allowing
longer configured context windows while keeping local high-frequency behavior
stable.

---

## 7. Attention Strategy

Layers alternate between local and global attention:

```txt
if layer_idx % global_layer_freq == global_layer_freq - 1:
    use MLA global attention
else:
    use GQA sliding-window attention
```

### MLA

Multi-Head Latent Attention compresses KV information:

```txt
x → W_DKV → c_kv
c_kv → W_UK → K
c_kv → W_UV → V
x → W_DQ → c_q → W_UQ → Q
x → W_QR / W_KR → decoupled RoPE Q/K
```

The cache stores compressed KV plus RoPE keys.

### GQA Sliding Window

Grouped Query Attention uses fewer KV heads than query heads:

```txt
groups = n_heads_q / n_heads_kv
K,V are repeated across query groups
```

The KV cache is clipped to `attention.local_window`.

### Masks

APEX masks combine:

- prefix-bidirectional attention over the prefix segment
- causal attention for generation tokens
- sliding-window locality on local GQA layers
- full causal visibility on global MLA layers

---

## 8. Transformer Block

One decoder block:

```txt
x
│
├─ RMSNorm
├─ Attention: MLA or GQA
├─ Residual add
│
├─ RMSNorm
├─ FFN or MoE
├─ Optional skip gate mask
├─ Residual add
│
▼
output
```

All blocks are pre-norm. The skip gate controls FFN execution per token with a
thresholded sigmoid gate.

---

## 9. FFN, MoE, Load Balancing, And Skip Gate

### Dense SwiGLU FFN

```txt
gate = silu(x W_gate)
up   = x W_up
out  = (gate * up) W_down
```

### MoE FFN

MoE combines:

- shared experts that always run
- routed experts selected by top-k router logits
- normalized routing weights

### Load Balancer

The load balancer tracks expert usage and adjusts per-expert router bias toward
balanced utilization.

### Skip Gate

The skip gate predicts whether FFN compute should run for a token:

```txt
gate = sigmoid(fc2(silu(fc1(x))))
run = gate >= threshold
```

---

## 10. PEFT Adapters

Linear projections can be plain or adapter-wrapped.

Supported methods:

| Method | Base Weight | Trainable Additions |
|---|---|---|
| LoRA | F32 | low-rank A/B matrices |
| QLoRA | 4-bit quantized | low-rank A/B matrices |
| DoRA | F32 | low-rank direction + magnitude vector |
| QDoRA | 4-bit quantized | direction update + magnitude vector |

LoRA update:

```txt
W' = W + B A * (alpha / r)
```

DoRA update:

```txt
W' = magnitude * normalize(W + B A * scale)
```

Adapters can be counted, exported, merged, and unloaded.

---

## 11. Generation

Generation uses a prefill step followed by one-token decode steps with KV cache
reuse.

Sampling controls:

- temperature
- top-k
- top-p
- repetition penalty
- EOS handling
- thinking-mode temperature/budget controls
- multi-token-head speculative decoding

The generator returns token IDs plus completion metadata:

```txt
token_ids
thinking_tokens
total_tokens
finished
```

Speculative decoding is implemented in `ApexGenerator::generate_with_speculative`.
The main model performs the prefill step, auxiliary multi-token heads greedily
draft future tokens from the last hidden state, and the main model verifies
`[main_token, draft_tokens...]` in one decode pass. Accepted draft tokens extend
the context directly. Rejected drafts are replaced by a sampled verifier token,
then the KV cache is rebuilt from the accepted context so later decode steps do
not carry rejected tokens.

Thinking-mode generation uses the standard one-token decode loop because the
thinking budget can force replacement tokens that need strict per-token control.

---

## 12. Training And Alignment Helpers

Training utilities include:

- JSONL readers for pretrain, SFT, preference, and vision rows
- vector-backed batch builders for pretrain, SFT, and preference samples
- streaming pretrain iterator that tokenizes text files line by line
- vision text collation with prompt-label masking
- pretrain next-token cross entropy
- SFT assistant-token-only loss
- vision SFT label expansion and loss
- cosine warmup schedule
- vector-backed AdamW update math and moment state
- global gradient norm calculation and clipping
- gradient accumulation and best-validation-loss state helpers
- checkpoint metadata
- single-process CPU smoke runners

Pretrain batches carry `input_ids` and `attention_mask`. The streaming iterator
emits padded final chunks only when at least half a sequence remains, and masks
padding tokens as zero so loss code can ignore them. Preference batches pad
chosen and rejected sequences separately and keep prompt lengths for response
log-probability slicing. Vision collation formats text around `<|img|>` and
creates labels with prompt positions set to `IGNORE_INDEX`; pixel loading stays
inside `vision::ImagePreprocessor`.

The optimizer utilities operate on flat `Vec<f32>` parameter and gradient
buffers. This keeps AdamW, weight decay, clipping, and accumulation semantics
testable on CPU while leaving full Candle autograd wiring as a future
training-loop layer.

Alignment utilities include:

- sequence log probability over response tokens
- DPO loss
- adapter-DPO step metrics
- GRPO-style normalized advantages
- clipped policy loss helper
- scalar reward model with last-non-padded hidden-state scoring
- Bradley-Terry reward-model loss
- process reward model that scores cumulative reasoning steps
- constitutional critique, scoring, revision, and DPO pair generation
- weighted combined reward from outcome, process, and constitutional signals

The alignment module is split into focused files:

```txt
alignment/dpo.rs              sequence log-probs, DPO, adapter-DPO metrics
alignment/grpo.rs             normalized advantages and clipped policy loss
alignment/reward.rs           scalar reward head and Bradley-Terry loss
alignment/prm.rs              process reward model step scoring
alignment/constitutional.rs   principle critique and revision wrapper
alignment/combined_reward.rs  outcome/process/constitutional reward mix
```

Reward models use the transformer as a hidden-state feature extractor. The
sequence reward reads the final non-padded hidden state, while PRM appends each
reasoning step to the prompt context and scores the last hidden state through a
sigmoid head. Constitutional critique is generator-agnostic through a small
`TextGenerator` trait; an empty generator gives the same safe no-op fallback used
for untrained/local validation paths.

---

## 13. Checkpoints

APEX Rust writes native artifacts:

```txt
model.safetensors       model tensors
adapter.safetensors     adapter tensors when PEFT is enabled
metadata.json           step, epoch, loss, version, notes
config.yaml             runtime configuration
```

Named tensor export walks the model graph. Quantized base weights are exported
as dequantized F32 tensors, while adapter tensors are kept in a separate adapter
payload when applicable.

The same named tensors can be loaded back into an existing model instance:

```txt
config.yaml + model.safetensors
  → ApexModel::new(config)
  → load_model_safetensors(...)

config.yaml + adapter.safetensors
  → ApexModel::new(adapter-enabled config)
  → load_adapter_safetensors(...)
```

Loading is strict for CLI commands. Missing tensors, unexpected tensors, or
shape mismatches fail early. QLoRA and QDoRA base weights are loaded from F32
payloads and re-quantized into the configured 4-bit representation in memory.

CLI commands that can consume saved tensors:

```txt
infer          --weights <model.safetensors> --adapter <adapter.safetensors>
inspect        --weights <model.safetensors> --adapter <adapter.safetensors>
benchmark      --weights <model.safetensors> --adapter <adapter.safetensors>
merge-adapter  --weights <model.safetensors> --adapter <adapter.safetensors>
```

---

## 14. Module Layout

```txt
src/model/
  apex_model.rs
  attention.rs
  block.rs
  ffn.rs
  linear.rs
  loading.rs
  load_balancer.rs
  mask.rs
  multi_token_head.rs
  norm.rs
  rope.rs
  skip_gate.rs

src/vision/
  preprocess.rs
  encoder.rs
  projector.rs
  wrapper.rs

src/train/
  losses.rs
  optimizer.rs
  scheduler.rs
  checkpoint.rs
  runner.rs

src/tokenizer/
  constants.rs
  core.rs
  trainer.rs

src/data/
  types.rs
  readers.rs
  batching.rs

src/utils/
  params.rs
  inspection.rs
  flops.rs
  diagram.rs
  shape.rs
```

Folder `mod.rs` files are intentionally small and only declare/re-export the
focused implementation files.

---

## 15. Runtime Boundaries

- CPU execution is the validated baseline.
- CUDA is not enabled by default.
- Training commands run data loading, forward/loss paths, metadata writing, and
  safetensors export.
- Checkpoints are native safetensors plus JSON/YAML sidecars and can be loaded
  by inference, inspection, benchmark, and adapter-merge commands.
- Numeric outputs are determined by Rust initialization and Candle operations.
