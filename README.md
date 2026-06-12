<div align="center">

# APEX Rust

### A Rust/Candle Language + Vision Model Toolkit

[![Rust](https://img.shields.io/badge/Rust-2021-orange.svg)](https://www.rust-lang.org/)
[![Candle](https://img.shields.io/badge/Candle-0.10.2-blue.svg)](https://github.com/huggingface/candle)
[![License](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-green.svg)](#license)
[![Status](https://img.shields.io/badge/Status-CPU%20Validated-brightgreen.svg)]()

APEX Rust is a standalone transformer implementation for text and vision-language
experiments. It combines a modern decoder stack, PEFT adapters, tokenizer/data
tools, generation, inspection, benchmarking, and native safetensors checkpoints
in one Rust crate.

</div>

---

## What Is APEX Rust?

APEX Rust is a CPU-first model runtime and experimentation crate built around
Hugging Face Candle. It includes:

- A decoder-only transformer with RMSNorm, RoPE/YaRN, MLA, GQA sliding-window
  attention, SwiGLU FFNs, MoE, load balancing, skip gates, tied LM head, and
  optional multi-token prediction heads.
- A vision-language path where images are encoded into continuous visual tokens
  and inserted at `<|img|>` inside the text context.
- PEFT adapters for LoRA, QLoRA, DoRA, and QDoRA.
- Tokenizer loading/training, JSONL data readers, loss functions, scheduler,
  generation, eval helpers, inspection, benchmark commands, and safetensors
  export.

APEX Rust does not ship trained production weights. Model quality depends on
training data, compute, and checkpoint quality. The crate focuses on a complete,
inspectable implementation that builds and runs locally on CPU.

---

## Capability Status

| Area | Status |
|---|---|
| YAML config schema and validation | Complete |
| Text, vision, PEFT, DPO, and inference YAML presets | Complete |
| Tokenizer JSON loading and BPE training | Complete |
| Pretrain, SFT, preference, and vision JSONL readers | Complete |
| Batch builders, streaming pretrain iterator, vision text collator | Complete |
| RMSNorm, RoPE/YaRN, masks, attention, FFN, MoE, skip gate | Complete |
| MLA and GQA sliding-window attention | Complete |
| KV-cache generation | Complete |
| Top-k, top-p, temperature, repetition penalty | Complete |
| Thinking-mode generation controls | Complete |
| Multi-token-head speculative decoding | Complete |
| LoRA, QLoRA, DoRA, QDoRA wrappers | Complete |
| 4-bit pack/unpack and quantize/dequantize helpers | Complete |
| Adapter merge/unload | Complete |
| Vision preprocessing, native patch encoder, and projector | Complete |
| Safetensors model and adapter save/load | Complete |
| Reward model, PRM, constitutional AI, combined reward | Complete |
| Inspection, FLOPs, parameter, diagram, shape-check utilities | Complete |
| AdamW math, gradient clipping, training-state helpers | Complete |
| Inspect, infer, benchmark, train, merge, tokenizer CLI | Complete |
| CUDA backend | Not enabled by default |

---

## Architecture At A Glance

```txt
Text token IDs [B, S]
        │
        ▼
Embedding × sqrt(d_model)
        │
        ├─────────────────────────────────────────────┐
        │ Optional image path                         │
        │                                             ▼
        │      Image [B, C, H, W] → Vision Encoder → Projector
        │                                             │
        └──────────── replace <|img|> with visual tokens
        │
        ▼
Transformer blocks × N
        │
        ├─ Local layers: GQA + sliding window
        ├─ Global layers: MLA
        ├─ Dense FFN or MoE FFN
        └─ Optional skip gate
        │
        ▼
Final RMSNorm
        │
        ▼
Tied LM head
        │
        ▼
Logits [B, S, vocab_size]
```

Core dimensions are controlled by `ApexConfig`:

| Field | Purpose |
|---|---|
| `model.d_model` | Hidden size |
| `model.n_layers` | Decoder depth |
| `model.n_heads_q` / `model.n_heads_kv` | Query and KV heads |
| `model.d_head` | Per-head attention width |
| `model.d_kv_compressed` / `model.d_q_compressed` | MLA latent widths |
| `model.d_ffn` | FFN intermediate width |
| `model.vocab_size` | Token vocabulary size |
| `attention.global_layer_freq` | MLA layer cadence |
| `attention.local_window` | GQA sliding-window size |
| `moe.n_experts` / `moe.n_active` / `moe.n_shared` | Expert routing |
| `vision.n_visual_tokens` | Number of tokens inserted for each image |

See [ARCHITECTURE.md](ARCHITECTURE.md) for the detailed model design.

---

## Project Layout

```txt
src/
  main.rs              CLI entry point
  lib.rs               Public crate exports
  config/              YAML config schema and presets
  data/                JSONL readers, batch builders, streaming pretrain
  model/               Transformer, attention, FFN, MoE, PEFT, checkpoints
  vision/              Image preprocessing, encoder, projector, wrapper
  tokenizer/           Tokenizer runtime, special tokens, BPE trainer
  generation/          Sampling and autoregressive generation
  train/               Losses, scheduler, checkpoint export, smoke runners
  alignment/           DPO, GRPO, reward model, PRM, constitutional AI
  eval/                Metrics, perplexity, benchmark helpers
  tensor/              Candle tensor utilities
  utils/               Inspection, FLOPs, parameters, diagrams, shape checks
configs/               YAML presets for text and vision model sizes
examples/              Small runnable examples
tests/                 Core and CLI tests
```

---

## Vision Preprocessing

`ImagePreprocessor` accepts CHW image buffers directly, converts HWC input into
CHW layout, expands grayscale to RGB when the config expects three channels,
clamps byte-scaled values into `[0, 1]`, resizes to `vision.image_size`, and
normalizes with CLIP-style channel statistics.

With the optional `image` feature enabled, it can also load PNG/JPEG files:

```bash
cargo test --all-features vision_preprocessor_loads_image_file
```

The model wrapper still expects a Candle tensor with shape `[B, C, H, W]`; use
`ImagePreprocessor::preprocess_to_tensor` or `batch_to_tensor` to build it.

---

## Quick Start

Build and inspect the default tiny model:

```bash
cargo run -- inspect
cargo run -- inspect --format json
```

Run random-token inference:

```bash
cargo run -- infer --random --max-new-tokens 8 --temperature 0
```

Run speculative decoding when the config enables multi-token heads:

```bash
cargo run -- infer --random --max-new-tokens 8 --temperature 0 --speculative
```

Run inference from saved artifacts:

```bash
cargo run -- infer \
  --config checkpoints/apex-rust/config.yaml \
  --weights checkpoints/apex-rust/model.safetensors \
  --prompt "Hello"
```

Benchmark a forward pass:

```bash
cargo run -- benchmark --seq-len 16 --repeats 3
```

Run a training smoke step:

```bash
cargo run -- train pretrain --dry-run --steps 1
cargo run -- train sft --data data/sft.jsonl --dry-run
cargo run -- train peft-sft --dry-run
cargo run -- train adapter-dpo --data data/preference.jsonl --dry-run
```

Train a tokenizer:

```bash
cargo run -- tokenizer train \
  --input data/text.jsonl \
  --output target/apex-tokenizer.json \
  --vocab-size 256
```

Use a custom config:

```bash
cargo run -- inspect --config configs/tiny.yaml
```

Built-in YAML presets:

| Text preset | Vision preset | Purpose |
|---|---|---|
| `configs/tiny.yaml` | `configs/tiny_vision.yaml` | Fast CPU checks and examples |
| `configs/small.yaml` | `configs/small_vision.yaml` | Small experiments |
| `configs/medium.yaml` | `configs/medium_vision.yaml` | Mid-size training runs |
| `configs/large.yaml` | `configs/large_vision.yaml` | Large model configuration |

Adapter presets:

```txt
configs/tiny_lora.yaml              configs/tiny_lora_dpo.yaml
configs/tiny_qlora.yaml             configs/tiny_qlora_dpo.yaml
configs/tiny_dora.yaml              configs/tiny_dora_dpo.yaml
configs/tiny_qdora.yaml             configs/tiny_qdora_dpo.yaml
configs/tiny_lora_inference.yaml    configs/tiny_qlora_inference.yaml
configs/tiny_dora_inference.yaml    configs/tiny_qdora_inference.yaml
```

Inference presets include a `generation` section for defaults such as
`max_new_tokens`, `temperature`, `top_p`, repetition penalty, thinking
temperatures, and speculative decoding.

---

## Examples

Runnable examples live in `examples/`:

```bash
cargo run --example forward_pass
cargo run --example generate
cargo run --example inspect_model
cargo run --example tokenizer_chat
cargo run --example peft_summary
cargo run --example vision_forward
```

Each example uses tiny CPU-friendly settings and is intended to show one
subsystem clearly.

---

## CLI Reference

```txt
apex-rust train pretrain      --config <yaml> --data <jsonl> --output-dir <dir>
apex-rust train sft           --config <yaml> --data <jsonl> --output-dir <dir>
apex-rust train peft-sft      --config <yaml> --data <jsonl> --output-dir <dir>
apex-rust train adapter-dpo   --config <yaml> --data <jsonl> --output-dir <dir>
apex-rust infer               --config <yaml> --weights <model.safetensors> --adapter <adapter.safetensors> --prompt <text> --speculative
apex-rust inspect             --config <yaml> --weights <model.safetensors> --adapter <adapter.safetensors> --format text|json
apex-rust benchmark           --config <yaml> --weights <model.safetensors> --seq-len <n> --repeats <n>
apex-rust merge-adapter       --config <yaml> --weights <model.safetensors> --adapter <adapter.safetensors> --output-dir <dir>
apex-rust tokenizer train     --input <text-file> --output <tokenizer.json>
```

With no config path, commands use tiny defaults intended for CPU validation.

---

## Data Schemas

Pretraining JSONL:

```json
{"text":"Rust systems programming with transformers."}
```

SFT JSONL:

```json
{"messages":[{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi"}]}
```

Preference JSONL:

```json
{"prompt":"Explain RMSNorm.","chosen":"RMSNorm normalizes by root mean square.","rejected":"It is dropout."}
```

Vision JSONL:

```json
{"image":"image.png","prompt":"<|img|> What is shown?","response":"A chart."}
```

The data API also exposes batch builders for pretrain, SFT, and preference
samples, a lazy streaming pretrain iterator over text files, padding helpers,
and a vision text collator that creates token labels while leaving image
decoding to the vision preprocessing layer.

---

## Checkpoints

Training and adapter commands write native artifacts:

```txt
output/
  model.safetensors
  adapter.safetensors      # when PEFT is enabled
  metadata.json
  config.yaml
  README.md
```

`model.safetensors` stores named F32 tensors. Adapter checkpoints store LoRA,
DoRA, QLoRA, or QDoRA adapter tensors separately when applicable.

Load paths are strict by default: tensor names and shapes must match the model
created from `--config`. QLoRA/QDoRA base weights are saved as dequantized F32
and re-quantized into the configured 4-bit representation when loaded.

The training support layer includes cosine warmup scheduling, loss functions,
checkpoint writing, vector-backed AdamW update math, global gradient clipping,
and gradient-accumulation state helpers. The current CLI training modes remain
CPU smoke loops around forward/loss/checkpoint paths; full Candle autograd
updates over every model tensor are a future integration layer.

---

## Validation

```bash
cargo fmt --check
cargo check
cargo clippy --all-targets --all-features
cargo test
```

The test suite covers config roundtrip, tokenizer/chat behavior, data readers,
RoPE/YaRN, masks, RMSNorm, load balancer, full model forward/losses, PEFT
insertion and merge, 4-bit quantization, generation sampling, data batching,
streaming masks, vision-token insertion, safetensors save/load, DPO/GRPO
helpers, reward/PRM scoring, constitutional critique, combined rewards, utility
reports, shape verification, and CLI smoke tests.

---

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
