# Apple Silicon Acceleration Plan

## Background

This project runs Qwen3 models using PagedAttention, prefix caching, and SDPA.
The CPU backend was implemented first for correctness. This document covers the
path to faster inference on Apple Silicon.

## Performance Landscape

| Backend | Throughput | Notes |
|---|---|---|
| CPU (ARM) | ~1–2 tok/s | Baseline; pure PyTorch, no GPU |
| PyTorch MPS | ~7–9 tok/s | Apple GPU via Metal; in this codebase |
| Core ML / ANE | ~33 tok/s | Neural Engine; **not suitable for LLMs** |
| Apple MLX | 230+ tok/s | Apple's own ML framework; full rewrite |

**Important**: The Apple Neural Engine (ANE/NPU) is optimized for convolutions and
vision models. Transformers run faster on the Apple GPU. Core ML on ANE is
counter-productively slow for decoder-only LLMs.

## Phase 1: PyTorch MPS (implemented)

Uses Metal Performance Shaders via `torch.backends.mps`. Incremental improvement
within the existing codebase — same PagedAttention, same SDPA, same scheduler.

### Changes made
- `nanovllm/config.py`: Added `device: str = "auto"` field. Auto-detects MPS on
  Apple Silicon, falls back to CPU otherwise. Can be overridden explicitly.
- `nanovllm/engine/model_runner.py`: Uses `config.device` instead of hardcoded
  `"cpu"`. All tensors (model weights, KV cache) land on the selected device.
- `nanovllm/layers/attention.py`: Float32 cast in SDPA is now conditional — only
  applied for bfloat16 on CPU (where it is required). MPS handles bfloat16 natively.

### Usage
```python
from nanovllm import LLM

# Default: auto-selects MPS on Apple Silicon
llm = LLM("~/huggingface/Qwen3-0.6B/")

# Explicit device selection
llm = LLM("~/huggingface/Qwen3-0.6B/", device="mps")   # force GPU
llm = LLM("~/huggingface/Qwen3-0.6B/", device="cpu")   # force CPU
```

### Known MPS limitations
- `@torch.compile` may skip compilation on some PyTorch versions (falls back to eager)
- Very long sequences (>32k tokens) may OOM on machines with limited unified memory
- MPS does not support all PyTorch ops; unsupported ops fallback to CPU automatically

### Memory sizing
Apple Silicon uses Unified Memory (CPU and GPU share the same RAM pool). The
`psutil.virtual_memory().available` measurement is correct for MPS — no separate
GPU VRAM to track. The default `memory_utilization=0.75` applies to this shared pool.

---

## Phase 2: Apple MLX (future)

[Apple MLX](https://github.com/ml-explore/mlx) achieves 230+ tok/s on Apple Silicon.
The `mlx-lm` package supports Qwen3 out of the box:

```bash
pip install mlx-lm
mlx_lm.generate --model ~/huggingface/Qwen3-0.6B/ --prompt "Hello, world"
```

MLX does NOT integrate with this codebase — it is a separate inference engine.
Adopting MLX would mean replacing `nanovllm` with an MLX-based implementation,
which is a full rewrite. The current codebase would serve as a reference for
scheduler, block manager, and prefix caching logic.

Key MLX differences:
- Lazy evaluation model (like JAX) — computation happens at `.eval()` call
- No PagedAttention equivalent yet; KV cache is a simple tensor
- Prefix caching would need to be reimplemented

---

## Decision Guide

| Goal | Recommendation |
|---|---|
| Correctness reference, CPU only | Current CPU backend |
| ~5–10x speedup, minimal change | Phase 1 (MPS) — done |
| Maximum throughput, rewrite OK | Phase 2 (MLX) |
| iOS / macOS app deployment | Core ML export (separate toolchain) |
