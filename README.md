# AutoParallel

English | [中文](README_zh.md)

Auto-parallelism strategy advisor for LLM training and inference.

Given a model architecture and GPU cluster, AutoParallel enumerates all valid parallel strategies (DP, PP, TP, CP, EP), estimates per-GPU memory consumption, models throughput with a cost model, and recommends optimal configurations — no OOM, maximum efficiency.

<p align="center">
  <img src="docs/images/architecture.png" alt="AutoParallel System Architecture" width="800"/>
</p>

## Features

- **Training + Inference**: separate strategy search with engine-specific cost models
- **Memory estimation**: model, gradients, optimizer, activations, KV cache — per-GPU and per-node CPU
- **Throughput modeling**: roofline compute + alpha-beta communication cost model
- **Hardware profiling**: optional GEMM + collective communication profiling for higher accuracy
- **Wide model support**: Dense, MoE, MLA, GQA, Lightning Attention (see [Supported Models](#supported-models))
- **GPU presets**: H200, H100, H800, A100, A800 with auto-detected specs
- **Engine-aware**: Megatron, FSDP, SGLang engine-specific optimizations (see [Supported Engines](#supported-engines))

## Installation

```bash
pip install autoparallel
```

Or from source:

```bash
git clone https://github.com/dingzhiqiang/AutoParallel.git
cd AutoParallel
pip install -e .
```

## Quick Start

AutoParallel has two modes: **training** (default) and **inference**. The tool reads the model's HuggingFace `config.json` to auto-detect architecture (Dense/MoE/MLA/GQA), or you can specify model parameters manually.

### Training (default mode)

Find the optimal training parallelism for a model on your cluster. The default engine is **Megatron** — the cost model accounts for Megatron-specific optimizations (EP AllToAll overlap, TP bandwidth degradation, etc.):

```bash
# Auto-detect model from HuggingFace config.json
# Default: --mode training --engine megatron
python -m autoparallel \
    --model-path /path/to/model --n-gpus 128 --gpu-type H200

# Use FSDP engine cost model instead of Megatron
python -m autoparallel \
    --model-path /path/to/model --n-gpus 64 --gpu-type A100 \
    --engine fsdp

# Manual model parameters (e.g., DeepSeek-V3 style MoE + MLA)
python -m autoparallel \
    --hidden-size 6144 --num-layers 78 --num-heads 64 \
    --num-experts 256 --expert-intermediate-size 2048 \
    --intermediate-size 12288 --vocab-size 154880 \
    --kv-lora-rank 512 --q-lora-rank 2048 \
    --n-gpus 128 --gpu-type H200
```

### Inference

Find the optimal inference parallelism. The default engine switches to **SGLang**:

```bash
# Default: --engine sglang
python -m autoparallel --mode inference \
    --model-path /path/to/model --n-gpus 64 --gpu-type H200 \
    --isl 4096 --osl 512 --infer-batch-size 32
```

### Hardware Profiling (optional)

Run hardware profiling for higher ranking accuracy (~90% → ~98%). Data is cached in `~/.cache/autoparallel/` and auto-loaded on subsequent runs:

```bash
# Auto-detect backend (Slurm/Ray/Local), single node
python -m autoparallel profile

# Two nodes via Slurm (measures both NVLink + IB)
python -m autoparallel profile --n-nodes 2 --backend slurm
```

## Supported Models

AutoParallel reads the model's HuggingFace `config.json` and auto-detects the architecture. Any model using the standard HuggingFace config fields is supported. Tested models include:

| Architecture | Representative Models | Key Features Detected |
| --- | --- | --- |
| **Dense Transformer** | LLaMA, LLaMA-2/3, GPT, Qwen-2/2.5 | `hidden_size`, `num_attention_heads`, `intermediate_size` |
| **GQA** | LLaMA-2/3, Qwen-2/2.5, GLM-4 | `num_key_value_heads` < `num_attention_heads` |
| **MoE** | DeepSeek-V2/V3, Qwen-MoE, Ling-MoE | `n_routed_experts`, `num_experts_per_tok`, `moe_intermediate_size` |
| **MLA** | DeepSeek-V2/V3, GLM-5.1 | `kv_lora_rank`, `q_lora_rank`, `qk_nope_head_dim` |
| **MoE + MLA** | DeepSeek-V3, GLM-5.1 | Both MoE and MLA fields present |
| **Lightning Attention** | GLM-5.1 | `group_norm_size` / `layer_group_size` |
| **first-k dense replace** | DeepSeek-V3, GLM-5.1 | `first_k_dense_replace` (first N layers use dense FFN) |

> **Note**: Any HuggingFace-format model with a `config.json` can be used — AutoParallel doesn't load model weights, only the architecture config.

## Supported Engines

The `--engine` flag selects the cost model preset. Different engines have different runtime behaviors that affect parallelism efficiency:

| Engine | Use Case | Default For | Key Cost Model Assumptions |
| --- | --- | --- | --- |
| **megatron** | Training | `--mode training` | EP AllToAll overlaps with FFN compute; TP BW degrades for TP>4; MoE activation factor=18 |
| **fsdp** | Training | — | EP AllToAll serialized with FFN; no TP BW degradation; MoE activation factor=14 |
| **sglang** | Inference | `--mode inference` | EP AllToAll overlaps with FFN; no TP BW degradation; MoE activation factor=14 |

**Why engine matters for training**: The Megatron engine overlaps EP AllToAll dispatch/combine with expert FFN computation (`cost = max(ep_time, ffn_time)`), while FSDP runs them sequentially (`cost = ep_time + ffn_time`). This significantly changes which TP/EP combination is optimal for MoE models.

```bash
# Training with Megatron cost model (default)
python -m autoparallel --model-path /path/to/model --n-gpus 128

# Training with FSDP cost model
python -m autoparallel --model-path /path/to/model --n-gpus 128 --engine fsdp

# Inference with SGLang cost model (default for inference)
python -m autoparallel --mode inference --model-path /path/to/model --n-gpus 64
```

## CLI Reference

### Mode

| Parameter | Default | Description |
| --- | --- | --- |
| `--mode {training,inference}` | `training` | Training or inference mode |

### Model (auto-detect)

| Parameter | Description |
| --- | --- |
| `--model-path PATH` | HuggingFace model directory (with config.json) |

### Model (manual)

| Parameter | Default | Description |
| --- | --- | --- |
| `--hidden-size` | 4096 | Hidden dimension |
| `--num-layers` | 32 | Number of transformer layers |
| `--num-heads` | 32 | Number of attention heads |
| `--num-kv-heads` | = num-heads | KV heads (GQA) |
| `--intermediate-size` | 11008 | FFN intermediate dimension |
| `--vocab-size` | 32000 | Vocabulary size |
| `--num-experts` | 0 | Number of MoE experts |
| `--num-experts-per-tok` | 0 | Experts activated per token |
| `--expert-intermediate-size` | 0 | Expert FFN intermediate dimension |
| `--n-shared-experts` | 0 | Number of shared experts |
| `--kv-lora-rank` | 0 | MLA KV LoRA rank |
| `--q-lora-rank` | 0 | MLA Q LoRA rank |
| `--group-norm-size` | 0 | Lightning Attention GroupNorm size |
| `--first-k-dense-replace` | 0 | First k layers use dense FFN |

### Cluster

| Parameter | Default | Description |
| --- | --- | --- |
| `--n-gpus` | 128 | Total number of GPUs |
| `--gpus-per-node` | 8 | GPUs per node |
| `--gpu-type` | H200 | GPU preset (H200/H100/H800/A100/A800) |
| `--gpu-memory-gb` | auto | Override GPU memory (GB, 0=use preset) |
| `--host-memory-gb` | auto | Override host CPU memory (GB) |
| `--gpu-flops` | auto | Override BF16 TFLOPS |
| `--bw-nvlink` | auto | Override NVLink bandwidth (GB/s) |
| `--bw-ib` | auto | Override IB bandwidth (GB/s) |

### Training Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `--max-tokens-per-mb` | 131072 | Max tokens per micro-batch |
| `--max-length` | 16384 | Max sequence length |
| `--batch-size` | 0 | Global batch size (0=unconstrained) |
| `--engine` | megatron | Engine preset (megatron/fsdp/sglang) |
| `--no-optimizer-cpu-offload` | false | Disable optimizer CPU offload |
| `--no-recompute` | false | Disable activation recompute |
| `--no-grad-reduce-in-fp32` | false | Use bf16 gradient accumulation |

### Inference Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `--isl` | 4096 | Input sequence length |
| `--osl` | 512 | Output sequence length |
| `--infer-batch-size` | 32 | Inference batch size |

### Profiling

| Parameter | Description |
| --- | --- |
| `--profile-data PATH` | Explicit profiling JSON path |
| `--no-profile` | Disable profiling data, use pure analytical model |

### Output

| Parameter | Description |
| --- | --- |
| `--json` | JSON format output |
| `--top N` | Show top N only (0=all) |
| `--find-min-nodes` | Search for minimum number of nodes |

## Output Example

### Training

```
============================================================================
Model: 786.3B params | MoE=True (256 experts) | MLA=True | Layers=78
max_tokens_per_mb=16384 | GPU=140GB | Host=1500GB/node | Engine=megatron
============================================================================
  #  DP  PP  TP  CP   EP  Tok/GPU  Layers Exp/R   Model  GrdBuf  ...  Fit?
----------------------------------------------------------------------------
  1   1   4  16   1   16    16384   19-20    16   24.1G   48.3G  ...   OK
  ...

Top-3 Recommended (by estimated throughput)
======================================================================
  #1  megatron:(attn:d1p8t8|ffn:d1p8e8)
      DP=1  PP=8  TP=8  CP=1  EP=8
      GPU: 95.2G / 140G (margin 45G)
      Score: 41.159 (100%)
       + TP=8 in-node NVLink
       + PP=8 bubble 10%
       + EP=8 in-node NVLink AllToAll
```

### Inference

```
Top-3 Recommended (by aggregate decode throughput)
======================================================================
  #1  tp1_ep8  (8 GPUs/instance x 8 instances)
      Memory: 33.3G / 140G | KV capacity: ~580K tokens
      Prefill: 12.3 ms (333K tok/s/inst)
      Decode:  8.1 ms/tok (123 tok/s/inst)
      Aggregate: 984 tok/s (100%)
```

## Cost Model

<p align="center">
  <img src="docs/images/cost_model.png" alt="Training Cost Model Pipeline" width="800"/>
</p>

<p align="center">
  <img src="docs/images/inference_model.png" alt="Inference Performance Model" width="800"/>
</p>

1. **ALPA alpha-beta communication model**: TP AllReduce, EP AllToAll, CP Ring
2. **Hierarchical EP AllToAll**: NVLink intra-node + IB cross-node traffic split
3. **Engine-aware optimizations**: EP overlap, TP BW degradation (Megatron-specific)
4. **Roofline inference model**: prefill compute-bound, decode memory-bandwidth-bound
5. **Profiling interpolation** (optional): measured GEMM + comm data, log-space interpolation

See [DESIGN.md](DESIGN.md) for details.

## Comparison with Other Systems

| Feature | AutoParallel | [Galvatron](https://github.com/PKU-DAIR/Hetu-Galvatron) | [Alpa](https://github.com/alpa-project/alpa) | [ColossalAI](https://github.com/hpcaitech/ColossalAI) | DeepSpeed |
| --- | --- | --- | --- | --- | --- |
| **Approach** | Pure advisor | Profiler + Search + Runtime | ILP + DP search | Config-based | ZeRO config |
| **DP/PP/TP** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **EP (Expert Parallel)** | ✓ | ✓ | ✗ | ✓ | ✓ (MoE) |
| **CP (Context Parallel)** | ✓ (MLA-aware) | ✗ | ✗ | ✗ | ✗ |
| **ZeRO stages** | ✗ (planned) | ✓ (1/2/3) | ✗ | ✓ | ✓ (1/2/3/Infinity) |
| **Sequence Parallel** | ✗ (planned) | ✓ (Megatron-SP, Ulysses) | ✗ | ✓ | ✗ |
| **MLA support** | ✓ | ✗ | ✗ | ✗ | ✗ |
| **FP8/Quantization** | ✗ (planned) | ✗ | ✗ | ✓ | ✗ |
| **Inference mode** | ✓ (Roofline) | ✗ | ✗ | ✗ | ✗ |
| **Engine-aware** | ✓ (Megatron/FSDP/SGLang) | ✗ | ✗ | ✗ | ✗ |
| **GPU dependency** | None | Profiling needed | ILP solver | Runtime | Runtime |
| **Training support** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Search algorithm** | Enumeration | Dynamic Programming | ILP | Config | Config |
| **Layer-wise strategy** | ✗ (planned) | ✓ | ✓ | ✗ | ✗ |

**AutoParallel's differentiators**:
1. Unified training + inference advisor in one tool
2. Engine-aware cost model (Megatron EP overlap, TP BW degradation)
3. MLA-aware CP costing (3.5% of standard MHA)
4. Zero GPU dependency — runs in seconds from config.json alone

## Roadmap

### Near-term

- [ ] **FP8 / Quantization support** — Model weight precision (FP8 E4M3, INT8, INT4/AWQ/GPTQ).
  FP8 halves weight memory (2→1 bytes/param) and doubles peak FLOPS on H100/H200.
  Many production models (DeepSeek-V3, Qwen3.5-FP8) ship with native FP8 weights.
  Implementation: add `--precision {bf16,fp8,int8,int4}` flag, adjust `dtype_bytes`
  and `gpu_flops` accordingly.

- [ ] **More inference engines** — Add vLLM and TensorRT-LLM engine presets.
  SGLang cookbook already covers all three; cost model differences are minor
  (memory allocation policy, chunked prefill, prefix caching).
  Implementation: add `VLLM_ENGINE` and `TRTLLM_ENGINE` configs.

- [ ] **ZeRO stages** — Model ZeRO-1 (optimizer sharding), ZeRO-2 (+gradient sharding),
  ZeRO-3 (+parameter sharding) for FSDP/DeepSpeed training.
  Changes optimizer and gradient memory formulas.

### Mid-term

- [ ] **Sequence Parallelism** — Model Megatron-SP (saves activation memory) and
  DeepSpeed-Ulysses (ring attention variant). SP interacts with TP for activation
  memory reduction.

- [ ] **Layer-wise heterogeneous parallelism** — Allow different parallelism configs per
  layer (like Galvatron). Useful for models with mixed dense/MoE layers
  (e.g., first-k dense replace).

- [ ] **Multi-stage RL workflow** — Model RLHF/GRPO resource allocation across
  rollout (inference) + training stages on a shared GPU cluster.

- [ ] **Interactive web UI** — Browser-based interface for exploring strategies,
  comparing configurations, and visualizing memory/throughput tradeoffs.

### Long-term

- [ ] **NVSwitch / heterogeneous topology** — Model NVSwitch all-to-all bandwidth
  (vs ring-based NVLink) and asymmetric interconnects.

- [ ] **Automatic profiling integration** — One-click hardware profiling with
  auto-calibration of the cost model. Currently profiling is optional Layer 2;
  make it seamless.

- [ ] **CI/CD integration** — Validate parallel configs before job submission.
  `autoparallel check --config train.yaml` to catch OOM before wasting GPU hours.

- [ ] **Multi-model serving** — Cost model for serving multiple models on a shared
  cluster (model multiplexing, memory sharing).

## GPU Presets

| GPU | Memory | Host Memory | BF16 TFLOPS | NVLink BW | IB BW |
| --- | --- | --- | --- | --- | --- |
| H200 | 141 GB | 1500 GB | 990 | 450 GB/s | 50 GB/s |
| H100 | 80 GB | 1000 GB | 990 | 450 GB/s | 50 GB/s |
| H800 | 80 GB | 1000 GB | 990 | 400 GB/s | 50 GB/s |
| A100 | 80 GB | 1000 GB | 312 | 300 GB/s | 25 GB/s |
| A800 | 80 GB | 1000 GB | 312 | 200 GB/s | 25 GB/s |

## Validation

AutoParallel's recommendations have been cross-validated against:

- **Real deployment**: GLM-5.1 on 128×H200, <5% memory error, correct ranking
- **SGLang official cookbook**: TP/EP recommendations align with SGLang's [auto-benchmark configs](https://github.com/sgl-project/sglang/tree/main/.claude/skills/llm-serving-auto-benchmark/configs/cookbook-llm)
- **Qwen model family**: 7 models from 8B to 397B tested (Dense, MoE, GQA)

See [BENCHMARK.md](BENCHMARK.md) ([中文](BENCHMARK_zh.md)) for details.

## Documentation

| Document | Description |
| --- | --- |
| [README.md](README.md) ([中文](README_zh.md)) | Quick start and CLI reference |
| [DESIGN.md](DESIGN.md) | Technical design with formulas (中文) |
| [PAPER.md](PAPER.md) ([中文](PAPER_zh.md)) | Paper-style write-up with contributions |
| [BENCHMARK.md](BENCHMARK.md) ([中文](BENCHMARK_zh.md)) | Real-world validation results |

## References

- [Alpa](https://github.com/alpa-project/alpa) — auto-parallelism for JAX
- [Galvatron](https://github.com/PKU-DAIR/Hetu-Galvatron) — automatic parallelism optimization
- XLA matmul/collective interpolation models

## License

Apache License 2.0
