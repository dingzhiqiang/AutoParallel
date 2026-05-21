# AutoParallel

English | [中文](README_zh.md)

Auto-parallelism strategy advisor for LLM training and inference.

Given a model architecture and GPU cluster, AutoParallel enumerates all valid parallel strategies (DP, PP, TP, CP, EP), estimates per-GPU memory consumption, models throughput with a cost model, and recommends optimal configurations — no OOM, maximum efficiency.

## Features

- **Training + Inference**: separate strategy search for training and inference workloads
- **Memory estimation**: model, gradients, optimizer, activations, KV cache — per-GPU and per-node CPU
- **Throughput modeling**: roofline compute + alpha-beta communication cost model
- **Hardware profiling**: optional GEMM + collective communication profiling for higher accuracy
- **Model support**: Dense, MoE, MLA, GQA, Lightning Attention, first-k dense replace
- **GPU presets**: H200, H100, H800, A100, A800 with auto-detected specs
- **Engine-aware**: Megatron and FSDP engine-specific optimizations (EP overlap, TP BW degradation)

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

```bash
# From HuggingFace config.json, auto-detect model architecture
python -m autoparallel \
    --model-path /path/to/model --n-gpus 128 --gpu-type H200

# Manual model parameters
python -m autoparallel \
    --hidden-size 6144 --num-layers 78 --num-heads 64 \
    --num-experts 256 --expert-intermediate-size 2048 \
    --intermediate-size 12288 --vocab-size 154880 \
    --kv-lora-rank 512 --q-lora-rank 2048 \
    --n-gpus 128 --gpu-type H200

# Inference mode
python -m autoparallel --mode inference \
    --model-path /path/to/model --n-gpus 64 --gpu-type H200 \
    --isl 4096 --osl 512 --infer-batch-size 32
```

## Supported Model Architectures

- **Dense Transformer**: LLaMA, GPT series
- **MoE (Mixture-of-Experts)**: DeepSeek-V2/V3 etc.
  - Shared experts, first-k dense replace
- **MLA (Multi-head Latent Attention)**: DeepSeek-V2/V3
  - Auto-detect kv_lora_rank, q_lora_rank
- **GQA (Grouped Query Attention)**: LLaMA-2/3 etc.
- **Lightning Attention + GroupNorm**: auto-handle group_norm_size constraints

## CLI Parameters

### Mode

| Parameter                     | Default    | Description             |
| ----------------------------- | ---------- | ----------------------- |
| `--mode {training,inference}` | `training` | Training or inference   |

### Model (auto-detect)

| Parameter           | Description                             |
| ------------------- | --------------------------------------- |
| `--model-path PATH` | HuggingFace model dir (with config.json)|

### Model (manual)

| Parameter                    | Default     | Description                        |
| ---------------------------- | ----------- | ---------------------------------- |
| `--hidden-size`              | 4096        | Hidden dimension                   |
| `--num-layers`               | 32          | Number of transformer layers       |
| `--num-heads`                | 32          | Number of attention heads          |
| `--num-kv-heads`             | = num-heads | KV heads (GQA)                     |
| `--intermediate-size`        | 11008       | FFN intermediate dimension         |
| `--vocab-size`               | 32000       | Vocabulary size                    |
| `--num-experts`              | 0           | Number of MoE experts              |
| `--num-experts-per-tok`      | 0           | Experts activated per token        |
| `--expert-intermediate-size` | 0           | Expert FFN intermediate dimension  |
| `--n-shared-experts`         | 0           | Number of shared experts           |
| `--kv-lora-rank`             | 0           | MLA KV LoRA rank                   |
| `--q-lora-rank`              | 0           | MLA Q LoRA rank                    |
| `--group-norm-size`          | 0           | Lightning Attention GroupNorm size |
| `--first-k-dense-replace`    | 0           | First k layers use dense FFN       |

### Cluster

| Parameter          | Default | Description                              |
| ------------------ | ------- | ---------------------------------------- |
| `--n-gpus`         | 128     | Total number of GPUs                     |
| `--gpus-per-node`  | 8       | GPUs per node                            |
| `--gpu-type`       | H200    | GPU preset (H200/H100/H800/A100/A800)   |
| `--gpu-memory-gb`  | 140     | Override GPU memory (GB)                 |
| `--host-memory-gb` | auto    | Override host CPU memory (GB)            |
| `--gpu-flops`      | auto    | Override BF16 TFLOPS                     |
| `--bw-nvlink`      | auto    | Override NVLink bandwidth (GB/s)         |
| `--bw-ib`          | auto    | Override IB bandwidth (GB/s)             |

### Training Parameters

| Parameter                    | Default  | Description                        |
| ---------------------------- | -------- | ---------------------------------- |
| `--max-tokens-per-mb`        | 131072   | Max tokens per micro-batch         |
| `--max-length`               | 16384    | Max sequence length                |
| `--batch-size`               | 0        | Global batch size (0=unconstrained)|
| `--engine`                   | megatron | Engine preset (megatron/fsdp)      |
| `--no-optimizer-cpu-offload` | false    | Disable optimizer CPU offload      |
| `--no-recompute`             | false    | Disable activation recompute       |

### Inference Parameters

| Parameter            | Default | Description               |
| -------------------- | ------- | ------------------------- |
| `--isl`              | 4096    | Input sequence length     |
| `--osl`              | 512     | Output sequence length    |
| `--infer-batch-size` | 32      | Inference batch size      |

## Hardware Profiling

Optional profiling for higher accuracy (~98% vs ~90% ranking accuracy):

```bash
# Auto-detect backend (Slurm/Ray/Local), single node
python -m autoparallel profile

# Two nodes, Slurm backend
python -m autoparallel profile --n-nodes 2 --backend slurm

# Specify GPUs and partition
python -m autoparallel profile \
    --gpus-per-node 8 --partition gpu --reservation my-res
```

Profiling data is cached in `~/.cache/autoparallel/` and auto-loaded on subsequent runs.

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

1. **ALPA alpha-beta communication model**: TP AllReduce, EP AllToAll, CP Ring
2. **Hierarchical EP AllToAll**: NVLink intra-node + IB cross-node traffic split
3. **Engine-aware optimizations**: EP overlap, TP BW degradation
4. **Roofline inference model**: prefill compute-bound, decode memory-bandwidth-bound
5. **Profiling interpolation** (optional): measured GEMM + comm data, log-space interpolation

See [DESIGN.md](DESIGN.md) for details.

## GPU Presets

| GPU  | Memory | Host Memory | BF16 TFLOPS | NVLink BW | IB BW   |
| ---- | ------ | ----------- | ----------- | --------- | ------- |
| H200 | 141 GB | 1500 GB     | 990         | 450 GB/s  | 50 GB/s |
| H100 | 80 GB  | 1000 GB     | 990         | 450 GB/s  | 50 GB/s |
| H800 | 80 GB  | 1000 GB     | 990         | 400 GB/s  | 50 GB/s |
| A100 | 80 GB  | 1000 GB     | 312         | 300 GB/s  | 25 GB/s |
| A800 | 80 GB  | 1000 GB     | 312         | 200 GB/s  | 25 GB/s |

## References

- [Alpa](https://github.com/alpa-project/alpa) — auto-parallelism for JAX
- [Galvatron](https://github.com/PKU-DAIR/Hetu-Galvatron) — automatic parallelism optimization
- XLA matmul/collective interpolation models

## License

Apache License 2.0
