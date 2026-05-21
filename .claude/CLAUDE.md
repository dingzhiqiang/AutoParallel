# CLAUDE.md - AutoParallel

## Project Overview

AutoParallel is an auto-parallelism strategy advisor for LLM training and inference.
Given a model architecture and GPU cluster, it enumerates all valid parallel strategies
(DP, PP, TP, CP, EP), estimates per-GPU memory, models throughput, and recommends
optimal configurations.

**Tech Stack**: Python 3.10+ | PyTorch (GPU detection + profiling only)

**Core Structure**:
- `autoparallel/__main__.py` — CLI entry point, cost model, memory estimation, strategy search
- `autoparallel/profiler/` — Hardware performance profiling (GEMM, collective comm)
- `autoparallel/profile_data/` — Profiling data management and interpolation

## Commands

```bash
# Training strategy search (default)
python -m autoparallel --model-path /path/to/model --n-gpus 128 --gpu-type H200

# Inference strategy search
python -m autoparallel --mode inference --model-path /path/to/model --n-gpus 64

# Hardware profiling
python -m autoparallel profile --n-nodes 2 --backend slurm

# Install
pip install -e .
```

## Key Design Decisions

- Cost model is engine-aware: Megatron (default for training), FSDP, SGLang (default for inference)
- Two-layer architecture: Layer 1 (analytical) + Layer 2 (profiling interpolation)
- Each profiling lookup falls back independently — not all-or-nothing
- EP AllToAll uses hierarchical modeling (intra NVLink + inter IB)

## Constraints

- No external areal dependencies — fully standalone
- `torch` is the only required dependency; `ray` is optional (for profiling)
- Only reads model config.json, never loads model weights
