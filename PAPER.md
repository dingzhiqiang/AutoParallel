# AutoParallel: Engine-Aware Auto-Parallelism for LLM Training and Inference

## Abstract

Optimizing distributed parallelism for large language models requires navigating a
combinatorial space of strategies (DP, PP, TP, CP, EP) under tight GPU memory constraints.
Existing cost models either ignore MoE-specific communication patterns (Alpa) or require
hardware benchmark databases (AIConfigurator). We present AutoParallel, an analytical cost
model that introduces three key improvements: (1) engine-aware EP AllToAll-FFN overlap
modeling, (2) TP NVLink bandwidth degradation correction, and (3) MLA-aware context
parallelism costing. On a 128-GPU H200 cluster training GLM-5.1 (671B MoE+MLA),
AutoParallel achieves <5% memory estimation error and correctly identifies the optimal
strategy where prior models fail. The system runs in seconds with zero GPU dependency.

## 1. Introduction

Large-scale LLM training requires combining multiple parallelism dimensions — Data
Parallel (DP), Pipeline Parallel (PP), Tensor Parallel (TP), Context Parallel (CP), and Expert
Parallel (EP). For a 671B MoE model on 128 GPUs, the valid strategy space exceeds 30
configurations. Manual exploration costs 10+ minutes per trial and carries OOM risk.

Existing auto-parallelism systems have limitations:

- **Alpa** [1] automates intra/inter-op parallelism search via ILP, but its cost model
  assumes serial compute-communication execution and lacks MoE/EP support.
- **AIConfigurator** [2] adds MoE EP AllToAll modeling with a hierarchical NVLink+IB
  cost model, but relies on a hardware performance database and does not model
  engine-specific runtime optimizations.
- **Galvatron** [3] uses DP-based search with analytical cost models, supporting
  heterogeneous clusters but lacking MoE/EP and context parallelism.

AutoParallel addresses these gaps with a purely analytical, engine-aware cost model that
supports the full parallelism space (DP, PP, TP, CP, EP) for both training and inference.

## 2. Method

### 2.1 Cost Model Architecture

AutoParallel uses a two-layer architecture (see [DESIGN.md](DESIGN.md) for full formulation):

- **Layer 1 (Analytical)**: Peak FLOPS roofline + alpha-beta communication model.
  Zero-configuration, works out of the box.
- **Layer 2 (Profiling)**: Optional GEMM + collective communication measurements with
  log-space interpolation, replacing analytical estimates point-by-point.

Each profiling lookup falls back independently — a query with profiling data uses the
measured value; a query without falls back to the analytical formula. This is not
all-or-nothing.

### 2.2 Key Contributions

#### Contribution 1: EP AllToAll-FFN Overlap Modeling

**Observation**: Megatron's MoE token dispatcher executes AllToAll dispatch/combine
concurrently with expert FFN computation. Prior models (Alpa, AIConfigurator) assume
serial execution, systematically overestimating EP communication cost.

**Impact**: Serial assumption biases the optimizer toward high-TP strategies (which reduce
EP volume), but in practice, low-TP + high-EP configurations are faster when overlap
is available.

**Modeling**:

```
// Megatron engine (overlap):
T_MoE = max(T_ep, T_routed_ffn)

// FSDP engine (serial):
T_MoE = T_ep + T_routed_ffn
```

This is controlled via the `EngineConfig` abstraction, making the cost model engine-aware.

**Validation**: On GLM-5.1 (128×H200, BS=16), the serial model incorrectly ranks TP=16
as #1. After adding overlap modeling, TP=8 CP=2 correctly ranks #1, matching measured
step times. See [BENCHMARK.md](BENCHMARK.md) for detailed results.

#### Contribution 2: TP NVLink Bandwidth Degradation

**Observation**: With TP > 4, multiple AllReduce rings compete for NVLink bandwidth.
Measured effective bandwidth at TP=8 is ~70% of TP=4 (see [BENCHMARK.md](BENCHMARK.md)).

**Modeling**: Apply a degradation factor for large TP groups:

```
B_eff = B_nvlink / sqrt(tp / 4)    when tp > 4
```

This correction is Megatron-specific (`tp_bw_degradation=True` in EngineConfig). FSDP
and SGLang engines do not exhibit this pattern.

#### Contribution 3: MLA-Aware Context Parallelism

**Observation**: MLA (Multi-head Latent Attention) compresses KV to extremely low
dimensions. CP ring attention transfers KV between adjacent ranks — with MLA, the
transfer volume drops to **3.5% of standard MHA**:

```
KV_dim(MLA)  = kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576
KV_dim(MHA)  = num_kv_heads × head_dim × 2 = 64 × 128 × 2 = 16,384
Ratio = 576 / 16,384 ≈ 3.5%
```

This makes CP=2 or CP=4 nearly free on MLA models (DeepSeek-V2/V3, GLM-5.1),
enabling strategies that prior models would penalize.

### 2.3 Engine-Aware Abstraction

Different training engines have different runtime behaviors. We parameterize these as
`EngineConfig`:

| Parameter | Megatron | FSDP | SGLang |
| --- | --- | --- | --- |
| EP AllToAll overlaps FFN | Yes | No | Yes |
| TP BW degradation (tp>4) | Yes | No | No |
| MoE activation factor | 18 | 14 | 14 |

This allows the same cost model to produce correct recommendations for different
engines without code changes.

### 2.4 Inference Model

For inference, AutoParallel uses a roofline model distinguishing two phases:

- **Prefill** (compute-bound): `T = F_compute / F_peak + T_comm`
- **Decode** (memory-bandwidth-bound): `T = max(F/F_peak, (W+KV)/B_hbm) + T_comm`

Multi-instance deployment maximizes aggregate throughput:
`TPS_aggregate = n_instances × TPS_decode`, where `n_instances = N_gpu / (TP × PP)`.

PP is disabled by default for inference (stages execute serially, only adding latency).

## 3. Evaluation

Detailed experimental results are in [BENCHMARK.md](BENCHMARK.md). Summary:

| Metric | Result |
| --- | --- |
| Memory estimation error | <5% (GLM-5.1, 128×H200) |
| Inference memory error | 2.1% (GLM-5.1 SGLang deployment) |
| SGLang cookbook alignment | TP/EP matches official cookbook configs |
| Ranking accuracy (with overlap) | Correctly identifies optimal strategy |
| Ranking accuracy (without overlap) | Incorrect — biased toward high TP |
| Search time | <1 second (pure analytical) |
| GPU dependency | None (reads config.json only) |

## 4. Comparison with Prior Work

| Feature | Alpa [1] | AIConfigurator [2] | Galvatron [3] | **AutoParallel** |
| --- | --- | --- | --- | --- |
| Communication model | α-β | α-β + silicon DB | α-β | α-β + hierarchical EP |
| MoE EP support | No | AllToAll | No | AllToAll + **overlap** |
| Context Parallel | No | Yes | No | Yes (**MLA-aware**) |
| Engine-aware | No | No | No | **EngineConfig** |
| TP BW degradation | No | No | No | **sqrt(tp/4) model** |
| Inference mode | No | Yes | No | **Roofline** (prefill+decode) |
| Dependency | ILP solver | HW benchmark DB | DP solver | **Zero** (pure analytical) |
| Training support | Yes | No | Yes | **Yes** |

## 5. Limitations and Future Work

1. **Activation estimation**: Uses empirical factors, not precise per-layer accounting.
2. **Communication topology**: Assumes uniform NVLink/IB; does not model NVSwitch
   or asymmetric topologies.
3. **Sequence parallelism**: Does not explicitly model Megatron's sequence parallelism
   optimization for activation memory.
4. **Multi-stage workflows**: Does not model RLHF/GRPO multi-stage resource allocation.
5. **FP8 quantization**: Currently models BF16 weights only; FP8 models need extension.
6. **Profiling calibration**: Could further improve accuracy with a small set of benchmark
   data points.

## References

1. Zheng, L. et al. "Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning." OSDI, 2022.
2. Xu, T. et al. "AIConfigurator: Lightning-Fast Configuration Optimization for Multi-Framework LLM Serving." 2025.
3. Miao, X. et al. "Galvatron: Efficient Transformer Training over Multiple GPUs Using Automatic Parallelism." VLDB, 2023.
4. Williams, S. et al. "Roofline: An Insightful Visual Performance Model for Multicore Architectures." CACM, 2009.
5. Wu, R. et al. "Rethinking Dynamic Networks and Heterogeneous Computing with Automatic Parallelization." APNET, 2025.
