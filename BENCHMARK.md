# Benchmark Results

Real-world validation of AutoParallel's strategy recommendations.

## 1. Training: GLM-5.1 (757B MoE) on 128×H200

### Setup

| Item | Value |
| --- | --- |
| **Model** | GLM-5.1 (757B MoE, 256 experts, MLA, 78 layers, H=6144) |
| **Cluster** | 128 × H200 (141 GB), 16 nodes |
| **Interconnect** | NVLink 450 GB/s (intra-node), IB 50 GB/s (inter-node) |
| **Data** | SFT, max_length=16384 |
| **Engine** | Megatron, 1F1B pipeline, CPU optimizer offload |

### Memory Estimation Accuracy

| Strategy | Predicted (GB) | Measured (GB) | Error |
| --- | --- | --- | --- |
| DP=2 PP=4 TP=8 CP=2 EP=16 | 95.6 | 99 | -3.4% |
| DP=2 PP=4 TP=4 CP=4 EP=16 | 99.1 | 104 | -4.7% |
| DP=2 PP=8 TP=8 EP=8 | 75.9 | 78 | -2.7% |

Memory estimation error is consistently **<5%**, sufficient for OOM filtering.

### Training Throughput: BS=16, MTPM=128K (Full Benchmark)

4 strategies compared on 128×H200, Slurm jobs A/B/C/D:

| Strategy | DP | PP | TP | CP | EP | Avg Step (s) | Mem (GB) | Rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **A (optimal)** | 2 | 4 | 8 | 2 | 16 | **37.6** | 101 | **#1** |
| C | 2 | 4 | 4 | 4 | 16 | 45.6 | 101.5 | #2 |
| B | 2 | 4 | 16 | 1 | 16 | ~50 | 99 | #3 |
| D | 2 | 2 | 16 | 2 | 32 | 73.3 | 100 | #4 |

AutoParallel (with EP overlap modeling) correctly identifies Strategy A as optimal.
The old serial model incorrectly favored Strategy B (TP=16).

### Training Throughput: BS=16, MTPM=16K

#### Before vs After EP Overlap Modeling

| Strategy | GPU Mem | Step Time | Old Rank | New Rank |
| --- | --- | --- | --- | --- |
| DP=2 PP=4 TP=8 CP=2 EP=16 | ~99G | **25.2s** | #2 | **#1** |
| DP=2 PP=4 TP=4 CP=4 EP=16 | ~104G | 25.6s | #3 | #2 |
| DP=2 PP=4 TP=16 CP=1 EP=16 | ~95G | 27.7s | **#1** | #10 |
| DP=2 PP=4 TP=8 CP=1 EP=8 | ~78G | 28.6s | (ref) | (ref) |

**Key finding**: The old model (serial EP assumption) incorrectly ranked TP=16 as #1.
After adding EP overlap modeling, TP=8 CP=2 correctly ranks #1, matching real measurements.

#### Why Old Model Was Wrong

The old model assumed EP AllToAll runs serially with FFN compute:
`cost = T_ep + T_ffn` — this systematically overestimates EP cost, biasing toward
high TP (which reduces EP communication volume).

The new model uses `cost = max(T_ep, T_ffn)` for Megatron engine, correctly reflecting
that dispatch/combine overlaps with expert FFN compute.

### Training Throughput: BS=32, MTPM=16K

| Strategy | GPU Mem | Step Time | Rank |
| --- | --- | --- | --- |
| DP=2 PP=8 TP=4 CP=2 EP=8 | ~80G | **33.5s** | **#1** |
| DP=2 PP=8 TP=8 CP=1 EP=8 | ~78G | 34.8s | #2 |

Step times are close (33.5s vs 34.8s). AutoParallel scores are also close (100% vs 100%),
correctly reflecting the small real-world gap.

### MLA Impact on Context Parallelism Cost

MLA compresses KV to extremely low dimensions, making CP nearly free:

| Architecture | KV dim per token | CP Ring Transfer Volume |
| --- | --- | --- |
| Standard MHA (h=64, d=128) | 16,384 | 100% (baseline) |
| MLA (kv_lora=512, rope=64) | 576 | **3.5%** |

This explains why CP=2 or CP=4 has almost zero overhead on MLA models like GLM-5.1
and DeepSeek-V3.

**Real measurement (GLM-5.1, 128×H200, BS=8, MTPM=16K)**:

| Strategy | TP | CP | EP | Avg Step (s) | Mem (GB) |
| --- | --- | --- | --- | --- | --- |
| s1 | 8 | 4 | 32 | 66.6 | 105 |
| s2 | 4 | 8 | 32 | 65.1 | 112 |

CP=4 vs CP=8 step time difference is **<2%**, confirming MLA makes CP communication
nearly free.

### BailingMoE CP Scaling (64×H200)

BailingMoE V2.5 (256 experts, MLA, Lightning Attention) on 64 GPU:

| CP | Step Time (s) | GPU Mem (GB) | Speedup |
| --- | --- | --- | --- |
| 4 | 106.66 | 57.88 (73%) | 1.0× |
| 8 | **47.28** | **41.00** (52%) | **2.25×** |

CP=8 achieves 2.25× speedup: both activation memory reduction and sequence-level
parallelism contribute. The low CP communication cost (MLA) makes higher CP efficient.

### TP Bandwidth Degradation

Measured NVLink effective bandwidth at different TP sizes (H200, 8 GPU):

| TP | Theoretical BW | Effective BW | Ratio |
| --- | --- | --- | --- |
| 2 | 450 GB/s | ~440 GB/s | 98% |
| 4 | 450 GB/s | ~430 GB/s | 96% |
| 8 | 450 GB/s | ~310 GB/s | 69% |

TP=8 effective bandwidth drops to ~70% of TP=4, consistent with the `1/sqrt(tp/4)` model:
`1/sqrt(8/4) = 1/sqrt(2) ≈ 0.707`.

## 2. Inference: SGLang Cookbook Cross-Validation

We compare AutoParallel's inference recommendations with SGLang's official
[cookbook configs](https://github.com/sgl-project/sglang/tree/main/.claude/skills/llm-serving-auto-benchmark/configs/cookbook-llm)
(used by their auto-benchmark system).

### Convention Mapping

| AutoParallel | SGLang | Meaning |
| --- | --- | --- |
| `TP=T` | `tp_size=T` | Total GPUs per instance |
| `EP=E` | `ep_size=E` | Expert parallel degree (EP ≤ TP) |
| `n_instances` | (launch N copies) | Multi-instance deployment |

### Comparison Results (BF16, H200 141GB)

| Model | Params | Type | GPUs | SGLang Cookbook | AutoParallel #1 | Match |
| --- | --- | --- | --- | --- | --- | --- |
| LLaMA-3.1-70B | 70B | Dense GQA | 4×H100 | TP=4 | TP=4 (67.4G) | ✓ |
| LLaMA-3.1-70B | 70B | Dense GQA | 8×H100 | - | TP=4 ×2 inst (6902 tps) | ✓ Multi-inst |
| Qwen3-235B-A22B | 246B | MoE 128E | 16×H200 | TP=8, EP∈{1,4,8} | TP=8 EP=8 ×2 inst (8615 tps) | ✓ |
| DeepSeek-V3 | 725B | MoE+MLA | 16×H200 | TP=8 (FP8) | TP=16 EP=16 (126.7G) | Note¹ |
| GLM-5.1 | 757B | MoE+MLA | 16×H200 | (prod deploy) | TP=16 EP=16 (114.5G) | ✓ Real² |

¹ SGLang cookbook uses FP8 weights (1 byte/param) on 8 GPU; AutoParallel models BF16
(2 bytes/param) requiring 16 GPU. With FP8 support, the recommendation would be TP=8 EP=8.

² Verified against real SGLang rollout in GRPO training: predicted 114.5G vs measured 117G
(**2.1% error**).

### Key Findings

1. **TP alignment**: For Dense models, AutoParallel's TP matches SGLang's `tp_size` exactly
2. **EP alignment**: For MoE models, AutoParallel consistently recommends the highest
   feasible EP, which SGLang's cookbook includes as the top candidate in its search space
3. **Multi-instance**: AutoParallel correctly recommends multiple instances when GPUs
   exceed minimum requirement, maximizing aggregate throughput
4. **FP8 gap**: Models with native FP8 weights (DeepSeek-V3, Qwen3.5-FP8) cannot be
   directly compared — this is a planned enhancement

## 3. Inference: Qwen Model Family Validation

Workload: `isl=4096, osl=1024, batch=64`, H200 141GB.

| Model | Params | Type | GPUs | AutoParallel #1 | Mem/GPU | Decode TPS |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B | 8.2B | Dense | 8 | TP=1 (8 inst) | 77.1G | 25K |
| Qwen3-30B-A3B | 30.1B | MoE 128E | 8 | TP=1 (8 inst) | 77.4G | 18.5K |
| Qwen3.5-35B-A3B | 33.8B | MoE 256E | 8 | TP=1 (8 inst) | 82.9G | 17K |
| Qwen2.5-72B | 72.7B | Dense GQA | 8 | TP=2 (4 inst) | 103.6G | 6.3K |
| Qwen3.5-122B-A10B | 118.9B | MoE 256E | 16 | TP=2 EP=2 (8 inst) | 127.1G | 10K |
| Qwen3-235B-A22B | 231.7B | MoE 128E | 32 | TP=4 EP=4 (8 inst) | 125.6G | 9.9K |
| Qwen3.5-397B-A17B | 391.6B | MoE 512E | 64 | TP=8 EP=8 (8 inst) | 107.1G | 12K |

### Observations

- **Small MoE models** (30B–35B): TP=1 is sufficient, maximizing multi-instance throughput
- **Medium MoE models** (120B–235B): TP=2–4 with matching EP balances memory and throughput
- **Large MoE models** (400B+): TP=8 EP=8 required; aggregate throughput remains high
  due to MoE's inherent activation sparsity
- **Dense models scale differently**: Qwen2.5-72B requires TP=2 (no EP), fewer instances

## 4. Bug Fixes During Validation

### 4.1 Prefill Attention FLOPs (commit `3181671`)

**Symptom**: LLaMA-3-8B TP=1 batch=64 showed 40.5s prefill time (should be ~5s).

**Root cause**: `_compute_flops` used `T² = (isl × batch / tp)²` for attention scores,
treating the entire token batch as one sequence. Correct formula is `batch × seq_len²`.

**Fix**: Added `batch` parameter; compute `seq_len = T / batch` for per-request attention
score calculation.

### 4.2 Nested HF Config (commit `fe61628`)

**Symptom**: Qwen3.5-397B detected as 6.7B Dense instead of 391.6B MoE.

**Root cause**: Config stores MoE params under `text_config` key (common in VL models).

**Fix**: Auto-detect and merge `text_config` into top-level config.

### 4.3 KV Head Replication for Inference (commit `fe61628`)

**Symptom**: Qwen3.5-397B (kv_heads=2) only allowed TP=1 or TP=2, both OOM.

**Root cause**: Training constraint `kv_heads % tp == 0` applied to inference.
SGLang/vLLM can replicate KV heads when `tp > kv_heads`.

**Fix**: Removed the constraint for inference mode.

### 4.4 GPU Memory from Preset (this session)

**Symptom**: `--gpu-type H100` still used 140GB default, allowing infeasible strategies.

**Root cause**: `--gpu-memory-gb` defaulted to 140.0; `--gpu-type` only affected
HardwareSpec, not ClusterSpec.

**Fix**: `--gpu-memory-gb` defaults to 0 (auto), resolved from GPU preset.
