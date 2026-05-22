# Benchmark Results

Real-world validation of AutoParallel's strategy recommendations.

## Setup

| Item | Value |
| --- | --- |
| **Model** | GLM-5.1 (671B MoE, 256 experts, MLA, 78 layers, H=6144) |
| **Cluster** | 128 × H200 (141 GB), 16 nodes |
| **Interconnect** | NVLink 450 GB/s (intra-node), IB 50 GB/s (inter-node) |
| **Data** | SFT, max_length=16384 |
| **Engine** | Megatron, 1F1B pipeline, CPU optimizer offload |

## Memory Estimation Accuracy

| Strategy | Predicted (GB) | Measured (GB) | Error |
| --- | --- | --- | --- |
| DP=2 PP=4 TP=8 CP=2 EP=16 | 95.6 | 99 | -3.4% |
| DP=2 PP=4 TP=4 CP=4 EP=16 | 99.1 | 104 | -4.7% |
| DP=2 PP=8 TP=8 EP=8 | 75.9 | 78 | -2.7% |

Memory estimation error is consistently **<5%**, sufficient for OOM filtering.

## Training Throughput: BS=16, MTPM=16K

### Before vs After EP Overlap Modeling

| Strategy | GPU Mem | Step Time | Old Rank | New Rank |
| --- | --- | --- | --- | --- |
| DP=2 PP=4 TP=8 CP=2 EP=16 | ~99G | **25.2s** | #2 | **#1** |
| DP=2 PP=4 TP=4 CP=4 EP=16 | ~104G | 25.6s | #3 | #2 |
| DP=2 PP=4 TP=16 CP=1 EP=16 | ~95G | 27.7s | **#1** | #10 |
| DP=2 PP=4 TP=8 CP=1 EP=8 | ~78G | 28.6s | (ref) | (ref) |

**Key finding**: The old model (serial EP assumption) incorrectly ranked TP=16 as #1.
After adding EP overlap modeling, TP=8 CP=2 correctly ranks #1, matching real measurements.

### Why Old Model Was Wrong

The old model assumed EP AllToAll runs serially with FFN compute:
`cost = T_ep + T_ffn` — this systematically overestimates EP cost, biasing toward
high TP (which reduces EP communication volume).

The new model uses `cost = max(T_ep, T_ffn)` for Megatron engine, correctly reflecting
that dispatch/combine overlaps with expert FFN compute.

## Training Throughput: BS=32, MTPM=16K

| Strategy | GPU Mem | Step Time | Rank |
| --- | --- | --- | --- |
| DP=2 PP=8 TP=4 CP=2 EP=8 | ~80G | **33.5s** | **#1** |
| DP=2 PP=8 TP=8 CP=1 EP=8 | ~78G | 34.8s | #2 |

Step times are close (33.5s vs 34.8s). AutoParallel scores are also close (100% vs 100%),
correctly reflecting the small real-world gap.

## MLA Impact on Context Parallelism Cost

MLA compresses KV to extremely low dimensions, making CP nearly free:

| Architecture | KV dim per token | CP Ring Transfer Volume |
| --- | --- | --- |
| Standard MHA (h=64, d=128) | 16,384 | 100% (baseline) |
| MLA (kv_lora=512, rope=64) | 576 | **3.5%** |

This explains why CP=2 or CP=4 has almost zero overhead on MLA models like GLM-5.1
and DeepSeek-V3.

## TP Bandwidth Degradation

Measured NVLink effective bandwidth at different TP sizes (H200, 8 GPU):

| TP | Theoretical BW | Effective BW | Ratio |
| --- | --- | --- | --- |
| 2 | 450 GB/s | ~440 GB/s | 98% |
| 4 | 450 GB/s | ~430 GB/s | 96% |
| 8 | 450 GB/s | ~310 GB/s | 69% |

TP=8 effective bandwidth drops to ~70% of TP=4, consistent with the `1/sqrt(tp/4)` model:
`1/sqrt(8/4) = 1/sqrt(2) ≈ 0.707`.
