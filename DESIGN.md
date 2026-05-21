# Parallel Advisor: 分层代价建模设计

## 概述

autoparallel 用**分层设计**估算分布式 LLM 训练/推理的并行策略效率：

- **Layer 1 (解析模型)**: Peak FLOPS roofline + alpha-beta 通信模型，零配置开箱即用
- **Layer 2 (Profiling 插值)**: 实测 GEMM 性能 + 集合通信延迟，按需开启提升精度

两层独立 fallback——每个 GEMM/通信查询点单独判断，有 profiling 数据就用插值， 否则退回解析公式。不是全有全无。

## 架构

```
autoparallel/
├── __main__.py              # CLI 入口 + 代价模型 + 显存估算
├── profiler/                # Profiling 采集
│   ├── gemm.py              # GEMM 性能测量 (单 GPU, ~3min)
│   ├── comm.py              # 集合通信测量 (torch.distributed, 1-2 节点)
│   └── launcher.py          # 启动器 (Slurm / Ray / Local)
├── profile_data/            # Profiling 数据管理
│   ├── loader.py            # 加载 + 对数空间插值
│   └── presets/             # 内置硬件 preset
│       └── H200_NVLink4_IB400.json
└── DESIGN.md                # 本文件
```

## 数据模型

### 硬件规格 (HardwareSpec)

GPU 型号预设，包含计算/通信/内存三类参数：

| 字段           | 示例 (H200) | 说明                   |
| -------------- | ----------- | ---------------------- |
| gpu_flops      | 990 TFLOPS  | BF16 峰值算力          |
| bw_nvlink      | 450 GB/s    | NVLink 单向带宽        |
| bw_ib          | 50 GB/s     | IB RDMA 单向带宽       |
| hbm_bw         | 4.8 TB/s    | HBM 带宽 (推理 decode) |
| gpu_memory_gb  | 141 GB      | HBM 显存               |
| host_memory_gb | 1500 GB     | 节点 CPU 内存          |

用户只需指定 `--gpu-type H200`，自动获得完整配置。各字段可通过 CLI 单独覆盖。

### GEMM 测量点

```json
{"M": 1024, "N": 6144, "K": 6144, "dtype": "bf16", "time_us": 52.3, "tflops": 1478}
```

- M: 行数 (tokens per GPU / tp)
- N: 输出维度 (hidden_size, intermediate_size / tp, etc.)
- K: 输入维度
- 测量范围: M ∈ \[64, 16384\], N/K ∈ \[1024, 24576\]

### 通信测量点

```json
{"op": "allreduce", "size_bytes": 1048576, "n_gpus": 8, "topology": "nvlink", "time_us": 23.5, "bw_GBs": 425}
```

- op: allreduce | alltoall | p2p
- topology: nvlink (节点内) | ib (跨节点) | mixed (部分跨节点)
- 测量范围: size_bytes ∈ \[1KB, 1GB\], n_gpus ∈ {2, 4, 8, 16}

## 显存估算

### GPU 显存 (per-rank)

| 组件            | 公式                                      | 分片方式   |
| --------------- | ----------------------------------------- | ---------- |
| Model           | params × dtype_bytes                      | TP, PP, EP |
| DDP Buffer      | params × grad_dtype (4 or 2)              | 不分片     |
| Gradient Shard  | params × 4 / dp (仅 !grad_reduce_in_fp32) | DP         |
| Optimizer (GPU) | params × 12 / dp (仅 !cpu_offload)        | DP         |
| Activation      | tokens × H × factor / tp + checkpoint     | TP, PP     |
| CUDA Context    | 8 + n_comm_groups GB                      | 固定       |

Total = (sum) × 1.05 (碎片) + CUDA context

### CPU/Host 内存 (per-node)

当 `optimizer_cpu_offload=True`（默认）时，optimizer states 不占 GPU 但占 CPU：

| 组件            | 公式                            | 说明                    |
| --------------- | ------------------------------- | ----------------------- |
| Optimizer (CPU) | params_per_rank × 12 / dp       | Adam: master + m + v    |
| 节点总量        | per_rank × gpus_per_node + 40GB | 40GB = OS + CUDA 运行时 |

如果 `cpu_total_per_node_gb > host_memory_gb`，策略标记为 `CPU!`（CPU OOM）。

Host memory 的默认值从 `GPU_PRESETS` 获取（H200=1500GB），也可通过 `--host-memory-gb` 覆盖。

## 插值策略

### GEMM 插值

对每个查询 (M, N, K)：

1. 精确匹配 → 直接返回
1. 固定 (N, K)，在 M 轴做对数空间线性插值
1. 若 (N, K) 也没精确匹配，找最近的 (N, K) 组合
1. 数据不足 → 返回 None，触发解析 fallback

### 通信插值

对每个查询 (op, size_bytes, n_gpus, topology)：

1. 固定 (op, n_gpus, topology)，在 size_bytes 轴做对数空间线性插值
1. 数据不足 → 返回 None，触发解析 fallback

## Profiling 启动

### 三种后端

| 后端  | 检测条件                            | GEMM                    | 通信(单节点)                   | 通信(跨节点)                              |
| ----- | ----------------------------------- | ----------------------- | ------------------------------ | ----------------------------------------- |
| Slurm | sinfo 可用                          | srun -N1 --gres=gpu:1   | srun -N1 --gres=gpu:8 torchrun | srun -N2 --gres=gpu:8 torchrun            |
| Ray   | RAY_ADDRESS 或 ray.is_initialized() | @ray.remote(num_gpus=1) | placement_group + torchrun     | placement_group(STRICT_SPREAD) + torchrun |
| Local | fallback                            | 直接运行                | torchrun --nproc_per_node=8    | 不支持(只测 NVLink)                       |

### 单节点 vs 双节点

- **1 节点**: 测 GEMM + NVLink 通信，IB 数据缺失时用解析 alpha-beta
- **2 节点**: 追加测 IB 通信，完整 profiling 数据

## 数据存储

优先级（从高到低）：

1. `--profile-data /path/to/custom.json` 用户显式指定
1. `~/.cache/autoparallel/{gpu_type}.json` 本地 profiling 结果
1. 内置 `profile_data/presets/{gpu_type}_*.json`
1. 无数据 → 纯解析模型

## 使用方式

```bash
# 运行 profiling (首次, ~5-15 min)
python -m autoparallel profile --n-nodes 2 --backend slurm

# 使用 profiling 数据 (自动加载)
python -m autoparallel \
    --model-path /path/to/model --n-gpus 128 --gpu-type H200

# 强制纯解析模型
python -m autoparallel --no-profile \
    --model-path /path/to/model --n-gpus 128 --gpu-type H200
```

## 可行性判断

一个策略必须同时满足两个约束才标记为 OK：

1. **GPU 显存**: `total_mem_gb ≤ gpu_memory_gb`
1. **CPU 内存**: `cpu_total_per_node_gb ≤ host_memory_gb`（仅 optimizer offload 时）

不满足 GPU 约束标记 `OOM`，不满足 CPU 约束标记 `CPU!`。

## 预期收益

| 场景                    | 纯解析          | +Profiling   | 改善点               |
| ----------------------- | --------------- | ------------ | -------------------- |
| GEMM 效率 (TP=8 小矩阵) | 假设 peak FLOPS | 实测利用率   | 修复 BS=64 TP 排序   |
| 通信延迟 (小消息)       | alpha 固定 5µs  | 实测 latency | 小 TP AllReduce 更准 |
| 总排序准确率            | ~90%            | ~98%         | 减少误判             |

## 参考

- XLA: `xla/service/gpu/model/matmul_interpolator.h` (GEMM 插值表)
- XLA: `xla/service/gpu/model/collective_interpolator.h` (通信插值表)
- Galvatron: `galvatron/profile_hardware/profile_allreduce.py` (通信 profiling)
- Galvatron: `galvatron/core/cost_model/components/layer_cost.py` (代价模型)
