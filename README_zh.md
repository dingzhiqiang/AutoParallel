# AutoParallel

[English](README.md) | 中文

LLM 训练与推理的自动并行策略推荐工具。

给定模型架构和 GPU 集群，AutoParallel 枚举所有合法的并行策略组合（DP, PP, TP, CP, EP），估算每张 GPU 的显存占用，基于代价模型对吞吐量排序，推荐最优配置——不 OOM，效率最高。

## 特性

- **训练 + 推理**：分别搜索训练和推理的最优并行策略
- **显存估算**：模型参数、梯度、优化器、激活值、KV Cache——逐 GPU 和逐节点 CPU 内存
- **吞吐量建模**：Roofline 计算 + alpha-beta 通信代价模型
- **硬件 Profiling**：可选的 GEMM + 集合通信实测，提升排序精度（~90% → ~98%）
- **模型支持**：Dense、MoE、MLA、GQA、Lightning Attention、first-k dense replace
- **GPU 预设**：H200、H100、H800、A100、A800，自动带出显存和带宽参数
- **引擎感知**：Megatron 和 FSDP 引擎特有优化（EP overlap、TP 带宽退化）

## 安装

```bash
pip install autoparallel
```

或从源码安装：

```bash
git clone https://github.com/dingzhiqiang/AutoParallel.git
cd AutoParallel
pip install -e .
```

## 快速开始

```bash
# 从 HuggingFace config.json 自动识别模型架构
python -m autoparallel \
    --model-path /path/to/model --n-gpus 128 --gpu-type H200

# 手动指定模型参数
python -m autoparallel \
    --hidden-size 6144 --num-layers 78 --num-heads 64 \
    --num-experts 256 --expert-intermediate-size 2048 \
    --intermediate-size 12288 --vocab-size 154880 \
    --kv-lora-rank 512 --q-lora-rank 2048 \
    --n-gpus 128 --gpu-type H200

# 推理模式
python -m autoparallel --mode inference \
    --model-path /path/to/model --n-gpus 64 --gpu-type H200 \
    --isl 4096 --osl 512 --infer-batch-size 32
```

## 支持的模型架构

- **Dense Transformer**：LLaMA、GPT 系列
- **MoE（混合专家）**：DeepSeek-V2/V3 等
  - 支持 shared experts、first-k dense replace
- **MLA（多头潜在注意力）**：DeepSeek-V2/V3
  - 自动识别 kv_lora_rank、q_lora_rank
- **GQA（分组查询注意力）**：LLaMA-2/3 等
- **Lightning Attention + GroupNorm**：自动处理 group_norm_size 约束

## CLI 参数

### 模式

| 参数                          | 默认       | 说明           |
| ----------------------------- | ---------- | -------------- |
| `--mode {training,inference}` | `training` | 训练或推理模式 |

### 模型（自动识别）

| 参数                | 说明                          |
| ------------------- | ----------------------------- |
| `--model-path PATH` | HF 模型目录（含 config.json） |

### 模型（手动指定）

| 参数                         | 默认        | 说明                               |
| ---------------------------- | ----------- | ---------------------------------- |
| `--hidden-size`              | 4096        | 隐藏维度                           |
| `--num-layers`               | 32          | Transformer 层数                   |
| `--num-heads`                | 32          | 注意力头数                         |
| `--num-kv-heads`             | = num-heads | KV 头数（GQA）                     |
| `--intermediate-size`        | 11008       | FFN 中间维度                       |
| `--vocab-size`               | 32000       | 词表大小                           |
| `--num-experts`              | 0           | MoE 专家数                         |
| `--num-experts-per-tok`      | 0           | 每 token 激活专家数                |
| `--expert-intermediate-size` | 0           | 专家 FFN 中间维度                  |
| `--n-shared-experts`         | 0           | 共享专家数                         |
| `--kv-lora-rank`             | 0           | MLA KV LoRA 秩                     |
| `--q-lora-rank`              | 0           | MLA Q LoRA 秩                      |
| `--group-norm-size`          | 0           | Lightning Attention GroupNorm 大小 |
| `--first-k-dense-replace`    | 0           | 前 k 层用 Dense FFN                |

### 集群

| 参数               | 默认 | 说明                                    |
| ------------------ | ---- | --------------------------------------- |
| `--n-gpus`         | 128  | GPU 总数                                |
| `--gpus-per-node`  | 8    | 每节点 GPU 数                           |
| `--gpu-type`       | H200 | GPU 型号预设 (H200/H100/H800/A100/A800) |
| `--gpu-memory-gb`  | 140  | 覆盖 GPU 显存 (GB)                      |
| `--host-memory-gb` | 自动 | 覆盖节点 CPU 内存 (GB)                  |
| `--gpu-flops`      | 自动 | 覆盖 BF16 TFLOPS                        |
| `--bw-nvlink`      | 自动 | 覆盖 NVLink 带宽 (GB/s)                 |
| `--bw-ib`          | 自动 | 覆盖 IB 带宽 (GB/s)                     |

**GPU 预设值**：

| GPU 型号 | 显存   | 节点内存 | BF16 TFLOPS | NVLink BW | IB BW   |
| -------- | ------ | -------- | ----------- | --------- | ------- |
| H200     | 141 GB | 1500 GB  | 990         | 450 GB/s  | 50 GB/s |
| H100     | 80 GB  | 1000 GB  | 990         | 450 GB/s  | 50 GB/s |
| H800     | 80 GB  | 1000 GB  | 990         | 400 GB/s  | 50 GB/s |
| A100     | 80 GB  | 1000 GB  | 312         | 300 GB/s  | 25 GB/s |
| A800     | 80 GB  | 1000 GB  | 312         | 200 GB/s  | 25 GB/s |

### 训练参数

| 参数                         | 默认     | 说明                           |
| ---------------------------- | -------- | ------------------------------ |
| `--max-tokens-per-mb`        | 131072   | 每 micro-batch 最大 token 数   |
| `--max-length`               | 16384    | 序列最大长度                   |
| `--batch-size`               | 0        | 全局 batch size（0=不约束 DP） |
| `--engine`                   | megatron | 引擎预设 (megatron/fsdp)       |
| `--no-optimizer-cpu-offload` | false    | 禁用 optimizer CPU offload     |
| `--no-recompute`             | false    | 禁用 activation recompute      |
| `--no-grad-reduce-in-fp32`   | false    | 用 bf16 梯度累加               |

### 推理参数

| 参数                 | 默认 | 说明                |
| -------------------- | ---- | ------------------- |
| `--isl`              | 4096 | 输入序列长度        |
| `--osl`              | 512  | 输出序列长度        |
| `--infer-batch-size` | 32   | 推理并发 batch size |

## 硬件 Profiling

可选的硬件性能采集，提升排序精度：

```bash
# 自动检测后端（Slurm/Ray/Local），单节点
python -m autoparallel profile

# 双节点，Slurm 后端
python -m autoparallel profile --n-nodes 2 --backend slurm

# 指定 GPU 数和分区
python -m autoparallel profile \
    --gpus-per-node 8 --partition gpu --reservation my-res
```

数据保存在 `~/.cache/autoparallel/`，后续运行自动加载。

## 引擎配置

代价模型的部分优化假设与引擎实现相关，通过 `--engine` 参数选择预设：

| 优化                    | Megatron | FSDP |
| ----------------------- | -------- | ---- |
| EP AllToAll 与 FFN 重叠 | Yes      | No   |
| TP 带宽退化（TP>4）     | Yes      | No   |
| MoE activation factor   | 18       | 14   |

## 显存估算

### GPU 显存（per-rank）

| 组件         | 说明                                             |
| ------------ | ------------------------------------------------ |
| Model        | 模型参数，按 TP/PP/EP 分片                       |
| DDP Buffer   | 梯度通信缓冲区（不分片）                         |
| Gradient     | fp32 梯度分片（仅 !grad_reduce_in_fp32）         |
| Optimizer    | Adam 状态 12 bytes/param / dp（仅 !cpu_offload） |
| Activation   | 检查点 + 工作显存，按 TP 分片                    |
| CUDA Context | 8GB + 1GB × 通信组数                             |

### CPU 内存（per-node）

当 `optimizer_cpu_offload=True`（默认）时，optimizer states 占 CPU 内存。超出 `host_memory_gb` 的策略标记为 `CPU!`。

## 输出示例

### 训练模式

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
       + TP=8 节点内 NVLink
       + PP=8 bubble 10%
       + EP=8 节点内 NVLink AllToAll
```

### 推理模式

```
Top-3 Recommended (by aggregate decode throughput)
======================================================================
  #1  tp1_ep8  (8 GPUs/instance x 8 instances)
      Memory: 33.3G / 140G | KV capacity: ~580K tokens
      Prefill: 12.3 ms (333K tok/s/inst)
      Decode:  8.1 ms/tok (123 tok/s/inst)
      Aggregate: 984 tok/s (100%)
```

## 代价模型

1. **ALPA alpha-beta 通信模型**：对 TP AllReduce、EP AllToAll、CP Ring 分别建模
2. **分层 EP AllToAll**：区分 NVLink 节点内和 IB 跨节点的流量比例
3. **引擎感知优化**：EP overlap、TP BW degradation 等引擎特定行为
4. **Roofline 推理模型**：Prefill compute-bound，Decode memory-bandwidth-bound
5. **Profiling 插值**（可选）：实测 GEMM + 通信数据，对数空间插值替代解析估算

详见 [DESIGN.md](DESIGN.md)。

## 参考

- [Alpa](https://github.com/alpa-project/alpa) — JAX 自动并行框架
- [Galvatron](https://github.com/PKU-DAIR/Hetu-Galvatron) — 自动并行优化系统
- XLA matmul/collective 插值模型

## 许可证

Apache License 2.0
