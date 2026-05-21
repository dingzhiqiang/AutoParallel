# AutoParallel: 自动并行策略推荐系统技术文档

## 1. 概述

AutoParallel 是一个面向大语言模型（LLM）分布式训练与推理的自动并行策略推荐系统。系统通过**分层代价建模**估算不同并行策略的显存占用和计算吞吐量，在秒级时间内完成搜索空间的遍历，推荐不 OOM 且效率最高的并行配置。

### 1.1 设计目标

- **零 GPU 依赖**：纯解析建模，不需要实际运行即可推荐策略
- **分层精度**：Layer 1 解析模型开箱即用；Layer 2 Profiling 插值按需提升精度
- **训练 + 推理**：分别针对训练和推理场景建模，覆盖不同的性能瓶颈
- **引擎感知**：Megatron / FSDP / SGLang 引擎的运行时行为差异纳入代价模型

### 1.2 系统架构

```
autoparallel/
├── __main__.py              # CLI 入口 + 策略搜索 + 代价模型 + 显存估算
├── profiler/                # 硬件性能采集（Layer 2）
│   ├── gemm.py              # 单 GPU GEMM 性能测量 (~3min)
│   ├── comm.py              # 集合通信测量 (torch.distributed, 1-2 节点)
│   └── launcher.py          # 多后端启动器 (Slurm / Ray / Local)
├── profile_data/            # Profiling 数据管理
│   ├── loader.py            # 数据加载 + 对数空间插值
│   └── presets/             # 内置 GPU 硬件 preset
│       └── H200_NVLink4_IB400.json
└── DESIGN.md                # 本文件
```

### 1.3 相关工作

| 系统 | 方法 | 特点 |
| --- | --- | --- |
| **Alpa** [Zheng et al., OSDI'22] | ILP + DP 搜索 intra/inter-op 并行 | 自动化程度高，但搜索开销大 |
| **Galvatron** [Miao et al., VLDB'23] | DP 搜索 + 解析代价模型 | 支持异构，但不支持 MoE/EP |
| **AIConfigurator** [Xu et al., 2025] | 算子级性能数据库 + 迭代级建模 | 框架感知，但仅面向推理 |
| **AutoParallel** (本系统) | 解析代价模型 + 可选 Profiling 插值 | 训练 + 推理，MoE/MLA/EP 支持 |

## 2. 数据模型

### 2.1 硬件规格 (HardwareSpec)

GPU 型号预设，包含计算、通信、内存三类参数。用户只需指定 `--gpu-type H200`，自动获得完整配置；各字段也可通过 CLI 单独覆盖。

| 字段 | 符号 | 示例 (H200) | 说明 |
| --- | --- | --- | --- |
| `gpu_flops` | $F_{peak}$ | 990 TFLOPS | BF16 峰值算力 |
| `bw_nvlink` | $B_{nvlink}$ | 450 GB/s | NVLink 单向带宽 |
| `bw_ib` | $B_{ib}$ | 50 GB/s | IB RDMA 单向带宽 |
| `hbm_bw` | $B_{hbm}$ | 4.8 TB/s | HBM 带宽（推理 decode roofline） |
| `alpha_nvlink` | $\alpha_{nvlink}$ | 5 µs | NVLink 启动延迟 |
| `alpha_ib` | $\alpha_{ib}$ | 10 µs | IB 启动延迟 |
| `gpu_memory_gb` | $M_{gpu}$ | 141 GB | HBM 显存 |
| `host_memory_gb` | $M_{host}$ | 1500 GB | 节点 CPU 内存 |

### 2.2 模型规格 (ModelSpec)

从 HuggingFace `config.json` 自动解析，覆盖以下架构特征：

| 架构特征 | 关键字段 | 判定条件 |
| --- | --- | --- |
| Dense Transformer | `hidden_size`, `num_attention_heads`, `intermediate_size` | 基础，始终存在 |
| GQA | `num_key_value_heads` | $n_{kv} < n_{heads}$ |
| MoE | `n_routed_experts`, `num_experts_per_tok`, `moe_intermediate_size` | $n_{experts} > 0$ |
| MLA | `kv_lora_rank`, `q_lora_rank`, `qk_nope_head_dim`, `v_head_dim` | $kv\_lora\_rank > 0$ |
| Lightning Attention | `group_norm_size` / `layer_group_size` | $group\_norm\_size > 0$ |
| first-k dense replace | `first_k_dense_replace` | $first\_k > 0$ |

### 2.3 并行策略空间

**训练 (ParallelStrategy)**：5 维并行空间

| 维度 | 符号 | 分片对象 | 约束 |
| --- | --- | --- | --- |
| Data Parallel | $d$ | batch | $d = N_{gpu} / (p \cdot t \cdot c)$ |
| Pipeline Parallel | $p$ | layers | $p \mid N_{gpu}$, $p \leq L$ |
| Tensor Parallel | $t$ | attention heads | $t \mid n_{heads}$, $n_{kv} \mod t = 0$ |
| Context Parallel | $c$ | sequence | $c \in \{1, 2, 4, 8\}$, $T \mod c = 0$ |
| Expert Parallel | $e$ | MoE experts | $e = t \cdot c$, $n_{experts} \mod e = 0$ |

恒等约束：$d \cdot p \cdot t \cdot c = N_{gpu}$

流水线调度约束（1F1B）：$batch\_size / d \geq 2p$

**推理 (InferenceStrategy)**：3 维 + 实例数

| 维度 | 约束 | 说明 |
| --- | --- | --- |
| TP | $t \mid n_{heads}$ | 可跨节点 |
| EP | $e \mid n_{experts}$, $e \leq t$ | SGLang 约束 |
| PP | 默认禁用 | 推理串行执行，PP 只增加延迟 |
| n_instances | $N_{gpu} / (t \cdot p)$ | 多实例提升聚合吞吐 |

### 2.4 引擎配置 (EngineConfig)

不同引擎的运行时行为差异影响代价模型的关键假设：

| 参数 | Megatron | FSDP | SGLang | 含义 |
| --- | --- | --- | --- | --- |
| `overlap_ep_alltoall` | True | False | True | EP AllToAll 是否与 FFN 计算重叠 |
| `tp_bw_degradation` | True | False | False | TP>4 时 NVLink 带宽是否退化 |
| `act_factor_dense` | 10 | 10 | 10 | Dense 层激活内存因子 |
| `act_factor_moe` | 18 | 14 | 14 | MoE 层激活内存因子 |

**重叠 vs 串行的影响**：

- Megatron 引擎将 EP AllToAll dispatch/combine 与 expert FFN 计算重叠执行：

  $T_{MoE} = \max(T_{ep}, T_{routed\_ffn})$

- FSDP 引擎串行执行：

  $T_{MoE} = T_{ep} + T_{routed\_ffn}$

这一差异对 MoE 模型的最优 TP/EP 组合选择有显著影响。

## 3. 显存估算模型

显存估算是可行性判断的基础——一个策略必须同时满足 GPU 显存和 CPU 内存约束才被标记为可行。

### 3.1 GPU 显存（per-rank）

每张 GPU 的显存占用由以下组件构成：

#### 3.1.1 模型参数

模型参数按 bf16 存储（$B=2$ bytes），按 TP/PP/EP 分片。以每个 Transformer 层为例：

**Attention 参数（Standard MHA/GQA）**：

$$P_{attn} = \frac{H \cdot n_{heads} \cdot d_{head} + H \cdot n_{kv} \cdot d_{head} \cdot 2 + n_{heads} \cdot d_{head} \cdot H}{t}$$

其中 $d_{head} = H / n_{heads}$。分子依次为 Q/K/V/O 投影，全部按 TP 分片。

**Attention 参数（MLA）**：

MLA 架构（DeepSeek-V2/V3, GLM-5.1）中，部分投影不按 TP 分片（replicated），部分按 head 维度分片：

$$P_{attn}^{MLA} = \underbrace{H \cdot r_q + H \cdot (r_{kv} + d_{rope})}_{replicated} + \frac{\underbrace{r_q \cdot n_{heads} \cdot d_{head} + r_{kv} \cdot n_{heads} \cdot (d_{nope} + d_v) + n_{heads} \cdot d_v \cdot H}_{TP\ 分片}}{t}$$

其中 $r_q$ = `q_lora_rank`, $r_{kv}$ = `kv_lora_rank`, $d_{rope}$ = `qk_rope_head_dim`, $d_{nope}$ = `qk_nope_head_dim`, $d_v$ = `v_head_dim`。

**FFN 参数（Dense）**：

$$P_{ffn}^{dense} = \frac{3 \cdot H \cdot d_{inter}}{t}$$

3 对应 gate/up/down 三个投影矩阵（SwiGLU/GeGLU 结构）。

**FFN 参数（MoE）**：

$$P_{ffn}^{MoE} = \frac{n_{experts}}{e} \cdot 3 \cdot H \cdot d_{expert} + \frac{n_{shared} \cdot 3 \cdot H \cdot d_{inter}}{t} + H \cdot n_{experts}$$

依次为：本地 routed experts（按 EP 分片）、shared experts（按 TP 分片）、router 权重。

**Embedding + Output Head**：

$$P_{embed} = \frac{V \cdot H}{t}$$

**每层参数汇总**：

$$P_{layer} = P_{attn} + P_{ffn} + 2H \quad (\text{2H 为 attn\_norm + ffn\_norm})$$

**Pipeline 分片**：每个 PP stage 承担 $\lceil L/p \rceil$ 层。取最坏情况（余数分配到前几个 stage）。

**模型参数显存**：

$$M_{model} = \left(\lceil L/p \rceil \cdot P_{layer} + P_{embed}\right) \cdot B$$

#### 3.1.2 梯度通信缓冲区（DDP Buffer）

DDP 需要完整的梯度缓冲区用于 AllReduce 通信，**不按 DP 分片**：

$$M_{grad\_buf} = P_{per\_rank} \cdot B_{grad}$$

其中 $B_{grad} = 4$（fp32）或 $2$（bf16），取决于 `grad_reduce_in_fp32` 设置。

#### 3.1.3 梯度分片

当 `grad_reduce_in_fp32=False` 时，梯度按 DP 分片存储：

$$M_{grad\_shard} = \frac{P_{per\_rank} \cdot 4}{d}$$

#### 3.1.4 优化器状态

Adam 优化器每个参数需要 12 bytes（master weights fp32 + exp\_avg fp32 + exp\_avg\_sq fp32）：

- **GPU 模式**（`cpu_offload=False`）：$M_{opt} = \frac{P_{per\_rank} \cdot 12}{d}$
- **CPU Offload 模式**（默认）：GPU 显存为 0，转移到 CPU 内存

#### 3.1.5 激活内存

激活内存与每张 GPU 处理的 token 数和隐藏维度相关：

$$M_{act} = T_{per\_gpu} \cdot H \cdot f_{act} \cdot B / t$$

其中 $T_{per\_gpu} = T_{mb} / (t \cdot c)$，$f_{act}$ 为引擎相关的激活因子：

| 引擎 | Dense $f_{act}$ | MoE $f_{act}$ | 差异原因 |
| --- | --- | --- | --- |
| Megatron | 10 | 18 | MoE 额外的 router logits、dispatch/combine 缓冲区 |
| FSDP | 10 | 14 | FSDP 实现的 MoE 激活更紧凑 |

当启用 activation recompute 时，仅保存 checkpoint 边界的激活，大幅减少激活内存。

#### 3.1.6 CUDA Context

$$M_{cuda} = 8 + n_{comm\_groups} \quad (\text{GB})$$

$n_{comm\_groups}$ 为 NCCL 通信组数（TP + DP + PP + EP）。

#### 3.1.7 总显存

$$M_{total} = (M_{model} + M_{grad\_buf} + M_{grad\_shard} + M_{opt} + M_{act}) \times 1.05 + M_{cuda}$$

1.05 为显存碎片系数。可行性判断：$M_{total} \leq M_{gpu}$。

### 3.2 CPU/Host 内存（per-node）

当 `optimizer_cpu_offload=True`（默认）时，优化器状态占 CPU 内存：

$$M_{opt\_cpu}^{per\_rank} = \frac{P_{per\_rank} \cdot 12}{d}$$

$$M_{cpu}^{per\_node} = M_{opt\_cpu}^{per\_rank} \times G_{per\_node} + 40\ \text{GB}$$

40 GB 为操作系统 + CUDA runtime 的固定开销。可行性判断：$M_{cpu}^{per\_node} \leq M_{host}$。

不满足 GPU 约束标记 `OOM`，不满足 CPU 约束标记 `CPU!`。

### 3.3 推理显存估算

推理显存由三部分组成，无梯度和优化器：

$$M_{infer} = M_{weights} + M_{kv\_cache} + M_{activation} + M_{cuda}$$

**权重**：同训练的模型参数计算，但无 DDP buffer 和梯度。

**KV Cache**：

$$M_{kv} = \frac{T_{max\_batch} \cdot \lceil L/p \rceil \cdot kv\_bytes\_per\_token}{t}$$

其中每层每 token 的 KV bytes：
- Standard: $kv\_bytes = n_{kv} \cdot d_{head} \cdot 2 \cdot 2$（K + V, bf16）
- MLA: $kv\_bytes = (r_{kv} + d_{rope}) \cdot 2$（压缩 KV）

**激活**：使用查表经验值，按 TP 和模型类型确定系数 $c$：

| TP | Dense $c$ | MoE $c$ |
| --- | --- | --- |
| 1 | 10 | 22 |
| 2 | 6 | 13 |
| 4 | 5 | 10 |
| 8 | 5 | 10 |

$$M_{act}^{infer} = \max\left(2 \cdot T_{prefill} \cdot H \cdot c,\ 70\ \text{MB}\right)$$

## 4. 训练吞吐量代价模型

训练代价模型采用 **ALPA 风格的 alpha-beta 通信模型** [Zheng et al., OSDI'22]，结合 **分层 EP AllToAll 建模**（参考 AIConfigurator [Xu et al., 2025]），估算每个策略的相对吞吐量。

### 4.1 计算代价（per-layer）

每层的计算量按 FLOPs 估算，除以 GPU 峰值算力得到计算时间。

#### 4.1.1 Attention 投影 FLOPs

**Standard MHA/GQA**：

$$F_{attn\_proj} = \frac{8 \cdot H^2 \cdot T}{t}$$

对应 Q/K/V/O 四个投影，每个为 $2HHT$ FLOPs（矩阵乘法 FLOPs = $2MNK$）。

**MLA**：

$$F_{attn\_proj}^{MLA} = 2T \cdot \left(\underbrace{H \cdot r_q + H \cdot (r_{kv} + d_{rope})}_{replicated} + \frac{\underbrace{r_q \cdot n_h \cdot d_h + r_{kv} \cdot n_h \cdot (d_{nope} + d_v) + n_h \cdot d_v \cdot H}_{TP\ 分片}}{t}\right)$$

#### 4.1.2 Attention Score FLOPs

**Standard**：

$$F_{attn\_score} = \frac{4 \cdot T^2 \cdot H}{t}$$

对应 $QK^T$（$2T^2 H$）和 $AV$（$2T^2 H$），按 TP 分片。

**MLA**：

$$F_{attn\_score}^{MLA} = \frac{2 \cdot T^2 \cdot n_h \cdot (d_{qk} + d_v)}{t}$$

其中 $d_{qk} = d_{nope} + d_{rope}$。

#### 4.1.3 FFN FLOPs

**Dense**：

$$F_{ffn}^{dense} = \frac{6 \cdot H \cdot d_{inter} \cdot T}{t}$$

6 对应 gate/up/down 三个矩阵乘法，每个 $2HdT$。

**MoE**（routed + shared experts）：

$$F_{routed} = \frac{k \cdot 6 \cdot H \cdot d_{expert} \cdot T}{e}$$

$$F_{shared} = \frac{n_{shared} \cdot 6 \cdot H \cdot d_{inter} \cdot T}{t}$$

$$F_{ffn}^{MoE} = F_{routed} + F_{shared}$$

$k$ = `num_experts_per_tok`（每 token 激活的专家数）。Routed experts 按 EP 分片，shared experts 按 TP 分片。

#### 4.1.4 计算时间

$$T_{compute} = \frac{F_{attn\_proj} + F_{attn\_score} + F_{ffn}}{F_{peak}}$$

其中 $F_{peak}$ 为 GPU 的 BF16 峰值算力（来自 HardwareSpec）。

### 4.2 通信代价（per-layer）

通信代价采用经典的 **alpha-beta 模型** [Hockney, 1994]：

$$T_{comm} = \alpha + \frac{V}{B}$$

$\alpha$ 为启动延迟，$V$ 为消息体积（bytes），$B$ 为带宽。

#### 4.2.1 TP AllReduce

Attention 输出需要 TP AllReduce。Ring AllReduce 的消息体积为 $2 \cdot \frac{t-1}{t} \cdot V_{payload}$：

- Dense 模型：每层 2 次 AllReduce（attention + FFN），$V_{payload} = H \cdot T \cdot 2$
- MoE 模型：每层 1 次 AllReduce（仅 attention），FFN 走 EP AllToAll

$$T_{tp} = \begin{cases} \alpha_{nvlink} + \frac{n_{ar} \cdot 2 \cdot \frac{t-1}{t} \cdot H \cdot T \cdot 2}{B_{nvlink}^{eff}} & \text{if } t \leq G_{per\_node} \\ \alpha_{ib} + \frac{n_{ar} \cdot 2 \cdot \frac{t-1}{t} \cdot H \cdot T \cdot 2}{B_{ib}} & \text{if } t > G_{per\_node} \end{cases}$$

**TP 带宽退化**（Megatron 引擎）：当 TP > 4 时，多个 NVLink ring 竞争带宽：

$$B_{nvlink}^{eff} = \frac{B_{nvlink}}{\sqrt{t / 4}} \quad (t > 4, \text{Megatron only})$$

#### 4.2.2 EP AllToAll（分层建模）

EP AllToAll 的消息体积（dispatch + combine 两次）：

$$V_{total} = 2 \cdot T \cdot H \cdot 2 \cdot \frac{e - 1}{e}$$

**节点内**（$e \leq G_{per\_node}$）：

$$T_{ep} = \alpha_{nvlink} + \frac{V_{total}}{B_{nvlink}}$$

**跨节点**（$e > G_{per\_node}$，分层 AllToAll）：

参考 AIConfigurator 的分层建模方法，将流量分为节点内和跨节点两部分：

$$f_{intra} = \frac{G_{per\_node} - 1}{e - 1}, \quad f_{inter} = 1 - f_{intra}$$

跨节点通信受多节点拥塞影响，引入拥塞因子 $\sqrt{n_{ep\_nodes}}$：

$$T_{ep} = \alpha_{ib} + \frac{V_{total} \cdot f_{intra}}{B_{nvlink}} + \frac{V_{total} \cdot f_{inter}}{B_{ib}} \cdot \sqrt{\frac{e}{G_{per\_node}}}$$

#### 4.2.3 CP Ring Attention

Context Parallel 使用 Ring Attention，相邻 rank 之间传输 KV：

$$V_{cp} = T \cdot d_{kv} \cdot 2 \cdot (c - 1)$$

- Standard: $d_{kv} = n_{kv} \cdot d_{head} \cdot 2$
- MLA: $d_{kv} = r_{kv} + d_{rope}$

$$T_{cp} = \alpha_{nvlink} \cdot (c - 1) + \frac{V_{cp}}{B_{nvlink}}$$

#### 4.2.4 PP P2P

Pipeline stage 之间传输激活值：

$$T_{pp} = \alpha_{ib} + \frac{T_{mb} / c \cdot H \cdot 2}{B_{ib}}$$

### 4.3 每层时间 → 总训练步时间

**每层时间**（前向 1x + 反向 2x = 3x）：

$$T_{layer} = 3 \times T_{per\_layer}$$

其中 $T_{per\_layer}$ 取决于引擎的 overlap 策略：

- **Megatron**（EP overlap）：
  $$T_{per\_layer} = T_{attn\_compute} + T_{tp} + \max(T_{ep}, T_{routed\_compute}) + T_{shared\_compute} + T_{cp}$$

- **FSDP**（串行）：
  $$T_{per\_layer} = T_{total\_compute} + T_{tp} + T_{ep} + T_{cp}$$

**Pipeline stage 时间**：

$$T_{stage} = T_{layer} \times \lceil L/p \rceil + T_{pp}$$

**1F1B Pipeline Schedule** [Narayanan et al., 2019]：

$$T_{total} = (p + n_{mb} - 1) \times T_{stage}$$

其中 $n_{mb} = batch\_size / d$（micro-batch 数）。Pipeline bubble 比例为 $(p-1)/(p + n_{mb} - 1)$。

**吞吐量得分**：

$$Score = \frac{d \cdot n_{mb}}{T_{total}}$$

得分越高，策略越优。

## 5. 推理性能模型

推理性能模型基于 **Roofline 模型** [Williams et al., 2009]，区分 prefill（compute-bound）和 decode（memory-bandwidth-bound）两个阶段。

### 5.1 Prefill（计算密集型）

Prefill 阶段一次性处理全部输入 token，受 GPU 计算能力限制：

$$T_{prefill}^{per\_layer} = \frac{F_{compute}(T_{pf})}{F_{peak}} + T_{comm}(T_{pf})$$

$$T_{prefill} = T_{prefill}^{per\_layer} \times \lceil L/p \rceil + T_{pp\_p2p} \times (p - 1)$$

其中 $T_{pf} = ISL \times batch / tp$，$F_{compute}$ 和 $T_{comm}$ 的计算方式与训练相同（参见 4.1、4.2）。

**Prefill 吞吐量**：

$$TPS_{prefill} = \frac{ISL \times batch}{T_{prefill}}$$

### 5.2 Decode（内存带宽密集型）

Decode 阶段每步只生成 1 个 token/request，受 HBM 读取带宽限制。采用 Roofline 模型：

$$T_{decode}^{per\_stage} = \max\left(\underbrace{\frac{F_{compute}(batch)}{F_{peak}} \times layers}_{compute},\ \underbrace{\frac{W_{bytes} + KV_{bytes}}{B_{hbm}}}_{memory\ read}\right) + T_{comm}(batch) \times layers$$

**权重读取**：$W_{bytes}$ 为本 stage 的模型权重大小（bf16）。

**KV Cache 读取**：

$$KV_{bytes} = \frac{batch \times \bar{S} \times layers \times kv\_bytes\_per\_token}{t}$$

其中 $\bar{S} = ISL + OSL/2$ 为平均序列长度。

**PP 串行**：推理 PP 无 microbatch 流水线，各 stage 串行执行：

$$T_{decode} = T_{decode}^{per\_stage} \times p + T_{pp\_p2p} \times (p - 1)$$

**Decode 吞吐量**：

$$TPS_{decode} = \frac{batch}{T_{decode}}$$

### 5.3 聚合吞吐量

多实例部署时，聚合 decode 吞吐量为：

$$TPS_{aggregate} = n_{instances} \times TPS_{decode}$$

推理策略按 $TPS_{aggregate}$ 降序排序。

## 6. Profiling 插值（Layer 2）

Layer 1 的解析模型假设 GPU 工作在 peak FLOPS，通信使用理论带宽——对于小矩阵 GEMM 和小消息 AllReduce，实际利用率显著低于理论值。Layer 2 通过实测数据替代解析公式，提升排序精度。

### 6.1 数据采集

#### GEMM Profiling

在单 GPU 上遍历 $(M, N, K)$ 网格，测量实际 GEMM 性能：

```json
{"M": 1024, "N": 6144, "K": 6144, "dtype": "bf16", "time_us": 52.3, "tflops": 1478}
```

- $M$: 行数（tokens per GPU / TP）
- $N$: 输出维度（hidden\_size, intermediate\_size / TP 等）
- $K$: 输入维度
- 测量范围: $M \in [64, 16384]$, $N/K \in [1024, 24576]$

#### 通信 Profiling

使用 `torch.distributed` 测量 AllReduce / AllToAll / P2P：

```json
{"op": "allreduce", "size_bytes": 1048576, "n_gpus": 8, "topology": "nvlink", "time_us": 23.5, "bw_GBs": 425}
```

- 拓扑: `nvlink`（节点内） | `ib`（跨节点） | `mixed`
- 测量范围: $size \in [1\text{KB}, 1\text{GB}]$, $n_{gpus} \in \{2, 4, 8, 16\}$

### 6.2 对数空间插值

实测数据稀疏，需要插值到任意查询点。参考 XLA 的 `matmul_interpolator` 和 `collective_interpolator` 实现。

#### GEMM 插值

对查询 $(M, N, K)$：

1. 精确匹配 $(N, K)$ → 在 $M$ 轴做对数空间线性插值
2. 无精确 $(N, K)$ → 找最近的 $(N, K)$ 组合
3. 数据不足 → 返回 None，退回 Layer 1 解析公式

**对数空间线性插值**：给定两个测量点 $(M_1, t_1)$ 和 $(M_2, t_2)$（$M_1 < M_q < M_2$）：

$$\log t_q = \log t_1 + \frac{\log M_q - \log M_1}{\log M_2 - \log M_1} \cdot (\log t_2 - \log t_1)$$

选择对数空间的原因：GEMM 性能与矩阵维度的关系近似幂律。

#### 通信插值

对查询 $(op, size, n_{gpus}, topo)$，固定 $(op, n_{gpus}, topo)$，在 $size$ 轴做对数空间插值。

### 6.3 数据加载优先级

1. `--profile-data /path/to/custom.json` — 用户显式指定
2. `~/.cache/autoparallel/{gpu_type}.json` — 本地 profiling 结果
3. 内置 `profile_data/presets/{gpu_type}_*.json` — 预置数据
4. 无数据 → 纯 Layer 1 解析模型

**独立 fallback**：每个查询点独立判断——有 profiling 数据就用插值，否则退回解析公式。不是全有全无。

### 6.4 Profiling 后端

| 后端 | 检测条件 | GEMM | 通信（单节点） | 通信（跨节点） |
| --- | --- | --- | --- | --- |
| Slurm | `sinfo` 可用 | `srun -N1 --gres=gpu:1` | `srun -N1 --gres=gpu:8 torchrun` | `srun -N2 --gres=gpu:8 torchrun` |
| Ray | `RAY_ADDRESS` 或 `ray.is_initialized()` | `@ray.remote(num_gpus=1)` | placement\_group + torchrun | STRICT\_SPREAD + torchrun |
| Local | fallback | 直接运行 | `torchrun --nproc_per_node=8` | 不支持（仅 NVLink） |

### 6.5 精度提升

| 场景 | 纯 Layer 1 | +Layer 2 Profiling | 改善 |
| --- | --- | --- | --- |
| GEMM 效率（TP=8 小矩阵） | 假设 peak FLOPS | 实测利用率 | 修复小 batch TP 排序 |
| 通信延迟（小消息） | $\alpha$ 固定 5µs | 实测 latency | 小 TP AllReduce 更准 |
| 总排序准确率 | ~90% | ~98% | 减少误判 |

## 7. 搜索算法

### 7.1 训练策略搜索

枚举所有合法的 $(d, p, t, c, e)$ 组合：

1. **生成候选维度**：
   - $t \in \text{divisors}(n_{heads})$, $t \leq 16$
   - $p \in \text{divisors}(N_{gpu})$, $p \leq \min(16, L)$
   - $c \in \{1, 2, 4, 8\}$

2. **约束过滤**：
   - GQA: $n_{kv} \mod t = 0$
   - GroupNorm: $(n_{heads}/t) \mod group\_norm\_size = 0$
   - CP: $(n_{heads}/t) \mod c = 0$（Lightning Attention）
   - Token 整除: $T_{mb} \mod c = 0$
   - MoE: $e = t \cdot c$, $n_{experts} \mod e = 0$
   - DP: $d = N_{gpu} / (p \cdot t \cdot c)$，须为正整数
   - Batch: $batch\_size \mod d = 0$（若指定）
   - 1F1B: $batch\_size / d \geq 2p$

3. **显存估算**（§3）→ 过滤 OOM

4. **吞吐量估算**（§4）→ 按效率得分降序排序

5. **输出 Top-K** 并附带优劣分析

### 7.2 推理策略搜索

1. 枚举 $t \in \text{divisors}(n_{heads})$, $t \leq N_{gpu}$
2. MoE: 枚举 $e \in \text{divisors}(n_{experts})$, $e \leq t$
3. PP: 默认禁用（推理串行执行，PP 只增加延迟）
4. 计算 $n_{instances} = N_{gpu} / (t \cdot p)$
5. 显存估算（§3.3）→ 过滤 OOM
6. 性能估算（§5）→ 按聚合 decode 吞吐量降序排序

## 8. 参考文献

1. Zheng, L. et al. "Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning." OSDI, 2022.
2. Xu, T. et al. "AIConfigurator: Lightning-Fast Configuration Optimization for Multi-Framework LLM Serving." 2025.
3. Wu, R. et al. "Rethinking Dynamic Networks and Heterogeneous Computing with Automatic Parallelization." APNET, 2025.
4. Miao, X. et al. "Galvatron: Efficient Transformer Training over Multiple GPUs Using Automatic Parallelism." VLDB, 2023.
5. Narayanan, D. et al. "PipeDream: Generalized Pipeline Parallelism for DNN Training." SOSP, 2019.
6. Williams, S. et al. "Roofline: An Insightful Visual Performance Model for Multicore Architectures." CACM, 2009.
7. Hockney, R. "The Communication Challenge for MPP: Intel Paragon and Meiko CS-2." Parallel Computing, 1994.
