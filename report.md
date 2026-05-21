# Parallel Strategy Advisor: 技术报告

## 1. 问题定义

大规模 LLM 训练需要组合多种并行策略（DP, PP, TP, CP, EP），配置空间巨大。以 GLM-5.1（671B MoE, 256 experts, MLA）在
128 GPU 上训练为例，合法策略超过 30 种。 人工试错成本极高（每次实验需 10+ 分钟），且 OOM 风险难以预判。

**目标**：给定模型架构 + 集群规模 + 显存约束，自动推荐吞吐最优的并行策略。

## 2. 相关工作

### 2.1 ALPA (OSDI'22)

ALPA 提出了自动并行化框架，用 ILP 搜索 intra-op（TP）和 inter-op（PP）的最优切分。 通信代价用 alpha-beta 模型：

```
T_comm = alpha + M / beta
```

其中 alpha 为启动延迟，beta 为带宽，M 为传输数据量。

**局限**：ALPA 的代价模型假设通信和计算串行执行，未建模 MoE 架构下的 EP 通信， 也没有考虑 Context Parallelism。

### 2.2 AIConfigurator

AIConfigurator 在 ALPA 基础上增加了：

- 对 MoE EP AllToAll 的建模（分层 NVLink + IB）
- 基于硬件性能数据库（silicon database）的精确建模
- Activation memory 的经验系数

**局限**：依赖硬件性能数据库（需要实际 benchmark 数据），不同引擎的运行时优化 （如 Megatron 的 EP overlap）未被建模。

## 3. 方法

### 3.1 显存估算模型

每 GPU 显存 = 模型权重 + DDP 梯度缓冲 + 梯度分片 + 优化器状态 + Activation + CUDA 上下文

#### 3.1.1 模型权重

按并行维度切分：

- **Attention 参数**: 按 TP 切分
  - MLA: q_a_proj, q_b_proj, kv_a_proj, kv_b_proj, o_proj
  - MHA/GQA: q_proj, k_proj, v_proj, o_proj
- **MoE FFN**: routed experts 按 EP 切分，shared experts 按 TP 切分
- **Dense FFN**: 按 TP 切分
- **Embedding/LM-Head**: 按 TP 切分
- **first_k_dense_replace**: 前 k 层使用 Dense FFN 而非 MoE

#### 3.1.2 梯度与优化器

- **DDP 梯度缓冲**: fp32 时为参数的 2x（4B vs 2B），Megatron 在 reduce-scatter 前 需要完整的 fp32 buffer
- **分布式优化器**: reduce-scatter 后每 GPU 只持有 1/DP 的梯度分片
- **CPU Offload**: 优化器状态（master weights + exp_avg + exp_avg_sq = 12B/param） 完全放 CPU，GPU
  侧为 0

#### 3.1.3 Activation

使用经验系数建模（参考 AIConfigurator）：

```
act_per_layer = tokens_per_gpu × H × act_factor × dtype_bytes / TP
```

act_factor 根据引擎和模型类型不同：

- Dense (Megatron): 10
- MoE (Megatron): 18（含 router logits, dispatch/combine buffers, expert 中间态）
- MoE (FSDP): 14

在 Megatron 1F1B 流水线调度下，同时在 flight 的 micro-batch 数 = 2 × PP。

**验证结果**（GLM-5.1, 128 GPU, MTPM=16K）:

| 策略                      | 预估 (GB) | 实际 (GB) | 误差  |
| ------------------------- | --------- | --------- | ----- |
| DP=2 PP=4 TP=8 CP=2 EP=16 | 95.6      | 99        | -3.4% |
| DP=2 PP=4 TP=4 CP=4 EP=16 | 99.1      | 104       | -4.7% |
| DP=2 PP=8 TP=8 EP=8       | 75.9      | 78        | -2.7% |

### 3.2 代价模型（训练吞吐）

#### 3.2.1 基础：ALPA alpha-beta 模型

每层训练时间 = 3 × (计算时间 + 通信时间)，其中 3x = 1x forward + 2x backward。

**计算时间**:

```
attn_proj_flops = 8 × H² × T / TP
attn_score_flops = 4 × T² × H / TP
ffn_flops = 6 × H × FFN_dim × T / TP  (dense)
          = top_k × 6 × H × expert_dim × T / EP  (routed experts)
          + n_shared × 6 × H × expert_dim × T / TP  (shared experts)
```

**TP AllReduce** (ring-based):

```
vol = n_allreduce × 2(TP-1)/TP × H × T × 2B
T_tp = alpha_nvlink + vol / bw_nvlink   (TP ≤ gpus_per_node)
```

**EP AllToAll** (hierarchical):

```
vol_per_dir = T × H × 2B × (EP-1) / EP
total_vol = 2 × vol_per_dir  (dispatch + combine)

// 节点内 NVLink + 跨节点 IB
intra_frac = (gpus_per_node - 1) / (EP - 1)
T_ep = alpha_ib + total_vol × intra_frac / bw_nvlink
     + total_vol × (1-intra_frac) / bw_ib × sqrt(ep_nodes)
```

**CP Ring Attention**:

```
kv_dim = kv_lora_rank + qk_rope_head_dim   (MLA)
       = num_kv_heads × head_dim × 2       (MHA/GQA)
vol = T × kv_dim × 2B × (CP-1)
T_cp = alpha_nvlink × (CP-1) + vol / bw_nvlink
```

**PP 1F1B Schedule**:

```
total_time = (PP + num_mb - 1) × stage_time
min_num_mb = 2 × PP  (Megatron 约束)
```

#### 3.2.2 创新点 1: EP AllToAll 与 FFN 计算重叠

**发现**：Megatron MoE 引擎的 token dispatcher 在 dispatch 和 combine 阶段与 expert FFN 计算并行执行。ALPA
和 AIConfigurator 均假设串行，导致高估 EP 通信代价。

**实验证据** (GLM-5.1, BS=16, MTPM=16K, 128 GPU):

| 策略                       | ALPA 模型排名 | 实际 step time | 实际排名 |
| -------------------------- | ------------- | -------------- | -------- |
| DP=2 PP=4 TP=8 CP=1 EP=8   | #1            | 28.6s          | #3       |
| DP=2 PP=4 TP=4 CP=4 EP=16  | #3            | 25.6s          | #1       |
| DP=2 PP=4 TP=16 CP=1 EP=16 | #2            | 27.7s          | #2       |

旧模型（串行假设）系统性地偏好高 TP（减少 EP 通信量），但实际上低 TP + 高 EP 在有 overlap 的情况下更快。

**改进后的建模**:

```python
if engine.overlap_ep_alltoall and model.is_moe and ep > 1:
    moe_time = max(ep_time, routed_compute)  # overlap: 取 max
    per_layer = attn_compute + tp_time + moe_time + shared_compute + cp_time
else:
    per_layer = total_compute + tp_time + ep_time + cp_time  # 串行: 取 sum
```

#### 3.2.3 创新点 2: TP NVLink 带宽退化

**发现**：当 TP > 4 时，多个 AllReduce ring 竞争 NVLink 带宽，有效带宽下降。 实测 TP=8 的有效带宽约为 TP=4 的 70%
(1/√2)。

**建模**:

```python
if engine.tp_bw_degradation and tp > 4:
    effective_bw = bw_nvlink / sqrt(tp / 4)
```

#### 3.2.4 创新点 3: 引擎感知代价模型 (EngineConfig)

不同训练引擎（Megatron, FSDP）的运行时优化不同，影响并行策略选择。我们引入 `EngineConfig` 抽象，将引擎特定行为参数化：

```python
@dataclass
class EngineConfig:
    overlap_ep_alltoall: bool = True    # EP AllToAll 与 FFN 重叠
    tp_bw_degradation: bool = True      # TP>4 带宽退化
    act_factor_dense: int = 10          # Dense activation 系数
    act_factor_moe: int = 18            # MoE activation 系数
```

预设：

- `MEGATRON_ENGINE`: overlap=True, degradation=True, moe_factor=18
- `FSDP_ENGINE`: overlap=False, degradation=False, moe_factor=14

#### 3.2.5 MLA 对 CP 代价的影响

MLA (Multi-head Latent Attention) 将 KV 压缩到极低维度：

```
kv_dim = kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576
```

对比标准 MHA（h=64, d=128）:

```
kv_dim_mha = 64 × 128 × 2 = 16384
```

MLA 的 CP ring transfer 数据量仅为 MHA 的 576/16384 ≈ 3.5%，使得 CP 通信代价 近乎可忽略。这解释了为什么 CP=2 甚至
CP=4 在 MLA 模型上几乎没有额外开销。

### 3.3 代价模型（推理）

#### 3.3.1 Prefill (Compute-bound)

与训练 forward 相同，但只有 1x（无 backward）：

```
prefill_time = per_layer_time × L / PP + pp_p2p × (PP-1)
prefill_tps = isl × batch_size / prefill_time
```

#### 3.3.2 Decode (Memory-bandwidth bound)

每步只处理 batch_size 个 token，瓶颈是 HBM 带宽：

```
weights_to_read = model_params_per_gpu × 2B
kv_to_read = batch_size × (isl + osl) × kv_dim × L / TP × 2B
decode_time = max(compute_time, (weights + kv) / hbm_bw) + comm_time
```

#### 3.3.3 多实例并行

推理关心总系统吞吐：

```
n_instances = n_gpus / (TP × EP × PP)
aggregate_tps = n_instances × decode_tps
```

## 4. 实验验证

### 4.1 实验设置

- **模型**: GLM-5.1 (671B MoE, 256 experts, MLA, 78 layers, H=6144)
- **集群**: 128 × H200 (140GB), 16 nodes, NVLink 450GB/s, IB 50GB/s
- **数据**: SFT, max_length=16384
- **引擎**: Megatron with 1F1B pipeline, CPU optimizer offload

### 4.2 BS=16, MTPM=16K

| 策略                       | 预估显存 | 实际显存 | Step Time | Advisor 排名 (旧) | Advisor 排名 (新) |
| -------------------------- | -------- | -------- | --------- | ----------------- | ----------------- |
| DP=2 PP=4 TP=8 CP=2 EP=16  | 95.6G    | ~99G     | 25.2s     | #2                | **#1**            |
| DP=2 PP=4 TP=4 CP=4 EP=16  | 99.1G    | ~104G    | 25.6s     | #3                | #2                |
| DP=2 PP=4 TP=16 CP=1 EP=16 | 94.0G    | ~95G     | 27.7s     | **#1**            | #10               |
| DP=2 PP=4 TP=8 CP=1 EP=8   | (ref)    | ~78G     | 28.6s     | (ref)             | (ref)             |

旧模型 #1 (TP=16) 实际排第三；新模型修正后 #1 (TP=8 CP=2) 与实际最优一致。

### 4.3 BS=32, MTPM=16K

| 策略                     | 预估显存 | 实际显存 | Step Time | Advisor 排名 (新) |
| ------------------------ | -------- | -------- | --------- | ----------------- |
| DP=2 PP=8 TP=4 CP=2 EP=8 | 78.1G    | ~80G     | 33.5s     | **#1**            |
| DP=2 PP=8 TP=8 CP=1 EP=8 | 75.9G    | ~78G     | 34.8s     | #2                |

两者 step time 接近（33.5s vs 34.8s），新模型得分也接近（100% vs 100%），正确 反映了实际差距不大的事实。

## 5. 与现有方法对比

| 特性       | ALPA       | AIConfigurator          | Parallel Advisor (本工作)    |
| ---------- | ---------- | ----------------------- | ---------------------------- |
| 通信模型   | alpha-beta | alpha-beta + silicon DB | alpha-beta + hierarchical    |
| MoE 支持   | 无         | EP AllToAll             | EP AllToAll + overlap        |
| CP 支持    | 无         | 有                      | 有 (MLA-aware)               |
| 引擎感知   | 无         | 无                      | EngineConfig (Megatron/FSDP) |
| TP BW 退化 | 无         | 无                      | sqrt(tp/4) 模型              |
| 推理模式   | 无         | 有                      | Roofline (prefill + decode)  |
| 依赖       | ILP solver | 硬件 benchmark DB       | 纯解析 (zero dependency)     |
| 准确度     | ~          | 高 (需 calibration)     | 中高 (显存误差 \<5%)         |

### 5.1 核心创新总结

1. **EP AllToAll-FFN Overlap 建模**: 首次在解析代价模型中建模 MoE 引擎的 dispatch/combine 与 expert 计算重叠，用
   max(ep_time, ffn_time) 替代 ep_time + ffn_time

1. **TP 带宽退化建模**: 建模大 TP group 下多 ring 竞争 NVLink 带宽的现象， 用 1/√(tp/4) 衰减因子修正有效带宽

1. **引擎感知抽象 (EngineConfig)**: 将引擎运行时优化参数化为可配置的 dataclass， 使同一代价模型适配不同训练引擎（Megatron vs
   FSDP）

1. **MLA-aware CP 代价**: 正确建模 MLA 架构下 CP ring transfer 的极低代价 （仅标准 MHA 的 3.5%），避免过度惩罚 CP
   策略

1. **零依赖纯解析模型**: 不依赖硬件 benchmark 数据库或 ILP solver， 仅需模型架构参数即可运行

## 6. 局限与未来工作

1. **Activation 估算精度**: 使用经验系数，无法精确反映不同层类型的差异
1. **通信拓扑**: 假设均匀的 NVLink/IB 拓扑，未建模 NVSwitch 等复杂互联
1. **Sequence Parallelism**: 未显式建模 Megatron 的 sequence parallelism 对 activation memory
   的优化
1. **多阶段训练**: 未建模 RLHF/GRPO 多阶段工作流的资源分配
1. **实际吞吐 calibration**: 可引入少量 benchmark 数据进行系数校准
