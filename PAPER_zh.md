# AutoParallel: 面向 LLM 训练和推理的引擎感知自动并行系统

## 摘要

优化大语言模型的分布式并行策略需要在严格的 GPU 显存约束下搜索策略组合空间（DP、PP、TP、CP、EP）。现有代价模型要么忽略 MoE 特有的通信模式（Alpa），要么依赖硬件 benchmark 数据库（AIConfigurator）。我们提出 AutoParallel，一个解析代价模型，引入三项关键改进：（1）引擎感知的 EP AllToAll-FFN overlap 建模，（2）TP NVLink 带宽退化校正，（3）MLA 感知的 Context Parallelism 代价计算。在 128 GPU H200 集群训练 GLM-5.1（671B MoE+MLA）的实验中，AutoParallel 实现 <5% 的显存估计误差，并正确识别最优策略（此前的模型会给出错误结果）。系统在秒级完成推荐，零 GPU 依赖。

## 1. 引言

大规模 LLM 训练需要组合多个并行维度——Data Parallel（DP）、Pipeline Parallel（PP）、Tensor Parallel（TP）、Context Parallel（CP）和 Expert Parallel（EP）。对于 671B MoE 模型在 128 GPU 上的部署，合法策略空间超过 30 种配置。手动探索每次尝试耗时 10 分钟以上，且存在 OOM 风险。

现有自动并行系统的局限性：

- **Alpa** [1] 通过 ILP 自动搜索 intra/inter-op 并行，但其代价模型假设计算和通信串行执行，且不支持 MoE/EP。
- **AIConfigurator** [2] 添加了 MoE EP AllToAll 建模和分层 NVLink+IB 代价模型，但依赖硬件性能数据库，且不建模引擎特定的运行时优化。
- **Galvatron** [3] 使用 DP 搜索和解析代价模型，支持异构集群但缺少 MoE/EP 和 Context Parallelism。

AutoParallel 通过纯解析、引擎感知的代价模型解决这些问题，支持完整的并行空间（DP、PP、TP、CP、EP），覆盖训练和推理场景。

## 2. 方法

### 2.1 代价模型架构

AutoParallel 使用两层架构（完整公式推导见 [DESIGN.md](DESIGN.md)）：

- **Layer 1（解析层）**：Peak FLOPS roofline + alpha-beta 通信模型。零配置，开箱即用。
- **Layer 2（Profiling 层）**：可选的 GEMM + 集合通信实测数据，使用对数空间插值逐点替代解析估算。

每个查询点独立 fallback——有 profiling 数据就用插值，否则退回解析公式。不是全有全无。

### 2.2 核心贡献

#### 贡献 1：EP AllToAll-FFN Overlap 建模

**观察**：Megatron 的 MoE token dispatcher 将 AllToAll dispatch/combine 与 expert FFN 计算并发执行。先前的模型（Alpa、AIConfigurator）假设串行执行，系统性地高估 EP 通信代价。

**影响**：串行假设使优化器偏向高 TP 策略（减少 EP 通信量），但实际上低 TP + 高 EP 配置在有 overlap 时更快。

**建模**：

```
// Megatron 引擎（overlap）：
T_MoE = max(T_ep, T_routed_ffn)

// FSDP 引擎（串行）：
T_MoE = T_ep + T_routed_ffn
```

通过 `EngineConfig` 抽象控制，使代价模型具备引擎感知能力。

**验证**：在 GLM-5.1（128×H200，BS=16）上，串行模型错误地将 TP=16 排为 #1。添加 overlap 建模后，TP=8 CP=2 正确排名 #1，与实测 step 时间一致。详见 [BENCHMARK.md](BENCHMARK.md)。

#### 贡献 2：TP NVLink 带宽退化

**观察**：TP > 4 时，多个 AllReduce ring 竞争 NVLink 带宽。实测 TP=8 的有效带宽约为 TP=4 的 70%（见 [BENCHMARK.md](BENCHMARK.md)）。

**建模**：对大 TP 组施加退化因子：

```
B_eff = B_nvlink / sqrt(tp / 4)    当 tp > 4
```

该校正为 Megatron 特有（`EngineConfig` 中 `tp_bw_degradation=True`）。FSDP 和 SGLang 引擎不表现此模式。

#### 贡献 3：MLA 感知的 Context Parallelism

**观察**：MLA（Multi-head Latent Attention）将 KV 压缩到极低维度。CP ring attention 在相邻 rank 间传输 KV——使用 MLA 时，传输量降至**标准 MHA 的 3.5%**：

```
KV_dim(MLA)  = kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576
KV_dim(MHA)  = num_kv_heads × head_dim × 2 = 64 × 128 × 2 = 16,384
比率 = 576 / 16,384 ≈ 3.5%
```

这使 CP=2 或 CP=4 在 MLA 模型（DeepSeek-V2/V3、GLM-5.1）上几乎无开销，使先前模型会惩罚的策略变得可行。

### 2.3 引擎感知抽象

不同训练引擎有不同的运行时行为。我们将其参数化为 `EngineConfig`：

| 参数 | Megatron | FSDP | SGLang |
| --- | --- | --- | --- |
| EP AllToAll 与 FFN 重叠 | 是 | 否 | 是 |
| TP 带宽退化（tp>4） | 是 | 否 | 否 |
| MoE 激活因子 | 18 | 14 | 14 |

同一套代价模型可以为不同引擎产出正确推荐，无需修改代码。

### 2.4 推理模型

推理场景下，AutoParallel 使用 roofline 模型区分两个阶段：

- **Prefill**（计算密集）：`T = F_compute / F_peak + T_comm`
- **Decode**（内存带宽密集）：`T = max(F/F_peak, (W+KV)/B_hbm) + T_comm`

多实例部署最大化聚合吞吐量：
`TPS_aggregate = n_instances × TPS_decode`，其中 `n_instances = N_gpu / (TP × PP)`。

PP 在推理模式默认禁用（stage 串行执行，只增加延迟不减少显存）。

## 3. 实验评估

详细实验结果见 [BENCHMARK.md](BENCHMARK.md)。摘要：

| 指标 | 结果 |
| --- | --- |
| 显存估计误差 | <5%（GLM-5.1，128×H200） |
| 排序准确率（含 overlap） | 正确识别最优策略 |
| 排序准确率（不含 overlap） | 错误——偏向高 TP |
| 搜索时间 | <1 秒（纯解析） |
| GPU 依赖 | 无（仅读取 config.json） |
| SGLang Cookbook 对齐 | TP/EP 推荐与官方 cookbook 一致 |
| 推理显存误差 | 2.1%（GLM-5.1 SGLang 实测） |

## 4. 与先前工作的对比

| 特性 | Alpa [1] | AIConfigurator [2] | Galvatron [3] | **AutoParallel** |
| --- | --- | --- | --- | --- |
| 通信模型 | α-β | α-β + 硅片数据库 | α-β | α-β + 分层 EP |
| MoE EP 支持 | 无 | AllToAll | 无 | AllToAll + **overlap** |
| Context Parallel | 无 | 有 | 无 | 有（**MLA 感知**） |
| 引擎感知 | 无 | 无 | 无 | **EngineConfig** |
| TP 带宽退化 | 无 | 无 | 无 | **sqrt(tp/4) 模型** |
| 推理模式 | 无 | 有 | 无 | **Roofline**（prefill+decode） |
| 依赖 | ILP 求解器 | 硬件 benchmark 数据库 | DP 求解器 | **零依赖**（纯解析） |
| 训练支持 | 有 | 无 | 有 | **有** |

## 5. 局限性与未来工作

1. **激活估计**：使用经验因子而非精确的逐层计算。
2. **通信拓扑**：假设均匀的 NVLink/IB；未建模 NVSwitch 或非对称拓扑。
3. **序列并行**：未显式建模 Megatron 的 Sequence Parallelism 对激活内存的优化。
4. **多阶段工作流**：未建模 RLHF/GRPO 多阶段资源分配。
5. **FP8 量化**：当前仅建模 BF16 权重；FP8 模型需要扩展。
6. **Profiling 校准**：少量 benchmark 数据点可进一步提升精度。

## 参考文献

1. Zheng, L. et al. "Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning." OSDI, 2022.
2. Xu, T. et al. "AIConfigurator: Lightning-Fast Configuration Optimization for Multi-Framework LLM Serving." 2025.
3. Miao, X. et al. "Galvatron: Efficient Transformer Training over Multiple GPUs Using Automatic Parallelism." VLDB, 2023.
4. Williams, S. et al. "Roofline: An Insightful Visual Performance Model for Multicore Architectures." CACM, 2009.
5. Wu, R. et al. "Rethinking Dynamic Networks and Heterogeneous Computing with Automatic Parallelization." APNET, 2025.
