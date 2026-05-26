# 发明专利技术交底书

**创新名称**：一种基于分层代价模型的引擎感知大模型训推自动并行方法

**关键词**：并行策略推荐、分布式训练、代价模型、MoE、MLA、显存估算

**申请说明（理由）**：

大语言模型（LLM）的分布式训练需要在 DP、PP、TP、CP、EP 五个并行维度中选择最优组合。当前业界主要依赖工程师经验手动配置，每次尝试需要 10 分钟以上的实际运行且存在 OOM 风险；现有自动并行系统（Alpa、Galvatron 等）不支持 MoE 专家并行和 MLA 架构，且不区分不同训练引擎的运行时差异，导致推荐结果不准确。

本方案提出基于分层代价模型的自动并行策略推荐方法，包含三项核心创新：（1）采用解析模型+可选 Profiling 插值的两层架构，Layer 1 无需 GPU 即可工作，Layer 2 通过实测数据的对数空间插值逐点替代解析估算提升精度；（2）根据引擎运行时行为差异对 EP AllToAll 与 FFN 计算的 overlap 进行差异化建模，并量化 TP>4 时 NVLink 带宽退化现象，提出 sqrt 校正公式；（3）根据 MLA 架构自适应计算 CP 通信量，将建模误差从 30 倍降至 3.5%。

本方案已在 128×H200 GPU 集群上验证，显存估计误差 <5%。在 757B MoE 模型的实测中，本方案推荐的最优策略（TP=8 CP=2）step time 为 37.6s，而未采用分层代价模型和引擎感知建模时推荐的 TP=16 策略 step time 约 50s，性能差距超过 30%。本方案 Layer 1 无需 GPU，秒级完成推荐，即将集成到内部 LLM 训练平台，可显著降低大模型训练的并行配置调试成本。

---

## 0、术语解释

- **DP (Data Parallel)**：数据并行，将训练 batch 切分到多张 GPU 上并行处理
- **PP (Pipeline Parallel)**：流水线并行，将模型的不同层分配到不同 GPU 上串行执行
- **TP (Tensor Parallel)**：张量并行，将单层的权重矩阵按列/行切分到多张 GPU 上
- **CP (Context Parallel)**：上下文并行，将输入序列沿 token 维度切分到多张 GPU，通过 Ring Attention 交换 KV
- **EP (Expert Parallel)**：专家并行，将 MoE 模型的不同专家分配到不同 GPU 上
- **MoE (Mixture of Experts)**：混合专家模型，每个 token 仅激活部分专家，提升模型容量的同时控制计算量
- **MLA (Multi-head Latent Attention)**：多头潜在注意力，将 KV 投影压缩到低维潜在空间，大幅减少 KV Cache 显存
- **AllToAll**：集合通信原语，每个 GPU 向其他所有 GPU 发送不同数据，用于 EP 中的 token 分发和收集
- **AllReduce**：集合通信原语，对所有 GPU 上的数据进行求和并广播结果，用于 TP 中的梯度/激活聚合
- **NVLink**：NVIDIA GPU 节点内高速互联总线，带宽 200-450 GB/s
- **IB (InfiniBand)**：节点间高速网络互联，带宽 25-50 GB/s
- **Roofline 模型**：性能分析模型，计算受限于峰值算力和内存带宽中的瓶颈
- **alpha-beta 模型**：通信代价模型，T = α（启动延迟）+ 数据量/带宽

## 1、应用本方案的产品

大语言模型分布式训练与推理系统（如 Megatron-LM、FSDP、SGLang 等框架的用户），包括但不限于 RLHF/GRPO 对齐训练、SFT 微调、大规模推理服务部署。

## 2、本方案的背景是什么？

大规模 LLM 训练需要组合多个并行维度（DP、PP、TP、CP、EP）。对于 757B MoE 模型在 128 张 GPU 上的部署，合法策略空间超过 30 种配置。手动探索每种配置需要 10 分钟以上的实际运行时间，且存在 OOM（显存溢出）风险，一次 OOM 即浪费数十张 GPU 的等待时间。

现有方案的不足：
- 人工经验法：依赖工程师经验逐个尝试，耗时且不可靠，尤其对 MoE+MLA 等新架构缺乏经验积累
- 现有自动并行系统（如 Alpa、Galvatron）不支持 MoE 专家并行或上下文并行，且不考虑不同训练引擎（Megatron、FSDP）的运行时行为差异

本方案即将集成到内部 LLM 训练平台，已在 128×H200 GPU 集群上验证。

## 3、行业内哪些竞争对手的业务、产品和本方案相关？

| 竞争对手 | 相关产品/项目 | 说明 |
|----------|-------------|------|
| Google/Alpa | Alpa | 基于 ILP+DP 的自动并行搜索系统（OSDI'22），面向 JAX |
| 北大/Galvatron | Galvatron | 基于动态规划的并行优化（VLDB'23），支持异构集群 |
| 清华/AIConfigurator | AIConfigurator | 算子级性能数据库 + 迭代级建模（2025），面向推理 |
| NVIDIA/DeepSpeed | Megatron-LM / DeepSpeed | 提供并行框架但依赖用户手动配置策略 |
| HPC-AI Tech | ColossalAI | 提供并行训练框架，含 FP8 支持，但无自动策略推荐 |

## 4、本方案是否有敏感的部分不适合作为专利申请公开？

无。

## 5、详细介绍与本方案相似的方案及其缺点

**方案一：Alpa 自动并行搜索（Google, OSDI'22）**

Alpa 将并行搜索分为 intra-op（ILP 求解 TP/数据切分）和 inter-op（DP 搜索 PP 划分）两阶段。其代价模型采用 alpha-beta 通信模型估算通信代价。

缺点：
- 代价模型假设计算与通信**串行执行**，不建模引擎级别的 overlap 优化（如 Megatron 的 EP AllToAll 与 FFN 计算重叠），导致系统性高估 EP 通信开销
- **不支持 MoE 专家并行（EP）**和上下文并行（CP），无法处理 DeepSeek-V3、GLM-5.1 等新一代 MoE 模型
- 依赖 ILP 求解器，搜索时间较长
- 项目已归档，不再维护

**方案二：Galvatron 自动并行优化（北大, VLDB'23）**

Galvatron 使用动态规划搜索并行策略，支持逐层异构并行配置，并通过 profiling 采集硬件性能数据。

缺点：
- **不支持 MoE 专家并行（EP）**和上下文并行（CP）
- 不建模 MLA（Multi-head Latent Attention）对通信量的影响
- 需要在目标 GPU 上运行 profiling，无法零依赖推荐
- 不区分不同训练引擎的运行时行为

**方案三：AIConfigurator（清华, 2025）**

AIConfigurator 建立算子级硬件性能数据库，通过分层 NVLink+IB 代价模型估算 EP AllToAll 通信代价。

缺点：
- **依赖硬件性能数据库**（silicon DB），需要预先针对特定 GPU 型号构建数据库
- 不建模引擎特定的**运行时优化**（如 Megatron 的 EP AllToAll-FFN overlap）
- **仅面向推理**，不支持训练场景的显存估算（梯度、优化器、激活等）
- 不建模 MLA 对 CP 通信量的影响

**三种方案的共同缺点**：

1. 均不区分不同训练引擎（Megatron vs FSDP）的运行时行为差异，使用统一的代价模型，导致对不同引擎给出相同（往往错误）的推荐
2. 均不建模 MLA 架构对 CP ring attention 通信量的 30 倍压缩效果，导致系统性高估 CP 代价
3. 均不建模 TP 在大规模（TP>4）下的 NVLink 带宽退化现象

## 6、本方案技术方案的详细阐述

### 6.1、系统结构与流程

*（附图：系统架构图 architecture.png 和代价模型流程图 cost_model.png，红色标注创新部分）*

本方案提出一种**引擎感知的分层代价模型**，用于自动推荐大语言模型分布式并行策略。系统输入为模型架构配置（HuggingFace config.json）和 GPU 集群规格，输出为按吞吐量排序的合法并行策略列表。主要流程如下：

**第一阶段：输入解析与策略空间枚举**

（1）从模型配置文件自动识别架构特征（Dense/MoE/MLA/GQA/Lightning Attention），提取隐藏维度 H、层数 L、注意力头数、专家数、KV LoRA 秩等参数，构建模型规格 ModelSpec；

（2）根据 GPU 型号预设（H200/H100/A100 等）自动加载硬件参数（峰值算力、NVLink 带宽、IB 带宽、HBM 带宽、显存容量），构建硬件规格 HardwareSpec；

（3）根据指定的训练引擎（Megatron/FSDP/SGLang），加载引擎配置 EngineConfig，其中参数化了不同引擎的运行时行为差异（详见创新点 1）；

（4）在五维并行空间 (DP, PP, TP, CP, EP) 中枚举所有满足硬件约束和模型约束的合法策略组合。约束条件包括：
- 恒等约束：DP × PP × TP × CP = N_gpu
- 注意力头整除：n_heads mod TP = 0，n_kv_heads mod TP = 0
- 专家整除：n_experts mod EP = 0，其中 EP = TP × CP
- 流水线调度约束（1F1B）：batch_size / DP ≥ 2 × PP
- 序列整除：max_tokens mod CP = 0

**第二阶段：分层显存估算与可行性过滤**

对每个候选策略，估算单 GPU 显存占用，包括以下组件：

（5）**模型参数显存**：根据模型架构（MHA/MLA/Dense FFN/MoE FFN）计算每层参数量，按 TP/PP/EP 分片后乘以数据类型字节数。其中 MLA 架构的部分投影矩阵（Q 压缩、KV 压缩）不按 TP 分片（replicated），需单独计算；

（6）**梯度与优化器显存**：DDP 梯度通信缓冲区（不按 DP 分片）、梯度分片（按 DP 分片）、Adam 优化器状态（12 bytes/param/DP，支持 CPU offload 转移到主机内存）；

（7）**激活显存**：与每 GPU 处理的 token 数和隐藏维度成正比，乘以引擎相关的激活因子（Megatron MoE 层为 18，FSDP 为 14，Dense 层统一为 10）；

（8）**CUDA 上下文**：固定 8GB + 每个 NCCL 通信组 1GB；

（9）总显存 = (模型 + 梯度缓冲 + 梯度分片 + 优化器 + 激活) × 1.05（碎片系数）+ CUDA 上下文。超过 GPU 显存标记为 OOM，超过主机 CPU 内存标记为 CPU OOM，过滤不可行策略。

**第三阶段：引擎感知吞吐量代价模型（核心创新）**

对每个可行策略，使用分层代价模型估算训练吞吐量：

**【创新点 1：引擎感知的 EP AllToAll-FFN Overlap 建模】**

（10）对 MoE 层的 EP AllToAll 通信代价，根据引擎配置采用不同的建模方式：
- Megatron 引擎：AllToAll dispatch/combine 与 expert FFN 计算并发执行，代价取两者的较大值：

  T_MoE = max(T_ep, T_routed_ffn)

- FSDP 引擎：AllToAll 与 FFN 串行执行，代价为两者之和：

  T_MoE = T_ep + T_routed_ffn

这一差异通过 EngineConfig 数据结构参数化，使同一套代价模型能为不同引擎产出正确推荐。先前的系统（Alpa、AIConfigurator、Galvatron）均假设串行执行，系统性地高估 EP 通信代价，导致偏向高 TP 策略。

（11）EP AllToAll 通信代价采用**分层建模**：区分节点内（NVLink）和跨节点（IB）两部分流量。当 EP 跨节点时，跨节点流量比例为 f_inter = 1 - (G_per_node - 1)/(EP - 1)，并引入拥塞因子 sqrt(n_ep_nodes)：

  T_ep = α_ib + V_total × f_intra / B_nvlink + V_total × f_inter / B_ib × sqrt(EP / G_per_node)

**【创新点 2：TP NVLink 带宽退化建模】**

（12）当 TP > 4 时，多个 AllReduce ring 竞争 NVLink 带宽，实测有效带宽下降。本方案引入退化因子：

  B_eff = B_nvlink / sqrt(TP / 4)    当 TP > 4

该校正仅在 Megatron 引擎下启用（EngineConfig.tp_bw_degradation=True），FSDP 和 SGLang 引擎不表现此模式。实测验证：H200 上 TP=8 的有效带宽为 TP=4 的约 70%，与 1/sqrt(2) ≈ 0.707 一致。

**【创新点 3：MLA 感知的 Context Parallelism 代价计算】**

（13）CP 使用 Ring Attention，相邻 rank 之间传输 KV。本方案根据注意力架构自适应计算 KV 传输维度：
- 标准 MHA：d_kv = n_kv_heads × d_head × 2
- MLA 架构：d_kv = kv_lora_rank + qk_rope_head_dim

以 GLM-5.1 为例：MLA 的 d_kv = 512 + 64 = 576，而标准 MHA 的 d_kv = 64 × 128 × 2 = 16384，**MLA 的 CP 通信量仅为标准 MHA 的 3.5%**。这使 CP=2 或 CP=4 在 MLA 模型上几乎零开销。实测验证：GLM-5.1 上 CP=4 vs CP=8 的 step time 差异 <2%。

先前的系统不区分 MHA 和 MLA，使用统一的 KV 维度计算 CP 通信量，导致对 MLA 模型过度惩罚 CP 策略。

（14）**TP AllReduce 通信代价**：采用 Ring AllReduce 建模，消息体积 = 2 × (TP-1)/TP × H × T × 2 bytes。对于 MoE 模型，每层仅 attention 部分需要 TP AllReduce（FFN 走 EP AllToAll），比 Dense 模型少一次 AllReduce。

（15）**PP P2P 通信代价**：流水线 stage 间传输激活值，代价 = α_ib + T_mb/CP × H × 2 / B_ib。

（16）**每层时间汇总**（前向 1x + 反向 2x = 3x）：
- Megatron 引擎：T_layer = 3 × [T_attn_compute + T_tp + max(T_ep, T_ffn) + T_shared + T_cp]
- FSDP 引擎：T_layer = 3 × [T_total_compute + T_tp + T_ep + T_cp]

（17）**Pipeline Schedule 建模**：采用 1F1B 调度，总时间 = (PP + n_mb - 1) × T_stage，其中 n_mb = batch_size/DP。Pipeline bubble 比例 = (PP-1)/(PP + n_mb - 1)。

（18）**吞吐量得分**：Score = DP × n_mb / T_total，按得分降序排序输出 Top-K 策略。

**第四阶段（可选）：Profiling 插值精度提升**

（19）Layer 1 的解析模型假设 GPU 工作在峰值算力——对于小矩阵 GEMM 和小消息通信，实际利用率显著低于理论值。Layer 2 通过实测数据替代解析公式：
- GEMM Profiling：遍历 (M, N, K) 网格测量实际 GEMM 性能
- 通信 Profiling：测量不同规模的 AllReduce/AllToAll 实际带宽

（20）采用**对数空间线性插值**：给定两个测量点 (M₁, t₁) 和 (M₂, t₂)，对查询点 M_q 进行对数空间插值。选择对数空间的原因是 GEMM 性能与矩阵维度的关系近似幂律。每个查询点独立 fallback——有 profiling 数据用插值，否则退回解析公式，不是全有全无。

**推理模式扩展**

（21）推理场景采用 **Roofline 模型**，区分两个阶段：
- Prefill（计算密集型）：T = F_compute / F_peak + T_comm
- Decode（内存带宽密集型）：T = max(F/F_peak, (W+KV)/B_hbm) + T_comm

（22）多实例部署最大化聚合吞吐量：TPS_aggregate = n_instances × TPS_decode，其中 n_instances = N_gpu / (TP × PP)。推理模式默认禁用 PP（各 stage 串行执行，只增加延迟不减少显存）。

### 6.2、是否还有其他解决方案？

- 本方案的代价模型目前采用解析公式 + 可选 profiling 插值。替代方案包括：使用机器学习模型（如 XGBoost、神经网络）从大量实测数据中学习代价预测函数，但需要大量训练数据且可解释性差。
- 显存估算中的激活内存目前使用经验因子。替代方案是对每个算子逐一追踪激活张量的生命周期，精度更高但实现复杂度大幅增加。
- EP AllToAll 的分层建模目前假设均匀的 NVLink/IB 拓扑。替代方案是引入完整的网络拓扑图（含 NVSwitch、多级交换机），但增加了用户配置的复杂度。

### 6.3、如何克服第 5 点中的缺点，以及本方案能够达到的技术效果

**克服方式**：

1. **克服"串行通信假设"缺点**：通过引擎感知的 EngineConfig 抽象，对不同引擎（Megatron/FSDP/SGLang）使用不同的 EP 代价公式（overlap vs 串行），避免一刀切的串行假设导致的排序错误。

2. **克服"不支持 MoE/EP/CP"缺点**：完整建模五维并行空间 (DP, PP, TP, CP, EP)，支持 MoE 专家并行的分层 AllToAll 通信建模，支持 MLA 感知的 CP 通信量计算。

3. **克服"依赖硬件数据库"缺点**：采用两层架构，Layer 1 纯解析模型零依赖即可工作，Layer 2 的 profiling 数据仅作为可选精度提升手段。

4. **克服"不区分引擎"缺点**：通过 EngineConfig 参数化引擎差异（EP overlap、TP 带宽退化、激活因子），同一套代价模型为不同引擎产出不同且正确的推荐。

**技术效果**：

| 指标 | 结果 |
|------|------|
| 显存估计误差 | <5%（GLM-5.1 757B MoE，128×H200 实测） |
| 推理显存误差 | 2.1%（GLM-5.1 SGLang 部署实测） |
| 排序准确性 | 正确识别最优策略（TP=8 CP=2 优于 TP=16，实测 37.6s vs ~50s） |
| CP 代价验证 | MLA 模型 CP=4 vs CP=8 差异 <2%（验证 3.5% 传输比） |
| TP 退化验证 | TP=8 有效带宽 = TP=4 的 69%（与 1/sqrt(2) 模型一致） |
| BailingMoE CP 扩展 | CP=8 比 CP=4 快 2.25 倍（64 GPU 实测） |
| 搜索时间 | <1 秒（纯解析模型） |
| GPU 依赖 | 零（仅读取 config.json） |

## 7、请提炼出本方案的关键技术创新点

1. **引擎感知的 EP AllToAll-FFN Overlap 代价建模**：根据训练引擎的实际运行时行为（Megatron 重叠 vs FSDP 串行），对 MoE 层的 EP AllToAll 通信代价采用不同的建模公式（max vs sum），通过 EngineConfig 参数化，解决了先前系统因统一串行假设导致的排序错误问题。

2. **TP NVLink 带宽退化校正模型**：发现并量化了 TP>4 时多个 AllReduce ring 竞争 NVLink 带宽导致有效带宽下降的现象，提出 B_eff = B_nvlink / sqrt(TP/4) 校正公式，实测吻合度达 97%。

3. **MLA 感知的 Context Parallelism 通信量自适应计算**：根据注意力架构（MHA vs MLA）自动选择 KV 传输维度计算公式，使 MLA 模型的 CP 通信量建模从标准 MHA 的 100% 降至 3.5%，正确反映了 CP 在 MLA 模型上几乎零开销的特性。

## 8、本方案是否涉及软件开源？

本方案的核心创新在于**方法层面**——引擎感知的代价建模方法、TP 带宽退化校正公式、MLA 感知的 CP 通信量计算方法。这些方法创新独立于具体的软件实现。

本方案的一个参考实现已在 GitHub 开源（Apache License 2.0）。开源的是软件实现代码，专利保护的是上述技术方法本身。开源实现使用 Python 标准库，核心推荐算法无第三方依赖。
