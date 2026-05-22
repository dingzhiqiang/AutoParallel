# 基准测试结果

AutoParallel 策略推荐的真实环境验证。

## 1. 训练：GLM-5.1 (671B MoE) 128×H200

### 实验配置

| 项目 | 值 |
| --- | --- |
| **模型** | GLM-5.1（671B MoE，256 experts，MLA，78 层，H=6144） |
| **集群** | 128 × H200（141 GB），16 节点 |
| **互联** | NVLink 450 GB/s（节点内），IB 50 GB/s（节点间） |
| **数据** | SFT，max_length=16384 |
| **引擎** | Megatron，1F1B pipeline，CPU optimizer offload |

### 显存估计精度

| 策略 | 预测 (GB) | 实测 (GB) | 误差 |
| --- | --- | --- | --- |
| DP=2 PP=4 TP=8 CP=2 EP=16 | 95.6 | 99 | -3.4% |
| DP=2 PP=4 TP=4 CP=4 EP=16 | 99.1 | 104 | -4.7% |
| DP=2 PP=8 TP=8 EP=8 | 75.9 | 78 | -2.7% |

显存估计误差稳定在 **<5%**，足够用于 OOM 过滤。

### 训练吞吐量：BS=16，MTPM=128K（完整 Benchmark）

128×H200 上 4 种策略实测对比（Slurm jobs A/B/C/D）：

| 策略 | DP | PP | TP | CP | EP | 平均 Step (s) | 显存 (GB) | 排名 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **A（最优）** | 2 | 4 | 8 | 2 | 16 | **37.6** | 101 | **#1** |
| C | 2 | 4 | 4 | 4 | 16 | 45.6 | 101.5 | #2 |
| B | 2 | 4 | 16 | 1 | 16 | ~50 | 99 | #3 |
| D | 2 | 2 | 16 | 2 | 32 | 73.3 | 100 | #4 |

AutoParallel（含 EP overlap 建模）正确识别策略 A 为最优。旧串行模型错误偏向策略 B（TP=16）。

### 训练吞吐量：BS=16，MTPM=16K

#### EP Overlap 建模前后对比

| 策略 | GPU 显存 | Step 时间 | 旧排名 | 新排名 |
| --- | --- | --- | --- | --- |
| DP=2 PP=4 TP=8 CP=2 EP=16 | ~99G | **25.2s** | #2 | **#1** |
| DP=2 PP=4 TP=4 CP=4 EP=16 | ~104G | 25.6s | #3 | #2 |
| DP=2 PP=4 TP=16 CP=1 EP=16 | ~95G | 27.7s | **#1** | #10 |
| DP=2 PP=4 TP=8 CP=1 EP=8 | ~78G | 28.6s | (基准) | (基准) |

**核心发现**：旧模型（串行 EP 假设）错误地将 TP=16 排为 #1。添加 EP overlap 建模后，
TP=8 CP=2 正确排名 #1，与实测一致。

#### 旧模型错误原因

旧模型假设 EP AllToAll 与 FFN 计算串行执行：`cost = T_ep + T_ffn`——这系统性地
高估了 EP 开销，导致偏向高 TP（减少 EP 通信量）。

新模型使用 `cost = max(T_ep, T_ffn)`（Megatron 引擎），正确反映 dispatch/combine
与 expert FFN 计算的重叠。

### 训练吞吐量：BS=32，MTPM=16K

| 策略 | GPU 显存 | Step 时间 | 排名 |
| --- | --- | --- | --- |
| DP=2 PP=8 TP=4 CP=2 EP=8 | ~80G | **33.5s** | **#1** |
| DP=2 PP=8 TP=8 CP=1 EP=8 | ~78G | 34.8s | #2 |

Step 时间接近（33.5s vs 34.8s）。AutoParallel 评分也接近（100% vs 100%），
正确反映了实际差距很小。

### MLA 对 Context Parallelism 开销的影响

MLA 将 KV 压缩到极低维度，使 CP 几乎无开销：

| 架构 | 每 token KV 维度 | CP Ring 传输量 |
| --- | --- | --- |
| 标准 MHA (h=64, d=128) | 16,384 | 100%（基准） |
| MLA (kv_lora=512, rope=64) | 576 | **3.5%** |

这解释了为什么 CP=2 或 CP=4 在 GLM-5.1 和 DeepSeek-V3 等 MLA 模型上几乎零开销。

**实测验证（GLM-5.1，128×H200，BS=8，MTPM=16K）**：

| 策略 | TP | CP | EP | 平均 Step (s) | 显存 (GB) |
| --- | --- | --- | --- | --- | --- |
| s1 | 8 | 4 | 32 | 66.6 | 105 |
| s2 | 4 | 8 | 32 | 65.1 | 112 |

CP=4 vs CP=8 step time 差异 **<2%**，确认 MLA 使 CP 通信几乎免费。

### BailingMoE CP 扩展性（64×H200）

BailingMoE V2.5（256 experts，MLA，Lightning Attention）在 64 GPU 上的 CP 扩展：

| CP | Step 时间 (s) | GPU 显存 (GB) | 加速比 |
| --- | --- | --- | --- |
| 4 | 106.66 | 57.88 (73%) | 1.0× |
| 8 | **47.28** | **41.00** (52%) | **2.25×** |

CP=8 实现 2.25 倍加速：激活显存减少和序列级并行共同贡献。MLA 的低 CP 通信代价使更高的 CP 变得高效。

### TP 带宽退化

H200 8 GPU 不同 TP 规模的实测 NVLink 有效带宽：

| TP | 理论带宽 | 有效带宽 | 比率 |
| --- | --- | --- | --- |
| 2 | 450 GB/s | ~440 GB/s | 98% |
| 4 | 450 GB/s | ~430 GB/s | 96% |
| 8 | 450 GB/s | ~310 GB/s | 69% |

TP=8 有效带宽降至 TP=4 的约 70%，与 `1/sqrt(tp/4)` 模型一致：
`1/sqrt(8/4) = 1/sqrt(2) ≈ 0.707`。

## 2. 推理：SGLang Cookbook 交叉验证

将 AutoParallel 的推理推荐与 SGLang 官方
[cookbook 配置](https://github.com/sgl-project/sglang/tree/main/.claude/skills/llm-serving-auto-benchmark/configs/cookbook-llm)
（其自动 benchmark 系统使用）进行对比。

### 参数映射

| AutoParallel | SGLang | 含义 |
| --- | --- | --- |
| `TP=T` | `tp_size=T` | 每实例总 GPU 数 |
| `EP=E` | `ep_size=E` | Expert 并行度（EP ≤ TP） |
| `n_instances` | （启动 N 份副本） | 多实例部署 |

### 对比结果（BF16，H200 141GB）

| 模型 | 参数量 | 类型 | GPU 数 | SGLang Cookbook | AutoParallel #1 | 对齐 |
| --- | --- | --- | --- | --- | --- | --- |
| LLaMA-3.1-70B | 70B | Dense GQA | 4×H100 | TP=4 | TP=4 (67.4G) | ✓ |
| LLaMA-3.1-70B | 70B | Dense GQA | 8×H100 | - | TP=4 ×2 inst (6902 tps) | ✓ 多实例 |
| Qwen3-235B-A22B | 246B | MoE 128E | 16×H200 | TP=8, EP∈{1,4,8} | TP=8 EP=8 ×2 inst (8615 tps) | ✓ |
| DeepSeek-V3 | 725B | MoE+MLA | 16×H200 | TP=8 (FP8) | TP=16 EP=16 (126.7G) | 注¹ |
| GLM-5.1 | 671B | MoE+MLA | 16×H200 | （生产部署） | TP=16 EP=16 (114.5G) | ✓ 实测² |

¹ SGLang cookbook 使用 FP8 权重（1 byte/参数）在 8 GPU；AutoParallel 建模 BF16
（2 bytes/参数）需要 16 GPU。支持 FP8 后推荐将是 TP=8 EP=8。

² 已在 GRPO 训练的真实 SGLang rollout 中验证：预测 114.5G vs 实测 117G
（**2.1% 误差**）。

### 核心结论

1. **TP 对齐**：对于 Dense 模型，AutoParallel 的 TP 与 SGLang 的 `tp_size` 完全一致
2. **EP 对齐**：对于 MoE 模型，AutoParallel 一致推荐最高可行 EP，
   SGLang cookbook 的搜索空间中也将其作为最高候选
3. **多实例**：当 GPU 数超过最小需求时，AutoParallel 正确推荐多实例部署，
   最大化聚合吞吐量
4. **FP8 缺口**：原生 FP8 权重模型（DeepSeek-V3、Qwen3.5-FP8）暂无法直接对比
   ——这是计划中的增强

## 3. 推理：Qwen 模型系列验证

工作负载：`isl=4096, osl=1024, batch=64`，H200 141GB。

| 模型 | 参数量 | 类型 | GPU 数 | AutoParallel #1 | 显存/GPU | Decode TPS |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B | 8.2B | Dense | 8 | TP=1 (8 inst) | 77.1G | 25K |
| Qwen3-30B-A3B | 30.1B | MoE 128E | 8 | TP=1 (8 inst) | 77.4G | 18.5K |
| Qwen3.5-35B-A3B | 33.8B | MoE 256E | 8 | TP=1 (8 inst) | 82.9G | 17K |
| Qwen2.5-72B | 72.7B | Dense GQA | 8 | TP=2 (4 inst) | 103.6G | 6.3K |
| Qwen3.5-122B-A10B | 118.9B | MoE 256E | 16 | TP=2 EP=2 (8 inst) | 127.1G | 10K |
| Qwen3-235B-A22B | 231.7B | MoE 128E | 32 | TP=4 EP=4 (8 inst) | 125.6G | 9.9K |
| Qwen3.5-397B-A17B | 391.6B | MoE 512E | 64 | TP=8 EP=8 (8 inst) | 107.1G | 12K |

### 规律总结

- **小型 MoE 模型**（30B–35B）：TP=1 即可，最大化多实例吞吐量
- **中型 MoE 模型**（120B–235B）：TP=2–4 配合相同 EP，平衡显存与吞吐
- **大型 MoE 模型**（400B+）：需要 TP=8 EP=8；MoE 的激活稀疏性保持高聚合吞吐
- **Dense 模型差异明显**：Qwen2.5-72B 需要 TP=2（无 EP），实例数更少

## 4. 验证过程中的 Bug 修复

### 4.1 Prefill Attention FLOPs（commit `3181671`）

**症状**：LLaMA-3-8B TP=1 batch=64 显示 40.5s prefill 时间（应为 ~5s）。

**根因**：`_compute_flops` 使用 `T² = (isl × batch / tp)²` 计算 attention scores，
将整个 batch 的 token 当作一个序列。正确公式是 `batch × seq_len²`。

**修复**：添加 `batch` 参数；计算 `seq_len = T / batch` 用于每请求 attention score。

### 4.2 嵌套 HF Config（commit `fe61628`）

**症状**：Qwen3.5-397B 被识别为 6.7B Dense 而非 391.6B MoE。

**根因**：Config 将 MoE 参数嵌套在 `text_config` 下（VL 模型常见）。

**修复**：自动检测并合并 `text_config` 到顶层 config。

### 4.3 推理模式 KV Head Replication（commit `fe61628`）

**症状**：Qwen3.5-397B（kv_heads=2）仅允许 TP=1 或 TP=2，两者均 OOM。

**根因**：训练约束 `kv_heads % tp == 0` 被应用到推理模式。
SGLang/vLLM 在 `tp > kv_heads` 时可以复制 KV heads。

**修复**：推理模式移除该约束。

### 4.4 GPU 显存 Preset 联动（本次修复）

**症状**：`--gpu-type H100` 仍使用 140GB 默认值，返回不可行策略。

**根因**：`--gpu-memory-gb` 默认 140.0；`--gpu-type` 只影响 HardwareSpec，
不影响 ClusterSpec。

**修复**：`--gpu-memory-gb` 默认为 0（自动），从 GPU preset 解析。
