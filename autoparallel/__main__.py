"""Parallel strategy advisor for distributed LLM training.

Enumerate valid parallel strategies (DP, PP, TP, CP, EP) for a given model
and cluster, estimate per-GPU memory, and recommend the best configuration.

Usage::

    python -m autoparallel \\
        --model-path /storage/openpsi/models/GLM-5.1 \\
        --n-gpus 128 --gpu-memory-gb 140 \\
        --max-tokens-per-mb 131072 --max-length 16384

    python -m autoparallel \\
        --hidden-size 6144 --num-layers 78 --num-heads 64 \\
        --num-experts 256 --expert-intermediate-size 2048 \\
        --intermediate-size 12288 --vocab-size 154880 \\
        --kv-lora-rank 512 --q-lora-rank 2048 \\
        --n-gpus 128 --gpu-memory-gb 140
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .profile_data.loader import ProfileLookup


@dataclass
class HardwareSpec:
    """GPU hardware performance parameters for cost modeling."""

    gpu_flops: float = 990e12  # BF16 FLOPS
    bw_nvlink: float = 450e9  # NVLink unidirectional bandwidth (bytes/s)
    bw_ib: float = 50e9  # IB RDMA unidirectional bandwidth (bytes/s)
    hbm_bw: float = 3.35e12  # HBM bandwidth (bytes/s), for decode roofline
    alpha_nvlink: float = 5e-6  # NVLink latency (seconds)
    alpha_ib: float = 20e-6  # IB latency (seconds)
    gpu_memory_gb: float = 80.0  # HBM per GPU (GB)
    host_memory_gb: float = 1000.0  # CPU RAM per node (GB)


GPU_PRESETS: dict[str, HardwareSpec] = {
    "H200": HardwareSpec(
        gpu_flops=990e12,
        bw_nvlink=450e9,
        bw_ib=50e9,
        hbm_bw=4.8e12,
        gpu_memory_gb=141,
        host_memory_gb=1500,
    ),
    "H100": HardwareSpec(
        gpu_flops=990e12,
        bw_nvlink=450e9,
        bw_ib=50e9,
        hbm_bw=3.35e12,
        gpu_memory_gb=80,
        host_memory_gb=1000,
    ),
    "H800": HardwareSpec(
        gpu_flops=990e12,
        bw_nvlink=400e9,
        bw_ib=50e9,
        hbm_bw=3.35e12,
        gpu_memory_gb=80,
        host_memory_gb=1000,
    ),
    "A100": HardwareSpec(
        gpu_flops=312e12,
        bw_nvlink=300e9,
        bw_ib=25e9,
        hbm_bw=2.0e12,
        gpu_memory_gb=80,
        host_memory_gb=1000,
    ),
    "A800": HardwareSpec(
        gpu_flops=312e12,
        bw_nvlink=200e9,
        bw_ib=25e9,
        hbm_bw=2.0e12,
        gpu_memory_gb=80,
        host_memory_gb=1000,
    ),
}


@dataclass
class ModelSpec:
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 32
    intermediate_size: int = 11008
    vocab_size: int = 32000
    num_experts: int = 0
    num_experts_per_tok: int = 0
    expert_intermediate_size: int = 0
    n_shared_experts: int = 0
    kv_lora_rank: int = 0
    q_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    first_k_dense_replace: int = 0
    group_norm_size: int = 0
    tie_word_embeddings: bool = False
    dtype_bytes: int = 2  # bf16

    @classmethod
    def from_hf_config(cls, path: str) -> ModelSpec:
        with open(f"{path}/config.json") as f:
            cfg = json.load(f)
        # Some models (Qwen3.5-MoE, VL models) nest params under text_config
        if "text_config" in cfg and "hidden_size" not in cfg:
            cfg = {**cfg, **cfg["text_config"]}
        # Qwen3.5 uses shared_expert_intermediate_size instead of n_shared_experts
        n_shared = cfg.get("n_shared_experts", 0) or 0
        shared_inter = cfg.get("shared_expert_intermediate_size", 0) or 0
        if n_shared == 0 and shared_inter > 0:
            n_shared = 1
        return cls(
            hidden_size=cfg.get("hidden_size", 4096),
            num_layers=cfg.get("num_hidden_layers", 32),
            num_heads=cfg.get("num_attention_heads", 32),
            num_kv_heads=cfg.get(
                "num_key_value_heads", cfg.get("num_attention_heads", 32)
            ),
            intermediate_size=cfg.get(
                "intermediate_size",
                cfg.get("shared_expert_intermediate_size", 11008),
            ),
            vocab_size=cfg.get("vocab_size", 32000),
            num_experts=cfg.get(
                "n_routed_experts",
                cfg.get("num_experts", cfg.get("num_local_experts", 0)),
            )
            or 0,
            num_experts_per_tok=cfg.get("num_experts_per_tok", 0) or 0,
            expert_intermediate_size=cfg.get("moe_intermediate_size", 0) or 0,
            n_shared_experts=n_shared,
            kv_lora_rank=cfg.get("kv_lora_rank", 0) or 0,
            q_lora_rank=cfg.get("q_lora_rank", 0) or 0,
            qk_nope_head_dim=cfg.get("qk_nope_head_dim", 0) or 0,
            qk_rope_head_dim=cfg.get("qk_rope_head_dim", 0) or 0,
            v_head_dim=cfg.get("v_head_dim", 0) or 0,
            first_k_dense_replace=cfg.get("first_k_dense_replace", 0) or 0,
            group_norm_size=cfg.get("group_norm_size", 0)
            or cfg.get("layer_group_size", 0)
            or 0,
            tie_word_embeddings=cfg.get("tie_word_embeddings", False),
        )

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    @property
    def is_mla(self) -> bool:
        return self.kv_lora_rank > 0

    @property
    def has_group_norm(self) -> bool:
        return self.group_norm_size > 0

    @property
    def has_dense_replace(self) -> bool:
        return self.first_k_dense_replace > 0

    @property
    def head_dim(self) -> int:
        if self.is_mla:
            return self.qk_nope_head_dim + self.qk_rope_head_dim
        return self.hidden_size // self.num_heads

    def total_params_billion(self) -> float:
        return self._total_params() / 1e9

    def _total_params(self) -> int:
        H = self.hidden_size
        V = self.vocab_size
        L = self.num_layers

        embedding = V * H

        if self.is_mla:
            attn_per_layer = (
                H * self.q_lora_rank  # q_a_proj
                + self.q_lora_rank * self.num_heads * self.head_dim  # q_b_proj
                + H * (self.kv_lora_rank + self.qk_rope_head_dim)  # kv_a_proj_with_mqa
                + self.kv_lora_rank
                * self.num_heads
                * (self.qk_nope_head_dim + self.v_head_dim)  # kv_b_proj
                + self.num_heads * self.v_head_dim * H  # o_proj
            )
        else:
            head_dim = H // self.num_heads
            attn_per_layer = (
                H * self.num_heads * head_dim  # q_proj
                + H * self.num_kv_heads * head_dim * 2  # k_proj + v_proj
                + self.num_heads * head_dim * H  # o_proj
            )

        if self.is_moe:
            ffn_per_expert = self.expert_intermediate_size * H * 3  # gate+up+down
            ffn_per_layer = (
                self.num_experts * ffn_per_expert
                + self.n_shared_experts * self.intermediate_size * H * 3
                + H * self.num_experts  # router
            )
        else:
            ffn_per_layer = self.intermediate_size * H * 3

        norm_per_layer = H * 2  # attn_norm + ffn_norm
        per_layer = attn_per_layer + ffn_per_layer + norm_per_layer

        output_head = V * H
        final_norm = H

        return embedding + L * per_layer + output_head + final_norm


@dataclass
class ParallelStrategy:
    dp: int
    pp: int
    tp: int
    cp: int
    ep: int
    n_gpus: int
    allocation_mode: str = ""

    # estimated memory (GB)
    model_mem_gb: float = 0.0
    ddp_buffer_mem_gb: float = 0.0
    grad_mem_gb: float = 0.0
    optimizer_mem_gb: float = 0.0
    activation_mem_gb: float = 0.0
    total_mem_gb: float = 0.0
    cpu_optimizer_mem_gb: float = 0.0
    cpu_total_per_node_gb: float = 0.0
    tokens_per_gpu: int = 0
    layers_per_stage: str = ""
    experts_per_rank: int = 0
    efficiency_score: float = 0.0

    def __post_init__(self):
        attn = f"d{self.dp}p{self.pp}t{self.tp}"
        if self.cp > 1:
            attn += f"c{self.cp}"
        if self.ep > 1:
            ffn = f"d{self.dp}p{self.pp}e{self.ep}"
            self.allocation_mode = f"megatron:(attn:{attn}|ffn:{ffn})"
        else:
            self.allocation_mode = f"megatron:{attn}"


@dataclass
class ClusterSpec:
    n_gpus: int = 128
    gpus_per_node: int = 8
    gpu_memory_gb: float = 140.0

    @property
    def n_nodes(self) -> int:
        return self.n_gpus // self.gpus_per_node


@dataclass
class EngineConfig:
    """Engine-specific optimization flags that affect the cost model.

    These reflect runtime behaviors of the training engine (e.g., Megatron)
    that change whether communication can be overlapped with computation.
    """

    overlap_ep_alltoall: bool = True
    """Megatron MoE overlaps EP AllToAll dispatch/combine with expert FFN
    computation. When True, the cost model uses max(ep_time, routed_ffn_time)
    instead of ep_time + routed_ffn_time."""

    tp_bw_degradation: bool = True
    """Model NVLink bandwidth degradation for large TP groups. With TP>4,
    multiple rings compete for NVLink bandwidth; effective BW drops by
    ~1/sqrt(tp/4)."""

    act_factor_dense: int = 10
    """Activation memory factor for dense layers (per token per hidden dim)."""

    act_factor_moe: int = 18
    """Activation memory factor for MoE layers. Higher than dense due to
    router logits, dispatch/combine buffers, and expert intermediate states."""


MEGATRON_ENGINE = EngineConfig(
    overlap_ep_alltoall=True,
    tp_bw_degradation=True,
    act_factor_moe=18,
)

FSDP_ENGINE = EngineConfig(
    overlap_ep_alltoall=False,
    tp_bw_degradation=False,
    act_factor_moe=14,
)

SGLANG_ENGINE = EngineConfig(
    overlap_ep_alltoall=True,
    tp_bw_degradation=False,
    act_factor_moe=14,
)

_CPU_SYSTEM_OVERHEAD_GB = 40.0


def _get_host_memory_gb(explicit: float, hw: HardwareSpec | None = None) -> float:
    """Resolve host memory: CLI override > HW preset > auto-detect > 1000GB."""
    if explicit > 0:
        return explicit
    if hw is not None and hw.host_memory_gb > 0:
        return hw.host_memory_gb
    try:
        import psutil

        return psutil.virtual_memory().total / (1024**3)
    except Exception:
        return 1000.0


def _divisors(n: int) -> list[int]:
    result = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            result.append(i)
            if i != n // i:
                result.append(n // i)
    return sorted(result)


def estimate_memory(
    model: ModelSpec,
    strategy: ParallelStrategy,
    max_tokens_per_mb: int,
    optimizer_cpu_offload: bool = True,
    recompute: bool = True,
    grad_reduce_in_fp32: bool = True,
    engine: EngineConfig | None = None,
) -> ParallelStrategy:
    """Estimate per-GPU memory for a given strategy. Updates strategy in-place."""
    if engine is None:
        engine = MEGATRON_ENGINE
    H = model.hidden_size
    L = model.num_layers
    B = model.dtype_bytes  # 2 for bf16

    pp = strategy.pp
    tp = strategy.tp
    cp = strategy.cp
    ep = strategy.ep
    dp = strategy.dp

    # --- Layers per pipeline stage ---
    base_layers = L // pp
    remainder = L % pp
    if remainder == 0:
        strategy.layers_per_stage = str(base_layers)
    else:
        strategy.layers_per_stage = f"{base_layers}-{base_layers + 1}"
    layers_this_stage = base_layers + (1 if remainder > 0 else 0)  # worst case

    # --- Model parameters per GPU (bf16) ---
    # Handle first_k_dense_replace: first N layers use dense FFN.
    # For memory estimation, pick the stage with the MOST parameters.
    # With MoE, later stages (all-MoE, no dense) often have more params
    # than stage 0 (which contains the cheaper dense layers).
    dense_layers_in_stage = 0
    moe_layers_in_stage = layers_this_stage
    if model.has_dense_replace and pp > 1:
        first_stage_layers = base_layers + (1 if remainder > 0 else 0)
        if model.first_k_dense_replace <= first_stage_layers:
            # Dense layers fit in stage 0; compare stage 0 vs later all-MoE stage
            dense_stage0 = min(model.first_k_dense_replace, layers_this_stage)
            moe_stage0 = layers_this_stage - dense_stage0
            # Later stage: all MoE (no dense replacement layers)
            dense_later = 0
            moe_later = layers_this_stage
            # Pick whichever has more total FFN params (computed after FFN section)
            # For now, set up both; we'll compare below
            _compare_stages = True
        else:
            dense_layers_in_stage = layers_this_stage
            moe_layers_in_stage = 0
            _compare_stages = False
    else:
        _compare_stages = False

    # Attention params: split by TP (except MLA q_a/kv_a which are replicated)
    if model.is_mla:
        # q_a (H→q_lora_rank) and kv_a (H→kv_lora_rank+rope) are replicated
        replicated_attn = H * model.q_lora_rank + H * (
            model.kv_lora_rank + model.qk_rope_head_dim
        )
        # q_b, kv_b, o_proj are split by TP (ColumnParallel / RowParallel)
        tp_split_attn = (
            model.q_lora_rank * model.num_heads * model.head_dim
            + model.kv_lora_rank
            * model.num_heads
            * (model.qk_nope_head_dim + model.v_head_dim)
            + model.num_heads * model.v_head_dim * H
        )
        attn_per_layer = replicated_attn + tp_split_attn / tp
    else:
        head_dim = H // model.num_heads
        attn_per_layer = (
            H * model.num_heads * head_dim
            + H * model.num_kv_heads * head_dim * 2
            + model.num_heads * head_dim * H
        ) / tp

    # FFN params: MoE split by EP, dense split by TP
    if model.is_moe:
        experts_local = model.num_experts // ep
        strategy.experts_per_rank = experts_local
        moe_ffn_per_layer = (
            experts_local * model.expert_intermediate_size * H * 3
            + model.n_shared_experts * model.intermediate_size * H * 3 / tp
            + H * model.num_experts  # router (replicated)
        )
        dense_ffn_per_layer = model.intermediate_size * H * 3 / tp

        if _compare_stages:
            # Compare stage 0 (with dense layers) vs later stage (all MoE)
            ffn_stage0 = (
                moe_ffn_per_layer * moe_stage0 + dense_ffn_per_layer * dense_stage0
            )
            ffn_later = (
                moe_ffn_per_layer * moe_later + dense_ffn_per_layer * dense_later
            )
            if ffn_later >= ffn_stage0:
                dense_layers_in_stage = dense_later
                moe_layers_in_stage = moe_later
            else:
                dense_layers_in_stage = dense_stage0
                moe_layers_in_stage = moe_stage0

        ffn_total = (
            moe_ffn_per_layer * moe_layers_in_stage
            + dense_ffn_per_layer * dense_layers_in_stage
        )
    else:
        ffn_per_layer = model.intermediate_size * H * 3 / tp
        ffn_total = ffn_per_layer * layers_this_stage
        strategy.experts_per_rank = 0

    norm_per_layer = H * 2
    params_total_per_layer = attn_per_layer + norm_per_layer
    # Embedding + output (first/last PP stage)
    embed_params = model.vocab_size * H / tp
    if model.tie_word_embeddings:
        embed_params = model.vocab_size * H / tp  # counted once
    params_total = layers_this_stage * params_total_per_layer + ffn_total + embed_params

    model_mem_bytes = params_total * B
    strategy.model_mem_gb = model_mem_bytes / (1024**3)

    # --- DDP grad buffer (full, not sharded by DP) ---
    grad_dtype_bytes = 4 if grad_reduce_in_fp32 else 2
    ddp_buffer_bytes = params_total * grad_dtype_bytes
    strategy.ddp_buffer_mem_gb = ddp_buffer_bytes / (1024**3)

    # --- fp32 gradient shard (distributed optimizer reduce-scatter output) ---
    # When grad_reduce_in_fp32=True, reduce-scatter operates in-place on the fp32
    # DDP buffer, so no separate allocation is needed for the grad shard.
    if grad_reduce_in_fp32:
        grad_shard_bytes = 0
    else:
        grad_shard_bytes = params_total * 4  # fp32, separate from bf16 DDP buffer
        if dp > 1:
            grad_shard_bytes /= dp
    strategy.grad_mem_gb = grad_shard_bytes / (1024**3)

    # --- Optimizer states ---
    # master weights + exp_avg + exp_avg_sq = 3 × fp32 = 12 bytes/param
    optimizer_state_bytes = params_total * 4 * 3
    if dp > 1:
        optimizer_state_bytes /= dp

    if optimizer_cpu_offload:
        optimizer_bytes = 0
        strategy.cpu_optimizer_mem_gb = optimizer_state_bytes / (1024**3)
    else:
        optimizer_bytes = optimizer_state_bytes
        strategy.cpu_optimizer_mem_gb = 0.0
    strategy.optimizer_mem_gb = optimizer_bytes / (1024**3)

    # --- Activation memory ---
    tokens_per_gpu = max_tokens_per_mb // cp
    strategy.tokens_per_gpu = tokens_per_gpu

    if recompute:
        # With full activation recomputation (recompute_num_layers=1):
        #
        # 1. Checkpoint memory: input hidden state per layer per in-flight
        #    micro-batch. With sequence parallelism (SP), the hidden state is
        #    distributed across TP ranks on the sequence dimension.
        #    Per layer per micro-batch: tokens_per_gpu * H * B / tp
        #
        # 2. Working memory: intermediate activations during backward recompute
        #    of ONE layer (freed after each layer). This is constant regardless
        #    of PP or layers_per_stage.
        #
        # In 1F1B schedule: peak in-flight micro-batches = PP.
        checkpoint_per_layer = tokens_per_gpu * H * B / tp
        n_mbs_in_flight = max(pp, 1)
        checkpoint_bytes = checkpoint_per_layer * layers_this_stage * n_mbs_in_flight

        working_factor = (
            engine.act_factor_moe if model.is_moe else engine.act_factor_dense
        )
        working_bytes = tokens_per_gpu * H * working_factor * B / tp

        act_bytes = checkpoint_bytes + working_bytes
    else:
        act_factor = 10
        act_per_layer = tokens_per_gpu * H * act_factor * B / tp
        act_bytes = act_per_layer * layers_this_stage

    # Communication buffers (all-to-all for MoE, etc.)
    comm_buffer = 0
    if model.is_moe:
        comm_buffer = tokens_per_gpu * H * B * 2  # dispatch + combine

    act_bytes += comm_buffer
    strategy.activation_mem_gb = act_bytes / (1024**3)

    # --- Total ---
    frag_factor = 1.05
    # CUDA context + NCCL buffers scale with number of process groups
    n_comm_groups = (
        (1 if tp > 1 else 0)
        + (1 if ep > 1 else 0)
        + (1 if cp > 1 else 0)
        + (1 if dp > 1 else 0)
        + (pp - 1 if pp > 1 else 0)  # PP P2P pairs
    )
    cuda_context_gb = 8.0 + n_comm_groups * 1.0
    strategy.total_mem_gb = (
        strategy.model_mem_gb
        + strategy.ddp_buffer_mem_gb
        + strategy.grad_mem_gb
        + strategy.optimizer_mem_gb
        + strategy.activation_mem_gb
    ) * frag_factor + cuda_context_gb

    return strategy


def compute_efficiency_score(
    s: ParallelStrategy,
    model: ModelSpec,
    max_tokens_per_mb: int,
    hw: HardwareSpec,
    gpus_per_node: int = 8,
    batch_size: int = 0,
    engine: EngineConfig | None = None,
    profile: ProfileLookup | None = None,
) -> float:
    """Estimate relative throughput using an analytical cost model.

    Uses ALPA-style alpha-beta communication model with hierarchical EP
    AllToAll (intra-node NVLink + inter-node IB) from AIConfigurator.
    Engine-specific optimizations (EP overlap, TP BW degradation) are
    controlled via EngineConfig.

    When *profile* is provided, GEMM and communication times are looked up
    from profiling data with log-space interpolation.  Each lookup falls
    back independently to the analytical formula when no data is available.
    """
    if engine is None:
        engine = MEGATRON_ENGINE
    H = model.hidden_size
    T = max_tokens_per_mb // (s.tp * s.cp)
    L = math.ceil(model.num_layers / s.pp)
    num_mb = batch_size // s.dp if batch_size > 0 else 8

    # === 1. Compute cost per layer (FLOPs → seconds) ===
    # Attention projection FLOPs
    if model.is_mla:
        # q_a, kv_a: replicated across TP (no head dimension)
        replicated_proj = H * model.q_lora_rank + H * (
            model.kv_lora_rank + model.qk_rope_head_dim
        )
        # q_b, kv_b, o_proj: split by TP (head dimension)
        tp_split_proj = (
            model.q_lora_rank * model.num_heads * model.head_dim
            + model.kv_lora_rank
            * model.num_heads
            * (model.qk_nope_head_dim + model.v_head_dim)
            + model.num_heads * model.v_head_dim * H
        )
        attn_proj_flops = 2.0 * T * (replicated_proj + tp_split_proj / s.tp)
    else:
        attn_proj_flops = 8.0 * H * H * T / s.tp

    # Attention score FLOPs (QK^T + AV)
    if model.is_mla:
        qk_dim = model.qk_nope_head_dim + model.qk_rope_head_dim
        attn_score_flops = (
            2.0 * T * T * model.num_heads * (qk_dim + model.v_head_dim) / s.tp
        )
    else:
        attn_score_flops = 4.0 * T * T * H / s.tp
    if model.is_moe:
        routed_flops = (
            float(model.num_experts_per_tok)
            * 6.0
            * H
            * model.expert_intermediate_size
            * T
            / s.ep
        )
        shared_flops = 0.0
        if model.n_shared_experts > 0:
            shared_flops = (
                float(model.n_shared_experts)
                * 6.0
                * H
                * model.intermediate_size
                * T
                / s.tp
            )
        ffn_flops = routed_flops + shared_flops
    else:
        routed_flops = 0.0
        shared_flops = 0.0
        ffn_flops = 6.0 * H * model.intermediate_size * T / s.tp

    attn_compute = (attn_proj_flops + attn_score_flops) / hw.gpu_flops
    routed_compute = routed_flops / hw.gpu_flops
    shared_compute = shared_flops / hw.gpu_flops
    total_compute = (attn_proj_flops + attn_score_flops + ffn_flops) / hw.gpu_flops

    # Profiling override: replace roofline with measured GEMM times
    if profile is not None:
        # Attention projections
        if model.is_mla:
            proj_N = int(
                model.q_lora_rank + model.kv_lora_rank + model.qk_rope_head_dim
            )
            proj_K = H
        else:
            proj_N = H
            proj_K = H
        t_attn_proj = profile.gemm_time_us(T, proj_N, proj_K)

        # FFN
        if model.is_moe:
            ffn_N_r = model.expert_intermediate_size
            ffn_N_s = (
                model.intermediate_size // s.tp if model.n_shared_experts > 0 else 0
            )
        else:
            ffn_N_r = 0
            ffn_N_s = model.intermediate_size // s.tp
        t_ffn_routed = profile.gemm_time_us(T, ffn_N_r, H) if ffn_N_r > 0 else None
        t_ffn_shared = profile.gemm_time_us(T, ffn_N_s, H) if ffn_N_s > 0 else None

        if t_attn_proj is not None:
            attn_compute = t_attn_proj * 1e-6  # us → seconds
        if t_ffn_routed is not None and model.is_moe:
            routed_compute = t_ffn_routed * 1e-6 * model.num_experts_per_tok
        if t_ffn_shared is not None:
            shared_compute = t_ffn_shared * 1e-6 * max(model.n_shared_experts, 1)
        total_compute = attn_compute + routed_compute + shared_compute

    # === 2. Communication cost per layer (alpha-beta model) ===

    # TP AllReduce (ring-based)
    tp_time = 0.0
    if s.tp > 1:
        n_ar = 1 if model.is_moe else 2
        vol = n_ar * 2.0 * (s.tp - 1) / s.tp * H * T * 2
        topo = "nvlink" if s.tp <= gpus_per_node else "ib"
        tp_profiled = (
            profile.comm_time_us("allreduce", int(vol), s.tp, topo)
            if profile is not None
            else None
        )
        if tp_profiled is not None:
            tp_time = tp_profiled * 1e-6
        elif s.tp <= gpus_per_node:
            effective_bw = hw.bw_nvlink
            if engine.tp_bw_degradation and s.tp > 4:
                effective_bw /= math.sqrt(s.tp / 4)
            tp_time = hw.alpha_nvlink + vol / effective_bw
        else:
            tp_time = hw.alpha_ib + vol / hw.bw_ib

    # EP AllToAll (hierarchical: intra NVLink + inter IB)
    ep_time = 0.0
    if s.ep > 1 and model.is_moe:
        vol_per_dir = float(T) * H * 2 * (s.ep - 1) / s.ep
        total_vol = 2.0 * vol_per_dir  # dispatch + combine
        topo = "nvlink" if s.ep <= gpus_per_node else "mixed"
        ep_profiled = (
            profile.comm_time_us("alltoall", int(total_vol), s.ep, topo)
            if profile is not None
            else None
        )
        if ep_profiled is not None:
            ep_time = ep_profiled * 1e-6
        elif s.ep <= gpus_per_node:
            ep_time = hw.alpha_nvlink + total_vol / hw.bw_nvlink
        else:
            intra_frac = (gpus_per_node - 1) / (s.ep - 1)
            inter_frac = 1.0 - intra_frac
            intra_vol = total_vol * intra_frac
            inter_vol = total_vol * inter_frac
            ep_nodes = s.ep / gpus_per_node
            congestion = math.sqrt(ep_nodes)
            ep_time = (
                hw.alpha_ib
                + intra_vol / hw.bw_nvlink
                + inter_vol / hw.bw_ib * congestion
            )

    # CP ring attention (KV transfer)
    cp_time = 0.0
    if s.cp > 1:
        if model.is_mla:
            kv_dim = model.kv_lora_rank + model.qk_rope_head_dim
        else:
            head_dim = H // model.num_heads
            kv_dim = model.num_kv_heads * head_dim * 2
        vol = float(T) * kv_dim * 2 * (s.cp - 1)
        cp_time = hw.alpha_nvlink * (s.cp - 1) + vol / hw.bw_nvlink

    # === 3. Per-layer → stage → total step time ===
    # MoE overlap: EP AllToAll runs concurrently with routed FFN compute
    if engine.overlap_ep_alltoall and model.is_moe and s.ep > 1:
        moe_time = max(ep_time, routed_compute)
        per_layer_compute_comm = (
            attn_compute + tp_time + moe_time + shared_compute + cp_time
        )
    else:
        per_layer_compute_comm = total_compute + tp_time + ep_time + cp_time

    per_layer_time = 3.0 * per_layer_compute_comm  # fwd=1x, bwd=2x
    stage_time = per_layer_time * L

    # PP P2P (activation transfer between pipeline stages)
    if s.pp > 1:
        p2p_vol = float(max_tokens_per_mb) / s.cp * H * 2
        stage_time += hw.alpha_ib + p2p_vol / hw.bw_ib

    # 1F1B pipeline schedule: total = (PP + num_mb - 1) * stage_time
    total_time = (s.pp + num_mb - 1) * stage_time

    return s.dp * num_mb / total_time if total_time > 0 else 0.0


def search_strategies(
    model: ModelSpec,
    cluster: ClusterSpec,
    max_tokens_per_mb: int,
    max_length: int = 16384,
    optimizer_cpu_offload: bool = True,
    recompute: bool = True,
    min_dp: int = 1,
    max_pp: int = 16,
    batch_size: int = 0,
    grad_reduce_in_fp32: bool = True,
    gpus_per_node: int = 8,
    hw: HardwareSpec | None = None,
    engine: EngineConfig | None = None,
    profile: ProfileLookup | None = None,
) -> list[ParallelStrategy]:
    """Enumerate all valid parallel strategies and estimate memory."""
    if hw is None:
        hw = HardwareSpec()
    if engine is None:
        engine = MEGATRON_ENGINE
    N = cluster.n_gpus
    results: list[ParallelStrategy] = []

    valid_tp = [t for t in _divisors(model.num_heads) if t <= 16]
    valid_pp = [p for p in _divisors(N) if p <= max_pp and p <= model.num_layers]
    valid_cp = [1, 2, 4, 8]

    for tp in valid_tp:
        # GQA constraint: num_kv_heads must be divisible by TP
        if not model.is_mla and model.num_kv_heads % tp != 0:
            continue

        # Lightning Attention GroupNorm constraint
        if model.has_group_norm:
            heads_per_tp = model.num_heads // tp
            if heads_per_tp % model.group_norm_size != 0:
                continue
            if heads_per_tp <= model.group_norm_size:
                continue

        for pp in valid_pp:
            for cp in valid_cp:
                # Lightning Attention + CP constraint
                if model.has_group_norm:
                    heads_per_tp = model.num_heads // tp
                    if heads_per_tp % cp != 0:
                        continue

                # tokens must be divisible by CP
                if max_tokens_per_mb % cp != 0:
                    continue

                # Constraint: TP × CP = EP (for MoE attn/ffn world balance)
                ep = tp * cp
                if model.is_moe:
                    if model.num_experts % ep != 0:
                        continue
                    if model.num_experts // ep < 1:
                        continue
                else:
                    ep = 1
                    if tp * cp > N // pp:
                        continue

                # DP = N / (PP × TP × CP)
                remaining = N // (pp * tp * cp)
                if remaining < min_dp:
                    continue
                if N != remaining * pp * tp * cp:
                    continue
                dp = remaining

                # Verify FFN world: DP × PP × EP = N
                if model.is_moe and dp * pp * ep != N:
                    continue

                # batch_size must be divisible by DP
                if batch_size > 0 and batch_size % dp != 0:
                    continue

                # min micro-batches = 2*PP (Megatron 1F1B schedule constraint)
                if batch_size > 0 and pp > 1:
                    n_mbs = batch_size // dp
                    if n_mbs < 2 * pp:
                        continue

                s = ParallelStrategy(dp=dp, pp=pp, tp=tp, cp=cp, ep=ep, n_gpus=N)
                estimate_memory(
                    model,
                    s,
                    max_tokens_per_mb,
                    optimizer_cpu_offload,
                    recompute,
                    grad_reduce_in_fp32,
                    engine,
                )
                if s.cpu_optimizer_mem_gb > 0:
                    s.cpu_total_per_node_gb = (
                        s.cpu_optimizer_mem_gb * gpus_per_node + _CPU_SYSTEM_OVERHEAD_GB
                    )
                s.efficiency_score = compute_efficiency_score(
                    s,
                    model,
                    max_tokens_per_mb,
                    hw,
                    gpus_per_node,
                    batch_size,
                    engine,
                    profile,
                )
                results.append(s)

    results.sort(key=lambda s: (s.total_mem_gb, -s.dp, s.pp))
    return results


def print_results(
    results: list[ParallelStrategy],
    gpu_memory_gb: float,
    model: ModelSpec,
    max_tokens_per_mb: int,
    gpus_per_node: int = 8,
    batch_size: int = 0,
    engine: EngineConfig | None = None,
    host_memory_gb: float = 0.0,
) -> None:
    if engine is None:
        engine = MEGATRON_ENGINE
    engine_names = {
        id(MEGATRON_ENGINE): "megatron",
        id(FSDP_ENGINE): "fsdp",
        id(SGLANG_ENGINE): "sglang",
    }
    engine_name = engine_names.get(id(engine), "custom")

    has_cpu_offload = any(s.cpu_optimizer_mem_gb > 0 for s in results)

    def _is_feasible(s: ParallelStrategy) -> bool:
        if s.total_mem_gb > gpu_memory_gb:
            return False
        if has_cpu_offload and host_memory_gb > 0 and s.cpu_total_per_node_gb > 0:
            if s.cpu_total_per_node_gb > host_memory_gb:
                return False
        return True

    print("=" * 140)
    print(
        f"Model: {model.total_params_billion():.1f}B params | "
        f"MoE={model.is_moe} ({model.num_experts} experts) | MLA={model.is_mla} | "
        f"Layers={model.num_layers}"
    )
    mem_info = f"max_tokens_per_mb={max_tokens_per_mb} | GPU={gpu_memory_gb:.0f}GB"
    if has_cpu_offload and host_memory_gb > 0:
        mem_info += f" | Host={host_memory_gb:.0f}GB/node"
    mem_info += f" | Engine={engine_name}"
    print(mem_info)
    print("=" * 140)

    hdr = (
        f"{'#':>3} {'DP':>3} {'PP':>3} {'TP':>3} {'CP':>3} {'EP':>4} "
        f"{'Tok/GPU':>8} {'Layers':>7} {'Exp/R':>5} "
        f"{'Model':>7} {'GrdBuf':>7} {'Grad':>7} {'Opt':>7} {'Act':>7} {'Total':>7} "
    )
    if has_cpu_offload:
        hdr += f"{'CPU/N':>7} "
    hdr += f"{'Fit?':>5} {'allocation_mode'}"
    print(hdr)
    print("-" * 140)

    for i, s in enumerate(results):
        gpu_ok = s.total_mem_gb <= gpu_memory_gb
        cpu_ok = (
            s.cpu_total_per_node_gb <= host_memory_gb
            if (has_cpu_offload and host_memory_gb > 0 and s.cpu_total_per_node_gb > 0)
            else True
        )
        if gpu_ok and cpu_ok:
            fits = "OK"
        elif not gpu_ok:
            fits = "OOM"
        else:
            fits = "CPU!"
        marker = " *" if fits == "OK" else ""

        row = (
            f"{i + 1:>3} {s.dp:>3} {s.pp:>3} {s.tp:>3} {s.cp:>3} {s.ep:>4} "
            f"{s.tokens_per_gpu:>8} {s.layers_per_stage:>7} {s.experts_per_rank:>5} "
            f"{s.model_mem_gb:>6.1f}G {s.ddp_buffer_mem_gb:>6.1f}G "
            f"{s.grad_mem_gb:>6.1f}G {s.optimizer_mem_gb:>6.1f}G "
            f"{s.activation_mem_gb:>6.1f}G {s.total_mem_gb:>6.1f}G "
        )
        if has_cpu_offload:
            row += f"{s.cpu_total_per_node_gb:>6.0f}G "
        row += f"{fits:>5}{marker} {s.allocation_mode}"
        print(row)

    ok_count = sum(1 for s in results if _is_feasible(s))
    print(f"\nTotal: {len(results)} strategies, {ok_count} feasible")

    if ok_count == 0:
        return

    fit = [s for s in results if _is_feasible(s)]
    fit.sort(key=lambda s: -s.efficiency_score)
    top_n = min(3, len(fit))
    best_score = fit[0].efficiency_score

    def num_mb_fn(s):
        return batch_size // s.dp if batch_size > 0 else 8

    print()
    print("=" * 70)
    print(f" Top-{top_n} Recommended (by estimated throughput)")
    print("=" * 70)

    for rank, s in enumerate(fit[:top_n], 1):
        margin = gpu_memory_gb - s.total_mem_gb
        rel = s.efficiency_score / best_score * 100 if best_score > 0 else 0
        num_mb = num_mb_fn(s)
        bubble_pct = (s.pp - 1) / (s.pp + num_mb - 1) * 100 if s.pp > 1 else 0

        print(f"\n  #{rank}  {s.allocation_mode}")
        print(f"      DP={s.dp}  PP={s.pp}  TP={s.tp}  CP={s.cp}  EP={s.ep}")
        print(
            f"      GPU: {s.total_mem_gb:.1f}G / {gpu_memory_gb:.0f}G (margin {margin:.0f}G)"
        )
        if has_cpu_offload and s.cpu_total_per_node_gb > 0 and host_memory_gb > 0:
            cpu_margin = host_memory_gb - s.cpu_total_per_node_gb
            print(
                f"      CPU: {s.cpu_total_per_node_gb:.0f}G / {host_memory_gb:.0f}G/node "
                f"(margin {cpu_margin:.0f}G)"
            )
        print(f"      Score: {s.efficiency_score:.3f} ({rel:.0f}%)")

        pros = []
        cons = []

        if s.tp <= gpus_per_node and s.tp > 1:
            pros.append(f"TP={s.tp} in-node NVLink")
        elif s.tp > gpus_per_node:
            cons.append(f"TP={s.tp} cross-node IB (slow AllReduce)")

        if s.dp > 1:
            pros.append(f"DP={s.dp} gradient parallelism")

        if s.pp > 1:
            if bubble_pct <= 30:
                pros.append(f"PP={s.pp} bubble {bubble_pct:.0f}% (num_mb={num_mb})")
            else:
                cons.append(f"PP={s.pp} bubble {bubble_pct:.0f}% (num_mb={num_mb})")

        if model.is_moe and s.ep > 1:
            if s.ep <= gpus_per_node:
                pros.append(f"EP={s.ep} in-node NVLink AllToAll")
            else:
                ep_inter_pct = (1 - (gpus_per_node - 1) / (s.ep - 1)) * 100
                cons.append(f"EP={s.ep} AllToAll (~{ep_inter_pct:.0f}% cross-node IB)")

        if s.cp > 1:
            if s.cp <= 2:
                pros.append(f"CP={s.cp} ring attention")
            else:
                cons.append(f"CP={s.cp} ring attention (high variance on long seq)")

        for p in pros:
            print(f"       + {p}")
        for c in cons:
            print(f"       - {c}")


# ===========================================================================
# Inference mode
# ===========================================================================


@dataclass
class InferenceStrategy:
    tp: int
    ep: int
    pp: int
    n_gpus_per_instance: int
    n_instances: int

    weight_mem_gb: float = 0.0
    kv_cache_mem_gb: float = 0.0
    activation_mem_gb: float = 0.0
    total_mem_gb: float = 0.0
    kv_cache_max_tokens: int = 0

    prefill_time_ms: float = 0.0
    decode_time_ms: float = 0.0
    prefill_tps: float = 0.0
    decode_tps: float = 0.0
    aggregate_decode_tps: float = 0.0

    allocation_mode: str = ""

    def __post_init__(self):
        # SGLang CLI: --tp T [--ep E] [--pp P]
        self.allocation_mode = f"tp{self.tp}"
        if self.ep > 1 and self.ep != self.tp:
            self.allocation_mode += f"_ep{self.ep}"
        if self.pp > 1:
            self.allocation_mode += f"_pp{self.pp}"


def _kv_bytes_per_token(model: ModelSpec) -> int:
    """KV cache bytes per token per layer (bf16)."""
    if model.is_mla:
        return (model.kv_lora_rank + model.qk_rope_head_dim) * 2
    head_dim = model.hidden_size // model.num_heads
    return model.num_kv_heads * head_dim * 2 * 2


def estimate_inference_memory(
    model: ModelSpec,
    s: InferenceStrategy,
    max_batch_tokens: int,
    isl: int,
    batch_size: int,
    gpus_per_node: int = 8,
) -> InferenceStrategy:
    """Estimate per-GPU memory for inference (weights + KV cache + activation)."""
    H = model.hidden_size
    L = model.num_layers
    B = model.dtype_bytes

    tp = s.tp
    ep = s.ep
    pp = s.pp
    layers_per_stage = math.ceil(L / pp)

    # --- Model weights ---
    if model.is_mla:
        replicated_attn = H * model.q_lora_rank + H * (
            model.kv_lora_rank + model.qk_rope_head_dim
        )
        tp_split_attn = (
            model.q_lora_rank * model.num_heads * model.head_dim
            + model.kv_lora_rank
            * model.num_heads
            * (model.qk_nope_head_dim + model.v_head_dim)
            + model.num_heads * model.v_head_dim * H
        )
        attn_per_layer = replicated_attn + tp_split_attn / tp
    else:
        head_dim = H // model.num_heads
        attn_per_layer = (
            H * model.num_heads * head_dim
            + H * model.num_kv_heads * head_dim * 2
            + model.num_heads * head_dim * H
        ) / tp

    if model.is_moe:
        experts_local = model.num_experts // ep
        moe_ffn = experts_local * model.expert_intermediate_size * H * 3
        shared_ffn = model.n_shared_experts * model.intermediate_size * H * 3 / tp
        router = H * model.num_experts
        ffn_per_layer = moe_ffn + shared_ffn + router
    else:
        ffn_per_layer = model.intermediate_size * H * 3 / tp

    norm_per_layer = H * 2
    params_per_layer = attn_per_layer + ffn_per_layer + norm_per_layer
    embed_params = model.vocab_size * H / tp
    total_params = layers_per_stage * params_per_layer + embed_params

    s.weight_mem_gb = total_params * B / (1024**3)

    # --- KV Cache ---
    kv_per_tok_per_layer = _kv_bytes_per_token(model)
    kv_bytes = max_batch_tokens * layers_per_stage * kv_per_tok_per_layer / tp
    s.kv_cache_mem_gb = kv_bytes / (1024**3)

    # --- Activation (AIConfigurator heuristic) ---
    prefill_tokens = isl * batch_size // pp
    if model.is_moe:
        c_dict = {1: 22, 2: 13, 4: 10, 8: 10}
    else:
        c_dict = {1: 10, 2: 6, 4: 5, 8: 5}
    c = c_dict.get(min(tp, 8), 5)
    act_bytes = 2.0 * prefill_tokens * H * c
    act_bytes = max(act_bytes, 70 * 1024 * 1024)
    s.activation_mem_gb = act_bytes / (1024**3)

    # --- Total ---
    cuda_context_gb = 5.0
    s.total_mem_gb = (
        s.weight_mem_gb + s.kv_cache_mem_gb + s.activation_mem_gb + cuda_context_gb
    )
    s.kv_cache_max_tokens = max_batch_tokens

    return s


def compute_inference_score(
    s: InferenceStrategy,
    model: ModelSpec,
    hw: HardwareSpec,
    gpus_per_node: int = 8,
    isl: int = 4096,
    osl: int = 512,
    batch_size: int = 32,
) -> InferenceStrategy:
    """Estimate prefill/decode performance using roofline model."""
    H = model.hidden_size
    L = model.num_layers
    layers = math.ceil(L / s.pp)
    tp = s.tp
    ep = s.ep

    # --- Helper: communication cost per layer ---
    def _comm_time(T: int) -> float:
        tp_time = 0.0
        if tp > 1:
            n_ar = 1 if model.is_moe else 2
            vol = n_ar * 2.0 * (tp - 1) / tp * H * T * 2
            if tp <= gpus_per_node:
                tp_time = hw.alpha_nvlink + vol / hw.bw_nvlink
            else:
                tp_time = hw.alpha_ib + vol / hw.bw_ib

        ep_time = 0.0
        if ep > 1 and model.is_moe:
            vol_per_dir = float(T) * H * 2 * (ep - 1) / ep
            total_vol = 2.0 * vol_per_dir
            if ep <= gpus_per_node:
                ep_time = hw.alpha_nvlink + total_vol / hw.bw_nvlink
            else:
                intra_frac = (gpus_per_node - 1) / (ep - 1)
                inter_frac = 1.0 - intra_frac
                intra_vol = total_vol * intra_frac
                inter_vol = total_vol * inter_frac
                ep_nodes = ep / gpus_per_node
                congestion = math.sqrt(ep_nodes)
                ep_time = (
                    hw.alpha_ib
                    + intra_vol / hw.bw_nvlink
                    + inter_vol / hw.bw_ib * congestion
                )
        return tp_time + ep_time

    # --- Helper: compute FLOPs per layer ---
    def _compute_flops(T: int, batch: int = 1) -> float:
        """Compute FLOPs per layer.

        For prefill: T = total tokens (isl * batch / tp), batch = actual batch size.
        Projection/FFN FLOPs scale with total tokens (batched GEMM).
        Attention score FLOPs are per-request (no cross-request attention),
        so use per-request seq_len = T / batch for the quadratic term.
        For decode: T = batch_size, batch = 1 (each token attends to its own KV).
        """
        seq_len = T / batch if batch > 1 else T
        if model.is_mla:
            replicated_proj = H * model.q_lora_rank + H * (
                model.kv_lora_rank + model.qk_rope_head_dim
            )
            tp_split_proj = (
                model.q_lora_rank * model.num_heads * model.head_dim
                + model.kv_lora_rank
                * model.num_heads
                * (model.qk_nope_head_dim + model.v_head_dim)
                + model.num_heads * model.v_head_dim * H
            )
            attn_proj = 2.0 * T * (replicated_proj + tp_split_proj / tp)
            qk_dim = model.qk_nope_head_dim + model.qk_rope_head_dim
            attn_score = (
                2.0 * batch * seq_len * seq_len * model.num_heads * (qk_dim + model.v_head_dim) / tp
            )
        else:
            attn_proj = 8.0 * H * H * T / tp
            attn_score = 4.0 * batch * seq_len * seq_len * H / tp
        if model.is_moe:
            routed = (
                float(model.num_experts_per_tok)
                * 6.0
                * H
                * model.expert_intermediate_size
                * T
                / ep
            )
            shared = 0.0
            if model.n_shared_experts > 0:
                shared = (
                    float(model.n_shared_experts)
                    * 6.0
                    * H
                    * model.intermediate_size
                    * T
                    / tp
                )
            ffn = routed + shared
        else:
            ffn = 6.0 * H * model.intermediate_size * T / tp
        return attn_proj + attn_score + ffn

    # === Prefill (compute-bound) ===
    T_prefill = isl * batch_size // tp
    prefill_compute = _compute_flops(T_prefill, batch=batch_size) / hw.gpu_flops
    prefill_comm = _comm_time(T_prefill)
    prefill_per_layer = prefill_compute + prefill_comm
    prefill_total = prefill_per_layer * layers
    if s.pp > 1:
        p2p_vol = float(isl * batch_size) * H * 2
        prefill_total += (hw.alpha_ib + p2p_vol / hw.bw_ib) * (s.pp - 1)

    s.prefill_time_ms = prefill_total * 1000
    s.prefill_tps = isl * batch_size / prefill_total if prefill_total > 0 else 0

    # === Decode (memory-bandwidth bound, roofline) ===
    # Each step: read all weights + KV cache from HBM, compute batch_size tokens
    # PP stages execute serially in inference (no microbatch pipelining)
    T_decode = batch_size

    # Per-stage compute time (very small for decode)
    stage_decode_compute = _compute_flops(T_decode) / hw.gpu_flops * layers

    # Per-stage memory read time
    weight_bytes = s.weight_mem_gb * (1024**3)
    kv_per_tok = _kv_bytes_per_token(model)
    avg_seq_len = isl + osl // 2
    kv_read_bytes = float(batch_size) * avg_seq_len * layers * kv_per_tok / tp
    stage_memory_time = (weight_bytes + kv_read_bytes) / hw.hbm_bw

    # Per-stage communication
    stage_decode_comm = _comm_time(T_decode) * layers

    # Roofline per stage: max(compute, memory) + communication
    stage_decode_time = max(stage_decode_compute, stage_memory_time) + stage_decode_comm

    # PP stages serial + P2P between stages
    decode_total = stage_decode_time * s.pp
    if s.pp > 1:
        p2p_vol = float(batch_size) * H * 2
        decode_total += (hw.alpha_ib + p2p_vol / hw.bw_ib) * (s.pp - 1)

    s.decode_time_ms = decode_total * 1000
    s.decode_tps = batch_size / decode_total if decode_total > 0 else 0
    s.aggregate_decode_tps = s.n_instances * s.decode_tps

    return s


def search_inference_strategies(
    model: ModelSpec,
    cluster: ClusterSpec,
    hw: HardwareSpec,
    isl: int = 4096,
    osl: int = 512,
    max_batch_tokens: int = 131072,
    batch_size: int = 32,
    allow_pp: bool = False,
) -> list[InferenceStrategy]:
    """Enumerate inference parallel strategies (TP, EP).

    SGLang model: launch with --tp T [--ep E].
    Constraints:
      - TP shards attention heads: num_heads % tp == 0, can cross nodes
      - EP shards experts: num_experts % ep == 0, ep <= tp
      - gpus_per_instance = tp * pp
      - PP is NOT recommended for inference (stages execute serially with no
        microbatch pipelining, only adding latency). Disabled by default; use
        allow_pp=True or --infer-allow-pp to enumerate PP>1 strategies.
    """
    N = cluster.n_gpus
    gpn = cluster.gpus_per_node
    results: list[InferenceStrategy] = []

    valid_tp = [t for t in _divisors(model.num_heads) if t <= N]
    valid_pp = [1, 2, 4] if allow_pp else [1]

    for tp in valid_tp:
        # Inference engines (SGLang/vLLM) can replicate KV heads when tp > kv_heads,
        # so we only require tp divides num_heads, not kv_heads.
        # Training requires kv_heads % tp == 0 (handled in search_strategies).
        if model.has_group_norm:
            heads_per_tp = model.num_heads // tp
            if heads_per_tp % model.group_norm_size != 0:
                continue
            if heads_per_tp <= model.group_norm_size:
                continue
        if model.is_moe:
            valid_ep = [e for e in _divisors(model.num_experts) if e <= tp]
        else:
            valid_ep = [1]
        for ep in valid_ep:
            for pp in valid_pp:
                gpus_per_inst = tp * pp
                if gpus_per_inst > N:
                    continue
                if N % gpus_per_inst != 0:
                    continue
                n_instances = N // gpus_per_inst

                s = InferenceStrategy(
                    tp=tp,
                    ep=ep,
                    pp=pp,
                    n_gpus_per_instance=gpus_per_inst,
                    n_instances=n_instances,
                )
                estimate_inference_memory(
                    model,
                    s,
                    max_batch_tokens,
                    isl,
                    batch_size,
                    gpn,
                )

                fixed_mem_gb = s.weight_mem_gb + s.activation_mem_gb + 5.0
                layers_per_stage = math.ceil(model.num_layers / pp)
                kv_per_tok_bytes = layers_per_stage * _kv_bytes_per_token(model) / tp
                if kv_per_tok_bytes > 0:
                    kv_budget_bytes = max(0.0, cluster.gpu_memory_gb - fixed_mem_gb) * (
                        1024**3
                    )
                    s.kv_cache_max_tokens = int(kv_budget_bytes / kv_per_tok_bytes)
                else:
                    s.kv_cache_max_tokens = 0

                if s.total_mem_gb > cluster.gpu_memory_gb:
                    continue

                compute_inference_score(s, model, hw, gpn, isl, osl, batch_size)
                results.append(s)

    results.sort(key=lambda s: -s.aggregate_decode_tps)
    return results


def print_inference_results(
    results: list[InferenceStrategy],
    gpu_memory_gb: float,
    model: ModelSpec,
    cluster: ClusterSpec,
    isl: int,
    osl: int,
    batch_size: int,
    allow_pp: bool = False,
) -> None:
    gpn = cluster.gpus_per_node
    n_nodes = cluster.n_nodes

    print("=" * 110)
    print(
        f"Model: {model.total_params_billion():.1f}B params | "
        f"MoE={model.is_moe} ({model.num_experts} experts) | MLA={model.is_mla} | "
        f"Layers={model.num_layers}"
    )
    print(
        f"Cluster: {cluster.n_gpus} GPUs ({n_nodes} nodes x {gpn} GPUs) | "
        f"{gpu_memory_gb:.0f}GB/GPU"
    )
    print(f"Workload: isl={isl} osl={osl} batch={batch_size}")
    print("=" * 110)

    print(
        f"{'#':>3} {'TP':>3} {'EP':>4} {'PP':>3} {'GPU/i':>6} {'Inst':>5} "
        f"{'Weight':>7} {'KV':>7} {'Act':>7} {'Total':>7} "
        f"{'KV cap':>8} {'Prefill':>9} {'Decode':>9} {'Agg tok/s':>10}"
    )
    print("-" * 110)

    for i, s in enumerate(results):
        kv_cap_k = s.kv_cache_max_tokens / 1000
        print(
            f"{i + 1:>3} {s.tp:>3} {s.ep:>4} {s.pp:>3} {s.n_gpus_per_instance:>6} "
            f"{s.n_instances:>5} "
            f"{s.weight_mem_gb:>6.1f}G {s.kv_cache_mem_gb:>6.1f}G "
            f"{s.activation_mem_gb:>6.1f}G {s.total_mem_gb:>6.1f}G "
            f"{kv_cap_k:>7.0f}K "
            f"{s.prefill_time_ms:>8.1f}ms {s.decode_time_ms:>8.1f}ms "
            f"{s.aggregate_decode_tps:>10.0f}"
        )

    if not results:
        print("\nNo feasible inference strategy found.")
        if not allow_pp:
            print("Hint: try --infer-allow-pp to enable pipeline parallelism.")
        return

    top_n = min(3, len(results))
    best_tps = results[0].aggregate_decode_tps

    print()
    print("=" * 70)
    print(f" Top-{top_n} Recommended (by aggregate decode throughput)")
    print("=" * 70)

    for rank, s in enumerate(results[:top_n], 1):
        rel = s.aggregate_decode_tps / best_tps * 100 if best_tps > 0 else 0
        kv_cap_k = s.kv_cache_max_tokens / 1000

        print(
            f"\n  #{rank}  {s.allocation_mode}  "
            f"({s.n_gpus_per_instance} GPUs/instance x {s.n_instances} instances)"
        )
        print(
            f"      Memory: {s.total_mem_gb:.1f}G / {gpu_memory_gb:.0f}G "
            f"| KV capacity: ~{kv_cap_k:.0f}K tokens"
        )
        print(
            f"      Prefill: {s.prefill_time_ms:.1f} ms "
            f"({s.prefill_tps / 1000:.0f}K tok/s/inst)"
        )
        print(
            f"      Decode:  {s.decode_time_ms:.1f} ms/tok "
            f"({s.decode_tps:.0f} tok/s/inst)"
        )
        print(f"      Aggregate: {s.aggregate_decode_tps:.0f} tok/s ({rel:.0f}%)")

        pros = []
        cons = []

        if s.tp == 1:
            pros.append("TP=1 decode zero AllReduce")
        elif s.tp <= gpn:
            pros.append(f"TP={s.tp} in-node NVLink")
        else:
            cons.append(f"TP={s.tp} cross-node IB (high decode latency)")

        if model.is_moe and s.ep > 1:
            if s.ep <= gpn:
                pros.append(f"EP={s.ep} in-node NVLink AllToAll")
            else:
                ep_inter_pct = (1 - (gpn - 1) / (s.ep - 1)) * 100
                cons.append(f"EP={s.ep} AllToAll (~{ep_inter_pct:.0f}% cross-node IB)")

        if s.n_instances > 1:
            pros.append(f"{s.n_instances} instances maximize aggregate throughput")

        if s.pp > 1:
            cons.append(f"PP={s.pp} adds latency (inter-stage P2P)")

        if s.tp == 1 and isl > 8192:
            cons.append("TP=1 prefill slow for long input sequences")

        for p in pros:
            print(f"       + {p}")
        for c in cons:
            print(f"       - {c}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel strategy advisor for distributed LLM training and inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Training mode (default)
  %(prog)s --model-path /path/to/model --n-gpus 128 --gpu-memory-gb 140

  # Inference mode
  %(prog)s --mode inference --model-path /path/to/model --n-gpus 64
""",
    )
    g = parser.add_argument_group("Mode")
    g.add_argument(
        "--mode",
        type=str,
        default="training",
        choices=["training", "inference"],
        help="Advisor mode (default: training)",
    )

    g = parser.add_argument_group("Model (auto-detect from HF config)")
    g.add_argument(
        "--model-path", type=str, help="Path to HF model directory with config.json"
    )

    g = parser.add_argument_group("Model (manual specification)")
    g.add_argument("--hidden-size", type=int, default=4096)
    g.add_argument("--num-layers", type=int, default=32)
    g.add_argument("--num-heads", type=int, default=32)
    g.add_argument("--num-kv-heads", type=int, default=None)
    g.add_argument("--intermediate-size", type=int, default=11008)
    g.add_argument("--vocab-size", type=int, default=32000)
    g.add_argument("--num-experts", type=int, default=0)
    g.add_argument("--num-experts-per-tok", type=int, default=0)
    g.add_argument("--expert-intermediate-size", type=int, default=0)
    g.add_argument("--n-shared-experts", type=int, default=0)
    g.add_argument("--kv-lora-rank", type=int, default=0)
    g.add_argument("--q-lora-rank", type=int, default=0)
    g.add_argument("--qk-nope-head-dim", type=int, default=0)
    g.add_argument("--qk-rope-head-dim", type=int, default=0)
    g.add_argument("--v-head-dim", type=int, default=0)
    g.add_argument("--group-norm-size", type=int, default=0)
    g.add_argument("--first-k-dense-replace", type=int, default=0)

    g = parser.add_argument_group("Cluster")
    g.add_argument("--n-gpus", type=int, default=128)
    g.add_argument("--gpus-per-node", type=int, default=8)
    g.add_argument("--gpu-memory-gb", type=float, default=0,
                    help="GPU HBM per GPU in GB (0=use GPU preset default)")
    g.add_argument(
        "--host-memory-gb",
        type=float,
        default=0,
        help="Host/CPU memory per node in GB (0=use GPU preset default)",
    )
    g.add_argument(
        "--gpu-type",
        type=str,
        default="H200",
        choices=list(GPU_PRESETS.keys()),
        help="GPU type preset for performance modeling (default: H200)",
    )
    g.add_argument(
        "--gpu-flops",
        type=float,
        default=0,
        help="Override BF16 TFLOPS (e.g., 990 for H100)",
    )
    g.add_argument(
        "--bw-nvlink",
        type=float,
        default=0,
        help="Override NVLink bandwidth in GB/s (e.g., 450)",
    )
    g.add_argument(
        "--bw-ib",
        type=float,
        default=0,
        help="Override IB/RDMA bandwidth in GB/s (e.g., 50)",
    )

    g = parser.add_argument_group("Training")
    g.add_argument("--max-tokens-per-mb", type=int, default=131072)
    g.add_argument("--max-length", type=int, default=16384)
    g.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Global batch size (adds DP divisibility constraint)",
    )
    g.add_argument("--no-optimizer-cpu-offload", action="store_true")
    g.add_argument(
        "--no-grad-reduce-in-fp32",
        action="store_true",
        help="Use bf16 gradient accumulation instead of fp32 (default is fp32)",
    )
    g.add_argument("--no-recompute", action="store_true")
    g.add_argument(
        "--engine",
        type=str,
        default=None,
        choices=["megatron", "fsdp", "sglang"],
        help="Engine preset for cost model (default: megatron for training, sglang for inference)",
    )

    g = parser.add_argument_group("Inference")
    g.add_argument(
        "--isl", type=int, default=4096, help="Input sequence length (default: 4096)"
    )
    g.add_argument(
        "--osl", type=int, default=512, help="Output sequence length (default: 512)"
    )
    g.add_argument(
        "--infer-batch-size",
        type=int,
        default=32,
        help="Inference concurrent batch size (default: 32)",
    )
    g.add_argument(
        "--infer-allow-pp",
        action="store_true",
        help="Allow PP>1 in inference mode (not recommended for SGLang)",
    )

    g = parser.add_argument_group("Output")
    g.add_argument("--json", action="store_true", help="Output as JSON")
    g.add_argument(
        "--top", type=int, default=0, help="Show only top N strategies (0=all)"
    )
    g.add_argument(
        "--find-min-nodes",
        action="store_true",
        help="Find minimum number of nodes needed",
    )

    g = parser.add_argument_group("Profiling")
    g.add_argument(
        "--profile-data",
        type=str,
        default=None,
        help="Path to profiling JSON (auto-detected from --gpu-type if omitted)",
    )
    g.add_argument(
        "--no-profile",
        action="store_true",
        help="Disable profiling data, use pure analytical model",
    )

    # Subcommand: profile
    sub = parser.add_subparsers(dest="subcommand")
    prof = sub.add_parser("profile", help="Run hardware profiling")
    prof.add_argument("--n-nodes", type=int, default=1, choices=[1, 2])
    prof.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "slurm", "ray", "local"],
    )
    prof.add_argument("--reservation", default=None)
    prof.add_argument("--partition", default=None)

    return parser.parse_args(argv)


def _build_hw(args: argparse.Namespace) -> HardwareSpec:
    """Build HardwareSpec from --gpu-type preset with optional overrides."""
    preset = GPU_PRESETS.get(args.gpu_type, HardwareSpec())
    hw = HardwareSpec(
        gpu_flops=args.gpu_flops * 1e12 if args.gpu_flops else preset.gpu_flops,
        bw_nvlink=args.bw_nvlink * 1e9 if args.bw_nvlink else preset.bw_nvlink,
        bw_ib=args.bw_ib * 1e9 if args.bw_ib else preset.bw_ib,
        hbm_bw=preset.hbm_bw,
        gpu_memory_gb=preset.gpu_memory_gb,
        host_memory_gb=preset.host_memory_gb,
    )
    return hw


def _find_min_nodes(
    model: ModelSpec, cluster: ClusterSpec, args: argparse.Namespace
) -> int:
    """Find the minimum number of nodes that have at least one feasible strategy."""
    gpn = cluster.gpus_per_node
    hw = _build_hw(args)
    host_mem = _get_host_memory_gb(getattr(args, "host_memory_gb", 0), hw)
    engine_map = {
        "megatron": MEGATRON_ENGINE,
        "fsdp": FSDP_ENGINE,
        "sglang": SGLANG_ENGINE,
    }
    if args.engine is not None:
        engine = engine_map[args.engine]
    else:
        engine = MEGATRON_ENGINE
    for n_nodes in range(1, 129):
        n_gpus = n_nodes * gpn
        c = ClusterSpec(
            n_gpus=n_gpus, gpus_per_node=gpn, gpu_memory_gb=cluster.gpu_memory_gb
        )
        results = search_strategies(
            model,
            c,
            max_tokens_per_mb=args.max_tokens_per_mb,
            max_length=args.max_length,
            optimizer_cpu_offload=not args.no_optimizer_cpu_offload,
            recompute=not args.no_recompute,
            batch_size=args.batch_size,
            grad_reduce_in_fp32=not args.no_grad_reduce_in_fp32,
            gpus_per_node=gpn,
            hw=hw,
            engine=engine,
        )
        feasible = [
            s
            for s in results
            if s.total_mem_gb <= cluster.gpu_memory_gb
            and (s.cpu_total_per_node_gb <= host_mem or s.cpu_total_per_node_gb == 0)
        ]
        if feasible:
            best = max(feasible, key=lambda s: s.efficiency_score)
            print(f"Minimum nodes: {n_nodes} ({n_gpus} GPUs)")
            print(f"  Best strategy: {best.allocation_mode}")
            print(
                f"  Estimated memory: {best.total_mem_gb:.1f}GB / {cluster.gpu_memory_gb:.0f}GB"
            )
            if best.cpu_total_per_node_gb > 0:
                print(
                    f"  Host memory: {best.cpu_total_per_node_gb:.0f}GB / {host_mem:.0f}GB per node"
                )
            print(f"  DP={best.dp} PP={best.pp} TP={best.tp} CP={best.cp} EP={best.ep}")
            print(f"  Total feasible strategies: {len(feasible)}")
            return 0
    print("No feasible strategy found for up to 128 nodes.")
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    # Handle "profile" subcommand
    if getattr(args, "subcommand", None) == "profile":
        from .profiler.launcher import launch_profiling

        launch_profiling(
            n_nodes=args.n_nodes,
            backend=args.backend,
            gpus_per_node=getattr(args, "gpus_per_node", 8),
            reservation=args.reservation,
            partition=args.partition,
        )
        return 0

    if args.model_path:
        model = ModelSpec.from_hf_config(args.model_path)
    else:
        model = ModelSpec(
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads or args.num_heads,
            intermediate_size=args.intermediate_size,
            vocab_size=args.vocab_size,
            num_experts=args.num_experts,
            num_experts_per_tok=args.num_experts_per_tok,
            expert_intermediate_size=args.expert_intermediate_size,
            n_shared_experts=args.n_shared_experts,
            kv_lora_rank=args.kv_lora_rank,
            q_lora_rank=args.q_lora_rank,
            qk_nope_head_dim=args.qk_nope_head_dim,
            qk_rope_head_dim=args.qk_rope_head_dim,
            v_head_dim=args.v_head_dim,
            group_norm_size=args.group_norm_size,
            first_k_dense_replace=args.first_k_dense_replace,
        )

    preset = GPU_PRESETS.get(args.gpu_type, HardwareSpec())
    gpu_mem = args.gpu_memory_gb if args.gpu_memory_gb > 0 else preset.gpu_memory_gb
    cluster = ClusterSpec(
        n_gpus=args.n_gpus,
        gpus_per_node=args.gpus_per_node,
        gpu_memory_gb=gpu_mem,
    )

    if args.find_min_nodes:
        return _find_min_nodes(model, cluster, args)

    hw = _build_hw(args)
    resolved_host_mem = _get_host_memory_gb(getattr(args, "host_memory_gb", 0), hw)

    # Auto-load profiling data
    profile = None
    if not getattr(args, "no_profile", False):
        from .profile_data.loader import auto_load

        profile = auto_load(
            gpu_type=getattr(args, "gpu_type", ""),
            profile_path=getattr(args, "profile_data", None),
        )
        if profile is not None:
            print("[advisor] Loaded profiling data for cost model enhancement")

    engine_map = {
        "megatron": MEGATRON_ENGINE,
        "fsdp": FSDP_ENGINE,
        "sglang": SGLANG_ENGINE,
    }
    if args.engine is not None:
        engine = engine_map[args.engine]
    elif args.mode == "inference":
        engine = SGLANG_ENGINE
    else:
        engine = MEGATRON_ENGINE

    if args.mode == "inference":
        results = search_inference_strategies(
            model,
            cluster,
            hw,
            isl=args.isl,
            osl=args.osl,
            max_batch_tokens=args.max_tokens_per_mb,
            batch_size=args.infer_batch_size,
            allow_pp=args.infer_allow_pp,
        )
        if args.json:
            out = [asdict(s) for s in results]
            print(json.dumps(out, indent=2))
        else:
            print_inference_results(
                results,
                cluster.gpu_memory_gb,
                model,
                cluster,
                isl=args.isl,
                osl=args.osl,
                batch_size=args.infer_batch_size,
                allow_pp=args.infer_allow_pp,
            )
        return 0

    results = search_strategies(
        model,
        cluster,
        max_tokens_per_mb=args.max_tokens_per_mb,
        max_length=args.max_length,
        optimizer_cpu_offload=not args.no_optimizer_cpu_offload,
        recompute=not args.no_recompute,
        batch_size=args.batch_size,
        grad_reduce_in_fp32=not args.no_grad_reduce_in_fp32,
        gpus_per_node=args.gpus_per_node,
        hw=hw,
        engine=engine,
        profile=profile,
    )

    if args.json:
        fit = [s for s in results if s.total_mem_gb <= cluster.gpu_memory_gb]
        if resolved_host_mem > 0:
            fit = [
                s
                for s in fit
                if s.cpu_total_per_node_gb <= resolved_host_mem
                or s.cpu_total_per_node_gb == 0
            ]
        fit.sort(key=lambda s: -s.efficiency_score)
        if args.top > 0:
            fit = fit[: args.top]
        out = [asdict(s) for s in fit]
        print(json.dumps(out, indent=2))
    else:
        print_results(
            results,
            cluster.gpu_memory_gb,
            model,
            args.max_tokens_per_mb,
            gpus_per_node=args.gpus_per_node,
            batch_size=args.batch_size,
            engine=engine,
            host_memory_gb=resolved_host_mem,
        )

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
