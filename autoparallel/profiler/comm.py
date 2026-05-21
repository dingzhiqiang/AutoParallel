"""Communication profiling: measure AllReduce/AllToAll/P2P at various sizes.

Requires torch.distributed (launch via torchrun). Auto-detects topology:
- world_size <= gpus_per_node: NVLink only
- world_size > gpus_per_node: NVLink + IB (cross-node)

Usage (standalone):
    torchrun --nproc_per_node=8 -m autoparallel.profiler.comm

Usage (2-node):
    torchrun --nproc_per_node=8 --nnodes=2 --master-addr=HOST --master-port=PORT \
        -m autoparallel.profiler.comm

Usage (via launcher):
    python -m autoparallel profile --mode comm
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _detect_gpu_type() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "unknown"
        name = torch.cuda.get_device_name(0).upper()
        for t in ("H200", "H100", "H800", "A100", "A800", "L20", "L40"):
            if t in name:
                return t
        return name.split()[0] if name else "unknown"
    except Exception:
        return "unknown"


def _benchmark_op(op_fn, n_warmup: int = 5, n_iters: int = 20) -> float:
    """Benchmark a distributed op, return median time in microseconds."""
    import torch

    # warmup
    for _ in range(n_warmup):
        op_fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(n_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        op_fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0)  # ms → us

    times.sort()
    return times[len(times) // 2]


def profile_comm(
    output_path: str | None = None,
    gpus_per_node: int = 8,
    n_warmup: int = 5,
    n_iters: int = 20,
) -> dict:
    """Profile collective communication across different group sizes and msg sizes."""
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group("nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % gpus_per_node))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    node_id = rank // gpus_per_node
    n_nodes = (world_size + gpus_per_node - 1) // gpus_per_node
    gpu_type = _detect_gpu_type()

    if rank == 0:
        print(f"Comm profiling: {world_size} GPUs, {n_nodes} nodes, {gpu_type}")

    # Message sizes: 1KB to 1GB, log-spaced
    sizes_bytes = [
        1024,
        4096,
        16384,
        65536,
        262144,
        1 * 1024 * 1024,
        4 * 1024 * 1024,
        16 * 1024 * 1024,
        64 * 1024 * 1024,
        256 * 1024 * 1024,
        1024 * 1024 * 1024,
    ]

    entries: list[dict] = []

    # --- NVLink groups (intra-node) ---
    intra_groups: dict[int, list[int]] = {}
    for g in range(world_size):
        nid = g // gpus_per_node
        intra_groups.setdefault(nid, []).append(g)

    my_intra_ranks = intra_groups[node_id]
    dist.new_group(my_intra_ranks)
    intra_size = len(my_intra_ranks)

    for group_size in [2, 4, 8]:
        if group_size > intra_size:
            continue
        sub_ranks = my_intra_ranks[:group_size]
        sub_group = dist.new_group(sub_ranks)
        is_member = rank in sub_ranks

        for op in ["allreduce", "alltoall"]:
            for size_b in sizes_bytes:
                n_elem = size_b // 2  # bf16 = 2 bytes
                if n_elem < group_size:
                    continue

                if op == "allreduce":
                    tensor = torch.randn(n_elem, dtype=torch.bfloat16, device=device)

                    def _op(t=tensor, g=sub_group):
                        dist.all_reduce(t, group=g)

                elif op == "alltoall":
                    inp = torch.randn(n_elem, dtype=torch.bfloat16, device=device)
                    out = torch.empty_like(inp)
                    inp_list = list(inp.chunk(group_size))
                    out_list = list(out.chunk(group_size))

                    def _op(ol=out_list, il=inp_list, g=sub_group):
                        dist.all_to_all(ol, il, group=g)

                if is_member:
                    time_us = _benchmark_op(_op, n_warmup, n_iters)
                else:
                    time_us = 0.0

                dist.barrier()

                if rank == 0:
                    bw = size_b / (time_us * 1e-6) / 1e9 if time_us > 0 else 0.0
                    entries.append(
                        {
                            "op": op,
                            "size_bytes": size_b,
                            "n_gpus": group_size,
                            "topology": "nvlink",
                            "time_us": round(time_us, 2),
                            "bw_GBs": round(bw, 2),
                        }
                    )

                if op == "allreduce":
                    del tensor
                torch.cuda.empty_cache()

    # --- Cross-node (IB) if multi-node ---
    if n_nodes >= 2:
        # AllReduce across all GPUs (mixed topology)
        for op in ["allreduce", "alltoall"]:
            for size_b in sizes_bytes:
                n_elem = size_b // 2
                if n_elem < world_size:
                    continue

                if op == "allreduce":
                    tensor = torch.randn(n_elem, dtype=torch.bfloat16, device=device)

                    def _op(t=tensor):
                        dist.all_reduce(t)

                elif op == "alltoall":
                    inp = torch.randn(n_elem, dtype=torch.bfloat16, device=device)
                    out = torch.empty_like(inp)
                    inp_list = list(inp.chunk(world_size))
                    out_list = list(out.chunk(world_size))

                    def _op(ol=out_list, il=inp_list):
                        dist.all_to_all(ol, il)

                time_us = _benchmark_op(_op, n_warmup, n_iters)
                dist.barrier()

                if rank == 0:
                    bw = size_b / (time_us * 1e-6) / 1e9 if time_us > 0 else 0.0
                    entries.append(
                        {
                            "op": op,
                            "size_bytes": size_b,
                            "n_gpus": world_size,
                            "topology": "mixed",
                            "time_us": round(time_us, 2),
                            "bw_GBs": round(bw, 2),
                        }
                    )

                torch.cuda.empty_cache()

        # P2P cross-node: rank 0 → rank gpus_per_node
        if world_size > gpus_per_node:
            src, dst = 0, gpus_per_node
            p2p_group = dist.new_group([src, dst])
            for size_b in sizes_bytes:
                n_elem = size_b // 2
                tensor = torch.randn(n_elem, dtype=torch.bfloat16, device=device)

                if rank == src:

                    def _op(t=tensor, d=dst, g=p2p_group):
                        dist.send(t, d, group=g)
                elif rank == dst:

                    def _op(t=tensor, s=src, g=p2p_group):
                        dist.recv(t, s, group=g)
                else:
                    _op = None

                if rank in (src, dst):
                    time_us = _benchmark_op(_op, n_warmup, n_iters)
                else:
                    time_us = 0.0

                dist.barrier()

                if rank == 0:
                    bw = size_b / (time_us * 1e-6) / 1e9 if time_us > 0 else 0.0
                    entries.append(
                        {
                            "op": "p2p",
                            "size_bytes": size_b,
                            "n_gpus": 2,
                            "topology": "ib",
                            "time_us": round(time_us, 2),
                            "bw_GBs": round(bw, 2),
                        }
                    )

                torch.cuda.empty_cache()

    if rank == 0:
        result = {
            "gpu_type": gpu_type,
            "metadata": {
                "gpu_name": torch.cuda.get_device_name(0),
                "world_size": world_size,
                "n_nodes": n_nodes,
                "gpus_per_node": gpus_per_node,
                "profiler": "comm",
            },
            "gemm_entries": [],
            "comm_entries": entries,
        }

        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(result, indent=2))
            print(f"Saved {len(entries)} comm entries to {output_path}")

        return result

    return {}


def main():
    parser = argparse.ArgumentParser(description="Profile communication performance")
    parser.add_argument("--output", default=None)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    output = args.output
    if not output:
        gpu = _detect_gpu_type()
        n_nodes = int(os.environ.get("WORLD_SIZE", "1"))
        n_nodes = (int(n_nodes) + args.gpus_per_node - 1) // args.gpus_per_node
        suffix = f"{n_nodes}node" if n_nodes > 1 else "1node"
        output = os.path.expanduser(
            f"~/.cache/parallel_advisor/comm_{gpu}_{suffix}.json"
        )

    profile_comm(output, args.gpus_per_node, args.warmup, args.iters)


if __name__ == "__main__":
    main()
