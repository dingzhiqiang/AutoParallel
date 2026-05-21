"""Profiling launcher: auto-detect environment and run GEMM + comm profiling.

Supports Slurm, Ray, and local (single-node) backends.
Merges results into a single JSON file for the advisor to load.

Usage:
    python -m autoparallel profile [--n-nodes {1,2}] [--backend {auto,slurm,ray,local}]
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _detect_gpu_type() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0).upper()
            for t in ("H200", "H100", "H800", "A100", "A800", "L20", "L40"):
                if t in name:
                    return t
        return "unknown"
    except Exception:
        return "unknown"


def _detect_backend() -> str:
    if shutil.which("sinfo"):
        try:
            subprocess.run(
                ["sinfo", "--version"],
                capture_output=True,
                timeout=5,
            )
            return "slurm"
        except Exception:
            pass

    if os.environ.get("RAY_ADDRESS"):
        return "ray"

    try:
        import ray

        if ray.is_initialized():
            return "ray"
    except ImportError:
        pass

    return "local"


def _find_free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _get_cache_dir() -> Path:
    d = Path(os.path.expanduser("~/.cache/autoparallel"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run_gemm_local(cache_dir: Path, gpu_type: str) -> Path:
    """Run GEMM profiling on local GPU."""
    out = cache_dir / f"gemm_{gpu_type}.json"
    cmd = [
        sys.executable,
        "-m",
        "autoparallel.profiler.gemm",
        "--output",
        str(out),
    ]
    print(f"[launcher] Running GEMM profiling → {out}")
    subprocess.run(cmd, check=True)
    return out


def _run_comm_local(cache_dir: Path, gpu_type: str, gpus_per_node: int) -> Path:
    """Run comm profiling on local node via torchrun."""
    out = cache_dir / f"comm_{gpu_type}_1node.json"
    port = _find_free_port()
    cmd = [
        "torchrun",
        f"--nproc_per_node={gpus_per_node}",
        "--nnodes=1",
        "--master-addr=localhost",
        f"--master-port={port}",
        "-m",
        "autoparallel.profiler.comm",
        "--output",
        str(out),
        "--gpus-per-node",
        str(gpus_per_node),
    ]
    print(f"[launcher] Running comm profiling (1 node, {gpus_per_node} GPUs) → {out}")
    subprocess.run(cmd, check=True)
    return out


def _run_slurm(
    cache_dir: Path,
    gpu_type: str,
    n_nodes: int,
    gpus_per_node: int,
    reservation: str | None = None,
    partition: str | None = None,
) -> tuple[Path, Path]:
    """Run GEMM + comm profiling via Slurm srun."""
    gemm_out = cache_dir / f"gemm_{gpu_type}.json"
    comm_out = cache_dir / f"comm_{gpu_type}_{n_nodes}node.json"

    # GEMM: single GPU
    srun_base = ["srun", "-N1", "--gres=gpu:1", "--ntasks=1"]
    if reservation:
        srun_base += [f"--reservation={reservation}"]
    if partition:
        srun_base += [f"--partition={partition}"]

    gemm_cmd = srun_base + [
        sys.executable,
        "-m",
        "autoparallel.profiler.gemm",
        "--output",
        str(gemm_out),
    ]
    print(f"[launcher] Slurm GEMM profiling → {gemm_out}")
    subprocess.run(gemm_cmd, check=True)

    # Comm: multi-GPU via torchrun
    srun_comm = [
        "srun",
        f"-N{n_nodes}",
        f"--gres=gpu:{gpus_per_node}",
        "--ntasks-per-node=1",
    ]
    if reservation:
        srun_comm += [f"--reservation={reservation}"]
    if partition:
        srun_comm += [f"--partition={partition}"]

    # Use srun to launch torchrun on each node
    comm_script = f"""#!/bin/bash
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
export MASTER_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
torchrun --nproc_per_node={gpus_per_node} --nnodes={n_nodes} \
    --node-rank=$SLURM_NODEID --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT \
    -m autoparallel.profiler.comm \
    --output {comm_out} --gpus-per-node {gpus_per_node}
"""
    script_path = cache_dir / "_comm_profile.sh"
    script_path.write_text(comm_script)
    script_path.chmod(0o755)

    srun_comm += ["bash", str(script_path)]
    print(f"[launcher] Slurm comm profiling ({n_nodes} nodes) → {comm_out}")
    subprocess.run(srun_comm, check=True)

    return gemm_out, comm_out


def _run_ray(
    cache_dir: Path,
    gpu_type: str,
    n_nodes: int,
    gpus_per_node: int,
) -> tuple[Path, Path]:
    """Run profiling via Ray."""
    import ray

    if not ray.is_initialized():
        ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))

    gemm_out = cache_dir / f"gemm_{gpu_type}.json"

    # GEMM: single GPU task
    @ray.remote(num_gpus=1)
    def _ray_gemm(out_path: str):
        from autoparallel.profiler.gemm import profile_gemm

        profile_gemm(out_path)

    print(f"[launcher] Ray GEMM profiling → {gemm_out}")
    ray.get(_ray_gemm.remote(str(gemm_out)))

    # Comm: use placement group for multi-node
    comm_out = cache_dir / f"comm_{gpu_type}_{n_nodes}node.json"
    pg = ray.util.placement_group(
        bundles=[{"GPU": gpus_per_node}] * n_nodes,
        strategy="STRICT_SPREAD",
    )
    ray.get(pg.ready())

    @ray.remote(num_gpus=gpus_per_node)
    def _ray_comm(out_path: str, gpn: int):
        port = _find_free_port()
        cmd = [
            "torchrun",
            f"--nproc_per_node={gpn}",
            "--nnodes=1",
            "--master-addr=localhost",
            f"--master-port={port}",
            "-m",
            "autoparallel.profiler.comm",
            "--output",
            out_path,
            "--gpus-per-node",
            str(gpn),
        ]
        subprocess.run(cmd, check=True)

    # For simplicity, run comm on each node independently and merge
    tasks = []
    for i in range(n_nodes):
        node_out = str(cache_dir / f"comm_{gpu_type}_node{i}.json")
        t = _ray_comm.options(
            placement_group=pg,
            placement_group_bundle_index=i,
        ).remote(node_out, gpus_per_node)
        tasks.append((t, node_out))

    ray.get([t for t, _ in tasks])

    # Take first node's results (NVLink is same across nodes)
    first_node_path = tasks[0][1]
    if Path(first_node_path).exists():
        import shutil as _sh

        _sh.copy2(first_node_path, comm_out)

    # Cleanup temp files
    for _, p in tasks:
        Path(p).unlink(missing_ok=True)

    ray.util.remove_placement_group(pg)

    return gemm_out, comm_out


def _merge_results(cache_dir: Path, gpu_type: str, *json_paths: Path) -> Path:
    """Merge GEMM and comm JSON files into a single profile."""
    from .loader import ProfilingData

    merged = ProfilingData(gpu_type=gpu_type)
    for p in json_paths:
        if p.exists():
            part = ProfilingData.load(p)
            merged.merge(part)

    out = cache_dir / f"{gpu_type}.json"
    merged.save(out)
    print(
        f"[launcher] Merged profile → {out} "
        f"({len(merged.gemm_entries)} GEMM + {len(merged.comm_entries)} comm entries)"
    )
    return out


def launch_profiling(
    n_nodes: int = 1,
    backend: str = "auto",
    gpus_per_node: int = 8,
    reservation: str | None = None,
    partition: str | None = None,
) -> str:
    """Run full profiling pipeline and return path to merged result."""
    cache_dir = _get_cache_dir()
    gpu_type = _detect_gpu_type()

    if backend == "auto":
        backend = _detect_backend()
    print(f"[launcher] Backend: {backend}, GPU: {gpu_type}, nodes: {n_nodes}")

    if backend == "slurm":
        gemm_out, comm_out = _run_slurm(
            cache_dir, gpu_type, n_nodes, gpus_per_node, reservation, partition
        )
    elif backend == "ray":
        gemm_out, comm_out = _run_ray(cache_dir, gpu_type, n_nodes, gpus_per_node)
    else:
        gemm_out = _run_gemm_local(cache_dir, gpu_type)
        comm_out = _run_comm_local(cache_dir, gpu_type, gpus_per_node)

    merged = _merge_results(cache_dir, gpu_type, gemm_out, comm_out)
    return str(merged)


def main():
    """CLI entry point for profiling."""
    import argparse

    parser = argparse.ArgumentParser(description="Run hardware profiling")
    parser.add_argument("--n-nodes", type=int, default=1, choices=[1, 2])
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "slurm", "ray", "local"],
    )
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--reservation", default=None)
    parser.add_argument("--partition", default=None)
    args = parser.parse_args()

    result_path = launch_profiling(
        n_nodes=args.n_nodes,
        backend=args.backend,
        gpus_per_node=args.gpus_per_node,
        reservation=args.reservation,
        partition=args.partition,
    )
    print(f"\nProfile saved to: {result_path}")
    print("The advisor will auto-load this on next run.")


if __name__ == "__main__":
    main()
