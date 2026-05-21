"""GEMM profiling: measure matmul performance at various (M, N, K) sizes.

Runs on a single GPU, takes ~3 minutes. Output is a JSON file with
GEMMEntry records that ProfileLookup can interpolate over.

Usage (standalone):
    python -m autoparallel.profiler.gemm [--output PATH]

Usage (via launcher):
    python -m autoparallel profile --mode gemm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _detect_gpu_type() -> str:
    """Best-effort GPU type detection from torch."""
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


def profile_gemm(
    output_path: str | None = None,
    dtype: str = "bf16",
    n_warmup: int = 5,
    n_iters: int = 20,
) -> dict:
    """Profile GEMM across a grid of (M, N, K) sizes.

    Returns dict with gemm_entries list, also writes to output_path if given.
    """
    import torch

    if not torch.cuda.is_available():
        print("ERROR: No GPU available", file=sys.stderr)
        sys.exit(1)

    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    device = torch.device("cuda:0")

    M_list = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    N_list = [1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576]
    K_list = [1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576]

    gpu_type = _detect_gpu_type()
    total = len(M_list) * len(N_list) * len(K_list)
    print(f"Profiling GEMM on {gpu_type} ({torch.cuda.get_device_name(0)})")
    print(f"  dtype={dtype}, warmup={n_warmup}, iters={n_iters}")
    print(
        f"  Grid: {len(M_list)} M x {len(N_list)} N x {len(K_list)} K = {total} points"
    )

    entries = []
    done = 0
    t0 = time.time()

    for M in M_list:
        for N in N_list:
            for K in K_list:
                A = torch.randn(M, K, dtype=torch_dtype, device=device)
                B = torch.randn(K, N, dtype=torch_dtype, device=device)

                # warmup
                for _ in range(n_warmup):
                    torch.matmul(A, B)
                torch.cuda.synchronize()

                # measure
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(n_iters):
                    torch.matmul(A, B)
                end.record()
                torch.cuda.synchronize()

                elapsed_ms = start.elapsed_time(end) / n_iters
                elapsed_us = elapsed_ms * 1000.0
                flops = 2.0 * M * N * K
                tflops = flops / (elapsed_us * 1e6) if elapsed_us > 0 else 0.0

                entries.append(
                    {
                        "M": M,
                        "N": N,
                        "K": K,
                        "dtype": dtype,
                        "time_us": round(elapsed_us, 2),
                        "tflops": round(tflops, 1),
                    }
                )

                del A, B

                done += 1
                if done % 100 == 0 or done == total:
                    elapsed = time.time() - t0
                    print(f"  [{done}/{total}] {elapsed:.0f}s elapsed")

    torch.cuda.empty_cache()

    result = {
        "gpu_type": gpu_type,
        "metadata": {
            "gpu_name": torch.cuda.get_device_name(0),
            "dtype": dtype,
            "n_warmup": n_warmup,
            "n_iters": n_iters,
            "profiler": "gemm",
        },
        "gemm_entries": entries,
        "comm_entries": [],
    }

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2))
        print(f"Saved {len(entries)} GEMM entries to {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Profile GEMM performance")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: ~/.cache/parallel_advisor/gemm_{gpu}.json)",
    )
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    output = args.output
    if not output:
        gpu = _detect_gpu_type()
        output = os.path.expanduser(f"~/.cache/parallel_advisor/gemm_{gpu}.json")

    profile_gemm(output, args.dtype, args.warmup, args.iters)


if __name__ == "__main__":
    main()
