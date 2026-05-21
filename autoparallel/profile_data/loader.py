"""Profiling data structures, loading, and interpolation."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

_PRESETS_DIR = Path(__file__).parent / "presets"
_CACHE_DIR = Path(os.path.expanduser("~/.cache/autoparallel"))


@dataclass
class GEMMEntry:
    """Single GEMM measurement point."""

    M: int
    N: int
    K: int
    dtype: str = "bf16"
    time_us: float = 0.0
    tflops: float = 0.0


@dataclass
class CommEntry:
    """Single collective communication measurement point."""

    op: str  # "allreduce" | "alltoall" | "p2p"
    size_bytes: int = 0
    n_gpus: int = 0
    topology: str = ""  # "nvlink" | "ib" | "mixed"
    time_us: float = 0.0
    bw_GBs: float = 0.0


@dataclass
class ProfilingData:
    """Container for all profiling measurements."""

    gpu_type: str = ""
    gemm_entries: list[GEMMEntry] = field(default_factory=list)
    comm_entries: list[CommEntry] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "gpu_type": self.gpu_type,
            "metadata": self.metadata,
            "gemm_entries": [asdict(e) for e in self.gemm_entries],
            "comm_entries": [asdict(e) for e in self.comm_entries],
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> ProfilingData:
        data = json.loads(Path(path).read_text())
        return cls(
            gpu_type=data.get("gpu_type", ""),
            metadata=data.get("metadata", {}),
            gemm_entries=[GEMMEntry(**e) for e in data.get("gemm_entries", [])],
            comm_entries=[CommEntry(**e) for e in data.get("comm_entries", [])],
        )

    def merge(self, other: ProfilingData) -> None:
        """Merge another ProfilingData into this one (e.g. gemm + comm)."""
        self.gemm_entries.extend(other.gemm_entries)
        self.comm_entries.extend(other.comm_entries)
        self.metadata.update(other.metadata)
        if not self.gpu_type:
            self.gpu_type = other.gpu_type


def _log_interp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    """Log-space linear interpolation."""
    if x0 == x1:
        return y0
    lx0, lx1 = math.log(max(x0, 1)), math.log(max(x1, 1))
    lx = math.log(max(x, 1))
    t = (lx - lx0) / (lx1 - lx0)
    t = max(0.0, min(1.0, t))
    return y0 + t * (y1 - y0)


class ProfileLookup:
    """Fast lookup with log-space interpolation over ProfilingData."""

    def __init__(self, data: ProfilingData):
        self._data = data
        self._gemm_index: dict[tuple[int, int], list[GEMMEntry]] = {}
        self._comm_index: dict[tuple[str, int, str], list[CommEntry]] = {}
        self._build_indices()

    def _build_indices(self) -> None:
        for e in self._data.gemm_entries:
            key = (e.N, e.K)
            self._gemm_index.setdefault(key, []).append(e)
        for entries in self._gemm_index.values():
            entries.sort(key=lambda e: e.M)

        for e in self._data.comm_entries:
            key = (e.op, e.n_gpus, e.topology)
            self._comm_index.setdefault(key, []).append(e)
        for entries in self._comm_index.values():
            entries.sort(key=lambda e: e.size_bytes)

    def gemm_time_us(self, M: int, N: int, K: int) -> float | None:
        """Look up GEMM time in microseconds. Returns None if no data."""
        entries = self._gemm_index.get((N, K))
        if not entries:
            entries = self._find_nearest_nk(N, K)
            if not entries:
                return None
        return self._interp_by_m(entries, M)

    def comm_time_us(
        self, op: str, size_bytes: int, n_gpus: int, topology: str
    ) -> float | None:
        """Look up collective comm time in microseconds. Returns None if no data."""
        entries = self._comm_index.get((op, n_gpus, topology))
        if not entries:
            return None
        return self._interp_by_size(entries, size_bytes)

    def _find_nearest_nk(self, N: int, K: int) -> list[GEMMEntry] | None:
        """Find entries with closest (N, K) using log distance."""
        if not self._gemm_index:
            return None
        best_key = None
        best_dist = float("inf")
        for nk in self._gemm_index:
            dn = abs(math.log(max(nk[0], 1)) - math.log(max(N, 1)))
            dk = abs(math.log(max(nk[1], 1)) - math.log(max(K, 1)))
            dist = dn + dk
            if dist < best_dist:
                best_dist = dist
                best_key = nk
        if best_key is None or best_dist > 1.0:
            return None
        return self._gemm_index[best_key]

    @staticmethod
    def _interp_by_m(entries: list[GEMMEntry], M: int) -> float:
        if len(entries) == 1:
            e = entries[0]
            return e.time_us * M / e.M if e.M > 0 else e.time_us

        for i, e in enumerate(entries):
            if e.M == M:
                return e.time_us
            if e.M > M:
                if i == 0:
                    return _log_interp(
                        M,
                        entries[0].M,
                        entries[1].M,
                        entries[0].time_us,
                        entries[1].time_us,
                    )
                return _log_interp(
                    M, entries[i - 1].M, e.M, entries[i - 1].time_us, e.time_us
                )
        return _log_interp(
            M, entries[-2].M, entries[-1].M, entries[-2].time_us, entries[-1].time_us
        )

    @staticmethod
    def _interp_by_size(entries: list[CommEntry], size_bytes: int) -> float:
        if len(entries) == 1:
            e = entries[0]
            return (
                e.time_us * size_bytes / e.size_bytes if e.size_bytes > 0 else e.time_us
            )

        for i, e in enumerate(entries):
            if e.size_bytes == size_bytes:
                return e.time_us
            if e.size_bytes > size_bytes:
                if i == 0:
                    return _log_interp(
                        size_bytes,
                        entries[0].size_bytes,
                        entries[1].size_bytes,
                        entries[0].time_us,
                        entries[1].time_us,
                    )
                return _log_interp(
                    size_bytes,
                    entries[i - 1].size_bytes,
                    e.size_bytes,
                    entries[i - 1].time_us,
                    e.time_us,
                )
        return _log_interp(
            size_bytes,
            entries[-2].size_bytes,
            entries[-1].size_bytes,
            entries[-2].time_us,
            entries[-1].time_us,
        )


def auto_load(
    gpu_type: str = "", profile_path: str | None = None
) -> ProfileLookup | None:
    """Auto-detect and load profiling data.

    Priority:
    1. Explicit path (--profile-data)
    2. ~/.cache/autoparallel/{gpu_type}.json
    3. Built-in presets
    4. None (pure analytical fallback)
    """
    if profile_path:
        p = Path(profile_path)
        if p.exists():
            return ProfileLookup(ProfilingData.load(p))

    if gpu_type:
        cache_path = _CACHE_DIR / f"{gpu_type}.json"
        if cache_path.exists():
            return ProfileLookup(ProfilingData.load(cache_path))

        for preset in _PRESETS_DIR.glob(f"{gpu_type}*.json"):
            return ProfileLookup(ProfilingData.load(preset))

    return None
