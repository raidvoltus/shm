"""
Resource monitor — cgroup-aware memory (container limit), not host-only.

Priority:
1. cgroup v2: memory.max / memory.current
2. cgroup v1: memory.limit_in_bytes / memory.usage_in_bytes
3. psutil fallback (local dev)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _read_int(path: Path) -> Optional[int]:
    try:
        text = path.read_text(encoding="utf-8").strip()
        if text in ("max", "unlimited", ""):
            return None  # unlimited
        val = int(text)
        # kernel often uses very large values for "unlimited"
        if val > 1 << 60:
            return None
        return val
    except (OSError, ValueError):
        return None


def read_cgroup_memory() -> tuple[Optional[int], Optional[int], str]:
    """Return (limit_bytes, usage_bytes, source)."""
    # cgroup v2
    v2_max = Path("/sys/fs/cgroup/memory.max")
    v2_cur = Path("/sys/fs/cgroup/memory.current")
    if v2_max.exists():
        limit = _read_int(v2_max)
        usage = _read_int(v2_cur) if v2_cur.exists() else None
        return limit, usage, "cgroup_v2"

    # cgroup v1
    v1_limit = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    v1_usage = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if v1_limit.exists():
        limit = _read_int(v1_limit)
        usage = _read_int(v1_usage) if v1_usage.exists() else None
        return limit, usage, "cgroup_v1"

    return None, None, "none"


def read_psutil_memory() -> tuple[Optional[int], Optional[int]]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.used)
    except Exception:
        # minimal fallback via /proc/meminfo
        try:
            total = available = None
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
            if total is not None:
                used = total - (available or 0)
                return total, used
        except Exception:
            pass
    return None, None


def read_cpu() -> tuple[int, Optional[float]]:
    cpu_count = os.cpu_count() or 1
    load = None
    try:
        load1, _, _ = os.getloadavg()
        load = load1 / max(cpu_count, 1)
    except (OSError, AttributeError):
        pass
    return cpu_count, load


def read_disk_free(path: str = "/") -> Optional[int]:
    try:
        st = os.statvfs(path)
        return int(st.f_bavail * st.f_frsize)
    except OSError:
        return None


@dataclass(frozen=True)
class ResourceSnapshot:
    memory_limit_bytes: Optional[int]
    memory_used_bytes: Optional[int]
    memory_available_bytes: Optional[int]
    memory_source: str
    cpu_count: int
    cpu_load: Optional[float]
    disk_available_bytes: Optional[int]

    @staticmethod
    def capture(*, inject: Optional[dict] = None) -> "ResourceSnapshot":
        """
        Capture live resources, or use inject={...} for adversarial tests.
        inject keys override detected values.
        """
        if inject is not None:
            limit = inject.get("memory_limit_bytes")
            used = inject.get("memory_used_bytes", 0)
            avail = inject.get("memory_available_bytes")
            if avail is None and limit is not None and used is not None:
                avail = max(0, limit - used)
            return ResourceSnapshot(
                memory_limit_bytes=limit,
                memory_used_bytes=used,
                memory_available_bytes=avail,
                memory_source=inject.get("memory_source", "injected"),
                cpu_count=int(inject.get("cpu_count", 1)),
                cpu_load=inject.get("cpu_load"),
                disk_available_bytes=inject.get("disk_available_bytes"),
            )

        limit, used, source = read_cgroup_memory()
        if limit is None and used is None:
            pt, pu = read_psutil_memory()
            if pt is not None:
                limit, used, source = pt, pu, "psutil_or_proc"
        avail = None
        if limit is not None and used is not None:
            avail = max(0, limit - used)
        elif limit is None and used is not None:
            # unlimited cgroup — treat available as unknown large
            avail = None
        cpu_count, cpu_load = read_cpu()
        disk = read_disk_free()
        return ResourceSnapshot(
            memory_limit_bytes=limit,
            memory_used_bytes=used,
            memory_available_bytes=avail,
            memory_source=source,
            cpu_count=cpu_count,
            cpu_load=cpu_load,
            disk_available_bytes=disk,
        )

    def to_dict(self) -> dict:
        return {
            "memory_limit_bytes": self.memory_limit_bytes,
            "memory_used_bytes": self.memory_used_bytes,
            "memory_available_bytes": self.memory_available_bytes,
            "memory_source": self.memory_source,
            "cpu_count": self.cpu_count,
            "cpu_load": self.cpu_load,
            "disk_available_bytes": self.disk_available_bytes,
        }
