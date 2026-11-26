"""Where this process runs on a CPU host: the CPUs and memory nodes it may use, which
sockets those span, their last-level caches, and the threading settings it inherited.

Facts only, read from sysfs and procfs. X-Forge binds nothing: the launcher places the
run (``numactl``, ``taskset``, the scheduler's cpuset) and this records what it got, so a
calibration or a timing can be tied to the placement it was measured under.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# Variables that set how threading runtimes place and size their pools.
THREAD_ENV = re.compile(
    r"^(OMP|KMP|GOMP|MKL|DNNL|ONEDNN|TBB|OPENBLAS|BLIS|NUMEXPR|VECLIB|DPCPP_CPU|SYCL)_"
)


def parse_list(text: str) -> list[int]:
    """``"0-3,8,10-11"`` -> ``[0, 1, 2, 3, 8, 10, 11]``."""
    out: list[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def format_list(ids) -> str:
    """The inverse of :func:`parse_list`, in compact ranges."""
    ids, parts = sorted(set(ids)), []
    for i in ids:
        if parts and parts[-1][1] == i - 1:
            parts[-1][1] = i
        else:
            parts.append([i, i])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in parts)


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _size_bytes(text: str | None) -> int | None:
    m = re.fullmatch(r"(\d+)([KMG]?)", text or "")
    return int(m[1]) * {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30}[m[2]] if m else None


def model_name(root: Path = Path("/")) -> str | None:
    for line in (_read(root / "proc/cpuinfo") or "").splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "model name":
            return value.strip()
    return None


def cpu_flags(root: Path = Path("/")) -> list[str]:
    """The instruction-set and feature flags the kernel reports for the first CPU."""
    for line in (_read(root / "proc/cpuinfo") or "").splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "flags":
            return value.split()
    return []


def placement(root: Path = Path("/"), cpus=None, env=None) -> dict:
    """The placement this process inherited. ``cpus`` defaults to its affinity mask."""
    cpus = sorted(os.sched_getaffinity(0) if cpus is None else cpus)
    sysfs = root / "sys/devices/system"
    status = _read(root / "proc/self/status") or ""
    mems = re.search(r"^Mems_allowed_list:\s*(\S+)", status, re.M)

    sockets, smt_siblings, caches = set(), 0, {}
    for c in cpus:
        topo = sysfs / f"cpu/cpu{c}/topology"
        pkg = _read(topo / "physical_package_id")
        if pkg is not None:
            sockets.add(int(pkg))
        sib = _read(topo / "thread_siblings_list")
        allowed = set(parse_list(sib or "")) & set(cpus)
        if len(allowed) > 1 and c != min(allowed):
            smt_siblings += 1  # a second hardware thread of a core already counted
        cache_dir = sysfs / f"cpu/cpu{c}/cache"
        levels = [(int(_read(d / "level") or 0), d) for d in cache_dir.glob("index*")]
        top = max((lv for lv, _ in levels), default=0)
        for lv, d in levels:
            shared = _read(d / "shared_cpu_list")
            if lv == top and shared and _read(d / "type") != "Instruction":
                caches[shared] = _size_bytes(_read(d / "size"))

    nodes = {}
    for d in sorted(sysfs.glob("node/node[0-9]*")):
        node_cpus = set(parse_list(_read(d / "cpulist") or ""))
        if node_cpus & set(cpus):
            nodes[int(d.name[4:])] = format_list(node_cpus & set(cpus))

    # The policy the launcher set (numactl --membind/--interleave/--preferred): numa_maps
    # shows it on every mapping that has no policy of its own, the first one included.
    maps = (_read(root / "proc/self/numa_maps") or "").split("\n", 1)[0].split()
    env = os.environ if env is None else env
    return {
        "cpus": format_list(cpus),
        "cpu_count": len(cpus),
        "smt_siblings": smt_siblings,  # hardware threads beyond the first of their core
        "sockets": sorted(sockets),
        "numa_nodes": nodes,  # node -> the allowed CPUs on it
        "memory_nodes": mems[1] if mems else None,  # the nodes its cpuset allows
        "memory_policy": maps[1] if len(maps) > 1 else None,  # e.g. default, bind:0
        # The last level, the CPUs' slices; unknown if any slice's size is.
        "llc_bytes": sum(caches.values()) if caches and all(caches.values()) else None,
        "thread_env": {k: v for k, v in sorted(env.items()) if THREAD_ENV.match(k)},
    }


def placement_tag(p: dict) -> str:
    """A short name for a placement, the same for the same CPUs and memory nodes."""
    tag = f"cpus{p['cpus']}_mem{p.get('memory_nodes') or 'any'}"
    policy = p.get("memory_policy")
    if policy and policy != "default":
        tag += f"_{policy}"
    return re.sub(r"[^\w.-]", "_", tag)
