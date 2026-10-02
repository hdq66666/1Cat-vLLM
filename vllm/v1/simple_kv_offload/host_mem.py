# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host memory checks for the pinned SimpleCPUOffload tier.

The CPU tier is pinned (cudaHostRegister'd) anonymous memory: it cannot be
swapped or reclaimed, so it has to fit next to everything else on the host
for the lifetime of the server.
"""

from vllm.logger import init_logger

logger = init_logger(__name__)

GiB = 1024**3


def read_mem_available_bytes() -> int | None:
    """MemAvailable from /proc/meminfo, or None where it is not available."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    # "MemAvailable:   69222068 kB"
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def check_host_mem_before_pinning(
    tier_bytes: int, min_available_gib: float, what: str
) -> None:
    """Refuse to pin ``tier_bytes`` if it would leave too little MemAvailable."""
    available = read_mem_available_bytes()
    if available is None:
        logger.warning("%s: MemAvailable unknown; skipping the host memory check", what)
        return
    remaining = available - tier_bytes
    logger.info(
        "%s: MemAvailable %.2f GiB, CPU KV tier %.2f GiB, %.2f GiB would remain "
        "(minimum %.2f GiB)",
        what,
        available / GiB,
        tier_bytes / GiB,
        remaining / GiB,
        min_available_gib,
    )
    if remaining < min_available_gib * GiB:
        raise ValueError(
            f"{what}: pinning a {tier_bytes / GiB:.2f} GiB CPU KV tier would leave "
            f"{remaining / GiB:.2f} GiB MemAvailable, below "
            f"min_host_mem_available_gib={min_available_gib}. Lower "
            "cpu_bytes_to_use or free host memory."
        )


def check_host_mem_after_pinning(min_available_gib: float, what: str) -> None:
    """Refuse to continue if MemAvailable is already below the floor."""
    available = read_mem_available_bytes()
    if available is None:
        return
    logger.info("%s: MemAvailable %.2f GiB after pinning", what, available / GiB)
    if available < min_available_gib * GiB:
        raise ValueError(
            f"{what}: MemAvailable is {available / GiB:.2f} GiB after the CPU KV "
            f"tier was pinned, below min_host_mem_available_gib="
            f"{min_available_gib}. Lower cpu_bytes_to_use or free host memory."
        )
