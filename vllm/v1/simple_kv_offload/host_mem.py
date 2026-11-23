# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host memory checks for the pinned SimpleCPUOffload tier.

The CPU tier is pinned (cudaHostRegister'd) anonymous memory: it cannot be
swapped or reclaimed, so it has to fit next to everything else on the host
for the lifetime of the server.
"""

import time

from vllm.logger import init_logger

logger = init_logger(__name__)

GiB = 1024**3

# While the server starts, the other processes (model loading, CUDA graph
# capture, warmup) briefly hold extra host memory, so one MemAvailable reading
# can sit several GiB below the steady state. With the same configuration the
# reading after pinning ranged from 9.3 to 18.8 GiB on one host.
SETTLE_POLL_S = 2.0


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


def _settled_mem_available(
    needed_bytes: int, settle_s: float
) -> tuple[int | None, int | None]:
    """Read MemAvailable until it reaches ``needed_bytes`` or ``settle_s``
    seconds have passed. Returns (last reading, lowest reading); (None, None)
    when MemAvailable cannot be read."""
    available = read_mem_available_bytes()
    if available is None:
        return None, None
    lowest = available
    deadline = time.monotonic() + settle_s
    while available < needed_bytes and time.monotonic() < deadline:
        time.sleep(min(SETTLE_POLL_S, max(0.0, deadline - time.monotonic())))
        reading = read_mem_available_bytes()
        if reading is None:
            return None, None
        available = reading
        lowest = min(lowest, reading)
    return available, lowest


def check_host_mem_before_pinning(
    tier_bytes: int, min_available_gib: float, what: str, settle_s: float = 0.0
) -> None:
    """Refuse to pin ``tier_bytes`` if it would leave too little MemAvailable.

    With ``settle_s`` > 0 a reading below the floor is retried for up to that
    many seconds, so a transient dip does not refuse the start."""
    available, lowest = _settled_mem_available(
        tier_bytes + int(min_available_gib * GiB), settle_s
    )
    if available is None or lowest is None:
        logger.warning("%s: MemAvailable unknown; skipping the host memory check", what)
        return
    if lowest < available:
        logger.info(
            "%s: MemAvailable rose from %.2f to %.2f GiB while waiting",
            what,
            lowest / GiB,
            available / GiB,
        )
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
            f"min_host_mem_available_gib={min_available_gib}"
            + (f" (waited {settle_s:.0f} s)" if settle_s > 0 else "")
            + ". Lower cpu_bytes_to_use or free host memory."
        )


def check_host_mem_after_pinning(
    min_available_gib: float, what: str, settle_s: float = 0.0
) -> None:
    """Refuse to continue if MemAvailable is below the floor after pinning.

    With ``settle_s`` > 0 a reading below the floor is retried for up to that
    many seconds, so a transient dip does not refuse the start."""
    available, lowest = _settled_mem_available(int(min_available_gib * GiB), settle_s)
    if available is None or lowest is None:
        return
    logger.info(
        "%s: MemAvailable %.2f GiB after pinning%s",
        what,
        available / GiB,
        f" (lowest {lowest / GiB:.2f} GiB while waiting)" if lowest < available else "",
    )
    if available < min_available_gib * GiB:
        raise ValueError(
            f"{what}: MemAvailable is {available / GiB:.2f} GiB after the CPU KV "
            f"tier was pinned, below min_host_mem_available_gib="
            f"{min_available_gib}"
            + (f" (waited {settle_s:.0f} s)" if settle_s > 0 else "")
            + ". Lower cpu_bytes_to_use or free host memory."
        )
