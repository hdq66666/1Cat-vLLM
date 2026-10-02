# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Options of SimpleCPUOffloadConnector, parsed from kv_connector_extra_config."""

from dataclasses import dataclass
from typing import Any

import vllm.envs as envs

# Default CPU capacity: 8 GB
DEFAULT_CPU_CAPACITY_BYTES = 8 * (1024**3)

COPY_BACKENDS = ("auto", "batch", "memcpy")

# Fault-injection switches that are only honoured with
# VLLM_SIMPLE_KV_OFFLOAD_TEST_MODE=1.
_TEST_SWITCHES = (
    "VLLM_SIMPLE_KV_OFFLOAD_TEST_UNORDERED_LOAD",
    "VLLM_SIMPLE_KV_OFFLOAD_TEST_NO_SKIP_ZERO",
    "VLLM_SIMPLE_KV_OFFLOAD_TEST_FAIL_COPY",
    "VLLM_SIMPLE_KV_OFFLOAD_DEBUG_ZERO_DELAY_CYCLES",
    "VLLM_SIMPLE_KV_OFFLOAD_DEBUG_COMPUTE_DELAY_CYCLES",
)


@dataclass(frozen=True)
class SimpleCPUOffloadOptions:
    # Server-wide CPU capacity (divided by world_size) unless
    # cpu_bytes_to_use_per_rank is given.
    cpu_bytes_to_use: int = DEFAULT_CPU_CAPACITY_BYTES
    cpu_bytes_to_use_per_rank: int | None = None
    lazy_offload: bool = False
    # "auto" self-tests cuMemcpyBatchAsync at startup and falls back to
    # per-block cudaMemcpyAsync; "batch" / "memcpy" force one of them.
    copy_backend: str = "auto"
    # Lazy mode: at most this many GPU blocks are selected for store per step.
    max_store_blocks_per_step: int = 16
    # Lazy mode: stop selecting stores when the GPU free pool would drop to
    # this many blocks. None = max_num_seqs * (1 + number of Mamba groups).
    store_reserve_free_blocks: int | None = None
    # Lazy mode: store a GPU block again after its CPU copy was evicted.
    allow_restore: bool = False
    # Refuse to start when host MemAvailable minus the CPU tier would fall
    # below this many GiB.
    min_host_mem_available_gib: float = 15.0
    # Log a warning while host MemAvailable is below this many GiB.
    host_mem_warn_gib: float = 12.0
    # A load or store that does not complete within this many seconds is
    # treated as a dead transfer and fails the engine.
    transfer_timeout_s: float = 60.0

    def cpu_bytes_per_rank(self, world_size: int) -> int:
        if self.cpu_bytes_to_use_per_rank is not None:
            return self.cpu_bytes_to_use_per_rank
        return self.cpu_bytes_to_use // world_size

    def cpu_bytes_total(self, world_size: int) -> int:
        return self.cpu_bytes_per_rank(world_size) * world_size


def _as_bool(key: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    # bool("false") is True; refuse anything that is not a real boolean.
    raise ValueError(
        f"kv_connector_extra_config[{key!r}] must be a JSON boolean, got {value!r}"
    )


def _as_int(key: str, value: Any, minimum: int) -> int:
    # --kv-offloading-size hands cpu_bytes_to_use over as a float.
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(
            f"kv_connector_extra_config[{key!r}] must be an integer, got {value!r}"
        )
    result = int(value)
    if result < minimum:
        raise ValueError(
            f"kv_connector_extra_config[{key!r}] must be >= {minimum}, got {result}"
        )
    return result


def _as_float(key: str, value: Any, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(
            f"kv_connector_extra_config[{key!r}] must be a number, got {value!r}"
        )
    result = float(value)
    if result < minimum:
        raise ValueError(
            f"kv_connector_extra_config[{key!r}] must be >= {minimum}, got {result}"
        )
    return result


def parse_options(extra_config: dict[str, Any] | None) -> SimpleCPUOffloadOptions:
    """Validate kv_connector_extra_config. Unknown keys are ignored."""
    cfg = extra_config or {}
    kwargs: dict[str, Any] = {}
    if "cpu_bytes_to_use" in cfg:
        kwargs["cpu_bytes_to_use"] = _as_int(
            "cpu_bytes_to_use", cfg["cpu_bytes_to_use"], 1
        )
    if "cpu_bytes_to_use_per_rank" in cfg:
        kwargs["cpu_bytes_to_use_per_rank"] = _as_int(
            "cpu_bytes_to_use_per_rank", cfg["cpu_bytes_to_use_per_rank"], 1
        )
    for key in ("lazy_offload", "allow_restore"):
        if key in cfg:
            kwargs[key] = _as_bool(key, cfg[key])
    if "copy_backend" in cfg:
        backend = cfg["copy_backend"]
        if backend not in COPY_BACKENDS:
            raise ValueError(
                "kv_connector_extra_config['copy_backend'] must be one of "
                f"{COPY_BACKENDS}, got {backend!r}"
            )
        kwargs["copy_backend"] = backend
    if "max_store_blocks_per_step" in cfg:
        kwargs["max_store_blocks_per_step"] = _as_int(
            "max_store_blocks_per_step", cfg["max_store_blocks_per_step"], 1
        )
    if cfg.get("store_reserve_free_blocks") is not None:
        kwargs["store_reserve_free_blocks"] = _as_int(
            "store_reserve_free_blocks", cfg["store_reserve_free_blocks"], 0
        )
    for key, minimum in (
        ("min_host_mem_available_gib", 0.0),
        ("host_mem_warn_gib", 0.0),
        ("transfer_timeout_s", 1.0),
    ):
        if key in cfg:
            kwargs[key] = _as_float(key, cfg[key], minimum)
    return SimpleCPUOffloadOptions(**kwargs)


def check_test_switches() -> None:
    """Refuse fault-injection switches outside test mode."""
    if envs.VLLM_SIMPLE_KV_OFFLOAD_TEST_MODE:
        return
    enabled = [name for name in _TEST_SWITCHES if getattr(envs, name)]
    if enabled:
        raise ValueError(
            f"{', '.join(enabled)} require VLLM_SIMPLE_KV_OFFLOAD_TEST_MODE=1"
        )


def test_switch(name: str) -> Any:
    """Value of a fault-injection switch, or its off value outside test mode."""
    assert name in _TEST_SWITCHES, name
    value = getattr(envs, name)
    if not envs.VLLM_SIMPLE_KV_OFFLOAD_TEST_MODE:
        return type(value)()
    return value
