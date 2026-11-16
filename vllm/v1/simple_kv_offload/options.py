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
    # A CPU hit is admitted only while the GPU pool can also take the
    # request's first compute step after the load plus a margin for the
    # running requests; otherwise it is deferred. After cpu_hit_admission_wait_s
    # the hit is given up as soon as a cold start of the request would be
    # admitted (so the request is never held back behind what the cold path
    # would do); after cpu_hit_admission_max_wait_s it is given up regardless.
    cpu_hit_admission_wait_s: float = 10.0
    cpu_hit_admission_max_wait_s: float = 60.0
    # Diagnostics. freed_block_quarantine_steps > 0: GPU blocks freed within
    # that many scheduler steps are neither selected for store nor (while
    # older free blocks exist) handed out by the allocator, so a load never
    # lands in a page a still-running step may touch. log_transfer_blocks:
    # log the GPU/CPU block ids of every store and load event.
    freed_block_quarantine_steps: int = 0
    log_transfer_blocks: bool = False
    # probe_block: fingerprint this GPU block (e.g. 0, the null block that no
    # request owns) at the end of every step on the compute stream, without
    # host synchronization, and log a warning whenever its content changes:
    # a change means some kernel writes through a block id it does not own.
    probe_block: int | None = None
    # probe_every: take the probe_block fingerprint only every that many
    # steps (1 = every step). A sparse probe barely perturbs the step timing.
    probe_every: int = 1
    # debug_host_sleep_ms / debug_gpu_sleep_cycles: after this step's
    # transfers were submitted, sleep on the host for that many milliseconds
    # or queue a spin of that many cycles on the compute stream. Timing
    # experiments only: they tell a host-side race (the host overruns a buffer
    # the GPU still reads) from a stream-ordering race (later GPU work races
    # a transfer).
    debug_host_sleep_ms: float = 0.0
    debug_gpu_sleep_cycles: int = 0
    # fingerprint: follow every transferred block with per-region checksums
    # taken on the transfer and compute streams and on the host, without
    # host synchronization, and warn about any difference (see fingerprint.py).
    # fingerprint_inject_gpu_load / _cpu_load: positive controls, change one
    # word of the first block of the N-th load (1-based) after / before it.
    fingerprint: bool = False
    fingerprint_inject_gpu_load: int = 0
    fingerprint_inject_cpu_load: int = 0
    # fingerprint_corrupt_plan: deliberately damage the blocks of given loads
    # ("<load number>:<mode>[=N];..." with modes zero_fa, zero_state, nan_fa,
    # swap_fa; see fingerprint.parse_corrupt_plan). Experiments only.
    fingerprint_corrupt_plan: str = ""
    # allow_request_directives: honour the "simple_cpu" entry of a request's
    # kv_transfer_params (experiments only; see manager.parse_directives):
    # corrupt the request's own load like fingerprint_corrupt_plan, gate its
    # load on the fingerprint verdict, or flip runtime controls in every
    # worker. fingerprint_gate_timeout_s: a gated load whose verdict has not
    # arrived within this many seconds after its copy completed is reported
    # to the scheduler as if it were good (with a warning).
    allow_request_directives: bool = False
    fingerprint_gate_timeout_s: float = 5.0
    # Margin of free GPU blocks kept for the running requests when admitting
    # a CPU hit. None = one block per request scheduled in the last step.
    cpu_hit_admission_margin_blocks: int | None = None
    # Prefix protection (lazy mode). A prefix that more than protect_hits
    # requests hit (GPU prefix cache or CPU tier) within protect_window_s
    # seconds is protected for protect_ttl_s seconds: its CPU copies are
    # held out of the LRU, so traffic that turns the tier over in the
    # meantime cannot evict them. 0 = off. With protect_refresh_on_hit every
    # later hit on a protected prefix restarts its time to live; without it
    # only meeting the hit rate again does. At most protect_max_fraction of
    # the CPU blocks are held this way; beyond that the protection that
    # would expire first is given up.
    protect_hits: int = 0
    protect_window_s: float = 600.0
    protect_ttl_s: float = 86400.0
    protect_max_fraction: float = 0.25
    protect_refresh_on_hit: bool = True

    def cpu_bytes_per_rank(self, world_size: int) -> int:
        if self.cpu_bytes_to_use_per_rank is not None:
            return self.cpu_bytes_to_use_per_rank
        return self.cpu_bytes_to_use // world_size

    def cpu_bytes_total(self, world_size: int) -> int:
        return self.cpu_bytes_per_rank(world_size) * world_size


def _as_bool(key: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    # bool("false") is True: accept only unambiguous spellings.
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    raise ValueError(
        f"kv_connector_extra_config[{key!r}] must be a boolean, got {value!r}"
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
    for key in (
        "lazy_offload",
        "allow_restore",
        "log_transfer_blocks",
        "fingerprint",
        "allow_request_directives",
        "protect_refresh_on_hit",
    ):
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
    for key in (
        "store_reserve_free_blocks",
        "cpu_hit_admission_margin_blocks",
        "freed_block_quarantine_steps",
        "probe_block",
        "debug_gpu_sleep_cycles",
        "fingerprint_inject_gpu_load",
        "fingerprint_inject_cpu_load",
        "protect_hits",
    ):
        if cfg.get(key) is not None:
            kwargs[key] = _as_int(key, cfg[key], 0)
    if cfg.get("probe_every") is not None:
        kwargs["probe_every"] = _as_int("probe_every", cfg["probe_every"], 1)
    for key, minimum in (
        ("min_host_mem_available_gib", 0.0),
        ("host_mem_warn_gib", 0.0),
        ("transfer_timeout_s", 1.0),
        ("cpu_hit_admission_wait_s", 0.0),
        ("cpu_hit_admission_max_wait_s", 0.0),
        ("debug_host_sleep_ms", 0.0),
        ("fingerprint_gate_timeout_s", 0.0),
        ("protect_window_s", 1.0),
        ("protect_ttl_s", 1.0),
        ("protect_max_fraction", 0.0),
    ):
        if key in cfg:
            kwargs[key] = _as_float(key, cfg[key], minimum)
    if kwargs.get("protect_max_fraction", 0.0) > 0.9:
        raise ValueError(
            "kv_connector_extra_config['protect_max_fraction'] must be <= 0.9, "
            f"got {kwargs['protect_max_fraction']}"
        )
    if cfg.get("fingerprint_corrupt_plan"):
        from vllm.v1.simple_kv_offload.fingerprint import parse_corrupt_plan

        plan = str(cfg["fingerprint_corrupt_plan"])
        parse_corrupt_plan(plan)  # validate early
        kwargs["fingerprint_corrupt_plan"] = plan
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
