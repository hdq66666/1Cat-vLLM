# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transfer statistics of SimpleCPUOffloadConnector.

Workers (one per TP rank) and the scheduler-side manager each produce a
``SimpleCPUOffloadStats`` whose ``data`` is a flat ``{name: number}`` dict, so
it pickles/msgspec-encodes without this class on the receiving side. The
aggregation rule is encoded in the key name:

* ``*_max`` keeps the maximum (latencies across ranks and steps),
* ``*_min`` keeps the minimum,
* ``*_last`` keeps the most recent value (scheduler-side gauges),
* everything else is summed (counters).
"""

from dataclasses import dataclass
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats

# Worker-side counters.
LOAD_EVENTS = "load_events"
LOAD_BLOCKS = "load_blocks"
LOAD_BYTES = "load_bytes"
LOAD_DMA_MS_SUM = "load_dma_ms_sum"
LOAD_DMA_MS_MAX = "load_dma_ms_max"
LOAD_READY_MS_SUM = "load_ready_ms_sum"
LOAD_READY_MS_MAX = "load_ready_ms_max"
STORE_EVENTS = "store_events"
STORE_BLOCKS = "store_blocks"
STORE_BYTES = "store_bytes"
STORE_DMA_MS_SUM = "store_dma_ms_sum"
STORE_DMA_MS_MAX = "store_dma_ms_max"
STORE_READY_MS_SUM = "store_ready_ms_sum"
STORE_READY_MS_MAX = "store_ready_ms_max"
VERIFY_CHECKED = "verify_checked_blocks"
VERIFY_MISMATCH = "verify_mismatch_blocks"
PENDING_COMPUTE_AT_LOAD = "pending_compute_at_load_launch"

# Scheduler-side counters and gauges.
STORES_SELECTED = "stores_selected_blocks"
RESTORE_SKIPPED = "restore_skipped_blocks"
INFLIGHT_SKIPPED = "inflight_dedup_skipped_blocks"
RESERVE_STOPS = "store_reserve_stops"
LOAD_REQUESTS = "load_requests"
LOAD_DST_RECENTLY_FREED = "load_dst_recently_freed_blocks"
CPU_HITS_REFUSED = "cpu_hits_refused"
STORE_PINS_LAST = "store_pinned_gpu_blocks_last"
CPU_USED_BLOCKS_LAST = "cpu_used_blocks_last"
CPU_CACHED_BLOCKS_LAST = "cpu_cached_blocks_last"
HOST_MEM_AVAILABLE_GIB_MIN = "host_mem_available_gib_min"


def _merge(key: str, a: float, b: float) -> float:
    if key.endswith("_max"):
        return max(a, b)
    if key.endswith("_min"):
        return min(a, b)
    if key.endswith("_last"):
        return b
    return a + b


@dataclass
class SimpleCPUOffloadStats(KVConnectorStats):
    def reset(self) -> None:
        self.data = {}

    def add(self, key: str, value: float = 1) -> None:
        """Fold one observation into ``data`` with the key's merge rule."""
        if key in self.data:
            self.data[key] = _merge(key, self.data[key], value)
        else:
            self.data[key] = value

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        merged = dict(self.data)
        for key, value in (other.data or {}).items():
            merged[key] = _merge(key, merged[key], value) if key in merged else value
        return SimpleCPUOffloadStats(data=merged)

    def reduce(self) -> dict[str, int | float]:
        d = self.data
        out: dict[str, int | float] = {}
        for kind, events, blocks, nbytes, dma_sum, dma_max, rdy_sum, rdy_max in (
            (
                "load",
                LOAD_EVENTS,
                LOAD_BLOCKS,
                LOAD_BYTES,
                LOAD_DMA_MS_SUM,
                LOAD_DMA_MS_MAX,
                LOAD_READY_MS_SUM,
                LOAD_READY_MS_MAX,
            ),
            (
                "store",
                STORE_EVENTS,
                STORE_BLOCKS,
                STORE_BYTES,
                STORE_DMA_MS_SUM,
                STORE_DMA_MS_MAX,
                STORE_READY_MS_SUM,
                STORE_READY_MS_MAX,
            ),
        ):
            n = d.get(events, 0)
            if not n:
                continue
            out[f"{kind}_events"] = int(n)
            out[f"{kind}_blocks"] = int(d.get(blocks, 0))
            out[f"{kind}_GiB"] = round(d.get(nbytes, 0) / 1024**3, 3)
            dma_ms = d.get(dma_sum, 0.0)
            if dma_ms > 0:
                out[f"{kind}_GBps"] = round(d.get(nbytes, 0) / 1e6 / dma_ms, 2)
            out[f"{kind}_dma_ms_avg"] = round(dma_ms / n, 2)
            out[f"{kind}_dma_ms_max"] = round(d.get(dma_max, 0.0), 2)
            out[f"{kind}_ready_ms_avg"] = round(d.get(rdy_sum, 0.0) / n, 2)
            out[f"{kind}_ready_ms_max"] = round(d.get(rdy_max, 0.0), 2)
        for key in (
            VERIFY_CHECKED,
            VERIFY_MISMATCH,
            PENDING_COMPUTE_AT_LOAD,
            STORES_SELECTED,
            RESTORE_SKIPPED,
            INFLIGHT_SKIPPED,
            RESERVE_STOPS,
            LOAD_REQUESTS,
            LOAD_DST_RECENTLY_FREED,
            CPU_HITS_REFUSED,
            STORE_PINS_LAST,
            CPU_USED_BLOCKS_LAST,
            CPU_CACHED_BLOCKS_LAST,
        ):
            if key in d:
                out[key] = int(d[key])
        if HOST_MEM_AVAILABLE_GIB_MIN in d:
            out[HOST_MEM_AVAILABLE_GIB_MIN] = round(d[HOST_MEM_AVAILABLE_GIB_MIN], 2)
        return out

    def is_empty(self) -> bool:
        return not any(self.data.values())

    def clone_and_reset(self) -> "SimpleCPUOffloadStats":
        snapshot = SimpleCPUOffloadStats(data=dict(self.data))
        self.reset()
        return snapshot


def build_stats(data: dict[str, Any] | None) -> SimpleCPUOffloadStats:
    return SimpleCPUOffloadStats(data=dict(data or {}))
