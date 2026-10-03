# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SimpleCPUOffloadConnector options, transfer statistics and host memory
checks (CPU only)."""

from __future__ import annotations

import io

import msgspec
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector,
)
from vllm.v1.simple_kv_offload import host_mem
from vllm.v1.simple_kv_offload import options as kv_options
from vllm.v1.simple_kv_offload import stats as kv_stats
from vllm.v1.simple_kv_offload.options import (
    DEFAULT_CPU_CAPACITY_BYTES,
    SimpleCPUOffloadOptions,
    check_test_switches,
    parse_options,
)
from vllm.v1.simple_kv_offload.stats import SimpleCPUOffloadStats, build_stats

GiB = 1024**3


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
def test_parse_options_defaults_and_production_config() -> None:
    assert parse_options(None) == SimpleCPUOffloadOptions()
    assert parse_options({}).cpu_bytes_to_use == DEFAULT_CPU_CAPACITY_BYTES

    options = parse_options(
        {
            "cpu_bytes_to_use": 32 * GiB,
            "lazy_offload": True,
            "copy_backend": "auto",
            "max_store_blocks_per_step": 16,
            "store_reserve_free_blocks": 25,
            "allow_restore": False,
            "min_host_mem_available_gib": 15,
            "host_mem_warn_gib": 12,
            "transfer_timeout_s": 60,
            "unknown_key": "ignored",
        }
    )
    assert options == SimpleCPUOffloadOptions(
        cpu_bytes_to_use=32 * GiB,
        lazy_offload=True,
        store_reserve_free_blocks=25,
    )
    assert options.cpu_bytes_per_rank(4) == 8 * GiB
    assert options.cpu_bytes_total(4) == 32 * GiB

    per_rank = parse_options(
        {"cpu_bytes_to_use": 32 * GiB, "cpu_bytes_to_use_per_rank": 6 * GiB}
    )
    assert per_rank.cpu_bytes_per_rank(4) == 6 * GiB
    assert per_rank.cpu_bytes_total(4) == 24 * GiB
    # --kv-offloading-size hands the size over as a float.
    assert parse_options({"cpu_bytes_to_use": 2.0 * GiB}).cpu_bytes_to_use == 2 * GiB
    no_reserve = parse_options({"store_reserve_free_blocks": None})
    assert no_reserve.store_reserve_free_blocks is None


@pytest.mark.parametrize(
    "extra_config",
    [
        {"lazy_offload": "yes"},
        {"lazy_offload": 2},
        {"allow_restore": None},
        {"cpu_bytes_to_use": True},
        {"cpu_bytes_to_use": 0},
        {"cpu_bytes_to_use": [1]},
        {"copy_backend": "cuda"},
        {"max_store_blocks_per_step": 0},
        {"store_reserve_free_blocks": -1},
        {"min_host_mem_available_gib": -1},
        {"transfer_timeout_s": 0.5},
        {"host_mem_warn_gib": False},
    ],
)
def test_parse_options_rejects(extra_config: dict) -> None:
    with pytest.raises(ValueError):
        parse_options(extra_config)


@pytest.mark.parametrize(
    "value,expected",
    [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        ("True", True),
        (" false ", False),
    ],
)
def test_parse_options_boolean_spellings(value, expected) -> None:
    # bool("false") would be True; only unambiguous spellings are accepted.
    assert parse_options({"lazy_offload": value}).lazy_offload is expected
    assert parse_options({"allow_restore": value}).allow_restore is expected


def test_test_switches_need_test_mode(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_SIMPLE_KV_OFFLOAD_TEST_MODE", raising=False)
    check_test_switches()
    monkeypatch.setenv("VLLM_SIMPLE_KV_OFFLOAD_TEST_UNORDERED_LOAD", "1")
    monkeypatch.setenv("VLLM_SIMPLE_KV_OFFLOAD_DEBUG_ZERO_DELAY_CYCLES", "1000")
    with pytest.raises(ValueError, match="TEST_MODE"):
        check_test_switches()
    # Outside test mode the switches read as their off value.
    assert kv_options.test_switch("VLLM_SIMPLE_KV_OFFLOAD_TEST_UNORDERED_LOAD") is False
    assert kv_options.test_switch("VLLM_SIMPLE_KV_OFFLOAD_DEBUG_ZERO_DELAY_CYCLES") == 0

    monkeypatch.setenv("VLLM_SIMPLE_KV_OFFLOAD_TEST_MODE", "1")
    check_test_switches()
    assert kv_options.test_switch("VLLM_SIMPLE_KV_OFFLOAD_TEST_UNORDERED_LOAD") is True
    delay = kv_options.test_switch("VLLM_SIMPLE_KV_OFFLOAD_DEBUG_ZERO_DELAY_CYCLES")
    assert delay == 1000
    with pytest.raises(AssertionError):
        kv_options.test_switch("VLLM_SIMPLE_KV_OFFLOAD_VERIFY")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def _worker_stats(rank: int) -> SimpleCPUOffloadStats:
    stats = SimpleCPUOffloadStats()
    stats.add(kv_stats.LOAD_EVENTS)
    stats.add(kv_stats.LOAD_BLOCKS, 10)
    stats.add(kv_stats.LOAD_BYTES, 10 * 1024**2)
    stats.add(kv_stats.LOAD_DMA_MS_SUM, 2.0)
    stats.add(kv_stats.LOAD_DMA_MS_MAX, 1.0 + rank)
    stats.add(kv_stats.LOAD_READY_MS_SUM, 3.0)
    stats.add(kv_stats.LOAD_READY_MS_MAX, 3.0)
    stats.add(kv_stats.HOST_MEM_AVAILABLE_GIB_MIN, 20.0 - rank)
    stats.add(kv_stats.CPU_USED_BLOCKS_LAST, rank)
    return stats


def test_stats_merge_rules() -> None:
    stats = SimpleCPUOffloadStats()
    assert stats.is_empty()
    for value in (3, 1, 2):
        stats.add(kv_stats.STORE_DMA_MS_MAX, value)
        stats.add(kv_stats.HOST_MEM_AVAILABLE_GIB_MIN, value)
        stats.add(kv_stats.STORE_PINS_LAST, value)
        stats.add(kv_stats.STORE_BLOCKS, value)
    assert stats.data == {
        kv_stats.STORE_DMA_MS_MAX: 3,
        kv_stats.HOST_MEM_AVAILABLE_GIB_MIN: 1,
        kv_stats.STORE_PINS_LAST: 2,
        kv_stats.STORE_BLOCKS: 6,
    }
    snapshot = stats.clone_and_reset()
    assert stats.data == {} and stats.is_empty()
    assert snapshot.data[kv_stats.STORE_BLOCKS] == 6


def test_stats_aggregate_over_ranks_and_reduce() -> None:
    merged = _worker_stats(0)
    for rank in range(1, 4):
        merged = merged.aggregate(_worker_stats(rank))
    assert isinstance(merged, SimpleCPUOffloadStats)
    data = merged.data
    assert data[kv_stats.LOAD_EVENTS] == 4
    assert data[kv_stats.LOAD_BYTES] == 40 * 1024**2
    assert data[kv_stats.LOAD_DMA_MS_SUM] == 8.0
    assert data[kv_stats.LOAD_DMA_MS_MAX] == 4.0
    assert data[kv_stats.HOST_MEM_AVAILABLE_GIB_MIN] == 17.0
    assert data[kv_stats.CPU_USED_BLOCKS_LAST] == 3

    reduced = merged.reduce()
    assert reduced["load_events"] == 4
    assert reduced["load_blocks"] == 40
    assert reduced["load_GiB"] == round(40 / 1024, 3)
    assert reduced["load_GBps"] == round(40 * 1024**2 / 1e6 / 8.0, 2)
    assert reduced["load_dma_ms_avg"] == 2.0
    assert reduced["load_dma_ms_max"] == 4.0
    assert reduced["load_ready_ms_avg"] == 3.0
    assert "store_events" not in reduced
    assert reduced[kv_stats.HOST_MEM_AVAILABLE_GIB_MIN] == 17.0
    assert reduced[kv_stats.CPU_USED_BLOCKS_LAST] == 3


def test_build_kv_connector_stats_round_trip() -> None:
    stats = _worker_stats(1)
    for payload in (
        dict(stats.data),
        msgspec.msgpack.decode(msgspec.msgpack.encode(stats.data)),
    ):
        rebuilt = SimpleCPUOffloadConnector.build_kv_connector_stats(data=payload)
        assert isinstance(rebuilt, SimpleCPUOffloadStats)
        assert rebuilt.data == stats.data
        assert rebuilt.reduce() == stats.reduce()
    empty = SimpleCPUOffloadConnector.build_kv_connector_stats()
    assert isinstance(empty, SimpleCPUOffloadStats) and empty.is_empty()
    assert build_stats(None).data == {}


# ---------------------------------------------------------------------------
# Host memory
# ---------------------------------------------------------------------------
def test_read_mem_available_bytes(monkeypatch) -> None:
    meminfo = "MemTotal:       263842396 kB\nMemAvailable:   69222068 kB\n"
    monkeypatch.setattr(
        host_mem, "open", lambda path: io.StringIO(meminfo), raising=False
    )
    assert host_mem.read_mem_available_bytes() == 69222068 * 1024

    monkeypatch.setattr(
        host_mem, "open", lambda path: io.StringIO("MemTotal: 1 kB\n"), raising=False
    )
    assert host_mem.read_mem_available_bytes() is None

    def missing(path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(host_mem, "open", missing, raising=False)
    assert host_mem.read_mem_available_bytes() is None


def test_host_mem_checks(monkeypatch) -> None:
    available: list[int | None] = [50 * GiB]
    monkeypatch.setattr(host_mem, "read_mem_available_bytes", lambda: available[0])

    # 50 - 32 = 18 GiB would remain: above the 15 GiB floor.
    host_mem.check_host_mem_before_pinning(32 * GiB, 15.0, "test")
    available[0] = 46 * GiB
    with pytest.raises(ValueError, match="min_host_mem_available_gib"):
        host_mem.check_host_mem_before_pinning(32 * GiB, 15.0, "test")
    host_mem.check_host_mem_after_pinning(15.0, "test")
    available[0] = 14 * GiB
    with pytest.raises(ValueError, match="after the CPU KV"):
        host_mem.check_host_mem_after_pinning(15.0, "test")

    # Unknown MemAvailable never blocks startup.
    available[0] = None
    host_mem.check_host_mem_before_pinning(32 * GiB, 15.0, "test")
    host_mem.check_host_mem_after_pinning(15.0, "test")
