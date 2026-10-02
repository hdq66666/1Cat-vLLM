# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""EngineCore refuses to start before the workers pin a SimpleCPUOffload tier
that would leave too little host memory (CPU only)."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

import vllm.v1.simple_kv_offload.host_mem as host_mem
from vllm.config import KVTransferConfig
from vllm.v1.engine.core import EngineCore, _check_simple_cpu_offload_host_mem

pytestmark = pytest.mark.cpu_test

GiB = 1024**3


def _vllm_config(
    kv_connector: str | None = "SimpleCPUOffloadConnector",
    extra_config: dict | None = None,
    enable_prefix_caching: bool = True,
    world_size: int = 4,
):
    kv_transfer_config = None
    if kv_connector is not None:
        kv_transfer_config = KVTransferConfig(
            kv_connector=kv_connector,
            kv_role="kv_both",
            kv_connector_extra_config=extra_config or {},
        )
    return SimpleNamespace(
        kv_transfer_config=kv_transfer_config,
        cache_config=SimpleNamespace(enable_prefix_caching=enable_prefix_caching),
        parallel_config=SimpleNamespace(
            world_size=world_size, data_parallel_rank_local=0
        ),
    )


@pytest.fixture
def mem_available(monkeypatch):
    """Set MemAvailable seen by the host memory checks (None = unknown)."""
    state: dict[str, Any] = {"bytes": None, "reads": 0}

    def read() -> int | None:
        state["reads"] += 1
        return state["bytes"]

    monkeypatch.setattr(host_mem, "read_mem_available_bytes", read)
    return state


PROD_CONFIG = {
    "cpu_bytes_to_use": 32 * GiB,
    "lazy_offload": True,
    "min_host_mem_available_gib": 15,
}


@pytest.mark.parametrize("available_gib", [46.9, 40])
def test_refuses_when_tier_leaves_too_little(mem_available, available_gib) -> None:
    # 32 GiB server-wide over 4 ranks; MemAvailable - 32 GiB < 15 GiB.
    mem_available["bytes"] = int(available_gib * GiB)
    with pytest.raises(ValueError, match="min_host_mem_available_gib=15"):
        _check_simple_cpu_offload_host_mem(_vllm_config(extra_config=PROD_CONFIG))


@pytest.mark.parametrize("available_gib", [47, 120])
def test_accepts_when_floor_remains(mem_available, available_gib) -> None:
    mem_available["bytes"] = available_gib * GiB
    _check_simple_cpu_offload_host_mem(_vllm_config(extra_config=PROD_CONFIG))
    assert mem_available["reads"] == 1


def test_counts_every_rank_of_a_per_rank_tier(mem_available) -> None:
    # 8 GiB per rank x 4 ranks = 32 GiB, whatever cpu_bytes_to_use says.
    config = {"cpu_bytes_to_use_per_rank": 8 * GiB, "min_host_mem_available_gib": 15}
    mem_available["bytes"] = 46 * GiB
    with pytest.raises(ValueError):
        _check_simple_cpu_offload_host_mem(_vllm_config(extra_config=config))
    mem_available["bytes"] = 48 * GiB
    _check_simple_cpu_offload_host_mem(_vllm_config(extra_config=config))


def test_unknown_mem_available_is_not_fatal(mem_available) -> None:
    mem_available["bytes"] = None
    _check_simple_cpu_offload_host_mem(_vllm_config(extra_config=PROD_CONFIG))
    assert mem_available["reads"] == 1


@pytest.mark.parametrize(
    "vllm_config",
    [
        _vllm_config(kv_connector=None),
        _vllm_config(kv_connector="ExampleConnector", extra_config=PROD_CONFIG),
        _vllm_config(extra_config=PROD_CONFIG, enable_prefix_caching=False),
    ],
    ids=["no-connector", "other-connector", "no-prefix-caching"],
)
def test_noop_without_a_pinned_tier(mem_available, vllm_config) -> None:
    mem_available["bytes"] = 1 * GiB
    _check_simple_cpu_offload_host_mem(vllm_config)
    assert mem_available["reads"] == 0


def test_invalid_options_fail_before_pinning(mem_available) -> None:
    with pytest.raises(ValueError, match="lazy_offload"):
        _check_simple_cpu_offload_host_mem(
            _vllm_config(extra_config={"lazy_offload": "false"})
        )
    assert mem_available["reads"] == 0


class _KVCachesInitialized(Exception):
    pass


@pytest.mark.parametrize("refuse", [True, False])
def test_engine_core_checks_before_initializing_kv_caches(
    monkeypatch, mem_available, refuse: bool
) -> None:
    """The check runs in EngineCore.__init__ right before the KV caches (and
    with them the workers' CPU tiers) are initialized."""
    initialize_kv_caches = MagicMock(side_effect=_KVCachesInitialized)
    monkeypatch.setattr(EngineCore, "_initialize_kv_caches", initialize_kv_caches)
    executor_class = MagicMock()
    mem_available["bytes"] = (30 if refuse else 60) * GiB

    engine = object.__new__(EngineCore)
    expected = ValueError if refuse else _KVCachesInitialized
    with pytest.raises(expected):
        EngineCore.__init__(
            engine,
            vllm_config=_vllm_config(extra_config=PROD_CONFIG),
            executor_class=executor_class,
            log_stats=False,
        )

    # The executor exists (workers are up) but nothing was pinned on refusal.
    executor_class.assert_called_once()
    assert initialize_kv_caches.called is not refuse
    assert mem_available["reads"] == 1
