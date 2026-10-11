# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata for SimpleCPUOffloadConnector."""

from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)

INVALID_JOB_ID = -1


@dataclass
class SimpleCPUOffloadMetadata(KVConnectorMetadata):
    """
    Metadata passed from scheduler to worker for CPU offload operations.

    The worker receives flat block lists keyed by a monotonic event_idx.
    Job->req_id translation is handled by the scheduler-side manager
    (via inverse maps), so the worker never knows about request identities.
    """

    # Load event per step. INVALID_JOB_ID means no blocks to load this step.
    load_event: int = INVALID_JOB_ID
    load_gpu_blocks: list[int] = field(default_factory=list)
    load_cpu_blocks: list[int] = field(default_factory=list)
    # Reverse map: load_event->req_ids, for tracking requests with finished load events
    load_event_to_reqs: dict[int, list[str]] = field(default_factory=dict)
    # The KV cache group of each loaded block's owner (fingerprint block
    # kinds; BLOCK_UNKNOWN without a hash), parallel to load_gpu_blocks;
    # empty when the fingerprint chain is off.
    load_block_kinds: list[int] = field(default_factory=list)
    # Per load event, kept while the event is pending like
    # load_event_to_reqs: the [start, end) span of each request's blocks in
    # the event's block list (same order as load_event_to_reqs), the
    # corrupt directives (start, end, mode) and whether any request asked
    # for the load to be gated on the fingerprint verdict.
    load_spans: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    load_corrupt: dict[int, list[tuple[int, int, str]]] = field(default_factory=dict)
    load_gate: dict[int, bool] = field(default_factory=dict)

    # Store event per step. INVALID_JOB_ID means no blocks to store this step.
    store_event: int = INVALID_JOB_ID
    store_gpu_blocks: list[int] = field(default_factory=list)
    store_cpu_blocks: list[int] = field(default_factory=list)
    store_block_kinds: list[int] = field(default_factory=list)

    # Runtime controls requested through request directives this step
    # (options.allow_request_directives); applied by every worker.
    controls: dict[str, str] = field(default_factory=dict)

    # Whether any requests were preempted this step and need flush pending transfers.
    need_flush: bool = False

    # Scheduler-side CPU block count; every worker checks it against its own
    # (0 = not set). A worker with fewer CPU blocks would DMA out of bounds.
    num_cpu_blocks: int = 0


@dataclass
class SimpleCPUOffloadWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker -> Scheduler metadata for completed store events.

    Each worker reports {event_idx: 1} for newly completed stores.
    ``aggregate()`` sums counts across workers within a step.
    The scheduler-side manager accumulates across steps and processes
    a store completion only when count reaches ``world_size``.
    """

    completed_store_events: dict[int, int]
    # CPU blocks whose content a gated load found bad on this worker (union
    # across workers): the scheduler keeps them out of circulation.
    bad_cpu_blocks: list[int] = field(default_factory=list)

    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "KVConnectorWorkerMetadata":
        assert isinstance(other, SimpleCPUOffloadWorkerMetadata)
        merged = dict(self.completed_store_events)
        for k, v in other.completed_store_events.items():
            merged[k] = merged.get(k, 0) + v
        return SimpleCPUOffloadWorkerMetadata(
            completed_store_events=merged,
            bad_cpu_blocks=sorted(set(self.bad_cpu_blocks) | set(other.bad_cpu_blocks)),
        )
