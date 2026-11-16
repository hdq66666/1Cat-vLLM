# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The engine keeps stepping while a KV connector has background work.

SimpleCPUOffloadConnector keeps offloading freed blocks after the last request
finished. Scheduler.has_requests() exposes that through the connector's
has_pending_push_work(), so EngineCore runs empty steps (connector metadata to
the workers, transfer completions back) and goes idle once the work is done.
"""

import queue
from collections import deque
from concurrent.futures import Future

import pytest

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    KVConnectorOutput,
    ModelRunnerOutput,
)
from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler
from vllm.v1.simple_kv_offload.metadata import (
    SimpleCPUOffloadMetadata,
    SimpleCPUOffloadWorkerMetadata,
)

from .utils import create_requests, create_scheduler, mock_kv

pytestmark = pytest.mark.cpu_test

# Generous bound on engine iterations; every loop below ends far earlier.
MAX_STEPS = 64


class _PushWork:
    """Background work of a connector that the workers finish one empty step
    at a time (e.g. CPU offload stores still in flight)."""

    def __init__(self, num_steps: int) -> None:
        self.remaining = num_steps

    def pending(self) -> bool:
        return self.remaining > 0

    def step_done(self) -> None:
        self.remaining = max(0, self.remaining - 1)


def _done(value):
    future: Future = Future()
    future.set_result(value)
    return future


def _sampled_output(
    scheduler: Scheduler,
    scheduler_output: SchedulerOutput,
    kv_connector_output: KVConnectorOutput | None = None,
) -> ModelRunnerOutput:
    req_ids = list(scheduler_output.num_scheduled_tokens)
    sampled = []
    for req_id in req_ids:
        request = scheduler.requests[req_id]
        partial = request.num_computed_tokens < request.num_tokens
        sampled.append([] if partial else [0])
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=sampled,
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=None,
        kv_connector_output=kv_connector_output,
    )


class _FakeExecutor:
    """MRV2-shaped executor: execute_model returns the output itself only for
    steps without tokens; sampled steps return it from sample_tokens."""

    def __init__(self, scheduler: Scheduler, push_work: _PushWork) -> None:
        self.scheduler = scheduler
        self.push_work = push_work
        self.scheduler_outputs: list[SchedulerOutput] = []
        self._pending_output: ModelRunnerOutput | None = None

    @property
    def num_empty_steps(self) -> int:
        return sum(
            output.total_num_scheduled_tokens == 0 for output in self.scheduler_outputs
        )

    def execute_model(self, scheduler_output: SchedulerOutput, non_block=False):
        self.scheduler_outputs.append(scheduler_output)
        if scheduler_output.total_num_scheduled_tokens == 0:
            # No forward: the step's connector results come back at once.
            self.push_work.step_done()
            output = EMPTY_MODEL_RUNNER_OUTPUT
        else:
            self._pending_output = _sampled_output(self.scheduler, scheduler_output)
            output = None
        return _done(output) if non_block else output

    def sample_tokens(self, grammar_output, non_block=False):
        output, self._pending_output = self._pending_output, None
        assert output is not None
        return _done(output) if non_block else output


def _make_engine(
    scheduler: Scheduler, executor, batch_queue_size: int
) -> EngineCoreProc:
    """An EngineCoreProc around the scheduler without workers or sockets."""
    engine = object.__new__(EngineCoreProc)
    engine.vllm_config = scheduler.vllm_config
    engine.scheduler = scheduler
    engine.model_executor = executor
    engine.aborts_queue = queue.Queue()
    engine.output_queue = queue.Queue()
    engine.engines_running = False
    engine.async_scheduling = batch_queue_size > 1
    engine.use_spec_decode = False
    engine.is_ec_consumer = True
    engine.is_pooling_model = False
    engine.batch_queue_size = batch_queue_size
    engine.batch_queue = None
    if batch_queue_size > 1:
        engine.batch_queue = deque(maxlen=batch_queue_size)
    engine._sm70_async_cpu_trace_step = 0
    engine.step_fn = (
        engine.step if engine.batch_queue is None else engine.step_with_batch_queue
    )
    return engine


def _run_until_idle(engine: EngineCoreProc) -> int:
    num_iterations = 0
    while engine.has_work():
        assert num_iterations < MAX_STEPS, "the engine never went idle"
        engine._process_engine_step()
        num_iterations += 1
    return num_iterations


def test_has_requests_follows_pending_push_work(monkeypatch) -> None:
    scheduler = create_scheduler(use_kv_connector=mock_kv(0, False))
    push_work = _PushWork(0)
    monkeypatch.setattr(scheduler.connector, "has_pending_push_work", push_work.pending)

    assert not scheduler.has_requests()
    push_work.remaining = 1
    assert not scheduler.has_unfinished_requests()
    assert not scheduler.has_finished_requests()
    assert scheduler.has_requests()
    push_work.remaining = 0
    assert not scheduler.has_requests()

    (request,) = create_requests(num_requests=1)
    scheduler.add_request(request)
    assert scheduler.has_requests()


def test_has_requests_without_connector() -> None:
    scheduler = create_scheduler()
    assert scheduler.connector is None
    assert not scheduler.has_requests()
    (request,) = create_requests(num_requests=1)
    scheduler.add_request(request)
    assert scheduler.has_requests()


@pytest.mark.parametrize("batch_queue_size", [1, 2])
def test_engine_steps_push_work_only_while_pending(
    monkeypatch, batch_queue_size: int
) -> None:
    scheduler = create_scheduler(
        use_kv_connector=mock_kv(0, False), async_scheduling=batch_queue_size > 1
    )
    push_work = _PushWork(3)
    monkeypatch.setattr(scheduler.connector, "has_pending_push_work", push_work.pending)
    executor = _FakeExecutor(scheduler, push_work)
    engine = _make_engine(scheduler, executor, batch_queue_size)

    _run_until_idle(engine)

    # Exactly one empty step per unit of work, each with connector metadata.
    assert executor.num_empty_steps == 3
    assert len(executor.scheduler_outputs) == 3
    assert all(o.kv_connector_metadata is not None for o in executor.scheduler_outputs)
    assert not push_work.pending()
    assert not engine.batch_queue
    # Idle: further steps do nothing.
    engine.step_fn()
    assert len(executor.scheduler_outputs) == 3


@pytest.mark.parametrize("batch_queue_size", [1, 2])
def test_engine_drains_push_work_after_last_request(
    monkeypatch, batch_queue_size: int
) -> None:
    scheduler = create_scheduler(
        use_kv_connector=mock_kv(0, False), async_scheduling=batch_queue_size > 1
    )
    push_work = _PushWork(2)
    monkeypatch.setattr(scheduler.connector, "has_pending_push_work", push_work.pending)
    executor = _FakeExecutor(scheduler, push_work)
    engine = _make_engine(scheduler, executor, batch_queue_size)
    for request in create_requests(num_requests=2, num_tokens=20, max_tokens=1):
        scheduler.add_request(request)

    _run_until_idle(engine)

    assert not scheduler.requests
    num_tokens = [o.total_num_scheduled_tokens for o in executor.scheduler_outputs]
    # The requests ran first; the work then took two empty steps.
    assert num_tokens[0] == 40
    assert num_tokens[-2:] == [0, 0]
    assert not push_work.pending()
    assert not engine.has_work()


@pytest.mark.skipif(
    not hasattr(SimpleCPUOffloadScheduler, "has_pending_work"),
    reason="needs the lazy CPU tier's has_pending_work()",
)
@pytest.mark.parametrize("batch_queue_size", [1, 2])
def test_simple_cpu_offload_drains_freed_blocks_when_idle(
    batch_queue_size: int,
) -> None:
    from .test_kv_load_skip_zero import PROMPT_LEN, _make_request, _make_scheduler

    scheduler = _make_scheduler()
    manager = scheduler.connector.scheduler_manager

    class _OffloadExecutor(_FakeExecutor):
        """Every store of a step completes within that step."""

        def execute_model(self, scheduler_output, non_block=False):
            meta = scheduler_output.kv_connector_metadata
            assert isinstance(meta, SimpleCPUOffloadMetadata)
            kv_output = KVConnectorOutput(
                kv_connector_worker_meta=SimpleCPUOffloadWorkerMetadata(
                    completed_store_events=(
                        {meta.store_event: 1} if meta.store_event >= 0 else {}
                    )
                )
            )
            self.scheduler_outputs.append(scheduler_output)
            output = _sampled_output(self.scheduler, scheduler_output, kv_output)
            if scheduler_output.total_num_scheduled_tokens == 0:
                return _done(output) if non_block else output
            self._pending_output = output
            return _done(None) if non_block else None

    executor = _OffloadExecutor(scheduler, _PushWork(0))
    engine = _make_engine(scheduler, executor, batch_queue_size)
    scheduler.add_request(_make_request("A", list(range(PROMPT_LEN))))

    _run_until_idle(engine)

    # Every hashed block A left behind reached the CPU tier before idling.
    gpu_pool = scheduler.kv_cache_manager.block_pool
    hashed = [blk for blk in gpu_pool.blocks if blk.block_hash is not None]
    assert hashed
    cpu_cached = manager.cpu_block_pool.cached_block_hash_to_block
    assert all(cpu_cached.get_one_block(blk.block_hash) for blk in hashed)
    assert executor.num_empty_steps >= 1
    assert not manager.has_pending_work()
    assert not scheduler.has_requests()
