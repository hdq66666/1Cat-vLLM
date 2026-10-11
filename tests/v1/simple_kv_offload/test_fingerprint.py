# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fingerprint chain of the CPU tier (options.fingerprint): every check must
stay silent on a faithful transfer and name the stage that broke otherwise."""

import pytest
import torch

from vllm.v1.simple_kv_offload.fingerprint import WATCH_STEPS, FingerprintChain
from vllm.v1.simple_kv_offload.options import parse_options

NUM_GPU, NUM_CPU, REGIONS = 12, 20, {"fa": 64, "state": 40}


def _make(device="cpu", **kwargs):
    gen = torch.Generator().manual_seed(0)
    gpu = {
        name: torch.randint(-128, 128, (NUM_GPU, size), dtype=torch.int8, generator=gen).to(device)
        for name, size in REGIONS.items()
    }
    cpu = {name: torch.zeros((NUM_CPU, size), dtype=torch.int8) for name, size in REGIONS.items()}
    if device != "cpu":
        cpu = {name: t.pin_memory() for name, t in cpu.items()}
    chain = FingerprintChain(gpu, cpu, rank=0, start_thread=False, **kwargs)
    return chain, gpu, cpu


def _store(chain, gpu, cpu, event, gpu_ids, cpu_ids, between=None, during=None, after=None):
    chain.store_launched(event, gpu_ids, cpu_ids, None)
    if between:
        between()
    pre = chain.snapshot(None)
    for name in gpu:
        cpu[name][cpu_ids] = gpu[name][gpu_ids].cpu()
    if during:
        during()
    post = chain.snapshot(None)
    if after:
        after()
    chain.transfer_issued(True, event, pre, post, None)


def _load(chain, gpu, cpu, event, cpu_ids, gpu_ids, before=None, after_copy=None):
    if before:
        before()
    chain.load_launched(event, cpu_ids, gpu_ids)
    for name in gpu:
        gpu[name][gpu_ids] = cpu[name][cpu_ids].to(gpu[name].device)
    if after_copy:
        after_copy()
    post = chain.snapshot(None)
    chain.transfer_issued(False, event, None, post, None)


def _steps(chain, n, at=None):
    for i in range(n):
        if at and i in at:
            at[i]()
        chain.tick(None)
        chain.drain()


def test_faithful_round_trip_is_silent():
    chain, gpu, cpu = _make()
    _store(chain, gpu, cpu, 0, [3, 4, 5], [7, 8, 9])
    chain.drain()
    assert chain.counters["stores"] == 1 and chain.counters["mismatch_events"] == 0
    for name in gpu:  # the GPU blocks are reused by somebody else
        gpu[name][[3, 4, 5]] = 1
    _load(chain, gpu, cpu, 0, [7, 8, 9], [1, 2, 6])
    chain.drain()
    chain.load_seen_complete(0)
    _steps(chain, WATCH_STEPS[-1] + 2)
    assert chain.counters == {
        "stores": 1,
        "loads": 1,
        "watches": len(WATCH_STEPS),
        "mismatch_events": 0,
        "skipped": 0,
        "stale": 0,
    }


def test_host_buffers_cross_inference_mode():
    # The engine thread runs in inference mode, the copy thread does not, and
    # both reuse the same host buffers.
    chain, gpu, cpu = _make()
    with torch.inference_mode():
        first = chain.snapshot(None)
    chain._release(first)
    second = chain.snapshot(None)
    assert second.host is first.host
    with torch.inference_mode():
        _store(chain, gpu, cpu, 0, [3], [7])
    chain.drain()
    _load(chain, gpu, cpu, 0, [7], [1])
    with torch.inference_mode():
        chain.drain()
    assert chain.counters["mismatch_events"] == 0 and chain.counters["loads"] == 1


def _flip(tensor, block):
    def fn():
        tensor[block, 8] ^= 1

    return fn


@pytest.mark.parametrize(
    "stage,kind",
    [
        ("between", "store launch->copy"),
        ("during", "store during-copy"),
        ("after", "store gpu->cpu"),
    ],
)
def test_store_stages(stage, kind):
    chain, gpu, cpu = _make()
    target = cpu["state"] if stage == "after" else gpu["state"]
    block = 8 if stage == "after" else 4
    _store(chain, gpu, cpu, 0, [3, 4, 5], [7, 8, 9], **{stage: _flip(target, block)})
    chain.drain()
    assert kind in chain.reports
    if stage == "between":
        # The copy itself was faithful to what it read.
        assert "store gpu->cpu" not in chain.reports


def test_cpu_block_changed_at_rest():
    chain, gpu, cpu = _make()
    _store(chain, gpu, cpu, 0, [3, 4], [7, 8])
    chain.drain()
    _load(chain, gpu, cpu, 0, [7, 8], [1, 2], before=_flip(cpu["fa"], 7))
    chain.drain()
    assert "load cpu-at-rest" in chain.reports
    assert "load end-to-end(gpu source vs gpu restored)" in chain.reports
    assert "load cpu->gpu" not in chain.reports


def test_load_delivers_other_bytes():
    chain, gpu, cpu = _make()
    _store(chain, gpu, cpu, 0, [3, 4], [7, 8])
    chain.drain()
    _load(chain, gpu, cpu, 0, [7, 8], [1, 2], after_copy=_flip(gpu["fa"], 2))
    chain.drain()
    assert chain.reports == [
        "load end-to-end(gpu source vs gpu restored)",
        "load cpu->gpu",
    ]


def test_stale_previous_content_is_named():
    chain, gpu, cpu = _make()
    _store(chain, gpu, cpu, 0, [3], [7])  # CPU block 7 holds GPU block 3
    chain.drain()
    old = {name: cpu[name][7].clone() for name in cpu}
    _store(chain, gpu, cpu, 1, [5], [7])  # ... and is then reused for block 5
    chain.drain()

    def deliver_old():
        for name in gpu:
            gpu[name][1] = old[name]

    _load(chain, gpu, cpu, 0, [7], [1], after_copy=deliver_old)
    chain.drain()
    assert "load cpu->gpu" in chain.reports
    assert chain.counters["stale"] == 1


def test_block_written_after_the_load():
    chain, gpu, cpu = _make()
    _store(chain, gpu, cpu, 0, [3, 4], [7, 8])
    chain.drain()
    _load(chain, gpu, cpu, 0, [7, 8], [1, 2])
    chain.drain()
    chain.load_seen_complete(0)

    def zero():
        for name in gpu:
            gpu[name][2] = 0

    _steps(chain, WATCH_STEPS[-1] + 2, at={3: zero})
    assert chain.reports == ["watch +4 steps", "watch +6 steps"]
    assert chain.counters["watches"] == len(WATCH_STEPS)


def test_later_store_into_the_same_cpu_block_is_not_compared():
    chain, gpu, cpu = _make()
    _store(chain, gpu, cpu, 0, [3], [7])
    # The checker is behind: a second store into CPU block 7 is launched and
    # copied before the first one is checked.
    _store(chain, gpu, cpu, 1, [5], [7])
    chain.drain()
    assert chain.counters["skipped"] == 1 and chain.counters["mismatch_events"] == 0


def test_injections_are_reported():
    chain, gpu, cpu = _make(inject_gpu_load=1)
    _store(chain, gpu, cpu, 0, [3, 4], [7, 8])
    chain.drain()
    _load(chain, gpu, cpu, 0, [7, 8], [1, 2])
    chain.drain()
    chain.load_seen_complete(0)
    _steps(chain, WATCH_STEPS[-1] + 2)
    assert chain.reports == ["watch +2 steps", "watch +4 steps", "watch +6 steps"]

    chain, gpu, cpu = _make(inject_cpu_load=1)
    _store(chain, gpu, cpu, 0, [3, 4], [7, 8])
    chain.drain()
    _load(chain, gpu, cpu, 0, [7, 8], [1, 2])
    chain.drain()
    assert "load cpu-at-rest" in chain.reports


@pytest.mark.parametrize(
    "mode,check",
    [
        ("zero_fa=2", lambda gpu, before: bool((gpu["fa"][[1, 2]] == 0).all())),
        ("zero_state=1", lambda gpu, before: bool((gpu["state"][6] == 0).all())),
        ("nan_fa=1", lambda gpu, before: bool((gpu["fa"][1] == -1).all())),
        (
            "swap_fa=1",
            lambda gpu, before: torch.equal(gpu["fa"][1], before["fa"][2])
            and not torch.equal(gpu["fa"][1], before["fa"][1]),
        ),
    ],
)
def test_corrupt_plan_damages_the_chosen_load(mode, check):
    chain, gpu, cpu = _make(corrupt_plan=f"2:{mode}")
    _store(chain, gpu, cpu, 0, [3, 4, 5], [7, 8, 9])
    chain.drain()
    for name in gpu:
        gpu[name][[3, 4, 5]] = 1
    # Load 1 is left alone, load 2 is damaged when it is seen complete.
    _load(chain, gpu, cpu, 0, [7, 8, 9], [9, 10, 11])
    chain.drain()
    chain.load_seen_complete(0)
    _load(chain, gpu, cpu, 1, [7, 8, 9], [1, 2, 6])
    chain.drain()
    before = {name: t.clone() for name, t in gpu.items()}
    chain.load_seen_complete(1)
    assert check(gpu, before)
    assert torch.equal(gpu["fa"][[9, 10, 11]], before["fa"][[9, 10, 11]])
    _steps(chain, WATCH_STEPS[-1] + 2)
    assert "watch +0 steps" in chain.reports


def test_corrupt_plan_is_validated():
    with pytest.raises(ValueError, match="unknown corrupt mode"):
        parse_options({"fingerprint_corrupt_plan": "3:melt"})
    plan = "3:zero_fa=8;19:nan_fa"
    assert parse_options({"fingerprint_corrupt_plan": plan}).fingerprint_corrupt_plan == plan


def test_options():
    assert parse_options({}).fingerprint is False
    options = parse_options(
        {"fingerprint": True, "fingerprint_inject_gpu_load": 2, "fingerprint_inject_cpu_load": 3}
    )
    assert options.fingerprint and options.fingerprint_inject_gpu_load == 2
    assert options.fingerprint_inject_cpu_load == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_streams_and_checker_thread():
    gen = torch.Generator().manual_seed(1)
    gpu = {
        name: torch.randint(-128, 128, (NUM_GPU, size), dtype=torch.int8, generator=gen).cuda()
        for name, size in REGIONS.items()
    }
    cpu = {
        name: torch.zeros((NUM_CPU, size), dtype=torch.int8).pin_memory()
        for name, size in REGIONS.items()
    }
    chain = FingerprintChain(gpu, cpu, rank=0)
    compute = torch.cuda.current_stream()
    store_stream, load_stream = torch.cuda.Stream(), torch.cuda.Stream()

    def store(event, gpu_ids, cpu_ids):
        store_stream.wait_stream(compute)
        chain.store_launched(event, gpu_ids, cpu_ids, compute)
        pre = chain.snapshot(store_stream)
        with torch.cuda.stream(store_stream):
            for name in gpu:
                for g, c in zip(gpu_ids, cpu_ids):
                    cpu[name][c].copy_(gpu[name][g], non_blocking=True)
        post = chain.snapshot(store_stream)
        done = torch.cuda.Event()
        done.record(store_stream)
        chain.transfer_issued(True, event, pre, post, done)

    def load(event, cpu_ids, gpu_ids):
        load_stream.wait_stream(compute)
        chain.load_launched(event, cpu_ids, gpu_ids)
        with torch.cuda.stream(load_stream):
            for name in gpu:
                for g, c in zip(gpu_ids, cpu_ids):
                    gpu[name][g].copy_(cpu[name][c], non_blocking=True)
        post = chain.snapshot(load_stream)
        done = torch.cuda.Event()
        done.record(load_stream)
        chain.transfer_issued(False, event, None, post, done)
        done.synchronize()
        chain.load_seen_complete(event)

    store(0, [3, 4, 5], [7, 8, 9])
    torch.cuda.synchronize()
    for name in gpu:
        gpu[name][[3, 4, 5]] = 1
    load(0, [7, 8, 9], [1, 2, 6])
    for step in range(WATCH_STEPS[-1] + 2):
        if step == 3:
            gpu["fa"][2, 8] ^= 1  # a write into a loaded block
        chain.tick(compute)
    chain.close()
    chain._thread.join(30)
    assert not chain._thread.is_alive()
    assert chain.counters["stores"] == 1 and chain.counters["loads"] == 1
    assert chain.counters["watches"] == len(WATCH_STEPS)
    assert chain.reports == ["watch +4 steps", "watch +6 steps"]
