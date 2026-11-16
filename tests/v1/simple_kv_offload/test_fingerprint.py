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
        name: torch.randint(
            -128, 128, (NUM_GPU, size), dtype=torch.int8, generator=gen
        ).to(device)
        for name, size in REGIONS.items()
    }
    cpu = {
        name: torch.zeros((NUM_CPU, size), dtype=torch.int8)
        for name, size in REGIONS.items()
    }
    if device != "cpu":
        cpu = {name: t.pin_memory() for name, t in cpu.items()}
    chain = FingerprintChain(gpu, cpu, rank=0, start_thread=False, **kwargs)
    return chain, gpu, cpu


def _store(
    chain, gpu, cpu, event, gpu_ids, cpu_ids, between=None, during=None, after=None
):
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
        "nonfinite_events": 0,
        "skipped": 0,
        "stale": 0,
    }
    assert chain.pop_load_verdict(0) == []
    assert chain.pop_load_verdict(0) is None


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
    assert (
        parse_options({"fingerprint_corrupt_plan": plan}).fingerprint_corrupt_plan
        == plan
    )


def test_options():
    assert parse_options({}).fingerprint is False
    options = parse_options(
        {
            "fingerprint": True,
            "fingerprint_inject_gpu_load": 2,
            "fingerprint_inject_cpu_load": 3,
        }
    )
    assert options.fingerprint and options.fingerprint_inject_gpu_load == 2
    assert options.fingerprint_inject_cpu_load == 3


@pytest.mark.skipif(not torch.accelerator.is_available(), reason="needs CUDA")
def test_cuda_streams_and_checker_thread():
    gen = torch.Generator().manual_seed(1)
    gpu = {
        name: torch.randint(
            -128, 128, (NUM_GPU, size), dtype=torch.int8, generator=gen
        ).cuda()
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
    torch.accelerator.synchronize()
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


# --- Typed views: non-finite counts, kind-aware corrupt modes, verdicts ---

from vllm.v1.simple_kv_offload.fingerprint import (  # noqa: E402
    BLOCK_ATTN,
    BLOCK_STATE,
    parse_corrupt_mode,
)


def _make_typed(**kwargs):
    """Regions with finite random content and typed views: "fa" holds fp16
    K then V halves, "state" one float32 state tensor."""
    chain, gpu, cpu = _make(**kwargs)
    gen = torch.Generator().manual_seed(1)
    gpu["fa"].view(torch.float16).uniform_(-1, 1, generator=gen)
    gpu["state"].view(torch.float32).uniform_(-1, 1, generator=gen)
    cpu["fa"].view(torch.float16).uniform_(-1, 1, generator=gen)
    cpu["state"].view(torch.float32).uniform_(-1, 1, generator=gen)

    def views(regions):
        fa = regions["fa"].view(torch.float16)
        half = fa.shape[1] // 2
        return {
            "fa": [("k", fa[:, :half]), ("v", fa[:, half:])],
            "state": [("state", regions["state"].view(torch.float32))],
        }

    gpu_views, cpu_views = views(gpu), views(cpu)
    chain = FingerprintChain(
        gpu,
        cpu,
        rank=0,
        start_thread=False,
        gpu_views=gpu_views,
        cpu_views=cpu_views,
        region_groups={"fa": BLOCK_ATTN, "state": BLOCK_STATE},
        state_groups={BLOCK_STATE},
        **kwargs,
    )
    return chain, gpu, cpu, gpu_views, cpu_views


def _store_typed(chain, gpu, cpu, event, gpu_ids, cpu_ids, kinds):
    chain.store_launched(event, gpu_ids, cpu_ids, None, kinds=kinds)
    pre = chain.snapshot(None)
    nf = chain.nonfinite(None, (True, event))
    for name in gpu:
        cpu[name][cpu_ids] = gpu[name][gpu_ids]
    post = chain.snapshot(None)
    chain.transfer_issued(True, event, pre, post, None, nf)


def _load_typed(chain, gpu, cpu, event, cpu_ids, gpu_ids, kinds, corrupt=None):
    chain.load_launched(event, cpu_ids, gpu_ids, kinds=kinds, corrupt=corrupt)
    for name in gpu:
        gpu[name][gpu_ids] = cpu[name][cpu_ids]
    post = chain.snapshot(None)
    nf = chain.nonfinite(None, (False, event))
    chain.transfer_issued(False, event, None, post, None, nf)


def test_nonfinite_counts_name_the_stage():
    chain, gpu, cpu, gv, cv = _make_typed()
    kinds = [BLOCK_ATTN, BLOCK_ATTN, BLOCK_STATE]
    # The state region of an attention block and the K/V of a state block
    # hold whatever: never counted.
    gv["state"][0][1][3, 0] = float("nan")
    gv["fa"][1][1][5, 0] = float("nan")
    _store_typed(chain, gpu, cpu, 0, [3, 4, 5], [7, 8, 9], kinds)
    chain.drain()
    assert chain.counters["nonfinite_events"] == 0, chain.reports

    # NaN in the V of an attention block before its store: the source is
    # named, and every later stage sees the same count.
    gv["fa"][1][1][4, 2] = float("nan")
    _store_typed(chain, gpu, cpu, 1, [3, 4, 5], [7, 8, 9], kinds)
    chain.drain()
    assert chain.reports == ["nonfinite store source(gpu before copy)"]
    for name in gpu:
        gpu[name][[3, 4, 5]] = 1
    _load_typed(chain, gpu, cpu, 0, [7, 8, 9], [1, 2, 6], kinds)
    chain.drain()
    assert chain.reports[-2:] == [
        "nonfinite load cpu-at-launch",
        "nonfinite load restored(gpu after copy)",
    ]
    assert chain.pop_load_verdict(0) == [1]
    assert chain.counters["mismatch_events"] == 0
    # Non-finite values appearing after the load are named at the watch.
    chain.load_seen_complete(0)
    _steps(chain, 2)
    gv["state"][0][1][6, 1] = float("inf")
    _steps(chain, WATCH_STEPS[-1])
    # The restored NaN is still there at every watch; the Inf written after
    # the load shows up from the +2 watch on.
    assert "nonfinite watch +0 steps" in chain.reports
    assert "nonfinite watch +2 steps" in chain.reports
    assert "watch +0 steps" not in chain.reports and "watch +2 steps" in chain.reports


def test_load_verdict_names_blocks_differing_from_the_cpu_copy():
    chain, gpu, cpu, gv, cv = _make_typed()
    kinds = [BLOCK_ATTN, BLOCK_ATTN, BLOCK_STATE]
    _store_typed(chain, gpu, cpu, 0, [3, 4, 5], [7, 8, 9], kinds)
    chain.drain()
    chain.load_launched(0, [7, 8, 9], [1, 2, 6], kinds=kinds)
    for name in gpu:
        gpu[name][[1, 2, 6]] = cpu[name][[7, 8, 9]]
    gpu["fa"][6, 0] ^= 1  # the copy delivered other bytes into position 2
    post = chain.snapshot(None)
    chain.transfer_issued(False, 0, None, post, None, chain.nonfinite(None, (False, 0)))
    chain.drain()
    assert "load cpu->gpu" in chain.reports
    assert chain.pop_load_verdict(0) == [2]


@pytest.mark.parametrize(
    "mode,damaged,part",
    [
        ("nan_v=1", [1], "v"),
        ("nan_v=2", [1, 2], "v"),
        ("nan_v_tail=1", [2], "v"),
        ("nan_k=1", [1], "k"),
        ("nan_k_tail=1", [2], "k"),
        ("inf_v=1", [1], "v"),
    ],
)
def test_corrupt_modes_hit_only_the_chosen_part(mode, damaged, part):
    chain, gpu, cpu, gv, cv = _make_typed()
    kinds = [BLOCK_ATTN, BLOCK_ATTN, BLOCK_STATE]
    gpu_ids = [1, 2, 6]
    _load_typed(chain, gpu, cpu, 0, [7, 8, 9], gpu_ids, kinds, corrupt=[(0, 3, mode)])
    chain.drain()
    chain.load_seen_complete(0)
    k, v = gv["fa"][0][1], gv["fa"][1][1]
    for gid in gpu_ids:
        hit = gid in damaged
        target, other = (v, k) if part == "v" else (k, v)
        assert bool(torch.isfinite(target[gid]).all()) == (not hit), (mode, gid)
        assert torch.isfinite(other[gid]).all(), (mode, gid)
        assert torch.isfinite(gv["state"][0][1][gid]).all()
    if mode == "inf_v=1":
        assert torch.isinf(v[1]).all()
    _steps(chain, 1)
    assert "watch +0 steps" in chain.reports


def test_corrupt_state_modes_use_block_kinds():
    chain, gpu, cpu, gv, cv = _make_typed()
    # The state block is not last in the list: kinds, not position, decide.
    kinds = [BLOCK_ATTN, BLOCK_STATE, BLOCK_ATTN]
    _load_typed(
        chain, gpu, cpu, 0, [7, 8, 9], [1, 6, 2], kinds, corrupt=[(0, 3, "nan_state")]
    )
    chain.drain()
    chain.load_seen_complete(0)
    assert torch.isnan(gv["state"][0][1][6]).all()
    assert torch.isfinite(gv["state"][0][1][[1, 2]]).all()
    assert torch.isfinite(gpu["fa"].view(torch.float16)[[1, 2, 6]]).all()

    chain, gpu, cpu, gv, cv = _make_typed()
    _load_typed(
        chain,
        gpu,
        cpu,
        0,
        [7, 8, 9],
        [1, 6, 2],
        kinds,
        corrupt=[(0, 3, "zero_state=1")],
    )
    chain.drain()
    chain.load_seen_complete(0)
    assert bool((gpu["state"][6] == 0).all()) and bool((gpu["fa"][6] == 0).all())
    assert not bool((gpu["fa"][2] == 0).all())


def test_corrupt_spans_apply_per_request():
    chain, gpu, cpu, gv, cv = _make_typed()
    kinds = [BLOCK_ATTN, BLOCK_STATE, BLOCK_ATTN, BLOCK_STATE]
    # Two requests in one load event; only the second one asked for damage.
    _load_typed(
        chain,
        gpu,
        cpu,
        0,
        [7, 8, 9, 10],
        [1, 6, 2, 9],
        kinds,
        corrupt=[(2, 4, "nan_v=1")],
    )
    chain.drain()
    chain.load_seen_complete(0)
    v = gv["fa"][1][1]
    assert torch.isfinite(v[1]).all() and torch.isnan(v[2]).all()


def test_part_modes_need_block_kinds():
    chain, gpu, cpu = _make()
    before = {name: t.clone() for name, t in gpu.items()}
    chain._corrupt("nan_v=1", 0, [1, 2, 6], None)
    chain._corrupt("nan_k_tail=1", 0, [1, 2, 6], None)
    assert all(torch.equal(gpu[name], before[name]) for name in gpu)


def test_corrupt_modes_are_validated():
    assert parse_corrupt_mode("nan_v=4") == ("nan_v", 4)
    assert parse_corrupt_mode(" inf_v ") == ("inf_v", 1)
    with pytest.raises(ValueError, match="unknown corrupt mode"):
        parse_corrupt_mode("melt=2")
    with pytest.raises(ValueError, match="count"):
        parse_corrupt_mode("nan_v=0")
    assert parse_options(
        {"fingerprint_corrupt_plan": "2:nan_v_tail=3"}
    ).fingerprint_corrupt_plan


def test_views_are_validated():
    chain, gpu, cpu = _make()
    with pytest.raises(ValueError, match="unknown part"):
        FingerprintChain(
            gpu,
            cpu,
            rank=0,
            start_thread=False,
            gpu_views={"fa": [("q", gpu["fa"].view(torch.float16))]},
        )
    with pytest.raises(ValueError, match="mixes"):
        FingerprintChain(
            gpu,
            cpu,
            rank=0,
            start_thread=False,
            gpu_views={
                "fa": [
                    ("k", gpu["fa"].view(torch.float16)),
                    ("state", gpu["fa"].view(torch.float16)),
                ]
            },
        )
    # Integer views are skipped: nothing is counted.
    chain = FingerprintChain(
        gpu, cpu, rank=0, start_thread=False, gpu_views={"fa": [("attn", gpu["fa"])]}
    )
    assert chain.count_nonfinite(None, [1], [BLOCK_ATTN]) is None


def test_regions_are_counted_for_their_own_group_only():
    """A block is counted only in the regions of its owner's KV cache group:
    an attention region of another group (e.g. the QSA ring) holds whatever
    an earlier owner left there."""
    chain, gpu, cpu, gv, cv = _make_typed()
    # Re-tag: "fa" belongs to group 2, no loaded block is owned by group 2.
    chain = FingerprintChain(
        gpu,
        cpu,
        rank=0,
        start_thread=False,
        gpu_views=gv,
        cpu_views=cv,
        region_groups={"fa": 2, "state": 1},
        state_groups={1},
    )
    gv["fa"][1][1][3, 0] = float("nan")
    snap = chain.count_nonfinite(None, [3, 4, 5], [0, 0, 1])
    counts = snap.wait()
    assert int(counts[0, 0]) == -1  # "fa" (group 2) not counted for group 0
    assert int(counts[1, 2]) == 0  # "state" counted for the group-1 block
    host = chain._host_nonfinite([7, 8, 9], [0, 0, 1])
    assert int(host[0, 0]) == -1 and int(host[1, 2]) == 0
    # With the block owned by group 2 the NaN is counted.
    counts = chain.count_nonfinite(None, [3, 4, 5], [2, 0, 1]).wait()
    assert int(counts[0, 0]) == 1
    # Corrupt modes tell attention from state blocks by state_groups.
    chain.load_launched(
        0, [7, 8, 9], [3, 4, 5], kinds=[2, 2, 1], corrupt=[(0, 3, "nan_v_tail=1")]
    )
    chain.load_seen_complete(0)
    assert torch.isnan(gv["fa"][1][1][4]).all()
    assert torch.isfinite(gv["fa"][1][1][5]).all()


@pytest.mark.parametrize(
    "mode,positions",
    [
        ("cpu_nan_v=1", [0]),
        ("cpu_nan_v_tail=1", [1]),
        ("cpu_zero_fa=2", [0, 1]),
        ("cpu_nan_state", [2]),
    ],
)
def test_cpu_corrupt_modes_damage_the_rows_before_the_copy(mode, positions):
    """cpu_* modes act on the CPU rows at launch: the load checks see the
    damage (cpu-at-rest and, for NaN, cpu-at-launch / restored) and the
    verdict names the positions, so a gated load fails on them."""
    chain, gpu, cpu, gv, cv = _make_typed()
    kinds = [BLOCK_ATTN, BLOCK_ATTN, BLOCK_STATE]
    _store_typed(chain, gpu, cpu, 0, [3, 4, 5], [7, 8, 9], kinds)
    chain.drain()
    before = {name: t.clone() for name, t in cpu.items()}
    _load_typed(chain, gpu, cpu, 0, [7, 8, 9], [1, 2, 6], kinds, corrupt=[(0, 3, mode)])
    chain.drain()
    assert chain.pop_load_verdict(0) == positions
    assert "load cpu-at-rest" in chain.reports
    if "nan" in mode:
        assert "nonfinite load cpu-at-launch" in chain.reports
        assert "nonfinite load restored(gpu after copy)" in chain.reports
    # Only the chosen rows changed, and nothing waits for load completion.
    cpu_ids = [7, 8, 9]
    for i, cid in enumerate(cpu_ids):
        changed = any(not torch.equal(cpu[n][cid], before[n][cid]) for n in cpu)
        assert changed == (i in positions), (mode, i)
    if mode.startswith("cpu_nan_v"):
        assert torch.isnan(cv["fa"][1][1][cpu_ids[positions[0]]]).all()
        assert torch.isfinite(cv["fa"][0][1][cpu_ids[positions[0]]]).all()
    assert not chain._corrupt_event
    chain.load_seen_complete(0)
    _steps(chain, 1)
    assert "watch +0 steps" not in chain.reports  # nothing damaged after the load
