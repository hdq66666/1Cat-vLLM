# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic fingerprint chain for the CPU tier (options.fingerprint).

Follows every transferred block through the tier with one checksum per KV
region, taken where the bytes are and without making the host wait for the
GPU. A checksum of every block of every region ("snapshot") costs one
reduction per region on a CUDA stream and is read back through pinned host
memory once its event has completed; a background thread does all waiting,
host-side checksumming and comparing.

Store of GPU block g into CPU block c:
    C    compute stream, at the launch point (the store stream waits for
         everything queued before it)
    PRE  store stream, right before the copy
    POST store stream, right after the copy
    H    host, from the pinned CPU block once the copy's event completed
  C != PRE    g was written between the launch point and the copy, i.e. while
              it was free and pinned for the store
  PRE != POST g was written while it was being copied
  POST != H   the CPU block does not hold what the GPU block held

Load of CPU block c into GPU block g:
    H0   host, from the pinned CPU block at launch
    L    load stream, right after the copy
    W_k  compute stream, k steps after the load was seen complete
  H0 != H(store)  the CPU block changed at rest
  L != H0         the GPU block does not hold what the CPU block held
  W_k != L        g was written after the load, while the request that owns
                  it was starting to use it

With typed views of the regions (gpu_views / cpu_views) the chain also counts
the non-finite values (NaN, Inf) of every transferred block in the regions
its owner uses: the GPU source before a store, the CPU block at a load's
launch, the GPU block after the load and at every watch step. A healthy
block holds none, so a count names the first stage at which a block carried
non-finite bytes.

Every load and store logs one line with a digest, so two restores of the same
CPU blocks can be compared from the log, and every mismatch logs a warning
naming the blocks and regions. A load's verdict (the positions whose block
differs from its CPU copy or carries non-finite values) is kept for the
worker, which may hold the load back from the scheduler on it.
"""

import queue
import threading
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Steps after a load was seen complete at which its destination blocks are
# checked again on the compute stream.
WATCH_STEPS = (0, 2, 4, 6)
_MAX_REPORTED = 4

# Block kinds of a transfer: what the block's owner keeps in it.
BLOCK_ATTN = 0  # full-attention K/V (plus whatever the state regions hold)
BLOCK_STATE = 1  # state-space state (plus whatever the K/V regions hold)
BLOCK_UNKNOWN = -1

# Parts of a region, for typed views: K or V of an attention layer, a whole
# attention region without a K/V split, or a state tensor.
REGION_PARTS = ("k", "v", "attn", "state")

# [regions, n] non-finite counts use -1 where a region was not counted.
NOT_COUNTED = -1

Views = dict[str, list[tuple[str, torch.Tensor]]]


class Snapshot:
    """Checksums of every block of every region at one point of one stream."""

    __slots__ = ("host", "event", "keep")

    def __init__(self, host: torch.Tensor, event: Any, keep: Any = None) -> None:
        self.host = host  # [regions, blocks] int64, valid once event completed
        self.event = event
        self.keep = keep  # tensors the queued kernels still read

    def wait(self) -> torch.Tensor:
        if self.event is not None:
            self.event.synchronize()
        self.keep = None
        return self.host


# zero_fa=N / nan_fa=N / swap_fa=N act on the first N attention blocks of a
# load (all regions), zero_state=N / nan_state=N on its last N state blocks.
# nan_v=N / inf_v=N / nan_k=N act only on the V (K) part of the first N
# attention blocks, nan_v_tail=N / nan_k_tail=N on the last N: keys stay
# valid, so the selected pages read non-finite values.
CORRUPT_MODES = (
    "zero_fa",
    "zero_state",
    "nan_fa",
    "swap_fa",
    "nan_state",
    "nan_v",
    "nan_v_tail",
    "inf_v",
    "nan_k",
    "nan_k_tail",
)
_PART_MODES = {
    "nan_v": "v",
    "nan_v_tail": "v",
    "inf_v": "v",
    "nan_k": "k",
    "nan_k_tail": "k",
}


def parse_corrupt_mode(mode: str) -> tuple[str, int]:
    """ "nan_v=4" -> ("nan_v", 4); the count defaults to 1."""
    kind, _, arg = mode.strip().partition("=")
    if kind not in CORRUPT_MODES:
        raise ValueError(f"unknown corrupt mode {mode!r}")
    count = int(arg) if arg else 1
    if count < 1:
        raise ValueError(f"corrupt mode {mode!r} needs a count >= 1")
    return kind, count


def parse_corrupt_plan(plan: str) -> dict[int, str]:
    """ "5:zero_state=4;21:zero_fa=8;37:nan_fa=1;53:swap_fa=8" -> {load
    number (1-based): mode}; see CORRUPT_MODES."""
    out: dict[int, str] = {}
    for item in filter(None, (part.strip() for part in plan.split(";"))):
        number, _, mode = item.partition(":")
        parse_corrupt_mode(mode)
        out[int(number)] = mode.strip()
    return out


def _digest(values: torch.Tensor) -> int:
    return int(values.sum().item()) & 0xFFFFFFFFFFFFFFFF


class FingerprintChain:
    def __init__(
        self,
        gpu_regions: dict[str, torch.Tensor],
        cpu_regions: dict[str, torch.Tensor],
        rank: int,
        inject_gpu_load: int = 0,
        inject_cpu_load: int = 0,
        corrupt_plan: str = "",
        start_thread: bool = True,
        gpu_views: Views | None = None,
        cpu_views: Views | None = None,
    ) -> None:
        self.rank = rank
        self.names = list(gpu_regions)
        # int64 views [blocks, words]: reductions over them allocate nothing
        # but their [blocks] result.
        self._gpu = [gpu_regions[name].view(torch.int64) for name in self.names]
        self._cpu = [cpu_regions[name].view(torch.int64) for name in self.names]
        self._on_cuda = self._gpu[0].is_cuda
        self._shape = (len(self.names), self._gpu[0].shape[0])
        self._inject_gpu_load = inject_gpu_load
        self._inject_cpu_load = inject_cpu_load
        self._inject_event = -1
        # Typed views per region, [(part, GPU view, CPU view or None)], the
        # first dim of a view being the block. Floating dtypes only. A region
        # holds either attention parts or state parts.
        self._parts: list[list[tuple[str, torch.Tensor, torch.Tensor | None]]] = []
        self._region_kind: list[str] = []
        for name in self.names:
            parts: list[tuple[str, torch.Tensor, torch.Tensor | None]] = []
            for part, view in (gpu_views or {}).get(name, []):
                if part not in REGION_PARTS:
                    raise ValueError(f"region {name}: unknown part {part!r}")
                if not view.is_floating_point():
                    continue
                cview = None
                for cpart, cv in (cpu_views or {}).get(name, []):
                    if cpart == part and cv.is_floating_point():
                        cview = cv
                        break
                parts.append((part, view, cview))
            kinds = {part for part, _, _ in parts}
            if "state" in kinds and len(kinds) > 1:
                raise ValueError(f"region {name} mixes state and attention parts")
            self._parts.append(parts)
            self._region_kind.append(
                "state" if "state" in kinds else "attn" if kinds else "other"
            )
        self._countable = any(self._parts)
        # Fault injection for experiments: {load number: mode}, see
        # parse_corrupt_plan(). Applied on the compute stream when the load
        # is seen complete, i.e. before the request's first step.
        self._corrupt_plan = parse_corrupt_plan(corrupt_plan)
        self._corrupt_event: dict[int, list[tuple[int, int, str]]] = {}

        # Main thread only.
        self._step = 0
        self._num_loads = 0
        self._watches: dict[int, list[tuple[int, int, list[int]]]] = {}
        self._load_blocks: dict[int, tuple[list[int], list[int] | None]] = {}
        # A CPU block's launch sequence number: a host checksum is only
        # meaningful while no later store was launched into the block.
        self._cpu_seq: dict[int, int] = {}
        self._seq = 0
        # (is_store, event) -> (GPU ids, block kinds) of launched transfers;
        # the copy thread reads an entry after the launch was queued to it.
        self._event_blocks: dict[
            tuple[bool, int], tuple[list[int], list[int] | None]
        ] = {}

        # Checker thread only.
        self._cpu_fp: dict[int, torch.Tensor] = {}  # at store completion
        # What the CPU block held before its current content: a load that
        # delivers this is reading a stale copy of the block.
        self._cpu_fp_prev: dict[int, torch.Tensor] = {}
        self._src_fp: dict[int, torch.Tensor] = {}  # GPU source before the store
        self._load_fp: dict[int, torch.Tensor] = {}  # event -> [regions, n] after load
        self._store_launch: dict[int, tuple[list[int], list[int], Snapshot, int]] = {}
        self._load_launch: dict[
            int,
            tuple[list[int], list[int], torch.Tensor, list[bool], torch.Tensor | None],
        ] = {}
        # Kinds of the last mismatches, newest last (tests, debugging).
        self.reports: list[str] = []
        self.counters = {
            "stores": 0,
            "loads": 0,
            "watches": 0,
            "mismatch_events": 0,
            "nonfinite_events": 0,
            "skipped": 0,
            "stale": 0,
        }
        # event -> positions of the loaded blocks that differ from their CPU
        # copy or carry non-finite values; set by the checker, taken by the
        # worker (pop_load_verdict).
        self._verdicts: dict[int, list[int]] = {}

        self._lock = threading.Lock()
        self._free: list[torch.Tensor] = []
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        if start_thread:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        logger.info(
            "SimpleCPU fp enabled: rank=%d regions=%d blocks=%d watch_steps=%s "
            "nonfinite=%s",
            rank,
            *self._shape,
            WATCH_STEPS,
            "counted" if self._countable else "not counted (no typed views)",
        )

    # ------------------------------------------------------------------
    # Snapshots (any thread; the stream must belong to the calling thread's
    # current device)
    # ------------------------------------------------------------------
    def _host_buffer(self) -> torch.Tensor:
        with self._lock:
            if self._free:
                return self._free.pop()
        return torch.empty(self._shape, dtype=torch.int64, pin_memory=self._on_cuda)

    def _release(self, snap: Snapshot | None) -> None:
        if snap is not None and tuple(snap.host.shape) == self._shape:
            with self._lock:
                self._free.append(snap.host)

    @torch.inference_mode()
    def snapshot(self, stream: Any) -> Snapshot:
        # Host buffers are shared by the engine thread (inference mode) and
        # the copy thread: every use of them runs in inference mode.
        host = self._host_buffer()
        if not self._on_cuda:
            host.copy_(torch.stack([r.sum(dim=1) for r in self._gpu]))
            return Snapshot(host, None)
        with torch.cuda.stream(stream):
            sums = torch.stack([r.sum(dim=1) for r in self._gpu])
            host.copy_(sums, non_blocking=True)
            event = torch.cuda.Event()
            event.record(stream)
        return Snapshot(host, event)

    @torch.inference_mode()
    def nonfinite(self, stream: Any, key: tuple[bool, int]) -> Snapshot | None:
        """Non-finite counts of a launched transfer's GPU blocks (copy
        thread); None when nothing is counted."""
        blocks = self._event_blocks.get(key)
        if blocks is None:
            return None
        return self.count_nonfinite(stream, *blocks)

    @torch.inference_mode()
    def count_nonfinite(
        self, stream: Any, gpu_ids: list[int], kinds: list[int] | None
    ) -> Snapshot | None:
        """[regions, n] counts of non-finite values of the given GPU blocks
        in the regions their kind uses, on the host once the snapshot's
        event completed. Nothing here waits for the GPU."""
        if not self._countable or not gpu_ids or kinds is None:
            return None
        positions = {
            "attn": [i for i, k in enumerate(kinds) if k == BLOCK_ATTN],
            "state": [i for i, k in enumerate(kinds) if k == BLOCK_STATE],
        }
        if not positions["attn"] and not positions["state"]:
            return None
        n = len(gpu_ids)
        device = self._gpu[0].device
        host = torch.full(
            (len(self.names), n),
            NOT_COUNTED,
            dtype=torch.int64,
            pin_memory=self._on_cuda,
        )
        keep: list[torch.Tensor] = []

        def index(values: list[int]) -> torch.Tensor:
            t = torch.tensor(values, dtype=torch.int64)
            if self._on_cuda:
                t = t.pin_memory().to(device, non_blocking=True)
                keep.append(t)
            return t

        def count() -> torch.Tensor:
            out = torch.full(
                (len(self.names), n), NOT_COUNTED, dtype=torch.int64, device=device
            )
            for r, parts in enumerate(self._parts):
                if not parts:
                    continue
                pos = positions[self._region_kind[r]]
                if not pos:
                    continue
                ids = index([gpu_ids[i] for i in pos])
                total: torch.Tensor | None = None
                for _, view, _ in parts:
                    c = (~torch.isfinite(view[ids])).flatten(1).sum(dim=1)
                    total = c if total is None else total + c
                out[r, index(pos)] = total
            return out

        if not self._on_cuda:
            host.copy_(count())
            return Snapshot(host, None)
        with torch.cuda.stream(stream):
            out = count()
            host.copy_(out, non_blocking=True)
            keep.append(out)
            event = torch.cuda.Event()
            event.record(stream)
        return Snapshot(host, event, keep)

    def _host_fp(self, cpu_ids: list[int]) -> torch.Tensor:
        """[regions, n] checksums of CPU blocks, read from host memory."""
        out = torch.empty((len(self.names), len(cpu_ids)), dtype=torch.int64)
        for r, region in enumerate(self._cpu):
            for i, cid in enumerate(cpu_ids):
                out[r, i] = region[cid].sum()
        return out

    def _host_nonfinite(
        self, cpu_ids: list[int], kinds: list[int] | None
    ) -> torch.Tensor | None:
        """[regions, n] non-finite counts of CPU blocks, read from host memory."""
        if not self._countable or kinds is None:
            return None
        out = torch.full(
            (len(self.names), len(cpu_ids)), NOT_COUNTED, dtype=torch.int64
        )
        for r, parts in enumerate(self._parts):
            views = [cv for _, _, cv in parts if cv is not None]
            if not parts or len(views) != len(parts):
                continue
            want = BLOCK_STATE if self._region_kind[r] == "state" else BLOCK_ATTN
            for i, cid in enumerate(cpu_ids):
                if kinds[i] != want:
                    continue
                out[r, i] = sum(int((~torch.isfinite(cv[cid])).sum()) for cv in views)
        return out

    # ------------------------------------------------------------------
    # Main thread
    # ------------------------------------------------------------------
    def store_launched(
        self,
        event_idx: int,
        gpu_ids: list[int],
        cpu_ids: list[int],
        stream: Any,
        kinds: list[int] | None = None,
    ) -> None:
        self._seq += 1
        for cid in cpu_ids:
            self._cpu_seq[cid] = self._seq
        self._event_blocks[(True, event_idx)] = (list(gpu_ids), kinds)
        snap = self.snapshot(stream)
        self._queue.put(
            ("store_launch", event_idx, list(gpu_ids), list(cpu_ids), snap, self._seq)
        )

    def load_launched(
        self,
        event_idx: int,
        cpu_ids: list[int],
        gpu_ids: list[int],
        kinds: list[int] | None = None,
        corrupt: list[tuple[int, int, str]] | None = None,
    ) -> None:
        """corrupt: (start, end, mode) spans of the load's block list to
        damage once the load is seen complete (see CORRUPT_MODES)."""
        self._num_loads += 1
        self._load_blocks[event_idx] = (list(gpu_ids), kinds)
        self._event_blocks[(False, event_idx)] = (list(gpu_ids), kinds)
        if self._num_loads == self._inject_gpu_load:
            self._inject_event = event_idx
        spans = list(corrupt or [])
        mode = self._corrupt_plan.get(self._num_loads)
        if mode is not None:
            spans.append((0, len(gpu_ids), mode))
        if spans:
            self._corrupt_event[event_idx] = spans
        if self._num_loads == self._inject_cpu_load:
            # Positive control: change one word of the first CPU block.
            self._cpu[0][cpu_ids[0], 0] ^= 1
            logger.warning(
                "SimpleCPU fp INJECT cpu rank=%d load event=%d cpu=%d region=%s",
                self.rank,
                event_idx,
                cpu_ids[0],
                self.names[0],
            )
        seqs = [self._cpu_seq.get(cid, 0) for cid in cpu_ids]
        self._queue.put(
            ("load_launch", event_idx, list(cpu_ids), list(gpu_ids), seqs, kinds)
        )

    def load_seen_complete(self, event_idx: int) -> None:
        blocks = self._load_blocks.pop(event_idx, None)
        if blocks is None:
            return
        gpu_ids, kinds = blocks
        for k in WATCH_STEPS:
            self._watches.setdefault(self._step + k, []).append((event_idx, k, gpu_ids))
        if event_idx == self._inject_event:
            self._inject_event = -1
            # Positive control: change one word of the first loaded GPU block
            # after the step-0 check, so the later checks must report it.
            self._watches.setdefault(self._step, []).append((event_idx, -1, gpu_ids))
        for start, end, mode in self._corrupt_event.pop(event_idx, []):
            self._corrupt(
                mode,
                event_idx,
                gpu_ids[start:end],
                kinds[start:end] if kinds is not None else None,
            )

    def _corrupt(
        self,
        mode: str,
        event_idx: int,
        gpu_ids: list[int],
        kinds: list[int] | None,
    ) -> None:
        """Deliberately damage loaded blocks (corrupt plans and request
        directives). Without block kinds the load list is taken as the
        full-attention blocks in prompt order followed by the state blocks."""
        kind, count = parse_corrupt_mode(mode)
        part = _PART_MODES.get(kind)
        if kinds is None and (part is not None or kind.endswith("_tail")):
            logger.warning(
                "SimpleCPU fp CORRUPT %s rank=%d load event=%d: needs block "
                "kinds, nothing damaged",
                mode,
                self.rank,
                event_idx,
            )
            return
        if kinds is None:
            attn = gpu_ids
            state = gpu_ids
        else:
            attn = [g for g, k in zip(gpu_ids, kinds) if k != BLOCK_STATE]
            state = [g for g, k in zip(gpu_ids, kinds) if k == BLOCK_STATE]
        if kind in ("zero_state", "nan_state"):
            targets = state[-count:]
        elif kind.endswith("_tail"):
            targets = attn[-count:]
        else:
            targets = attn[:count]
        value = float("inf") if kind == "inf_v" else float("nan")
        touched = 0
        for r, region in enumerate(self._gpu):
            for i, gid in enumerate(targets):
                if kind in ("zero_fa", "zero_state"):
                    region[gid].zero_()
                elif kind == "nan_fa":
                    region[gid].fill_(-1)  # all bits set: NaN in fp16/fp32
                elif kind == "nan_state":
                    if self._region_kind[r] == "state":
                        region[gid].fill_(-1)
                elif kind == "swap_fa":
                    # Another block of the same request: finite, wrong content.
                    other = gpu_ids[(len(gpu_ids) // 2 + i) % len(gpu_ids)]
                    region[gid].copy_(region[other])
                else:
                    for p, view, _ in self._parts[r]:
                        if p == part:
                            view[gid].fill_(value)
                            touched += 1
        if part is not None and not touched:
            logger.warning(
                "SimpleCPU fp CORRUPT %s rank=%d load event=%d: no %s views, "
                "nothing damaged",
                mode,
                self.rank,
                event_idx,
                part.upper(),
            )
            return
        logger.warning(
            "SimpleCPU fp CORRUPT %s rank=%d load event=%d gpu=%s",
            mode,
            self.rank,
            event_idx,
            targets,
        )

    def tick(self, stream: Any) -> None:
        """Once per engine step, after the step's transfers were submitted."""
        due = self._watches.pop(self._step, None)
        self._step += 1
        if not due:
            return
        snap: Snapshot | None = None
        for event_idx, k, gpu_ids in due:
            if k < 0:
                self._gpu[0][gpu_ids[0], 0] ^= 1
                logger.warning(
                    "SimpleCPU fp INJECT gpu rank=%d load event=%d gpu=%d region=%s",
                    self.rank,
                    event_idx,
                    gpu_ids[0],
                    self.names[0],
                )
                continue
            if snap is None:
                snap = self.snapshot(stream)
            kinds = self._event_blocks.get((False, event_idx), (None, None))[1]
            nf = self.count_nonfinite(stream, gpu_ids, kinds)
            self._queue.put(("watch", event_idx, k, gpu_ids, snap, nf))
            if k == WATCH_STEPS[-1]:
                self._event_blocks.pop((False, event_idx), None)
        if snap is not None:
            self._queue.put(("release", snap))

    def pop_load_verdict(self, event_idx: int) -> list[int] | None:
        """Positions of a completed load's bad blocks (differing from the
        CPU copy, or non-finite), or None while the checker has not
        finished with the load."""
        with self._lock:
            return self._verdicts.pop(event_idx, None)

    # ------------------------------------------------------------------
    # Copy thread
    # ------------------------------------------------------------------
    def transfer_issued(
        self,
        is_store: bool,
        event_idx: int,
        pre: Snapshot | None,
        post: Snapshot,
        done: Any,
        nonfinite: Snapshot | None = None,
    ) -> None:
        self._queue.put(
            ("store" if is_store else "load", event_idx, pre, post, done, nonfinite)
        )

    def close(self) -> None:
        self._queue.put(None)

    # ------------------------------------------------------------------
    # Checker thread
    # ------------------------------------------------------------------
    def _run(self) -> None:
        try:
            if self._on_cuda:
                torch.accelerator.set_device_index(self._gpu[0].device.index)
            while True:
                job = self._queue.get()
                if job is None:
                    return
                self.process(job)
        except BaseException:
            logger.exception("SimpleCPU fp checker thread failed; checks stopped")

    def drain(self) -> None:
        """Process queued jobs on the calling thread (tests)."""
        while True:
            try:
                job = self._queue.get_nowait()
            except queue.Empty:
                return
            if job is not None:
                self.process(job)

    @torch.inference_mode()
    def process(self, job: tuple) -> None:
        kind = job[0]
        if kind == "store_launch":
            _, event_idx, gpu_ids, cpu_ids, snap, seq = job
            self._store_launch[event_idx] = (gpu_ids, cpu_ids, snap, seq)
        elif kind == "load_launch":
            _, event_idx, cpu_ids, gpu_ids, seqs, kinds = job
            # The CPU blocks are pinned for the load: checksum them now. If a
            # store was launched into one of them meanwhile (this thread fell
            # far behind), the checksum of that block proves nothing.
            host0 = self._host_fp(cpu_ids)
            valid = [
                self._cpu_seq.get(cid, 0) == seq for cid, seq in zip(cpu_ids, seqs)
            ]
            host0_nf = self._host_nonfinite(cpu_ids, kinds)
            self._load_launch[event_idx] = (cpu_ids, gpu_ids, host0, valid, host0_nf)
        elif kind == "store":
            self._check_store(*job[1:])
        elif kind == "load":
            self._check_load(*job[1:])
        elif kind == "watch":
            self._check_watch(*job[1:])
        elif kind == "release":
            self._release(job[1])

    def _report(
        self,
        what: str,
        event_idx: int,
        before: torch.Tensor,
        after: torch.Tensor,
        gpu_ids: list[int],
        cpu_ids: list[int] | None,
    ) -> list[int]:
        """Warn about differing blocks; before/after are [regions, n].
        Returns the positions of the differing blocks."""
        diff = before != after
        bad_blocks = diff.any(dim=0).nonzero().flatten().tolist()
        if not bad_blocks:
            return []
        self.counters["mismatch_events"] += 1
        self.reports = (self.reports + [what])[-64:]
        details = []
        for i in bad_blocks[:_MAX_REPORTED]:
            regions = diff[:, i].nonzero().flatten().tolist()
            names = ",".join(
                f"{self.names[r]}:{int(before[r, i]) & 0xFFFFFFFF:08x}->"
                f"{int(after[r, i]) & 0xFFFFFFFF:08x}"
                for r in regions[:3]
            )
            zero = bool((after[:, i] == 0).all())
            details.append(
                f"[pos={i} gpu={gpu_ids[i]}"
                + (f" cpu={cpu_ids[i]}" if cpu_ids is not None else "")
                + f" regions={len(regions)}/{len(self.names)}"
                + (" NOW-ALL-ZERO" if zero else "")
                + f" {names}]"
            )
        logger.warning(
            "SimpleCPU fp MISMATCH %s rank=%d event=%d blocks=%d/%d positions=%s %s",
            what,
            self.rank,
            event_idx,
            len(bad_blocks),
            len(gpu_ids),
            bad_blocks[:16],
            " ".join(details),
        )
        return bad_blocks

    def _report_nonfinite(
        self,
        what: str,
        event_idx: int,
        counts: torch.Tensor | None,
        gpu_ids: list[int],
        cpu_ids: list[int] | None,
    ) -> list[int]:
        """Warn about blocks carrying non-finite values; counts is
        [regions, n] with NOT_COUNTED where a region was not counted.
        Returns the positions of those blocks."""
        if counts is None:
            return []
        bad = (counts > 0).any(dim=0).nonzero().flatten().tolist()
        if not bad:
            return []
        self.counters["nonfinite_events"] += 1
        self.reports = (self.reports + [f"nonfinite {what}"])[-64:]
        details = []
        for i in bad[:_MAX_REPORTED]:
            regions = (counts[:, i] > 0).nonzero().flatten().tolist()
            names = ",".join(
                f"{self.names[r]}:{int(counts[r, i])}" for r in regions[:3]
            )
            details.append(
                f"[pos={i} gpu={gpu_ids[i]}"
                + (f" cpu={cpu_ids[i]}" if cpu_ids is not None else "")
                + f" regions={len(regions)}"
                + f" total={int(counts[:, i].clamp(min=0).sum())}"
                + f" {names}]"
            )
        logger.warning(
            "SimpleCPU fp NONFINITE %s rank=%d event=%d blocks=%d/%d positions=%s %s",
            what,
            self.rank,
            event_idx,
            len(bad),
            len(gpu_ids),
            bad[:16],
            " ".join(details),
        )
        return bad

    @staticmethod
    def _nf_total(counts: torch.Tensor | None) -> str:
        if counts is None:
            return "-"
        return str(int(counts.clamp(min=0).sum()))

    def _check_store(
        self,
        event_idx: int,
        pre: Snapshot,
        post: Snapshot,
        done: Any,
        nonfinite: Snapshot | None = None,
    ) -> None:
        launch = self._store_launch.pop(event_idx, None)
        if done is not None:
            done.synchronize()
        if launch is None:
            self._release(pre)
            self._release(post)
            return
        gpu_ids, cpu_ids, at_launch, seq = launch
        c = at_launch.wait()[:, gpu_ids].clone()
        p0 = pre.wait()[:, gpu_ids].clone()
        p1 = post.wait()[:, gpu_ids].clone()
        for snap in (at_launch, pre, post):
            self._release(snap)
        nf = nonfinite.wait().clone() if nonfinite is not None else None
        host = self._host_fp(cpu_ids)
        bad = len(
            self._report("store launch->copy", event_idx, c, p0, gpu_ids, cpu_ids)
        )
        bad += len(
            self._report("store during-copy", event_idx, p0, p1, gpu_ids, cpu_ids)
        )
        # A later store into one of the CPU blocks makes the host checksum
        # meaningless for that block.
        fresh = [i for i, cid in enumerate(cpu_ids) if self._cpu_seq.get(cid) == seq]
        if len(fresh) != len(cpu_ids):
            self.counters["skipped"] += len(cpu_ids) - len(fresh)
        if fresh:
            bad += len(
                self._report(
                    "store gpu->cpu",
                    event_idx,
                    p1[:, fresh],
                    host[:, fresh],
                    [gpu_ids[i] for i in fresh],
                    [cpu_ids[i] for i in fresh],
                )
            )
        bad += len(
            self._report_nonfinite(
                "store source(gpu before copy)", event_idx, nf, gpu_ids, cpu_ids
            )
        )
        for i in fresh:
            old = self._cpu_fp.get(cpu_ids[i])
            if old is not None:
                self._cpu_fp_prev[cpu_ids[i]] = old
            self._cpu_fp[cpu_ids[i]] = host[:, i].clone()
            self._src_fp[cpu_ids[i]] = p0[:, i].clone()
        self.counters["stores"] += 1
        logger.info(
            "SimpleCPU fp store rank=%d event=%d blocks=%d digest=%016x nf=%s %s",
            self.rank,
            event_idx,
            len(cpu_ids),
            _digest(host),
            self._nf_total(nf),
            "ok" if not bad else f"BAD({bad})",
        )

    def _check_load(
        self,
        event_idx: int,
        pre: Snapshot | None,
        post: Snapshot,
        done: Any,
        nonfinite: Snapshot | None = None,
    ) -> None:
        launch = self._load_launch.pop(event_idx, None)
        if done is not None:
            done.synchronize()
        self._release(pre)
        if launch is None:
            self._release(post)
            with self._lock:
                self._verdicts[event_idx] = []
            return
        cpu_ids, gpu_ids, host0, valid, host0_nf = launch
        loaded = post.wait()[:, gpu_ids].clone()
        self._release(post)
        nf = nonfinite.wait().clone() if nonfinite is not None else None
        self._load_fp[event_idx] = loaded
        verdict: set[int] = set()
        bad = 0
        if not all(valid):
            self.counters["skipped"] += valid.count(False)
        known = [i for i, cid in enumerate(cpu_ids) if valid[i] and cid in self._cpu_fp]
        if known:
            at_store = torch.stack([self._cpu_fp[cpu_ids[i]] for i in known], dim=1)
            source = torch.stack([self._src_fp[cpu_ids[i]] for i in known], dim=1)
            kg = [gpu_ids[i] for i in known]
            kc = [cpu_ids[i] for i in known]
            at_rest = self._report(
                "load cpu-at-rest", event_idx, at_store, host0[:, known], kg, kc
            )
            verdict.update(known[i] for i in at_rest)
            bad += len(at_rest)
            bad += len(
                self._report(
                    "load end-to-end(gpu source vs gpu restored)",
                    event_idx,
                    source,
                    loaded[:, known],
                    kg,
                    kc,
                )
            )
        ok = [i for i in range(len(cpu_ids)) if valid[i]]
        stale = [
            i
            for i in ok
            if cpu_ids[i] in self._cpu_fp_prev
            and not torch.equal(loaded[:, i], host0[:, i])
            and torch.equal(loaded[:, i], self._cpu_fp_prev[cpu_ids[i]])
        ]
        if stale:
            self.counters["stale"] += len(stale)
            logger.warning(
                "SimpleCPU fp STALE rank=%d event=%d: %d loaded blocks hold the "
                "PREVIOUS content of their CPU block: positions=%s cpu=%s",
                self.rank,
                event_idx,
                len(stale),
                stale[:16],
                [cpu_ids[i] for i in stale[:16]],
            )
        cpu2gpu = self._report(
            "load cpu->gpu",
            event_idx,
            host0[:, ok],
            loaded[:, ok],
            [gpu_ids[i] for i in ok],
            [cpu_ids[i] for i in ok],
        )
        verdict.update(ok[i] for i in cpu2gpu)
        bad += len(cpu2gpu)
        bad += len(
            self._report_nonfinite(
                "load cpu-at-launch", event_idx, host0_nf, gpu_ids, cpu_ids
            )
        )
        restored_nf = self._report_nonfinite(
            "load restored(gpu after copy)", event_idx, nf, gpu_ids, cpu_ids
        )
        verdict.update(restored_nf)
        bad += len(restored_nf)
        with self._lock:
            self._verdicts[event_idx] = sorted(verdict)
        self.counters["loads"] += 1
        logger.info(
            "SimpleCPU fp load rank=%d event=%d blocks=%d known=%d digest=%016x "
            "nf=%s %s",
            self.rank,
            event_idx,
            len(cpu_ids),
            len(known),
            _digest(loaded),
            self._nf_total(nf),
            "ok" if not bad else f"BAD({bad})",
        )

    def _check_watch(
        self,
        event_idx: int,
        k: int,
        gpu_ids: list[int],
        snap: Snapshot,
        nonfinite: Snapshot | None = None,
    ) -> None:
        loaded = self._load_fp.get(event_idx)
        nf = nonfinite.wait().clone() if nonfinite is not None else None
        if loaded is None:
            return
        now = snap.wait()[:, gpu_ids].clone()
        bad = len(
            self._report(f"watch +{k} steps", event_idx, loaded, now, gpu_ids, None)
        )
        bad += len(
            self._report_nonfinite(f"watch +{k} steps", event_idx, nf, gpu_ids, None)
        )
        self.counters["watches"] += 1
        logger.info(
            "SimpleCPU fp watch rank=%d event=%d step=+%d digest=%016x nf=%s %s",
            self.rank,
            event_idx,
            k,
            _digest(now),
            self._nf_total(nf),
            "ok" if not bad else f"BAD({bad})",
        )
        if k == WATCH_STEPS[-1]:
            self._load_fp.pop(event_idx, None)
