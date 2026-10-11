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

Every load and store logs one line with a digest, so two restores of the same
CPU blocks can be compared from the log, and every mismatch logs a warning
naming the blocks and regions.
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


class Snapshot:
    """Checksums of every block of every region at one point of one stream."""

    __slots__ = ("host", "event")

    def __init__(self, host: torch.Tensor, event: Any) -> None:
        self.host = host  # [regions, blocks] int64, valid once event completed
        self.event = event

    def wait(self) -> torch.Tensor:
        if self.event is not None:
            self.event.synchronize()
        return self.host


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
        start_thread: bool = True,
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

        # Main thread only.
        self._step = 0
        self._num_loads = 0
        self._watches: dict[int, list[tuple[int, int, list[int]]]] = {}
        self._load_blocks: dict[int, list[int]] = {}
        # A CPU block's launch sequence number: a host checksum is only
        # meaningful while no later store was launched into the block.
        self._cpu_seq: dict[int, int] = {}
        self._seq = 0

        # Checker thread only.
        self._cpu_fp: dict[int, torch.Tensor] = {}  # at store completion
        self._src_fp: dict[int, torch.Tensor] = {}  # GPU source before the store
        self._load_fp: dict[int, torch.Tensor] = {}  # event -> [regions, n] after load
        self._store_launch: dict[int, tuple[list[int], list[int], Snapshot, int]] = {}
        self._load_launch: dict[
            int, tuple[list[int], list[int], torch.Tensor, list[bool]]
        ] = {}
        # Kinds of the last mismatches, newest last (tests, debugging).
        self.reports: list[str] = []
        self.counters = {
            "stores": 0,
            "loads": 0,
            "watches": 0,
            "mismatch_events": 0,
            "skipped": 0,
        }

        self._lock = threading.Lock()
        self._free: list[torch.Tensor] = []
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        if start_thread:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        logger.info(
            "SimpleCPU fp enabled: rank=%d regions=%d blocks=%d watch_steps=%s",
            rank,
            *self._shape,
            WATCH_STEPS,
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
        if snap is not None:
            with self._lock:
                self._free.append(snap.host)

    def snapshot(self, stream: Any) -> Snapshot:
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

    def _host_fp(self, cpu_ids: list[int]) -> torch.Tensor:
        """[regions, n] checksums of CPU blocks, read from host memory."""
        out = torch.empty((len(self.names), len(cpu_ids)), dtype=torch.int64)
        for r, region in enumerate(self._cpu):
            for i, cid in enumerate(cpu_ids):
                out[r, i] = region[cid].sum()
        return out

    # ------------------------------------------------------------------
    # Main thread
    # ------------------------------------------------------------------
    def store_launched(
        self, event_idx: int, gpu_ids: list[int], cpu_ids: list[int], stream: Any
    ) -> None:
        self._seq += 1
        for cid in cpu_ids:
            self._cpu_seq[cid] = self._seq
        snap = self.snapshot(stream)
        self._queue.put(
            ("store_launch", event_idx, list(gpu_ids), list(cpu_ids), snap, self._seq)
        )

    def load_launched(
        self, event_idx: int, cpu_ids: list[int], gpu_ids: list[int]
    ) -> None:
        self._num_loads += 1
        self._load_blocks[event_idx] = list(gpu_ids)
        if self._num_loads == self._inject_gpu_load:
            self._inject_event = event_idx
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
        self._queue.put(("load_launch", event_idx, list(cpu_ids), list(gpu_ids), seqs))

    def load_seen_complete(self, event_idx: int) -> None:
        gpu_ids = self._load_blocks.pop(event_idx, None)
        if gpu_ids is None:
            return
        for k in WATCH_STEPS:
            self._watches.setdefault(self._step + k, []).append(
                (event_idx, k, gpu_ids)
            )
        if event_idx == self._inject_event:
            self._inject_event = -1
            # Positive control: change one word of the first loaded GPU block
            # after the step-0 check, so the later checks must report it.
            self._watches.setdefault(self._step, []).append((event_idx, -1, gpu_ids))

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
            self._queue.put(("watch", event_idx, k, gpu_ids, snap))
        if snap is not None:
            self._queue.put(("release", snap))

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
    ) -> None:
        self._queue.put(("store" if is_store else "load", event_idx, pre, post, done))

    def close(self) -> None:
        self._queue.put(None)

    # ------------------------------------------------------------------
    # Checker thread
    # ------------------------------------------------------------------
    def _run(self) -> None:
        try:
            if self._on_cuda:
                torch.cuda.set_device(self._gpu[0].device)
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

    def process(self, job: tuple) -> None:
        kind = job[0]
        if kind == "store_launch":
            _, event_idx, gpu_ids, cpu_ids, snap, seq = job
            self._store_launch[event_idx] = (gpu_ids, cpu_ids, snap, seq)
        elif kind == "load_launch":
            _, event_idx, cpu_ids, gpu_ids, seqs = job
            # The CPU blocks are pinned for the load: checksum them now. If a
            # store was launched into one of them meanwhile (this thread fell
            # far behind), the checksum of that block proves nothing.
            host0 = self._host_fp(cpu_ids)
            valid = [self._cpu_seq.get(cid, 0) == seq for cid, seq in zip(cpu_ids, seqs)]
            self._load_launch[event_idx] = (cpu_ids, gpu_ids, host0, valid)
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
    ) -> int:
        """Warn about differing blocks; before/after are [regions, n]."""
        diff = before != after
        bad_blocks = diff.any(dim=0).nonzero().flatten().tolist()
        if not bad_blocks:
            return 0
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
        return len(bad_blocks)

    def _check_store(
        self, event_idx: int, pre: Snapshot, post: Snapshot, done: Any
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
        host = self._host_fp(cpu_ids)
        bad = self._report("store launch->copy", event_idx, c, p0, gpu_ids, cpu_ids)
        bad += self._report("store during-copy", event_idx, p0, p1, gpu_ids, cpu_ids)
        # A later store into one of the CPU blocks makes the host checksum
        # meaningless for that block.
        fresh = [i for i, cid in enumerate(cpu_ids) if self._cpu_seq.get(cid) == seq]
        if len(fresh) != len(cpu_ids):
            self.counters["skipped"] += len(cpu_ids) - len(fresh)
        if fresh:
            bad += self._report(
                "store gpu->cpu",
                event_idx,
                p1[:, fresh],
                host[:, fresh],
                [gpu_ids[i] for i in fresh],
                [cpu_ids[i] for i in fresh],
            )
        for i in fresh:
            self._cpu_fp[cpu_ids[i]] = host[:, i].clone()
            self._src_fp[cpu_ids[i]] = p0[:, i].clone()
        self.counters["stores"] += 1
        logger.info(
            "SimpleCPU fp store rank=%d event=%d blocks=%d digest=%016x %s",
            self.rank,
            event_idx,
            len(cpu_ids),
            _digest(host),
            "ok" if not bad else f"BAD({bad})",
        )

    def _check_load(
        self, event_idx: int, pre: Snapshot | None, post: Snapshot, done: Any
    ) -> None:
        launch = self._load_launch.pop(event_idx, None)
        if done is not None:
            done.synchronize()
        self._release(pre)
        if launch is None:
            self._release(post)
            return
        cpu_ids, gpu_ids, host0, valid = launch
        loaded = post.wait()[:, gpu_ids].clone()
        self._release(post)
        self._load_fp[event_idx] = loaded
        bad = 0
        if not all(valid):
            self.counters["skipped"] += valid.count(False)
        known = [
            i for i, cid in enumerate(cpu_ids) if valid[i] and cid in self._cpu_fp
        ]
        if known:
            at_store = torch.stack([self._cpu_fp[cpu_ids[i]] for i in known], dim=1)
            source = torch.stack([self._src_fp[cpu_ids[i]] for i in known], dim=1)
            kg = [gpu_ids[i] for i in known]
            kc = [cpu_ids[i] for i in known]
            bad += self._report("load cpu-at-rest", event_idx, at_store, host0[:, known], kg, kc)
            bad += self._report(
                "load end-to-end(gpu source vs gpu restored)",
                event_idx,
                source,
                loaded[:, known],
                kg,
                kc,
            )
        ok = [i for i in range(len(cpu_ids)) if valid[i]]
        bad += self._report(
            "load cpu->gpu",
            event_idx,
            host0[:, ok],
            loaded[:, ok],
            [gpu_ids[i] for i in ok],
            [cpu_ids[i] for i in ok],
        )
        self.counters["loads"] += 1
        logger.info(
            "SimpleCPU fp load rank=%d event=%d blocks=%d known=%d digest=%016x %s",
            self.rank,
            event_idx,
            len(cpu_ids),
            len(known),
            _digest(loaded),
            "ok" if not bad else f"BAD({bad})",
        )

    def _check_watch(
        self, event_idx: int, k: int, gpu_ids: list[int], snap: Snapshot
    ) -> None:
        loaded = self._load_fp.get(event_idx)
        if loaded is None:
            return
        now = snap.wait()[:, gpu_ids].clone()
        bad = self._report(f"watch +{k} steps", event_idx, loaded, now, gpu_ids, None)
        self.counters["watches"] += 1
        logger.info(
            "SimpleCPU fp watch rank=%d event=%d step=+%d digest=%016x %s",
            self.rank,
            event_idx,
            k,
            _digest(now),
            "ok" if not bad else f"BAD({bad})",
        )
        if k == WATCH_STEPS[-1]:
            self._load_fp.pop(event_idx, None)
