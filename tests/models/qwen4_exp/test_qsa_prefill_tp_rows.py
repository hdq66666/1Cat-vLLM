# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_SM70_QSA_PREFILL_TP_ROWS: four ranks score a quarter of a large
prefill chunk each and all-gather the block ids; the result must equal the
replicated selector bit for bit. One process plays all four ranks."""

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="the cuBLAS selector and the TP admission are SM70 only",
)


class _FakeRank:
    """A TP4 group seen from one rank. First pass: record this rank's local
    block ids. Second pass: hand back all four ranks' ids, rank-major."""

    world_size = 4

    def __init__(self, rank: int, store: dict):
        self.rank_in_group = rank
        self.store = store
        self.calls = 0

    def all_gather(self, t: torch.Tensor, dim: int = 0) -> torch.Tensor:
        assert dim == 0
        self.calls += 1
        if len(self.store) < self.world_size:
            self.store[self.rank_in_group] = t.clone()
            return torch.cat([t] * self.world_size)
        return torch.cat([self.store[r] for r in range(self.world_size)])


def _inputs(rows: int, context: int, seed: int = 0):
    torch.manual_seed(seed)
    page = 16
    keys = context // 4
    pages = keys // page
    q = torch.randn(rows, 4, 128, device="cuda", dtype=torch.float16)
    k_cache = torch.randn(pages + 3, page, 1, 128, device="cuda", dtype=torch.float16)
    page_table = torch.randperm(pages + 3, device="cuda", dtype=torch.int32)[:pages]
    page_table = page_table.unsqueeze(0).contiguous()
    token_to_req = torch.zeros(rows, device="cuda", dtype=torch.int32)
    # The last chunk of a `context`-token prompt.
    query_positions = torch.arange(
        context - rows, context, device="cuda", dtype=torch.int32
    )
    sequence_lengths = torch.tensor([context], device="cuda", dtype=torch.int32)
    return (q, k_cache, page_table, token_to_req, query_positions, sequence_lengths)


def _select(args, tp_group=None):
    return qsa_ops.qsa_select_paged_tokens(*args, 2048, 4, tp_group=tp_group)


@pytest.mark.parametrize(
    "rows,context", [(4096, 16384), (4097, 16384), (7345, 34688), (8160, 131072)]
)
def test_tp_rows_equal_replicated_selection(rows, context):
    args = _inputs(rows, context)
    expected = _select(args)
    store: dict = {}
    ranks = [_FakeRank(r, store) for r in range(4)]
    for rank in ranks:
        _select(args, rank)
    outs = [_select(args, rank) for rank in ranks]
    assert all(rank.calls == 2 for rank in ranks)
    for out in outs:
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [4095, 9217])
def test_tp_rows_fall_back_outside_the_admitted_chunk_sizes(rows):
    args = _inputs(rows, 40960)
    expected = _select(args)
    rank = _FakeRank(0, {})
    actual = _select(args, rank)
    assert rank.calls == 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _two_request_inputs(prefill_rows: int, context: int, other_rows: int, seed=1):
    """A prefill chunk next to a short second request (the paged scorer)."""
    torch.manual_seed(seed)
    page = 16
    pages = (context // 4) // page
    other_context = 8192
    other_pages = (other_context // 4) // page
    rows = prefill_rows + other_rows
    q = torch.randn(rows, 4, 128, device="cuda", dtype=torch.float16)
    k_cache = torch.randn(
        pages + other_pages + 3, page, 1, 128, device="cuda", dtype=torch.float16
    )
    order = torch.randperm(pages + other_pages + 3, device="cuda", dtype=torch.int32)
    page_table = torch.full((2, pages), 0, device="cuda", dtype=torch.int32)
    page_table[0] = order[:pages]
    page_table[1, :other_pages] = order[pages : pages + other_pages]
    token_to_req = torch.cat(
        (
            torch.zeros(prefill_rows, dtype=torch.int32, device="cuda"),
            torch.ones(other_rows, dtype=torch.int32, device="cuda"),
        )
    )
    query_positions = torch.cat(
        (
            torch.arange(context - prefill_rows, context, device="cuda"),
            torch.arange(other_context - other_rows, other_context, device="cuda"),
        )
    ).to(torch.int32)
    sequence_lengths = torch.tensor(
        [context, other_context], device="cuda", dtype=torch.int32
    )
    return (q, k_cache, page_table, token_to_req, query_positions, sequence_lengths)


@pytest.mark.parametrize("prefill_rows,other_rows", [(8160, 32), (6528, 10)])
def test_tp_rows_partition_the_paged_scorer_of_mixed_steps(prefill_rows, other_rows):
    args = _two_request_inputs(prefill_rows, 40960, other_rows)
    expected = _select(args)
    store: dict = {}
    ranks = [_FakeRank(r, store) for r in range(4)]
    for rank in ranks:
        _select(args, rank)
    outs = [_select(args, rank) for rank in ranks]
    assert all(rank.calls == 2 for rank in ranks)
    for out in outs:
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
