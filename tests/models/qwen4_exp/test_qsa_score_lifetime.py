# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU lifetime regressions adapted from upstream 996e3a161.

Use weak references to check that the next tile is allocated only after the
previous tile's temporary tensors lose their Python owners. These checks do
not execute CUDA kernels or establish CUDA graph/stream correctness.
"""

import weakref
from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize("exact", (False, True))
@pytest.mark.parametrize("partition_rows", (False, True), ids=("local", "tp4"))
def test_qsa_paged_selection_releases_logits_between_row_tiles(
    monkeypatch, exact, partition_rows
):
    """Upstream's row-tile lifetime check, including the local mixed TP4 path."""
    columns = 16
    rows = 4097 if partition_rows else 5
    rows_per_tile = 512 if partition_rows else 2
    query = torch.empty(rows, 4, 128)
    output = torch.empty(rows, 2051, dtype=torch.int32)
    events = []
    previous_scores: list[weakref.ReferenceType[torch.Tensor]] = []

    def assert_released():
        assert all(ref() is None for ref in previous_scores)

    def score(tile, *args, **kwargs):
        assert_released()
        events.append("score")
        logits = torch.ones(tile.shape[0], columns)
        visible = torch.full((tile.shape[0],), columns, dtype=torch.int32)
        previous_scores.extend((weakref.ref(logits), weakref.ref(visible)))
        return logits, visible

    def select(logits, lengths, indices, *args):
        events.append("select")
        assert logits.shape[0] == lengths.numel() == indices.shape[0]
        indices.zero_()

    def expand(*args):
        events.append("expand")
        args[-1].fill_(-1)

    def all_gather(blocks, dim):
        assert_released()
        assert dim == 0
        events.append("gather")
        return blocks.repeat(4, 1)

    tp_group = (
        SimpleNamespace(world_size=4, rank_in_group=0, all_gather=all_gather)
        if partition_rows
        else None
    )
    monkeypatch.setattr(qsa_ops, "_LOGITS_WORKSPACE_BYTES", rows_per_tile * columns * 4)
    monkeypatch.setattr(qsa_ops, "_SM70_QSA_MTP_TOPK", False)
    monkeypatch.setattr(qsa_ops, "_use_sm70_qsa_indexer_cublas", lambda *a: False)
    monkeypatch.setattr(qsa_ops, "_use_sm70_qsa_lexicographic_topk", lambda *a: exact)
    monkeypatch.setattr(qsa_ops, "_sm70_qsa_lexicographic_topk_op", lambda: select)
    monkeypatch.setattr(
        qsa_ops.current_platform, "has_device_capability", lambda *a: False
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(qsa_ops, "qsa_mqa_paged", score)
    monkeypatch.setattr(qsa_ops.torch.ops._C, "persistent_topk", select, raising=False)
    monkeypatch.setattr(qsa_ops, "expand_qsa_block_indices_cuda", expand)
    actual = qsa_ops.qsa_select_paged_tokens(
        query,
        torch.ones(1, columns, 1, 128),
        torch.zeros(1, 1, dtype=torch.int32),
        torch.zeros(rows, dtype=torch.int32),
        torch.arange(rows),
        torch.tensor([columns * 4], dtype=torch.int32),
        2048,
        4,
        output,
        tp_group=tp_group,
    )

    assert actual is output
    assert torch.all(output == -1)
    assert_released()
    assert len(previous_scores) == 6
    expected = (
        ["score", "select"] * 3 + ["gather", "expand"]
        if partition_rows
        else ["score", "select", "expand"] * 3
    )
    assert events == expected


def test_qsa_cublas_releases_scores_between_column_tiles(monkeypatch):
    """Upstream's cuBLAS tile check with the local constant-based interface."""
    monkeypatch.setattr(qsa_ops, "_SM70_INDEXER_SCORE_TILE_BYTES", 1024 * 1024)
    query = torch.randn(512, 4, 16)
    keys = torch.randn(600, 16)
    valid = torch.ones(600, dtype=torch.uint8)
    visible = torch.full((512,), 600, dtype=torch.int32)
    previous: list[weakref.ReferenceType[torch.Tensor]] = []
    original_mm = torch.mm

    def multiply(left, right, *, out_dtype):
        assert out_dtype == torch.float32
        assert all(ref() is None for ref in previous)
        result = original_mm(left, right)
        previous.append(weakref.ref(result))
        return result

    class Epilogue:
        def __getitem__(self, grid):
            def apply(scores, visibility, mask, logits, *args, **kwargs):
                width, column = args[3:5]
                heads = kwargs["NUM_HEADS"]
                logits[:, column : column + width].copy_(
                    scores.reshape(query.shape[0], heads, width).relu().sum(dim=1)
                )

            return apply

    expected = original_mm(query.reshape(-1, 16), keys.T)
    expected = expected.reshape(512, 4, 600).relu().sum(dim=1)
    monkeypatch.setattr(torch, "mm", multiply)
    monkeypatch.setattr(qsa_ops, "_qsa_relu_headsum_visible_kernel", Epilogue())
    actual, actual_visible = qsa_ops._qsa_mqa_cublas(query, keys, valid, visible)

    torch.testing.assert_close(actual, expected)
    assert actual_visible is visible
    assert len(previous) == 3
    assert all(ref() is None for ref in previous)
