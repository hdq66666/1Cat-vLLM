# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BlockPool free-step stamps and the freed-block quarantine (CPU only)."""

import pytest

from vllm.v1.core.block_pool import BlockPool

pytestmark = pytest.mark.cpu_test


def _pool(num_blocks: int = 8) -> BlockPool:
    return BlockPool(num_gpu_blocks=num_blocks, enable_caching=True, hash_block_size=16)


def test_free_blocks_stamps_the_current_step() -> None:
    pool = _pool()
    blocks = pool.get_new_blocks(3)
    assert all(b.freed_at_step == -1 for b in blocks)
    pool.current_step = 7
    pool.free_blocks(blocks)
    assert all(b.freed_at_step == 7 for b in blocks)


def test_quarantine_prefers_blocks_freed_long_ago() -> None:
    pool = _pool(8)
    old = pool.get_new_blocks(2)
    young = pool.get_new_blocks(2)
    rest = pool.get_new_blocks(3)  # everything else is allocated
    pool.current_step = 1
    pool.free_blocks(old)  # freed at step 1
    pool.current_step = 10
    pool.free_blocks(young)  # freed at step 10; queue head: young (uncached first)
    pool.quarantine_steps = 3
    pool.current_step = 11

    got = pool.get_new_blocks(2)
    assert {b.block_id for b in got} == {b.block_id for b in old}
    assert pool.quarantine_fallbacks == 0

    # Only young blocks are left: they are handed out anyway and counted.
    got2 = pool.get_new_blocks(2)
    assert {b.block_id for b in got2} == {b.block_id for b in young}
    assert pool.quarantine_fallbacks == 2
    assert pool.get_num_free_blocks() == 0
    pool.free_blocks(got + got2 + rest)


def test_quarantine_off_keeps_queue_order() -> None:
    pool = _pool(8)
    a = pool.get_new_blocks(3)
    pool.current_step = 5
    pool.free_blocks(a)
    # Uncached scratch is recycled first: the freed blocks come back first.
    got = pool.get_new_blocks(3)
    assert [b.block_id for b in got] == [b.block_id for b in a]
