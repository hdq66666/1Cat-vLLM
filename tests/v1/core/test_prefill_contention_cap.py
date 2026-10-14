# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_SM70_PREFILL_CONTENTION_TOKENS: shorter prefill steps while another
request is decoding."""

from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import ModelRunnerOutput

from .utils import create_requests, create_scheduler


def _req(computed: int, prompt: int, total: int | None = None):
    return SimpleNamespace(
        num_computed_tokens=computed,
        num_prompt_tokens=prompt,
        num_tokens=prompt if total is None else total,
    )


def _cap(cap, request, computed, new, running):
    fake = SimpleNamespace(prefill_contention_tokens=cap, running=running)
    return Scheduler._contention_prefill_cap(fake, request, computed, new)


def test_cap_applies_only_while_another_request_decodes():
    prefill = _req(0, 30000)
    decoding = _req(500, 400, 501)
    other_prefill = _req(100, 9000)
    assert _cap(2048, prefill, 0, 8192, [prefill, decoding]) == 2048
    # Nobody else, or only other prefills: the whole budget.
    assert _cap(2048, prefill, 0, 8192, [prefill]) == 8192
    assert _cap(2048, prefill, 0, 8192, [prefill, other_prefill]) == 8192
    # Off, or the chunk is already short enough.
    assert _cap(0, prefill, 0, 8192, [prefill, decoding]) == 8192
    assert _cap(2048, prefill, 0, 1500, [prefill, decoding]) == 1500


def test_cap_leaves_decode_and_spec_steps_alone():
    me = _req(400, 400, 401)
    decoding = _req(500, 400, 501)
    # A decode step with drafts can schedule several tokens; never cap it.
    assert _cap(2, me, 400, 5, [me, decoding]) == 5


def test_cap_counts_tokens_already_found_in_the_cache():
    # A waiting request whose prompt is fully covered by a cache hit.
    me = _req(0, 4000)
    decoding = _req(500, 400, 501)
    assert _cap(1024, me, 4000, 3000, [decoding]) == 3000
    assert _cap(1024, me, 1000, 3000, [decoding]) == 1024


@pytest.mark.parametrize("cap", [0, 512])
def test_scheduler_caps_prefill_next_to_a_decoding_request(cap):
    scheduler = create_scheduler(max_num_batched_tokens=8192, num_blocks=2000)
    scheduler.prefill_contention_tokens = cap
    short = create_requests(num_requests=1, num_tokens=32, req_ids=["short"])[0]
    scheduler.add_request(short)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens["short"] == 32
    scheduler.update_from_output(
        out,
        ModelRunnerOutput(
            req_ids=["short"],
            req_id_to_index={"short": 0},
            sampled_token_ids=[[0]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )

    # "short" now decodes. A long prompt arrives (waiting path) ...
    long = create_requests(num_requests=1, num_tokens=4000, req_ids=["long"])[0]
    scheduler.add_request(long)
    out = scheduler.schedule()
    assert out.num_scheduled_tokens["short"] == 1
    first = out.num_scheduled_tokens["long"]
    assert first == (512 if cap else 4000)
    if not cap:
        return
    scheduler.update_from_output(
        out,
        ModelRunnerOutput(
            req_ids=["short", "long"],
            req_id_to_index={"short": 0, "long": 1},
            sampled_token_ids=[[0], []],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    # ... and keeps short steps on the running path.
    out = scheduler.schedule()
    assert out.num_scheduled_tokens["long"] == 512
