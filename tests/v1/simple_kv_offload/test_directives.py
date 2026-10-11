# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request directives of the CPU tier (kv_transfer_params["simple_cpu"],
options.allow_request_directives): strictly validated, experiments only."""

import pytest

from vllm.v1.simple_kv_offload.manager import (
    CONTROLS,
    Directives,
    parse_directives,
)
from vllm.v1.simple_kv_offload.options import parse_options

pytestmark = pytest.mark.cpu_test


def test_no_directives():
    assert parse_directives(None) is None
    assert parse_directives({}) is None
    assert parse_directives({"other_connector": {"x": 1}}) is None


def test_directives_are_parsed():
    got = parse_directives(
        {
            "simple_cpu": {
                "corrupt": "nan_v_tail=4",
                "gate": True,
                "control": {"qsa_grouped_page4": 0, "compute_nans_in_logits": "1"},
            }
        }
    )
    assert got == Directives(
        corrupt="nan_v_tail=4",
        gate=True,
        controls=(("qsa_grouped_page4", "0"), ("compute_nans_in_logits", "1")),
    )
    assert parse_directives({"simple_cpu": {"gate": False}}) == Directives()


@pytest.mark.parametrize(
    "entry,match",
    [
        ("gate", "must be an object"),
        ({"melt": 1}, "unknown simple_cpu directives"),
        ({"corrupt": "melt=1"}, "unknown corrupt mode"),
        ({"corrupt": "nan_v=0"}, "count"),
        ({"gate": "yes"}, "gate must be a boolean"),
        ({"control": {"reboot": "1"}}, "not allowed"),
        ({"control": {"qsa_grouped_page4": "maybe"}}, "not allowed"),
    ],
)
def test_directives_are_rejected(entry, match):
    with pytest.raises(ValueError, match=match):
        parse_directives({"simple_cpu": entry})


def test_controls_have_two_states():
    assert set(CONTROLS) == {
        "qsa_grouped_page4",
        "qsa_grouped_pad_fix",
        "compute_nans_in_logits",
    }
    assert all(values == ("0", "1") for values in CONTROLS.values())


def test_options():
    options = parse_options({})
    assert options.allow_request_directives is False
    assert options.fingerprint_gate_timeout_s == 5.0
    options = parse_options(
        {"allow_request_directives": True, "fingerprint_gate_timeout_s": 2}
    )
    assert (
        options.allow_request_directives and options.fingerprint_gate_timeout_s == 2.0
    )
    with pytest.raises(ValueError):
        parse_options({"fingerprint_gate_timeout_s": -1})
    with pytest.raises(ValueError):
        parse_options({"allow_request_directives": "yes"})
