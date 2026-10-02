# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Responses API behaviour that coding-agent clients (pi, dsh) rely on.

Those clients build every turn from stream events alone (they never read
response.completed.output), send reasoning.effort values the chat template
may not accept, and need a terminal event even when generation fails.

Tests marked ``needs_qwen38`` use the real Qwen3.8 tokenizer, chat template
and the qwen3 / qwen3_coder parsers. Point QWEN38_TOKENIZER_DIR at a directory
with tokenizer.json, tokenizer_config.json and chat_template.jinja.
"""

import asyncio
import functools
import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)

import vllm.envs as envs
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ErrorResponse,
    RequestResponseMetadata,
)
from vllm.entrypoints.openai.responses.context import SimpleContext
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
    validate_reasoning_effort_map,
)
from vllm.entrypoints.openai.responses.serving import (
    OpenAIServingResponses,
    _strip_leading_message_whitespace,
)
from vllm.entrypoints.openai.responses.streaming_events import (
    SimpleStreamingEventProcessor,
    _StateType,
)
from vllm.entrypoints.openai.responses.utils import (
    construct_chat_messages_with_tool_call,
)
from vllm.entrypoints.openai.sse_keep_alive import (
    SSE_KEEP_ALIVE_COMMENT,
    with_sse_keep_alive,
)
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams

QWEN38_DIR = os.environ.get("QWEN38_TOKENIZER_DIR", "")
needs_qwen38 = pytest.mark.skipif(
    not QWEN38_DIR, reason="QWEN38_TOKENIZER_DIR is not set"
)

WEATHER_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


@functools.cache
def _tokenizer():
    from vllm.tokenizers import get_tokenizer

    return get_tokenizer(QWEN38_DIR)


def _chat_template_kwargs(request: ResponsesRequest) -> dict:
    return request.build_chat_params(None, "auto").chat_template_kwargs


@pytest.fixture
def effort_map(monkeypatch):
    # The parsed map is cached per raw value, so no cache reset is needed.
    monkeypatch.setattr(
        envs,
        "VLLM_RESPONSES_REASONING_EFFORT_MAP",
        '{"high": "xhigh", "max": "xhigh", "minimal": "low"}',
    )


# --------------------------------------------------------------------------
# reasoning.effort -> chat template kwargs
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("effort", "expected"),
    [("high", "xhigh"), ("max", "xhigh"), ("minimal", "low"), ("medium", "medium")],
)
def test_reasoning_effort_is_mapped(effort_map, effort, expected):
    kwargs = _chat_template_kwargs(
        ResponsesRequest(input="hi", reasoning={"effort": effort})
    )
    assert kwargs["reasoning_effort"] == expected
    assert kwargs["enable_thinking"] is True


def test_reasoning_effort_unmapped_without_config(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_RESPONSES_REASONING_EFFORT_MAP", "")
    kwargs = _chat_template_kwargs(
        ResponsesRequest(input="hi", reasoning={"effort": "high"})
    )
    assert kwargs["reasoning_effort"] == "high"


def test_reasoning_effort_none_turns_thinking_off(effort_map):
    kwargs = _chat_template_kwargs(
        ResponsesRequest(input="hi", reasoning={"effort": "none"})
    )
    assert kwargs["enable_thinking"] is False
    assert "reasoning_effort" not in kwargs


def test_explicit_enable_thinking_wins_over_effort(effort_map):
    kwargs = _chat_template_kwargs(
        ResponsesRequest(
            input="hi",
            reasoning={"effort": "high"},
            chat_template_kwargs={"enable_thinking": False},
        )
    )
    assert kwargs["enable_thinking"] is False


def test_malformed_effort_map_is_rejected(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_RESPONSES_REASONING_EFFORT_MAP", "{high: xhigh}")
    with pytest.raises(ValueError, match="not valid JSON"):
        validate_reasoning_effort_map()
    monkeypatch.setattr(envs, "VLLM_RESPONSES_REASONING_EFFORT_MAP", '["high"]')
    with pytest.raises(ValueError, match="JSON object"):
        validate_reasoning_effort_map()


@needs_qwen38
@pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "high", "max"])
def test_every_openai_effort_renders_with_the_qwen38_template(effort_map, effort):
    """No OpenAI effort value may make the template raise."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(QWEN38_DIR)
    template = (Path(QWEN38_DIR) / "chat_template.jinja").read_text()
    kwargs = _chat_template_kwargs(
        ResponsesRequest(input="hi", reasoning={"effort": effort})
    )
    kwargs.pop("continue_final_message")
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        chat_template=template,
        tokenize=False,
        **kwargs,
    )
    if effort == "none":
        assert prompt.endswith("<think>\n\n</think>\n\n")
    else:
        assert prompt.endswith("<think>\n")


# --------------------------------------------------------------------------
# Streaming event processor
# --------------------------------------------------------------------------


def _tool_delta(index, call_id=None, name=None, arguments=None):
    return DeltaMessage(
        tool_calls=[
            DeltaToolCall(
                index=index,
                id=call_id,
                type="function" if call_id else None,
                function=DeltaFunctionCall(name=name, arguments=arguments),
            )
        ]
    )


def _run(processor, deltas):
    events = []
    for delta in deltas:
        delta = processor.strip_leading_content_whitespace(delta)
        if delta is None:
            continue
        target, tool_call = processor.resolve_target_state(delta)
        if target == _StateType.NONE:
            continue
        if processor.needs_transition(target, tool_call):
            events += processor.close_current()
            events += processor.open(target, tool_call)
        events += processor.emit_delta(delta, MagicMock(), None)
    return events


def test_message_text_starts_without_whitespace():
    processor = SimpleStreamingEventProcessor()
    events = _run(
        processor,
        [
            DeltaMessage(reasoning="think"),
            DeltaMessage(content="\n\n"),
            DeltaMessage(content="  The"),
            DeltaMessage(content="  answer"),
        ],
    )
    events += processor.close_current()
    text_deltas = [e.delta for e in events if e.type == "response.output_text.delta"]
    assert text_deltas == ["The", "  answer"]
    assert processor.output_items[-1].content[0].text == "The  answer"


def test_item_ids_call_ids_and_events_match_openai():
    processor = SimpleStreamingEventProcessor()
    events = _run(
        processor,
        [
            DeltaMessage(reasoning="plan"),
            DeltaMessage(content="Checking."),
            _tool_delta(0, "call_abc", "get_weather", ""),
            _tool_delta(0, arguments='{"city": "Paris"}'),
            _tool_delta(1, "call_def", "get_weather", '{"city": "Rome"}'),
        ],
    )
    events += processor.close_current()

    added = [e.item for e in events if e.type == "response.output_item.added"]
    done = [e.item for e in events if e.type == "response.output_item.done"]
    assert [i.id[:3] for i in added] == ["rs_", "msg", "fc_", "fc_"]
    assert [i.id for i in added] == [i.id for i in done]
    assert [i.call_id for i in added[2:]] == ["call_abc", "call_def"]
    assert [i.call_id for i in done[2:]] == ["call_abc", "call_def"]
    # The final response reuses exactly what was streamed.
    assert processor.output_items == done

    # Reasoning parts use the standard content_part events.
    types = [e.type for e in events]
    assert "response.reasoning_part.added" not in types
    part_added = [e for e in events if e.type == "response.content_part.added"]
    assert part_added[0].part.type == "reasoning_text"

    # Every function call gets arguments.done, also without argument deltas.
    args_done = [e for e in events if e.type == "response.function_call_arguments.done"]
    assert [e.arguments for e in args_done] == ['{"city": "Paris"}', '{"city": "Rome"}']


def test_complete_current_streams_the_missing_tail():
    processor = SimpleStreamingEventProcessor()
    _run(processor, [_tool_delta(0, "call_abc", "get_weather", "{")])
    tail = processor.complete_current('{"city": "Paris"}')
    assert [e.delta for e in tail] == ['"city": "Paris"}']
    done = processor.close_current()
    assert done[-1].item.arguments == '{"city": "Paris"}'


def test_complete_current_replaces_diverged_text():
    processor = SimpleStreamingEventProcessor()
    _run(processor, [_tool_delta(0, "call_abc", "get_weather", '{"x": 1')])
    assert processor.complete_current('{"city": "Paris"}') == []
    done = processor.close_current()
    assert done[0].arguments == '{"city": "Paris"}'
    assert done[-1].item.arguments == '{"city": "Paris"}'


def test_complete_current_without_reference_changes_nothing():
    processor = SimpleStreamingEventProcessor()
    _run(processor, [DeltaMessage(content="partial")])
    assert processor.complete_current(None) == []
    assert processor.close_current()[-1].item.content[0].text == "partial"


def _message(text):
    return ResponseOutputMessage(
        id="msg_1",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )


def _reasoning(text="thought"):
    return ResponseReasoningItem(
        id="rs_1",
        type="reasoning",
        summary=[],
        content=[{"type": "reasoning_text", "text": text}],
    )


def test_full_parse_drops_whitespace_after_reasoning_only():
    items = _strip_leading_message_whitespace([_reasoning(), _message("\n\nNo.")])
    assert items[1].content[0].text == "No."
    # Nothing but whitespace after the reasoning: the message is dropped.
    assert len(_strip_leading_message_whitespace([_reasoning(), _message("\n ")])) == 1
    # Without reasoning before it (e.g. a continuation), text is untouched.
    items = _strip_leading_message_whitespace([_message(" world")])
    assert items[0].content[0].text == " world"


def test_stream_keeps_whitespace_when_no_reasoning_came_first():
    processor = SimpleStreamingEventProcessor()
    _run(processor, [DeltaMessage(content=" world"), DeltaMessage(content="\n1.")])
    assert processor.close_current()[-1].item.content[0].text == " world\n1."


def _call(name, arguments):
    return ResponseFunctionToolCall(
        type="function_call",
        id="fc_x",
        call_id=f"call_{name}",
        name=name,
        arguments=arguments,
        status="completed",
    )


def test_tool_reconciliation_requires_matching_names():
    """A full parse that found an extra call before the open one must not
    hand the open call another call's arguments, nor send it twice."""
    processor = SimpleStreamingEventProcessor()
    _run(
        processor,
        [
            _tool_delta(0, "call_A", "A", '{"x": 1}'),
            _tool_delta(1, "call_C", "C", '{"x": 3}'),
        ],
    )
    full_parse = [
        _call("A", '{"x": 1}'),
        _call("B", '{"x": 2}'),
        _call("C", '{"x": 3}'),
    ]
    expected = OpenAIServingResponses._expected_open_item_text(processor, full_parse)
    assert expected is None
    processor.complete_current(expected)
    processor.close_current()
    assert [i.arguments for i in processor.output_items] == ['{"x": 1}', '{"x": 3}']
    assert OpenAIServingResponses._unstreamed_items(processor, full_parse) == []


def test_missing_trailing_calls_are_added_when_names_match():
    processor = SimpleStreamingEventProcessor()
    _run(processor, [_tool_delta(0, "call_A", "A", '{"x": 1}')])
    full_parse = [_call("A", '{"x": 1}'), _call("B", '{"x": 2}')]
    expected = OpenAIServingResponses._expected_open_item_text(processor, full_parse)
    assert expected == '{"x": 1}'
    processor.close_current()
    missing = OpenAIServingResponses._unstreamed_items(processor, full_parse)
    assert [c.name for c in missing] == ["B"]


def test_no_message_added_when_reasoning_split_differs():
    processor = SimpleStreamingEventProcessor()
    _run(processor, [_tool_delta(0, "call_A", "A", '{"x": 1}')])
    processor.close_current()
    full_parse = [_reasoning("head"), _message("tail")]
    assert OpenAIServingResponses._unstreamed_items(processor, full_parse) == []


# --------------------------------------------------------------------------
# Serving: real Qwen3.8 tokenizer and parsers
# --------------------------------------------------------------------------


def _serving():
    engine_client = MagicMock()
    model_config = MagicMock()
    model_config.max_model_len = 262144
    model_config.hf_config.model_type = "qwen4_exp"
    model_config.get_diff_sampling_param.return_value = {}
    engine_client.model_config = model_config
    return OpenAIServingResponses(
        engine_client=engine_client,
        models=MagicMock(),
        openai_serving_render=MagicMock(),
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
        reasoning_parser="qwen3",
        enable_auto_tools=True,
        tool_parser="qwen3_coder",
        default_chat_template_kwargs={"enable_thinking": False},
    )


async def _stream(request, chunks, prompt, finish_reason="stop"):
    """Drive the streaming path with token chunks, as the engine would."""
    tokenizer = _tokenizer()
    serving = _serving()
    context = SimpleContext()
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    async def generator():
        for i, text in enumerate(chunks):
            last = i == len(chunks) - 1
            completion = CompletionOutput(
                index=0,
                text=text,
                token_ids=tokenizer.encode(text, add_special_tokens=False),
                cumulative_logprob=0.0,
                logprobs=None,
                finish_reason=finish_reason if last else None,
                stop_reason=None,
            )
            context.append_output(
                RequestOutput(
                    request_id="req",
                    prompt=prompt,
                    prompt_token_ids=prompt_ids,
                    prompt_logprobs=None,
                    outputs=[completion],
                    finished=last,
                    num_cached_tokens=0,
                )
            )
            yield context

    return [
        event
        async for event in serving.responses_stream_generator(
            request,
            SamplingParams(max_tokens=1024),
            generator(),
            context,
            "Qwen3.8",
            tokenizer,
            RequestResponseMetadata(request_id="req"),
        )
    ]


THINKING_PROMPT = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n"
PLAIN_PROMPT = THINKING_PROMPT + "\n</think>\n\n"


@needs_qwen38
@pytest.mark.asyncio
async def test_thinking_stream_matches_final_response():
    request = ResponsesRequest(
        input="hi", stream=True, reasoning={"effort": "low"}, store=False
    )
    events = await _stream(
        request,
        ["Let me", " think.\n", "</think>\n\n", "The answer", " is 42."],
        THINKING_PROMPT,
    )
    done = [e.item for e in events if e.type == "response.output_item.done"]
    assert [i.type for i in done] == ["reasoning", "message"]
    assert done[0].content[0].text == "Let me think.\n"
    assert done[1].content[0].text == "The answer is 42."

    completed = events[-1]
    assert completed.type == "response.completed"
    final = completed.response.output
    assert [(i.type, i.id) for i in final] == [(i.type, i.id) for i in done]
    reasoning_tokens = sum(
        len(_tokenizer().encode(text, add_special_tokens=False))
        for text in ["Let me", " think.\n"]
    )
    usage = completed.response.usage
    assert usage.output_tokens_details.reasoning_tokens == reasoning_tokens


@needs_qwen38
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chunks",
    [
        # One token per delta.
        [
            "<tool_call>\n<function=get_weather>\n",
            "<parameter=city>\n",
            "Paris\n",
            "</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ],
        # Speculative decoding: the body and the end arrive together.
        [
            "<tool_call>\n<function=get_weather>\n",
            "<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
        ],
        # The whole call in the last delta.
        [
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n"
            "</parameter>\n</function>\n</tool_call>",
        ],
    ],
    ids=["token-by-token", "burst", "single-delta"],
)
async def test_streamed_tool_call_is_complete(chunks):
    request = ResponsesRequest(
        input="weather in Paris?", stream=True, tools=[WEATHER_TOOL], store=False
    )
    events = await _stream(request, chunks, PLAIN_PROMPT)

    added = [e.item for e in events if e.type == "response.output_item.added"]
    done = [e.item for e in events if e.type == "response.output_item.done"]
    assert [i.type for i in done] == ["function_call"]
    assert json.loads(done[0].arguments) == {"city": "Paris"}
    streamed_args = "".join(
        e.delta for e in events if e.type == "response.function_call_arguments.delta"
    )
    assert json.loads(streamed_args) == {"city": "Paris"}
    assert added[0].call_id == done[0].call_id

    completed = events[-1]
    assert completed.type == "response.completed"
    (final,) = completed.response.output
    assert (final.id, final.call_id, final.arguments) == (
        done[0].id,
        done[0].call_id,
        done[0].arguments,
    )


@needs_qwen38
@pytest.mark.asyncio
async def test_literal_end_marker_with_thinking_off_stays_content():
    """With thinking off, '</think>' in the answer is just text."""
    request = ResponsesRequest(input="hi", stream=True, store=False)
    events = await _stream(
        request, ["Qwen3 ends reasoning with ", "</think>", "; easy."], PLAIN_PROMPT
    )
    done = [e.item for e in events if e.type == "response.output_item.done"]
    assert [i.type for i in done] == ["message"]
    text = "Qwen3 ends reasoning with </think>; easy."
    assert done[0].content[0].text == text
    final = events[-1].response
    assert [i.type for i in final.output] == ["message"]
    assert final.usage.output_tokens_details.reasoning_tokens == 0


@needs_qwen38
@pytest.mark.asyncio
async def test_tool_argument_containing_end_marker_with_thinking_off():
    bash_tool = {
        "type": "function",
        "name": "bash",
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    }
    request = ResponsesRequest(
        input="find it", stream=True, tools=[bash_tool], store=False
    )
    events = await _stream(
        request,
        [
            "<tool_call>\n<function=bash>\n",
            "<parameter=command>\ngrep -rn '</think>' vllm\n</parameter>\n",
            "</function>\n</tool_call>",
        ],
        PLAIN_PROMPT,
    )
    done = [e.item for e in events if e.type == "response.output_item.done"]
    assert [i.type for i in done] == ["function_call"]
    assert json.loads(done[0].arguments) == {"command": "grep -rn '</think>' vllm"}
    assert [i.type for i in events[-1].response.output] == ["function_call"]


@needs_qwen38
@pytest.mark.asyncio
async def test_tool_choice_none_streams_markup_as_text_like_a_full_parse():
    request = ResponsesRequest(
        input="hi", stream=True, tools=[WEATHER_TOOL], tool_choice="none", store=False
    )
    chunks = [
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n",
        "</parameter>\n</function>\n</tool_call>",
    ]
    events = await _stream(request, chunks, PLAIN_PROMPT)
    done = [e.item for e in events if e.type == "response.output_item.done"]
    assert [i.type for i in done] == ["message"]
    assert done[0].content[0].text == "".join(chunks)


@needs_qwen38
@pytest.mark.asyncio
async def test_generation_error_ends_stream_with_error_event():
    request = ResponsesRequest(input="hi", stream=True, store=False)
    events = await _stream(
        request, ["partial", " answer"], PLAIN_PROMPT, finish_reason="error"
    )
    assert events[-1].type == "error"
    assert events[-1].message == "Internal server error"
    assert not any(e.type == "response.completed" for e in events)


@pytest.mark.asyncio
async def test_unexpected_failure_after_stream_ends_with_error_event(monkeypatch):
    serving = OpenAIServingResponses.__new__(OpenAIServingResponses)
    serving.use_harmony = False

    async def no_events(*args, **kwargs):
        if False:
            yield

    async def broken_final(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(serving, "_process_simple_streaming_events", no_events)
    monkeypatch.setattr(serving, "responses_full_generator", broken_final)
    events = [
        e
        async for e in serving.responses_stream_generator(
            ResponsesRequest(input="hi", stream=True),
            SamplingParams(max_tokens=8),
            no_events(),
            SimpleContext(),
            "m",
            MagicMock(),
            RequestResponseMetadata(request_id="req"),
        )
    ]
    assert [e.type for e in events] == [
        "response.created",
        "response.in_progress",
        "error",
    ]
    assert events[-1].message == "boom"


# --------------------------------------------------------------------------
# Request handling
# --------------------------------------------------------------------------


def test_hosted_tools_are_dropped():
    serving = _serving()
    request = ResponsesRequest(
        input="hi",
        tools=[WEATHER_TOOL, {"type": "web_search"}],
        tool_choice="required",
    )
    assert serving._drop_unsupported_tools(request) is None
    assert [t.type for t in request.tools] == ["function"]
    assert request.tool_choice == "required"


def test_only_hosted_tools_with_required_is_a_400():
    serving = _serving()
    request = ResponsesRequest(
        input="hi", tools=[{"type": "web_search"}], tool_choice="required"
    )
    error = serving._drop_unsupported_tools(request)
    assert isinstance(error, ErrorResponse)
    assert error.error.code == 400


def test_only_hosted_tools_with_auto_disables_tools():
    serving = _serving()
    request = ResponsesRequest(input="hi", tools=[{"type": "web_search"}])
    assert serving._drop_unsupported_tools(request) is None
    assert request.tools == []
    assert request.tool_choice == "none"


def test_unknown_input_item_is_a_value_error():
    with pytest.raises(ValueError, match="Unsupported input item type"):
        construct_chat_messages_with_tool_call([{"type": "item_reference", "id": "x"}])


# --------------------------------------------------------------------------
# SSE keep-alive
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_keep_alive_comment_while_idle():
    async def slow():
        await asyncio.sleep(0.35)
        yield "data: 1\n\n"

    chunks = [c async for c in with_sse_keep_alive(slow(), 0.1)]
    assert chunks[-1] == "data: 1\n\n"
    assert chunks[:-1] and set(chunks[:-1]) == {SSE_KEEP_ALIVE_COMMENT}


def test_keep_alive_disabled_returns_generator_unchanged():
    async def gen():
        yield "x"

    stream = gen()
    assert with_sse_keep_alive(stream, 0) is stream
