"""Test the Bedrock provider's tool use stream parsing.

We do not call AWS; instead we monkey patch the Bedrock client factory
on a ``BedrockProvider`` instance and hand it a scripted list of
``converse_stream`` events. The test verifies that text deltas become
:class:`TextDelta` events, tool use deltas accumulate into a single
:class:`ToolUseRequest`, and ``messageStop`` closes the stream with a
:class:`Stop` event.
"""

from __future__ import annotations

import json

from atop_web.llm import tools as tools_mod
from atop_web.llm.provider import BedrockProvider


class _FakeClient:
    def __init__(self, scripted_events):
        self._events = scripted_events
        self.last_request = None

    def converse_stream(self, **kwargs):
        self.last_request = kwargs
        return {"stream": iter(self._events)}


def _make_provider(scripted_events):
    p = BedrockProvider(model="test-model", region="us-east-1")
    p._client = _FakeClient(scripted_events)
    return p


def test_chat_with_tools_parses_text_delta_and_tool_use_frames():
    scripted = [
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"text": "Hello "},
            }
        },
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"text": "world"},
            }
        },
        {
            "contentBlockStart": {
                "contentBlockIndex": 1,
                "start": {
                    "toolUse": {
                        "toolUseId": "call-1",
                        "name": "get_capture_info",
                    }
                },
            }
        },
        {
            "contentBlockDelta": {
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": '{"a":'}},
            }
        },
        {
            "contentBlockDelta": {
                "contentBlockIndex": 1,
                "delta": {"toolUse": {"input": "1}"}},
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {"messageStop": {"stopReason": "tool_use"}},
    ]
    provider = _make_provider(scripted)
    dummy_tool = tools_mod.ToolSpec(
        name="get_capture_info",
        description="Return capture metadata.",
        input_schema={"type": "object", "properties": {}},
        handler=lambda _args: {"ok": True},
    )
    events = list(
        provider.chat_with_tools(
            "system prompt",
            [{"role": "user", "content": "hi"}],
            [dummy_tool],
        )
    )
    # First two frames: text deltas.
    assert isinstance(events[0], tools_mod.TextDelta)
    assert events[0].text == "Hello "
    assert isinstance(events[1], tools_mod.TextDelta)
    # Single tool use request, input accumulated across the two deltas.
    tool_events = [e for e in events if isinstance(e, tools_mod.ToolUseRequest)]
    assert len(tool_events) == 1
    req = tool_events[0]
    assert req.call_id == "call-1"
    assert req.name == "get_capture_info"
    assert req.arguments == {"a": 1}
    # Terminal frame is a Stop with the tool_use reason.
    stop = events[-1]
    assert isinstance(stop, tools_mod.Stop)
    assert stop.reason == "tool_use"

    # Also confirm the outbound shape: the tool spec was lowered to the
    # ``toolSpec`` shape Bedrock expects.
    sent = provider._client.last_request
    assert sent["toolConfig"]["tools"][0]["toolSpec"]["name"] == "get_capture_info"
    assert (
        sent["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"]
        == {"type": "object", "properties": {}}
    )


def test_chat_with_tools_emits_stop_when_stream_ends_without_message_stop():
    # Simulating a malformed stream: ``messageStop`` is missing.
    scripted = [
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"text": "ok"},
            }
        },
    ]
    provider = _make_provider(scripted)
    events = list(
        provider.chat_with_tools(
            "system",
            [{"role": "user", "content": "hi"}],
            [],
        )
    )
    assert any(isinstance(e, tools_mod.Stop) for e in events)


def test_chat_with_tools_lowers_tool_result_message():
    # The provider must accept a ``role: tool`` message with a ToolResult
    # and forward it as a user turn containing a ``toolResult`` block.
    scripted = [{"messageStop": {"stopReason": "end_turn"}}]
    provider = _make_provider(scripted)
    dummy_tool = tools_mod.ToolSpec(
        name="dummy",
        description="",
        input_schema={"type": "object", "properties": {}},
        handler=lambda _a: {},
    )
    result = tools_mod.ToolResult(
        call_id="c1", name="dummy", content={"k": "v"}
    )
    list(
        provider.chat_with_tools(
            "sys",
            [
                {"role": "user", "content": "q"},
                {"role": "tool", "tool_result": result},
            ],
            [dummy_tool],
        )
    )
    sent = provider._client.last_request["messages"]
    assert sent[0]["role"] == "user"
    # The tool result lands as a ``user`` message with a toolResult block.
    assert sent[1]["role"] == "user"
    block = sent[1]["content"][0]["toolResult"]
    assert block["toolUseId"] == "c1"
    assert block["status"] == "success"
    assert block["content"][0]["json"] == {"k": "v"}


def test_chat_with_tools_merges_parallel_tool_results_into_single_user_msg():
    # Bedrock requires every toolUse in an assistant turn to be answered
    # by a toolResult block in the next user turn. When the model fires
    # two tools in parallel, the neutral transcript holds two separate
    # ``role: tool`` entries, and the provider must merge them into one
    # user message containing both toolResult blocks.
    scripted = [{"messageStop": {"stopReason": "end_turn"}}]
    provider = _make_provider(scripted)
    dummy_tool = tools_mod.ToolSpec(
        name="dummy",
        description="",
        input_schema={"type": "object", "properties": {}},
        handler=lambda _a: {},
    )
    call_a = tools_mod.ToolCall(call_id="a1", name="dummy", arguments={})
    call_b = tools_mod.ToolCall(call_id="b2", name="dummy", arguments={})
    result_a = tools_mod.ToolResult(
        call_id="a1", name="dummy", content={"which": "a"}
    )
    result_b = tools_mod.ToolResult(
        call_id="b2", name="dummy", content={"which": "b"}, is_error=True
    )
    list(
        provider.chat_with_tools(
            "sys",
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "tool_calls": [call_a, call_b]},
                {"role": "tool", "tool_result": result_a},
                {"role": "tool", "tool_result": result_b},
            ],
            [dummy_tool],
        )
    )
    sent = provider._client.last_request["messages"]
    # user, assistant(tool_calls), merged user(toolResults A+B)
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    merged = sent[2]["content"]
    assert len(merged) == 2
    assert merged[0]["toolResult"]["toolUseId"] == "a1"
    assert merged[0]["toolResult"]["status"] == "success"
    assert merged[0]["toolResult"]["content"][0]["json"] == {"which": "a"}
    assert merged[1]["toolResult"]["toolUseId"] == "b2"
    assert merged[1]["toolResult"]["status"] == "error"
    assert merged[1]["toolResult"]["content"][0]["json"] == {"which": "b"}


def test_chat_with_tools_merges_three_parallel_tool_results():
    scripted = [{"messageStop": {"stopReason": "end_turn"}}]
    provider = _make_provider(scripted)
    dummy_tool = tools_mod.ToolSpec(
        name="dummy",
        description="",
        input_schema={"type": "object", "properties": {}},
        handler=lambda _a: {},
    )
    calls = [
        tools_mod.ToolCall(call_id=f"c{i}", name="dummy", arguments={})
        for i in range(3)
    ]
    results = [
        tools_mod.ToolResult(call_id=f"c{i}", name="dummy", content={"i": i})
        for i in range(3)
    ]
    list(
        provider.chat_with_tools(
            "sys",
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "tool_calls": calls},
                {"role": "tool", "tool_result": results[0]},
                {"role": "tool", "tool_result": results[1]},
                {"role": "tool", "tool_result": results[2]},
            ],
            [dummy_tool],
        )
    )
    sent = provider._client.last_request["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    merged = sent[2]["content"]
    assert [b["toolResult"]["toolUseId"] for b in merged] == ["c0", "c1", "c2"]


def test_chat_with_tools_does_not_merge_across_non_tool_boundary():
    # Defensive: if a non-tool message sits between tool messages, the
    # merge window must end at that boundary.
    scripted = [{"messageStop": {"stopReason": "end_turn"}}]
    provider = _make_provider(scripted)
    dummy_tool = tools_mod.ToolSpec(
        name="dummy",
        description="",
        input_schema={"type": "object", "properties": {}},
        handler=lambda _a: {},
    )
    r1 = tools_mod.ToolResult(call_id="x", name="dummy", content={"k": 1})
    r2 = tools_mod.ToolResult(call_id="y", name="dummy", content={"k": 2})
    list(
        provider.chat_with_tools(
            "sys",
            [
                {"role": "user", "content": "q"},
                {"role": "tool", "tool_result": r1},
                {"role": "assistant", "content": "thinking"},
                {"role": "tool", "tool_result": r2},
            ],
            [dummy_tool],
        )
    )
    sent = provider._client.last_request["messages"]
    assert [m["role"] for m in sent] == ["user", "user", "assistant", "user"]
    assert sent[1]["content"][0]["toolResult"]["toolUseId"] == "x"
    assert sent[3]["content"][0]["toolResult"]["toolUseId"] == "y"


def test_supports_tools_is_true_on_bedrock_and_false_on_none():
    from atop_web.llm.provider import NoneProvider

    assert BedrockProvider().supports_tools() is True
    assert NoneProvider().supports_tools() is False
