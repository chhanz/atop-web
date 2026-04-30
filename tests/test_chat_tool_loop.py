"""End to end test for the Phase 20 tool use loop in ``stream_chat``.

The test uses a scripted provider that mimics the Bedrock tool use
streaming contract (TextDelta, ToolUseRequest, Stop). We feed two
turns: the first returns a tool call; the router must execute the
matching local handler and feed the result back in the second turn.
The second turn returns a ``Stop`` and a short text answer, and the
router must close the chat with a ``done`` event.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atop_web.llm import chat
from atop_web.llm.provider import LLMProvider
from atop_web.llm.tools import Stop, TextDelta, ToolUseRequest
from atop_web.parser.reader import parse_file


@pytest.fixture(scope="module")
def rawlog(rawlog_path: Path):
    return parse_file(rawlog_path, max_samples=30)


class _ScriptedToolProvider(LLMProvider):
    """Provider that enacts a scripted sequence of turns.

    Each element in ``script`` is a list of ``ProviderEvent`` instances
    representing one call to ``chat_with_tools``. The provider records
    every ``messages`` argument it receives so the test can assert the
    tool result was appended correctly.
    """

    name = "scripted"
    model = "test"

    def __init__(self, script):
        self._script = list(script)
        self.received_messages = []

    def health(self):
        return {"ok": True, "provider": self.name, "model": self.model, "detail": ""}

    def complete_json(self, system, user, schema):
        raise NotImplementedError

    def stream(self, system, user, history=None):
        raise NotImplementedError

    def supports_tools(self) -> bool:
        return True

    def chat_with_tools(self, system, messages, tools, *, temperature=0.2, max_tokens=1500):
        # Record a shallow copy of the transcript so the test can verify
        # what the router fed in on each turn.
        self.received_messages.append(list(messages))
        if not self._script:
            yield Stop(reason="end_turn")
            return
        events = self._script.pop(0)
        for ev in events:
            yield ev


def test_tool_loop_executes_handler_and_resumes(rawlog):
    script = [
        # Turn 1: model asks to call get_capture_info.
        [
            ToolUseRequest(
                call_id="c1",
                name="get_capture_info",
                arguments={},
            ),
            Stop(reason="tool_use"),
        ],
        # Turn 2: model emits a short answer and stops.
        [
            TextDelta(text="Capture covers "),
            TextDelta(text="a short window."),
            Stop(reason="end_turn"),
        ],
    ]
    provider = _ScriptedToolProvider(script)
    req = chat.ChatRequest(message="how long is this capture?")
    events = list(chat.stream_chat(provider, rawlog, req))

    types = [e.type for e in events]
    # Router must emit exactly one tool_call + one tool_result for the
    # single tool invocation, then the text tokens, then done.
    assert types.count("tool_call") == 1
    assert types.count("tool_result") == 1
    assert "token" in types
    assert types[-1] == "done"

    # The tool_result payload came from the local handler, not the
    # scripted model, so we can assert on its shape.
    tool_result_ev = next(e for e in events if e.type == "tool_result")
    payload = tool_result_ev.payload["content"]
    assert payload["sample_count"] == len(rawlog.samples)
    assert payload["start"].endswith("Z")

    # The router must have appended a ``role: tool`` message before
    # turn 2 so the model sees the result.
    assert len(provider.received_messages) == 2
    turn2 = provider.received_messages[1]
    tool_msgs = [m for m in turn2 if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_result"].name == "get_capture_info"

    # Final assistant text assembled from the two TextDelta frames.
    tokens = "".join(e.payload["text"] for e in events if e.type == "token")
    assert tokens == "Capture covers a short window."


def test_tool_loop_caps_at_budget(rawlog, monkeypatch):
    # Force a tight budget so the cap trips in a test friendly way.
    monkeypatch.setattr(chat, "MAX_TOOL_CALLS_PER_TURN", 2)

    # Three consecutive turns that each emit a tool call - the router
    # must stop after the budget is exceeded and emit an error event.
    def make_turn(call_id):
        return [
            ToolUseRequest(
                call_id=call_id,
                name="get_capture_info",
                arguments={},
            ),
            Stop(reason="tool_use"),
        ]

    provider = _ScriptedToolProvider(
        [make_turn("c1"), make_turn("c2"), make_turn("c3")]
    )
    req = chat.ChatRequest(message="loop forever")
    events = list(chat.stream_chat(provider, rawlog, req))
    types = [e.type for e in events]
    assert "error" in types
    error_event = next(e for e in events if e.type == "error")
    assert "budget" in error_event.payload["message"]


def test_tool_loop_returns_handler_error_as_tool_result(rawlog):
    # Ask for an unknown tool; the router must synthesize a tool_result
    # with ``is_error`` instead of crashing.
    script = [
        [
            ToolUseRequest(
                call_id="x",
                name="not_a_real_tool",
                arguments={},
            ),
            Stop(reason="tool_use"),
        ],
        [TextDelta(text="ok"), Stop(reason="end_turn")],
    ]
    provider = _ScriptedToolProvider(script)
    req = chat.ChatRequest(message="do the thing")
    events = list(chat.stream_chat(provider, rawlog, req))
    tool_results = [e for e in events if e.type == "tool_result"]
    assert len(tool_results) == 1
    assert tool_results[0].payload["is_error"] is True
    assert "unknown tool" in tool_results[0].payload["content"]["error"]
