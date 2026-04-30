"""Chat orchestration: wrap the streaming provider with context + parsing.

``stream_chat`` is the single entry point used by the SSE route. It picks a
context builder (``all`` vs ``range``), serializes the payload, fires the
provider stream, and yields structured events the route serializes to SSE:

* ``{"type": "token", "text": "..."}`` - raw model output chunks.
* ``{"type": "range_hint", "start": "...", "end": "...", "reason": "..."}``
  - parsed from inline ``<range .../>`` tags as they arrive.
* ``{"type": "done", "total_chars": N, "truncated": bool, "context_chars": N}``
  - final meta, emitted once after the model stops.
* ``{"type": "error", "message": "..."}`` - provider failure.

The range hint parser keeps a rolling buffer so tags split across chunks
still resolve. It strips the tag from the token stream before forwarding
text so the UI does not display raw XML to the user.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Iterator

from atop_web.llm import context, prompts, tools as tools_module
from atop_web.llm.provider import LLMProvider, LLMProviderError
from atop_web.llm.tools import (
    Stop,
    TextDelta,
    ToolCall,
    ToolResult,
    ToolUseRequest,
    build_tool_specs,
)
from atop_web.parser.reader import RawLog

logger = logging.getLogger(__name__)

DEFAULT_RANGE_LABEL = "range"

# Upper bound on tool calls per chat turn so a misbehaving model cannot
# loop forever. Set in Phase 20 at 8 because worst case the model makes
# one call per handler; raising this is fine later.
MAX_TOOL_CALLS_PER_TURN = 8


MAX_HISTORY = 20  # hard cap; older turns are dropped so we stay under budget.

# Matches ``<range start="..." end="..." reason="..."/>`` with flexible
# attribute order. The regex is greedy-safe because each attribute is
# bounded by double quotes.
_RANGE_TAG_RE = re.compile(
    r"<range\s+(?P<attrs>[^>]*?)/>",
    re.IGNORECASE,
)
_ATTR_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')
# A minimal "could still grow into a range tag" heuristic so we hold back
# partial prefixes until we know whether they close.
_POSSIBLE_PREFIX_RE = re.compile(r"<(r(a(n(g(e)?)?)?)?)?$", re.IGNORECASE)


@dataclass
class ChatRequest:
    message: str
    time_range_start: int | None = None
    time_range_end: int | None = None
    history: list[dict] = field(default_factory=list)


@dataclass
class ChatEvent:
    type: str
    payload: dict


def parse_iso_epoch(text: str | None) -> int | None:
    """Accept ``None``, empty string, or ISO8601 and return epoch seconds."""
    if not text:
        return None
    clean = text.strip()
    if not clean:
        return None
    if clean.endswith("Z"):
        clean = clean[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(clean)
    except ValueError as exc:
        raise ValueError(f"invalid ISO8601 timestamp: {text!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _format_iso(epoch: int | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def extract_range_hints(buffer: str) -> tuple[list[dict], str, str]:
    """Pull complete ``<range/>`` tags out of ``buffer``.

    Returns ``(hints, safe_text, remaining_buffer)``:

    * ``hints`` - parsed tag dicts (``start``, ``end``, ``reason``) in
      document order.
    * ``safe_text`` - the prefix of ``buffer`` that is guaranteed not to
      contain any partial tag; the caller forwards this as token output.
    * ``remaining_buffer`` - text held back because it might still grow
      into a ``<range/>`` tag. The caller passes it into the next call.

    The parser is intentionally forgiving about attribute ordering so it
    matches real model output; missing attributes default to empty
    strings but the tag is still emitted so the UI can show the reason
    even without one bound.
    """
    hints: list[dict] = []
    idx = 0
    safe = []
    while True:
        match = _RANGE_TAG_RE.search(buffer, idx)
        if not match:
            break
        safe.append(buffer[idx : match.start()])
        attrs = dict(_ATTR_RE.findall(match.group("attrs")))
        # ``reason`` is the legacy attribute name; ``label`` is the new
        # preferred one. Accept either so the prompt can stabilize on
        # ``label`` without breaking existing replies already in flight.
        reason = (attrs.get("label") or attrs.get("reason") or "").strip()
        hints.append(
            {
                "start": attrs.get("start", "").strip(),
                "end": attrs.get("end", "").strip(),
                "reason": reason or DEFAULT_RANGE_LABEL,
            }
        )
        idx = match.end()

    tail = buffer[idx:]
    # Hold back any trailing fragment that might still close into a tag.
    hold_at = _find_hold_at(tail)
    if hold_at is None:
        safe.append(tail)
        remaining = ""
    else:
        safe.append(tail[:hold_at])
        remaining = tail[hold_at:]
    return hints, "".join(safe), remaining


def _find_hold_at(tail: str) -> int | None:
    """Return the index at which we must start holding ``tail`` back.

    If we see a ``<`` that might begin a ``<range`` tag but has not closed
    yet, hold everything from that ``<`` onward. Otherwise return ``None``
    so the caller can forward ``tail`` in full.
    """
    last_lt = tail.rfind("<")
    if last_lt < 0:
        return None
    suffix = tail[last_lt:]
    if ">" in suffix:
        # Closed tag that the main regex did not match (eg. ``<br>``):
        # leave it in the safe text so we don't silently swallow content.
        return None
    if suffix == "<" or _POSSIBLE_PREFIX_RE.match(suffix) or suffix.lower().startswith("<range"):
        return last_lt
    return None


def _build_user_message(ctx_json: str, user_message: str) -> str:
    return (
        "Context (JSON):\n"
        + ctx_json
        + "\n\nUser question: "
        + user_message.strip()
    )


def _hint_to_epochs(hint: dict) -> tuple[int | None, int | None]:
    """Parse ``hint.start`` / ``hint.end`` into epoch seconds.

    Returns ``(None, None)`` when either side is unparseable; the caller
    drops the hint in that case. Accepts ISO8601 strings with a ``Z``
    suffix or epoch seconds as a fallback (robust against older
    responses).
    """
    try:
        s = parse_iso_epoch(hint.get("start")) if hint.get("start") else None
    except ValueError:
        s = None
    try:
        e = parse_iso_epoch(hint.get("end")) if hint.get("end") else None
    except ValueError:
        e = None
    return s, e


def _validate_and_widen_hint(
    hint: dict,
    capture_start: int | None,
    capture_end: int | None,
    interval_seconds: int | None,
) -> dict | None:
    """Return a sanitized hint, or ``None`` to drop the hallucinated value.

    The model can still hallucinate dates even after we serialize all
    context timestamps to ISO8601; this is the last line of defense:

    * Drop if either endpoint is unparseable.
    * Drop if start >= end.
    * Drop if either endpoint falls outside the capture window with a
      small interval-sized tolerance so a model that names the very
      first or last sample does not get falsely rejected.
    * Widen to ``interval_seconds * 2`` when the window is narrower, so
      the UI's filtered query returns samples. Phase 18.5 did this on
      the client; doing it here too keeps the SSE payload honest.
    """
    start_epoch, end_epoch = _hint_to_epochs(hint)
    if start_epoch is None or end_epoch is None:
        logger.warning(
            "range_hint dropped: unparseable timestamps (start=%r, end=%r)",
            hint.get("start"), hint.get("end"),
        )
        return None
    if start_epoch >= end_epoch:
        logger.warning(
            "range_hint dropped: non positive width (start=%s, end=%s)",
            hint.get("start"), hint.get("end"),
        )
        return None
    if capture_start is not None and capture_end is not None:
        tolerance = max(interval_seconds or 0, 1)
        lo = capture_start - tolerance
        hi = capture_end + tolerance
        if start_epoch < lo or end_epoch > hi:
            logger.warning(
                "range_hint dropped: outside capture window "
                "(hint=%s..%s, capture=%s..%s)",
                hint.get("start"), hint.get("end"),
                _epoch_to_iso(capture_start), _epoch_to_iso(capture_end),
            )
            return None
    # Auto widen narrow windows so the filtered query always has samples.
    min_width = max((interval_seconds or 0) * 2, 60)
    width = end_epoch - start_epoch
    widened = False
    if width < min_width:
        center = (start_epoch + end_epoch) // 2
        half = min_width // 2
        start_epoch = center - half
        end_epoch = center + half
        if capture_start is not None and start_epoch < capture_start:
            end_epoch += capture_start - start_epoch
            start_epoch = capture_start
        if capture_end is not None and end_epoch > capture_end:
            start_epoch -= end_epoch - capture_end
            end_epoch = capture_end
            if capture_start is not None and start_epoch < capture_start:
                start_epoch = capture_start
        widened = True
    out = {
        "start": _epoch_to_iso(start_epoch),
        "end": _epoch_to_iso(end_epoch),
        "reason": hint.get("reason") or DEFAULT_RANGE_LABEL,
    }
    if widened:
        out["widened"] = True
    return out


def _epoch_to_iso(epoch: int) -> str:
    return datetime.fromtimestamp(int(epoch), tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def stream_chat(
    provider: LLMProvider,
    rawlog: RawLog,
    request: ChatRequest,
) -> Iterator[ChatEvent]:
    """Yield :class:`ChatEvent` objects for the SSE route to serialize.

    The generator catches ``LLMProviderError`` and yields an ``error``
    event so the route always terminates cleanly. Any other exception
    propagates so the route can log it as a 500.
    """
    # Phase 20: prefer the tool use loop when the provider advertises
    # support. We keep the pre compute path below as a fallback for the
    # ``none`` provider (which errors out anyway) and future skeletons.
    if provider.supports_tools():
        yield from _stream_chat_with_tools(provider, rawlog, request)
        return

    if request.time_range_start is not None or request.time_range_end is not None:
        payload = context.build_range_context(
            rawlog, request.time_range_start, request.time_range_end
        )
    else:
        payload = context.build_all_context(rawlog)
    ctx_json, truncated = context.serialize_context(payload)

    # Capture window used for server side hint validation. We pull from
    # the raw samples directly so the guard works even if the context
    # builder later trims the capture fields for token budget reasons.
    capture_start = (
        rawlog.samples[0].curtime if rawlog.samples else None
    )
    capture_end = (
        rawlog.samples[-1].curtime if rawlog.samples else None
    )
    interval_seconds = context._median_interval_seconds(rawlog.samples)

    history = [
        h for h in request.history[-MAX_HISTORY:]
        if isinstance(h, dict) and h.get("role") in ("user", "assistant")
    ]
    user_text = _build_user_message(ctx_json, request.message)

    def emit_hint(raw_hint: dict):
        sanitized = _validate_and_widen_hint(
            raw_hint, capture_start, capture_end, interval_seconds
        )
        if sanitized is None:
            return None
        return ChatEvent("range_hint", sanitized)

    buffer = ""
    total_chars = 0
    try:
        for chunk in provider.stream(prompts.SYSTEM_CHAT, user_text, history):
            if not chunk:
                continue
            buffer += chunk
            hints, safe_text, buffer = extract_range_hints(buffer)
            for hint in hints:
                ev = emit_hint(hint)
                if ev is not None:
                    yield ev
            if safe_text:
                total_chars += len(safe_text)
                yield ChatEvent("token", {"text": safe_text})
    except LLMProviderError as exc:
        yield ChatEvent("error", {"message": str(exc)})
        return

    # Flush anything left in the buffer (including a trailing fragment that
    # never turned into a tag).
    if buffer:
        hints, safe_text, _ = extract_range_hints(buffer + " ")
        for hint in hints:
            ev = emit_hint(hint)
            if ev is not None:
                yield ev
        if safe_text.strip():
            total_chars += len(safe_text)
            yield ChatEvent("token", {"text": safe_text})

    yield ChatEvent(
        "done",
        {
            "total_chars": total_chars,
            "truncated": truncated,
            "context_chars": len(ctx_json),
            "mode": payload.get("mode"),
        },
    )


# ---------------------------------------------------------------------------
# Phase 20: tool use loop
# ---------------------------------------------------------------------------


def _stream_chat_with_tools(
    provider: LLMProvider,
    rawlog: RawLog,
    request: ChatRequest,
) -> Iterator[ChatEvent]:
    """Tool enabled chat loop.

    The user turn is primed with a compact context header (capture
    info, active range selection) so the model knows the bounds for
    tag validation without having to call ``get_capture_info`` first.
    Detailed metric questions go through tools on demand; the chat
    router relays text deltas to the SSE stream as ``token`` events and
    tool round trips as ``tool_call`` / ``tool_result`` events so the
    UI can show "thinking" state if desired (Phase 21+).
    """
    tools = build_tool_specs(rawlog)
    tool_by_name = {t.name: t for t in tools}

    capture_start = rawlog.samples[0].curtime if rawlog.samples else None
    capture_end = rawlog.samples[-1].curtime if rawlog.samples else None
    interval_seconds = context._median_interval_seconds(rawlog.samples)

    primer_lines = [
        "Capture metadata:",
        f"- hostname: {rawlog.header.nodename}",
        f"- atop version: {rawlog.header.aversion}",
        f"- samples: {len(rawlog.samples)}",
        f"- start: {_epoch_to_iso(capture_start) if capture_start else 'n/a'}",
        f"- end: {_epoch_to_iso(capture_end) if capture_end else 'n/a'}",
        f"- interval_seconds: {interval_seconds}",
    ]
    if request.time_range_start is not None or request.time_range_end is not None:
        primer_lines.append(
            "- active_selection_start: "
            + (
                _epoch_to_iso(request.time_range_start)
                if request.time_range_start is not None
                else "n/a"
            )
        )
        primer_lines.append(
            "- active_selection_end: "
            + (
                _epoch_to_iso(request.time_range_end)
                if request.time_range_end is not None
                else "n/a"
            )
        )
    primer_lines.append("")
    primer_lines.append("User question:")
    primer_lines.append(request.message.strip())
    user_prompt = "\n".join(primer_lines)

    messages: list[dict] = []
    for entry in request.history[-MAX_HISTORY:]:
        if not isinstance(entry, dict):
            continue
        if entry.get("role") in ("user", "assistant") and isinstance(
            entry.get("content"), str
        ):
            messages.append({"role": entry["role"], "content": entry["content"]})
    messages.append({"role": "user", "content": user_prompt})

    def emit_hint(raw_hint: dict):
        sanitized = _validate_and_widen_hint(
            raw_hint, capture_start, capture_end, interval_seconds
        )
        if sanitized is None:
            return None
        return ChatEvent("range_hint", sanitized)

    buffer = ""
    total_chars = 0
    tool_calls_made = 0

    for turn in range(MAX_TOOL_CALLS_PER_TURN + 1):
        try:
            event_iter = provider.chat_with_tools(
                prompts.SYSTEM_CHAT_TOOLS,
                messages,
                tools,
            )
        except LLMProviderError as exc:
            yield ChatEvent("error", {"message": str(exc)})
            return

        pending_calls: list[ToolCall] = []
        assistant_text = ""
        stop_reason = "end_turn"
        try:
            for ev in event_iter:
                if isinstance(ev, TextDelta):
                    if not ev.text:
                        continue
                    assistant_text += ev.text
                    buffer += ev.text
                    hints, safe_text, buffer = extract_range_hints(buffer)
                    for hint in hints:
                        mapped = emit_hint(hint)
                        if mapped is not None:
                            yield mapped
                    if safe_text:
                        total_chars += len(safe_text)
                        yield ChatEvent("token", {"text": safe_text})
                elif isinstance(ev, ToolUseRequest):
                    pending_calls.append(
                        ToolCall(
                            call_id=ev.call_id,
                            name=ev.name,
                            arguments=ev.arguments or {},
                        )
                    )
                elif isinstance(ev, Stop):
                    stop_reason = ev.reason
                    break
        except LLMProviderError as exc:
            yield ChatEvent("error", {"message": str(exc)})
            return

        # Record the assistant turn (text + any tool calls) so the next
        # model turn sees what it asked for.
        assistant_msg: dict = {"role": "assistant"}
        if assistant_text:
            assistant_msg["content"] = assistant_text
        if pending_calls:
            assistant_msg["tool_calls"] = pending_calls
        if pending_calls or assistant_text:
            messages.append(assistant_msg)

        if not pending_calls:
            # Flush any trailing buffer fragment (non tag text that was
            # held back). Identical pattern to the legacy path.
            if buffer:
                hints, safe_text, _ = extract_range_hints(buffer + " ")
                for hint in hints:
                    mapped = emit_hint(hint)
                    if mapped is not None:
                        yield mapped
                if safe_text.strip():
                    total_chars += len(safe_text)
                    yield ChatEvent("token", {"text": safe_text})
                buffer = ""
            yield ChatEvent(
                "done",
                {
                    "total_chars": total_chars,
                    "tool_calls": tool_calls_made,
                    "stop_reason": stop_reason,
                },
            )
            return

        # Execute each tool call locally and feed the result back in.
        if tool_calls_made + len(pending_calls) > MAX_TOOL_CALLS_PER_TURN:
            logger.warning(
                "tool call budget exceeded after %d calls; ending turn",
                tool_calls_made,
            )
            yield ChatEvent(
                "error",
                {
                    "message": (
                        "model exceeded tool call budget "
                        f"({MAX_TOOL_CALLS_PER_TURN} per turn)"
                    )
                },
            )
            return

        for call in pending_calls:
            tool_calls_made += 1
            yield ChatEvent(
                "tool_call",
                {
                    "id": call.call_id,
                    "name": call.name,
                    "arguments": call.arguments,
                },
            )
            spec = tool_by_name.get(call.name)
            if spec is None:
                result = ToolResult(
                    call_id=call.call_id,
                    name=call.name,
                    content={"error": f"unknown tool: {call.name!r}"},
                    is_error=True,
                )
            else:
                try:
                    payload = spec.call(call.arguments)
                    result = ToolResult(
                        call_id=call.call_id,
                        name=call.name,
                        content=payload,
                    )
                except Exception as exc:
                    logger.exception("tool %s raised", call.name)
                    result = ToolResult(
                        call_id=call.call_id,
                        name=call.name,
                        content={"error": f"{type(exc).__name__}: {exc}"},
                        is_error=True,
                    )
            yield ChatEvent(
                "tool_result",
                {
                    "id": result.call_id,
                    "name": result.name,
                    "is_error": result.is_error,
                    "content": result.content,
                },
            )
            messages.append({"role": "tool", "tool_result": result})

        # Loop back: the model gets a new turn with the tool results
        # appended to the transcript.
        continue

    # Loop exit without return should not happen because the budget
    # check emits an error first; if it somehow does, terminate.
    yield ChatEvent(
        "error",
        {"message": "tool loop exited without a terminal stop"},
    )
