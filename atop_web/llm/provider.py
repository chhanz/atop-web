"""LLM provider abstraction.

The provider boundary is the only place in the app that talks to a language
model. Each concrete provider implements two things:

* ``health()`` - cheap check returning ``{"ok": bool, "provider", "model",
  "detail"}``. The UI uses this to decide whether to render LLM specific
  cards.
* ``complete_json(system, user, schema)`` - run a JSON only completion and
  return the parsed object. On shape errors the provider retries up to two
  times with a follow up prompt asking for valid JSON.

The package ships:

* ``NoneProvider`` - default, always healthy, never calls anything.
* ``BedrockProvider`` - AWS Bedrock Converse API, calls Claude Sonnet by
  default.
* ``OllamaProvider`` / ``OpenAIProvider`` / ``AnthropicProvider`` /
  ``GeminiProvider`` - skeletons that raise ``NotImplementedError`` until a
  future phase fleshes them out.
"""

from __future__ import annotations

import abc
import json
import os
import threading
from collections.abc import Iterator
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from atop_web.llm.tools import ProviderEvent, ToolResult, ToolSpec

PROVIDER_NONE = "none"
PROVIDER_BEDROCK = "bedrock"
PROVIDER_OLLAMA = "ollama"
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_GEMINI = "gemini"


class LLMProviderError(RuntimeError):
    """Raised when a completion cannot be produced in the requested shape."""


class LLMProvider(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def health(self) -> dict:
        """Return ``{ok, provider, model, detail}``.

        ``ok=True`` means the provider is configured and reachable enough to
        plausibly answer. It does not guarantee a specific call will succeed.
        """

    @abc.abstractmethod
    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        """Produce a JSON object conforming to ``schema``.

        Implementations should enforce JSON only output at the provider
        level where possible (response_format, tool use, etc.) and fall back
        to two repair retries otherwise. On persistent failure raise
        ``LLMProviderError``.
        """

    @abc.abstractmethod
    def stream(
        self,
        system: str,
        user: str,
        history: list[dict] | None = None,
    ) -> Iterator[str]:
        """Yield text tokens as they arrive from the model.

        ``history`` is a list of ``{"role": "user|assistant", "content": str}``
        messages (oldest first) that precede the current ``user`` turn.
        Providers that cannot stream raise ``LLMProviderError``.
        """

    # ------------------------------------------------------------------
    # Tool use (Phase 20). Default implementations make adoption opt in
    # per provider so skeletons (Ollama / OpenAI / ...) do not have to
    # implement this until they are fleshed out.
    # ------------------------------------------------------------------

    def supports_tools(self) -> bool:
        """Whether the provider can run a tool use loop.

        Subclasses flip this to ``True`` and override
        :meth:`chat_with_tools`. Defaults to ``False`` so callers can
        safely fall back to the plain streaming path.
        """
        return False

    def chat_with_tools(
        self,
        system: str,
        messages: list[dict],
        tools: "list[ToolSpec]",
        *,
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> "Iterator[ProviderEvent]":
        """Run a single model turn with tool use support.

        ``messages`` is a provider neutral transcript: each entry is one
        of

        * ``{"role": "user", "content": str}``
        * ``{"role": "assistant", "content": str}``
        * ``{"role": "assistant", "tool_calls": [ToolCall, ...]}``
        * ``{"role": "tool", "tool_result": ToolResult}``

        The provider adapter lowers this to whatever shape its API
        expects. The generator yields :class:`TextDelta` / :class:`ToolUseRequest`
        and terminates with a single :class:`Stop`.
        """
        raise LLMProviderError(
            f"{self.name} provider does not support chat_with_tools yet"
        )


# ---------------------------------------------------------------------------
# No op provider (default)
# ---------------------------------------------------------------------------


class NoneProvider(LLMProvider):
    """Default stub. Reports ok but cannot actually complete anything.

    The UI inspects ``health().provider == "none"`` and hides LLM features
    accordingly, so it is fine for ``complete_json`` to raise here.
    """

    name = PROVIDER_NONE

    def health(self) -> dict:
        return {
            "ok": True,
            "provider": self.name,
            "model": None,
            "detail": "LLM disabled (LLM_PROVIDER=none)",
        }

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        raise LLMProviderError(
            "LLM is disabled. Set LLM_PROVIDER to bedrock, ollama, openai, "
            "anthropic, or gemini to enable completions."
        )

    def stream(
        self,
        system: str,
        user: str,
        history: list[dict] | None = None,
    ) -> Iterator[str]:
        raise LLMProviderError(
            "LLM is disabled. Set LLM_PROVIDER to bedrock, ollama, openai, "
            "anthropic, or gemini to enable streaming."
        )
        # Unreachable yield marker so type checkers see this as a generator.
        yield ""  # pragma: no cover


# ---------------------------------------------------------------------------
# AWS Bedrock (real)
# ---------------------------------------------------------------------------


def _coerce_json(text: str) -> dict:
    """Best effort JSON parse that strips common wrappers (``json\\n{...}``)."""
    text = text.strip()
    if text.startswith("```"):
        # Drop leading fence line (``` or ```json) and trailing ```.
        first_nl = text.find("\n")
        if first_nl >= 0:
            text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[: -3]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract the outermost JSON object.
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


class BedrockProvider(LLMProvider):
    """AWS Bedrock Converse API.

    Reads configuration from environment variables:

    * ``BEDROCK_MODEL`` (default ``global.anthropic.claude-sonnet-4-6``)
    * ``AWS_REGION`` / ``AWS_DEFAULT_REGION`` (via boto3)
    * credentials via the usual boto3 chain

    The client is created lazily and cached on the instance so health
    checks stay cheap.
    """

    name = PROVIDER_BEDROCK
    DEFAULT_MODEL = "global.anthropic.claude-sonnet-4-6"
    MAX_REPAIR_ATTEMPTS = 2

    def __init__(self, *, model: str | None = None, region: str | None = None) -> None:
        self.model = model or os.environ.get("BEDROCK_MODEL", self.DEFAULT_MODEL)
        self.region = region or os.environ.get("AWS_REGION") or os.environ.get(
            "AWS_DEFAULT_REGION"
        )
        self._client = None
        self._client_error: str | None = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self._client_error is not None:
            raise LLMProviderError(self._client_error)
        try:
            import boto3  # type: ignore
        except ImportError as exc:
            self._client_error = f"boto3 not installed: {exc}"
            raise LLMProviderError(self._client_error) from exc
        try:
            kwargs = {}
            if self.region:
                kwargs["region_name"] = self.region
            self._client = boto3.client("bedrock-runtime", **kwargs)
        except Exception as exc:
            self._client_error = f"bedrock client init failed: {exc}"
            raise LLMProviderError(self._client_error) from exc
        return self._client

    def health(self) -> dict:
        # The health probe stays local: we only confirm that the client can
        # be constructed. Actually calling the API would cost money on every
        # page load.
        try:
            self._get_client()
        except LLMProviderError as exc:
            return {
                "ok": False,
                "provider": self.name,
                "model": self.model,
                "detail": str(exc),
            }
        return {
            "ok": True,
            "provider": self.name,
            "model": self.model,
            "detail": f"region={self.region or 'default'}",
        }

    def _converse(self, system: str, user: str) -> str:
        client = self._get_client()
        try:
            resp = client.converse(
                modelId=self.model,
                system=[{"text": system}],
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": user}],
                    }
                ],
                inferenceConfig={"temperature": 0.2, "maxTokens": 1200},
            )
        except Exception as exc:
            raise LLMProviderError(f"bedrock converse failed: {exc}") from exc

        try:
            parts = resp["output"]["message"]["content"]
            text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        except (KeyError, TypeError) as exc:
            raise LLMProviderError(f"unexpected bedrock response shape: {exc}") from exc
        if not text:
            raise LLMProviderError("bedrock returned empty text")
        return text

    def stream(
        self,
        system: str,
        user: str,
        history: list[dict] | None = None,
    ) -> Iterator[str]:
        """Yield text chunks from Bedrock ``converse_stream``.

        ``history`` entries become prior ``user``/``assistant`` messages so
        the model sees the conversation in order. Any Bedrock error or
        unexpected frame raises ``LLMProviderError`` so the caller can
        surface it to the UI instead of silently truncating.
        """
        client = self._get_client()
        messages: list[dict] = []
        for entry in history or []:
            role = entry.get("role")
            content = entry.get("content")
            if role not in ("user", "assistant") or not isinstance(content, str):
                continue
            if not content.strip():
                continue
            messages.append({"role": role, "content": [{"text": content}]})
        messages.append({"role": "user", "content": [{"text": user}]})
        try:
            resp = client.converse_stream(
                modelId=self.model,
                system=[{"text": system}],
                messages=messages,
                inferenceConfig={"temperature": 0.2, "maxTokens": 1500},
            )
        except Exception as exc:
            raise LLMProviderError(
                f"bedrock converse_stream failed: {exc}"
            ) from exc
        stream = resp.get("stream")
        if stream is None:
            raise LLMProviderError(
                "bedrock converse_stream returned no stream handle"
            )
        try:
            for event in stream:
                delta = event.get("contentBlockDelta", {}).get("delta", {})
                text = delta.get("text")
                if text:
                    yield text
                if "messageStop" in event:
                    break
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(
                f"bedrock stream interrupted: {exc}"
            ) from exc

    def supports_tools(self) -> bool:
        # Bedrock Converse tool use is available on Claude 3.x / 4.x and
        # on the Sonnet / Haiku / Opus families we ship by default. We
        # keep the gate optimistic (True) and rely on the API to reject
        # calls against models that truly lack support; that way new
        # capable models start working without a code change.
        return True

    def chat_with_tools(
        self,
        system: str,
        messages: list[dict],
        tools,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ):
        # Deferred import: provider.py must not pull tools.py at module
        # load because tools.py depends on the parser, which pulls in
        # binary decoding code irrelevant to provider wiring.
        from atop_web.llm import tools as tools_module

        client = self._get_client()
        bedrock_messages = self._to_bedrock_messages(messages)
        tool_config = {
            "tools": [
                {
                    "toolSpec": {
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": {"json": t.input_schema},
                    }
                }
                for t in tools
            ]
        }
        try:
            resp = client.converse_stream(
                modelId=self.model,
                system=[{"text": system}],
                messages=bedrock_messages,
                inferenceConfig={
                    "temperature": temperature,
                    "maxTokens": max_tokens,
                },
                toolConfig=tool_config,
            )
        except Exception as exc:
            raise LLMProviderError(
                f"bedrock converse_stream (tools) failed: {exc}"
            ) from exc
        stream = resp.get("stream")
        if stream is None:
            raise LLMProviderError(
                "bedrock converse_stream (tools) returned no stream"
            )

        # Tool use arrives as one contentBlockStart (type=toolUse) with
        # the call id and name, followed by contentBlockDelta events
        # carrying a ``toolUse.input`` JSON fragment, then a
        # contentBlockStop. We accumulate the fragments per index so we
        # can emit exactly one ToolUseRequest when the block closes.
        pending_tool_blocks: dict[int, dict] = {}
        try:
            for event in stream:
                if "contentBlockStart" in event:
                    block = event["contentBlockStart"]
                    idx = block.get("contentBlockIndex")
                    start = block.get("start") or {}
                    tool_use = start.get("toolUse")
                    if idx is not None and tool_use:
                        pending_tool_blocks[idx] = {
                            "call_id": tool_use.get("toolUseId"),
                            "name": tool_use.get("name"),
                            "input": "",
                        }
                    continue
                if "contentBlockDelta" in event:
                    delta = event["contentBlockDelta"]
                    idx = delta.get("contentBlockIndex")
                    payload = delta.get("delta") or {}
                    text = payload.get("text")
                    if text:
                        yield tools_module.TextDelta(text=text)
                        continue
                    tool_use = payload.get("toolUse")
                    if tool_use and idx in pending_tool_blocks:
                        pending_tool_blocks[idx]["input"] += tool_use.get(
                            "input", ""
                        )
                    continue
                if "contentBlockStop" in event:
                    idx = event["contentBlockStop"].get("contentBlockIndex")
                    block = pending_tool_blocks.pop(idx, None)
                    if block is not None and block.get("name"):
                        try:
                            args = (
                                json.loads(block["input"])
                                if block["input"]
                                else {}
                            )
                        except json.JSONDecodeError:
                            args = {"_raw_input": block["input"]}
                        yield tools_module.ToolUseRequest(
                            call_id=block["call_id"] or "",
                            name=block["name"],
                            arguments=args,
                        )
                    continue
                if "messageStop" in event:
                    reason = event["messageStop"].get("stopReason") or "end_turn"
                    yield tools_module.Stop(reason=reason)
                    return
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(
                f"bedrock tool stream interrupted: {exc}"
            ) from exc
        # If the stream ended without an explicit messageStop, emit one
        # so the router can close the turn cleanly.
        yield tools_module.Stop(reason="end_turn")

    def _to_bedrock_messages(self, messages: list[dict]) -> list[dict]:
        """Lower provider neutral messages to Bedrock Converse format.

        Consecutive ``role == "tool"`` entries are merged into a single
        Bedrock user message whose ``content`` array holds one
        ``toolResult`` block per neutral entry. Bedrock requires every
        ``toolUse`` block in an assistant turn to be answered by a
        ``toolResult`` block in the next user turn, so parallel tool
        calls must land in the same message - splitting them across
        messages triggers a ValidationException.
        """
        out: list[dict] = []
        i = 0
        n = len(messages)
        while i < n:
            msg = messages[i]
            if msg.get("role") == "tool":
                blocks: list[dict] = []
                while i < n and messages[i].get("role") == "tool":
                    blocks.append(self._tool_result_block(messages[i]))
                    i += 1
                out.append({"role": "user", "content": blocks})
                continue
            out.append(self._to_bedrock_message(msg))
            i += 1
        return out

    def _tool_result_block(self, msg: dict) -> dict:
        """Build a single ``toolResult`` content block from a tool message."""
        result = msg["tool_result"]
        return {
            "toolResult": {
                "toolUseId": result.call_id,
                "content": [{"json": result.content}],
                "status": "error" if result.is_error else "success",
            }
        }

    def _to_bedrock_message(self, msg: dict) -> dict:
        """Lower a provider neutral message to Bedrock Converse format."""
        role = msg.get("role")
        if role == "user":
            return {"role": "user", "content": [{"text": str(msg["content"])}]}
        if role == "assistant" and msg.get("tool_calls"):
            blocks = []
            for call in msg["tool_calls"]:
                blocks.append(
                    {
                        "toolUse": {
                            "toolUseId": call.call_id,
                            "name": call.name,
                            "input": call.arguments,
                        }
                    }
                )
            text = msg.get("content")
            if text:
                blocks.insert(0, {"text": str(text)})
            return {"role": "assistant", "content": blocks}
        if role == "assistant":
            return {
                "role": "assistant",
                "content": [{"text": str(msg.get("content") or "")}],
            }
        if role == "tool":
            return {"role": "user", "content": [self._tool_result_block(msg)]}
        raise ValueError(f"unsupported message role: {role!r}")

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        augmented_system = (
            system
            + "\n\nYou MUST reply with a single JSON object that matches the "
            + "schema provided in the user message. Do not include prose, "
            + "markdown fences, or comments. JSON only."
        )
        augmented_user = (
            user
            + "\n\nJSON schema (informal): "
            + json.dumps(schema, ensure_ascii=False)
        )

        last_error: Exception | None = None
        attempt_user = augmented_user
        for _ in range(self.MAX_REPAIR_ATTEMPTS + 1):
            text = self._converse(augmented_system, attempt_user)
            try:
                return _coerce_json(text)
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                attempt_user = (
                    augmented_user
                    + "\n\nYour previous reply was not valid JSON. "
                    + f"Error: {exc}. Reply with JSON ONLY."
                )
                continue
        raise LLMProviderError(
            f"bedrock produced no valid JSON after retries: {last_error}"
        )


# ---------------------------------------------------------------------------
# Skeletons (to be implemented in future phases)
# ---------------------------------------------------------------------------


class _UnimplementedProvider(LLMProvider):
    """Base class for providers that are declared but not yet implemented."""

    description: str = ""

    def health(self) -> dict:
        return {
            "ok": False,
            "provider": self.name,
            "model": None,
            "detail": f"{self.name} provider is not implemented yet",
        }

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        raise NotImplementedError(
            f"{self.name} provider is not implemented yet ({self.description})"
        )

    def stream(
        self,
        system: str,
        user: str,
        history: list[dict] | None = None,
    ) -> Iterator[str]:
        raise NotImplementedError(
            f"{self.name} provider is not implemented yet ({self.description})"
        )
        yield ""  # pragma: no cover


class OllamaProvider(_UnimplementedProvider):
    name = PROVIDER_OLLAMA
    description = "local Ollama HTTP endpoint"


class OpenAIProvider(_UnimplementedProvider):
    name = PROVIDER_OPENAI
    description = "OpenAI compatible chat completions"


class AnthropicProvider(_UnimplementedProvider):
    name = PROVIDER_ANTHROPIC
    description = "Anthropic Messages API"


class GeminiProvider(_UnimplementedProvider):
    name = PROVIDER_GEMINI
    description = "Google Gemini API"


PROVIDERS: dict[str, type[LLMProvider]] = {
    PROVIDER_NONE: NoneProvider,
    PROVIDER_BEDROCK: BedrockProvider,
    PROVIDER_OLLAMA: OllamaProvider,
    PROVIDER_OPENAI: OpenAIProvider,
    PROVIDER_ANTHROPIC: AnthropicProvider,
    PROVIDER_GEMINI: GeminiProvider,
}


# ---------------------------------------------------------------------------
# Process wide cache
# ---------------------------------------------------------------------------

_provider_lock = threading.Lock()
_provider_cache: tuple[str, LLMProvider] | None = None


def _resolve_name(value: str | None) -> str:
    if not value:
        return PROVIDER_NONE
    value = value.strip().lower()
    if value not in PROVIDERS:
        # Unknown provider name: be conservative and disable LLM features.
        return PROVIDER_NONE
    return value


def get_provider() -> LLMProvider:
    """Return a cached provider instance chosen by ``LLM_PROVIDER`` env."""
    global _provider_cache
    name = _resolve_name(os.environ.get("LLM_PROVIDER"))
    with _provider_lock:
        if _provider_cache is not None and _provider_cache[0] == name:
            return _provider_cache[1]
        cls = PROVIDERS[name]
        instance = cls()
        _provider_cache = (name, instance)
        return instance


def reset_provider_cache() -> None:
    """Clear the cached instance. Intended for tests and env changes."""
    global _provider_cache
    with _provider_lock:
        _provider_cache = None
