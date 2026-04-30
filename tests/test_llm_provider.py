"""Unit tests for the LLM provider layer.

Bedrock is exercised by injecting a fake boto3 client so we never touch
AWS from CI. The four skeleton providers must exist and raise
``NotImplementedError`` so future phases can replace them without import
churn.
"""

from __future__ import annotations

import json

import pytest

from atop_web.llm import provider as provider_mod
from atop_web.llm.provider import (
    AnthropicProvider,
    BedrockProvider,
    GeminiProvider,
    LLMProviderError,
    NoneProvider,
    OllamaProvider,
    OpenAIProvider,
    PROVIDERS,
    get_provider,
    reset_provider_cache,
)


# None provider ---------------------------------------------------------------


def test_none_provider_reports_ok():
    p = NoneProvider()
    snap = p.health()
    assert snap["ok"] is True
    assert snap["provider"] == "none"
    assert snap["model"] is None
    assert snap["detail"]


def test_none_provider_complete_json_raises():
    p = NoneProvider()
    with pytest.raises(LLMProviderError):
        p.complete_json("sys", "user", {})


# Skeleton providers ----------------------------------------------------------


@pytest.mark.parametrize(
    "cls,name",
    [
        (OllamaProvider, "ollama"),
        (OpenAIProvider, "openai"),
        (AnthropicProvider, "anthropic"),
        (GeminiProvider, "gemini"),
    ],
)
def test_skeleton_provider_raises_not_implemented(cls, name):
    p = cls()
    assert p.name == name
    snap = p.health()
    assert snap["ok"] is False
    assert snap["provider"] == name
    with pytest.raises(NotImplementedError):
        p.complete_json("s", "u", {})


def test_providers_mapping_is_complete():
    for key in ("none", "bedrock", "ollama", "openai", "anthropic", "gemini"):
        assert key in PROVIDERS


# Provider cache --------------------------------------------------------------


def test_get_provider_defaults_to_none(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    reset_provider_cache()
    assert isinstance(get_provider(), NoneProvider)


def test_get_provider_unknown_falls_back_to_none(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "totally-not-a-provider")
    reset_provider_cache()
    assert isinstance(get_provider(), NoneProvider)


def test_get_provider_respects_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    reset_provider_cache()
    inst = get_provider()
    assert isinstance(inst, BedrockProvider)
    # Subsequent calls must return the cached instance.
    assert get_provider() is inst


# Bedrock provider (fake boto3) ----------------------------------------------


class _FakeBedrockClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


def _install_fake_boto3(monkeypatch, client):
    import sys
    import types

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda name, **kwargs: client
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)


def _make_response(text: str) -> dict:
    return {
        "output": {
            "message": {
                "content": [{"text": text}],
            }
        }
    }


def test_bedrock_complete_json_happy_path(monkeypatch):
    client = _FakeBedrockClient(
        [_make_response('{"issues": [{"title": "t", "severity": "info", "detail": "d"}]}')]
    )
    _install_fake_boto3(monkeypatch, client)

    p = BedrockProvider(model="fake-model", region="us-east-1")
    out = p.complete_json("sys", "user", {"type": "object"})
    assert out == {
        "issues": [{"title": "t", "severity": "info", "detail": "d"}]
    }
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["modelId"] == "fake-model"
    # System and user texts are forwarded; the user side must include the
    # schema so the model has shape hints.
    assert call["system"][0]["text"].startswith("sys")
    user_text = call["messages"][0]["content"][0]["text"]
    assert "user" in user_text
    assert "JSON schema" in user_text


def test_bedrock_retries_on_bad_json(monkeypatch):
    client = _FakeBedrockClient(
        [
            _make_response("not json"),
            _make_response('```json\n{"issues": []}\n```'),
        ]
    )
    _install_fake_boto3(monkeypatch, client)

    p = BedrockProvider(model="fake", region="us-east-1")
    out = p.complete_json("sys", "user", {})
    assert out == {"issues": []}
    assert len(client.calls) == 2


def test_bedrock_gives_up_after_retries(monkeypatch):
    client = _FakeBedrockClient(
        [_make_response("bad") for _ in range(5)]
    )
    _install_fake_boto3(monkeypatch, client)

    p = BedrockProvider(model="fake", region="us-east-1")
    with pytest.raises(LLMProviderError):
        p.complete_json("sys", "user", {})
    # Initial attempt plus MAX_REPAIR_ATTEMPTS retries.
    assert len(client.calls) == BedrockProvider.MAX_REPAIR_ATTEMPTS + 1


def test_bedrock_health_reports_model(monkeypatch):
    client = _FakeBedrockClient([])
    _install_fake_boto3(monkeypatch, client)

    p = BedrockProvider(model="fake-model", region="us-west-2")
    snap = p.health()
    assert snap["ok"] is True
    assert snap["provider"] == "bedrock"
    assert snap["model"] == "fake-model"
    assert "us-west-2" in snap["detail"]


def test_bedrock_health_reports_error_when_boto3_missing(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "boto3", None)
    p = BedrockProvider(model="fake")
    snap = p.health()
    assert snap["ok"] is False
    assert "boto3" in snap["detail"]
