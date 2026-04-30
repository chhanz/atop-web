"""Optional LLM integration package.

atop-web works without any LLM. When ``LLM_PROVIDER`` is unset or set to
``none``, ``get_provider()`` returns a stub that reports itself as healthy
but raises on any actual completion call. This keeps the rest of the app
happy with a uniform interface while the UI can hide LLM only features by
inspecting ``/api/llm/health``.
"""

from atop_web.llm.provider import (
    LLMProvider,
    LLMProviderError,
    NoneProvider,
    get_provider,
    reset_provider_cache,
)

__all__ = [
    "LLMProvider",
    "LLMProviderError",
    "NoneProvider",
    "get_provider",
    "reset_provider_cache",
]
