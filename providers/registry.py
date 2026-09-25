from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ProviderCapability:
    DOCUMENT_UNDERSTANDING = "document_understanding"
    VISUAL_ROUTING = "visual_routing"
    VISUAL_UNDERSTANDING = "visual_understanding"
    EMBEDDING = "embedding"
    RERANKING = "reranking"
    STRUCTURED_GENERATION = "structured_generation"
    VERIFICATION = "verification"
    IMAGE_GENERATION = "image_generation"


@dataclass(frozen=True)
class ProviderEntry:
    capability: str
    name: str
    provider: Any


class ProviderRegistry:
    """Runtime capability registry.

    LangGraph nodes and services consume capabilities rather than model names.
    Provider-specific construction remains in the composition root (app.factory).
    """

    def __init__(self):
        self._providers: dict[str, dict[str, Any]] = {}
        self._defaults: dict[str, str] = {}

    def register(self, capability: str, provider: Any, *, name: str = "default", default: bool = True) -> Any:
        bucket = self._providers.setdefault(capability, {})
        bucket[name] = provider
        if default or capability not in self._defaults:
            self._defaults[capability] = name
        return provider

    def resolve(self, capability: str, *, name: str | None = None, required: bool = True) -> Any | None:
        bucket = self._providers.get(capability, {})
        selected = name or self._defaults.get(capability)
        provider = bucket.get(selected) if selected else None
        if provider is None and required:
            raise KeyError(f"No provider registered for capability '{capability}'" + (f" with name '{name}'" if name else ""))
        return provider

    def capabilities(self) -> dict[str, list[str]]:
        return {capability: sorted(entries) for capability, entries in self._providers.items()}
