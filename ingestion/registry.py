from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.schemas import IngestionResult


Handler = Callable[[Path, str, str], IngestionResult]


@dataclass(frozen=True)
class ProcessorRegistration:
    name: str
    suffixes: frozenset[str]
    handler: Handler


class IngestionProcessorRegistry:
    """Extension registry for input processors.

    New file families can be added without editing IngestionRouter.ingest(). MIME
    resolution can be added later while retaining this contract.
    """

    def __init__(self):
        self._entries: list[ProcessorRegistration] = []

    def register(self, name: str, suffixes: set[str] | frozenset[str], handler: Handler) -> None:
        normalized = frozenset(s.lower() if s.startswith(".") else f".{s.lower()}" for s in suffixes)
        self._entries = [entry for entry in self._entries if entry.name != name]
        self._entries.append(ProcessorRegistration(name=name, suffixes=normalized, handler=handler))

    def resolve(self, path: Path) -> ProcessorRegistration | None:
        suffix = path.suffix.lower()
        for entry in self._entries:
            if suffix in entry.suffixes:
                return entry
        return None

    def supported_suffixes(self) -> list[str]:
        return sorted({suffix for entry in self._entries for suffix in entry.suffixes})
