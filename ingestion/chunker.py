from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Iterable

from app.schemas import Chunk, IngestionResult, SourceElement


class StructureAwareChunker:
    def __init__(self, *, target_chars: int = 3200, overlap_chars: int = 450):
        self.target_chars = target_chars
        self.overlap_chars = overlap_chars

    def chunk(self, result: IngestionResult, *, session_id: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        buffer: list[SourceElement] = []
        size = 0

        def flush() -> None:
            nonlocal buffer, size
            if not buffer:
                return
            text = "\n\n".join(element.text for element in buffer).strip()
            raw = "\n\n".join((element.raw_text or element.text) for element in buffer).strip()
            if not text:
                buffer, size = [], 0
                return
            first = buffer[0]
            last = buffer[-1]
            seed = f"{session_id}:{result.source_id}:{len(chunks)}:{text[:160]}".encode("utf-8")
            chunk_id = hashlib.sha1(seed).hexdigest()[:24]
            pages = [e.page for e in buffer if e.page is not None]
            slides = [e.slide for e in buffer if e.slide is not None]
            sections = [e.section for e in buffer if e.section]
            metadata = {
                "chunk_id": chunk_id,
                "session_id": session_id,
                "source_id": result.source_id,
                "filename": result.filename,
                "media_type": result.media_type,
                "strategy": result.strategy,
                "page_start": min(pages) if pages else -1,
                "page_end": max(pages) if pages else -1,
                "slide_start": min(slides) if slides else -1,
                "slide_end": max(slides) if slides else -1,
                "section": sections[-1] if sections else "",
                "element_start": first.element_id,
                "element_end": last.element_id,
            }
            chunks.append(Chunk(chunk_id, text, metadata, raw_text=raw))

            # Character-level overlap while preserving the final element when possible.
            if self.overlap_chars > 0 and buffer:
                overlap: list[SourceElement] = []
                overlap_size = 0
                for element in reversed(buffer):
                    overlap.insert(0, element)
                    overlap_size += len(element.text)
                    if overlap_size >= self.overlap_chars:
                        break
                buffer = overlap
                size = sum(len(e.text) for e in buffer)
            else:
                buffer, size = [], 0

        for element in result.elements:
            # Preserve table/picture blocks instead of splitting them internally.
            element_size = len(element.text)
            if buffer and size + element_size > self.target_chars:
                flush()
            buffer.append(element)
            size += element_size
            if element_size >= self.target_chars:
                flush()
        flush()
        return chunks
