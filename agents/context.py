from __future__ import annotations

from collections import OrderedDict
from typing import Any


def _meta(doc: dict[str, Any]) -> dict[str, Any]:
    return dict(doc.get("metadata") or {})


def _position_key(doc: dict[str, Any]) -> tuple[Any, ...]:
    meta = _meta(doc)
    page = meta.get("page_start", -1)
    slide = meta.get("slide_start", -1)
    # -1 means not applicable; push it after real positions of the same type.
    page_key = page if isinstance(page, int) and page >= 0 else 10**9
    slide_key = slide if isinstance(slide, int) and slide >= 0 else 10**9
    return (
        str(meta.get("source_id", "")),
        slide_key,
        page_key,
        str(meta.get("section", "")),
        str(meta.get("chunk_id", doc.get("id", ""))),
    )


def _location(meta: dict[str, Any]) -> str:
    if isinstance(meta.get("slide_start"), int) and meta.get("slide_start", -1) >= 0:
        start = meta["slide_start"]
        end = meta.get("slide_end", start)
        return f"SLIDE {start}" if end == start else f"SLIDES {start}-{end}"
    if isinstance(meta.get("page_start"), int) and meta.get("page_start", -1) >= 0:
        start = meta["page_start"]
        end = meta.get("page_end", start)
        return f"PAGE {start}" if end == start else f"PAGES {start}-{end}"
    return "LOCATION UNKNOWN"


def group_hierarchical(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group an ordered corpus by source and structural section.

    The full chunk list is retained in each group; no summarization happens in
    Phase 3. This is the handoff structure for Phase 4 hierarchical generation.
    """
    ordered = sorted(documents, key=_position_key)
    groups: "OrderedDict[tuple[str, str], dict[str, Any]]" = OrderedDict()
    for doc in ordered:
        meta = _meta(doc)
        source_id = str(meta.get("source_id", "unknown-source"))
        section = str(meta.get("section") or "Unsectioned")
        key = (source_id, section)
        if key not in groups:
            groups[key] = {
                "source_id": source_id,
                "filename": str(meta.get("filename", "")),
                "section": section,
                "chunks": [],
            }
        groups[key]["chunks"].append(doc)
    return list(groups.values())


def group_qa(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"source_id": "retrieved", "filename": "", "section": "Top-K evidence", "chunks": documents}]


def render_context(
    groups: list[dict[str, Any]],
    *,
    max_chars: int,
    chunk_to_alias: dict[str, str] | None = None,
) -> str:
    """Create a bounded, provenance-rich text handoff.

    The structured `context_groups` state always retains all retrieved chunks.
    This rendering is intentionally bounded so a downstream model call can use
    it directly when the corpus is small enough.
    """
    if max_chars <= 0:
        return ""

    parts: list[str] = []
    used = 0

    # Round-robin across groups avoids spending the entire budget on the first
    # section of a long transformation request.
    positions = [0 for _ in groups]
    remaining = True
    while remaining and used < max_chars:
        remaining = False
        for index, group in enumerate(groups):
            chunks = group.get("chunks") or []
            pos = positions[index]
            if pos >= len(chunks):
                continue
            remaining = True
            doc = chunks[pos]
            positions[index] += 1
            meta = _meta(doc)
            source = meta.get("filename") or meta.get("source_id") or group.get("filename") or group.get("source_id")
            section = meta.get("section") or group.get("section") or "Unsectioned"
            chunk_id = str(meta.get("chunk_id") or doc.get("id") or "unknown")
            evidence_label = (chunk_to_alias or {}).get(chunk_id, chunk_id)
            evidence_key = "EVIDENCE" if chunk_to_alias is not None else "CHUNK"
            block = (
                f"SOURCE: {source}\n"
                f"SECTION: {section}\n"
                f"{_location(meta)}\n"
                f"{evidence_key}: {evidence_label}\n\n"
                f"{str(doc.get('text', '')).strip()}\n"
            )
            if used + len(block) > max_chars:
                remaining = False
                break
            parts.append(block)
            used += len(block) + 5
            if used >= max_chars:
                break

    return "\n---\n\n".join(parts)
