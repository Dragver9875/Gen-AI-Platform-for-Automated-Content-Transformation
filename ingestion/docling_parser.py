from __future__ import annotations

import re
from typing import Any

from app.schemas import SourceElement
from ingestion.normalizer import normalize_text


def _page_from_prov(item: dict[str, Any]) -> int | None:
    prov = item.get("prov")
    if isinstance(prov, list) and prov:
        candidate = prov[0]
        if isinstance(candidate, dict):
            page = candidate.get("page_no") or candidate.get("page")
            if page is not None:
                try:
                    return int(page)
                except (TypeError, ValueError):
                    return None
    return None


def elements_from_docling_json(data: dict[str, Any]) -> list[SourceElement]:
    elements: list[SourceElement] = []
    seen: set[str] = set()

    for bucket_name, kind in (("texts", "text"), ("tables", "table"), ("pictures", "picture"), ("key_value_items", "key_value")):
        bucket = data.get(bucket_name)
        if not isinstance(bucket, list):
            continue
        for index, item in enumerate(bucket):
            if not isinstance(item, dict):
                continue
            text = item.get("text") or item.get("orig") or item.get("caption") or ""
            if bucket_name == "tables" and not text:
                grid = ((item.get("data") or {}).get("grid") if isinstance(item.get("data"), dict) else None)
                if isinstance(grid, list):
                    rows = []
                    for row in grid:
                        if isinstance(row, list):
                            rows.append(" | ".join(str(cell.get("text", "") if isinstance(cell, dict) else cell) for cell in row))
                    text = "\n".join(rows)
            if bucket_name == "pictures" and not text:
                annotations = item.get("annotations")
                if isinstance(annotations, list):
                    text = " ".join(str(a.get("text") or a.get("label") or "") for a in annotations if isinstance(a, dict))
            text = normalize_text(str(text))
            if not text:
                continue
            element_id = str(item.get("self_ref") or f"{bucket_name}-{index}")
            if element_id in seen:
                continue
            seen.add(element_id)
            label = str(item.get("label") or kind)
            elements.append(
                SourceElement(
                    element_id=element_id,
                    kind=label,
                    text=text,
                    raw_text=str(item.get("orig") or text),
                    page=_page_from_prov(item),
                    metadata={"docling_bucket": bucket_name},
                )
            )
    return elements


def elements_from_markdown(markdown: str) -> list[SourceElement]:
    markdown = normalize_text(markdown)
    if not markdown:
        return []
    elements: list[SourceElement] = []
    section: str | None = None
    blocks = re.split(r"\n\s*\n", markdown)
    for index, block in enumerate(blocks):
        block = block.strip()
        if not block:
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", block)
        if heading:
            section = heading.group(2).strip()
            elements.append(SourceElement(f"md-{index}", "heading", section, section=section))
        else:
            elements.append(SourceElement(f"md-{index}", "paragraph", block, section=section))
    return elements
