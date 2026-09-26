from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf


_FILENAME_HINTS = (
    "pdf",
    "scan",
    "scanned",
    "page",
    "document",
    "doc",
    "invoice",
    "receipt",
    "report",
    "paper",
    "article",
    "form",
    "letter",
    "screenshot",
)


@dataclass(frozen=True)
class DocumentImageAssessment:
    is_document_like: bool
    score: float
    reasons: tuple[str, ...]


def assess_document_image(path: str | Path) -> DocumentImageAssessment:
    """Cheap, deterministic precheck for scanned/document-page images.

    This intentionally performs no OCR and loads no ML weights. It combines
    filename hints with page-like geometry and coarse luminance/edge statistics.
    The result is only a routing hint; semantic extraction remains the job of the
    configured document/VLM provider.
    """

    path = Path(path)
    score = 0.0
    reasons: list[str] = []
    name = path.stem.lower().replace("-", "_")

    if any(hint in name for hint in _FILENAME_HINTS):
        score += 0.45
        reasons.append("filename_hint")

    try:
        pix = pymupdf.Pixmap(str(path))
    except Exception:
        return DocumentImageAssessment(score >= 0.45, min(score, 1.0), tuple(reasons))

    try:
        width, height = int(pix.width), int(pix.height)
        if width <= 0 or height <= 0:
            return DocumentImageAssessment(score >= 0.45, min(score, 1.0), tuple(reasons))

        short_long_ratio = min(width, height) / max(width, height)
        # Covers A-series/US-letter-ish pages in either orientation without
        # treating very panoramic photographs as documents.
        if 0.58 <= short_long_ratio <= 0.85:
            score += 0.15
            reasons.append("page_aspect_ratio")

        channels = max(1, int(pix.n))
        samples = pix.samples
        step_x = max(1, width // 96)
        step_y = max(1, height // 96)

        total = 0
        white = 0
        dark = 0
        edge_pairs = 0
        edge_hits = 0

        for y in range(0, height, step_y):
            previous_luma: int | None = None
            row_base = y * width * channels
            for x in range(0, width, step_x):
                idx = row_base + x * channels
                if channels >= 3:
                    r, g, b = samples[idx], samples[idx + 1], samples[idx + 2]
                    luma = int(0.299 * r + 0.587 * g + 0.114 * b)
                else:
                    luma = samples[idx]

                total += 1
                if luma >= 220:
                    white += 1
                if luma <= 95:
                    dark += 1

                if previous_luma is not None:
                    edge_pairs += 1
                    if abs(luma - previous_luma) >= 34:
                        edge_hits += 1
                previous_luma = luma

        if total:
            white_fraction = white / total
            dark_fraction = dark / total
            edge_fraction = edge_hits / edge_pairs if edge_pairs else 0.0

            if white_fraction >= 0.50:
                score += 0.15
                reasons.append("light_page_background")
            if 0.008 <= dark_fraction <= 0.38:
                score += 0.10
                reasons.append("text_like_dark_content")
            if edge_fraction >= 0.035:
                score += 0.10
                reasons.append("text_like_edge_density")
    finally:
        pix = None

    score = min(score, 1.0)
    return DocumentImageAssessment(score >= 0.50, score, tuple(reasons))
