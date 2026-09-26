from __future__ import annotations

import json
from generation.content_ir import ContentIR
from generation.crr import TransformationConfig


def _base(content: ContentIR, config: TransformationConfig) -> str:
    return (
        "VERIFIED CONTENT IR:\n" + json.dumps(content.model_dump(mode="json"), ensure_ascii=False) +
        "\n\nTRANSFORMATION CONFIG:\n" + json.dumps(config.model_dump(mode="json"), ensure_ascii=False)
    )


def artifact_system_prompt() -> str:
    return (
        "You are an artifact decoder. The supplied Content IR is already verified. "
        "Do not add factual claims that are absent from it. Produce only the requested structured artifact IR. "
        "Preserve exact numbers, dates, names, and quoted terminology."
    )


def text_prompt(content: ContentIR, config: TransformationConfig) -> str:
    return _base(content, config) + "\n\nProduce polished final text suitable for the configured artifact type, audience, tone and language."


def pdf_prompt(content: ContentIR, config: TransformationConfig) -> str:
    return _base(content, config) + (
        "\n\nProduce a complete Typst document in typst_source. The document should be professional, readable, "
        "and self-contained. Use only standard Typst syntax and the supplied verified content."
    )


def pptx_prompt(content: ContentIR, config: TransformationConfig) -> str:
    return _base(content, config) + (
        "\n\nPlan an editable presentation. Return slides with concise titles, bullets/body, optional metrics, "
        "speaker notes and visual_prompt. Keep each slide focused and avoid inventing data."
    )


def svg_prompt(content: ContentIR, config: TransformationConfig) -> str:
    return _base(content, config) + (
        "\n\nProduce one complete, standards-compliant SVG string for a factual infographic. "
        "All visible numbers and labels must match the verified content exactly. Avoid external assets."
    )


def image_prompt(content: ContentIR, config: TransformationConfig) -> str:
    return _base(content, config) + (
        "\n\nProduce a concise creative image-generation prompt. Do not request visible factual text, statistics, "
        "or labels inside the generated image; factual text should be rendered separately."
    )
