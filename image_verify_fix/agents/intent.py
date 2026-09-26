from __future__ import annotations

import re
from dataclasses import dataclass

from agents.state import RequestMode


_TRANSFORM_ACTIONS = {
    "summarize", "summarise", "convert", "transform", "rewrite", "rephrase",
    "refine", "translate", "draft", "prepare", "produce", "generate", "create",
    "make", "turn", "professionalize", "professionalise", "shorten", "expand",
}

_ARTIFACT_TERMS = {
    "summary", "advisory", "executive brief", "executive briefing", "executive summary",
    "linkedin post", "twitter post", "tweet", "thread", "press release", "report",
    "presentation", "ppt", "pptx", "pdf", "infographic",
}

_QA_PREFIXES = (
    "what ", "why ", "how ", "when ", "where ", "who ", "which ", "does ",
    "do ", "did ", "is ", "are ", "was ", "were ", "can ", "could ", "should ",
    "list ", "find ", "identify ", "explain ", "tell me ", "give me the ",
)


@dataclass(frozen=True)
class IntentDecision:
    intent: str
    confidence: float
    reason: str


class IntentRouter:
    """Deterministic Phase 3 intent router.

    A future release can replace this with a hosted LLM classifier without changing the
    graph contract. Explicit request_mode always wins over heuristic routing.
    """

    def classify(self, query: str, request_mode: RequestMode = "auto") -> IntentDecision:
        text = re.sub(r"\s+", " ", (query or "").strip().lower())
        if not text:
            return IntentDecision("index_only", 1.0, "No query supplied; ingestion/indexing only.")

        if request_mode == "qa":
            return IntentDecision("qa", 1.0, "Explicit request_mode=qa.")
        if request_mode == "transform":
            return IntentDecision("transform", 1.0, "Explicit request_mode=transform.")

        action_hits = [term for term in _TRANSFORM_ACTIONS if term in text]
        if action_hits:
            return IntentDecision(
                "transform",
                min(0.98, 0.84 + 0.03 * len(action_hits)),
                f"Transformation action detected: {', '.join(sorted(action_hits)[:4])}.",
            )

        # Artifact names by themselves do not override a clear question. This
        # avoids routing "What does the executive summary say?" as a transform.
        if text.endswith("?") or text.startswith(_QA_PREFIXES):
            return IntentDecision("qa", 0.90, "Question/search language detected.")

        artifact_hits = [term for term in _ARTIFACT_TERMS if term in text]
        if artifact_hits:
            return IntentDecision(
                "transform",
                0.76,
                f"Requested artefact language detected: {', '.join(sorted(artifact_hits)[:3])}.",
            )

        # Most free-form requests over an indexed source are safer as focused
        # retrieval unless they explicitly ask to create/transform an artefact.
        return IntentDecision("qa", 0.62, "No transformation cue found; defaulting to focused retrieval.")
