from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from verification.models import DeterministicIssue


_NUMBER_RE = re.compile(
    r"(?<![\w.])(?P<currency>[$€£₹])?\s*(?P<number>[-+]?\d[\d,]*(?:\.\d+)?)"
    r"\s*(?P<unit>%|percent|per\s+cent|million|billion|thousand|crore|lakh)?(?!\w)",
    re.IGNORECASE,
)
_DATE_PATTERNS = [
    re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b"),
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*|\s+)\d{4}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{4}\b",
        re.IGNORECASE,
    ),
]
_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9&./-]{1,}\b")
_MULTIWORD_ENTITY_RE = re.compile(r"\b(?:[A-Z][\w&.'-]+\s+){1,4}[A-Z][\w&.'-]+\b")
_QUOTED_RE = re.compile(r"[\"'“”‘’]([^\"'“”‘’]{2,80})[\"'“”‘’]")


@dataclass(frozen=True)
class NumericLiteral:
    raw: str
    value: str
    kind: str


def _fold(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _normalize_number(value: str) -> str:
    value = value.replace(",", "")
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return str(int(number))
    return ("%.12f" % number).rstrip("0").rstrip(".")


def extract_numeric_literals(text: str) -> list[NumericLiteral]:
    text = text or ""
    date_spans: list[tuple[int, int]] = []
    for pattern in _DATE_PATTERNS:
        date_spans.extend(match.span() for match in pattern.finditer(text))
    out: list[NumericLiteral] = []
    for match in _NUMBER_RE.finditer(text):
        if any(match.start() >= start and match.end() <= end for start, end in date_spans):
            continue
        currency = match.group("currency") or ""
        unit = _fold(match.group("unit") or "")
        if unit in {"percent", "per cent", "%"}:
            unit = "%"
        number = _normalize_number(match.group("number"))
        if currency:
            kind = f"currency:{currency}"
            value = f"{currency}{number}"
        elif unit:
            kind = f"unit:{unit}"
            value = f"{number}{unit}"
        else:
            kind = "number"
            value = number
        out.append(NumericLiteral(match.group(0).strip(), value, kind))
    return out


def extract_dates(text: str) -> list[str]:
    values: list[str] = []
    for pattern in _DATE_PATTERNS:
        values.extend(match.group(0) for match in pattern.finditer(text or ""))
    return list(dict.fromkeys(values))


def extract_high_confidence_entities(text: str) -> list[str]:
    values = [m.group(0) for m in _ACRONYM_RE.finditer(text or "")]
    values.extend(m.group(0) for m in _MULTIWORD_ENTITY_RE.finditer(text or ""))
    values.extend(m.group(1) for m in _QUOTED_RE.finditer(text or ""))
    # Filter common title-like phrases that are weak entity signals.
    stop = {"Key Findings", "Executive Summary", "Recommended Actions", "Source Evidence"}
    return list(dict.fromkeys(v.strip() for v in values if v.strip() and v.strip() not in stop))


class DeterministicLiteralValidator:
    """Conservative deterministic checks layered on top of semantic verification.

    Only *conflicting* comparable numeric/date literals hard-fail a claim. Missing
    high-confidence entities are emitted as warnings and left to the semantic
    verifier, avoiding brittle English-only NER assumptions.
    """

    def validate(self, claim_text: str, evidence_text: str) -> list[DeterministicIssue]:
        issues: list[DeterministicIssue] = []
        claim_numbers = extract_numeric_literals(claim_text)
        evidence_numbers = extract_numeric_literals(evidence_text)
        evidence_by_kind: dict[str, set[str]] = {}
        for item in evidence_numbers:
            evidence_by_kind.setdefault(item.kind, set()).add(item.value)

        for item in claim_numbers:
            comparable = evidence_by_kind.get(item.kind, set())
            if item.value in comparable:
                continue
            if comparable:
                issues.append(
                    DeterministicIssue(
                        kind="numeric_conflict",
                        severity="conflict",
                        literal=item.raw,
                        message=(
                            f"Claim literal '{item.raw}' is not present in cited evidence; "
                            f"evidence contains comparable values: {', '.join(sorted(comparable))}."
                        ),
                    )
                )
            else:
                issues.append(
                    DeterministicIssue(
                        kind="numeric_literal_unmatched",
                        severity="warning",
                        literal=item.raw,
                        message=f"Claim literal '{item.raw}' was not matched exactly in cited evidence.",
                    )
                )

        claim_dates = extract_dates(claim_text)
        evidence_dates = extract_dates(evidence_text)
        evidence_dates_folded = {_fold(v) for v in evidence_dates}
        for date in claim_dates:
            if _fold(date) in evidence_dates_folded:
                continue
            severity = "conflict" if evidence_dates else "warning"
            issues.append(
                DeterministicIssue(
                    kind="date_conflict" if severity == "conflict" else "date_literal_unmatched",
                    severity=severity,
                    literal=date,
                    message=(
                        f"Claim date '{date}' is not present in cited evidence"
                        + (f"; evidence contains: {', '.join(evidence_dates)}." if evidence_dates else ".")
                    ),
                )
            )

        evidence_folded = _fold(evidence_text)
        for entity in extract_high_confidence_entities(claim_text):
            if _fold(entity) not in evidence_folded:
                issues.append(
                    DeterministicIssue(
                        kind="entity_literal_unmatched",
                        severity="warning",
                        literal=entity,
                        message=f"High-confidence entity/identifier '{entity}' was not matched in cited evidence.",
                    )
                )
        return issues
