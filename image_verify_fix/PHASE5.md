# Phase 5 — Factuality Verification and Bounded Repair

Phase 5 extends the Phase 4 Canonical Response Representation (CRR) pipeline with claim-level grounding verification and a bounded repair loop.

## What Phase 5 does

1. Verifies every CRR claim only against the chunks cited by that claim.
2. Uses a hosted structured-output verifier model; no local model weights are loaded.
3. Applies deterministic checks for critical numeric/date conflicts and conservative entity/identifier mismatches.
4. Computes the faithfulness score locally rather than trusting a model-produced score.
5. Routes failed CRRs through a source-grounded repair call.
6. Re-verifies the repaired CRR.
7. Stops after `PHASE5_MAX_REPAIR_ATTEMPTS` and returns `phase5_complete_with_issues` if unresolved claims remain.

## Status values

- `phase5_complete`: every factual claim is supported.
- `phase5_complete_with_issues`: the repair limit was reached with unresolved claims.
- `indexed_ready`: the request only indexed files and did not request generation.
- `error`: an unrecoverable pipeline/provider error occurred.

## Verification statuses

- `supported`
- `partially_supported`
- `unsupported`
- `insufficient_evidence`

The faithfulness score is calculated as:

- supported = 1.0
- partially supported = 0.5
- unsupported = 0.0
- insufficient evidence = 0.0

An artifact passes only if it meets `PHASE5_MIN_FAITHFULNESS_SCORE` and contains no partially-supported, unsupported, or insufficient-evidence claims.

## Provider behavior

If `VERIFIER_API_URL` is omitted, Phase 5 reuses the Phase 4 hosted LLM for verification. A separate verifier endpoint can be configured without changing the graph. Repair uses the main Phase 4 LLM so repaired output follows the same CRR schema and transformation instructions.

## Deterministic checks

The deterministic layer is intentionally conservative. It can hard-fail clear conflicts such as `42%` in a claim when the cited evidence contains a different percentage. Unmatched named entities/identifiers are warnings unless the semantic verifier also finds a support problem. This avoids relying on brittle English-only NER heuristics in a multilingual system.

## Run

```bash
python -m scripts.run_phase5 \
  --user-id user-123 \
  --session-id session-1 \
  --file report.pdf \
  --query "Turn the report into an executive summary" \
  --mode transform \
  --artifact-type executive_summary
```
