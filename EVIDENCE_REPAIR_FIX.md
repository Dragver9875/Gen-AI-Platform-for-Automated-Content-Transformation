# Evidence Alias + Non-Fatal Repair Fix

## Evidence aliases

The application now assigns short aliases (`E1`, `E2`, ...) to retrieved chunks before any LLM prompt is built. Model-facing Phase 4, Phase 5 verification, and Phase 5 repair prompts use only these aliases. Raw vector/database chunk IDs remain internal and are restored after model output validation.

This prevents long hash-like chunk IDs from being copied incorrectly or hallucinated by the model.

## Non-fatal repair failure

If Phase 5 determines that repair is needed but the repair request fails (for example an HTTP 402/429/5xx or provider outage), the graph:

1. keeps the current CRR unchanged;
2. keeps the failed verification report;
3. records `repair_degraded=true` plus the provider error in verification metadata;
4. increments the repair attempt counter;
5. finalizes as `phase5_complete_with_issues` rather than `error`.

No unresolved claim is promoted to supported and no repair output is fabricated.
