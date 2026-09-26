# Image routing + Phase 5 structured-output fix

## Runtime failures addressed

1. Hosted SigLIP zero-shot classification is not consistently available via HF Inference Providers. SigLIP is now optional and disabled by default (`SIGLIP_ENABLED=false`). The document-page heuristic runs first; ambiguous images use the already-required VLM for routing.
2. `HostedLLMProvider.generate_json()` previously fell back only on HTTP errors. A 200 response containing empty or malformed JSON now triggers the next mode: `json_schema -> json_object -> prompt_only`.
3. OpenAI-compatible response extraction now supports list-based content blocks and tool-call function arguments.
4. If every semantic-verifier structured-output mode fails, verification degrades conservatively to `insufficient_evidence` instead of putting the entire graph in `error`. Repair is skipped for a degraded verifier and the graph finalizes with issues.

## Environment

Recommended local defaults:

```env
SIGLIP_ENABLED=false
SIGLIP_API_URL=
VLM_API_URL=
LLM_API_URL=
VERIFIER_API_URL=
```

`HF_TOKEN` is still used by the VLM and main LLM through the HF router.
