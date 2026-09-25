# Phase 6 — Generic Artifact Generation

Phase 6 extends the verified Phase 5 CRR into concrete artifacts while preserving the existing QA/transform LangGraph topology.

## What changed before Phase 6

The graph's intent routing remains `qa` vs `transform`. Genericization is implemented around it:

- provider capabilities are resolved through `ProviderRegistry`;
- ingestion file families are registered through `IngestionProcessorRegistry`;
- retrieval strategy is configurable (`hybrid`, `dense`, `lexical`);
- verification policy is selected through named profiles;
- CRR is converted to artifact-agnostic `ContentIR`;
- output formats are resolved through `ArtifactRegistry` and `artifacts/specs/default.json`;
- `TaskSpec` provides a service-level request contract without replacing the current intent router.

## Phase 6 flow

```text
verified CRR
   ↓
Content IR
   ↓
Artifact Registry
   ├─ text → model-generated TextArtifactIR → text serializer
   ├─ pdf → model-generated Typst source → Typst compiler
   ├─ pptx → model-generated PresentationIR → editable PPTX serializer
   ├─ infographic/svg → model-generated SVG → SVG serializer
   └─ image → model-generated image prompt → hosted image model API
```

Only successfully verified CRRs are rendered. `phase5_complete_with_issues` does not produce final artifacts.

## Artifact plugins

The LangGraph does not have a separate PDF/PPTX/image branch. One `generate_artifacts` node calls the registry. Aliases/formats are configured in `artifacts/specs/default.json`.

Default formats:

- `text` (`txt`, `md`, `markdown` aliases)
- `pdf`
- `pptx` (`ppt`, `presentation` aliases)
- `svg` (`infographic`, `factual_image` aliases)
- `image` (`creative_image`, `png` aliases)

## PDF

The LLM generates complete Typst source. The serializer only validates and compiles it. If Typst is not available, the generated `.typ` source is preserved and the artifact is returned with `status=source_only` rather than silently falling back to another decoder.

For deployment, install the Typst CLI in the application image and set `PHASE6_TYPST_BINARY` when it is not on `PATH`.

## PPTX

The LLM generates a `PresentationIR`; `python-pptx` serializes that IR into an editable `.pptx`. The serializer does not decide the content, slide ordering, or narrative.

## Factual image / infographic

The LLM emits a complete SVG. Active/external content is rejected before writing the file.

## Creative image

The LLM first creates a `CreativeImageIR` prompt. A hosted image-generation endpoint (for example a hosted FLUX endpoint) returns the actual image bytes. No image-generation weights are loaded by this repo.

## Security boundary

Model-generated artifact code is treated as untrusted:

- Typst rejects imports/includes, file reads and external URLs.
- SVG rejects scripts, `foreignObject`, JavaScript URLs and external hrefs.
- output directories sanitize user/session identifiers to prevent path traversal.

## Run

```bash
python -m scripts.run_phase6 \
  --user-id user-123 \
  --session-id report-session \
  --file report.pdf \
  --query "Convert this into an executive briefing" \
  --mode transform \
  --artifact-type executive_summary \
  --format text \
  --format pdf \
  --format pptx
```
