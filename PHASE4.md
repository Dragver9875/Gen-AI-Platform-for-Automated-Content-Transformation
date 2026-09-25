# Phase 4: Grounded Generation and CRR

## Scope

Phase 4 consumes the Phase 3 retrieval handoff and produces a format-independent Canonical Response Representation (CRR). It does **not** perform semantic factuality verification or artifact rendering; those belong to Phases 5 and 6.

## Added files

- `providers/llm.py` — hosted SLM adapter (`openai` and `hf` endpoint styles)
- `generation/crr.py` — transformation config, CRR, section-digest schemas
- `generation/prompts.py` — source-grounded structured-output prompts
- `generation/service.py` — QA and hierarchical transformation generation
- `agents/phase4_nodes.py` — LangGraph generation nodes
- `agents/phase4_graph.py` — Phase 1–4 graph and session-aware façade
- `app/phase4.py` — Phase 4 orchestrator factory
- `scripts/run_phase4.py` — CLI runner
- `tests/test_phase4_generation.py` — Phase 4 regression tests

## Data flow

```text
Phase 3 context
    │
    ├── QA ───────────────→ one grounded CRR call
    │
    └── Transformation
          ↓
       section groups
          ↓
       section digests
          ↓
       final CRR synthesis
```

## Phase boundary

Phase 4 guarantees:

1. Schema-valid CRR JSON.
2. Source-grounded prompting.
3. Chunk-level evidence references.
4. Unknown/nonexistent evidence IDs are removed.
5. Long-document transformations use hierarchical section coverage.

Phase 4 does **not** decide whether a cited chunk truly entails a claim. Phase 5 must perform that verification and repair unsupported claims.
