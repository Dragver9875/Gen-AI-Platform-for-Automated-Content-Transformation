# Phase 9 — Evaluation and Metrics

Phase 9 is implemented before the deployment layer so quality baselines exist before API/UI/queue infrastructure is added.

## Metric families

- **Retrieval:** Precision@K, Recall@K, HitRate@K, MRR, nDCG@K.
- **Reference text:** normalized exact match, token precision/recall/F1, ROUGE-L precision/recall/F1, length ratio.
- **Grounding:** Phase 5 faithfulness, supported/partial/unsupported/insufficient claim rates, mean verifier confidence, repair attempts.
- **Artifacts:** generation success/failure/source-only rates, non-empty output rate, requested-format coverage.
- **Runtime/provider telemetry:** end-to-end latency, provider request/error/retry counts, p50/p95 API latency, token usage when exposed by providers, and optional price-book cost estimation.

## Dataset format

JSON or JSONL cases can include only the ground truth available. `relevant_chunk_ids`, `reference_text`, and `expected_formats` are optional; their metric families are omitted when not supplied.

## Run

```bash
python -m scripts.run_evals --dataset scripts/eval_data/sample.json --output eval_reports/latest.json
```

Optional pricing:

```json
{
  "llm": {"input_per_million": 0.0, "output_per_million": 0.0},
  "image_generation": {"per_request": 0.0}
}
```

Pass it using `--pricing-json pricing.json`. Pricing is deliberately external because hosted-provider prices change.
