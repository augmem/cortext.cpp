# Roadmap

## Current Priority

Ship the current Cortext release.

Near-term work should favor:
- release readiness
- stability
- benchmark validation
- documentation cleanup
- product-path hardening

New retrieval architecture work is deferred until after shipping.

## Post-Ship Candidate: ColBERT-Style Late Interaction

Use a Liquid AI CPU-friendly ColBERT model as a text reranker, not as a full replacement for the primary embedding stack.

Rationale:
- improves exact text retrieval precision
- preserves the current dense embedding pipeline for recall
- avoids forcing a full token-matrix storage/index redesign before release

### Phase 1: Retrieval Reranker

Primary insertion point:
- `src/operations/graph_retrieval.cpp`

Planned integration:
- keep current dense retrieval and graph expansion for recall
- rerank top-N eligible text candidates after `filter_candidates(...)`
- apply reranked relevance before `select_diversified(...)`
- preserve existing association/label seeding and diversification behavior
- fall back to current dense scoring for non-text or missing-text candidates

### Deferred Semantic Evidence Scoring

Do not plan label, relation, or summary reranking until a production semantic
evidence subsystem is explicitly reintroduced. The v1 runtime intentionally has
no semantic extraction or summarization pass to hook into.

## Explicitly Deferred

Do not do this before shipping:
- replace the primary embedding model with ColBERT
- make token-space late interaction a hard dependency of the live loop
- replace the multimodal embedding path with a text-only ColBERT stack

## Success Criteria For Post-Ship Evaluation

Measure whether the reranker improves:
- exact entity retrieval
- retrieval semantic overlap
- interrupt precision and recall
- summary/source grounding quality

Do not accept the change if it materially degrades:
- per-turn latency
- write-path simplicity
- multimodal behavior

## Release Positioning

Cortext should be treated as a v1 release candidate.

The current priority is shipping the existing architecture cleanly, documenting
it clearly, and learning from real usage before taking on larger runtime
substitutions.

## Post-v1 Architecture Direction

Post-v1 work is expected to make two major infrastructure moves:

- move the event-driven system to `stateforward/sml.cpp`
- use that change to improve system structure and memory safety
- move inference to `stateforward/emel.cpp` once that library is complete and production-ready

These are follow-on goals. They should not block the v1 release unless the
current architecture shows a concrete release-blocking defect.
