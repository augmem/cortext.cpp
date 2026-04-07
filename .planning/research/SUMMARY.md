# Project Research Summary

**Project:** cortext
**Domain:** Local-first multimodal memory engine for augmem.ai
**Researched:** 2026-04-07
**Confidence:** MEDIUM-HIGH

## Executive Summary

Cortext should be treated as an embeddable local-first memory engine, not as a chat product. The research is consistent on the core shape: keep the existing C++/SQLite foundation, assign one runtime per job instead of forcing a single universal inference runtime, and restructure the internals into a fast online path plus a durable background-maintenance path. Experts build this class of system around durable evidence, hybrid retrieval, explicit memory lifecycles, and strong observability because "has memory" is no longer the differentiator; correctness, attribution, and explainability are.

The recommended runtime direction is clear. Keep SQLite as the source of truth, expand it with WAL, FTS5, sqlite-vec, and durable job/explanation tables, use ONNX Runtime as the primary portable runtime for encoders and multimodal models, use llama.cpp as the default GGUF text runtime, promote sherpa-onnx to the default realtime speech stack, and keep LiteRT only for targeted Gemma/mobile/NPU deployments. Architecturally, the next milestone should formalize internal boundaries for ingest, speech, jobs, query/explanations, and runtime policy while preserving the public `Cortext` facade and C API.

The biggest risks are not infrastructure choices but memory-quality failures that look acceptable in demos: destructive consolidation that loses evidence, dense-only retrieval that misses exact facts and corrections, stale facts that out-rank newer truths, speaker-attribution errors that permanently pollute identity memory, silent fallback paths that hide degraded quality, and hot-path latency spikes from background work leaking into interactive flows. The mitigation is equally consistent across the research: preserve evidence and provenance, make retrieval hybrid and explainable, add explicit version/correction semantics, gate identity-affecting voice writes on attribution confidence, expose degraded modes in telemetry, and keep heavy maintenance out of the synchronous path.

## Key Findings

Detailed inputs: [STACK.md](./STACK.md), [FEATURES.md](./FEATURES.md), [ARCHITECTURE.md](./ARCHITECTURE.md), [PITFALLS.md](./PITFALLS.md)

### Recommended Stack

The stack recommendation is an explicit runtime ownership model, not a replatform. Cortext already has the right embeddable shape. The work is to clarify defaults, remove overlap, and make the engine easier to tune and inspect.

**Core technologies:**
- `C++20` + `CMake` + stable C API: keep the embeddable engine core and preserve current public integration boundaries.
- `SQLite` + WAL + migrations + `sqlite-objstore`: keep one durable local-first memory substrate instead of splitting truth across multiple stores.
- `sqlite-vec`: keep as the embedded dense-retrieval layer, but treat it as a scoped internal dependency because upstream is still pre-v1.
- `FTS5`/BM25: add lexical retrieval so recall is hybrid rather than vector-only.
- `ONNX Runtime`: keep as the primary portable runtime for embeddings, multimodal encoders, and hardware-accelerated on-device inference.
- `llama.cpp`: keep as the default GGUF runtime for local text generation, reranking, summarization, and lightweight reasoning.
- `sherpa-onnx`: promote to the default realtime speech stack for streaming ASR, VAD, diarization, speaker ID, and optional TTS.
- `whisper.cpp`: retain as offline/fallback ASR and benchmarking baseline, not the default speech path.
- `LiteRT` / `LiteRT-LM`: keep only as a targeted path for Gemma and mobile/NPU-optimized shipping targets.
- `OpenTelemetry` + local SQLite trace tables: expand telemetry into retrieval-native observability with persisted explanation and degraded-mode artifacts.

**Critical runtime direction:**
- Use capability ownership by job: ONNX for encoders, llama.cpp for GGUF text tasks, sherpa-onnx for realtime voice, LiteRT only where hardware-specific packaging makes it worth the extra path.
- Do not add a remote-first vector database or standardize around daemon-style model orchestration.
- Do not make ONNX Runtime GenAI or LiteRT the universal runtime contract for the whole engine.

### Expected Features

The feature research separates table stakes from actual differentiation. Cortext needs to harden the baseline memory-engine contract before chasing more ambitious graph or adaptive-memory behaviors.

**Must have (table stakes):**
- Scoped memory primitives: `add`, `get`, `search`, `update`, `delete`, `list/history`.
- Explicit namespaces for `user`, `session`, `agent`, and shared/group memory.
- Hybrid retrieval with semantic search, lexical signals, metadata filters, and recency controls.
- Prompt-ready recall plus structured/raw recall for apps, agents, UI, and analytics.
- Memory correction lifecycle with dedupe, merge, invalidate, update, and delete behavior.
- Multimodal ingestion into one canonical memory schema with source anchors.
- Speaker/session attribution for voice and multi-party workflows.
- Privacy/governance surfaces for erasure, export, auditability, and scope isolation.
- Retrieval inspectability with provenance, confidence, and ranking rationale.

**Should have (competitive differentiators):**
- Temporal or bitemporal memory semantics so new facts can supersede old ones cleanly.
- Budget-aware working-memory hydration for different consumers and latency/token budgets.
- Cross-modal recall and reverse lookup across text, image, and audio evidence.
- Realtime voice memory with incremental correction instead of batch-only transcript memory.
- Shared memory layers with permissions and mutability rules.
- Explanation-grade provenance and replay.
- Knowledge-graph augmentation over memory records.
- Maintenance surfaces for consolidation, decay, and replay outside the hot path.

**Defer (v2+ until the base is cleaner):**
- Full temporal graph reasoning and rich multi-hop graph retrieval.
- Highly adaptive salience-based write policy.
- Full explanation-grade replay across graph paths and deep maintenance automation.

### Architecture Approach

The architecture recommendation is a two-speed engine built around SQLite as both the canonical store and the durable coordination backbone. Keep the public facade stable, but internally separate capture and retrieval from heavier extraction/consolidation work. Retrieval, speech, background jobs, and integrations should communicate through durable records and stable interfaces, not direct coupling or example-level orchestration.

**Major components:**
1. Compatibility facade: preserve `Cortext` and C API behavior while routing into refactored internals.
2. Capture adapters: normalize text, image, and audio into a shared signal envelope with source IDs, timestamps, modality metadata, and correlation IDs.
3. Realtime speech service: own VAD, chunking, ASR partial/final events, diarization, barge-in, and playback control as engine-private infrastructure.
4. Online memory pipeline: run the low-latency path for encode, retrieve, working-memory updates, and minimal durable writes.
5. SQLite store + durable job queue: persist canonical memory state, signal events, jobs, attempts, explanation artifacts, and model-run metadata.
6. Background workers: execute consolidation, extraction, graph/fact refresh, and maintenance in bounded batches.
7. Query + explanation layer: return ranked memories with score breakdowns, provenance, and drill-down handles through stable read models.
8. Runtime registry and execution lanes: centralize model selection and scheduling into `low_latency` and `background` lanes.

### Critical Pitfalls

1. **Destructive compression replacing evidence** — keep raw evidence, facts, and summaries as separate retrievable layers; never let summaries become the only durable truth.
2. **Dense-only retrieval optimizing similarity instead of usefulness** — ship hybrid retrieval with lexical, dense, temporal, and fact/graph signals plus explicit score decomposition.
3. **No version semantics for changing facts** — add supersession, conflict, observed/valid timestamps, and correction-chain tests before long-lived memory is trusted.
4. **Speaker-attribution errors polluting durable identity memory** — store speaker confidence separately, delay profile-affecting writes until attribution stabilizes, and test overlap/barge-in fixtures.
5. **Hidden degraded paths masking quality loss** — convert silent fallbacks into explicit degraded-mode states with counters, traces, and regression tests.
6. **Realtime path absorbing maintenance work** — enforce stage-level latency budgets and move summarization, extraction, graph refresh, and compaction to background workers.
7. **Evaluation rewarding demos but missing real failures** — keep long-horizon harnesses and add scenario suites for corrections, abstention, privacy, overlap, and stale-memory conflicts.
8. **Persistent memory leaking across contexts** — add task/context gating, redact logs by default, and evaluate contextual-integrity violations explicitly.

## Implications for Roadmap

Based on the combined research, the roadmap should start by hardening the engine's internal control plane and retrieval correctness. Do not lead with graph intelligence or richer voice behavior before the online path, background path, and explanation surfaces are stable.

### Phase 1: Two-Speed Core Refactor
**Rationale:** Everything else depends on separating low-latency online work from heavy background maintenance without changing the public API.
**Delivers:** Private `ingest`, `realtime`, `jobs`, `query`, and `runtime` boundaries; stage-specific pipeline builders; SQLite signal/job tables; standardized telemetry contracts; stable public facade preserved.
**Addresses:** Maintenance surfaces, retrieval inspectability foundation, runtime ownership clarity.
**Avoids:** Hot-path latency spikes, silent degraded paths, example-layer orchestration becoming product architecture.

### Phase 2: Explainable Hybrid Retrieval + Evidence Preservation
**Rationale:** Retrieval quality and inspectability are the most important product-facing correctness gap and should be fixed before broader app/agent surfaces are expanded.
**Delivers:** FTS5/BM25 lane, hybrid ranker, evidence-level indexing, provenance-rich retrieval artifacts, score decomposition, read-model/query APIs, degraded-mode telemetry.
**Addresses:** Hybrid retrieval, prompt-ready plus structured recall, inspectability, evidence preservation.
**Uses:** SQLite, sqlite-vec, FTS5, OpenTelemetry, query/explanation layer.
**Avoids:** Destructive compression, dense-only failure modes, opaque ranking behavior.

### Phase 3: Scoped Memory Lifecycle + Temporal Correctness
**Rationale:** Once retrieval is explainable, the engine needs stable identity, scope, and correction semantics so newer truths reliably beat older ones.
**Delivers:** Namespaces for `user`/`session`/`agent`/shared memory, stable IDs, `update`/`invalidate`/`delete`/`history` APIs, supersession/conflict metadata, freshness-aware ranking, export/erasure foundations.
**Addresses:** Scoped memory primitives, correction lifecycle, governance controls, baseline temporal semantics.
**Implements:** Canonical memory schema, lifecycle state, temporal metadata in the store and ranker.
**Avoids:** Single-pool contamination, stale facts outranking corrections, unfixable append-only memory.

### Phase 4: Realtime Voice + Speaker-Safe Ingestion
**Rationale:** Voice is a major augmem.ai differentiator, but it should land only after queueing, provenance, and correction semantics are in place.
**Delivers:** Engine-private realtime speech service, sherpa-onnx default speech path, streaming ASR/VAD/diarization integration, confidence-gated durable writes, replay fixtures for overlap and interruption, thin example/UI clients.
**Addresses:** Speaker/session attribution, realtime voice memory, multimodal normalization, incremental correction for speech-derived memory.
**Uses:** sherpa-onnx, ONNX Runtime, low-latency execution lane, signal journal.
**Avoids:** Identity pollution from diarization errors, voice latency regressions, speech logic trapped in example code.

### Phase 5: Advanced Memory Surfaces
**Rationale:** Only after correctness and latency foundations are stable should Cortext push into the differentiators that make augmem.ai meaningfully better than a generic memory layer.
**Delivers:** Budget-aware working-memory hydration, cross-modal recall, selective graph augmentation, adaptive write-policy experiments, stronger contextual privacy gates, richer maintenance commands.
**Addresses:** Competitive differentiators from the feature research.
**Implements:** Query-layer shaping, graph/fact refresh workers, policy-aware retrieval gating.
**Avoids:** Overbuilding on unstable foundations, shipping sophisticated recall without correctness or privacy controls.

### Phase Ordering Rationale

- Phase 1 comes first because queueing, runtime ownership, and telemetry are architectural prerequisites for every later phase.
- Phase 2 comes before lifecycle and voice work because retrieval explainability is the fastest route to correctness, tuning, and safe downstream integration.
- Phase 3 comes before Phase 4 because scoped identities and correction semantics are required before voice-derived memories can safely become durable.
- Phase 4 waits until the engine can tolerate realtime ingestion without running consolidation or repair work inline.
- Phase 5 intentionally groups advanced differentiators after the base engine can preserve evidence, explain itself, and maintain latency budgets.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 3:** Temporal/bitemporal semantics and conflict-resolution policy need explicit design choices beyond standard CRUD/history behavior.
- **Phase 4:** Device benchmarking is still needed for speech-runtime tuning, diarization confidence thresholds, and execution-provider choices across target hardware.
- **Phase 5:** Graph augmentation, adaptive write policy, and privacy/contextual-integrity guardrails need focused validation so they do not become expensive or unsafe experiments.

Phases with standard patterns (skip research-phase unless scope changes):
- **Phase 1:** Two-speed refactor, durable SQLite job queues, and OTel correlation are well-documented patterns and already fit the current codebase.
- **Phase 2:** Hybrid retrieval with persisted explanations and score breakdowns follows established patterns and is strongly supported by the research set.
- **Phase 3 base CRUD/scopes work:** Namespaces, stable IDs, and correction APIs are standard; only the deeper temporal semantics need extra research.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | The runtime direction is anchored in official runtime/storage docs and aligns tightly with the existing repo's embeddable local-first constraints. |
| Features | MEDIUM-HIGH | The table-stakes/differentiator split is consistent across multiple current memory-engine ecosystems, but exact augmem.ai product priorities still need validation. |
| Architecture | HIGH | The two-speed engine, SQLite-backed job coordination, runtime lanes, and explanation read models are strongly supported by both the repo shape and official platform docs. |
| Pitfalls | MEDIUM-HIGH | The failure modes are well-supported by current benchmarks/papers and match observed Cortext stress points, but exact mitigation thresholds still require implementation-time testing. |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- **Speech/runtime benchmarking:** Validate sherpa-onnx default choices, whisper.cpp fallback boundaries, and ONNX execution-provider selection on target augmem.ai hardware before locking runtime policy.
- **Temporal correctness policy:** Decide how `observed-at`, `asserted-at`, `valid-from`, `superseded-by`, and unresolved conflicts should interact in ranking and API responses.
- **Privacy/context gating:** Define augmem.ai-specific contextual-integrity rules early so retrieval does not over-surface sensitive memory in the name of personalization.
- **sqlite-vec risk management:** Treat sqlite-vec as an embedded dependency with explicit scope and regression coverage because the project is still pre-v1.
- **Integration contract detail:** The research is engine-strong but not app-contract-specific; downstream bindings and augmem.ai product surfaces may need additional shaping once roadmap phases are chosen.

## Sources

### Primary (HIGH confidence)
- Internal project context: `.planning/PROJECT.md`, `.planning/codebase/CONCERNS.md`, `.planning/codebase/TESTING.md`
- SQLite WAL documentation: https://sqlite.org/wal.html
- OpenTelemetry C++ / signals / GenAI semantic conventions: https://opentelemetry.io/docs/languages/cpp/ , https://opentelemetry.io/docs/concepts/signals/ , https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
- ONNX Runtime documentation: execution providers, QNN EP, CoreML EP, I/O binding, quantization, reduced operator config, logging/tracing, GenAI docs
- LiteRT documentation: overview, CMake builds, delegates
- llama.cpp README
- whisper.cpp README

### Secondary (MEDIUM confidence)
- sherpa-onnx documentation and README
- LangChain long-term memory docs and changelog
- Zep concepts, sessions, episodes, and retrieval docs
- Mem0 open-source and platform feature docs
- Letta shared memory blocks docs
- Qdrant multimodal retrieval docs
- Soniox diarization concepts docs

### Benchmark / Research Inputs (MEDIUM confidence)
- LongMemEval
- LoCoMo
- EvolMem
- CIMemories
- Overlap-aware low-latency online speaker diarization
- WhisperX practitioner-facing diarization caveats

---
*Research completed: 2026-04-07*
*Ready for roadmap: yes*
