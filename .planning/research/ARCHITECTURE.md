# Architecture Patterns

**Domain:** Local-first multimodal memory engine for augmem.ai  
**Researched:** 2026-04-07  
**Overall confidence:** HIGH for storage/observability/runtime structure, MEDIUM for speech-runtime selection details that still need device benchmarking.

## Executive Summary

Keep the public `Cortext` facade and C API stable, but restructure the internals into a two-speed engine:

1. A **small synchronous online path** for ingesting signals, retrieving context, updating working memory, and returning a result fast enough for live chat and voice.
2. A **durable background path** for consolidation, extraction, summarization, graph maintenance, and other heavier jobs that should not sit on the user-facing critical path.

The core design choice is to use SQLite as both:

- the authoritative local memory store, and
- the durable coordination backbone for queued work, replay, and inspection.

That means retrieval, consolidation, speech, and agent/app integrations do not call each other directly. They communicate through stable internal boundaries and durable records:

```text
Capture/Adapters
  -> Signal Journal
  -> Online Memory Pipeline
  -> Retrieval Read Model + Working Memory
  -> App/Agent Response

Signal Journal / Online Pipeline
  -> Background Job Queue
  -> Consolidation + Extraction Workers
  -> Graph/Fact Updates
  -> Read Model Refresh / Explanation Artifacts
```

This is the right brownfield move for cortext because the repo already has the hard parts of a local engine:

- stable facade and FFI surface
- composable operation pipeline
- SQLite transactions and WAL helpers
- OpenTelemetry wrapper
- multiple model runtimes

The missing piece is not a new engine. It is an internal control plane that lets realtime capture, retrieval, consolidation, and inference run at different cadences without being entangled.

## Architecture Diagram

```text
                   +---------------------------+
                   |   App / Agent Adapters    |
                   | chat, bindings, tools     |
                   +-------------+-------------+
                                 |
                                 v
                   +---------------------------+
                   | Compatibility Facade      |
                   | Cortext / C API / JSON    |
                   +------+------+-------------+
                          |      |
          process/retrieve|      |subscribe/query/explain
                          |      v
                          |  +---------------------------+
                          |  | Query + Explanation Layer |
                          |  | retrieval read model      |
                          |  +-------------+-------------+
                          |                ^
                          v                |
 +------------------+   +------------------+------------------+
 | Capture Adapters |-> | Online Memory Pipeline             |
 | text/image/audio |   | encode, score, retrieve, WM update |
 +--------+---------+   +------------------+------------------+
          |                                   |
          v                                   v
 +------------------+               +--------------------------+
 | Realtime Speech  |               | Durable Job / Event Log  |
 | VAD/ASR/diarize  |               | SQLite tables + outbox   |
 +--------+---------+               +------------+-------------+
          |                                      |
          v                                      v
 +------------------+               +--------------------------+
 | Model Runtime    |<------------->| Background Workers       |
 | ASR/enc/sum/ext  |               | consolidate/extract/GC   |
 +--------+---------+               +------------+-------------+
          |                                      |
          +------------------+-------------------+
                             v
                   +---------------------------+
                   | SQLite Memory Store       |
                   | memories/facts/edges/jobs |
                   +---------------------------+
```

## Component Boundaries

| Component | Responsibility | Communicates With | Repo implication |
|-----------|----------------|-------------------|------------------|
| Compatibility facade | Preserve `Cortext` and C API behavior while routing work into the new internals | Online pipeline, query layer, job queue | Keep `include/cortext/*` stable; treat internal refactor as private |
| Capture adapters | Normalize text, image, and audio inputs into a shared signal envelope with timestamps, source IDs, modality metadata, and correlation IDs | Realtime speech, online pipeline, job log | Add a private `src/ingest/` or `src/runtime/ingest/` instead of spreading capture logic across examples |
| Realtime speech service | Own microphone session state, VAD, chunking, partial/final ASR, diarization, barge-in events, and TTS playback control | Capture adapters, model runtime, online pipeline | Move `examples/chat/voice_session.*` behavior behind a library-private service; leave the example as UI only |
| Model runtime layer | Expose capability-based interfaces for ASR, embeddings, summarization, extraction, diarization, TTS, and image/text encoding | Online pipeline, speech service, background workers | Introduce a runtime registry and execution lanes instead of letting each caller choose models ad hoc |
| Online memory pipeline | Fast path for encode, retrieval, working-memory mutation, light persistence, and interrupt decisions | Model runtime, SQLite store, query layer, job queue | Keep `SignalProcessor`, but split the giant root pipeline into named stage builders |
| Background workers | Run heavy consolidation, summarization, extraction, graph/fact refresh, backfill, and maintenance jobs out of band | Job queue, model runtime, SQLite store, query layer | Add `src/jobs/` or `src/background/` with deterministic worker entrypoints |
| Query + explanation layer | Return ranked memories plus why they were surfaced, what scores contributed, and what provenance backs them | SQLite store, online pipeline, background workers, adapters | Add read-model APIs instead of reconstructing explanations inside apps |
| SQLite memory store | Persist canonical memory state plus queued jobs, checkpoints, explanation artifacts, and model-run metadata | All internal services | Keep SQLite; evolve schema rather than replacing it |
| Observability layer | Correlate spans, logs, and metrics across capture, inference, retrieval, consolidation, and integrations | Every component | Build on existing telemetry wrapper; standardize names/attributes rather than ad hoc logging |
| Integration adapters | Bindings, chat app, tools, agents, and future app hooks | Compatibility facade, query layer, optional event subscriptions | Keep adapters thin; no business logic in examples |

## Data Flow

### 1. Realtime voice flow

1. Microphone frames enter the realtime speech service.
2. VAD segments audio into short utterance windows; diarization attaches a speaker ID if available.
3. Streaming ASR produces partial and final transcript events.
4. Finalized utterance segments are normalized into a signal envelope and appended to the local signal journal/job log.
5. The online memory pipeline runs the fast retrieval/working-memory path only.
6. The query layer returns retrieved memories plus explanation metadata to the caller.
7. Heavy work such as summarization, extraction, or graph maintenance is enqueued for background workers.

**Architecture rule:** the voice path must never block on consolidation or deep summarization.

### 2. Text and image flow

1. Text or image enters through the facade.
2. A capture adapter adds shared metadata: source, time bounds, correlation ID, modality, session ID.
3. The online pipeline encodes, retrieves, updates short-lived state, and persists minimal canonical changes.
4. If thresholds are crossed, the pipeline emits background jobs instead of doing all downstream work inline.

### 3. Background consolidation flow

1. The online path writes durable consolidation or extraction jobs into SQLite.
2. Background workers lease jobs, execute in bounded batches, and commit results in short transactions.
3. Workers write summaries, labels, facts, graph edges, and refreshed explanation artifacts.
4. Query-side materialized views or caches are refreshed or invalidated.
5. Job outcome and model-run telemetry are recorded for later debugging.

### 4. App and agent retrieval flow

1. An app or agent asks for context with an explicit intent and token/latency budget.
2. The query layer reads prepared retrieval views plus explanation artifacts from SQLite.
3. The adapter receives:
   - ranked memories
   - concise attribution/explanation fields
   - optional provenance handles for drill-down
4. The adapter formats those results for chat, voice, tools, or bindings without changing core ranking logic.

## Patterns To Follow

### Pattern 1: Two-speed execution

**What:** Separate the engine into an online decision loop and a background maintenance loop.  
**When:** Immediately. This is the highest-leverage refactor.  
**Why:** Retrieval and interruption decisions are latency-sensitive; consolidation and extraction are throughput-sensitive.

For cortext, that means:

- `ProcessText` / `ProcessAudio` / `ProcessImage` stay synchronous
- consolidation jobs become queued and lease-based
- retrieval explanations are written as artifacts, not reconstructed from logs later

### Pattern 2: SQLite-backed durable outbox/job queue

**What:** Use SQLite tables for `signal_events`, `background_jobs`, `job_attempts`, and optionally `retrieval_explanations` / `model_runs`.  
**When:** Before adding more realtime voice and agent features.  
**Why:** A local-first engine needs replayable, inspectable coordination, not in-memory callbacks.

This fits SQLite well, but respect WAL constraints:

- WAL improves reader/writer overlap and allows checkpoints to be moved to idle time or a separate thread/process.
- WAL still permits only one writer at a time.
- Therefore background workers must use short write transactions and avoid long-lived write locks.

### Pattern 3: Capability-based model runtime with execution lanes

**What:** Hide concrete runtimes behind capabilities like `StreamingAsr`, `OfflineAsr`, `EmbeddingEncoder`, `Summarizer`, `Extractor`, `SpeakerDiarizer`, `TtsSynthesizer`.  
**When:** Before optimizing performance.  
**Why:** Performance tuning only stays modular if scheduling and model selection are not spread across apps.

Use at least two execution lanes:

- `low_latency`: streaming ASR, embedding, retrieval-critical inference
- `background`: summarization, extraction, backfill, rescoring

This is where runtime-specific optimization belongs:

- ONNX Runtime quantization and reduced-operator builds for hot-path models
- platform execution providers where device-specific benchmarks prove a win
- whisper.cpp as a fallback or quality mode, not the default interactive path

**Recommendation:** Prefer sherpa-onnx as the primary realtime speech stack because it explicitly supports offline streaming ASR, VAD, speaker identification/segmentation, and TTS on top of ONNX Runtime. Keep whisper.cpp available for non-streaming or fallback transcription modes.

### Pattern 4: Read models for retrieval and explainability

**What:** Maintain query-oriented artifacts separate from write-oriented tables.  
**When:** Before large app/agent integration work.  
**Why:** Agents and UIs need stable explanation payloads, not raw internal processor state.

Useful read-model artifacts:

- retrieval rank breakdowns
- top contributing scores/features
- signal provenance IDs
- fact/graph evidence links
- last surfaced / last used metadata

### Pattern 5: Observability contracts, not just logging

**What:** Standardize spans, metrics, and logs around a shared request/job correlation model.  
**When:** In parallel with queue extraction.  
**Why:** Without correlation, realtime performance work and retrieval debugging become guesswork.

Recommended span families:

- `cortext.capture.*`
- `cortext.speech.*`
- `cortext.model.*`
- `cortext.retrieve.*`
- `cortext.consolidate.*`
- `cortext.job.*`
- `cortext.integration.*`

Recommended required attributes:

- `conversation.id` or session/thread equivalent
- `signal.id`
- `job.id`
- `model.name`
- `model.runtime`
- `modality`
- `source_id`
- `latency_ms`
- `queue_delay_ms`
- `retrieval_count`
- `consolidation_mode`

### Pattern 6: Split pipeline assembly by stage, not one giant root

**What:** Replace a monolithic `BuildPipelineRoot()` with named builders.  
**When:** Early in the refactor.  
**Why:** The current operation set is composable, but one large root makes stage ownership blurry.

Suggested private builders:

- `BuildOnlineIngestPipeline()`
- `BuildOnlineRetrievalPipeline()`
- `BuildProbePipeline()`
- `BuildConsolidationPipeline()`
- `BuildMaintenancePipeline()`

That preserves the operation pattern already in the repo while making future performance work more targeted.

## Anti-Patterns To Avoid

### Anti-Pattern 1: Putting voice orchestration in the example layer

**What:** Letting `examples/chat` remain the owner of capture, diarization, interrupt, and playback coordination.  
**Why bad:** Voice becomes tied to one UI instead of becoming a reusable engine subsystem.  
**Instead:** Move the orchestration into a library-private realtime module and keep examples as thin clients.

### Anti-Pattern 2: Running consolidation inline with the user-facing transaction

**What:** ASR finalization or text processing directly triggering summarization/extraction/graph work in the same critical path.  
**Why bad:** Latency spikes, writer contention, and poor barge-in behavior.  
**Instead:** Commit the minimal online result, enqueue heavy work, and return.

### Anti-Pattern 3: Treating observability as debug-only logging

**What:** Logs without stable correlation IDs, queue metrics, or per-model timings.  
**Why bad:** You can see failures, but not explain why the engine was slow or why a memory surfaced.  
**Instead:** Use structured spans/logs/metrics with stable naming and attributes across the whole path.

### Anti-Pattern 4: Letting every caller choose runtimes directly

**What:** App code picking whisper, sherpa, Gemma, Phi, or ONNX settings ad hoc.  
**Why bad:** Scheduling, tuning, and fallback policy become impossible to reason about.  
**Instead:** Centralize runtime policy in a model registry and execution-lane scheduler.

### Anti-Pattern 5: One-table-fits-all query access

**What:** Making apps reconstruct retrieval explanations by joining raw storage tables on demand.  
**Why bad:** Slow, inconsistent, and hard to keep stable for bindings.  
**Instead:** Maintain explicit read models or explanation artifacts.

## Scalability Considerations

| Concern | Near term: single device, single session | Mid term: many sessions / agents on one device | Longer term: large local history |
|---------|-------------------------------------------|-----------------------------------------------|----------------------------------|
| Realtime latency | Keep the online path minimal and synchronous | Introduce execution lanes and queue budgets | Precompute read artifacts; prune or tier cold work |
| SQLite contention | One short writer path, WAL enabled, checkpoint on idle | Separate read connections from write scheduling | Batch maintenance writes; monitor WAL growth and lease times |
| Speech throughput | Streaming ASR + VAD only on hot path | Reuse diarization and ASR sessions; avoid model churn | Tier models by device class and session priority |
| Retrieval explainability | Persist reason codes per retrieval | Add explanation read models for agents/UI | Periodically compact explanation artifacts |
| Model cost | Quantize hot-path models first | Device-specific execution-provider selection | Reduced operator builds and model packaging by target |

## Suggested Build Order Implications

1. **Extract internal boundaries before chasing more performance**
   - Create private modules for `ingest`, `realtime speech`, `jobs`, `query`, and `runtime`.
   - Keep the public API stable.
   - Split `BuildPipelineRoot()` into named stage builders.

2. **Add durable job orchestration**
   - Introduce SQLite-backed job/outbox tables and a worker lease model.
   - Move consolidation, extraction, and heavy graph work off the online path.
   - Add replayable signal-event records for debugging.

3. **Make retrieval explainable as a first-class product surface**
   - Persist retrieval breakdowns and provenance artifacts.
   - Add query APIs that return both memories and explanation fields.
   - This should happen before deeper app/agent integration, otherwise integrations will hard-code unstable logic.

4. **Move voice from example code into engine-private infrastructure**
   - Promote shared realtime speech orchestration into the library.
   - Keep UI code in `examples/chat` thin.
   - This unlocks reuse across future agents and apps.

5. **Then optimize on-device inference**
   - Quantize hot-path ONNX models.
   - Use reduced-operator builds for constrained targets.
   - Profile model sessions and queue delay separately.
   - Only after the scheduler exists should device-specific execution-provider tuning begin.

6. **Only after those pieces land, widen integration surfaces**
   - Add richer subscriptions, app hooks, and agent-oriented retrieval APIs.
   - Otherwise API shape will be driven by temporary internals and will churn.

## Concrete Brownfield Recommendations For This Repo

- Keep `include/cortext/cortext.hpp` and `include/cortext/capi.h` stable.
- Keep `SignalProcessor`, but reposition it as the online decision engine.
- Replace one large private pipeline builder in `src/cortext.cpp` with stage-specific private builders.
- Add private directories roughly like:
  - `src/ingest/`
  - `src/realtime/`
  - `src/jobs/`
  - `src/query/`
  - `src/runtime/`
- Move voice session orchestration out of `examples/chat/voice_session.*` and leave that directory as a UI shell.
- Extend the SQLite schema with:
  - durable signal events
  - background jobs and attempts
  - retrieval explanation artifacts
  - model run/profiling metadata
- Standardize telemetry naming/attributes before more feature work.

## Sources

- SQLite WAL documentation: https://sqlite.org/wal.html  
  Confidence: HIGH. Used for checkpointing and single-writer guidance.
- OpenTelemetry signals: https://opentelemetry.io/docs/concepts/signals/  
  Confidence: HIGH. Used for traces/metrics/logs structuring.
- OpenTelemetry GenAI spans: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/  
  Confidence: MEDIUM-HIGH. Useful for retrieval/model span naming, but the spec is still marked developmental.
- ONNX Runtime reduced operator config: https://onnxruntime.ai/docs/reference/operators/reduced-operator-config-file.html  
  Confidence: HIGH. Used for smaller on-device runtime packaging guidance.
- ONNX Runtime quantization: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html  
  Confidence: HIGH. Used for hot-path model optimization guidance.
- ONNX Runtime logging and tracing: https://onnxruntime.ai/docs/performance/tune-performance/logging_tracing.html  
  Confidence: HIGH. Used for model-runtime observability guidance.
- sherpa-onnx project documentation/README: https://github.com/k2-fsa/sherpa-onnx  
  Confidence: MEDIUM-HIGH. Used for offline streaming ASR, VAD, speaker, and TTS capability assessment.
- whisper.cpp project README: https://github.com/ggml-org/whisper.cpp  
  Confidence: MEDIUM-HIGH. Used for fallback/offline ASR packaging and runtime characteristics.
