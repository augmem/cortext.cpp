# cortext

## What This Is

cortext is the memory engine powering augmem.ai for augmenting human and LLM memory. It is a brownfield C++ system that ingests multimodal signals, persists durable memory traces, and resurfaces relevant context for applications, agents, analyses, and realtime chat experiences.

## Core Value

Important context should resurface at the right time for humans and models without requiring manual memory management.

## Requirements

### Validated

- ✓ Process text, audio, and image signals into persisted memories with contextual retrieval — existing
- ✓ Run a configurable F/S/T-driven memory pipeline with working-memory hydration and SQLite-backed persistence — existing
- ✓ Support multiple local model backends and runtime paths for encoding, extraction, and summarization — existing
- ✓ Expose the engine through native APIs, bindings, examples, and benchmark tooling — existing

### Active

- [ ] Improve realtime memory augmentation for both human conversation and LLM/agent workflows
- [ ] Make retrieval surfacing more explainable, inspectable, and tunable in live applications
- [ ] Build a higher-performance on-device speech stack for voice memory capture, speaker separation, and assistant interaction

### Out of Scope

- Managed cloud memory service as the primary product surface — this repo remains engine-first and local-first
- Broad augmem.ai application features that do not improve the core memory engine — they belong above the engine boundary
- Public API churn without explicit approval — the engine must stay usable by existing native and binding consumers

## Context

- The repository already contains a substantial brownfield codebase with a public C++ API, C API, multiple bindings, examples, benchmarks, and a paper/experiment workflow.
- The engine is centered on memory formation, retrieval, consolidation, and working-memory behavior controlled by the three knobs F, S, and T.
- Local model execution already spans ONNX Runtime, LiteRT-LM, GGUF/llama.cpp-style assets, sherpa-onnx, and whisper.cpp, but realtime voice performance and speaker handling remain active areas of churn.
- The immediate product framing is that cortext is the engine behind augmem.ai, so engine quality, observability, and on-device behavior matter more than standalone demo polish.

## Constraints

- **Tech stack**: C++20 with CMake, SQLite, and local model runtimes — the current engine architecture is already in production use and should be evolved, not replaced casually
- **API stability**: Public headers in `include/` and the C API require explicit approval before breaking changes — bindings and examples depend on them
- **Research traceability**: Algorithm and experiment changes must be reflected in `docs/paper/sections/` and the generated manuscript — this repo treats paper evidence as part of the product record
- **Performance**: On-device latency matters, especially for speech and realtime interaction paths — augmem.ai needs memory augmentation that feels live, not batch-oriented

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Keep cortext as a brownfield engine rather than restart from scratch | The repo already contains the core memory pipeline, persistence layer, bindings, experiments, and demos needed to power augmem.ai | — Pending |
| Treat local-first multimodal memory as the product center | The engine must work for both humans and LLMs without depending on a hosted memory backend | — Pending |
| Preserve F/S/T as the main behavioral control surface | Repository guidance and current algorithms already organize behavior around these knobs | — Pending |
| Track planning artifacts in git | This project uses docs, experiments, and roadmap artifacts as durable engineering state | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? -> Move to Out of Scope with reason
2. Requirements validated? -> Move to Validated with phase reference
3. New requirements emerged? -> Add to Active
4. Decisions to log? -> Add to Key Decisions
5. "What This Is" still accurate? -> Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check -> still the right priority?
3. Audit Out of Scope -> reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-07 after initialization*
