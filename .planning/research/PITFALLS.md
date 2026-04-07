# Domain Pitfalls

**Domain:** local-first human + LLM memory engine for augmem.ai
**Researched:** 2026-04-07
**Confidence:** MEDIUM-HIGH

The next mistakes for Cortext to avoid are not generic "pick the wrong vector database" failures. The common failures in this ecosystem are correctness failures hidden behind plausible demos: compressing away evidence, retrieving semantically similar but operationally wrong memories, polluting identity with unstable speaker attribution, doing too much work on the realtime path, and declaring success from narrow benchmarks that miss updates, abstention, and long-horizon drift.

This matters directly to Cortext because the current repo already shows stress in the same places: a large synchronous retrieval path, silent retrieval fallbacks, active churn in voice/speaker code, and strong long-horizon harnesses that are still more valuable than the current automated app/runtime coverage.

## Critical Pitfalls

### Pitfall 1: Destructive Compression Replaces Evidence
**What goes wrong:** The system summarizes or consolidates too early, then later retrieval depends on compressed text that omitted the exact fact, wording, timestamp, or speaker attribution needed for a correct answer.

**Why it happens:** Memory systems optimize for storage and prompt budget first, and treat summaries as a lossless replacement for raw evidence. Recent long-memory work argues for finer-grained session decomposition and fact-augmented indexing precisely because coarse histories and coarse summaries hurt downstream recall.

**Consequences:** Cortext resurfaces plausible but incomplete memory, misses factual details needed for explanation, and cannot justify why a retrieved memory should be trusted.

**Warning signs:**
- Retrieval quality looks better in demo summaries than in evidence-level inspection.
- The system can answer gist questions but fails exact preference, date, or who-said-what questions.
- Consolidation improves token budget while downstream answer accuracy stagnates or drops.
- Users report "that sounds close, but not what I said."

**Prevention:**
- Keep raw evidence and extracted facts as first-class retrievable objects; do not let summaries become the only durable representation.
- Index at multiple granularities: turn, fact/assertion, session, and synthesized summary.
- Make retrieval return provenance by default: source span, speaker, timestamp, and why this item beat alternatives.
- Treat consolidation as an additional view over evidence, not a destructive rewrite.

**Future phase:** `Retrieval Granularity + Evidence Preservation`

### Pitfall 2: Dense-Only Retrieval Optimizes Similarity Instead of Usefulness
**What goes wrong:** The engine retrieves semantically nearby memories but misses exact lexical facts, time-bounded facts, contradictory updates, or sparse but decisive cues.

**Why it happens:** Teams over-trust embeddings and KNN recall. Current research on long-term assistant memory improves results by combining session decomposition, fact-augmented key expansion, and time-aware query expansion, which is a strong signal that naive dense retrieval is not enough.

**Consequences:** Retrieval looks acceptable on casual chat but fails on user-specific preferences, scheduling facts, revisions, and adversarial questions. The engine also becomes hard to tune because "similarity score" is not the same thing as "useful memory."

**Warning signs:**
- BM25 or exact-text features unexpectedly rescue failures from dense retrieval.
- Retrieval works for topic continuity but not for corrections, dates, names, or negations.
- Engineers spend time retuning thresholds without being able to explain failure classes.
- Top-k results are topically related but not answer-bearing.

**Prevention:**
- Ship hybrid retrieval intentionally: lexical, dense, temporal, and graph/fact signals in one ranking stack.
- Expand queries with entities, facts, aliases, and time scopes when the question implies them.
- Evaluate retrieval separately from answer generation: evidence recall, answer-bearing hit rate, contradiction hit rate, and abstention quality.
- Instrument "retrieved but not used" and "used but unsupported" cases.

**Future phase:** `Hybrid Retrieval + Query Expansion`

### Pitfall 3: Memory Has No Version Semantics, So Old Facts Beat New Ones
**What goes wrong:** The engine stores user facts as timeless truths. Later corrections, preference changes, relationship changes, or task-context changes do not cleanly supersede earlier memories.

**Why it happens:** Many systems treat memory as an append-only bag of snippets. Benchmarks such as LongMemEval explicitly include knowledge updates, temporal reasoning, and abstention because long-term memory systems fail here even when they do fine on plain fact recall.

**Consequences:** Cortext answers with stale or internally contradictory memories, especially in long-running human-assistant relationships. Trust drops faster here than from simple misses because the system sounds confidently outdated.

**Warning signs:**
- The same user attribute appears with multiple values and no clear winner.
- Retrieval quality degrades as the database ages, even without model regressions.
- Manual inspection shows contradictions but ranking has no concept of recency or validity interval.
- Evaluation sets underweight "used to be true" versus "is true now."

**Prevention:**
- Make temporal validity explicit in storage and ranking: observed-at, asserted-at, last-confirmed-at, superseded-by, contradiction links.
- Score freshness and conflict status directly instead of relying on generic relevance.
- Add tests for correction chains, renamed entities, changed preferences, and session-local overrides.
- Prefer abstention when the system sees unresolved conflict.

**Future phase:** `Temporal/Bitemporal Memory Correctness`

### Pitfall 4: Speaker Attribution Errors Pollute Memory Permanently
**What goes wrong:** ASR text is correct enough, but the wrong speaker gets attached to the utterance or to extracted facts. Once stored, identity pollution spreads into retrieval and consolidation.

**Why it happens:** Online diarization is hard under overlap, short turns, interruptions, and latency constraints. The diarization literature explicitly uses overlap-aware segmentation and cannot-link constraints to avoid wrongful merges, and even practical open-source stacks like WhisperX still flag word-level diarization improvement as unfinished work.

**Consequences:** Cortext remembers the assistant's statement as the user's preference, merges two humans into one profile, or attributes one person's correction to another. This is worse than a transcription typo because it corrupts durable identity state.

**Warning signs:**
- Overlap, interruption, and barge-in scenarios produce unstable speaker labels across replays.
- "Who said this?" cannot be answered confidently from persisted evidence.
- Speaker IDs are treated as ground truth even when diarization confidence is low.
- A diarization bug causes downstream memory drift days later.

**Prevention:**
- Separate transcript confidence from speaker confidence; low-confidence attribution should not become durable identity memory.
- Delay profile-affecting writes until speaker attribution stabilizes across a short buffer or second pass.
- Store diarization confidence, overlap flags, and attribution provenance alongside each utterance/fact.
- Build replay fixtures with overlap, interruptions, same-gender voices, and assistant barge-in to test identity stability.

**Future phase:** `Realtime Voice + Speaker Attribution`

### Pitfall 5: Hidden Degraded Paths Make Retrieval Quietly Worse
**What goes wrong:** The engine keeps running after a retrieval, storage, or model-path failure, but quality silently drops because fallbacks are invisible or unmeasured.

**Why it happens:** Memory engines favor availability, so they catch exceptions and continue. That is defensible only if degraded mode is explicit. In Cortext's current codebase, graph retrieval already has silent fallback branches, which means this is not hypothetical.

**Consequences:** Teams ship a system that "works" while the best signals are unavailable. Users experience erratic memory quality and engineers cannot correlate failures with root cause.

**Warning signs:**
- Quality regressions appear without corresponding crashes or alerts.
- Retrieval latency and answer quality swing independently.
- Debugging requires reading logs manually because there is no structured degraded-mode telemetry.
- Offline experiments and real app behavior disagree for no obvious reason.

**Prevention:**
- Convert silent fallback paths into explicit degraded-mode states with counters, traces, and surfaced diagnostics.
- Track per-stage availability and contribution: seed search, fact recall, graph expansion, reconstruction, summarization, speaker attribution.
- Fail closed for identity-affecting writes and fail open only for clearly non-destructive enrichments.
- Add regression tests that assert degraded-mode telemetry, not just returned payloads.

**Future phase:** `Retrieval Observability + Degraded-Path Telemetry`

### Pitfall 6: The Realtime Path Absorbs Background Maintenance Work
**What goes wrong:** Interactive retrieval or ingest starts doing fact lifecycle recomputation, reconstruction cleanup, schema-sensitive work, or expensive multi-pass DB traversal inline.

**Why it happens:** Local-first systems try to avoid background services, so maintenance work gets piggybacked onto user actions. This usually looks fine in small demos, then produces step-function latency spikes on real histories.

**Consequences:** Augmentation feels non-live. Voice interaction becomes laggy, interruptions are mistimed, and users stop trusting the system to stay out of the way.

**Warning signs:**
- P50 looks acceptable but P95/P99 degrade sharply with memory growth.
- The same query gets slower as history grows even when answer complexity does not.
- Maintenance tasks appear inside the same trace as retrieval/response generation.
- Engineers need to cap sweep sizes or memory counts to keep demos responsive.

**Prevention:**
- Put explicit latency budgets on ingest, retrieval, and voice-loop stages.
- Move fact lifecycle maintenance, backfills, reconstructions, and compaction into bounded background work or explicit maintenance commands.
- Profile retrieval by stage and refuse new scoring complexity without budget data.
- Preserve a "fast path" for interactive use, even if it returns less enriched context.

**Future phase:** `Realtime Latency Budget + Background Maintenance`

### Pitfall 7: Evaluation Rewards Recall Demos but Misses Real Memory Failures
**What goes wrong:** The system looks strong on handcrafted examples or a single benchmark, but fails on knowledge updates, abstention, multi-session reasoning, privacy boundaries, or efficiency under sustained use.

**Why it happens:** Memory evaluation is still immature. LoCoMo, LongMemEval, EvolMem, and CIMemories all exist because prior evaluation was too narrow. The recent evidence is uncomfortable: long-memory assistants still drop sharply on sustained interactions, no model dominates across memory dimensions, agent memory mechanisms can add efficiency cost without consistent benefit, and privacy violations can accumulate across tasks and repeated runs.

**Consequences:** Cortext can optimize itself into the wrong target, overfit to one harness, and miss the product failures that matter most for augmem.ai users.

**Warning signs:**
- Success is reported mostly through anecdotal conversations or one headline metric.
- Benchmarks do not include abstention, corrections, privacy/context gating, or speaker identity stability.
- Retrieval recall is measured, but not whether the final answer was supported by the retrieved evidence.
- Long-horizon runs are too expensive or noisy to be part of normal development.

**Prevention:**
- Treat evaluation as a matrix, not a score: retrieval hit rate, evidence support, correction handling, abstention, speaker attribution stability, latency, and privacy/contextual-integrity violations.
- Keep the existing long-horizon harnesses, but add targeted scenario suites for corrections, overlapping speakers, assistant interruption, and stale-memory conflict.
- Version datasets and traces so regressions are comparable across code, model, and schema changes.
- Require every retrieval/prompt change to show both quality impact and latency impact.

**Future phase:** `Long-Horizon Evaluation + Scenario Harness`

### Pitfall 8: Persistent Memory Leaks Across Contexts
**What goes wrong:** The engine retrieves or surfaces sensitive memory in contexts where it is not needed, or logs enough raw memory content that "local-first" still behaves like uncontrolled persistence.

**Why it happens:** Teams optimize for personalization utility and forget contextual integrity. Recent benchmark work shows strong models still leak attributes in inappropriate contexts, and violation rates can grow across tasks and repeated runs.

**Consequences:** The product becomes unsafe even if retrieval is technically accurate. For a human-memory product, that is a trust-ending failure.

**Warning signs:**
- Retrieval ranking has no notion of "appropriate for this task/user/request."
- Logs and diagnostics capture raw personal content by default.
- Memory recall is rewarded even when the answer would be better with abstention or less detail.
- Re-running the same task can surface different private attributes.

**Prevention:**
- Add task/context gating to memory surfacing, not just relevance scoring.
- Make logs redacted by default and separate developer diagnostics from user memory storage.
- Evaluate contextual-integrity violations explicitly, including repeated-run instability.
- Default to minimal sufficient memory in prompts instead of maximal recall.

**Future phase:** `Privacy + Contextual Integrity Guardrails`

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Retrieval Granularity + Evidence Preservation | Coarse summaries replace exact evidence | Keep raw evidence/facts retrievable and provenance-rich |
| Hybrid Retrieval + Query Expansion | Dense similarity outranks exact answer-bearing memory | Blend lexical, dense, temporal, and fact signals |
| Temporal/Bitemporal Memory Correctness | Stale facts beat updated facts | Add explicit supersession/conflict semantics |
| Realtime Voice + Speaker Attribution | Wrong speaker identity becomes durable memory | Gate writes on speaker confidence and overlap handling |
| Retrieval Observability + Degraded-Path Telemetry | Silent fallbacks mask quality loss | Expose degraded mode explicitly in metrics and tests |
| Realtime Latency Budget + Background Maintenance | Maintenance work lands on the interactive path | Move heavy work off-path and enforce latency budgets |
| Long-Horizon Evaluation + Scenario Harness | Benchmarks miss the failure modes that matter | Use scenario suites plus long-horizon regression traces |
| Privacy + Contextual Integrity Guardrails | Accurate memory is surfaced in inappropriate contexts | Add context gating, redaction, and leak-oriented eval |

## What Cortext Should Explicitly Avoid Next

- Do not make summaries the authoritative memory record.
- Do not ship another retrieval-scoring iteration without evidence-level observability.
- Do not write profile-affecting memory from low-confidence speaker attribution.
- Do not add more retrieval complexity to the synchronous hot path before stage-level latency budgets exist.
- Do not evaluate memory quality only with topical-chat success or recall-at-k.
- Do not treat "local-first" as a substitute for contextual privacy controls.

## Sources

- Project context: `/Users/gabrielwillen/VSCode/cortext/.planning/PROJECT.md`
- Codebase concerns: `/Users/gabrielwillen/VSCode/cortext/.planning/codebase/CONCERNS.md`
- Testing patterns: `/Users/gabrielwillen/VSCode/cortext/.planning/codebase/TESTING.md`
- LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory (ICLR 2025, arXiv 2410.10813): https://arxiv.org/abs/2410.10813
- LoCoMo: Evaluating Very Long-term Conversational Memory of LLM Agents (ICLR 2025): https://openreview.net/forum?id=xKDZAW0He3
- EvolMem: A Cognitive-Driven Benchmark for Multi-Session Dialogue Memory (arXiv 2601.03543, 2026): https://arxiv.org/abs/2601.03543
- Overlap-aware low-latency online speaker diarization based on end-to-end local segmentation (ASRU 2021, arXiv 2109.06483): https://arxiv.org/abs/2109.06483
- WhisperX official repository/README, used here only for current practitioner-facing diarization caveats and alignment notes: https://github.com/m-bain/whisperX
- CIMemories: A Compositional Benchmark For Contextual Integrity In LLMs (ICLR 2026): https://openreview.net/forum?id=YnNIp38v1M

## Confidence Notes

- **High confidence:** destructive compression, hybrid retrieval needs, temporal updates, speaker-overlap risk, and the need for long-horizon evaluation are all directly supported by current benchmarks/papers and match Cortext's current stress points.
- **Medium confidence:** the exact phase boundaries are proposed for roadmap usefulness, not copied from an existing roadmap.
