# Feature Landscape

**Project:** cortext
**Domain:** Memory engine for augmenting human and LLM memory across text, voice, and multimodal workflows
**Researched:** 2026-04-07
**Overall confidence:** MEDIUM-HIGH

## Framing

This file is intentionally engine-scoped. It covers the capabilities a memory engine should expose to applications, agents, and realtime voice systems. It does **not** cover generic chat-product features.

The current ecosystem has largely converged on a few baseline expectations:

- durable memory primitives with scoped retrieval
- memory correction and deletion, not append-only storage
- multimodal ingestion into a common searchable memory substrate
- prompt-ready recall plus structured/raw retrieval surfaces

The differentiators are no longer "has memory" but rather: temporal reasoning, attribution quality, explainability, adaptive write behavior, and cross-modal recall that still works under realtime constraints.

## Table Stakes

Features users and downstream applications should expect from a serious memory engine. Missing any of these makes the engine feel incomplete or toy-like.

| Feature | Why Expected | Complexity | Dependencies |
|---------|--------------|------------|--------------|
| Scoped memory primitives (`add`, `get`, `search`, `update`, `delete`, `list/history`) | Current memory systems expose explicit memory operations and scoped retrieval rather than opaque transcript stuffing. | Medium | Durable store, stable memory IDs, namespace model |
| Namespaces for `user`, `session`, `agent`, and shared/group memory | Memory must be partitionable by owner and conversation while still allowing selective shared recall. | Medium | Identity model, session model, access rules |
| Hybrid retrieval with semantic search, metadata filters, and recency controls | Vector-only search is no longer enough; production memory needs precise filtering and ranking knobs. | Medium-High | Embeddings, metadata indexes, ranking layer |
| Prompt-ready recall plus structured/raw recall | Callers need both a convenient context string for prompting and machine-readable memory objects for UI, analytics, and policy logic. | Medium | Retrieval API, hydration layer, provenance fields |
| Memory correction lifecycle (`dedupe`, `merge`, `update`, `invalidate`, `delete`) | Users change preferences, facts expire, and bad memories must be fixable. Append-only memory is not acceptable. | High | Provenance, stable IDs, conflict detection, write policy |
| Multimodal ingestion into a normalized memory schema | Text, images, and audio-derived content must become first-class memories in one system, not separate silos. | High | Modality adapters, asset references, canonical memory schema |
| Speaker/session attribution for multi-party conversations | Voice and group workflows break down if memories are not attributed to the correct person or agent. | High | Session model, participant identity, diarization/transcript pipeline |
| Privacy and governance controls | Human memory augmentation requires erasure, export, auditability, and strong local/project scoping. | Medium | Delete APIs, history/audit trail, namespace isolation |
| Retrieval inspectability | Applications need to know why a memory surfaced, what source created it, and how confident the engine was. | Medium | Scores, source anchors, telemetry, traceable IDs |

## Differentiators

Features that materially separate a strong memory engine from a merely competent one.

| Feature | Value Proposition | Complexity | Dependencies |
|---------|-------------------|------------|--------------|
| Temporal or bitemporal memory model | Lets the engine represent not just facts, but when facts became valid, changed, or expired. This is the difference between "user likes Adidas" and "used to like Adidas until last month." | High | Provenance, lifecycle correction, timestamped ingestion |
| Knowledge-graph augmentation over base memory records | Enables multi-hop recall, relationship-aware retrieval, and richer context than flat vector neighbors alone. | High | Entity extraction, relation storage, hybrid retrieval |
| Adaptive write policy tied to salience instead of "store everything" | A real memory engine should learn what deserves durable storage, not blindly persist every turn. This is especially important for long voice sessions. | High | Telemetry, write scoring, consolidation pipeline, evaluation harness |
| Budget-aware working-memory hydration | Different consumers need different memory shapes: terse fact pack for voice, richer evidence bundle for an agent, more narrative summary for humans. | High | Retrieval inspectability, summarization, token/latency budgets |
| Cross-modal recall and reverse lookup | Text should retrieve relevant screenshots or audio moments; images or audio should retrieve related text facts. This is where multimodal memory becomes genuinely useful. | High | Unified or linked embeddings, source anchors, asset store |
| Realtime voice memory with incremental correction | High-quality live memory for speech requires streaming ASR, speaker attribution, revisions, and interruption-safe writes instead of waiting for batch transcripts. | High | Streaming speech stack, diarization, revision-aware memory updates |
| Shared memory layers with permissions and mutability rules | Personal memory, agent procedural memory, organization memory, and read-only policy memory should coexist without collapsing into one pool. | Medium-High | Namespace model, ACL/policy layer, typed memory classes |
| Explanation-grade provenance and replay | The engine can show the episode, source span, graph path, and retrieval path that produced a memory decision. This is valuable for human trust and tuning. | High | Episode model, retrieval traces, graph/path metadata |
| Maintenance surfaces for consolidation, decay, and replay | Exposing explicit maintenance commands allows large databases to stay healthy without doing all expensive work on the interactive path. | Medium-High | Background jobs, lifecycle state, compaction/consolidation APIs |

## Anti-Features

Features or approaches that should be explicitly avoided.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Treating transcript history as "memory" | Full transcripts are too noisy, expensive, and hard to correct. They are evidence, not memory. | Separate raw episodes from distilled memories and working memory |
| Single undifferentiated memory pool | Mixing user facts, agent procedures, session state, and shared knowledge produces contamination and bad recall. | Use explicit scopes and typed memory lanes |
| Vector-only memory with no invalidation or provenance | Similarity search alone cannot represent contradiction, expiry, or why a memory should be trusted. | Add lifecycle state, source anchors, and update/delete semantics |
| Synchronous deep consolidation on the hot path | Heavy summarization, graph expansion, or lifecycle recomputation will make voice and live chat feel laggy. | Keep interactive writes cheap and move expensive work to background maintenance |
| Hidden autonomous writes with no inspection or erasure path | Human-facing memory that users cannot inspect, correct, or delete will fail trust and compliance requirements. | Expose inspect/edit/delete/export controls at the engine boundary |
| Modality-specific silos | Separate "text memory", "voice memory", and "image memory" stores create brittle recall and duplicate logic. | Normalize all modalities into one memory model with modality-aware evidence |
| Memory behavior hard-coded to one model/provider | The engine becomes brittle and expensive to evolve if memory quality depends on one vendor-specific prompt or API. | Keep extraction, summarization, and ranking behind swappable backend interfaces |

## Feature Dependencies

```text
Identity + namespaces
  -> scoped memory primitives
  -> shared/personal/agent memory layers

Scoped memory primitives + canonical IDs
  -> update/delete/invalidate/history
  -> prompt-ready + structured retrieval surfaces

Canonical memory schema + modality adapters
  -> multimodal ingestion
  -> cross-modal recall

Session model + speaker attribution
  -> reliable voice memory
  -> multi-party group memory

Provenance + timestamped ingestion
  -> correction lifecycle
  -> temporal reasoning
  -> explanation-grade replay

Hybrid retrieval + inspectability
  -> budget-aware working-memory hydration
  -> graph-augmented retrieval

Telemetry + evals + maintenance APIs
  -> adaptive write policy
  -> background consolidation/decay
```

## Brownfield Recommendation

Prioritize for the next milestone:

1. Scoped memory primitives plus inspectable retrieval
2. Multimodal normalization with source anchors
3. Speaker/session attribution for voice and group workflows
4. Memory correction lifecycle (`update` / `invalidate` / `delete`)
5. Budget-aware working-memory hydration

Defer until the base is cleaner:

- full temporal graph reasoning
- highly adaptive write policies
- explanation-grade replay across graph paths

Rationale: Cortext already has a durable pipeline, multimodal processing, retrieval, consolidation, and working-memory behavior. The highest-value next step is making those capabilities more explicit, inspectable, and correctable at the engine boundary before pushing further into more sophisticated graph or realtime behaviors.

## Sources

Official/current references used for this split:

- LangChain docs, "Long-term memory" and memory concepts: https://docs.langchain.com/oss/python/langchain/long-term-memory and https://docs.langchain.com/oss/javascript/concepts/memory
- LangChain changelog, semantic search for LangGraph memory (2024-12-06): https://changelog.langchain.com/announcements/semantic-search-for-langgraph-s-long-term-memory
- Zep docs, concepts / sessions / memory retrieval / episodes: https://help.getzep.com/v2/concepts , https://help.getzep.com/v2/sessions , https://help.getzep.com/v2/sdk-reference/memory/get , https://help.getzep.com/graphiti/core-concepts/adding-episodes
- Mem0 docs, update lifecycle / metadata filtering / multimodal / group chat / MCP / reranking: https://docs.mem0.ai/open-source/features/custom-update-memory-prompt , https://docs.mem0.ai/open-source/features/metadata-filtering , https://docs.mem0.ai/open-source/features/multimodal-support , https://docs.mem0.ai/platform/features/group-chat , https://docs.mem0.ai/platform/features/mcp-integration , https://docs.mem0.ai/open-source/features/reranker-search
- Letta docs, shared memory blocks: https://docs.letta.com/tutorials/shared-memory-blocks/
- Qdrant docs, overview and multimodal retrieval: https://qdrant.tech/documentation/overview/ and https://qdrant.tech/documentation/multimodal-search/
- Soniox docs, speaker diarization and realtime tradeoffs: https://soniox.com/docs/stt/concepts/speaker-diarization
