# Cortext Database Schema

Entity Relationship Diagram for signal storage and retrieval based on `docs/research/algorithms.md` and `docs/design.md`.

## Overview

```mermaid
---
title: Cortext Signal Storage & Retrieval Schema
config:
  layout: elk
---
erDiagram
    direction LR

    %% Core Vector Storage (sqlite-vec virtual table)
    EMBEDDINGS ||--o| MEMORIES : "has_metadata"
    EMBEDDINGS ||--o| MEMORY_FEEDBACK : "has_feedback"
    EMBEDDINGS ||--o{ SIGNAL_METRICS : "wrote_signal"
    EMBEDDINGS ||--o{ GRAPH_NODES : "vectorizes"
    EMBEDDINGS }|--o| EPISODES : "belongs_to"
    ACCUMULATOR_STATE ||--o{ EMBEDDINGS : "produces"

    EMBEDDINGS {
        int embedding_id PK "rowid"
        blob embedding "float[256] - sqlite-vec"
        text type "memory|summary|concept"
        real strength "Section 5.1 decay model"
        real use_frequency "Section 5.1 EWMA"
        real stability "Section 5.3.3 per-memory"
        real connectivity "Section 7.2 graph metric"
        real drift_mag "Section 3.1.3"
        real influence "Section 5.4 composite"
        real sustained_influence "Section 5.4 EWMA"
        real contextual_gain "Section 5.2"
        real redundancy "Section 7.2"
        real pre_activation "Section 6.5"
        real lability_state "Section 6.3"
        int suppression_count "Section 6.4 RIF"
        int cluster_id FK "Section 7.3"
        int last_access "timestamp"
        int created_at "timestamp"
    }

    %% Memory Metadata (The unit of retrieval)
    MEMORIES ||--o{ SIGNALS : "contains"
    MEMORIES ||--o| OBJSTORE : "stores_concatenated_blob"
    MEMORIES }|--|| EPISODES : "belongs_to"
    MEMORIES {
        int embedding_id PK,FK
        int start_ts "timestamp"
        int end_ts "timestamp"
        int n_signals "number of signals in memory"
        text primary_modality "text|audio|image"
        blob content_blob_id FK "concatenated/summary blob"
        real s_max "max signal score"
        real s_avg "average signal score"
        blob emotion "6d vector - Section 2.2.2"
        blob ambient_mood "6d vector - Section 2.2.4"
        int episode_id FK
        text status "active|consolidated|evicted"
    }

    %% Signal Metadata (Discrete inputs)
    SIGNALS {
        int signal_id PK
        int embedding_id FK "Memory this signal belongs to"
        int timestamp "signal arrival"
        text modality "text|audio|image"
        text mime
        blob blob_id FK "objstore references individual signal content"
        int serial_position "order within memory"
        real score "signal score from metrics"
    }

    %% Memory Feedback (retrieval/usage tracking)
    MEMORY_FEEDBACK {
        int embedding_id PK,FK
        int retrieved_count "Section 5.2"
        int used_count "Section 5.2"
        real influence_factor "Section 5.2"
        real mean_influence "Section 5.4"
        int last_used "timestamp"
        blob original_embedding "Section 6.3 reconsolidation"
        int lability_ts "Section 6.3 - Unix timestamp"
        int recovery_time_start "Section 6.4 RIF"
    }

    %% Graph Nodes (Section 12.4 - sqlite-graph)
    GRAPH_NODES ||--o{ GRAPH_EDGES : "connects"
    GRAPH_NODES {
        text node_id PK
        text type "entity|goal|concept"
        text label
        int embedding_id FK "nullable"
        json metadata
        int created_at
    }

    %% Graph Edges (Section 12.4 - sqlite-graph)
    GRAPH_EDGES {
        text source_id PK,FK
        text target_id PK,FK
        text edge_type PK "co_occurs|implies|contradicts|reinforces|derived_from|causes"
        real weight
        real decay_rate
        int last_reinforced
    }

    %% =======================================================================
    %% CONSOLIDATION & EXTRACTION TABLES (Section 7)
    %% =======================================================================

    %% Consolidation Summaries (Section 7.3 Clustering)
    CONSOLIDATION_SUMMARIES ||--o{ CONSOLIDATION_SOURCES : "has_sources"
    CONSOLIDATION_SUMMARIES ||--o{ EXTRACTION_ENTITIES : "extracts"
    CONSOLIDATION_SUMMARIES ||--o{ EXTRACTION_RELATIONS : "extracts"
    CONSOLIDATION_SUMMARIES {
        text summary_id PK
        text summary_text
        blob centroid "256d cluster center"
        int cluster_size
    }

    %% Consolidation Sources (Section 7.3 - source tracking)
    CONSOLIDATION_SOURCES {
        text summary_id FK
        int source_embedding_id FK
    }

    %% Extraction Entities (Section 7.4 Semantic Extraction)
    EXTRACTION_ENTITIES {
        text summary_id PK,FK
        text name PK
        text type PK
        real salience
        int embedding_id FK
    }

    %% Extraction Relations (Section 7.5 Knowledge Graph)
    EXTRACTION_RELATIONS {
        text summary_id FK
        text subject
        text predicate
        text object
        real confidence
    }

    %% Entity Index (Section 7.5 - name to node mapping)
    ENTITY_INDEX }|--|| GRAPH_NODES : "maps_to"
    ENTITY_INDEX {
        text name PK
        text node_id FK
    }

    %% Goal Nodes (Goal Alignment algorithm)
    GOAL_NODES }|--|| GRAPH_NODES : "is_goal"
    GOAL_NODES {
        text node_id PK,FK
    }

    %% Binary Object Store (sqlite-objstore virtual table)
    OBJSTORE {
        blob id PK "SHA256 content hash"
        blob data "raw content: text, audio PCM, image bytes"
    }

    %% Episodes (Section 3.1.3)
    EPISODES {
        int id PK
        int start_ts
        int end_ts
        text boundary_type "drift|explicit|timeout"
        blob centroid "256d vector"
    }

    %% Accumulator State (Section 4.4 - Write Pacing Accumulator)
    ACCUMULATOR_STATE {
        text source_id PK "per-stream accumulator"
        blob mu_acc "256d running mean"
        real drift_acc "accumulated drift"
        real s_sum "score sum"
        real s_max "max score"
        int n_signals "count"
        blob e_peak "256d peak embedding"
        int t_start "memory start timestamp"
        int last_write_ts "for write refractory"
        int last_signal_ts "for gap detection"
        real eta_acc "drift EWMA"
        real coherence_prev "previous coherence"
    }

    %% Signal Metrics (Section 3.1-3.3 - Composite Scoring)
    SIGNAL_METRICS {
        int id PK
        int timestamp
        int embedding_id FK "Memory this signal contributed to"
        real relevance
        real mismatch
        real surprise
        real rarity
        real drift
        real contradiction
        real utility
        real periphery
        real coverage
        real salience
        real valence
        real arousal
        real composite_score
        real threshold_t
        int write_decision
        real coherence "Section 3.1.1"
        real focus_spread "Section 3.1.2"
        real f_effective "Section 3.1.1"
    }
```

## Timestamp Units Convention

**All timestamp columns in the database are stored in milliseconds since Unix epoch.** This ensures consistency with the Signal API (`signal.timestamp`) and standard system time conventions.

| Column Pattern | Unit | Examples |
|----------------|------|----------|
| `timestamp` | milliseconds | `signals.timestamp`, `signal_metrics.timestamp` |
| `created_at` | milliseconds | `embeddings.created_at` |
| `last_access` | milliseconds | `embeddings.last_access` |
| `t_start` | milliseconds | `accumulator_state.t_start` |
| `*_ts` suffix | milliseconds | `last_write_ts`, `end_ts`, `episode_start_ts` |
| `updated_at` | milliseconds | `processor_state.updated_at` |

**API convention:** All timestamps in the Signal API (`signal.timestamp`) are in **milliseconds** since Unix epoch.

**Internal calculations:** Decay and rate formulas convert to seconds before computation: `t_seconds = t_milliseconds / 1000.0`

## SQLite Extensions

The schema relies on three embedded SQLite extensions:

| Extension | Virtual Table | Purpose | Used By |
|-----------|--------------|---------|---------|
| **sqlite-vec** | `EMBEDDINGS` | 256d float vector storage + KNN search | Vector retrieval, similarity |
| **sqlite-objstore** | `OBJSTORE` | Content-addressed blob storage | Raw payload (text, audio, images) |
| **sqlite-graph** | `GRAPH_NODES`, `GRAPH_EDGES` | Cypher-compatible knowledge graph | Semantic relationships |

### sqlite-vec Usage

```sql
-- Create embeddings table with 256d float vectors and metadata
CREATE VIRTUAL TABLE embeddings USING vec0(
    embedding_id INTEGER PRIMARY KEY,
    embedding float[256],
    type text,
    strength float,
    use_frequency float,
    stability float,
    connectivity float,
    drift_mag float,
    influence float,
    sustained_influence float,
    contextual_gain float,
    redundancy float,
    pre_activation float,
    lability_state float,
    suppression_count integer,
    cluster_id integer,
    last_access integer,
    +created_at integer
);

-- KNN search for k nearest neighbors
SELECT embedding_id, distance
FROM embeddings
WHERE embedding MATCH ?query_vector
  AND k = 10;
```

### sqlite-objstore Usage

```sql
-- Create objstore virtual table
CREATE VIRTUAL TABLE IF NOT EXISTS objstore USING objstore();

-- Store payload, returns SHA256 hash
SELECT objstore_put(?payload) AS blob_id;

-- Retrieve payload by hash
SELECT objstore_get(?blob_id) AS data;
```

## Entity Descriptions

### Core Tables

| Entity | Algorithm | Purpose |
|--------|-----------|---------|
| `EMBEDDINGS` | 5.1-5.4 | Vector storage with decay (sqlite-vec) |
| `MEMORIES` | 4.4, 6.1 | Groups of coherent signals (the unit of retrieval) |
| `SIGNALS` | 2.2 | Individual momentary inputs |
| `MEMORY_FEEDBACK` | 5.2 | Usage tracking and reconsolidation |
| `OBJSTORE` | - | Raw binary storage (sqlite-objstore) |
| `SIGNAL_METRICS` | 3.1-3.3 | Per-signal algorithmic telemetry |
| `ACCUMULATOR_STATE` | 4.4 | Real-time state for building memories |
| `PROCESSOR_STATE` | 1.4-8.4 | Global algorithmic parameters |
| `GRAPH_NODES`, `GRAPH_EDGES` | 7.4-7.6 | Semantic Knowledge Graph |
| `EPISODES` | 3.1.3 | Temporal episodic boundaries |
| `RECENT_CONTEXT`, `RECENT_SCORES` | 2.1, 4.1 | Sliding memory windows |
| `WORKING_MEMORY_SLOTS` | 6.1 | Active thought maintenance |

### Consolidation & Extraction Tables (Section 7)

| Entity | Algorithm | Purpose |
|--------|-----------|---------|
| `CONSOLIDATION_SUMMARIES` | 7.3 | Cluster summaries from consolidation |
| `CONSOLIDATION_SOURCES` | 7.3 | Source signal tracking |
| `EXTRACTION_ENTITIES` | 7.4 | Named entities extracted from memories |
| `EXTRACTION_RELATIONS` | 7.5 | Semantic relations |

### Relationship Cardinalities

| Relationship | Cardinality | Description |
|--------------|-------------|-------------|
| EMBEDDINGS → MEMORIES | 1:1 | Every embedding corresponds to one Memory |
| MEMORIES → SIGNALS | 1:N | One Memory groups multiple Signals |
| SIGNALS → OBJSTORE | 1:1 | Each Signal has a physical blob |
| MEMORIES → OBJSTORE | 1:0..1 | Memories can have a combined summary blob |
| EMBEDDINGS → SIGNAL_METRICS | 1:N | Tracking which signals built the embedding |
| PROCESSOR_STATE → WORKING_MEMORY_SLOTS | 1:N | Working Memory persistence |

## State Resumption

On startup, the processor loads persisted state:

1. **`processor_state`**: All scalar adaptive parameters.
2. **`accumulator_state`**: Rebuilds ongoing memories from unfinished streams.
3. **`blender_weights`**: RLS-fitted metric weights.
4. **`working_memory_slots`**: Re-populates active thoughts.
5. **`recent_context`**: Last N context embeddings for immediate relevance.
