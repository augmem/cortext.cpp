#pragma once

namespace cortext::store
{

// ===========================================================================
// v2 EMBEDDINGS - Minimal sqlite-vec table (only 3 columns)
// All metadata moved to MEMORIES table
// ===========================================================================

// Column list for embeddings INSERT (sqlite-vec virtual table)
constexpr const char *kEmbeddingsInsertColumns = "embedding, created_at";

// Default values for embeddings INSERT
// Placeholders: embedding vector, created_at timestamp
constexpr const char *kEmbeddingsInsertDefaults = "?, ?";

// ===========================================================================
// v2 MEMORIES - Full metadata table
// ===========================================================================

// Core insert columns (minimal required for new memory)
constexpr const char *kMemoriesInsertColumns =
    "embedding_id, source_id, kind, start_ts, end_ts, n_signals, modality, "
    "s_max, s_avg, emotion, ambient_mood, created_at";

// Default values for new LONG_TERM memory
constexpr const char *kMemoriesLongTermDefaults =
    "?, ?, 'LONG_TERM', ?, ?, ?, ?, ?, ?, ?, ?, ?";

// Default values for new WORKING memory
constexpr const char *kMemoriesWorkingDefaults =
    "?, ?, 'WORKING', ?, ?, ?, ?, ?, ?, ?, ?, ?";

// Default values for ASSOCIATION (consolidation centroids)
constexpr const char *kMemoriesAssociationDefaults =
    "?, ?, 'ASSOCIATION', ?, ?, ?, ?, ?, ?, ?, ?, ?";

// Default values for LABEL (concept nodes)
constexpr const char *kMemoriesLabelDefaults =
    "?, ?, 'LABEL', ?, ?, ?, ?, ?, ?, ?, ?, ?";

// ===========================================================================
// v2 SIGNALS - Per-signal metrics (inline, no separate signal_metrics)
// ===========================================================================

constexpr const char *kSignalsInsertColumns =
    "memory_id, source_id, embedding_id, blob_id, timestamp, modality, mime, "
    "serial_position, score, created_at";

constexpr const char *kSignalsInsertDefaults = "?, ?, ?, ?, ?, ?, ?, ?, ?, ?";

} // namespace cortext::store
