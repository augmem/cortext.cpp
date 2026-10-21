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
// v2 SIGNALS - Per-signal metrics (inline, no separate signal_metrics)
// ===========================================================================

constexpr const char *kSignalsInsertColumns =
    "memory_id, source_id, embedding_id, blob_id, timestamp, modality, mime, "
    "serial_position, score, created_at";

constexpr const char *kSignalsInsertDefaults = "?, ?, ?, ?, ?, ?, ?, ?, ?, ?";

} // namespace cortext::store
