#pragma once

namespace cortext::store
{

// SQL fragments for embeddings table INSERTs
// Note: sqlite-vec has max 16 auxiliary columns, doesn't support DEFAULT values
// Columns: embedding, type, strength, use_frequency, stability, connectivity,
//          drift_mag, influence, sustained_influence, contextual_gain, redundancy,
//          pre_activation, lability_state, suppression_count, cluster_id,
//          last_access, created_at (16 auxiliary + 1 vector = 17 total)

// Column list for full embeddings INSERT (excluding rowid/embedding_id)
constexpr const char *kEmbeddingsInsertColumns =
    "embedding, type, strength, use_frequency, stability, connectivity, "
    "drift_mag, influence, sustained_influence, contextual_gain, redundancy, "
    "pre_activation, lability_state, suppression_count, created_at";

// Column list for reconsolidation (no created_at, has lability_state)
constexpr const char *kEmbeddingsReconsolidateColumns =
    "embedding_id, embedding, type, strength, use_frequency, stability, "
    "connectivity, drift_mag, influence, sustained_influence, contextual_gain, "
    "redundancy, pre_activation, lability_state, suppression_count";

// Default values placeholder for 'memory' type (?, 'memory', defaults..., ?)
// The two ? are: embedding vector, created_at timestamp
constexpr const char *kEmbeddingsMemoryDefaults =
    "?, 'memory', 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, ?";

// Default values for reconsolidation (embedding_id, embedding, type, defaults..., lability_state)
// The three ? are: embedding_id, embedding vector, lability_state
constexpr const char *kEmbeddingsReconsolidateDefaults =
    "?, ?, 'memory', 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ?, 0";

// Default values for centroid type (used in consolidation_summarize)
// Single ? is for embedding vector
constexpr const char *kEmbeddingsCentroidDefaults =
    "'centroid', 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0";

// Default values for concept type (used in concept_detection)
// Two ? are: embedding_id, embedding vector
constexpr const char *kEmbeddingsConceptDefaults =
    "?, ?, 'concept', 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0";

} // namespace cortext::store
