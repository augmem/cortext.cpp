#include "cortext/store/schema.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

namespace cortext::store
{

namespace
{

/// @brief Represents a single migration step.
struct Migration
{
  int64_t id;                            // Unique integer ID (e.g., 2024010101)
  std::string description;               // Human-readable description
  std::vector<std::string> up_statements; // SQL statements to apply
};

/// @brief Reads the set of already-applied migration IDs from the tracking table.
///
/// Migration IDs are monotonically increasing integers (e.g., YYYYMMDDnn format)
/// that must be unique across all registered migrations to prevent collisions.
std::set<int64_t>
GetAppliedMigrations (Store &store)
{
  std::set<int64_t> applied_ids;
  auto rows = store.Execute ("SELECT id FROM cortext_schema_migrations");
  for (const auto &row : rows)
    {
      if (row.count ("id"))
        {
          auto val = row.at ("id");
          if (val.type () == typeid (int))
            applied_ids.insert (std::any_cast<int> (val));
          else if (val.type () == typeid (long long))
            applied_ids.insert (std::any_cast<long long> (val));
        }
    }
  return applied_ids;
}

void
ApplySingleMigrationInTransaction (Store &store, const Migration &m)
{
  try
    {
      for (const auto &sql : m.up_statements)
        {
          try
            {
              store.Execute (sql);
            }
          catch (const std::exception &e)
            {
              telemetry::LogError (
                  "Migration statement failed",
                  { telemetry::Attribute::String ("component", "store.schema"),
                    telemetry::Attribute::Int64 ("migration_id", m.id),
                    telemetry::Attribute::String ("migration_description", m.description),
                    telemetry::Attribute::String ("sql", sql),
                    telemetry::Attribute::String ("error", e.what ()) });
              throw;
            }
        }
      store.Execute (
          "INSERT INTO cortext_schema_migrations (id, description, applied_at) "
          "VALUES (?, ?, strftime('%s', 'now'))",
          { m.id, m.description });
    }
  catch (const std::exception &e)
    {
      telemetry::LogError (
          "Schema migration failed",
          { telemetry::Attribute::String ("component", "store.schema"),
            telemetry::Attribute::Int64 ("migration_id", m.id),
            telemetry::Attribute::String ("migration_description", m.description),
            telemetry::Attribute::String ("error", e.what ()) });
      throw;
    }
}

/// @brief Returns the core schema migrations.
///
/// All schema is defined here in a single migration. Future schema changes
/// should add new migrations with higher IDs.
std::vector<Migration>
GetCoreMigrations ()
{
  return {
      // ==========================================================================
      // Migration 0: v2 Schema per docs/paper/diagrams/entity-relationship.qmd
      // Clean slate - EPISODES -> MEMORIES -> SIGNALS -> (EMBEDDINGS, BLOBS)
      // ==========================================================================
      {
          0,
          "Core schema v2 (docs/paper/diagrams/entity-relationship.qmd)",
          {
              // ------------------------------------------------------------------
              // sqlite-objstore virtual table (BLOBS)
              // ------------------------------------------------------------------
              "CREATE VIRTUAL TABLE IF NOT EXISTS blobs USING objstore()",

              // ------------------------------------------------------------------
              // EPISODES - Episodic boundaries (top of hierarchy)
              // ------------------------------------------------------------------
              "CREATE TABLE IF NOT EXISTS episodes ("
              "  episode_id INTEGER PRIMARY KEY,"
              "  start_ts INTEGER NOT NULL,"
              "  end_ts INTEGER,"
              "  boundary_type TEXT,"
              "  centroid BLOB,"
              "  created_at INTEGER NOT NULL"
              ")",

              // ------------------------------------------------------------------
              // LONG-TERM GRAPH MEMORY NODES
              //
              // The durable LTM is the persisted graph. This table stores graph
              // nodes: episodic memories, associations, labels, and working
              // memory records. Per-node adaptive state from memory_feedback,
              // rif_state, and emotional_tags is folded into the same row.
              // ------------------------------------------------------------------
              "CREATE TABLE IF NOT EXISTS memories ("
              "  memory_id INTEGER PRIMARY KEY,"
              "  episode_id INTEGER,"
              "  embedding_id INTEGER,"
              "  blob_id BLOB,"
              "  source_id TEXT NOT NULL,"
              "  kind TEXT NOT NULL DEFAULT 'LONG_TERM',"
              "  label TEXT,"
              "  start_ts INTEGER NOT NULL,"
              "  end_ts INTEGER,"
              "  n_signals INTEGER NOT NULL DEFAULT 1,"
              "  modality TEXT NOT NULL DEFAULT 'text',"
              // Score metrics
              "  s_max REAL NOT NULL DEFAULT 0.0,"
              "  s_avg REAL NOT NULL DEFAULT 0.0,"
              "  s_emotion_max REAL NOT NULL DEFAULT 0.0,"
              "  s_arousal_avg REAL NOT NULL DEFAULT 0.0,"
              "  boundary_score REAL NOT NULL DEFAULT 0.0,"
              // Emotion vectors (6d)
              "  emotion BLOB,"
              "  ambient_mood BLOB,"
              // Decay metrics (from embeddings v1)
              "  strength REAL NOT NULL DEFAULT 1.0,"
              "  trace_fast REAL NOT NULL DEFAULT 0.0,"
              "  trace_med REAL NOT NULL DEFAULT 0.0,"
              "  trace_slow REAL NOT NULL DEFAULT 0.0,"
              "  trace_ultra REAL NOT NULL DEFAULT 0.0,"
              "  use_frequency REAL NOT NULL DEFAULT 0.0,"
              "  stability REAL NOT NULL DEFAULT 1.0,"
              "  connectivity REAL NOT NULL DEFAULT 0.0,"
              "  drift_mag REAL NOT NULL DEFAULT 0.0,"
              "  influence REAL NOT NULL DEFAULT 0.0,"
              "  sustained_influence REAL NOT NULL DEFAULT 0.0,"
              "  contextual_gain REAL NOT NULL DEFAULT 0.0,"
              "  redundancy REAL NOT NULL DEFAULT 0.0,"
              "  pre_activation REAL NOT NULL DEFAULT 0.0,"
              "  lability_state REAL NOT NULL DEFAULT 0.0,"
              "  suppression_count INTEGER NOT NULL DEFAULT 0,"
              "  cluster_id INTEGER,"
              "  last_access INTEGER,"
              "  created_at INTEGER NOT NULL,"
              "  context BLOB,"
              // Memory feedback (from memory_feedback v1)
              "  retrieved_count INTEGER NOT NULL DEFAULT 0,"
              "  used_count INTEGER NOT NULL DEFAULT 0,"
              "  influence_factor REAL NOT NULL DEFAULT 0.0,"
              "  mean_influence REAL NOT NULL DEFAULT 0.0,"
              "  last_used INTEGER,"
              "  original_centroid BLOB,"
              "  lability_ts INTEGER,"
              "  recovery_time_start INTEGER,"
              // RIF state (from rif_state v1)
              "  suppression REAL NOT NULL DEFAULT 0.0,"
              "  suppression_ts INTEGER,"
              // Source monitoring
              "  source_origin TEXT DEFAULT 'source',"
              "  source_reliability REAL NOT NULL DEFAULT 0.7,"
              "  source_contradiction_count INTEGER NOT NULL DEFAULT 0,"
              "  source_last_verified_ts INTEGER NOT NULL DEFAULT 0,"
              // Flashbulb (from emotional_tags v1)
              "  flashbulb INTEGER NOT NULL DEFAULT 0,"
              "  emotional_intensity REAL NOT NULL DEFAULT 0.0,"
              "  half_life_bonus REAL NOT NULL DEFAULT 0.0,"
              "  detail_suppression REAL NOT NULL DEFAULT 0.0,"
              "  gist_components INTEGER NOT NULL DEFAULT 0,"
              "  cascade_radius INTEGER NOT NULL DEFAULT 0,"
              "  cascade_decay REAL NOT NULL DEFAULT 0.0,"
              // Synaptic tagging
              "  tag_strength REAL NOT NULL DEFAULT 0.0,"
              "  tag_expires_at INTEGER NOT NULL DEFAULT 0"
              ")",

              // ------------------------------------------------------------------
              // SIGNALS - Extended signal metadata with inline metrics
              // (was split: signals + signal_metrics)
              // ------------------------------------------------------------------
              "CREATE TABLE IF NOT EXISTS signals ("
              "  signal_id INTEGER PRIMARY KEY,"
              "  memory_id INTEGER,"
              "  source_id TEXT NOT NULL,"
              "  embedding_id INTEGER NOT NULL,"
              "  blob_id BLOB,"
              "  timestamp INTEGER NOT NULL,"
              "  modality TEXT NOT NULL DEFAULT 'text',"
              "  mime TEXT,"
              "  serial_position INTEGER NOT NULL DEFAULT 0,"
              // Inline metrics (from signal_metrics v1)
              "  score REAL NOT NULL DEFAULT 0.0,"
              "  relevance REAL,"
              "  mismatch REAL,"
              "  surprise REAL,"
              "  rarity REAL,"
              "  drift REAL,"
              "  contradiction REAL,"
              "  utility REAL,"
              "  periphery REAL,"
              "  coverage REAL,"
              "  salience REAL,"
              "  valence REAL,"
              "  arousal REAL,"
              "  threshold_t REAL,"
              "  write_decision INTEGER,"
              "  coherence REAL,"
              "  focus_spread REAL,"
              "  f_effective REAL,"
              "  created_at INTEGER NOT NULL"
              ")",

              // ------------------------------------------------------------------
              // EMBEDDINGS - Minimal sqlite-vec virtual table
              // Most metadata moved to MEMORIES
              // ------------------------------------------------------------------
              "CREATE VIRTUAL TABLE IF NOT EXISTS embeddings USING vec0("
              "embedding_id INTEGER PRIMARY KEY,"
              "embedding float[256],"
              "+created_at integer"
              ")",

              // ------------------------------------------------------------------
              // LONG-TERM GRAPH MEMORY EDGES
              //
              // Durable graph edges such as has_label, derived_from, reinforces,
              // and relation links. Together, memories + associations are LTM.
              // ------------------------------------------------------------------
              "CREATE TABLE IF NOT EXISTS associations ("
              "  source_memory_id INTEGER NOT NULL,"
              "  target_memory_id INTEGER NOT NULL,"
              "  edge_type TEXT NOT NULL,"
              "  weight REAL NOT NULL DEFAULT 1.0,"
              "  decay_rate REAL,"
              "  last_reinforced INTEGER,"
              "  PRIMARY KEY (source_memory_id, target_memory_id, edge_type)"
              ")",

              // ------------------------------------------------------------------
              // ACCUMULATORS - Write pacing per source (was accumulator_state)
              // ------------------------------------------------------------------
              "CREATE TABLE IF NOT EXISTS accumulators ("
              "  source_id TEXT PRIMARY KEY,"
              "  episode_id INTEGER,"
              "  mu_acc BLOB,"
              "  drift_acc REAL NOT NULL DEFAULT 0.0,"
              "  s_sum REAL NOT NULL DEFAULT 0.0,"
              "  s_max REAL NOT NULL DEFAULT 0.0,"
              "  n INTEGER NOT NULL DEFAULT 0,"
              "  e_peak BLOB,"
              "  emo_max REAL NOT NULL DEFAULT 0.0,"
              "  arousal_sum REAL NOT NULL DEFAULT 0.0,"
              "  acc_signals_window BLOB,"
              "  drift_accum REAL NOT NULL DEFAULT 0.0,"
              "  drift_at_last_interrupt REAL NOT NULL DEFAULT 0.0,"
              "  drift_acc_pacing REAL NOT NULL DEFAULT 0.0,"
              "  x_last_check BLOB,"
              "  prev_x BLOB,"
              "  c_t BLOB,"
              "  t_start INTEGER NOT NULL DEFAULT 0,"
              "  last_write_ts INTEGER NOT NULL DEFAULT 0,"
              "  last_signal_ts INTEGER NOT NULL DEFAULT 0,"
              "  eta_acc REAL NOT NULL DEFAULT 0.0,"
              "  coherence_prev REAL NOT NULL DEFAULT 1.0"
              ")",

              // ------------------------------------------------------------------
              // STATE - Unified processor state (singleton, id=1)
              // Includes: processor_state + 4 blender tables
              // ------------------------------------------------------------------
              "CREATE TABLE IF NOT EXISTS state ("
              "  id INTEGER PRIMARY KEY CHECK (id = 1),"
              "  signals_processed INTEGER NOT NULL DEFAULT 0,"
              // Threshold state
              "  theta_dynamic REAL NOT NULL DEFAULT 0.2,"
              "  theta_target REAL NOT NULL DEFAULT 0.2,"
              "  hysteresis REAL NOT NULL DEFAULT 0.05,"
              "  half_life REAL NOT NULL DEFAULT 120.0,"
              // Focus state
              "  weight_relevance REAL NOT NULL DEFAULT 0.5,"
              "  attention_width REAL NOT NULL DEFAULT 1.57,"
              "  coverage_gain_floor REAL NOT NULL DEFAULT 0.65,"
              "  mismatch_weight REAL NOT NULL DEFAULT 0.5,"
              // Sensitivity state
              "  weight_novelty REAL NOT NULL DEFAULT 0.3,"
              "  weight_surprise REAL NOT NULL DEFAULT 0.2,"
              "  weight_valence REAL NOT NULL DEFAULT 0.4,"
              "  weight_arousal REAL NOT NULL DEFAULT 0.0,"
              "  emotion_gain REAL NOT NULL DEFAULT 1.0,"
              "  score_gain REAL NOT NULL DEFAULT 1.0,"
              "  rate_target REAL NOT NULL DEFAULT 0.2,"
              // Emotion state
              "  emotion_intensity REAL NOT NULL DEFAULT 0.0,"
              "  valence REAL NOT NULL DEFAULT 0.5,"
              "  arousal REAL NOT NULL DEFAULT 0.0,"
              "  mood_vector BLOB,"
              "  last_mood_ts INTEGER NOT NULL DEFAULT 0,"
              "  flashbulb_rate REAL NOT NULL DEFAULT 0.0,"
              // Stability state
              "  rate_decay REAL NOT NULL DEFAULT 0.60,"
              "  periphery_half_life REAL NOT NULL DEFAULT 120.0,"
              "  salience_half_life REAL NOT NULL DEFAULT 120.0,"
              "  drift_weight REAL NOT NULL DEFAULT 0.5,"
              "  retention_ema REAL NOT NULL DEFAULT 0.0,"
              // Rate control
              "  m_rate REAL NOT NULL DEFAULT 0.0,"
              "  rho_hat_prev REAL NOT NULL DEFAULT 0.0,"
              "  dt_ema REAL NOT NULL DEFAULT 0.0,"
              "  rate_ticks INTEGER NOT NULL DEFAULT 0,"
              "  last_rate_timestamp INTEGER NOT NULL DEFAULT 0,"
              "  reliability REAL NOT NULL DEFAULT 1.0,"
              // Uncertainty
              "  u_uncertainty REAL NOT NULL DEFAULT 0.0,"
              // Embedding prediction
              "  last_embedding BLOB,"
              "  x_pred_ema BLOB,"
              "  delta_half_life_adj REAL NOT NULL DEFAULT 0.0,"
              "  sustained_influence REAL NOT NULL DEFAULT 0.0,"
              // Working memory
              "  wm_maintenance_cost REAL NOT NULL DEFAULT 0.0,"
              "  wm_slot_count INTEGER NOT NULL DEFAULT 0,"
              "  wm_last_accepted INTEGER NOT NULL DEFAULT 0,"
              "  wm_last_chunked INTEGER NOT NULL DEFAULT 0,"
              // Consolidation
              "  last_consolidation_ts INTEGER NOT NULL DEFAULT 0,"
              "  consolidation_count INTEGER NOT NULL DEFAULT 0,"
              "  memories_since_consolidation INTEGER NOT NULL DEFAULT 0,"
              "  is_processing_signal INTEGER NOT NULL DEFAULT 0,"
              "  last_retrieval_ts INTEGER NOT NULL DEFAULT 0,"
              // Episode tracking
              "  episode_start_ts INTEGER NOT NULL DEFAULT 0,"
              "  last_interrupt_tick INTEGER NOT NULL DEFAULT 0,"
              "  last_signal_timestamp INTEGER NOT NULL DEFAULT 0,"
              "  updated_at INTEGER NOT NULL DEFAULT 0,"
              // Outcome prediction + neuromodulators
              "  outcome_pred REAL NOT NULL DEFAULT 0.0,"
              "  neuromod_ach REAL NOT NULL DEFAULT 0.0,"
              "  neuromod_ne REAL NOT NULL DEFAULT 0.0,"
              "  neuromod_da REAL NOT NULL DEFAULT 0.0,"
              "  osc_phase REAL NOT NULL DEFAULT 0.0,"
              // Write rate
              "  write_rate_timestamps BLOB,"
              // Blender weights (from blender_weights v1)
              "  w_relevance REAL,"
              "  w_mismatch REAL,"
              "  w_surprise REAL,"
              "  w_rarity REAL,"
              "  w_drift REAL,"
              "  w_contradiction REAL,"
              "  w_utility REAL,"
              "  w_periphery REAL,"
              "  w_coverage REAL,"
              "  w_salience REAL,"
              "  w_valence REAL,"
              "  w_arousal REAL,"
              "  blender_ready INTEGER NOT NULL DEFAULT 0,"
              "  blender_update_count INTEGER NOT NULL DEFAULT 0,"
              // Blender covariance (from blender_covariance v1)
              "  blender_P_matrix BLOB,"
              // Blender coefficients (from blender_coefficients v1)
              "  blender_coefficients BLOB,"
              // Blender coeff covariance (from blender_coeff_covariance v1)
              "  blender_coeff_P_matrix BLOB"
              ")",

              // ------------------------------------------------------------------
              // EPISODES indexes
              // ------------------------------------------------------------------
              "CREATE INDEX IF NOT EXISTS idx_episodes_end ON episodes(end_ts) WHERE end_ts IS NULL",

              // ------------------------------------------------------------------
              // MEMORIES indexes
              // ------------------------------------------------------------------
              "CREATE INDEX IF NOT EXISTS idx_memories_episode ON memories(episode_id, start_ts)",
              "CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories(embedding_id)",
              "CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind)",
              "CREATE INDEX IF NOT EXISTS idx_memories_working_active "
              "ON memories(end_ts, memory_id) "
              "WHERE kind = 'WORKING' AND end_ts IS NULL",
              "CREATE INDEX IF NOT EXISTS idx_memories_working_closed_end "
              "ON memories(end_ts, memory_id) "
              "WHERE kind = 'WORKING' AND end_ts IS NOT NULL",
              "CREATE INDEX IF NOT EXISTS idx_memories_cluster ON memories(cluster_id) WHERE cluster_id IS NOT NULL",

              // ------------------------------------------------------------------
              // SIGNALS indexes
              // ------------------------------------------------------------------
              "CREATE INDEX IF NOT EXISTS idx_signals_memory ON signals(memory_id, serial_position)",
              "CREATE INDEX IF NOT EXISTS idx_signals_source ON signals(source_id) WHERE memory_id IS NULL",
              "CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(timestamp DESC)",

              // ------------------------------------------------------------------
              // ASSOCIATIONS indexes
              // ------------------------------------------------------------------
              "CREATE INDEX IF NOT EXISTS idx_associations_source ON associations(source_memory_id, edge_type)",
              "CREATE INDEX IF NOT EXISTS idx_associations_target ON associations(target_memory_id, edge_type)",
              "CREATE INDEX IF NOT EXISTS idx_associations_source_weight "
              "ON associations(source_memory_id, weight DESC, last_reinforced DESC, target_memory_id)",
              "CREATE INDEX IF NOT EXISTS idx_associations_target_weight "
              "ON associations(target_memory_id, weight DESC, last_reinforced DESC, source_memory_id)",
              "CREATE INDEX IF NOT EXISTS idx_associations_source_edge_weight "
              "ON associations(source_memory_id, edge_type, weight DESC, last_reinforced DESC, target_memory_id)",
              "CREATE INDEX IF NOT EXISTS idx_associations_target_edge_weight "
              "ON associations(target_memory_id, edge_type, weight DESC, last_reinforced DESC, source_memory_id)",

              // ------------------------------------------------------------------
              // ACCUMULATORS indexes
              // ------------------------------------------------------------------
              "CREATE INDEX IF NOT EXISTS idx_accumulators_episode ON accumulators(episode_id)",

              // ------------------------------------------------------------------
              // Views (Computed Windows) - replace sliding window tables
              // ------------------------------------------------------------------
              "CREATE VIEW IF NOT EXISTS recent_context AS "
              "SELECT s.signal_id, s.embedding_id, e.embedding, s.timestamp "
              "FROM signals s "
              "JOIN embeddings e ON s.embedding_id = e.embedding_id "
              "ORDER BY s.timestamp DESC "
              "LIMIT 64",

              "CREATE VIEW IF NOT EXISTS recent_scores AS "
              "SELECT signal_id, score, timestamp "
              "FROM signals "
              "WHERE score IS NOT NULL "
              "ORDER BY timestamp DESC "
              "LIMIT 100",

              "CREATE VIEW IF NOT EXISTS recent_ids AS "
              "SELECT embedding_id, timestamp "
              "FROM signals "
              "ORDER BY timestamp DESC "
              "LIMIT 1024",

              "CREATE VIEW IF NOT EXISTS recent_retrievals AS "
              "SELECT memory_id, last_access as last_retrieval_ts "
              "FROM memories "
              "WHERE last_access IS NOT NULL "
              "  AND kind != 'WORKING' "
              "ORDER BY last_access DESC "
              "LIMIT 128",
          },
      },
      {
          3,
          "Long-term memory eviction audit log",
          {
              "CREATE TABLE IF NOT EXISTS memory_evictions ("
              "  eviction_id INTEGER PRIMARY KEY,"
              "  memory_id INTEGER NOT NULL,"
              "  embedding_id INTEGER,"
              "  source_id TEXT NOT NULL,"
              "  kind TEXT NOT NULL,"
              "  label TEXT,"
              "  start_ts INTEGER NOT NULL,"
              "  end_ts INTEGER,"
              "  created_at INTEGER NOT NULL,"
              "  last_access INTEGER,"
              "  strength REAL NOT NULL DEFAULT 0.0,"
              "  use_frequency REAL NOT NULL DEFAULT 0.0,"
              "  contextual_gain REAL NOT NULL DEFAULT 0.0,"
              "  retrieved_count INTEGER NOT NULL DEFAULT 0,"
              "  used_count INTEGER NOT NULL DEFAULT 0,"
              "  n_signals INTEGER NOT NULL DEFAULT 0,"
              "  modality TEXT NOT NULL DEFAULT 'text',"
              "  eviction_reason TEXT NOT NULL,"
              "  evicted_at INTEGER NOT NULL"
              ")",
              "CREATE INDEX IF NOT EXISTS idx_memory_evictions_time "
              "ON memory_evictions(evicted_at DESC, eviction_id DESC)",
              "CREATE INDEX IF NOT EXISTS idx_memory_evictions_memory "
              "ON memory_evictions(memory_id)",
              "CREATE INDEX IF NOT EXISTS idx_memory_evictions_reason "
              "ON memory_evictions(eviction_reason, kind)",
          },
      },
      {
          4,
          "Constructive recall reconstruction ledger",
          {
              "CREATE TABLE IF NOT EXISTS memory_reconstructions ("
              "  reconstruction_id INTEGER PRIMARY KEY,"
              "  memory_id INTEGER NOT NULL,"
              "  embedding_id INTEGER NOT NULL,"
              "  blob_id BLOB,"
              "  created_at INTEGER NOT NULL,"
              "  uncertainty REAL NOT NULL DEFAULT 0.0,"
              "  trigger TEXT NOT NULL DEFAULT 'retrieval',"
              "  source_confidence REAL NOT NULL DEFAULT 1.0,"
              "  context_similarity REAL NOT NULL DEFAULT 0.0"
              ")",
              "CREATE INDEX IF NOT EXISTS idx_memory_reconstructions_memory "
              "ON memory_reconstructions(memory_id, reconstruction_id DESC)",
              "CREATE INDEX IF NOT EXISTS idx_memory_reconstructions_embedding "
              "ON memory_reconstructions(embedding_id)",
          },
      },
      {
          5,
          "Meta-learning coefficient store",
          {
              "CREATE TABLE IF NOT EXISTS meta_learning_coeffs ("
              "  family TEXT PRIMARY KEY,"
              "  alpha_f REAL NOT NULL,"
              "  alpha_s REAL NOT NULL,"
              "  alpha_t REAL NOT NULL,"
              "  beta REAL NOT NULL,"
              "  a REAL NOT NULL,"
              "  b REAL NOT NULL,"
              "  update_count INTEGER NOT NULL DEFAULT 0,"
              "  updated_at INTEGER NOT NULL DEFAULT 0"
              ")",
          },
      },
      {
          6,
          "Soft Anchor shadow state and links",
          {
              "CREATE TABLE IF NOT EXISTS soft_anchors ("
              "  anchor_id TEXT PRIMARY KEY,"
              "  status TEXT NOT NULL,"
              "  semantic_centroid BLOB,"
              "  entity_centroid BLOB,"
              "  full_centroid BLOB,"
              "  semantic_radius REAL NOT NULL DEFAULT 0.0,"
              "  entity_radius REAL NOT NULL DEFAULT 0.0,"
              "  full_radius REAL NOT NULL DEFAULT 0.0,"
              "  source_id TEXT,"
              "  first_step INTEGER NOT NULL DEFAULT 0,"
              "  last_step INTEGER NOT NULL DEFAULT 0,"
              "  first_ts INTEGER NOT NULL DEFAULT 0,"
              "  last_ts INTEGER NOT NULL DEFAULT 0,"
              "  last_boundary_id INTEGER NOT NULL DEFAULT 0,"
              "  anchor_strength REAL NOT NULL DEFAULT 0.0,"
              "  support_count INTEGER NOT NULL DEFAULT 0,"
              "  contradiction_count INTEGER NOT NULL DEFAULT 0,"
              "  recent_memory_ids TEXT,"
              "  updated_at INTEGER NOT NULL DEFAULT 0"
              ")",
              "CREATE TABLE IF NOT EXISTS soft_anchor_links ("
              "  memory_id INTEGER NOT NULL,"
              "  anchor_id TEXT NOT NULL,"
              "  anchor_strength REAL NOT NULL DEFAULT 0.0,"
              "  anchor_label TEXT NOT NULL DEFAULT 'none',"
              "  evidence_kind TEXT NOT NULL DEFAULT 'unknown',"
              "  memory_tier TEXT NOT NULL DEFAULT 'WM',"
              "  score REAL NOT NULL DEFAULT 0.0,"
              "  margin REAL NOT NULL DEFAULT 0.0,"
              "  entropy REAL NOT NULL DEFAULT 0.0,"
              "  support_count INTEGER NOT NULL DEFAULT 0,"
              "  contradiction_count INTEGER NOT NULL DEFAULT 0,"
              "  created_step INTEGER NOT NULL DEFAULT 0,"
              "  updated_step INTEGER NOT NULL DEFAULT 0,"
              "  created_at INTEGER NOT NULL DEFAULT 0,"
              "  PRIMARY KEY (memory_id, anchor_id)"
              ")",
              "CREATE INDEX IF NOT EXISTS idx_soft_anchors_status "
              "ON soft_anchors(status, last_step)",
              "CREATE INDEX IF NOT EXISTS idx_soft_anchor_links_anchor "
              "ON soft_anchor_links(anchor_id, updated_step)",
              "CREATE INDEX IF NOT EXISTS idx_soft_anchor_links_memory "
              "ON soft_anchor_links(memory_id)",
          },
      },
      {
          7,
          "Memory embedding lookup index for graph retrieval",
          {
              "CREATE INDEX IF NOT EXISTS idx_memories_embedding "
              "ON memories(embedding_id)",
          },
      },
      {
          8,
          "Normalize memory source metadata",
          {
              "UPDATE memories "
              "SET source_origin = 'source', source_reliability = 0.7 "
              "WHERE source_origin IS NULL "
              "   OR source_origin != 'source' "
              "   OR source_reliability IN (0.5, 0.6, 0.8, 0.9)",
          },
      },
      {
          9,
          "Realtime retrieval lookup indexes",
          {
              "CREATE INDEX IF NOT EXISTS idx_signals_embedding "
              "ON signals(embedding_id)",
              "CREATE INDEX IF NOT EXISTS idx_signals_source_ts_serial "
              "ON signals(source_id, timestamp, serial_position)",
              "CREATE INDEX IF NOT EXISTS idx_memories_kind_source "
              "ON memories(kind, source_id)",
              "CREATE INDEX IF NOT EXISTS idx_memories_source_start "
              "ON memories(source_id, start_ts)",
              "CREATE INDEX IF NOT EXISTS idx_memories_last_access "
              "ON memories(last_access DESC) "
              "WHERE last_access IS NOT NULL AND kind != 'WORKING'",
              "CREATE INDEX IF NOT EXISTS idx_memories_label_created "
              "ON memories(created_at DESC) WHERE kind = 'LABEL'",
          },
      },
      {
          10,
          "Realtime graph retrieval query indexes",
          {
              "CREATE INDEX IF NOT EXISTS idx_associations_edge_source_target "
              "ON associations(edge_type, source_memory_id, target_memory_id)",
              "CREATE INDEX IF NOT EXISTS idx_associations_edge_target_source "
              "ON associations(edge_type, target_memory_id, source_memory_id)",
          },
      },
      {
          11,
          "Current memory embedding surface",
          {
              "CREATE VIRTUAL TABLE IF NOT EXISTS current_memory_embeddings "
              "USING vec0("
              "memory_id INTEGER PRIMARY KEY,"
              "embedding float[256],"
              "+embedding_id INTEGER,"
              "+created_at INTEGER"
              ")",
              "DELETE FROM current_memory_embeddings",
              "INSERT INTO current_memory_embeddings("
              "memory_id, embedding, embedding_id, created_at"
              ") "
              "WITH latest AS ("
              "  SELECT mr.memory_id, mr.embedding_id, mr.created_at "
              "  FROM memory_reconstructions mr "
              "  JOIN ("
              "    SELECT memory_id, MAX(reconstruction_id) AS reconstruction_id "
              "    FROM memory_reconstructions "
              "    GROUP BY memory_id"
              "  ) mx ON mx.memory_id = mr.memory_id "
              "       AND mx.reconstruction_id = mr.reconstruction_id"
              "), current AS ("
              "  SELECT m.memory_id, "
              "         COALESCE(latest.embedding_id, m.embedding_id) AS embedding_id, "
              "         COALESCE(latest.created_at, m.created_at, 0) AS created_at "
              "  FROM memories m "
              "  LEFT JOIN latest ON latest.memory_id = m.memory_id "
              "  WHERE COALESCE(latest.embedding_id, m.embedding_id) IS NOT NULL "
              "    AND m.kind NOT IN ('WORKING', 'LABEL')"
              ") "
              "SELECT current.memory_id, e.embedding, current.embedding_id, "
              "       current.created_at "
              "FROM current "
              "JOIN embeddings e ON e.embedding_id = current.embedding_id",
          },
      },
      {
          12,
          "Realtime memory recency lookup index",
          {
              "CREATE INDEX IF NOT EXISTS idx_memories_created_desc "
              "ON memories(created_at DESC)",
          },
      },
      {
          13,
          "Long-term memory eviction lookup index",
          {
              "CREATE INDEX IF NOT EXISTS idx_memories_ltm_strength_created "
              "ON memories(strength, created_at, embedding_id) "
              "WHERE kind = 'LONG_TERM'",
          },
      },
      {
          14,
          "Covering strength index for retention aggregate",
          {
              "DROP INDEX IF EXISTS idx_memories_strength",
              "CREATE INDEX IF NOT EXISTS idx_memories_strength "
              "ON memories(strength, last_access, created_at, start_ts)",
          },
      },
      {
          15,
          "Retrievable current embedding surface",
          {
              "DELETE FROM current_memory_embeddings "
              "WHERE memory_id NOT IN ("
              "  SELECT memory_id FROM memories "
              "  WHERE kind NOT IN ('WORKING', 'LABEL')"
              ")",
          },
      },
      {
          16,
          "Predictive pre-activation lookup index",
          {
              "CREATE INDEX IF NOT EXISTS idx_memories_pre_activation_embedding "
              "ON memories(pre_activation, embedding_id) "
              "WHERE pre_activation > 0.0",
              "CREATE INDEX IF NOT EXISTS idx_memories_pre_activation_embedding_active "
              "ON memories(embedding_id) "
              "WHERE pre_activation > 0.0",
          },
      },
      {
          17,
          "Association fanout order indexes",
          {
              "CREATE INDEX IF NOT EXISTS idx_associations_source_weight "
              "ON associations(source_memory_id, weight DESC, last_reinforced DESC, target_memory_id)",
              "CREATE INDEX IF NOT EXISTS idx_associations_target_weight "
              "ON associations(target_memory_id, weight DESC, last_reinforced DESC, source_memory_id)",
          },
      },
      {
          18,
          "Association edge-filtered fanout order indexes",
          {
              "CREATE INDEX IF NOT EXISTS idx_associations_source_edge_weight "
              "ON associations(source_memory_id, edge_type, weight DESC, last_reinforced DESC, target_memory_id)",
              "CREATE INDEX IF NOT EXISTS idx_associations_target_edge_weight "
              "ON associations(target_memory_id, edge_type, weight DESC, last_reinforced DESC, source_memory_id)",
          },
      },
      {
          19,
          "Long-term retention sample index",
          {
              "CREATE INDEX IF NOT EXISTS idx_memories_ltm_created_strength "
              "ON memories(created_at DESC, start_ts DESC, strength) "
              "WHERE kind = 'LONG_TERM'",
          },
      },
      {
          20,
          "Drop obsolete strength retention index",
          {
              "DROP INDEX IF EXISTS idx_memories_strength",
          },
      },
      {
          21,
          "Bound working-memory lookup indexes",
          {
              "DROP INDEX IF EXISTS idx_memories_working",
              "CREATE INDEX IF NOT EXISTS idx_memories_working_active "
              "ON memories(end_ts, memory_id) "
              "WHERE kind = 'WORKING' AND end_ts IS NULL",
              "CREATE INDEX IF NOT EXISTS idx_memories_working_closed_end "
              "ON memories(end_ts, memory_id) "
              "WHERE kind = 'WORKING' AND end_ts IS NOT NULL",
          },
      },
      {
          22,
          "Exclude working rows from durable recency lookups",
          {
              "DROP VIEW IF EXISTS recent_retrievals",
              "CREATE VIEW IF NOT EXISTS recent_retrievals AS "
              "SELECT memory_id, last_access as last_retrieval_ts "
              "FROM memories "
              "WHERE last_access IS NOT NULL "
              "  AND kind != 'WORKING' "
              "ORDER BY last_access DESC "
              "LIMIT 128",
              "DROP INDEX IF EXISTS idx_memories_last_access",
              "CREATE INDEX IF NOT EXISTS idx_memories_last_access "
              "ON memories(last_access DESC) "
              "WHERE last_access IS NOT NULL AND kind != 'WORKING'",
          },
      },
      {
          23,
          "Predictive pre-activation active embedding lookup",
          {
              "CREATE INDEX IF NOT EXISTS idx_memories_pre_activation_embedding_active "
              "ON memories(embedding_id) "
              "WHERE pre_activation > 0.0",
          },
      },
      {
          24,
          "Track working-memory strength decay timestamp",
          {
              "ALTER TABLE memories ADD COLUMN strength_updated_at INTEGER",
              "UPDATE memories SET strength_updated_at = last_access "
              "WHERE kind = 'WORKING' AND strength_updated_at IS NULL",
          },
      },
      {
          25,
          "Persist consolidation throughput range",
          {
              "ALTER TABLE state ADD COLUMN consolidation_rate_floor REAL NOT NULL DEFAULT 0.0",
              "ALTER TABLE state ADD COLUMN consolidation_rate_peak REAL NOT NULL DEFAULT 0.0",
          },
      },
      {
          26,
          "Persist consolidation throughput initialization",
          {
              "ALTER TABLE state ADD COLUMN consolidation_rate_initialized INTEGER NOT NULL DEFAULT 0",
              "UPDATE state SET consolidation_rate_initialized = 1 "
              "WHERE consolidation_rate_floor != 0.0 "
              "OR consolidation_rate_peak != 0.0",
          },
      },
      {
          27,
          "Persist consolidation throughput recommendation latch",
          {
              "ALTER TABLE state ADD COLUMN consolidation_rate_armed INTEGER NOT NULL DEFAULT 1",
          },
      },
  };
}

void
ApplyMigrationsThrough (Store &store, int64_t maximum_id)
{
  // Migrations must own the write lock before reading the applied set. Use
  // direct SQL so this does not depend on the configurable default transaction
  // mode used by normal processing transactions.
  store.Execute ("BEGIN IMMEDIATE");
  bool committed = false;

  try
    {
      store.Execute (
          "CREATE TABLE IF NOT EXISTS cortext_schema_migrations ("
          "  id INTEGER PRIMARY KEY,"
          "  description TEXT,"
          "  applied_at INTEGER"
          ")");

      std::set<int64_t> applied_ids = GetAppliedMigrations (store);
      auto migrations = GetCoreMigrations ();

      std::sort (migrations.begin (), migrations.end (),
                 [] (const Migration &a, const Migration &b) {
                   return a.id < b.id;
                 });

      for (const auto &m : migrations)
        {
          if (m.id > maximum_id || applied_ids.count (m.id))
            {
              continue;
            }
          ApplySingleMigrationInTransaction (store, m);
        }

      store.Execute ("COMMIT");
      committed = true;
    }
  catch (...)
    {
      if (!committed)
        {
          try
            {
              store.Execute ("ROLLBACK");
            }
          catch (...)
            {
            }
        }
      throw;
    }
}

} // namespace

#if defined(CORTEXT_TESTING)
std::set<int64_t>
DebugGetAppliedMigrationIdsForTest (Store &store)
{
  return GetAppliedMigrations (store);
}

void
DebugApplyCoreMigrationsThroughForTest (Store &store, int64_t maximum_id)
{
  ApplyMigrationsThrough (store, maximum_id);
}
#endif

bool
TableExists (Store &store, const std::string &table_name)
{
  auto rows = store.Execute (
      "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
      { table_name });
  return !rows.empty ();
}

bool
ColumnExists (Store &store, const std::string &table_name,
              const std::string &column_name)
{
  // Quote the table identifier to prevent injection
  std::string quoted_table = "\"";
  for (char c : table_name)
    {
      if (c == '"')
        quoted_table += "\"\"";
      else
        quoted_table += c;
    }
  quoted_table += "\"";

  // PRAGMA table_info returns one row per column.
  // Columns: cid, name, type, notnull, dflt_value, pk
  auto rows
      = store.Execute ("PRAGMA table_info(" + quoted_table + ")", {});
  for (const auto &row : rows)
    {
      auto it = row.find ("name");
      if (it != row.end ())
        {
          // Check for string or text type in the variant/any
          if (it->second.type () == typeid (std::string))
            {
              if (std::any_cast<std::string> (it->second) == column_name)
                return true;
            }
        }
    }
  return false;
}

void
ApplyMigrations (Store &store)
{
  ApplyMigrationsThrough (store, std::numeric_limits<int64_t>::max ());
}

} // namespace cortext::store
