#include "cortext/store/schema.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <algorithm>
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
  int id;                                // Unique integer ID (e.g., 2024010101)
  std::string description;               // Human-readable description
  std::vector<std::string> up_statements; // SQL statements to apply
};

/// @brief Reads the set of already-applied migration IDs from the tracking table.
///
/// Migration IDs are monotonically increasing integers (e.g., YYYYMMDDnn format)
/// that must be unique across all registered migrations to prevent collisions.
std::set<int>
GetAppliedMigrations (Store &store)
{
  std::set<int> applied_ids;
  auto rows = store.Execute ("SELECT id FROM cortext_schema_migrations");
  for (const auto &row : rows)
    {
      if (row.count ("id"))
        {
          auto val = row.at ("id");
          if (val.type () == typeid (int))
            applied_ids.insert (std::any_cast<int> (val));
          else if (val.type () == typeid (long long))
            applied_ids.insert (static_cast<int> (std::any_cast<long long> (val)));
        }
    }
  return applied_ids;
}

void
ApplySingleMigration (Store &store, const Migration &m)
{
  auto tx = store.Begin ();
  try
    {
      for (const auto &sql : m.up_statements)
        {
          try
            {
              tx->Execute (sql);
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
      tx->Execute (
          "INSERT INTO cortext_schema_migrations (id, description, applied_at) "
          "VALUES (?, ?, strftime('%s', 'now'))",
          { m.id, m.description });
      tx->Commit ();
    }
  catch (const std::exception &e)
    {
      tx->Rollback ();
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
              // nodes: episodic memories, summaries/cues, labels, and working
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
              // Metacognition
              "  fok_state REAL NOT NULL DEFAULT 0.0,"
              "  retrieval_strength REAL NOT NULL DEFAULT 0.0,"
              "  metacognitive_confidence REAL NOT NULL DEFAULT 0.0,"
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
              "CREATE INDEX IF NOT EXISTS idx_memories_working ON memories(end_ts) WHERE kind = 'WORKING'",
              "CREATE INDEX IF NOT EXISTS idx_memories_strength ON memories(strength, last_access)",
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
              "ORDER BY last_access DESC "
              "LIMIT 128",
          },
      },
      {
          1,
          "Bitemporal fact assertions and retrieval cache",
          {
              "CREATE TABLE IF NOT EXISTS fact_assertions ("
              "  fact_id INTEGER PRIMARY KEY,"
              "  subject TEXT NOT NULL,"
              "  predicate TEXT NOT NULL,"
              "  object TEXT NOT NULL,"
              "  canonical_subject TEXT NOT NULL,"
              "  canonical_predicate TEXT NOT NULL,"
              "  canonical_object TEXT NOT NULL,"
              "  valid_start_ts INTEGER,"
              "  valid_end_ts INTEGER,"
              "  recorded_at_ts INTEGER NOT NULL,"
              "  superseded_at_ts INTEGER,"
              "  confidence REAL NOT NULL DEFAULT 0.5,"
              "  summary_memory_id INTEGER NOT NULL,"
              "  created_at INTEGER NOT NULL"
              ")",

              "CREATE TABLE IF NOT EXISTS fact_evidence ("
              "  fact_id INTEGER NOT NULL,"
              "  source_memory_id INTEGER NOT NULL,"
              "  evidence_type TEXT NOT NULL,"
              "  support_weight REAL NOT NULL DEFAULT 1.0,"
              "  PRIMARY KEY (fact_id, source_memory_id, evidence_type)"
              ")",

              "CREATE TABLE IF NOT EXISTS fact_cache ("
              "  fact_id INTEGER PRIMARY KEY,"
              "  embedding_id INTEGER,"
              "  fact_text TEXT NOT NULL,"
              "  is_current INTEGER NOT NULL DEFAULT 0,"
              "  valid_start_ts INTEGER,"
              "  valid_end_ts INTEGER,"
              "  recorded_at_ts INTEGER NOT NULL,"
              "  superseded_at_ts INTEGER,"
              "  updated_at INTEGER NOT NULL"
              ")",

              "CREATE INDEX IF NOT EXISTS idx_fact_assertions_canonical "
              "ON fact_assertions(canonical_subject, canonical_predicate, canonical_object)",
              "CREATE INDEX IF NOT EXISTS idx_fact_assertions_open "
              "ON fact_assertions(canonical_subject, canonical_predicate, superseded_at_ts, valid_end_ts)",
              "CREATE INDEX IF NOT EXISTS idx_fact_assertions_valid "
              "ON fact_assertions(valid_start_ts, valid_end_ts)",
              "CREATE INDEX IF NOT EXISTS idx_fact_assertions_recorded "
              "ON fact_assertions(recorded_at_ts, superseded_at_ts)",
              "CREATE INDEX IF NOT EXISTS idx_fact_assertions_summary "
              "ON fact_assertions(summary_memory_id)",
              "CREATE INDEX IF NOT EXISTS idx_fact_evidence_source "
              "ON fact_evidence(source_memory_id, evidence_type)",
              "CREATE INDEX IF NOT EXISTS idx_fact_cache_current "
              "ON fact_cache(is_current, recorded_at_ts)",
          },
      },
      {
          2,
          "Fact lifecycle support, archival, and bounded retention",
          {
              "ALTER TABLE fact_assertions "
              "ADD COLUMN support_mass REAL NOT NULL DEFAULT 0.0",
              "ALTER TABLE fact_assertions "
              "ADD COLUMN source_diversity INTEGER NOT NULL DEFAULT 0",
              "ALTER TABLE fact_assertions "
              "ADD COLUMN contradiction_mass REAL NOT NULL DEFAULT 0.0",
              "ALTER TABLE fact_assertions "
              "ADD COLUMN confirmation_count INTEGER NOT NULL DEFAULT 0",
              "ALTER TABLE fact_assertions "
              "ADD COLUMN challenge_count INTEGER NOT NULL DEFAULT 0",
              "ALTER TABLE fact_assertions "
              "ADD COLUMN compressed_support_count INTEGER NOT NULL DEFAULT 0",
              "ALTER TABLE fact_assertions "
              "ADD COLUMN last_confirmation_ts INTEGER NOT NULL DEFAULT 0",
              "ALTER TABLE fact_assertions "
              "ADD COLUMN last_challenge_ts INTEGER",
              "ALTER TABLE fact_assertions "
              "ADD COLUMN severity_class TEXT NOT NULL DEFAULT 'medium'",
              "ALTER TABLE fact_assertions "
              "ADD COLUMN lifecycle_state TEXT NOT NULL DEFAULT 'active'",
              "ALTER TABLE fact_assertions "
              "ADD COLUMN archived_at INTEGER",
              "ALTER TABLE fact_assertions "
              "ADD COLUMN last_maintenance_ts INTEGER NOT NULL DEFAULT 0",
              "ALTER TABLE fact_cache "
              "ADD COLUMN lifecycle_state TEXT NOT NULL DEFAULT 'active'",
              "ALTER TABLE fact_cache "
              "ADD COLUMN support_mass REAL NOT NULL DEFAULT 0.0",
              "CREATE INDEX IF NOT EXISTS idx_fact_assertions_lifecycle "
              "ON fact_assertions(lifecycle_state, severity_class, archived_at)",
              "CREATE INDEX IF NOT EXISTS idx_fact_assertions_maintenance "
              "ON fact_assertions(last_maintenance_ts, recorded_at_ts)",
              "CREATE INDEX IF NOT EXISTS idx_fact_cache_lifecycle "
              "ON fact_cache(lifecycle_state, is_current, updated_at)",
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
          "Neutral source metadata and LTM deep summaries",
          {
              "UPDATE memories "
              "SET source_origin = 'source', source_reliability = 0.7 "
              "WHERE source_origin IS NULL "
              "   OR source_origin != 'source' "
              "   OR source_reliability IN (0.5, 0.6, 0.8, 0.9)",
              "UPDATE memories "
              "SET kind = 'LONG_TERM' "
              "WHERE kind = 'ASSOCIATION' AND source_id LIKE 'summary_%'",
          },
      },
  };
}

} // namespace

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
  store.Execute (
      "CREATE TABLE IF NOT EXISTS cortext_schema_migrations ("
      "  id INTEGER PRIMARY KEY,"
      "  description TEXT,"
      "  applied_at INTEGER"
      ")");

  std::set<int> applied_ids = GetAppliedMigrations (store);
  auto migrations = GetCoreMigrations ();

  // Sort by ID (should already be sorted, but be explicit)
  std::sort (migrations.begin (), migrations.end (),
             [] (const Migration &a, const Migration &b) {
               return a.id < b.id;
             });

  for (const auto &m : migrations)
    {
      if (applied_ids.count (m.id))
        {
          continue;
        }
      ApplySingleMigration (store, m);
    }
}

} // namespace cortext::store
