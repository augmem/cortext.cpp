#include "cortext/store/schema.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>

namespace cortext::store
{

namespace
{

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

} // namespace

void
SchemaRegistry::Register (Migration migration)
{
  migrations_.push_back (std::move (migration));
}

std::vector<Migration>
SchemaRegistry::GetSortedMigrations () const
{
  std::vector<Migration> sorted = migrations_;
  std::sort (sorted.begin (), sorted.end (),
             [] (const Migration &a, const Migration &b) {
               return a.id < b.id;
             });
  return sorted;
}

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
           if (it->second.type() == typeid(std::string)) {
               if (std::any_cast<std::string>(it->second) == column_name) return true;
           }
        }
    }
  return false;
}

void
RegisterCoreSchema (SchemaRegistry &registry)
{
  // ==========================================================================
  // Migration 0: Complete schema per docs/diagrams/database.md
  // Clean slate - all tables in single migration
  // ==========================================================================
  registry.Register ({
      0,
      "Core schema (all tables per docs/diagrams/database.md)",
      {
          // ------------------------------------------------------------------
          // sqlite-objstore virtual table
          // ------------------------------------------------------------------
          "CREATE VIRTUAL TABLE IF NOT EXISTS objstore USING objstore()",

          // ------------------------------------------------------------------
          // EMBEDDINGS - sqlite-vec virtual table with auxiliary columns
          // (Section 3.1, 5.1-5.4) - single unified table per database.md
          // Note: sqlite-vec has max 16 auxiliary columns, doesn't support DEFAULTs
          // embedding_id INTEGER PRIMARY KEY aliases rowid (doesn't count as auxiliary)
          // ------------------------------------------------------------------
          "CREATE VIRTUAL TABLE IF NOT EXISTS embeddings USING vec0("
          "embedding_id INTEGER PRIMARY KEY,"
          "embedding float[256],"
          "+type text,"
          "+strength float,"
          "+use_frequency float,"
          "+stability float,"
          "+connectivity float,"
          "+drift_mag float,"
          "+influence float,"
          "+sustained_influence float,"
          "+contextual_gain float,"
          "+redundancy float,"
          "+pre_activation float,"
          "+lability_state float,"
          "+suppression_count integer,"
          "+cluster_id integer,"
          "+last_access integer,"
          "+created_at integer"
          ")",

          // ------------------------------------------------------------------
          // MEMORIES - Memory metadata + payload reference (Section 2.2.2, 6.6)
          // ------------------------------------------------------------------
          "CREATE TABLE IF NOT EXISTS memories ("
          "  embedding_id INTEGER PRIMARY KEY,"
          "  source_id TEXT,"
          "  modality TEXT,"
          "  mime TEXT,"
          "  timestamp INTEGER,"
          "  blob_id BLOB,"
          "  event_emotion BLOB,"
          "  ambient_mood BLOB,"
          "  episode_id INTEGER,"
          "  serial_position INTEGER,"
          "  width INTEGER,"
          "  height INTEGER,"
          "  channels INTEGER,"
          "  sample_rate INTEGER,"
          "  num_samples INTEGER"
          ")",

          // ------------------------------------------------------------------
          // MEMORY_FEEDBACK - Retrieval/usage tracking (Section 5.2, 5.4, 6.3, 6.4)
          // ------------------------------------------------------------------
          "CREATE TABLE IF NOT EXISTS memory_feedback ("
          "  embedding_id INTEGER PRIMARY KEY,"
          "  retrieved_count INTEGER NOT NULL DEFAULT 0,"
          "  used_count INTEGER NOT NULL DEFAULT 0,"
          "  influence_factor REAL NOT NULL DEFAULT 0.0,"
          "  mean_influence REAL NOT NULL DEFAULT 0.0,"
          "  last_used INTEGER NOT NULL DEFAULT 0,"
          "  original_embedding BLOB,"
          "  lability_ts INTEGER,"
          "  recovery_time_start INTEGER"
          ")",

          // ------------------------------------------------------------------
          // GRAPH_NODES - Knowledge graph nodes (Section 7.4-7.6)
          // ------------------------------------------------------------------
          "CREATE TABLE IF NOT EXISTS graph_nodes ("
          "  node_id TEXT PRIMARY KEY,"
          "  type TEXT NOT NULL,"
          "  label TEXT,"
          "  embedding_id INTEGER,"
          "  metadata TEXT,"
          "  created_at INTEGER"
          ")",

          // ------------------------------------------------------------------
          // GRAPH_EDGES - Knowledge graph edges (Section 7.4-7.6)
          // ------------------------------------------------------------------
          "CREATE TABLE IF NOT EXISTS graph_edges ("
          "  source_id TEXT NOT NULL,"
          "  target_id TEXT NOT NULL,"
          "  edge_type TEXT NOT NULL,"
          "  weight REAL NOT NULL DEFAULT 1.0,"
          "  decay_rate REAL,"
          "  last_reinforced INTEGER,"
          "  PRIMARY KEY (source_id, target_id, edge_type)"
          ")",
          "CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source_id)",
          "CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target_id)",
          "CREATE INDEX IF NOT EXISTS idx_graph_edges_type ON graph_edges(edge_type)",

          // ------------------------------------------------------------------
          // CONSOLIDATION_SUMMARIES - Cluster summaries (Section 7.3)
          // ------------------------------------------------------------------
          "CREATE TABLE IF NOT EXISTS consolidation_summaries ("
          "  summary_id TEXT PRIMARY KEY,"
          "  summary_text TEXT,"
          "  centroid BLOB,"
          "  cluster_size INTEGER"
          ")",

          // ------------------------------------------------------------------
          // CONSOLIDATION_SOURCES - Source tracking (Section 7.3)
          // ------------------------------------------------------------------
          "CREATE TABLE IF NOT EXISTS consolidation_sources ("
          "  summary_id TEXT,"
          "  source_embedding_id INTEGER"
          ")",

          // ------------------------------------------------------------------
          // EXTRACTION_ENTITIES - Named entities (Section 7.4)
          // ------------------------------------------------------------------
          "CREATE TABLE IF NOT EXISTS extraction_entities ("
          "  summary_id TEXT NOT NULL,"
          "  name TEXT NOT NULL,"
          "  type TEXT NOT NULL,"
          "  salience REAL,"
          "  embedding_id INTEGER,"
          "  PRIMARY KEY (summary_id, name, type)"
          ")",

          // ------------------------------------------------------------------
          // EXTRACTION_RELATIONS - Semantic relations (Section 7.5)
          // ------------------------------------------------------------------
          "CREATE TABLE IF NOT EXISTS extraction_relations ("
          "  summary_id TEXT NOT NULL,"
          "  subject TEXT NOT NULL,"
          "  predicate TEXT NOT NULL,"
          "  object TEXT NOT NULL,"
          "  confidence REAL"
          ")",

          // ------------------------------------------------------------------
          // ENTITY_INDEX - Name to node mapping (Section 7.5)
          // ------------------------------------------------------------------
          "CREATE TABLE IF NOT EXISTS entity_index ("
          "  name TEXT PRIMARY KEY,"
          "  node_id TEXT NOT NULL"
          ")",

          // ------------------------------------------------------------------
          // GOAL_NODES - Goal alignment targets
          // ------------------------------------------------------------------
          "CREATE TABLE IF NOT EXISTS goal_nodes ("
          "  node_id TEXT PRIMARY KEY"
          ")",

          // ------------------------------------------------------------------
          // RIF_STATE - Retrieval-Induced Forgetting (Section 8.4)
          // ------------------------------------------------------------------
          "CREATE TABLE IF NOT EXISTS rif_state ("
          "  embedding_id INTEGER PRIMARY KEY,"
          "  suppression REAL NOT NULL DEFAULT 0.0,"
          "  ts INTEGER"
          ")",

          // ------------------------------------------------------------------
          // EMOTIONAL_TAGS - Flashbulb/emotional consolidation (Section 8.7)
          // ------------------------------------------------------------------
          "CREATE TABLE IF NOT EXISTS emotional_tags ("
          "  embedding_id INTEGER PRIMARY KEY,"
          "  flashbulb INTEGER NOT NULL DEFAULT 0,"
          "  intensity REAL NOT NULL DEFAULT 0.0,"
          "  arousal REAL NOT NULL DEFAULT 0.0,"
          "  valence REAL NOT NULL DEFAULT 0.5,"
          "  half_life_bonus REAL NOT NULL DEFAULT 0.0,"
          "  detail_suppression REAL NOT NULL DEFAULT 0.0,"
          "  gist_components INTEGER NOT NULL DEFAULT 0,"
          "  cascade_radius INTEGER NOT NULL DEFAULT 0,"
          "  cascade_decay REAL NOT NULL DEFAULT 0.0,"
          "  flashbulb_threshold_eff REAL NOT NULL DEFAULT 0.0,"
          "  ts INTEGER"
          ")",

          // ------------------------------------------------------------------
          // CONSOLIDATION_CANDIDATES - Scoring candidates (Section 9.2)
          // ------------------------------------------------------------------
          "CREATE TABLE IF NOT EXISTS consolidation_candidates ("
          "  embedding_id INTEGER PRIMARY KEY,"
          "  score REAL NOT NULL,"
          "  created_at INTEGER,"
          "  reason TEXT"
          ")",

          // ------------------------------------------------------------------
          // EPISODES - Episodic boundaries (Section 3.1.3)
          // ------------------------------------------------------------------
          "CREATE TABLE IF NOT EXISTS episodes ("
          "  id INTEGER PRIMARY KEY,"
          "  start_ts INTEGER NOT NULL,"
          "  end_ts INTEGER,"
          "  boundary_type TEXT,"
          "  centroid BLOB"
          ")",
          "CREATE INDEX IF NOT EXISTS idx_episodes_start ON episodes(start_ts)",

          // ------------------------------------------------------------------
          // SIGNAL_METRICS - Per-signal composite scoring (Section 3.1-3.3)
          // ------------------------------------------------------------------
          "CREATE TABLE IF NOT EXISTS signal_metrics ("
          "  id INTEGER PRIMARY KEY,"
          "  timestamp INTEGER NOT NULL,"
          "  embedding_id INTEGER,"
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
          "  composite_score REAL,"
          "  threshold_t REAL,"
          "  write_decision INTEGER,"
          "  coherence REAL,"
          "  focus_spread REAL,"
          "  f_effective REAL"
          ")",
          "CREATE INDEX IF NOT EXISTS idx_signal_metrics_ts ON signal_metrics(timestamp)",

          // ------------------------------------------------------------------
          // PROCESSOR_STATE - Full algorithm state (singleton, id=1)
          // ------------------------------------------------------------------
          "CREATE TABLE IF NOT EXISTS processor_state ("
          "  id INTEGER PRIMARY KEY CHECK (id = 1),"
          "  signals_processed INTEGER NOT NULL DEFAULT 0,"
          // Threshold state (Section 4)
          "  theta_dynamic REAL NOT NULL DEFAULT 0.2,"
          "  theta_target REAL NOT NULL DEFAULT 0.2,"
          "  hysteresis REAL NOT NULL DEFAULT 0.05,"
          "  half_life REAL NOT NULL DEFAULT 120.0,"
          // Focus state (Section 2.1)
          "  weight_relevance REAL NOT NULL DEFAULT 0.5,"
          "  weight_relevance_prior REAL NOT NULL DEFAULT 0.5,"
          "  attention_width REAL NOT NULL DEFAULT 1.57,"
          "  attention_width_prior REAL NOT NULL DEFAULT 1.57,"
          "  coverage_gain_floor REAL NOT NULL DEFAULT 0.65,"
          "  coverage_gain_floor_prior REAL NOT NULL DEFAULT 0.65,"
          "  mismatch_weight REAL NOT NULL DEFAULT 0.5,"
          "  mismatch_weight_prior REAL NOT NULL DEFAULT 0.5,"
          // Sensitivity state (Section 2.2)
          "  weight_novelty REAL NOT NULL DEFAULT 0.3,"
          "  weight_novelty_prior REAL NOT NULL DEFAULT 0.3,"
          "  weight_surprise REAL NOT NULL DEFAULT 0.2,"
          "  weight_surprise_prior REAL NOT NULL DEFAULT 0.2,"
          "  weight_valence REAL NOT NULL DEFAULT 0.4,"
          "  weight_valence_prior REAL NOT NULL DEFAULT 0.4,"
          "  weight_arousal REAL NOT NULL DEFAULT 0.0,"
          "  weight_arousal_prior REAL NOT NULL DEFAULT 0.0,"
          "  emotion_gain REAL NOT NULL DEFAULT 1.0,"
          "  emotion_gain_prior REAL NOT NULL DEFAULT 1.0,"
          "  score_gain REAL NOT NULL DEFAULT 1.0,"
          "  score_gain_prior REAL NOT NULL DEFAULT 1.0,"
          "  rate_target REAL NOT NULL DEFAULT 0.2,"
          "  rate_target_prior REAL NOT NULL DEFAULT 0.2,"
          "  base_rate_prior REAL NOT NULL DEFAULT 0.2,"
          "  weight_emotion_prior REAL NOT NULL DEFAULT 0.2,"
          // Emotion state (Section 2.2.2-4)
          "  emotion_intensity REAL NOT NULL DEFAULT 0.0,"
          "  valence REAL NOT NULL DEFAULT 0.5,"
          "  arousal REAL NOT NULL DEFAULT 0.0,"
          "  mood_vector BLOB,"
          // Stability state (Section 2.3)
          "  rate_decay REAL NOT NULL DEFAULT 0.60,"
          "  rate_decay_prior REAL NOT NULL DEFAULT 0.60,"
          "  periphery_half_life REAL NOT NULL DEFAULT 120.0,"
          "  periphery_half_life_prior REAL NOT NULL DEFAULT 120.0,"
          "  salience_half_life REAL NOT NULL DEFAULT 120.0,"
          "  salience_half_life_prior REAL NOT NULL DEFAULT 120.0,"
          "  drift_weight REAL NOT NULL DEFAULT 0.5,"
          "  drift_weight_prior REAL NOT NULL DEFAULT 0.5,"
          "  retention_ema REAL NOT NULL DEFAULT 0.0,"
          "  hysteresis_band_prior REAL NOT NULL DEFAULT 0.02,"
          "  half_life_prior REAL NOT NULL DEFAULT 120.0,"
          // Rate control (Section 4.2)
          "  m_rate REAL NOT NULL DEFAULT 0.0,"
          "  dt_ema REAL NOT NULL DEFAULT 0.0,"
          "  rate_ticks INTEGER NOT NULL DEFAULT 0,"
          "  last_rate_timestamp INTEGER NOT NULL DEFAULT 0,"
          "  reliability REAL NOT NULL DEFAULT 1.0,"
          // Uncertainty (Section 1.4)
          "  u_uncertainty REAL NOT NULL DEFAULT 0.0,"
          // Embedding prediction (Section 3.1.4)
          "  last_embedding BLOB,"
          "  delta_x_trend BLOB,"
          "  delta_half_life_adj REAL NOT NULL DEFAULT 0.0,"
          "  sustained_influence REAL NOT NULL DEFAULT 0.0,"
          // Working memory (Section 6.1)
          "  wm_maintenance_cost REAL NOT NULL DEFAULT 0.0,"
          "  wm_slot_count INTEGER NOT NULL DEFAULT 0,"
          "  wm_last_accepted INTEGER NOT NULL DEFAULT 0,"
          "  wm_last_chunked INTEGER NOT NULL DEFAULT 0,"
          // Metacognition (Section 6.2)
          "  fok_state REAL NOT NULL DEFAULT 0.0,"
          "  retrieval_strength REAL NOT NULL DEFAULT 0.0,"
          "  metacognitive_confidence REAL NOT NULL DEFAULT 0.0,"
          // Consolidation (Section 7.1)
          "  last_consolidation_ts INTEGER NOT NULL DEFAULT 0,"
          "  consolidation_count INTEGER NOT NULL DEFAULT 0,"
          "  is_processing_signal INTEGER NOT NULL DEFAULT 0,"
          "  last_retrieval_ts INTEGER NOT NULL DEFAULT 0,"
          // Streaming pacing (Section 10)
          "  drift_accum REAL NOT NULL DEFAULT 0.0,"
          "  drift_at_last_interrupt REAL NOT NULL DEFAULT 0.0,"
          // Episode tracking (Section 3.1.3)
          "  episode_start_ts INTEGER NOT NULL DEFAULT 0,"
          "  last_interrupt_tick INTEGER NOT NULL DEFAULT 0,"
          // Initialization flags
          "  focus_priors_initialized INTEGER NOT NULL DEFAULT 0,"
          "  sensitivity_priors_initialized INTEGER NOT NULL DEFAULT 0,"
          "  stability_priors_initialized INTEGER NOT NULL DEFAULT 0,"
          // Timestamps
          "  last_signal_timestamp INTEGER NOT NULL DEFAULT 0,"
          "  updated_at INTEGER NOT NULL DEFAULT 0,"
          // Rate window (Section 2.2)
          "  write_rate_timestamps BLOB"
          ")",

          // ------------------------------------------------------------------
          // BLENDER_WEIGHTS - RLS-fitted metric weights (Section 3.2)
          // ------------------------------------------------------------------
          "CREATE TABLE IF NOT EXISTS blender_weights ("
          "  id INTEGER PRIMARY KEY CHECK (id = 1),"
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
          "  update_count INTEGER NOT NULL DEFAULT 0"
          ")",

          // ------------------------------------------------------------------
          // BLENDER_COVARIANCE - RLS P matrix 12x12 (Section 3.2)
          // ------------------------------------------------------------------
          "CREATE TABLE IF NOT EXISTS blender_covariance ("
          "  id INTEGER PRIMARY KEY CHECK (id = 1),"
          "  P_matrix BLOB"
          ")",

          // ------------------------------------------------------------------
          // BLENDER_COEFFICIENTS - RLS learned coefficients (Section 3.2)
          // ------------------------------------------------------------------
          "CREATE TABLE IF NOT EXISTS blender_coefficients ("
          "  id INTEGER PRIMARY KEY CHECK (id = 1),"
          "  coefficients BLOB,"
          "  update_count INTEGER NOT NULL DEFAULT 0"
          ")",

          // ------------------------------------------------------------------
          // BLENDER_COEFF_COVARIANCE - RLS P matrix 48x48 (Section 3.2)
          // ------------------------------------------------------------------
          "CREATE TABLE IF NOT EXISTS blender_coeff_covariance ("
          "  id INTEGER PRIMARY KEY CHECK (id = 1),"
          "  P_matrix BLOB"
          ")",

          // ------------------------------------------------------------------
          // RECENT_CONTEXT - Sliding window of context embeddings
          // ------------------------------------------------------------------
          "CREATE TABLE IF NOT EXISTS recent_context ("
          "  id INTEGER PRIMARY KEY,"
          "  embedding BLOB NOT NULL,"
          "  timestamp INTEGER NOT NULL,"
          "  seq_order INTEGER NOT NULL"
          ")",
          "CREATE INDEX IF NOT EXISTS idx_recent_context_seq ON recent_context(seq_order)",

          // ------------------------------------------------------------------
          // RECENT_SCORES - Sliding window of composite scores
          // ------------------------------------------------------------------
          "CREATE TABLE IF NOT EXISTS recent_scores ("
          "  id INTEGER PRIMARY KEY,"
          "  score REAL NOT NULL,"
          "  timestamp INTEGER NOT NULL"
          ")",
          "CREATE INDEX IF NOT EXISTS idx_recent_scores_ts ON recent_scores(timestamp)",

          // ------------------------------------------------------------------
          // OBSERVED_RETENTION_HISTORY - Retention observations (Section 2.3.2)
          // ------------------------------------------------------------------
          "CREATE TABLE IF NOT EXISTS observed_retention_history ("
          "  id INTEGER PRIMARY KEY,"
          "  retention_value REAL NOT NULL,"
          "  timestamp INTEGER NOT NULL"
          ")",
          "CREATE INDEX IF NOT EXISTS idx_retention_history_ts ON observed_retention_history(timestamp)",

          // ------------------------------------------------------------------
          // WORKING_MEMORY_SLOTS - Session persistence (Section 6.1)
          // ------------------------------------------------------------------
          "CREATE TABLE IF NOT EXISTS working_memory_slots ("
          "  id INTEGER PRIMARY KEY,"
          "  slot_index INTEGER NOT NULL,"
          "  embedding_id INTEGER,"
          "  strength REAL NOT NULL DEFAULT 0.0,"
          "  timestamp INTEGER NOT NULL,"
          "  embedding BLOB NOT NULL"
          ")",
          "CREATE INDEX IF NOT EXISTS idx_wm_slots_index ON working_memory_slots(slot_index)",

          // ------------------------------------------------------------------
          // RECENT_IDS - LRU tracking for novelty detection (Section 8.6, Algorithm 27)
          // ------------------------------------------------------------------
          "CREATE TABLE IF NOT EXISTS recent_ids ("
          "  id INTEGER PRIMARY KEY,"
          "  embedding_id INTEGER NOT NULL,"
          "  access_type INTEGER NOT NULL,"  // 0=inserted, 1=retrieved
          "  timestamp INTEGER NOT NULL,"
          "  seq_order INTEGER NOT NULL"
          ")",
          "CREATE INDEX IF NOT EXISTS idx_recent_ids_embedding ON recent_ids(embedding_id)",
          "CREATE INDEX IF NOT EXISTS idx_recent_ids_seq ON recent_ids(seq_order)",

          // ------------------------------------------------------------------
          // RECENT_RETRIEVALS - Implicit feedback cache (Section 5.2)
          // ------------------------------------------------------------------
          "CREATE TABLE IF NOT EXISTS recent_retrievals ("
          "  id INTEGER PRIMARY KEY,"
          "  embedding_id INTEGER NOT NULL,"
          "  timestamp INTEGER NOT NULL,"
          "  seq_order INTEGER NOT NULL"
          ")",
          "CREATE INDEX IF NOT EXISTS idx_recent_retrievals_embedding ON recent_retrievals(embedding_id)",
          "CREATE INDEX IF NOT EXISTS idx_recent_retrievals_seq ON recent_retrievals(seq_order)",
      },
  });
}

void
ApplyMigrations (Store &store, const SchemaRegistry &registry)
{
  store.Execute (
      "CREATE TABLE IF NOT EXISTS cortext_schema_migrations ("
      "  id INTEGER PRIMARY KEY,"
      "  description TEXT,"
      "  applied_at INTEGER"
      ")");
  std::set<int> applied_ids = GetAppliedMigrations (store);
  SchemaRegistry full_registry = registry;
  RegisterCoreSchema(full_registry);
  auto migrations = full_registry.GetSortedMigrations ();
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
