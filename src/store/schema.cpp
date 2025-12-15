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
  registry.Register ({
      1,
      "Core tables (memory_index, memory_feedback, embeddings)",
      {
          // Embeddings table (vector store)
          "CREATE TABLE IF NOT EXISTS embeddings ("
          "  embedding_id INTEGER PRIMARY KEY,"
          "  embedding BLOB"
          ")",

          // Memory index (metadata + payload pointer)
          "CREATE TABLE IF NOT EXISTS memory_index ("
          "  embedding_id INTEGER PRIMARY KEY,"
          "  modality TEXT,"
          "  mime TEXT,"
          "  content_key TEXT,"
          "  source_id TEXT,"
          "  timestamp INTEGER,"
          "  width INTEGER,"
          "  height INTEGER,"
          "  channels INTEGER,"
          "  sample_rate INTEGER,"
          "  num_samples INTEGER,"
          "  blob_id BLOB"
          ")",

          // Memory feedback (metrics/learning)
          "CREATE TABLE IF NOT EXISTS memory_feedback ("
          "  embedding_id INTEGER PRIMARY KEY,"
          "  retrieved_count INTEGER NOT NULL DEFAULT 0,"
          "  used_count INTEGER NOT NULL DEFAULT 0,"
          "  contextual_gain REAL NOT NULL DEFAULT 0.0,"
          "  use_frequency REAL NOT NULL DEFAULT 0.0,"
          "  last_used INTEGER NOT NULL DEFAULT 0,"
          "  strength REAL NOT NULL DEFAULT 1.0,"
          "  original_embedding BLOB,"
          "  lability_state REAL NOT NULL DEFAULT 0.0,"
          "  lability_ts INTEGER"
          ")",
      },
  });

  registry.Register ({
      2,
      "Ensure objstore virtual table",
      {
          // Only works if extension is loaded; harmless if fails usually, but
          // better to wrap in try-catch or check module existence if possible.
          // For now, we assume if extension is missing, this will fail safely or
          // be caught by the migration runner loop.
          // Note: CREATE VIRTUAL TABLE IF NOT EXISTS is supported in newer SQLite.
          "CREATE VIRTUAL TABLE IF NOT EXISTS objstore USING objstore()",
      },
  });

  // Migration 3: Processor state persistence tables (Algorithms 1-8, 15-17, 19)
  registry.Register ({
      3,
      "Processor state persistence tables",
      {
          // PROCESSOR_STATE - Single row for all scalar adaptive parameters
          "CREATE TABLE IF NOT EXISTS processor_state ("
          "  id INTEGER PRIMARY KEY CHECK (id = 1),"
          "  signals_processed INTEGER NOT NULL DEFAULT 0,"
          "  u_t REAL NOT NULL DEFAULT 0.0,"
          "  weight_relevance_prior REAL NOT NULL DEFAULT 0.5,"
          "  weight_relevance REAL NOT NULL DEFAULT 0.5,"
          "  attention_width REAL NOT NULL DEFAULT 1.57,"
          "  T_dynamic REAL NOT NULL DEFAULT 0.2,"
          "  hysteresis REAL NOT NULL DEFAULT 0.05,"
          "  half_life REAL NOT NULL DEFAULT 120.0,"
          "  rate_target REAL NOT NULL DEFAULT 0.0,"
          "  sustained_influence REAL NOT NULL DEFAULT 0.0,"
          "  last_signal_timestamp INTEGER NOT NULL DEFAULT 0,"
          "  updated_at INTEGER NOT NULL DEFAULT 0"
          ")",

          // BLENDER_WEIGHTS - RLS-fitted metric weights (Algorithm 7)
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

          // BLENDER_COVARIANCE - RLS P matrix as BLOB (12x12 = 144 floats)
          "CREATE TABLE IF NOT EXISTS blender_covariance ("
          "  id INTEGER PRIMARY KEY CHECK (id = 1),"
          "  P_matrix BLOB"
          ")",
      },
  });

  // Migration 4: Episode and signal tracking tables
  registry.Register ({
      4,
      "Episode and signal tracking tables",
      {
          // EPISODES - Algorithm 12 episode boundaries
          "CREATE TABLE IF NOT EXISTS episodes ("
          "  id INTEGER PRIMARY KEY,"
          "  start_ts INTEGER NOT NULL,"
          "  end_ts INTEGER,"
          "  boundary_type TEXT,"
          "  centroid BLOB"
          ")",
          "CREATE INDEX IF NOT EXISTS idx_episodes_start ON episodes(start_ts)",

          // SIGNAL_METRICS - Algorithm 7 composite scoring observability
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
          "  write_decision INTEGER"
          ")",
          "CREATE INDEX IF NOT EXISTS idx_signal_metrics_ts ON "
          "signal_metrics(timestamp)",

          // GENERATION_TRACE - Algorithm 19 output embeddings
          "CREATE TABLE IF NOT EXISTS generation_trace ("
          "  id INTEGER PRIMARY KEY,"
          "  embedding BLOB NOT NULL,"
          "  timestamp INTEGER NOT NULL,"
          "  topic_id INTEGER,"
          "  semantic_drift REAL"
          ")",
          "CREATE INDEX IF NOT EXISTS idx_generation_trace_ts ON "
          "generation_trace(timestamp)",
      },
  });

  // Migration 5: Rolling window persistence tables
  registry.Register ({
      5,
      "Rolling window persistence tables",
      {
          // RECENT_CONTEXT - Rolling window of context embeddings
          "CREATE TABLE IF NOT EXISTS recent_context ("
          "  id INTEGER PRIMARY KEY,"
          "  embedding BLOB NOT NULL,"
          "  timestamp INTEGER NOT NULL,"
          "  seq_order INTEGER NOT NULL"
          ")",
          "CREATE INDEX IF NOT EXISTS idx_recent_context_seq ON "
          "recent_context(seq_order)",

          // RECENT_SCORES - Rolling window of composite scores
          "CREATE TABLE IF NOT EXISTS recent_scores ("
          "  id INTEGER PRIMARY KEY,"
          "  score REAL NOT NULL,"
          "  timestamp INTEGER NOT NULL"
          ")",
          "CREATE INDEX IF NOT EXISTS idx_recent_scores_ts ON "
          "recent_scores(timestamp)",
      },
  });

  // Migration 6: Vector search index (sqlite-vec)
  // NOTE: vec0 requires fixed dimension at table creation. The table is created
  // empty; embeddings are synced via reconsolidation.cpp writes. The dimension
  // (256) must match what the encoder produces.
  registry.Register ({
      6,
      "Vector search index (sqlite-vec)",
      {
          // vec0 virtual table with 256-dimensional embeddings for KNN search
          "CREATE VIRTUAL TABLE IF NOT EXISTS vec_embeddings USING vec0("
          "  embedding_id INTEGER PRIMARY KEY,"
          "  embedding float[256]"
          ")",
      },
  });

  // Migration 7: Extend memory_feedback table with additional columns
  // per docs/algorithms.md schema requirements
  registry.Register ({
      7,
      "Extend memory_feedback table (episode_id, mean_influence, stability)",
      {
          "ALTER TABLE memory_feedback ADD COLUMN episode_id INTEGER",
          "ALTER TABLE memory_feedback ADD COLUMN mean_influence REAL NOT NULL "
          "DEFAULT 0.0",
          "ALTER TABLE memory_feedback ADD COLUMN stability REAL NOT NULL "
          "DEFAULT 1.0",
      },
  });

  // Migration 8: Extend processor_state table with additional columns
  // for full algorithm state persistence per docs/algorithms.md
  registry.Register ({
      8,
      "Extend processor_state table (priors, dynamic state, threshold state)",
      {
          // Focus priors (Algorithm 1)
          "ALTER TABLE processor_state ADD COLUMN coverage_gain_floor_prior "
          "REAL NOT NULL DEFAULT 0.65",
          "ALTER TABLE processor_state ADD COLUMN mismatch_weight_prior REAL "
          "NOT NULL DEFAULT 0.5",
          "ALTER TABLE processor_state ADD COLUMN attention_width_prior REAL "
          "NOT NULL DEFAULT 1.57",

          // Sensitivity priors (Algorithm 3)
          "ALTER TABLE processor_state ADD COLUMN base_rate_prior REAL NOT "
          "NULL DEFAULT 0.2",
          "ALTER TABLE processor_state ADD COLUMN weight_novelty_prior REAL "
          "NOT NULL DEFAULT 0.3",
          "ALTER TABLE processor_state ADD COLUMN weight_surprise_prior REAL "
          "NOT NULL DEFAULT 0.2",
          "ALTER TABLE processor_state ADD COLUMN weight_valence_prior REAL "
          "NOT NULL DEFAULT 0.4",
          "ALTER TABLE processor_state ADD COLUMN weight_arousal_prior REAL "
          "NOT NULL DEFAULT 0.0",
          "ALTER TABLE processor_state ADD COLUMN weight_emotion_prior REAL "
          "NOT NULL DEFAULT 0.2",
          "ALTER TABLE processor_state ADD COLUMN emotion_gain_prior REAL NOT "
          "NULL DEFAULT 1.0",
          "ALTER TABLE processor_state ADD COLUMN score_gain_prior REAL NOT "
          "NULL DEFAULT 1.0",
          "ALTER TABLE processor_state ADD COLUMN rate_target_prior REAL NOT "
          "NULL DEFAULT 0.2",

          // Sensitivity dynamic (Algorithm 4)
          "ALTER TABLE processor_state ADD COLUMN weight_novelty REAL NOT NULL "
          "DEFAULT 0.3",

          // Stability priors (Algorithm 5)
          "ALTER TABLE processor_state ADD COLUMN hysteresis_band_prior REAL "
          "NOT NULL DEFAULT 0.02",
          "ALTER TABLE processor_state ADD COLUMN half_life_prior REAL NOT "
          "NULL DEFAULT 120.0",
          "ALTER TABLE processor_state ADD COLUMN rate_decay_prior REAL NOT "
          "NULL DEFAULT 0.60",
          "ALTER TABLE processor_state ADD COLUMN periphery_half_life_prior "
          "REAL NOT NULL DEFAULT 120.0",
          "ALTER TABLE processor_state ADD COLUMN salience_half_life_prior "
          "REAL NOT NULL DEFAULT 120.0",
          "ALTER TABLE processor_state ADD COLUMN drift_weight_prior REAL NOT "
          "NULL DEFAULT 0.5",

          // Stability dynamic (Algorithm 6)
          "ALTER TABLE processor_state ADD COLUMN rate_decay REAL NOT NULL "
          "DEFAULT 0.60",
          "ALTER TABLE processor_state ADD COLUMN periphery_half_life REAL NOT "
          "NULL DEFAULT 120.0",
          "ALTER TABLE processor_state ADD COLUMN salience_half_life REAL NOT "
          "NULL DEFAULT 120.0",

          // Threshold state (Algorithm 8)
          "ALTER TABLE processor_state ADD COLUMN m_rate REAL NOT NULL DEFAULT "
          "0.0",
          "ALTER TABLE processor_state ADD COLUMN rate_ticks INTEGER NOT NULL "
          "DEFAULT 0",
          "ALTER TABLE processor_state ADD COLUMN dt_ema REAL NOT NULL DEFAULT "
          "0.0",
          "ALTER TABLE processor_state ADD COLUMN last_rate_timestamp INTEGER "
          "NOT NULL DEFAULT 0",
      },
  });

  // Migration 9: Phase 3 - Observation windows, emotion state, RLS coefficients
  registry.Register ({
      9,
      "Phase 3: Observation windows, emotion state, RLS coefficients",
      {
          // Task 2: Observed retention history table
          "CREATE TABLE IF NOT EXISTS observed_retention_history ("
          "  id INTEGER PRIMARY KEY,"
          "  retention_seconds REAL NOT NULL,"
          "  timestamp INTEGER NOT NULL"
          ")",
          "CREATE INDEX IF NOT EXISTS idx_retention_history_ts ON "
          "observed_retention_history(timestamp)",

          // Task 3: Emotion state persistence (EWMA-smoothed values)
          "ALTER TABLE processor_state ADD COLUMN emotion_intensity_ewma REAL "
          "NOT NULL DEFAULT 0.0",
          "ALTER TABLE processor_state ADD COLUMN valence_ewma REAL NOT NULL "
          "DEFAULT 0.5",
          "ALTER TABLE processor_state ADD COLUMN arousal_ewma REAL NOT NULL "
          "DEFAULT 0.0",

          // Task 1: RLS coefficient storage (Algorithm 7)
          // 12 metrics x 4 coefficients = 48 doubles for learned coefficients
          "CREATE TABLE IF NOT EXISTS blender_coefficients ("
          "  id INTEGER PRIMARY KEY CHECK (id = 1),"
          "  coefficients BLOB"
          ")",
          // 48x48 covariance matrix = 2304 doubles
          "CREATE TABLE IF NOT EXISTS blender_coeff_covariance ("
          "  id INTEGER PRIMARY KEY CHECK (id = 1),"
          "  P_matrix BLOB"
          ")",
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
