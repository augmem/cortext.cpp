#include "cortext/store/schema.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include "schema_internal.hpp"
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

using schema_internal::kBaselineVersion;
using schema_internal::Migration;

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

/// @brief The schema of migrations 0..kBaselineVersion, created directly.
///
/// Each statement here yields the same sqlite_schema entry the legacy chain
/// leaves behind (tests/schema_baseline.test.cpp compares them exactly).
/// Column order follows the chain: columns added by ALTER TABLE come last.
const std::vector<std::string> &
BaselineStatements ()
{
  static const std::vector<std::string> statements = {
      // ------------------------------------------------------------------
      // Object store payloads (sqlite-objstore) and episodes
      // ------------------------------------------------------------------
      "CREATE VIRTUAL TABLE blobs USING objstore()",

      "CREATE TABLE episodes ("
      "  episode_id INTEGER PRIMARY KEY,"
      "  start_ts INTEGER NOT NULL,"
      "  end_ts INTEGER,"
      "  boundary_type TEXT,"
      "  centroid BLOB,"
      "  created_at INTEGER NOT NULL"
      ")",
      "CREATE INDEX idx_episodes_end ON episodes(end_ts) "
      "WHERE end_ts IS NULL",

      // ------------------------------------------------------------------
      // Long-term graph memory nodes: episodic memories, associations,
      // labels and working-memory records.
      // ------------------------------------------------------------------
      "CREATE TABLE memories ("
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
      "  s_max REAL NOT NULL DEFAULT 0.0,"
      "  s_avg REAL NOT NULL DEFAULT 0.0,"
      "  s_emotion_max REAL NOT NULL DEFAULT 0.0,"
      "  s_arousal_avg REAL NOT NULL DEFAULT 0.0,"
      "  boundary_score REAL NOT NULL DEFAULT 0.0,"
      "  emotion BLOB,"
      "  strength REAL NOT NULL DEFAULT 1.0,"
      "  drift_mag REAL NOT NULL DEFAULT 0.0,"
      "  cluster_id INTEGER,"
      "  last_access INTEGER,"
      "  created_at INTEGER NOT NULL,"
      "  retrieved_count INTEGER NOT NULL DEFAULT 0,"
      "  used_count INTEGER NOT NULL DEFAULT 0,"
      "  last_used INTEGER,"
      "  source_origin TEXT DEFAULT 'source',"
      "  strength_updated_at INTEGER"
      ")",
      "CREATE INDEX idx_memories_episode ON memories(episode_id, start_ts)",
      "CREATE INDEX idx_memories_embedding ON memories(embedding_id)",
      "CREATE INDEX idx_memories_kind ON memories(kind)",
      "CREATE INDEX idx_memories_kind_source ON memories(kind, source_id)",
      "CREATE INDEX idx_memories_source_start ON memories(source_id, start_ts)",
      "CREATE INDEX idx_memories_created_desc ON memories(created_at DESC)",
      "CREATE INDEX idx_memories_cluster ON memories(cluster_id) "
      "WHERE cluster_id IS NOT NULL",
      "CREATE INDEX idx_memories_working_active ON memories(end_ts, memory_id) "
      "WHERE kind = 'WORKING' AND end_ts IS NULL",
      "CREATE INDEX idx_memories_working_closed_end "
      "ON memories(end_ts, memory_id) "
      "WHERE kind = 'WORKING' AND end_ts IS NOT NULL",
      "CREATE INDEX idx_memories_label_created ON memories(created_at DESC) "
      "WHERE kind = 'LABEL'",
      "CREATE INDEX idx_memories_last_access ON memories(last_access DESC) "
      "WHERE last_access IS NOT NULL AND kind != 'WORKING'",
      "CREATE INDEX idx_memories_ltm_strength_created "
      "ON memories(strength, created_at, embedding_id) "
      "WHERE kind = 'LONG_TERM'",
      "CREATE INDEX idx_memories_ltm_unclustered_strength_created "
      "ON memories(strength, created_at, embedding_id) "
      "WHERE kind = 'LONG_TERM' AND cluster_id IS NULL",
      // Reference-time retrieval (operations/temporal_seed_internal.hpp).
      "CREATE INDEX idx_memories_ltm_start ON memories(start_ts) "
      "WHERE kind = 'LONG_TERM'",
      "CREATE INDEX idx_memories_ltm_end ON memories(end_ts) "
      "WHERE kind = 'LONG_TERM'",

      // ------------------------------------------------------------------
      // Signals with inline metrics
      // ------------------------------------------------------------------
      "CREATE TABLE signals ("
      "  signal_id INTEGER PRIMARY KEY,"
      "  memory_id INTEGER,"
      "  source_id TEXT NOT NULL,"
      "  embedding_id INTEGER NOT NULL,"
      "  blob_id BLOB,"
      "  timestamp INTEGER NOT NULL,"
      "  modality TEXT NOT NULL DEFAULT 'text',"
      "  mime TEXT,"
      "  serial_position INTEGER NOT NULL DEFAULT 0,"
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
      "  write_decision INTEGER,"
      "  coherence REAL,"
      "  focus_spread REAL,"
      "  f_effective REAL,"
      "  created_at INTEGER NOT NULL"
      ")",
      "CREATE INDEX idx_signals_memory ON signals(memory_id, serial_position)",
      "CREATE INDEX idx_signals_source ON signals(source_id) "
      "WHERE memory_id IS NULL",
      "CREATE INDEX idx_signals_ts ON signals(timestamp DESC)",
      "CREATE INDEX idx_signals_embedding ON signals(embedding_id)",
      "CREATE INDEX idx_signals_source_ts_serial "
      "ON signals(source_id, timestamp, serial_position)",
      "CREATE INDEX idx_signals_blob ON signals(blob_id) "
      "WHERE blob_id IS NOT NULL",

      // ------------------------------------------------------------------
      // Vectors (sqlite-vec): every stored embedding, and the current
      // retrievable embedding per memory.
      // ------------------------------------------------------------------
      "CREATE VIRTUAL TABLE embeddings USING vec0("
      "embedding_id INTEGER PRIMARY KEY,"
      "embedding float[256],"
      "+created_at integer"
      ")",
      "CREATE VIRTUAL TABLE current_memory_embeddings USING vec0("
      "memory_id INTEGER PRIMARY KEY,"
      "embedding float[256],"
      "+embedding_id INTEGER,"
      "+created_at INTEGER"
      ")",

      // ------------------------------------------------------------------
      // Long-term graph memory edges
      // ------------------------------------------------------------------
      "CREATE TABLE associations ("
      "  source_memory_id INTEGER NOT NULL,"
      "  target_memory_id INTEGER NOT NULL,"
      "  edge_type TEXT NOT NULL,"
      "  weight REAL NOT NULL DEFAULT 1.0,"
      "  decay_rate REAL,"
      "  last_reinforced INTEGER,"
      "  PRIMARY KEY (source_memory_id, target_memory_id, edge_type)"
      ")",
      "CREATE INDEX idx_associations_source "
      "ON associations(source_memory_id, edge_type)",
      "CREATE INDEX idx_associations_target "
      "ON associations(target_memory_id, edge_type)",
      "CREATE INDEX idx_associations_source_weight "
      "ON associations(source_memory_id, weight DESC, last_reinforced DESC, "
      "target_memory_id)",
      "CREATE INDEX idx_associations_target_weight "
      "ON associations(target_memory_id, weight DESC, last_reinforced DESC, "
      "source_memory_id)",
      "CREATE INDEX idx_associations_source_edge_weight "
      "ON associations(source_memory_id, edge_type, weight DESC, "
      "last_reinforced DESC, target_memory_id)",
      "CREATE INDEX idx_associations_target_edge_weight "
      "ON associations(target_memory_id, edge_type, weight DESC, "
      "last_reinforced DESC, source_memory_id)",
      "CREATE INDEX idx_associations_edge_source_target "
      "ON associations(edge_type, source_memory_id, target_memory_id)",
      "CREATE INDEX idx_associations_edge_target_source "
      "ON associations(edge_type, target_memory_id, source_memory_id)",

      // ------------------------------------------------------------------
      // Write pacing per source
      // ------------------------------------------------------------------
      "CREATE TABLE accumulators ("
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
      "CREATE INDEX idx_accumulators_episode ON accumulators(episode_id)",

      // ------------------------------------------------------------------
      // Unified processor state (singleton, id = 1)
      // ------------------------------------------------------------------
      "CREATE TABLE state ("
      "  id INTEGER PRIMARY KEY CHECK (id = 1),"
      "  signals_processed INTEGER NOT NULL DEFAULT 0,"
      // Focus
      "  weight_relevance REAL NOT NULL DEFAULT 0.5,"
      "  attention_width REAL NOT NULL DEFAULT 1.57,"
      "  coverage_gain_floor REAL NOT NULL DEFAULT 0.65,"
      "  mismatch_weight REAL NOT NULL DEFAULT 0.5,"
      // Sensitivity
      "  weight_novelty REAL NOT NULL DEFAULT 0.3,"
      "  weight_surprise REAL NOT NULL DEFAULT 0.2,"
      "  weight_valence REAL NOT NULL DEFAULT 0.4,"
      "  weight_arousal REAL NOT NULL DEFAULT 0.0,"
      "  emotion_gain REAL NOT NULL DEFAULT 1.0,"
      "  score_gain REAL NOT NULL DEFAULT 1.0,"
      // Emotion
      "  emotion_intensity REAL NOT NULL DEFAULT 0.0,"
      "  valence REAL NOT NULL DEFAULT 0.5,"
      "  arousal REAL NOT NULL DEFAULT 0.0,"
      "  flashbulb_rate REAL NOT NULL DEFAULT 0.0,"
      // Rate observation
      "  m_rate REAL NOT NULL DEFAULT 0.0,"
      "  dt_ema REAL NOT NULL DEFAULT 0.0,"
      "  last_rate_timestamp INTEGER NOT NULL DEFAULT 0,"
      "  u_uncertainty REAL NOT NULL DEFAULT 0.0,"
      // Embedding prediction
      "  last_embedding BLOB,"
      "  x_pred_ema BLOB,"
      // Working memory
      "  wm_maintenance_cost REAL NOT NULL DEFAULT 0.0,"
      "  wm_slot_count INTEGER NOT NULL DEFAULT 0,"
      "  wm_last_accepted INTEGER NOT NULL DEFAULT 0,"
      "  wm_last_chunked INTEGER NOT NULL DEFAULT 0,"
      // Consolidation and episode tracking
      "  last_consolidation_ts INTEGER NOT NULL DEFAULT 0,"
      "  consolidation_count INTEGER NOT NULL DEFAULT 0,"
      "  memories_since_consolidation INTEGER NOT NULL DEFAULT 0,"
      "  is_processing_signal INTEGER NOT NULL DEFAULT 0,"
      "  last_retrieval_ts INTEGER NOT NULL DEFAULT 0,"
      "  episode_start_ts INTEGER NOT NULL DEFAULT 0,"
      "  last_interrupt_tick INTEGER NOT NULL DEFAULT 0,"
      "  last_signal_timestamp INTEGER NOT NULL DEFAULT 0,"
      "  updated_at INTEGER NOT NULL DEFAULT 0,"
      "  neuromod_ach REAL NOT NULL DEFAULT 0.0,"
      "  write_rate_timestamps BLOB,"
      // Blender weights, covariance and coefficients
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
      "  blender_P_matrix BLOB,"
      "  blender_coefficients BLOB,"
      "  blender_coeff_P_matrix BLOB,"
      // Consolidation throughput range and recommendation latch
      "  consolidation_rate_floor REAL NOT NULL DEFAULT 0.0,"
      "  consolidation_rate_peak REAL NOT NULL DEFAULT 0.0,"
      "  consolidation_rate_initialized INTEGER NOT NULL DEFAULT 0,"
      "  consolidation_rate_armed INTEGER NOT NULL DEFAULT 1"
      ")",

      // ------------------------------------------------------------------
      // Constructive recall reconstruction ledger
      // ------------------------------------------------------------------
      "CREATE TABLE memory_reconstructions ("
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
      "CREATE INDEX idx_memory_reconstructions_memory "
      "ON memory_reconstructions(memory_id, reconstruction_id DESC)",
      "CREATE INDEX idx_memory_reconstructions_embedding "
      "ON memory_reconstructions(embedding_id)",

      // ------------------------------------------------------------------
      // Row-addressed sparse retrieval route
      // ------------------------------------------------------------------
      "CREATE TABLE cortext_sparse_route_meta("
      "singleton INTEGER PRIMARY KEY CHECK(singleton = 1), "
      "schema_version INTEGER NOT NULL CHECK(schema_version = 1), "
      "embedding_dim INTEGER NOT NULL CHECK(embedding_dim > 0), "
      "entry_memory_id INTEGER NOT NULL CHECK(entry_memory_id >= 0), "
      "max_level INTEGER NOT NULL CHECK(max_level >= 0), "
      "active_count INTEGER NOT NULL CHECK(active_count >= 0), "
      "generation INTEGER NOT NULL CHECK(generation >= 0), "
      // Consolidation-rebuilt activation centroid
      "activation_entry_memory_id INTEGER NOT NULL DEFAULT 0 "
      "CHECK(activation_entry_memory_id >= 0), "
      "activation_generation INTEGER NOT NULL DEFAULT 0 "
      "CHECK(activation_generation >= 0), "
      "activation_centroid BLOB, "
      "activation_identity_ids BLOB)",
      "CREATE TABLE cortext_sparse_route_nodes("
      "memory_id INTEGER PRIMARY KEY, "
      "embedding BLOB NOT NULL, "
      "level INTEGER NOT NULL CHECK(level >= 0), "
      "links BLOB NOT NULL, "
      "active INTEGER NOT NULL CHECK(active IN (0, 1)), "
      "generation INTEGER NOT NULL CHECK(generation >= 0))",
      "CREATE TABLE cortext_sparse_route_build("
      "singleton INTEGER PRIMARY KEY CHECK(singleton = 1), "
      "schema_version INTEGER NOT NULL CHECK(schema_version = 1), "
      "embedding_dim INTEGER NOT NULL CHECK(embedding_dim > 0), "
      "cursor_memory_id INTEGER NOT NULL CHECK(cursor_memory_id >= 0), "
      "entry_memory_id INTEGER NOT NULL CHECK(entry_memory_id >= 0), "
      "max_level INTEGER NOT NULL CHECK(max_level >= 0), "
      "active_count INTEGER NOT NULL CHECK(active_count >= 0), "
      "generation INTEGER NOT NULL CHECK(generation > 0))",
      "CREATE TABLE cortext_sparse_route_dirty("
      "memory_id INTEGER PRIMARY KEY CHECK(memory_id > 0))",
      "CREATE INDEX idx_sparse_route_nodes_active "
      "ON cortext_sparse_route_nodes(generation, active, memory_id)",
      "CREATE INDEX idx_sparse_route_nodes_generation "
      "ON cortext_sparse_route_nodes(generation)",
      "CREATE INDEX idx_sparse_route_nodes_entry "
      "ON cortext_sparse_route_nodes(generation, active, level DESC, "
      "memory_id ASC)",

      // ------------------------------------------------------------------
      // Exact signal embeddings outside the global vector route
      // ------------------------------------------------------------------
      "CREATE TABLE cortext_active_signal_embeddings("
      "slot INTEGER PRIMARY KEY CHECK(slot >= 0), "
      "signal_id INTEGER NOT NULL UNIQUE, "
      "embedding BLOB NOT NULL, "
      "created_at INTEGER NOT NULL, "
      "capacity INTEGER NOT NULL CHECK(capacity > 0), "
      "FOREIGN KEY(signal_id) REFERENCES signals(signal_id) "
      "ON DELETE CASCADE)",

      // ------------------------------------------------------------------
      // Computed windows
      // ------------------------------------------------------------------
      "CREATE VIEW recent_context AS "
      "SELECT s.signal_id, s.embedding_id, "
      "COALESCE(a.embedding, e.embedding) AS embedding, s.timestamp "
      "FROM signals s "
      "LEFT JOIN cortext_active_signal_embeddings a "
      "ON a.signal_id = s.signal_id "
      "LEFT JOIN embeddings e ON s.embedding_id = e.embedding_id "
      "ORDER BY s.timestamp DESC "
      "LIMIT 64",
      "CREATE VIEW recent_scores AS "
      "SELECT signal_id, score, timestamp "
      "FROM signals "
      "WHERE score IS NOT NULL "
      "ORDER BY timestamp DESC "
      "LIMIT 100",
      "CREATE VIEW recent_ids AS "
      "SELECT embedding_id, timestamp "
      "FROM signals "
      "ORDER BY timestamp DESC "
      "LIMIT 1024",
      "CREATE VIEW recent_retrievals AS "
      "SELECT memory_id, last_access as last_retrieval_ts "
      "FROM memories "
      "WHERE last_access IS NOT NULL "
      "  AND kind != 'WORKING' "
      "ORDER BY last_access DESC "
      "LIMIT 128",

      // ------------------------------------------------------------------
      // Keyword index over LONG_TERM memory text (FTS5, contentless).
      // rowid = memory_id; text stays in the object store.
      // ------------------------------------------------------------------
      "CREATE VIRTUAL TABLE memory_text_fts USING fts5("
      "  text, content='', contentless_delete=1,"
      "  tokenize='porter unicode61 remove_diacritics 2')",
      "CREATE TRIGGER memories_text_fts_delete "
      "AFTER DELETE ON memories WHEN old.kind = 'LONG_TERM' "
      "BEGIN DELETE FROM memory_text_fts WHERE rowid = old.memory_id; "
      "END",
      // Upgraded stores queue one-time C++ backfills here
      // (memory_text_index_internal::RunPendingBackfill). A fresh store
      // holds no memories, so it queues none.
      "CREATE TABLE cortext_pending_backfills ("
      "  name TEXT PRIMARY KEY)",

      // ------------------------------------------------------------------
      // Shared-space vectors and late-interaction reranker codes per
      // stored signal payload, keyed by content id (objstore BLAKE3 id).
      // ------------------------------------------------------------------
      "CREATE TABLE memory_signal_vectors ("
      "  blob_id BLOB PRIMARY KEY,"
      "  modality TEXT NOT NULL,"
      "  embedding BLOB NOT NULL)",
      "CREATE INDEX idx_memory_signal_vectors_modality "
      "ON memory_signal_vectors(modality)",
      "CREATE TRIGGER signals_vector_delete "
      "AFTER DELETE ON signals WHEN old.blob_id IS NOT NULL "
      "BEGIN DELETE FROM memory_signal_vectors "
      "WHERE blob_id = old.blob_id AND NOT EXISTS ("
      "  SELECT 1 FROM signals WHERE blob_id = old.blob_id); END",
      // Codes live in their own rowid table so vector reads do not pull
      // them (models/aist_late_interaction.hpp Serialize format).
      "CREATE TABLE memory_signal_li_codes ("
      "  blob_id BLOB NOT NULL UNIQUE,"
      "  codes BLOB NOT NULL)",
      "CREATE TRIGGER signals_li_codes_delete "
      "AFTER DELETE ON signals WHEN old.blob_id IS NOT NULL "
      "BEGIN DELETE FROM memory_signal_li_codes "
      "WHERE blob_id = old.blob_id AND NOT EXISTS ("
      "  SELECT 1 FROM signals WHERE blob_id = old.blob_id); END",

      // ------------------------------------------------------------------
      // Standing set: stored signals that state a lasting directive
      // (operations/standing_memory_internal.hpp). terms: word stems.
      // ------------------------------------------------------------------
      "CREATE TABLE standing_signals ("
      "  signal_id INTEGER PRIMARY KEY,"
      "  start_ts INTEGER NOT NULL,"
      "  terms TEXT NOT NULL DEFAULT '')",
      "CREATE TRIGGER signals_standing_delete "
      "AFTER DELETE ON signals "
      "BEGIN DELETE FROM standing_signals "
      "WHERE signal_id = old.signal_id; END",
  };
  return statements;
}

/// @brief Migrations newer than the baseline.
///
/// Add new schema changes here with ids above kBaselineVersion. Fresh stores
/// run them after the baseline; upgraded stores after the legacy chain.
std::vector<Migration>
PostBaselineMigrations ()
{
  return {};
}

/// @brief True when the store holds no cortext schema yet.
///
/// The ledger is empty and no memories table exists: a store with cortext
/// tables but no ledger rows keeps the idempotent legacy chain.
bool
IsEmptyStore (Store &store, const std::set<int64_t> &applied_ids)
{
  if (!applied_ids.empty ())
    return false;
  return store
      .Execute ("SELECT 1 FROM sqlite_master "
                "WHERE type = 'table' AND name = 'memories' LIMIT 1")
      .empty ();
}

/// @brief Creates the baseline schema and records every legacy id as
/// applied, so the ledger matches a store that ran the chain.
void
ApplyBaselineInTransaction (Store &store,
                            const std::vector<Migration> &legacy)
{
  for (const auto &sql : BaselineStatements ())
    {
      try
        {
          store.Execute (sql);
        }
      catch (const std::exception &e)
        {
          telemetry::LogError (
              "Baseline schema statement failed",
              { telemetry::Attribute::String ("component", "store.schema"),
                telemetry::Attribute::Int64 ("baseline_version",
                                             kBaselineVersion),
                telemetry::Attribute::String ("sql", sql),
                telemetry::Attribute::String ("error", e.what ()) });
          throw;
        }
    }
  for (const auto &m : legacy)
    store.Execute (
        "INSERT INTO cortext_schema_migrations (id, description, applied_at) "
        "VALUES (?, ?, strftime('%s', 'now'))",
        { m.id, m.description });
}

enum class SchemaPath
{
  // Empty stores get the baseline; other stores upgrade incrementally.
  kBaselineWhenEmpty,
  // Every store runs the legacy chain (equivalence and upgrade tests).
  kLegacyChain,
};

void
ApplyMigrationsThrough (Store &store, int64_t maximum_id, SchemaPath path)
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
      auto migrations = schema_internal::LegacyMigrations ();
      std::sort (migrations.begin (), migrations.end (),
                 [] (const Migration &a, const Migration &b) {
                   return a.id < b.id;
                 });

      if (path == SchemaPath::kBaselineWhenEmpty
          && maximum_id >= kBaselineVersion
          && IsEmptyStore (store, applied_ids))
        {
          ApplyBaselineInTransaction (store, migrations);
          for (const auto &m : migrations)
            applied_ids.insert (m.id);
        }

      for (auto &m : PostBaselineMigrations ())
        migrations.push_back (std::move (m));

      for (const auto &m : migrations)
        {
          if (m.id > maximum_id || applied_ids.count (m.id))
            {
              continue;
            }
          ApplySingleMigrationInTransaction (store, m);
        }

      const auto memory_text_fts = store.Execute (
          "SELECT 1 FROM sqlite_master "
          "WHERE type = 'table' AND name = 'memory_text_fts' LIMIT 1");
      if (!memory_text_fts.empty ())
        {
          // Connection-local helpers for keyword queries: a scratch FTS5
          // table with the index's tokenizer maps query words to index terms,
          // and vocabularies give those terms' document frequencies
          // (memory_text_index_internal::Search).
          store.Execute (
              "CREATE VIRTUAL TABLE IF NOT EXISTS temp.memory_query_terms "
              "USING fts5(text, content='',"
              "  tokenize='porter unicode61 remove_diacritics 2')");
          store.Execute (
              "CREATE VIRTUAL TABLE IF NOT EXISTS temp.memory_query_term_stems "
              "USING fts5vocab(temp, memory_query_terms, 'instance')");
          store.Execute (
              "CREATE VIRTUAL TABLE IF NOT EXISTS temp.memory_text_vocab "
              "USING fts5vocab(main, memory_text_fts, 'row')");
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
  ApplyMigrationsThrough (store, maximum_id, SchemaPath::kBaselineWhenEmpty);
}

void
DebugApplyLegacyMigrationChainThroughForTest (Store &store,
                                              int64_t maximum_id)
{
  ApplyMigrationsThrough (store, maximum_id, SchemaPath::kLegacyChain);
}

std::vector<std::string>
DebugBaselineStatementsForTest ()
{
  return BaselineStatements ();
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
  ApplyMigrationsThrough (store, std::numeric_limits<int64_t>::max (),
                          SchemaPath::kBaselineWhenEmpty);
}

} // namespace cortext::store
