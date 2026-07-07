#include "cortext/operations/memory_storage.hpp"
#include "constructive_recall_internal.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/processor/accumulator_state.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/signal.hpp"
#include "cortext/core/sparse.hpp"
#include "cortext/store/object_store.hpp"
#include "cortext/store/schema_helpers.hpp"
#include "cortext/store/store.hpp"
#include "cortext/store/utils.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cortext::operations
{

namespace
{
using SteadyClock = std::chrono::steady_clock;

double
ElapsedMillis (SteadyClock::time_point start)
{
  return std::chrono::duration<double, std::milli> (SteadyClock::now () - start)
      .count ();
}

/// @brief Determine primary modality from signal records (majority vote)
std::string
GetPrimaryModality (const std::vector<SignalRecord> &signals,
                    const std::string &fallback_modality)
{
  if (signals.empty ())
    {
      return fallback_modality;
    }

  std::unordered_map<std::string, int> counts;
  for (const auto &sig : signals)
    {
      counts[sig.modality]++;
    }

  std::string primary = signals[0].modality;
  int max_count = 0;
  for (const auto &kv : counts)
    {
      if (kv.second > max_count)
        {
          max_count = kv.second;
          primary = kv.first;
        }
    }
  return primary;
}

/// @brief Serialize 6d emotion/mood vector to blob
std::vector<char>
SerializeEmotionVector (const std::array<double, 6> &vec)
{
  std::vector<char> blob (sizeof (double) * 6);
  std::memcpy (blob.data (), vec.data (), blob.size ());
  return blob;
}

std::string
SourceOriginFor ()
{
  return "source";
}

double
SourcePriorReliability (double F, double S, double T)
{
  return core::SourceReliabilityPrior (F, S, T);
}

bool
DecodeEmbeddingAny (const std::any &value, int expected_dim,
                    Eigen::VectorXf &out)
{
  if (value.type () == typeid (std::vector<float>))
    {
      const auto &vec = std::any_cast<const std::vector<float> &> (value);
      if (expected_dim > 0 && static_cast<int> (vec.size ()) != expected_dim)
        {
          return false;
        }
      out.resize (static_cast<Eigen::Index> (vec.size ()));
      for (std::size_t i = 0; i < vec.size (); ++i)
        {
          out[static_cast<Eigen::Index> (i)] = vec[i];
        }
      return out.size () > 0;
    }
  return expected_dim > 0 && core::DecodeFloatBlob (value, expected_dim, out);
}

std::vector<float>
ToFloatVector (const Eigen::VectorXf &v)
{
  std::vector<float> out;
  out.resize (static_cast<std::size_t> (v.size ()));
  for (int i = 0; i < v.size (); ++i)
    {
      out[static_cast<std::size_t> (i)] = v[i];
    }
  return out;
}

void
AppendUniqueSupersessionRows (
    std::vector<std::map<std::string, std::any>> &rows,
    std::vector<std::map<std::string, std::any>> extra_rows,
    std::unordered_set<long long> &seen_memory_ids)
{
  for (auto &row : extra_rows)
    {
      auto memory_it = row.find ("memory_id");
      if (memory_it == row.end ())
        {
          continue;
        }
      const auto memory_id = store::AnyToLongLong (memory_it->second);
      if (!memory_id || *memory_id <= 0
          || !seen_memory_ids.insert (*memory_id).second)
        {
          continue;
        }
      rows.push_back (std::move (row));
    }
}

std::vector<std::map<std::string, std::any>>
LoadSupersessionCandidateRows (Transaction &tx,
                               const Eigen::VectorXf &embedding_to_store,
                               long long memory_id, long long end_ts,
                               int candidate_limit)
{
  std::vector<std::map<std::string, std::any>> rows;
  std::unordered_set<long long> seen_memory_ids;
  const std::vector<float> query_embedding = ToFloatVector (
      embedding_to_store);
  try
    {
      auto current_rows = tx.Execute (
          "SELECT m.memory_id, cme.embedding "
          "FROM ("
          "  SELECT memory_id, embedding "
          "  FROM current_memory_embeddings "
          "  WHERE embedding MATCH ? "
          "    AND k = ?"
          ") cme "
          "JOIN memories m ON m.memory_id = cme.memory_id "
          "WHERE m.memory_id != ? "
          "  AND m.kind IN ('LONG_TERM', 'ASSOCIATION') "
          "  AND COALESCE(m.start_ts, 0) < ?",
          { query_embedding, static_cast<long long> (candidate_limit),
            memory_id, end_ts });
      AppendUniqueSupersessionRows (rows, std::move (current_rows),
                                    seen_memory_ids);
    }
  catch (const std::exception &e)
    {
      telemetry::LogDebug (
          "cortext.memory_storage.supersession_current_knn_unavailable",
          { telemetry::Attribute::String ("error", e.what ()) });
    }

  try
    {
      auto historical_rows = tx.Execute (
          "SELECT m.memory_id, e.embedding "
          "FROM embeddings e "
          "JOIN memories m ON m.embedding_id = e.embedding_id "
          "WHERE e.embedding MATCH ? "
          "  AND k = ? "
          "  AND m.memory_id != ? "
          "  AND m.embedding_id IS NOT NULL "
          "  AND m.kind IN ('LONG_TERM', 'ASSOCIATION') "
          "  AND COALESCE(m.start_ts, 0) < ?",
          { query_embedding, static_cast<long long> (candidate_limit),
            memory_id, end_ts });
      AppendUniqueSupersessionRows (rows, std::move (historical_rows),
                                    seen_memory_ids);
    }
  catch (const std::exception &e)
    {
      telemetry::LogDebug (
          "cortext.memory_storage.supersession_historical_knn_unavailable",
          { telemetry::Attribute::String ("error", e.what ()) });
    }

  if (rows.empty ())
    {
      auto recent_rows = tx.Execute (
          "SELECT m.memory_id, "
          "       CASE WHEN cme.memory_id IS NOT NULL "
          "            THEN cme.embedding ELSE e.embedding END AS embedding "
          "FROM memories m "
          "LEFT JOIN current_memory_embeddings cme "
          "  ON cme.memory_id = m.memory_id "
          "JOIN embeddings e "
          "  ON e.embedding_id = COALESCE(cme.embedding_id, m.embedding_id) "
          "WHERE m.memory_id != ? "
          "  AND m.embedding_id IS NOT NULL "
          "  AND m.kind IN ('LONG_TERM', 'ASSOCIATION') "
          "  AND COALESCE(m.start_ts, 0) < ? "
          "ORDER BY m.memory_id DESC "
          "LIMIT ?",
          { memory_id, end_ts, static_cast<long long> (candidate_limit) });
      AppendUniqueSupersessionRows (rows, std::move (recent_rows),
                                    seen_memory_ids);
    }

  return rows;
}

struct SupersessionEdge
{
  long long target_memory_id = 0;
  double similarity = 0.0;
  double weight = 0.0;
};

void
WriteSupersessionEdges (OperationContext &context, Transaction &tx,
                        long long memory_id,
                        const Eigen::VectorXf &embedding_to_store,
                        long long end_ts)
{
  if (memory_id <= 0 || embedding_to_store.size () <= 0)
    {
      return;
    }

  const auto &cfg = context.GetConfig ();
  const int embedding_dim = static_cast<int> (embedding_to_store.size ());
  const int candidate_limit = std::max (
      1, core::SupersessionCandidateLimit (cfg.focus, cfg.sensitivity,
                                           cfg.stability));
  const int max_edges = core::SupersessionMaxEdges (
      cfg.focus, cfg.sensitivity, cfg.stability);
  const auto rows = LoadSupersessionCandidateRows (
      tx, embedding_to_store, memory_id, end_ts, candidate_limit);

  const double similarity_threshold = core::SupersessionSimilarityThreshold (
      cfg.focus, cfg.sensitivity, cfg.stability);
  const double duplicate_threshold = core::SupersessionDuplicateThreshold (
      cfg.focus, cfg.sensitivity, cfg.stability);
  int scanned_count = 0;
  int decoded_count = 0;
  int below_topic_count = 0;
  int duplicate_count = 0;
  double best_similarity = -1.0;
  std::vector<SupersessionEdge> edges;
  edges.reserve (static_cast<std::size_t> (max_edges));
  int edge_count = 0;
  for (const auto &row : rows)
    {
      ++scanned_count;
      auto memory_it = row.find ("memory_id");
      auto embedding_it = row.find ("embedding");
      if (memory_it == row.end () || embedding_it == row.end ())
        {
          continue;
        }
      const auto target_id_opt = store::AnyToLongLong (memory_it->second);
      if (!target_id_opt || *target_id_opt <= 0)
        {
          continue;
        }
      Eigen::VectorXf target_embedding;
      if (!DecodeEmbeddingAny (embedding_it->second, embedding_dim,
                               target_embedding))
        {
          continue;
        }
      ++decoded_count;
      const double similarity = core::CosineSimilarity (embedding_to_store,
                                                        target_embedding);
      best_similarity = std::max (best_similarity, similarity);
      if (similarity < similarity_threshold)
        {
          ++below_topic_count;
          continue;
        }
      if (similarity >= duplicate_threshold)
        {
          ++duplicate_count;
          continue;
        }
      const double weight = core::SupersessionEdgeWeight (
          similarity, cfg.focus, cfg.sensitivity, cfg.stability);
      edges.push_back ({ *target_id_opt, similarity, weight });
    }

  std::sort (edges.begin (), edges.end (),
             [] (const SupersessionEdge &a, const SupersessionEdge &b) {
               if (a.similarity != b.similarity)
                 {
                   return a.similarity > b.similarity;
                 }
               return a.target_memory_id > b.target_memory_id;
             });
  if (static_cast<int> (edges.size ()) > max_edges)
    {
      edges.resize (static_cast<std::size_t> (max_edges));
    }

  for (const auto &edge : edges)
    {
      tx.Execute (
          "INSERT OR REPLACE INTO associations "
          "(source_memory_id, target_memory_id, edge_type, weight, "
          "last_reinforced) "
          "VALUES (?, ?, 'supersedes', ?, ?)",
          { memory_id, edge.target_memory_id, edge.weight, end_ts });
      tx.Execute (
          "UPDATE memories "
          "SET source_contradiction_count = source_contradiction_count + 1 "
          "WHERE memory_id = ?",
          { edge.target_memory_id });
      ++edge_count;
    }

  telemetry::LogDebug (
      "cortext.memory_storage.supersession_scan",
      { telemetry::Attribute::Int64 ("source_memory_id", memory_id),
        telemetry::Attribute::Int64 ("candidate_count", scanned_count),
        telemetry::Attribute::Int64 ("decoded_count", decoded_count),
        telemetry::Attribute::Int64 ("below_topic_count", below_topic_count),
        telemetry::Attribute::Int64 ("duplicate_count", duplicate_count),
        telemetry::Attribute::Int64 ("edge_count", edge_count),
        telemetry::Attribute::Int64 ("max_edges", max_edges),
        telemetry::Attribute::Double ("best_similarity", best_similarity),
        telemetry::Attribute::Double ("similarity_threshold",
                                      similarity_threshold),
        telemetry::Attribute::Double ("duplicate_threshold",
                                      duplicate_threshold) });
}

} // namespace

void
MemoryStorage::Execute (OperationContext &context, Transaction &tx) const
{
  // Check write gate decision - if rejected, discard entirely
  if (!context.GetAccumulatorWriteDecision ())
    {
      telemetry::AddCounter ("cortext.memory_storage.rejected_total", 1);
      return;
    }

  const auto &signal = context.GetSignal ();
  const auto &cfg = context.GetConfig ();
  if (signal.retention == Retention::Ephemeral)
    {
      telemetry::AddCounter ("cortext.memory_storage.ephemeral_skip_total", 1);
      return;
    }

  Store *store = context.GetStore ();
  if (!store)
    {
      telemetry::AddCounter ("cortext.memory_storage.no_store_total", 1);
      return;
    }

  auto &p_ctx = context.GetProcessorContext ();

  // Get accumulator state for this source
  auto acc_it = p_ctx.accumulator_states.find (signal.source_id);
  if (acc_it == p_ctx.accumulator_states.end ())
    {
      telemetry::AddCounter ("cortext.memory_storage.no_accumulator_total", 1);
      return;
    }
  auto &acc = acc_it->second;

  // Use nested transaction for atomicity - all writes succeed or none
  const auto savepoint_begin_start = SteadyClock::now ();
  auto savepoint = tx.Begin ();
  auto object_savepoint = context.GetObjectTransaction ()
                              ? context.GetObjectTransaction ()->Begin ()
                              : nullptr;
  context.AddOperationTiming ("MemoryStorage.begin_savepoints",
                              ElapsedMillis (savepoint_begin_start));
  bool savepoint_finished = false;
  bool object_savepoint_finished = false;
  auto rollback_savepoints = [&] {
    std::exception_ptr rollback_error;
    if (object_savepoint && !object_savepoint_finished)
      {
        try
          {
            object_savepoint->Rollback ();
            object_savepoint_finished = true;
          }
        catch (...)
          {
            if (!rollback_error)
              {
                rollback_error = std::current_exception ();
              }
          }
      }
    if (!savepoint_finished)
      {
        try
          {
            savepoint->Rollback ();
            savepoint_finished = true;
          }
        catch (...)
          {
            rollback_error = std::current_exception ();
          }
      }
    if (rollback_error)
      {
        std::rethrow_exception (rollback_error);
      }
  };

  try
    {
      using store::AnyToLongLong;
      using store::BlobFromAny;
      using store::EigenToFloatVec;

      // Section 4.4: Use representative embedding if available (memory write)
      const auto &rep_emb = context.GetRepresentativeEmbedding ();
      const Eigen::VectorXf &embedding_to_store
          = rep_emb.has_value () ? *rep_emb : acc.mu_acc;

      // Convert embedding from Eigen to vector<float> for SQL storage
      const std::vector<float> emb_vec = EigenToFloatVec (embedding_to_store);

      // 1. Compute memory-level aggregates from accumulator (Section 4.4)
      const int n_signals = std::max (acc.n_signals, 1);
      const double s_avg = acc.s_sum / static_cast<double> (n_signals);
      const double s_max = acc.s_max;
      const double s_emotion_max = acc.s_emotion_max;
      const double s_arousal_avg
          = acc.s_arousal_sum / static_cast<double> (n_signals);
      const uint64_t start_ts = acc.t_start;
      const uint64_t end_ts = signal.timestamp;
      const double drift_mag = acc.drift_acc;
      const double boundary_score
          = context.GetBoundaryScore ().value_or (0.0);

      // 2. Get emotion and mood from context (Section 6.1.1)
      const auto &emotion_probs = context.GetEmotionProbabilities ();
      const auto &mood_vec = p_ctx.mood_vector;
      const std::vector<char> emotion_blob = SerializeEmotionVector (emotion_probs);
      const std::vector<char> mood_blob = SerializeEmotionVector (mood_vec);

      // 3. Determine primary modality from tracked signals
      const std::string primary_modality
          = GetPrimaryModality (acc.signals, signal.modality);

      const std::string origin = SourceOriginFor ();
      const double source_reliability
          = SourcePriorReliability (cfg.focus, cfg.sensitivity, cfg.stability);

      // 4. Require tracked per-signal records for persistence.
      if (acc.signals.empty ())
        {
          rollback_savepoints ();
          telemetry::AddCounter (
              "cortext.memory_storage.missing_signals_total", 1);
          return;
        }

      // 5. Store memory-level content blob by concatenating signal blobs.
      const auto content_start = SteadyClock::now ();
      double content_get_ms = 0.0;
      double content_put_ms = 0.0;
      std::vector<unsigned char> content_payload;
      std::vector<const SignalRecord *> ordered_signals;
      ordered_signals.reserve (acc.signals.size ());
      for (const auto &rec : acc.signals)
        {
          ordered_signals.push_back (&rec);
        }
      std::sort (ordered_signals.begin (), ordered_signals.end (),
                 [] (const SignalRecord *a, const SignalRecord *b) {
                   return a->serial_position < b->serial_position;
                 });
      const bool text_mode = std::all_of (
          ordered_signals.begin (), ordered_signals.end (),
          [] (const SignalRecord *rec) {
            return rec && rec->modality == "text";
          });
      const bool audio_mode = std::all_of (
          ordered_signals.begin (), ordered_signals.end (),
          [] (const SignalRecord *rec) {
            return rec && rec->modality == "audio";
          });
      const bool single_payload_mode = ordered_signals.size () == 1;
      const bool memory_blob_supported
          = text_mode || audio_mode || single_payload_mode;
      std::vector<unsigned char> content_blob_id;
      if (single_payload_mode && !ordered_signals.empty ()
          && ordered_signals[0] && !ordered_signals[0]->blob_id.empty ())
        {
          content_blob_id = ordered_signals[0]->blob_id;
        }
      else if (memory_blob_supported)
        {
          for (const auto *rec : ordered_signals)
            {
              if (!rec || rec->blob_id.empty ())
                {
                  continue;
                }
              const auto object_get_start = SteadyClock::now ();
              auto bytes_opt = GetObject (object_savepoint.get (), *savepoint,
                                          rec->blob_id);
              content_get_ms += ElapsedMillis (object_get_start);
              if (bytes_opt)
                {
                  const auto &bytes = *bytes_opt;
                  if (!bytes.empty ())
                    {
                      if (text_mode && !content_payload.empty ())
                        {
                          content_payload.push_back ('\n');
                        }
                      content_payload.insert (content_payload.end (),
                                              bytes.begin (), bytes.end ());
                    }
                }
            }
        }
      if (memory_blob_supported && content_payload.empty () && signal.payload
          && !signal.payload->empty ())
        {
          content_payload = *signal.payload;
        }
      if (content_blob_id.empty () && !content_payload.empty ())
        {
          const auto object_put_start = SteadyClock::now ();
          content_blob_id
              = PutObject (object_savepoint.get (), *savepoint,
                           content_payload);
          content_put_ms += ElapsedMillis (object_put_start);
        }
      context.AddOperationTiming ("MemoryStorage.content_get_objects",
                                  content_get_ms);
      context.AddOperationTiming ("MemoryStorage.content_put_object",
                                  content_put_ms);
      context.AddOperationTiming ("MemoryStorage.content_blob",
                                  ElapsedMillis (content_start));
      // 6. Insert memory embedding (v2: minimal sqlite-vec table)
      const std::string insert_sql = std::string ("INSERT INTO embeddings (")
                                     + store::kEmbeddingsInsertColumns
                                     + ") VALUES ("
                                     + store::kEmbeddingsInsertDefaults + ")";
      const auto insert_embedding_start = SteadyClock::now ();
      savepoint->Execute (insert_sql,
                          { emb_vec, static_cast<long long> (end_ts) });
      context.AddOperationTiming ("MemoryStorage.insert_memory_embedding",
                                  ElapsedMillis (insert_embedding_start));

      const auto embedding_id_start = SteadyClock::now ();
      auto id_rows = savepoint->Execute ("SELECT last_insert_rowid() AS id", {});
      context.AddOperationTiming ("MemoryStorage.select_embedding_id",
                                  ElapsedMillis (embedding_id_start));
      if (id_rows.empty () || id_rows[0].count ("id") == 0)
        {
          rollback_savepoints ();
          telemetry::AddCounter (
              "cortext.memory_storage.embedding_error_total", 1);
          return;
        }
      const auto id_opt = AnyToLongLong (id_rows[0].at ("id"));
      if (!id_opt)
        {
          rollback_savepoints ();
          telemetry::AddCounter (
              "cortext.memory_storage.embedding_error_total", 1);
          return;
        }
      const long long embedding_id = *id_opt;

      // 7. Insert MEMORIES row with aggregated metadata (v2 schema)
      std::any episode_id_any;
      if (p_ctx.episode_start_ts > 0)
        {
          episode_id_any = static_cast<long long> (p_ctx.episode_start_ts);
        }

      std::vector<float> ctx_vec;
      if (acc.c_t.size () > 0)
        {
          ctx_vec = EigenToFloatVec (acc.c_t);
        }

      const auto initial_trace = core::MemoryInitialTracePolicy (
          cfg.focus, cfg.sensitivity, cfg.stability);
      const double initial_strength = core::MemoryInitialStrengthPolicy (
          cfg.focus, cfg.sensitivity, cfg.stability);
      const double initial_stability = core::MemoryInitialStabilityPolicy (
          cfg.focus, cfg.sensitivity, cfg.stability);
      const auto insert_memory_start = SteadyClock::now ();
      savepoint->Execute (
          "INSERT INTO memories ("
          "  embedding_id, source_id, kind, start_ts, end_ts, n_signals, "
          "  modality, s_max, s_avg, s_emotion_max, s_arousal_avg, boundary_score, "
          "  drift_mag, emotion, ambient_mood, episode_id, "
          "  blob_id, created_at, context, source_origin, source_reliability, "
          "  strength, stability, "
          "  trace_fast, trace_med, trace_slow, trace_ultra"
          ") VALUES (?, ?, 'LONG_TERM', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
          { embedding_id, signal.source_id, static_cast<long long> (start_ts),
            static_cast<long long> (end_ts),
            static_cast<long long> (n_signals), primary_modality,
            s_max, s_avg, s_emotion_max, s_arousal_avg, boundary_score,
            drift_mag, emotion_blob, mood_blob, episode_id_any,
            content_blob_id.empty () ? std::any () : std::any (content_blob_id),
            static_cast<long long> (end_ts),
            ctx_vec.empty () ? std::any () : std::any (ctx_vec),
            origin, source_reliability,
            initial_strength, initial_stability,
            initial_trace.fast, initial_trace.medium, initial_trace.slow,
            initial_trace.ultra });
      context.AddOperationTiming ("MemoryStorage.insert_memory",
                                  ElapsedMillis (insert_memory_start));

      // 8. Get memory_id from inserted memories row
      const auto memory_id_start = SteadyClock::now ();
      auto mem_id_rows
          = savepoint->Execute ("SELECT last_insert_rowid() AS id", {});
      context.AddOperationTiming ("MemoryStorage.select_memory_id",
                                  ElapsedMillis (memory_id_start));
      const long long memory_id
          = mem_id_rows.empty ()
                ? 0
                : AnyToLongLong (mem_id_rows[0].at ("id")).value_or (0);

      // 9. Insert SIGNALS rows (one per tracked signal)
      std::optional<long long> stored_signal_id;
      std::optional<long long> current_signal_id;
      double signal_embedding_insert_ms = 0.0;
      double signal_row_insert_ms = 0.0;
      double signal_id_select_ms = 0.0;
      for (const auto &sig_rec : acc.signals)
        {
          const std::vector<float> signal_embedding
              = sig_rec.embedding.size () > 0
                    ? EigenToFloatVec (sig_rec.embedding)
                    : emb_vec;
          const auto signal_embedding_start = SteadyClock::now ();
          savepoint->Execute (insert_sql,
                              { signal_embedding,
                                static_cast<long long> (sig_rec.timestamp) });
          signal_embedding_insert_ms += ElapsedMillis (signal_embedding_start);

          const auto signal_embedding_id_start = SteadyClock::now ();
          auto signal_embedding_id_rows
              = savepoint->Execute ("SELECT last_insert_rowid() AS id", {});
          signal_id_select_ms += ElapsedMillis (signal_embedding_id_start);
          if (signal_embedding_id_rows.empty ()
              || signal_embedding_id_rows[0].count ("id") == 0)
            {
              rollback_savepoints ();
              telemetry::AddCounter (
                  "cortext.memory_storage.signal_embedding_error_total", 1);
              return;
            }
          const long long signal_embedding_id
              = AnyToLongLong (signal_embedding_id_rows[0].at ("id"))
                    .value_or (0);
          if (signal_embedding_id <= 0)
            {
              rollback_savepoints ();
              telemetry::AddCounter (
                  "cortext.memory_storage.signal_embedding_error_total", 1);
              return;
            }

          const auto signal_row_start = SteadyClock::now ();
          savepoint->Execute (
              "INSERT INTO signals ("
              "  memory_id, source_id, embedding_id, timestamp, modality, "
              "  mime, blob_id, serial_position, score, created_at"
              ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
              { memory_id, signal.source_id, signal_embedding_id,
                static_cast<long long> (sig_rec.timestamp), sig_rec.modality,
                sig_rec.mime,
                sig_rec.blob_id.empty () ? std::any ()
                                         : std::any (sig_rec.blob_id),
                static_cast<long long> (sig_rec.serial_position), sig_rec.score,
                static_cast<long long> (sig_rec.timestamp) });
          signal_row_insert_ms += ElapsedMillis (signal_row_start);
          const auto signal_id_start = SteadyClock::now ();
          auto signal_id_rows
              = savepoint->Execute ("SELECT last_insert_rowid() AS id", {});
          signal_id_select_ms += ElapsedMillis (signal_id_start);
          if (!signal_id_rows.empty ()
              && signal_id_rows[0].count ("id") != 0)
            {
              const auto signal_id
                  = AnyToLongLong (signal_id_rows[0].at ("id"));
              if (signal_id && *signal_id > 0)
                {
                  stored_signal_id = *signal_id;
                  if (sig_rec.timestamp == signal.timestamp
                      && sig_rec.modality == signal.modality)
                    {
                      current_signal_id = *signal_id;
                    }
                }
            }
        }
      context.AddOperationTiming ("MemoryStorage.insert_signal_embeddings",
                                  signal_embedding_insert_ms);
      context.AddOperationTiming ("MemoryStorage.insert_signal_rows",
                                  signal_row_insert_ms);
      context.AddOperationTiming ("MemoryStorage.select_signal_ids",
                                  signal_id_select_ms);

      // 10. Leave signal tracking until accumulator resets (used by WM gating)
      const auto supersession_start = SteadyClock::now ();
      WriteSupersessionEdges (context, *savepoint, memory_id,
                              embedding_to_store,
                              static_cast<long long> (end_ts));
      context.AddOperationTiming ("MemoryStorage.supersession_edges",
                                  ElapsedMillis (supersession_start));

      if (!constructive_recall::Disabled () && memory_id > 0)
        {
          const auto reconstruction_start = SteadyClock::now ();
          constructive_recall::AppendReconstructionWithEmbeddingId (
              *savepoint, memory_id, embedding_id, content_blob_id,
              static_cast<long long> (end_ts), 0.0, "initial", 1.0, 1.0);
          context.AddOperationTiming ("MemoryStorage.initial_reconstruction",
                                      ElapsedMillis (reconstruction_start));
        }

      // Commit external object content before releasing the DB savepoint. If
      // DB commit then fails, content-addressed payloads may be orphaned; the
      // reverse order can leave DB rows pointing at uncommitted object content.
      if (object_savepoint)
        {
          const auto object_commit_start = SteadyClock::now ();
          object_savepoint->Commit ();
          object_savepoint_finished = true;
          object_savepoint.reset ();
          context.AddOperationTiming ("MemoryStorage.commit_object_savepoint",
                                      ElapsedMillis (object_commit_start));
        }
      const auto savepoint_commit_start = SteadyClock::now ();
      savepoint->Commit ();
      savepoint_finished = true;
      context.AddOperationTiming ("MemoryStorage.commit_savepoints",
                                  ElapsedMillis (savepoint_commit_start));

      // Set stored_embedding_id in context for output
      context.SetStoredEmbeddingId (embedding_id);
      if (memory_id > 0)
        {
          context.SetStoredMemoryId (memory_id);
          context.SetStoredSignalId (
              current_signal_id.has_value () ? current_signal_id
                                             : stored_signal_id);
        }
      else
        {
          context.SetStoredMemoryId (std::nullopt);
          context.SetStoredSignalId (std::nullopt);
        }
      p_ctx.memories_since_consolidation += 1;
      if (memory_id > 0 && embedding_to_store.size () > 0)
        {
          const auto surface_start = SteadyClock::now ();
	          ProcessorContext::RetrievalSurfaceEntry surface_entry;
	          surface_entry.memory_id = memory_id;
	          surface_entry.embedding_id = embedding_id;
	          surface_entry.created_at = static_cast<long long> (end_ts);
	          surface_entry.start_ts = static_cast<long long> (start_ts);
          surface_entry.event_ts = static_cast<long long> (start_ts);
          surface_entry.kind = "LONG_TERM";
	          surface_entry.source_id = signal.source_id;
	          surface_entry.modality = primary_modality;
	          surface_entry.source_reliability = source_reliability;
          if (acc.c_t.size () > 0)
            {
              surface_entry.context_embedding = acc.c_t;
            }
          surface_entry.embedding = embedding_to_store;
          p_ctx.UpsertRetrievalSurface (std::move (surface_entry));

          const int k_key = core::SparseKeySize (
              context.GetConfig ().focus, context.GetConfig ().sensitivity,
              context.GetConfig ().stability);
          const std::string key = core::SparseKey (embedding_to_store, k_key);
          if (!key.empty ())
            {
              p_ctx.index_store[key].push_back (memory_id);
              p_ctx.index_reverse[memory_id] = key;
            }
          context.AddOperationTiming ("MemoryStorage.retrieval_surface_update",
                                      ElapsedMillis (surface_start));
        }
      telemetry::AddCounter ("cortext.memory_storage.stored_total", 1);
      telemetry::AddCounter ("cortext.memory_storage.signals_written_total",
                             static_cast<int64_t> (n_signals));

      // Debug logging
      telemetry::LogDebug (
          "cortext.memory_storage",
          { telemetry::Attribute::Bool ("stored", true),
            telemetry::Attribute::Int64 ("embedding_id", embedding_id),
            telemetry::Attribute::Int64 ("n_signals", n_signals),
            telemetry::Attribute::String ("primary_modality", primary_modality),
            telemetry::Attribute::Double ("s_max", s_max),
            telemetry::Attribute::Double ("s_avg", s_avg) });
    }
  catch (const std::exception &e)
    {
      rollback_savepoints ();
      telemetry::AddCounter ("cortext.memory_storage.error_total", 1);
      telemetry::LogError (
          "MemoryStorage failed",
          { telemetry::Attribute::String ("error", e.what ()) });
      throw;
    }
}

} // namespace cortext::operations
