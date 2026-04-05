#include "cortext/operations/memory_storage.hpp"
#include "constructive_recall_internal.hpp"
#include "cortext/processor/accumulator_state.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/signal.hpp"
#include "cortext/core/sparse.hpp"
#include "cortext/store/schema_helpers.hpp"
#include "cortext/store/store.hpp"
#include "cortext/store/utils.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include <algorithm>
#include <cstring>
#include <string>
#include <unordered_map>

namespace cortext::operations
{

namespace
{

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

/// @brief Map source_id to an origin label for source monitoring.
std::string
SourceOriginFor (const std::string &source_id)
{
  if (source_id.find ("user") != std::string::npos)
    return "user";
  if (source_id.find ("assistant") != std::string::npos)
    return "assistant";
  if (source_id.find ("system") != std::string::npos)
    return "system";
  return "external";
}

/// @brief Baseline source reliability prior.
double
SourcePriorReliability (const std::string &origin)
{
  if (origin == "user")
    return 0.8;
  if (origin == "assistant")
    return 0.6;
  if (origin == "system")
    return 0.9;
  return 0.7;
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
  auto savepoint = tx.Begin ();

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

      const std::string origin = SourceOriginFor (signal.source_id);
      const double source_reliability = SourcePriorReliability (origin);

      // 4. Require tracked per-signal records for persistence.
      if (acc.signals.empty ())
        {
          savepoint->Rollback ();
          telemetry::AddCounter (
              "cortext.memory_storage.missing_signals_total", 1);
          return;
        }

      // 5. Store memory-level content blob by concatenating signal blobs.
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
      const bool text_mode = (primary_modality == "text");
      for (const auto *rec : ordered_signals)
        {
          if (!rec || rec->blob_id.empty ())
            {
              continue;
            }
          auto blob_rows = savepoint->Execute (
              "SELECT objstore_get(?1) AS data", { rec->blob_id });
          if (!blob_rows.empty () && blob_rows[0].count ("data") != 0)
            {
              auto bytes = BlobFromAny (blob_rows[0].at ("data"));
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
      if (content_payload.empty () && signal.payload
          && !signal.payload->empty ())
        {
          content_payload = *signal.payload;
        }
      std::vector<unsigned char> content_blob_id;
      if (!content_payload.empty ())
        {
          auto blob_rows = savepoint->Execute ("SELECT objstore_put(?1) AS id",
                                               { content_payload });
          if (!blob_rows.empty () && blob_rows[0].count ("id") != 0)
            {
              content_blob_id = BlobFromAny (blob_rows[0].at ("id"));
            }
        }

      // 6. Insert memory embedding (v2: minimal sqlite-vec table)
      const std::string insert_sql = std::string ("INSERT INTO embeddings (")
                                     + store::kEmbeddingsInsertColumns
                                     + ") VALUES ("
                                     + store::kEmbeddingsInsertDefaults + ")";
      savepoint->Execute (insert_sql,
                          { emb_vec, static_cast<long long> (end_ts) });

      auto id_rows = savepoint->Execute ("SELECT last_insert_rowid() AS id", {});
      if (id_rows.empty () || id_rows[0].count ("id") == 0)
        {
          savepoint->Rollback ();
          telemetry::AddCounter (
              "cortext.memory_storage.embedding_error_total", 1);
          return;
        }
      const auto id_opt = AnyToLongLong (id_rows[0].at ("id"));
      if (!id_opt)
        {
          savepoint->Rollback ();
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

      savepoint->Execute (
          "INSERT INTO memories ("
          "  embedding_id, source_id, kind, start_ts, end_ts, n_signals, "
          "  modality, s_max, s_avg, s_emotion_max, s_arousal_avg, boundary_score, "
          "  drift_mag, emotion, ambient_mood, episode_id, "
          "  blob_id, created_at, context, source_origin, source_reliability, "
          "  trace_fast, trace_med, trace_slow, trace_ultra"
          ") VALUES (?, ?, 'LONG_TERM', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
          { embedding_id, signal.source_id, static_cast<long long> (start_ts),
            static_cast<long long> (end_ts),
            static_cast<long long> (n_signals), primary_modality,
            s_max, s_avg, s_emotion_max, s_arousal_avg, boundary_score,
            drift_mag, emotion_blob, mood_blob, episode_id_any,
            content_blob_id.empty () ? std::any () : std::any (content_blob_id),
            static_cast<long long> (end_ts),
            ctx_vec.empty () ? std::any () : std::any (ctx_vec),
            origin, source_reliability,
            1.0, 0.0, 0.0, 0.0 });

      // 8. Get memory_id from inserted memories row
      auto mem_id_rows
          = savepoint->Execute ("SELECT last_insert_rowid() AS id", {});
      const long long memory_id
          = mem_id_rows.empty ()
                ? 0
                : AnyToLongLong (mem_id_rows[0].at ("id")).value_or (0);

      // 9. Insert SIGNALS rows (one per tracked signal)
      for (const auto &sig_rec : acc.signals)
        {
          long long signal_embedding_id = embedding_id;
          if (sig_rec.embedding.size () > 0)
            {
              const std::vector<float> sig_vec
                  = EigenToFloatVec (sig_rec.embedding);
              savepoint->Execute (insert_sql,
                                  { sig_vec,
                                    static_cast<long long> (sig_rec.timestamp) });
              auto sig_id_rows
                  = savepoint->Execute ("SELECT last_insert_rowid() AS id", {});
              if (!sig_id_rows.empty () && sig_id_rows[0].count ("id") != 0)
                {
                  signal_embedding_id
                      = AnyToLongLong (sig_id_rows[0].at ("id"))
                            .value_or (embedding_id);
                }
            }

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
        }

      // 10. Leave signal tracking until accumulator resets (used by WM gating)
      if (!constructive_recall::Disabled () && memory_id > 0)
        {
          constructive_recall::AppendReconstructionWithEmbeddingId (
              *savepoint, memory_id, embedding_id, content_blob_id,
              static_cast<long long> (end_ts), 0.0, "initial", 1.0, 1.0);
        }

      // Commit the savepoint
      savepoint->Commit ();

      // Set stored_embedding_id in context for output
      context.SetStoredEmbeddingId (embedding_id);
      p_ctx.memories_since_consolidation += 1;
      if (memory_id > 0 && embedding_to_store.size () > 0)
        {
          const int k_key = core::SparseKeySize (context.GetConfig ().focus);
          const std::string key = core::SparseKey (embedding_to_store, k_key);
          if (!key.empty ())
            {
              p_ctx.index_store[key].push_back (memory_id);
              p_ctx.index_reverse[memory_id] = key;
            }
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
      savepoint->Rollback ();
      telemetry::AddCounter ("cortext.memory_storage.error_total", 1);
      telemetry::LogError (
          "MemoryStorage failed",
          { telemetry::Attribute::String ("error", e.what ()) });
    }
}

} // namespace cortext::operations
