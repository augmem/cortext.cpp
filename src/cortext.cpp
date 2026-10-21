#include "cortext/cortext.hpp"
#include "cortext_pipeline_internal.hpp"
#include "cortext/clock.hpp"
#include "cortext/core/algorithms.hpp"
#include "cortext/core/knobs.hpp"
#include "cortext/core/utils.hpp"
#include "cortext/internal/cancellation.hpp"
#include "cortext/models/embedding_model_pin.hpp"
#include "cortext/internal/replay_ingress.hpp"
#include "encoder/text_encoder_factory.hpp"
#include "models/aist_late_interaction.hpp"
#include "models/aist_semantic_cache_internal.hpp"
#include "operations/constructive_recall_internal.hpp"
#include "operations/retrieval_trace_state.hpp"
#include "operations/signal_record_rollback_internal.hpp"
#include "operations/sparse_retrieval_knobs_internal.hpp"
#include "contiguity_internal.hpp"
#include "data_time_clock_internal.hpp"
#include "experimental_env.hpp"
#include "store/memory_text_index_internal.hpp"
#include "signal_filter.hpp"
#include "streaming_text_probe.hpp"

#include "cortext/processor.hpp"
#include "cortext/processor/operation_set.hpp"
#include "cortext/signal.hpp"
#include "cortext/store/object_store.hpp"
#include "cortext/store/sqlite_store.hpp"
#include "cortext/store/utils.hpp"
#include "cortext/telemetry/telemetry.hpp"

#include "cortext/operations/blend.hpp"
#include "cortext/operations/graph_retrieval.hpp"

#include "cortext/operations/embedding_prediction_error.hpp"
#include "cortext/operations/centroids.hpp"
#include "cortext/operations/precision.hpp"
#include "cortext/operations/sensitivity.hpp"
#include "cortext/operations/threshold.hpp"
#include "cortext/operations/uncertainty.hpp"
#include "cortext/processor/operation.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iterator>
#include <limits>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cortext/operations/boundary.hpp"
#include "cortext/operations/coherence.hpp"
#include "cortext/operations/effective_focus.hpp"
#include "cortext/operations/focus.hpp"
#include "cortext/operations/focus_spread.hpp"
#include "cortext/operations/interrupt_gate.hpp"
#include "cortext/operations/memory_usage_record.hpp"
#include "cortext/operations/metrics.hpp"
#include "cortext/operations/neuromodulators.hpp"
#include "cortext/operations/recent_context.hpp"
#include "cortext/operations/signal_metrics_persistence.hpp"
#include "cortext/operations/stability.hpp"
#include "cortext/operations/write_gate.hpp"
#include "cortext/operations/memory_storage.hpp"
#include "cortext/operations/working_memory.hpp"
#include "cortext/operations/detect_memory_usage.hpp"
#include "cortext/operations/drift_accumulation.hpp"
#include "cortext/operations/streaming_pacing.hpp"

// Section 4.4: Memory Accumulation
#include "cortext/operations/accumulator.hpp"
#include "cortext/operations/accumulator_scores.hpp"
#include "cortext/operations/accumulator_reset.hpp"
#include "cortext/operations/coherence.hpp"
#include "cortext/operations/boundary.hpp"
#include "cortext/operations/spike_bypass.hpp"
#include "cortext/operations/write_gate.hpp"

// Phase 4: Knowledge Graph Enhancement
#include "cortext/telemetry/telemetry.hpp"

namespace cortext
{

namespace
{

constexpr const char *kMaintenanceSourceId = "cortext/maintenance";

bool
SourceBlobsDisabled ()
{
  return internal::experimental_env::Flag ("CORTEXT_DISABLE_SOURCE_BLOBS");
}

#if !defined(CORTEXT_DISABLE_IMAGE)
std::string
RawImageMime (int width, int height, int channels)
{
  return "image/raw;width=" + std::to_string (width)
         + ";height=" + std::to_string (height)
         + ";channels=" + std::to_string (channels);
}

std::size_t
RawImageBytes (int width, int height, int channels)
{
  return static_cast<std::size_t> (width) * static_cast<std::size_t> (height)
         * static_cast<std::size_t> (channels);
}
#endif

#if !defined(CORTEXT_DISABLE_AUDIO)
void
ValidateAudioInput (const float *pcm)
{
  if (pcm == nullptr)
    {
      throw std::invalid_argument ("pcm must not be NULL");
    }
}
#endif

#if !defined(CORTEXT_DISABLE_IMAGE)
void
ValidateImageInput (const std::uint8_t *data, int width, int height,
                    int channels)
{
  if (data == nullptr)
    {
      throw std::invalid_argument ("data must not be NULL");
    }
  if (width <= 0 || height <= 0 || channels <= 0)
    {
      throw std::invalid_argument (
          "width, height, and channels must be positive");
    }
}
#endif

void
ValidateMedia (const Cortext::Media &media)
{
  if (media.size == 0)
    {
      return;
    }
  if (!media.data)
    {
      throw std::invalid_argument (
          "media.data must not be NULL when media.size is non-zero");
    }
  if (media.mimetype.empty ())
    {
      throw std::invalid_argument (
          "media.mimetype must not be empty when media data is provided");
    }
}

// The durable payload is the caller's Media when it carries bytes, else the
// canonical processing input (raw pixels / f32le PCM). Every stored signal
// thus keeps a payload, and with it a searchable shared-space vector.
void
ApplyStoredMedia (Signal &signal, const Cortext::Media &media,
                  const void *canonical, std::size_t canonical_bytes,
                  std::string canonical_mimetype)
{
  ValidateMedia (media);
  if (media.size == 0)
    {
      const auto *bytes = static_cast<const unsigned char *> (canonical);
      signal.payload
          = std::vector<unsigned char> (bytes, bytes + canonical_bytes);
      signal.mimetype = std::move (canonical_mimetype);
      return;
    }

  signal.payload = std::vector<unsigned char> (media.data,
                                               media.data + media.size);
  signal.mimetype = media.mimetype;
}

#if defined(CORTEXT_DISABLE_AUDIO)
[[noreturn]] void
ThrowAudioSupportDisabled ()
{
  throw std::runtime_error (
      "Cortext audio support was disabled at build time; configure with "
      "CORTEXT_ENABLE_AUDIO=ON.");
}
#endif

#if defined(CORTEXT_DISABLE_IMAGE)
[[noreturn]] void
ThrowImageSupportDisabled ()
{
  throw std::runtime_error (
      "Cortext image support was disabled at build time; configure with "
      "CORTEXT_ENABLE_IMAGE=ON.");
}
#endif

std::string
RowString (const std::map<std::string, std::any> &row, const char *key)
{
  auto it = row.find (key);
  if (it == row.end () || !it->second.has_value ())
    {
      return {};
    }
  if (it->second.type () == typeid (std::string))
    {
      return std::any_cast<std::string> (it->second);
    }
  return {};
}

/// @brief Converts a std::vector<float> to an Eigen::VectorXf using Eigen::Map.
inline Eigen::VectorXf
ToEigen (const std::vector<float> &v)
{
  return Eigen::Map<const Eigen::VectorXf> (v.data (),
                                            static_cast<int> (v.size ()));
}

void
NormalizeVector (std::vector<float> &v)
{
  double sum = 0.0;
  for (float value : v)
    {
      sum += static_cast<double> (value) * value;
    }
  const double norm = std::sqrt (sum);
  if (norm <= 1e-12 || !std::isfinite (norm))
    {
      return;
    }
  const float inv_norm = static_cast<float> (1.0 / norm);
  for (float &value : v)
    {
      value *= inv_norm;
    }
}

std::vector<float>
RetrievalEmbeddingView (const std::vector<float> &encoded)
{
  std::vector<float> out = encoded;
  if (out.size () > 256)
    {
      out.resize (256);
      NormalizeVector (out);
    }
  return out;
}

std::vector<float>
ToStdVector (const Eigen::VectorXf &v)
{
  return std::vector<float> (v.data (), v.data () + v.size ());
}

internal::SignalFilterConfig
MakeSignalFilterConfig (const Cortext::Config &cfg)
{
  internal::SignalFilterConfig filter_cfg;
  filter_cfg.focus = cfg.focus;
  filter_cfg.sensitivity = cfg.sensitivity;
  filter_cfg.stability = cfg.stability;
  filter_cfg.audio_enabled = cfg.signal_filter_audio_enabled;
  filter_cfg.image_enabled = cfg.signal_filter_image_enabled;
  filter_cfg.text_enabled = cfg.signal_filter_text_enabled;
  return filter_cfg;
}

inline double
ToMillis (std::chrono::steady_clock::duration duration)
{
  return std::chrono::duration_cast<std::chrono::duration<double, std::milli>> (
             duration)
      .count ();
}

void
ApplySignalFilterDecision (Cortext::ProcessorOutput &output,
                           const internal::SignalFilterDecision &decision)
{
  output.signal_filter_evaluated = decision.evaluated;
  output.signal_filter_accepted = decision.accepted;
  output.signal_filter_modality = decision.modality;
  output.signal_filter_reason = decision.reason;
  output.signal_filter_score = decision.score;
  output.signal_filter_threshold = decision.threshold;
  output.signal_filter_quiet_seconds = decision.quiet_seconds;
}

// Query reference time travels on the signal as time metadata. Only an
// Ephemeral (query) signal carries it, so ingest-time retrieval of storing
// calls is unchanged; as_of without a reference time is ignored.
void
ApplyRetrievalTime (Signal &signal, const Cortext::RetrievalTime &time)
{
  if (signal.retention != Retention::Ephemeral || !time.reference_ms)
    return;
  signal.reference_ts = time.reference_ms;
  signal.reference_as_of = time.as_of;
}

Cortext::Context
MakeFilteredContext (const internal::SignalFilterDecision &decision,
                     std::chrono::steady_clock::time_point total_start)
{
  Cortext::Context result;
  ApplySignalFilterDecision (result.output, decision);
  result.total_ms = ToMillis (std::chrono::steady_clock::now () - total_start);
  return result;
}

void
ApplyRetrievalScore (Cortext::Context::Memory &memory,
                     const std::unordered_map<long long,
                                              operations::retrieval_trace::RankedCandidate>
                         &ranked_by_memory_id)
{
  auto it = ranked_by_memory_id.find (memory.id);
  if (it == ranked_by_memory_id.end ())
    {
      return;
    }
  memory.composite_score = it->second.score;
  memory.relevance = it->second.relevance;
}

// An object-store read transaction shared by a run of hydrations, so each
// payload read does not open and commit a transaction of its own. Reads only:
// the run must not write. object_tx is set when the cortext has an object
// store, else db_tx is the metadata transaction objects are read through.
struct SharedObjectReads
{
  ObjectTransaction *object_tx = nullptr;
  Transaction *db_tx = nullptr;
};

bool
LoadObjectPayload (Store *store, ObjectStore *object_store,
                   const std::vector<unsigned char> &blob_id,
                   std::vector<unsigned char> &out,
                   SharedObjectReads *shared = nullptr)
{
  if (blob_id.empty ())
    {
      return false;
    }
  try
    {
      if (shared && (shared->object_tx || shared->db_tx))
        {
          auto payload = shared->object_tx
                             ? shared->object_tx->Get (blob_id)
                             : GetObject (nullptr, *shared->db_tx, blob_id);
          if (payload && !payload->empty ())
            {
              out = std::move (*payload);
              return true;
            }
        }
      else if (object_store)
        {
          auto tx = object_store->Begin ();
          auto payload = tx->Get (blob_id);
          tx->Commit ();
          if (payload && !payload->empty ())
            {
              out = std::move (*payload);
              return true;
            }
        }
      else if (store)
        {
          auto tx = store->Begin ();
          auto payload = GetObject (nullptr, *tx, blob_id);
          tx->Commit ();
          if (payload && !payload->empty ())
            {
              out = std::move (*payload);
              return true;
            }
        }
    }
  catch (const std::exception &e)
    {
      telemetry::LogWarn (
          "Failed to load objstore payload",
          { telemetry::Attribute::String ("component", "cortext"),
            telemetry::Attribute::String ("error", e.what ()) });
    }
  catch (...)
    {
      telemetry::LogWarn (
          "Failed to load objstore payload (unknown error)",
          { telemetry::Attribute::String ("component", "cortext") });
    }
  return false;
}

bool
IsLikelyTextPayload (const std::vector<unsigned char> &payload)
{
  if (payload.empty ())
    {
      return false;
    }
  std::size_t printable = 0;
  for (unsigned char c : payload)
    {
      if (c == 0)
        {
          return false;
        }
      if (c == '\n' || c == '\r' || c == '\t'
          || (c >= 0x20 && c < 0x7F) || c >= 0x80)
        {
          ++printable;
        }
    }
  return printable == payload.size ();
}

bool
PayloadMatchesMemorySurface (const std::string &memory_modality,
                             const std::string &memory_mime,
                             const std::string &signal_modality,
                             const std::string &signal_mime,
                             const std::vector<unsigned char> &payload)
{
  if (!memory_modality.empty () && signal_modality != memory_modality)
    {
      return false;
    }
  // Text modality can carry caller-provided binary payloads (for example a
  // token KV cache). Only canonical text/plain payloads require text-content
  // validation; a custom MIME type declares an opaque payload surface.
  if (memory_mime == "text/plain"
      || (memory_mime.empty () && memory_modality == "text")
      || signal_mime == "text/plain")
    {
      return IsLikelyTextPayload (payload);
    }
  return true;
}

/// @brief Load all signal blobs for a memory, ordered by serial_position.
/// @param store The store instance.
/// @param memory_id The memory_id to query signals for.
/// @param out Vector of blobs, one per signal.
/// @param out_timestamps Stored time of each blob's signal, in blob order.
/// @return true if any blobs were loaded.
bool
LoadSignalBlobs (Store *store, long long memory_id,
                 ObjectStore *object_store,
                 const std::string &memory_modality,
                 const std::string &memory_mime,
                 int max_signal_blobs,
                 std::size_t *loaded_signal_blobs,
                 std::vector<std::vector<unsigned char>> &out,
                 std::vector<std::string> &out_mimetypes,
                 std::vector<std::uint64_t> &out_timestamps,
                 SharedObjectReads *shared = nullptr)
{
  if (!store || memory_id <= 0)
    return false;
  // A mixed memory returns the payloads of all its signals.
  const bool mixed = memory_modality == "mixed";
  const std::string modality_filter = mixed ? std::string () : memory_modality;
  const std::string mime_filter = mixed ? std::string () : memory_mime;

  try
    {
      const long long limit = std::max (1, max_signal_blobs);
      long long cursor_position = std::numeric_limits<long long>::max ();
      long long cursor_signal_id = std::numeric_limits<long long>::max ();
      std::vector<std::vector<unsigned char>> matched_payloads;
      std::vector<std::string> matched_mimetypes;
      std::vector<std::uint64_t> matched_timestamps;
      matched_payloads.reserve (static_cast<std::size_t> (limit));
      while (static_cast<long long> (matched_payloads.size ()) < limit)
        {
          auto rows = store->Execute (
              "SELECT signal_id, modality, mime, blob_id, timestamp, "
              "       COALESCE(serial_position, signal_id) AS sort_position "
              "FROM signals "
              "WHERE memory_id = ? AND blob_id IS NOT NULL "
              "  AND (? = '' OR modality = ?) "
              "  AND (COALESCE(serial_position, signal_id) < ? "
              "       OR (COALESCE(serial_position, signal_id) = ? "
              "           AND signal_id < ?)) "
              "ORDER BY sort_position DESC, signal_id DESC "
              "LIMIT ?",
              { memory_id, modality_filter, modality_filter,
                cursor_position, cursor_position, cursor_signal_id, limit });
          if (rows.empty ())
            break;

          for (const auto &row : rows)
            {
              const auto position = store::AnyToLongLong (
                  row.at ("sort_position"));
              const auto signal_id = store::AnyToLongLong (
                  row.at ("signal_id"));
              if (!position || !signal_id)
                throw std::runtime_error (
                    "invalid fallback signal ordering row");
              cursor_position = *position;
              cursor_signal_id = *signal_id;

              auto it = row.find ("blob_id");
              if (it != row.end () && it->second.has_value ())
                {
                  auto blob_id = store::BlobFromAny (it->second);
                  if (!blob_id.empty ())
                    {
                      std::vector<unsigned char> payload;
                      if (LoadObjectPayload (store, object_store, blob_id,
                                             payload, shared))
                        {
                          const std::string signal_modality
                              = RowString (row, "modality");
                          const std::string signal_mime
                              = RowString (row, "mime");
                          if (PayloadMatchesMemorySurface (
                                  modality_filter, mime_filter,
                                  signal_modality, signal_mime, payload))
                            {
                              matched_payloads.push_back (
                                  std::move (payload));
                              matched_mimetypes.push_back (signal_mime);
                              const auto ts = row.find ("timestamp");
                              matched_timestamps.push_back (
                                  static_cast<std::uint64_t> (std::max (
                                      0LL,
                                      ts == row.end ()
                                          ? 0LL
                                          : store::AnyToLongLong (ts->second)
                                                .value_or (0))));
                              if (static_cast<long long> (
                                      matched_payloads.size ())
                                  == limit)
                                break;
                            }
                        }
                    }
                }
            }
          if (static_cast<long long> (rows.size ()) < limit)
            break;
        }
      std::reverse (matched_payloads.begin (), matched_payloads.end ());
      std::reverse (matched_mimetypes.begin (), matched_mimetypes.end ());
      std::reverse (matched_timestamps.begin (), matched_timestamps.end ());
      if (loaded_signal_blobs)
        *loaded_signal_blobs += matched_payloads.size ();
      out.insert (out.end (),
                  std::make_move_iterator (matched_payloads.begin ()),
                  std::make_move_iterator (matched_payloads.end ()));
      out_mimetypes.insert (out_mimetypes.end (),
                            std::make_move_iterator (matched_mimetypes.begin ()),
                            std::make_move_iterator (matched_mimetypes.end ()));
      out_timestamps.insert (out_timestamps.end (), matched_timestamps.begin (),
                             matched_timestamps.end ());
      return !out.empty ();
    }
  catch (const std::exception &e)
    {
      telemetry::LogWarn (
          "Failed to load signal blobs",
          { telemetry::Attribute::String ("component", "cortext"),
            telemetry::Attribute::Int64 ("memory_id", memory_id),
            telemetry::Attribute::String ("error", e.what ()) });
    }
  catch (...)
    {
      telemetry::LogWarn (
          "Failed to load signal blobs (unknown error)",
          { telemetry::Attribute::String ("component", "cortext"),
            telemetry::Attribute::Int64 ("memory_id", memory_id) });
    }
  return false;
}

bool
LoadContentBlob (Store *store, ObjectStore *object_store,
                 const std::vector<unsigned char> &blob_id,
                 const std::string &memory_modality,
                 const std::string &memory_mime,
                 std::uint64_t blob_timestamp,
                 std::vector<std::vector<unsigned char>> &out,
                 std::vector<std::string> &out_mimetypes,
                 std::vector<std::uint64_t> &out_timestamps,
                 SharedObjectReads *shared = nullptr)
{
  if (!store || blob_id.empty ())
    return false;

  std::vector<unsigned char> payload;
  if (LoadObjectPayload (store, object_store, blob_id, payload, shared)
      && PayloadMatchesMemorySurface (memory_modality, memory_mime,
                                      memory_modality, memory_mime, payload))
    {
      out.push_back (std::move (payload));
      out_mimetypes.push_back (memory_mime);
      out_timestamps.push_back (blob_timestamp);
      return true;
    }
  return false;
}

void
HydrateMemory (Store *store, ObjectStore *object_store, long long id,
               Cortext::Context::Memory &m,
               int max_fallback_signal_blobs,
               std::size_t *loaded_fallback_signal_blobs,
               SharedObjectReads *shared = nullptr)
{
  if (!store)
    return;
  try
    {
      // v2: Query uses unified memories table
      // Mimetype is at the signal level - get from first signal
      auto rows = store->Execute (
          "SELECT "
          "  m.memory_id, m.modality, m.source_id, m.start_ts, m.end_ts, "
          "  m.created_at, "
          "  m.n_signals, m.s_max, m.s_avg, "
          "  m.blob_id, "
          "  COALESCE(m.retrieved_count, 0) AS retrieved_count, "
          "  COALESCE(m.used_count, 0) AS used_count, "
          "  COALESCE("
          "    (SELECT s.mime FROM signals s WHERE s.memory_id = m.memory_id "
          "       AND s.modality = m.modality "
          "     ORDER BY s.serial_position LIMIT 1), "
          "    (SELECT s.mime FROM signals s WHERE s.memory_id = m.memory_id "
          "     ORDER BY s.serial_position LIMIT 1)"
          "  ) AS signal_mime, "
          "  COALESCE("
          "    (SELECT s.timestamp FROM signals s "
          "     WHERE s.memory_id = m.memory_id AND s.modality = m.modality "
          "     ORDER BY s.serial_position LIMIT 1), "
          "    (SELECT s.timestamp FROM signals s WHERE s.memory_id = m.memory_id "
          "     ORDER BY s.serial_position LIMIT 1)"
          "  ) AS signal_ts "
          "FROM memories m "
          "WHERE m.memory_id = ?",
          { id });

      if (!rows.empty ())
        {
          const auto &row = rows[0];

          auto get_s = [&row] (const char *k) -> std::string {
            auto it = row.find (k);
            if (it == row.end () || !it->second.has_value ())
              return {};
            if (it->second.type () == typeid (std::string))
              return std::any_cast<std::string> (it->second);
            return {};
          };

          auto get_ll = [&row] (const char *k) -> long long {
            auto it = row.find (k);
            if (it == row.end () || !it->second.has_value ())
              return 0LL;
            if (it->second.type () == typeid (long long))
              return std::any_cast<long long> (it->second);
            if (it->second.type () == typeid (int))
              return static_cast<long long> (std::any_cast<int> (it->second));
            return 0LL;
          };

          [[maybe_unused]] auto get_dbl = [&row] (const char *k, double def = 0.0) -> double {
            auto it = row.find (k);
            if (it == row.end () || !it->second.has_value ())
              return def;
            return store::AnyToDouble (it->second, def);
          };

          [[maybe_unused]] auto get_blob = [&row] (const char *k) {
            auto it = row.find (k);
            if (it == row.end () || !it->second.has_value ())
              return std::vector<unsigned char> ();
            return store::BlobFromAny (it->second);
          };

          // v2: Populate from unified memories table
          const long long memory_id = get_ll ("memory_id");
          m.id = memory_id;
          m.modality = get_s ("modality");
          m.mimetype = get_s ("signal_mime");  // Get from first signal
          m.source_id = get_s ("source_id");
          const long long end_ts = get_ll ("end_ts");
          const long long start_ts = get_ll ("start_ts");
          const long long created_at = get_ll ("created_at");
          const long long timestamp
              = end_ts > 0 ? end_ts
                           : (start_ts > 0 ? start_ts : created_at);
          m.timestamp = static_cast<std::uint64_t> (timestamp);
          // A memory-level blob (a reconstruction, or the memory's content
          // joined from its signals) carries the time of the signal it
          // starts with, the one signal_mime comes from.
          const auto memory_blob_ts = static_cast<std::uint64_t> (
              std::max (0LL, get_ll ("signal_ts")));

          const auto reconstruction
              = operations::constructive_recall::LoadLatestReconstruction (
                  store, memory_id);
          if (reconstruction.has_value () && !reconstruction->blob_id.empty ())
            {
              std::vector<unsigned char> payload;
              if (LoadObjectPayload (store, object_store,
                                     reconstruction->blob_id, payload,
                                     shared)
                  && PayloadMatchesMemorySurface (
                      m.modality, m.mimetype, m.modality, m.mimetype, payload))
                {
                  m.content.push_back (std::move (payload));
                  m.content_mimetypes.push_back (m.mimetype);
                  m.content_timestamps.push_back (memory_blob_ts);
                }
            }

          if (m.content.empty ())
            {
              const auto blob_id = get_blob ("blob_id");
              LoadContentBlob (store, object_store, blob_id, m.modality,
                               m.mimetype, memory_blob_ts, m.content,
                               m.content_mimetypes, m.content_timestamps,
                               shared);
            }

          if (m.content.empty ())
            {
              // v2 fallback: old databases may only have signal-level blobs.
              LoadSignalBlobs (store, memory_id, object_store, m.modality,
                               m.mimetype, max_fallback_signal_blobs,
                               loaded_fallback_signal_blobs,
                               m.content, m.content_mimetypes,
                               m.content_timestamps, shared);
            }

          // v2: Counts are inline on memories table
          m.retrieved_count = get_ll ("retrieved_count");
          m.used_count = get_ll ("used_count");
        }
    }
  catch (const std::exception &e)
    {
      telemetry::LogWarn (
          "Failed to hydrate memory",
          { telemetry::Attribute::String ("component", "cortext"),
            telemetry::Attribute::Int64 ("memory_id", id),
            telemetry::Attribute::String ("error", e.what ()) });
    }
  catch (...)
    {
      telemetry::LogWarn (
          "Failed to hydrate memory (unknown error)",
          { telemetry::Attribute::String ("component", "cortext"),
            telemetry::Attribute::Int64 ("memory_id", id) });
    }
}

bool
HasHydratedContent (const Cortext::Context::Memory &memory)
{
  for (const auto &blob : memory.content)
    {
      if (!blob.empty ())
        {
          return true;
        }
    }
  return false;
}

void
StripSourceBlobsIfDisabled (Cortext::Context::Memory &memory)
{
  if (SourceBlobsDisabled ())
    {
      memory.content.clear ();
      memory.content_mimetypes.clear ();
      memory.content_timestamps.clear ();
    }
}

// Episodic contiguity completion for retrieved conversational turns.
//
// Recalling one moment of a conversation brings back what was said around it
// (the temporal-contiguity effect of episodic recall), and the effect fades
// with distance. A retrieved turn is rarely self-contained: a question's
// answer, the statement an acknowledgement refers to, or the user message an
// assistant reply was written for lives one or two turns away. For each
// retrieved turn, of any modality, the adjacent turns from the same source
// stream (adjacent LONG_TERM rows in write order, within a short time gap;
// contiguity_internal.hpp) are hydrated as extra content blobs of that same
// retrieved memory under a per-context byte budget, text clipped to the
// window that best covers a text query's terms and other media whole,
// charged as one clipped neighbour each: first the immediate successor and
// predecessor of every anchor in rank order, then, with what budget is
// left, the turns two steps away. A 2026-10 sweep found the old 6 KiB budget and
// lag-2 cap (top 6 anchors) binding; 12 KiB with lag 2 for every anchor raised
// judged sufficiency on Eval 1 by +0.037 (budgets past 12 KiB change nothing). Cost is O(anchors): primary-key range lookups
// bounded by a fixed id span plus one indexed supersession check per
// neighbor, issued as two statements (lag 1, lag 2) for all anchors. Query
// retrieval does not spend anchor slots on these neighbours
// (DeferAttachedNeighbours in graph_retrieval.cpp).
constexpr std::size_t kContiguityAnchorLimit = 11;
constexpr std::size_t kContiguityNeighborMaxBytes = 480;
constexpr std::size_t kContiguityContextBudgetBytes = 12288;
constexpr long long kContiguityIdScanSpan = internal::contiguity::kIdScanSpan;
constexpr long long kContiguityMaxGapMs = internal::contiguity::kMaxGapMs;

bool
IsTextMime (const std::string &mimetype)
{
  return mimetype.empty () || mimetype.rfind ("text/", 0) == 0;
}

// True when the memory carries text among its content (a text memory, or a
// mixed memory with text signals); index i of content is text when its MIME
// type is text/*.
bool
HasTextContent (const Cortext::Context::Memory &memory)
{
  for (std::size_t i = 0; i < memory.content.size (); ++i)
    {
      const std::string mime = i < memory.content_mimetypes.size ()
                                   ? memory.content_mimetypes[i]
                                   : memory.mimetype;
      if (!memory.content[i].empty () && IsTextMime (mime))
        return true;
    }
  return false;
}

// Before inserting attached blobs: one MIME type and one signal time per
// existing content blob. Every hydration path records a blob's time, so the
// 0 (unknown) padding only covers content a caller built without one.
void
AlignContentMimetypes (Cortext::Context::Memory &memory)
{
  memory.content_mimetypes.resize (memory.content.size (), memory.mimetype);
  memory.content_timestamps.resize (memory.content.size (), 0);
}

std::string
SourceStreamOf (const std::string &source_id)
{
  return internal::contiguity::StreamOf (source_id);
}

long long
RowLongLong (const std::map<std::string, std::any> &row, const char *key)
{
  const auto it = row.find (key);
  if (it == row.end () || !it->second.has_value ())
    {
      return 0;
    }
  return store::AnyToLongLong (it->second).value_or (0);
}

// Clip text bytes to max_bytes on a UTF-8 code point boundary.
std::vector<unsigned char>
ClipTextBlob (const std::vector<unsigned char> &text, std::size_t max_bytes)
{
  if (text.size () <= max_bytes)
    {
      return text;
    }
  std::size_t cut = max_bytes;
  while (cut > 0 && (text[cut] & 0xC0u) == 0x80u)
    {
      --cut;
    }
  std::vector<unsigned char> out (text.begin (),
                                  text.begin () + static_cast<std::ptrdiff_t> (cut));
  static constexpr char kEllipsis[] = "...";
  out.insert (out.end (), kEllipsis, kEllipsis + 3);
  return out;
}

// Terms that pick the clipped window of attached text: the query's
// keywords (the keyword channel's stopword-filtered words) of 3+ bytes,
// lowercased and cut to 5 bytes so inflections match ("assemble",
// "assembly"). Only a query (Ephemeral retention) that carries text has
// terms; any other query, and every ingest-time hydration, keeps the
// leading window, so ingest pays nothing for the windowing scan.
std::vector<std::string>
ClipTermsOf (const cortext::Signal *query)
{
  if (!query || query->retention != Retention::Ephemeral || !query->payload
      || !IsTextMime (query->mimetype))
    return {};
  const std::string_view text (
      reinterpret_cast<const char *> (query->payload->data ()),
      query->payload->size ());
  std::vector<std::string> terms;
  for (auto &w : store::memory_text_index_internal::QueryWords (text, 32))
    {
      if (w.size () < 3)
        continue;
      if (w.size () > 5)
        w.resize (5);
      if (std::find (terms.begin (), terms.end (), w) == terms.end ())
        terms.push_back (std::move (w));
    }
  return terms;
}

// Clips text to max_bytes keeping the window with the most distinct query
// terms: the first kClipHeadBytes (turn header: time, speaker) stay, then
// the best window. Falls back to the head clip when no window beats it.
std::vector<unsigned char>
ClipTextAroundTerms (const std::vector<unsigned char> &text,
                     std::size_t max_bytes,
                     const std::vector<std::string> &terms)
{
  if (text.size () <= max_bytes)
    return text;
  constexpr std::size_t kClipHeadBytes = 64;
  const auto is_word = [] (unsigned char c) {
    return c >= 0x80 || std::isalnum (c);
  };
  // (position, term index) of every word that starts with a term.
  std::vector<std::pair<std::size_t, std::size_t> > hits;
  for (std::size_t i = 0; i < text.size ();)
    {
      while (i < text.size () && !is_word (text[i]))
        ++i;
      const std::size_t start = i;
      while (i < text.size () && is_word (text[i]))
        ++i;
      if (i == start)
        continue;
      for (std::size_t t = 0; t < terms.size (); ++t)
        {
          const auto &term = terms[t];
          if (i - start < term.size ())
            continue;
          bool match = true;
          for (std::size_t k = 0; k < term.size () && match; ++k)
            match = static_cast<char> (std::tolower (text[start + k])) == term[k];
          if (match)
            {
              hits.emplace_back (start, t);
              break;
            }
        }
    }
  const auto distinct_in = [&] (std::size_t lo, std::size_t hi) {
    std::vector<bool> seen (terms.size (), false);
    std::size_t n = 0;
    for (const auto &[pos, t] : hits)
      if (pos >= lo && pos < hi && !seen[t])
        {
          seen[t] = true;
          ++n;
        }
    return n;
  };
  const std::size_t head_score = distinct_in (0, max_bytes);
  const std::size_t window = max_bytes - kClipHeadBytes - 8;
  std::size_t best_start = 0, best_score = head_score;
  for (const auto &[pos, t] : hits)
    {
      (void)t;
      // Start a little before the hit so the sentence around it survives.
      std::size_t lo = pos > 96 ? pos - 96 : 0;
      if (lo < kClipHeadBytes)
        continue;
      while (lo < pos && is_word (text[lo - 1]))
        ++lo;
      const std::size_t score = distinct_in (lo, lo + window);
      if (score > best_score)
        {
          best_score = score;
          best_start = lo;
        }
    }
  if (best_start == 0)
    return ClipTextBlob (text, max_bytes);
  std::vector<unsigned char> head (
      text.begin (), text.begin () + static_cast<std::ptrdiff_t> (kClipHeadBytes));
  auto out = ClipTextBlob (head, kClipHeadBytes - 3);
  out.push_back (' ');
  std::size_t lo = best_start;
  while (lo < text.size () && (text[lo] & 0xC0u) == 0x80u)
    ++lo;
  std::vector<unsigned char> tail (text.begin () + static_cast<std::ptrdiff_t> (lo),
                                   text.end ());
  auto clipped = ClipTextBlob (tail, window);
  out.insert (out.end (), clipped.begin (), clipped.end ());
  return out;
}

// Hydrations for the attachment steps, which read but never write. A memory
// a step hydrates again (a neighbour that did not fit the budget, a
// predecessor that is also a neighbour of another anchor) is served from the
// first hydration; the fallback signal rows that hydration loaded are counted
// again, as a fresh hydration would count them. Payload reads go through one
// shared object-store transaction when one is supplied.
class AttachmentHydrations
{
public:
  AttachmentHydrations (Store *store, ObjectStore *object_store,
                        SharedObjectReads *shared,
                        int max_fallback_signal_blobs,
                        std::size_t *loaded_fallback_signal_blobs,
                        std::vector<std::string> clip_terms = {})
      : store_ (store), object_store_ (object_store), shared_ (shared),
        max_fallback_signal_blobs_ (max_fallback_signal_blobs),
        loaded_fallback_signal_blobs_ (loaded_fallback_signal_blobs),
        clip_terms_ (std::move (clip_terms))
  {
  }

  // Content of a memory to attach to another memory, whatever its
  // modalities: its text blobs joined by spaces into one clipped text/plain
  // blob, then each non-text blob whole. cost charges text by its bytes and
  // each non-text blob as one full clipped neighbour. timestamps holds each
  // blob's stored signal time (the joined text: its first text blob's).
  struct AttachedParts
  {
    std::vector<std::vector<unsigned char> > blobs;
    std::vector<std::string> mimetypes;
    std::vector<std::uint64_t> timestamps;
    std::size_t cost = 0;
  };

  AttachedParts
  AttachedContent (long long memory_id, std::size_t max_text_bytes)
  {
    AttachedParts parts;
    std::uint64_t text_timestamp = 0;
    auto text = AttachedText (memory_id, max_text_bytes, &text_timestamp);
    if (!text.empty ())
      {
        parts.cost += text.size ();
        parts.blobs.push_back (std::move (text));
        parts.mimetypes.push_back ("text/plain");
        parts.timestamps.push_back (text_timestamp);
      }
    const Hydrated &memory = Hydrate (memory_id);
    for (std::size_t i = 0; i < memory.content.size (); ++i)
      {
        const std::string &mime = i < memory.content_mimetypes.size ()
                                      ? memory.content_mimetypes[i]
                                      : memory.mimetype;
        if (memory.content[i].empty () || IsTextMime (mime))
          continue;
        parts.cost += max_text_bytes;
        parts.blobs.push_back (memory.content[i]);
        parts.mimetypes.push_back (mime);
        parts.timestamps.push_back (BlobTimestamp (memory, i));
      }
    return parts;
  }

  // Text of a memory to attach to another memory's content: its content
  // blobs joined by spaces and clipped to max_bytes, or empty when the
  // memory has no text surface. *timestamp, when given, receives the stored
  // signal time of the first text blob joined (0 when there is none).
  std::vector<unsigned char>
  AttachedText (long long memory_id, std::size_t max_bytes,
                std::uint64_t *timestamp = nullptr)
  {
    if (timestamp)
      *timestamp = 0;
    const Hydrated &memory = Hydrate (memory_id);
    if (!memory.has_content)
      {
        return {};
      }
    std::vector<unsigned char> text;
    for (std::size_t i = 0; i < memory.content.size (); ++i)
      {
        const auto &blob = memory.content[i];
        const std::string &mime = i < memory.content_mimetypes.size ()
                                      ? memory.content_mimetypes[i]
                                      : memory.mimetype;
        if (blob.empty () || !IsTextMime (mime))
          {
            continue;
          }
        if (!text.empty ())
          {
            text.push_back (' ');
          }
        else if (timestamp)
          {
            *timestamp = BlobTimestamp (memory, i);
          }
        text.insert (text.end (), blob.begin (), blob.end ());
        if (clip_terms_.empty () && text.size () > max_bytes)
          {
            break;
          }
      }
    if (!clip_terms_.empty ())
      return ClipTextAroundTerms (text, max_bytes, clip_terms_);
    return ClipTextBlob (text, max_bytes);
  }

private:
  struct Hydrated
  {
    std::string modality;
    std::string mimetype;
    std::vector<std::vector<unsigned char> > content;
    std::vector<std::string> content_mimetypes;
    std::vector<std::uint64_t> content_timestamps;
    bool has_content = false;
    std::size_t fallback_signal_blobs = 0;
  };

  static std::uint64_t
  BlobTimestamp (const Hydrated &memory, std::size_t i)
  {
    return i < memory.content_timestamps.size ()
               ? memory.content_timestamps[i]
               : 0;
  }

  const Hydrated &
  Hydrate (long long memory_id)
  {
    auto it = cache_.find (memory_id);
    if (it == cache_.end ())
      {
        Cortext::Context::Memory memory;
        std::size_t fallback_signal_blobs = 0;
        HydrateMemory (store_, object_store_, memory_id, memory,
                       max_fallback_signal_blobs_, &fallback_signal_blobs,
                       shared_);
        Hydrated hydrated;
        hydrated.modality = std::move (memory.modality);
        hydrated.mimetype = std::move (memory.mimetype);
        hydrated.has_content = HasTextContent (memory);
        hydrated.content = std::move (memory.content);
        hydrated.content_mimetypes = std::move (memory.content_mimetypes);
        hydrated.content_timestamps = std::move (memory.content_timestamps);
        hydrated.fallback_signal_blobs = fallback_signal_blobs;
        it = cache_.emplace (memory_id, std::move (hydrated)).first;
      }
    if (loaded_fallback_signal_blobs_)
      {
        *loaded_fallback_signal_blobs_ += it->second.fallback_signal_blobs;
      }
    return it->second;
  }

  Store *store_;
  ObjectStore *object_store_;
  SharedObjectReads *shared_;
  int max_fallback_signal_blobs_;
  std::size_t *loaded_fallback_signal_blobs_;
  std::vector<std::string> clip_terms_;
  std::unordered_map<long long, Hydrated> cache_;
};

// Standing block budget (Context::standing_memory): directive signals are
// short (standing_memory_internal::kMaxTextBytes each), so this holds the
// best few; text is charged by its bytes, other media as
// Cortext::kMediaEvidenceBytes each.
constexpr std::size_t kStandingPacketBytes = 2048;

// Appends a stored directive signal's payload to `memory`'s content and
// returns its charge against kStandingPacketBytes; 0 when the signal or its
// payload is missing.
std::size_t
LoadStandingSignal (Store *store, ObjectStore *object_store,
                    long long signal_id, Cortext::Context::Memory &memory)
{
  if (!store || signal_id <= 0)
    return 0;
  try
    {
      const auto rows = store->Execute (
          "SELECT blob_id, modality, mime, timestamp FROM signals "
          "WHERE signal_id = ?",
          { signal_id });
      if (rows.empty ())
        return 0;
      const auto blob = rows[0].find ("blob_id");
      if (blob == rows[0].end () || !blob->second.has_value ())
        return 0;
      std::vector<unsigned char> payload;
      if (!LoadObjectPayload (store, object_store,
                              store::BlobFromAny (blob->second), payload)
          || payload.empty ())
        return 0;
      std::string mimetype = RowString (rows[0], "mime");
      const std::string modality = RowString (rows[0], "modality");
      if (mimetype.empty ())
        mimetype = "text/plain";
      const bool text = IsTextMime (mimetype)
                        && (modality.empty () || modality == "text");
      const std::size_t cost
          = text ? payload.size () : Cortext::kMediaEvidenceBytes;
      memory.content.push_back (std::move (payload));
      memory.content_mimetypes.push_back (std::move (mimetype));
      memory.content_timestamps.push_back (static_cast<std::uint64_t> (
          std::max (0LL, RowLongLong (rows[0], "timestamp"))));
      return cost;
    }
  catch (const std::exception &e)
    {
      telemetry::LogWarn (
          "Failed to load standing signal",
          { telemetry::Attribute::String ("component", "cortext"),
            telemetry::Attribute::String ("error", e.what ()) });
      return 0;
    }
}

// Evidence budget (Cortext::EvidenceBudget): the bytes of evidence a query
// packet (standing block, retrieved memories and their attached neighbours)
// may hold. Text is charged by its bytes and other media as
// Cortext::kMediaEvidenceBytes each.
std::size_t
EvidenceCharge (const std::string &mime, const std::string &modality,
                std::size_t bytes)
{
  const bool text
      = IsTextMime (mime) && (modality.empty () || modality == "text");
  return text ? bytes : Cortext::kMediaEvidenceBytes;
}

// A packet blob is text by its MIME type, or by the memory's modality when
// it has none.
std::size_t
BlobCharge (const Cortext::Context::Memory &m, std::size_t i)
{
  const std::string &mime = i < m.content_mimetypes.size ()
                                ? m.content_mimetypes[i]
                                : m.mimetype;
  const bool text = mime.empty ()
                        ? m.modality.empty () || m.modality == "text"
                        : mime.rfind ("text/", 0) == 0;
  return text ? m.content[i].size () : Cortext::kMediaEvidenceBytes;
}

// A retrieved memory's own content before neighbours are attached around it:
// its first blob's buffer (moved, never copied, by the attachment steps) and
// its blob count.
struct OwnContent
{
  const unsigned char *first = nullptr;
  std::size_t count = 0;
};

std::vector<OwnContent>
OwnContentOf (const std::vector<Cortext::Context::Memory> &memories)
{
  std::vector<OwnContent> own (memories.size ());
  for (std::size_t i = 0; i < memories.size (); ++i)
    if (!memories[i].content.empty () && !memories[i].content[0].empty ())
      own[i] = { memories[i].content[0].data (), memories[i].content.size () };
  return own;
}

struct EvidenceFit
{
  std::size_t bytes = 0;      // evidence charge after the fit
  std::size_t trimmed = 0;    // grouped memories cut to their best members
  std::size_t stripped = 0;   // memories that dropped attached neighbours
  std::size_t dropped = 0;    // memories left out
  std::size_t member_cap = 0; // member cap the fit settled on
};

// Fits retrieved memories (and their attached neighbours) to `budget` less
// `standing_bytes`. A memory grouped from several signals is focusable when
// each member has a stored shared-space vector (memory_signal_vectors) and a
// payload and its own content is found among its attached neighbours; its
// floor is the best member of each modality by cosine to the query (cosines
// are not comparable across modalities: a text query sits closer to every
// text member than to any image). Memories join in rank order at their
// floor (focusable) or whole, with their neighbours when those fit too,
// else without them, else not at all. One member cap shared by every
// focusable memory then grows to the largest with which the packet fits:
// each keeps its floor, then its other members best first while its
// members fit the cap. Kept members return in their original order
// (consecutive text joined by newlines, as the stored memory text is) in
// place of the memory's own content, between its attached neighbours.
// Cost: query-time only, and only for packets over the budget: one indexed
// signal read for the retrieved memories plus their members' payloads.
EvidenceFit
FitEvidenceToBudget (Store *store, ObjectStore *object_store,
                     SharedObjectReads *shared,
                     std::vector<Cortext::Context::Memory> &memories,
                     const std::vector<OwnContent> &own,
                     std::size_t standing_bytes,
                     const Eigen::VectorXf *query, std::size_t budget)
{
  EvidenceFit fit;
  std::size_t total = standing_bytes;
  for (const auto &m : memories)
    for (std::size_t i = 0; i < m.content.size (); ++i)
      total += BlobCharge (m, i);
  fit.bytes = total;
  if (total <= budget || memories.empty ())
    return fit;

  struct Member
  {
    std::vector<unsigned char> blob_id;
    std::string modality;
    std::string mime;
    std::uint64_t timestamp = 0;
    double score = -2.0;
    bool has_vector = false;
    bool floor = false;
    bool keep = true;
    std::size_t cost = 0;
    std::vector<unsigned char> payload;
  };
  std::unordered_map<long long, std::vector<Member> > members;
  const float query_norm = query ? query->norm () : 0.0f;
  if (store && query && query->size () > 0 && query_norm > 0.0f)
    {
      std::string ids;
      std::vector<std::any> params;
      for (const auto &m : memories)
        {
          if (m.id <= 0)
            continue;
          ids += ids.empty () ? "?" : ",?";
          params.emplace_back (static_cast<long long> (m.id));
        }
      try
        {
          const auto rows = params.empty ()
              ? std::vector<std::map<std::string, std::any> > ()
              : store->Execute (
                    "SELECT s.memory_id AS memory_id, s.blob_id AS blob_id, "
                    "       s.modality AS modality, s.mime AS mime, "
                    "       s.timestamp AS timestamp, "
                    "       v.embedding AS embedding "
                    "FROM signals s "
                    "LEFT JOIN memory_signal_vectors v "
                    "  ON v.blob_id = s.blob_id "
                    "WHERE s.memory_id IN (" + ids + ") "
                    "ORDER BY s.memory_id, s.serial_position, s.signal_id",
                    params);
          const int dim = static_cast<int> (query->size ());
          Eigen::VectorXf vec;
          for (const auto &row : rows)
            {
              Member member;
              if (const auto it = row.find ("blob_id");
                  it != row.end () && it->second.has_value ())
                member.blob_id = store::BlobFromAny (it->second);
              member.modality = RowString (row, "modality");
              member.mime = RowString (row, "mime");
              member.timestamp = static_cast<std::uint64_t> (
                  std::max (0LL, RowLongLong (row, "timestamp")));
              if (const auto it = row.find ("embedding");
                  it != row.end () && it->second.has_value ()
                  && core::DecodeFloatBlob (it->second, dim, vec))
                {
                  const float n = vec.norm ();
                  if (n > 0.0f)
                    {
                      member.score = static_cast<double> (
                          query->dot (vec) / (query_norm * n));
                      member.has_vector = true;
                    }
                }
              members[RowLongLong (row, "memory_id")].push_back (
                  std::move (member));
            }
        }
      catch (const std::exception &e)
        {
          telemetry::LogWarn (
              "Failed to read retrieved memory members",
              { telemetry::Attribute::String ("component", "cortext"),
                telemetry::Attribute::String ("error", e.what ()) });
          members.clear ();
        }
    }

  struct Entry
  {
    std::size_t own_begin = 0;
    std::size_t own_count = 0;
    std::size_t own_charge = 0;
    std::size_t attach_charge = 0;
    std::vector<Member> *list = nullptr; // set when focusable
    std::vector<std::size_t> order;      // floor first, then by cosine
    std::size_t floor_charge = 0;
    std::size_t member_charge = 0;
    bool joined = false;
    bool with_neighbours = false;
  };
  std::vector<Entry> entries (memories.size ());
  for (std::size_t rank = 0; rank < memories.size (); ++rank)
    {
      auto &m = memories[rank];
      auto &e = entries[rank];
      AlignContentMimetypes (m);
      e.own_count = m.content.size ();
      bool located = false;
      if (rank < own.size () && own[rank].first)
        for (std::size_t i = 0; i < m.content.size (); ++i)
          if (m.content[i].data () == own[rank].first
              && i + own[rank].count <= m.content.size ())
            {
              e.own_begin = i;
              e.own_count = own[rank].count;
              located = true;
              break;
            }
      for (std::size_t i = 0; i < m.content.size (); ++i)
        {
          const bool own_blob
              = i >= e.own_begin && i < e.own_begin + e.own_count;
          (own_blob ? e.own_charge : e.attach_charge) += BlobCharge (m, i);
        }
      const auto it = members.find (m.id);
      if (it == members.end () || it->second.size () < 2 || !located)
        continue;
      auto &list = it->second;
      if (std::any_of (list.begin (), list.end (), [] (const Member &x) {
            return !x.has_vector || x.blob_id.empty ();
          }))
        continue;
      bool complete = true;
      for (auto &x : list)
        {
          if (!LoadObjectPayload (store, object_store, x.blob_id, x.payload,
                                  shared))
            {
              complete = false;
              break;
            }
          // Text members are charged one more byte for the newline that
          // joins consecutive kept text.
          x.cost = EvidenceCharge (x.mime, x.modality, x.payload.size ());
          if (x.cost == x.payload.size ())
            ++x.cost;
          e.member_charge += x.cost;
        }
      if (!complete)
        {
          e.member_charge = 0;
          continue;
        }
      std::unordered_map<std::string, std::size_t> best_of;
      for (std::size_t i = 0; i < list.size (); ++i)
        {
          const auto b = best_of.find (list[i].modality);
          if (b == best_of.end () || list[i].score > list[b->second].score)
            best_of[list[i].modality] = i;
        }
      for (const auto &[modality, i] : best_of)
        {
          list[i].floor = true;
          e.floor_charge += list[i].cost;
        }
      e.order.resize (list.size ());
      for (std::size_t i = 0; i < list.size (); ++i)
        e.order[i] = i;
      std::stable_sort (e.order.begin (), e.order.end (),
                        [&] (std::size_t a, std::size_t b) {
                          if (list[a].floor != list[b].floor)
                            return list[a].floor;
                          return list[a].score > list[b].score;
                        });
      e.list = &list;
    }

  // Memories join in rank order at their floor.
  std::size_t used = standing_bytes;
  std::size_t fixed = standing_bytes;
  std::size_t widest = 0;
  for (auto &e : entries)
    {
      const std::size_t least = e.list ? e.floor_charge : e.own_charge;
      if (used + e.attach_charge + least <= budget)
        {
          e.joined = true;
          e.with_neighbours = true;
          used += e.attach_charge + least;
          fixed += e.attach_charge;
        }
      else if (used + least <= budget)
        {
          e.joined = true;
          used += least;
        }
      else
        continue;
      if (e.list)
        widest = std::max (widest, e.member_charge);
      else
        fixed += e.own_charge;
    }
  // Packet bytes with member cap `cap`; marks kept members when `mark`.
  const auto fill = [&] (std::size_t cap, bool mark) {
    std::size_t sum = fixed;
    for (auto &e : entries)
      {
        if (!e.joined || !e.list)
          continue;
        std::size_t kept = 0;
        for (const std::size_t i : e.order)
          {
            auto &x = (*e.list)[i];
            const bool keep = x.floor || kept + x.cost <= cap;
            if (mark)
              x.keep = keep;
            if (keep)
              kept += x.cost;
          }
        sum += kept;
      }
    return sum;
  };
  std::size_t lo = 0;
  std::size_t hi = widest;
  while (lo < hi)
    {
      const std::size_t mid = lo + (hi - lo + 1) / 2;
      if (fill (mid, false) <= budget)
        lo = mid;
      else
        hi = mid - 1;
    }
  fit.member_cap = lo;
  (void)fill (lo, true);

  std::vector<Cortext::Context::Memory> fitted;
  fitted.reserve (memories.size ());
  for (std::size_t rank = 0; rank < memories.size (); ++rank)
    {
      auto &m = memories[rank];
      auto &e = entries[rank];
      if (!e.joined)
        {
          ++fit.dropped;
          continue;
        }
      const bool trim
          = e.list
            && std::any_of (e.list->begin (), e.list->end (),
                            [] (const Member &x) { return !x.keep; });
      if (!trim && e.with_neighbours)
        {
          fitted.push_back (std::move (m));
          continue;
        }
      std::vector<std::vector<unsigned char> > content;
      std::vector<std::string> mimetypes;
      std::vector<std::uint64_t> timestamps;
      const auto take = [&] (std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i)
          {
            content.push_back (std::move (m.content[i]));
            mimetypes.push_back (std::move (m.content_mimetypes[i]));
            timestamps.push_back (m.content_timestamps[i]);
          }
      };
      if (e.with_neighbours)
        take (0, e.own_begin);
      if (trim)
        {
          bool last_text = false;
          for (auto &x : *e.list)
            {
              if (!x.keep)
                continue;
              const bool text = IsTextMime (x.mime)
                                && (x.modality.empty ()
                                    || x.modality == "text");
              if (text && last_text)
                {
                  content.back ().push_back ('\n');
                  content.back ().insert (content.back ().end (),
                                          x.payload.begin (),
                                          x.payload.end ());
                  continue;
                }
              content.push_back (std::move (x.payload));
              mimetypes.push_back (text ? std::string ("text/plain")
                                        : x.mime);
              // Consecutive kept text joins this blob: it keeps the time of
              // the member it starts with.
              timestamps.push_back (x.timestamp);
              last_text = text;
            }
          ++fit.trimmed;
        }
      else
        take (e.own_begin, e.own_begin + e.own_count);
      if (e.with_neighbours)
        take (e.own_begin + e.own_count, m.content.size ());
      else if (e.attach_charge > 0)
        ++fit.stripped;
      m.content = std::move (content);
      m.content_mimetypes = std::move (mimetypes);
      m.content_timestamps = std::move (timestamps);
      fitted.push_back (std::move (m));
    }
  // The charges above bound the fitted content; this keeps the bound if a
  // stored memory's own content is larger than its members.
  fit.bytes = standing_bytes;
  for (const auto &m : fitted)
    for (std::size_t i = 0; i < m.content.size (); ++i)
      fit.bytes += BlobCharge (m, i);
  while (fit.bytes > budget && !fitted.empty ())
    {
      for (std::size_t i = 0; i < fitted.back ().content.size (); ++i)
        fit.bytes -= BlobCharge (fitted.back (), i);
      fitted.pop_back ();
      ++fit.dropped;
    }
  memories = std::move (fitted);
  return fit;
}

// A step from a turn to the adjacent LONG_TERM turn in an anchor's stream.
struct ContiguityProbe
{
  long long position_id = 0;
  long long anchor_start_ts = 0;
  std::string anchor_stream;
  bool forward = false;
};

// The adjacent turn a probe found.
struct ContiguousTurn
{
  // The adjacent turn when it belongs to the same stretch of conversation
  // (same stream, inside the contiguity gap), else 0. A walk continues from
  // it even when it is not attachable.
  long long position_id = 0;
  // Set to position_id when the turn is a text turn not hidden by a
  // correction from the same stretch of conversation.
  long long memory_id = 0;
};

// Resolves every probe in one statement: one UNION ALL arm per probe, each
// a primary-key range lookup bounded by a fixed id span plus one indexed
// supersession check. Forward arms come first so the statement text depends
// only on how many probes go each way. Supersessions from other
// conversations are topical successors and do not hide the turn.
std::vector<ContiguousTurn>
FindContiguousTurns (Store *store, const std::vector<ContiguityProbe> &probes,
                     std::optional<long long> as_of_ts)
{
  std::vector<ContiguousTurn> turns (probes.size ());
  if (!store || probes.empty ())
    {
      return turns;
    }
  std::string sql;
  std::vector<std::any> params;
  params.reserve (probes.size () * 9);
  for (const bool forward : { true, false })
    {
      for (std::size_t k = 0; k < probes.size (); ++k)
        {
          const auto &probe = probes[k];
          if (probe.forward != forward)
            {
              continue;
            }
          if (!sql.empty ())
            {
              sql += " UNION ALL ";
            }
          sql += "SELECT ? AS probe, n.memory_id, n.start_ts, n.source_id, "
                 "  n.modality, "
                 "  EXISTS (SELECT 1 FROM associations a "
                 "          JOIN memories c ON c.memory_id = a.source_memory_id "
                 "          WHERE a.target_memory_id = n.memory_id "
                 "            AND a.edge_type = 'supersedes' "
                 "            AND c.start_ts BETWEEN n.start_ts - ? "
                 "                               AND n.start_ts + ? "
                 "            AND ((c.source_id >= ? AND c.source_id < ? "
                 "                  AND instr(substr(c.source_id, ?), '/') = 0) "
                 "                 OR (c.source_id = ? "
                 "                     AND instr(c.source_id, '/') = 0))) "
                 "    AS corrected "
                 "FROM (SELECT memory_id, start_ts, source_id, modality "
                 "      FROM memories WHERE ";
          sql += forward ? "memory_id > ? AND memory_id <= ? "
                           "AND kind = 'LONG_TERM' "
                           "ORDER BY memory_id ASC LIMIT 1) n"
                         : "memory_id < ? AND memory_id >= ? "
                           "AND kind = 'LONG_TERM' "
                           "ORDER BY memory_id DESC LIMIT 1) n";
          const std::string stream_prefix = probe.anchor_stream + "/";
          const std::string stream_end
              = probe.anchor_stream + "0"; // '0' follows '/'
          const long long bound
              = forward ? probe.position_id + kContiguityIdScanSpan
                        : probe.position_id - kContiguityIdScanSpan;
          params.emplace_back (static_cast<long long> (k));
          params.emplace_back (kContiguityMaxGapMs);
          params.emplace_back (kContiguityMaxGapMs);
          params.emplace_back (stream_prefix);
          params.emplace_back (stream_end);
          params.emplace_back (
              static_cast<long long> (stream_prefix.size () + 1));
          params.emplace_back (probe.anchor_stream);
          params.emplace_back (probe.position_id);
          params.emplace_back (bound);
        }
    }
  const auto rows = store->Execute (sql, params);
  for (const auto &row : rows)
    {
      const long long k = RowLongLong (row, "probe");
      if (k < 0 || static_cast<std::size_t> (k) >= probes.size ())
        {
          continue;
        }
      const auto &probe = probes[static_cast<std::size_t> (k)];
      const long long neighbor_id = RowLongLong (row, "memory_id");
      if (neighbor_id <= 0
          || SourceStreamOf (RowString (row, "source_id"))
                 != probe.anchor_stream)
        {
          continue;
        }
      const long long gap
          = RowLongLong (row, "start_ts") - probe.anchor_start_ts;
      if (gap > kContiguityMaxGapMs || gap < -kContiguityMaxGapMs)
        {
          continue;
        }
      // An as-of query sees no turn stored after its reference time; the
      // stretch ends there.
      if (as_of_ts && RowLongLong (row, "start_ts") > *as_of_ts)
        {
          continue;
        }
      auto &turn = turns[static_cast<std::size_t> (k)];
      turn.position_id = neighbor_id;
      if (RowLongLong (row, "corrected") == 0)
        {
          turn.memory_id = neighbor_id;
        }
    }
  return turns;
}

// Kind and start time of the first `limit` retrieved memories, in one
// indexed primary-key lookup.
struct AnchorRow
{
  std::string kind;
  long long start_ts = 0;
};

std::unordered_map<long long, AnchorRow>
LoadAnchorRows (Store *store,
                const std::vector<Cortext::Context::Memory> &retrieved,
                std::size_t limit)
{
  std::unordered_map<long long, AnchorRow> out;
  std::vector<std::any> params;
  std::string placeholders;
  for (std::size_t i = 0; i < std::min (limit, retrieved.size ()); ++i)
    {
      if (retrieved[i].id <= 0)
        continue;
      if (!params.empty ())
        placeholders += ", ";
      placeholders += "?";
      params.emplace_back (retrieved[i].id);
    }
  if (!store || params.empty ())
    return out;
  try
    {
      const auto rows = store->Execute (
          "SELECT memory_id, kind, start_ts FROM memories "
          "WHERE memory_id IN (" + placeholders + ")",
          params);
      for (const auto &row : rows)
        out[RowLongLong (row, "memory_id")]
            = { RowString (row, "kind"), RowLongLong (row, "start_ts") };
    }
  catch (const std::exception &e)
    {
      out.clear ();
      telemetry::LogWarn (
          "Failed to load retrieved anchor rows",
          { telemetry::Attribute::String ("component", "cortext"),
            telemetry::Attribute::String ("error", e.what ()) });
    }
  return out;
}

// Attaches adjacent-turn context to retrieved text turns: lag 1 for every
// anchor in rank order, then lag 2. The adjacent turns are found up front,
// all lag-1 steps in one statement and all lag-2 steps in another; which of
// them are attached is then decided in the same order as a turn-by-turn walk.
// Returns the number of neighbor blobs attached; their memory ids are added
// to attached_ids.
std::size_t
AttachContiguousTurns (Store *store,
                       std::vector<Cortext::Context::Memory> &retrieved,
                       const std::unordered_map<long long, AnchorRow>
                           &anchor_rows,
                       AttachmentHydrations &hydrations,
                       std::unordered_set<long long> &attached_ids,
                       std::optional<long long> as_of_ts)
{
  if (!store || retrieved.empty () || SourceBlobsDisabled ())
    {
      return 0;
    }
  std::unordered_set<long long> retrieved_ids;
  retrieved_ids.reserve (retrieved.size ());
  for (const auto &memory : retrieved)
    {
      retrieved_ids.insert (memory.id);
    }
  std::unordered_set<long long> excluded_ids = retrieved_ids;
  std::size_t budget = kContiguityContextBudgetBytes;
  std::size_t attached = 0;
  const std::size_t anchors
      = std::min (kContiguityAnchorLimit, retrieved.size ());
  struct AnchorWalk
  {
    std::size_t index = 0;
    long long start_ts = 0;
    std::string stream;
    ContiguousTurn successor;
    ContiguousTurn predecessor;
    ContiguousTurn successor_lag2;
    ContiguousTurn predecessor_lag2;
  };
  // Attaches the adjacent turn to the anchor when it is still unattached
  // and fits the budget; returns the walked position (0 when the stretch
  // ends).
  auto attach_adjacent = [&] (const AnchorWalk &walk,
                              const ContiguousTurn &turn,
                              bool forward) -> long long {
    if (turn.memory_id <= 0 || excluded_ids.count (turn.memory_id) != 0)
      {
        return turn.position_id;
      }
    auto parts = hydrations.AttachedContent (turn.memory_id,
                                             kContiguityNeighborMaxBytes);
    if (!parts.blobs.empty () && parts.cost <= budget)
      {
        auto &anchor = retrieved[walk.index];
        budget -= parts.cost;
        AlignContentMimetypes (anchor);
        const auto at = forward ? anchor.content.size () : 0;
        anchor.content.insert (
            anchor.content.begin () + static_cast<std::ptrdiff_t> (at),
            std::make_move_iterator (parts.blobs.begin ()),
            std::make_move_iterator (parts.blobs.end ()));
        anchor.content_mimetypes.insert (
            anchor.content_mimetypes.begin () + static_cast<std::ptrdiff_t> (at),
            parts.mimetypes.begin (), parts.mimetypes.end ());
        anchor.content_timestamps.insert (
            anchor.content_timestamps.begin () + static_cast<std::ptrdiff_t> (at),
            parts.timestamps.begin (), parts.timestamps.end ());
        excluded_ids.insert (turn.memory_id);
        attached_ids.insert (turn.memory_id);
        ++attached;
      }
    return turn.position_id;
  };
  try
    {
      std::vector<AnchorWalk> walks;
      walks.reserve (anchors);
      for (std::size_t i = 0; i < anchors; ++i)
        {
          const auto &anchor = retrieved[i];
          if (anchor.id <= 0 || !HasHydratedContent (anchor))
            {
              continue;
            }
          const auto anchor_row = anchor_rows.find (anchor.id);
          if (anchor_row == anchor_rows.end ()
              || anchor_row->second.kind != "LONG_TERM")
            {
              continue;
            }
          AnchorWalk walk;
          walk.index = i;
          walk.start_ts = anchor_row->second.start_ts;
          walk.stream = SourceStreamOf (anchor.source_id);
          walks.push_back (std::move (walk));
        }
      if (walks.empty ())
        {
          return 0;
        }
      // Lag-1 steps of every anchor, both ways.
      std::vector<ContiguityProbe> probes;
      probes.reserve (walks.size () * 2);
      for (const auto &walk : walks)
        {
          const long long anchor_id = retrieved[walk.index].id;
          probes.push_back ({ anchor_id, walk.start_ts, walk.stream, true });
          probes.push_back ({ anchor_id, walk.start_ts, walk.stream, false });
        }
      auto turns = FindContiguousTurns (store, probes, as_of_ts);
      for (std::size_t w = 0; w < walks.size (); ++w)
        {
          walks[w].successor = turns[2 * w];
          walks[w].predecessor = turns[2 * w + 1];
        }
      // Lag-2 steps continue from the lag-1 turns of the top anchors.
      probes.clear ();
      std::vector<ContiguousTurn *> lag2_targets;
      for (auto &walk : walks)
        {
          if (walk.successor.position_id > 0)
            {
              probes.push_back ({ walk.successor.position_id, walk.start_ts,
                                  walk.stream, true });
              lag2_targets.push_back (&walk.successor_lag2);
            }
          if (walk.predecessor.position_id > 0)
            {
              probes.push_back ({ walk.predecessor.position_id,
                                  walk.start_ts, walk.stream, false });
              lag2_targets.push_back (&walk.predecessor_lag2);
            }
        }
      if (!probes.empty ())
        {
          turns = FindContiguousTurns (store, probes, as_of_ts);
          for (std::size_t k = 0; k < lag2_targets.size (); ++k)
            {
              *lag2_targets[k] = turns[k];
            }
        }
      // Lag 1, in rank order. Forward association is stronger than
      // backward: the successor (the reply) is attached first.
      std::size_t walked = 0;
      for (auto &walk : walks)
        {
          if (budget == 0)
            break;
          attach_adjacent (walk, walk.successor, true);
          if (budget > 0)
            attach_adjacent (walk, walk.predecessor, false);
          else
            walk.predecessor = {};
          ++walked;
        }
      // Lag 2 with what is left of the budget: contiguity decays with
      // distance, so the second turn out comes after every first one.
      for (std::size_t w = 0; w < walked; ++w)
        {
          const auto &walk = walks[w];
          if (budget == 0)
            break;
          if (walk.successor.position_id > 0)
            attach_adjacent (walk, walk.successor_lag2, true);
          if (budget > 0 && walk.predecessor.position_id > 0)
            attach_adjacent (walk, walk.predecessor_lag2, false);
        }
    }
  catch (const std::exception &e)
    {
      telemetry::LogWarn (
          "Failed to attach contiguous turns",
          { telemetry::Attribute::String ("component", "cortext"),
            telemetry::Attribute::String ("error", e.what ()) });
    }
  return attached;
}

// Recall of superseded predecessors for retrieved memories.
//
// At write time every new memory gets 'supersedes' edges to the few older
// memories it is most similar to (weight grows with similarity), and
// retrieval treats any memory with an active incoming 'supersedes' edge as
// replaced: it is never a seed again. Those edges are topical, not only
// corrections, so a large share of older statements (an earlier course
// count, a previous status, the session where an event was first mentioned)
// becomes unreachable as soon as a similar later turn arrives. People do not
// lose the earlier version of a fact when they learn the newer one: recalling
// the update brings back the earlier episode it updated. Within one stretch
// of conversation (same source stream, inside the contiguity gap) a newer
// statement is a correction that replaces the stale one outright, so those
// predecessors stay suppressed. For each of the top retrieved
// memories, the older memories it supersedes (strongest edges first) are
// hydrated as extra content blobs of that same retrieved memory, placed
// before its own content so the packet reads oldest to newest, clipped and
// under a per-context byte budget. Cost is O(anchors): one indexed lookup on
// idx_associations_source for all anchors plus a bounded number of
// hydrations.
// Ranking, the retrieval pool and the number of retrieved memories are
// unchanged.
constexpr std::size_t kSupersededRecallAnchorLimit = 11;
constexpr long long kSupersededRecallPerAnchor = 2;
constexpr std::size_t kSupersededRecallMaxBytes = 512;
constexpr std::size_t kSupersededRecallContextBudgetBytes = 4096;

// A superseded target of an anchor, strongest edge first.
struct SupersededTarget
{
  long long memory_id = 0;
};

// The `per_anchor` strongest LONG_TERM text targets of the 'supersedes'
// edges out of each anchor (by weight, then newest target), in one statement
// over idx_associations_source.
std::unordered_map<long long, std::vector<SupersededTarget> >
LoadSupersededTargets (Store *store, const std::vector<long long> &anchor_ids,
                       long long per_anchor, std::optional<long long> as_of_ts)
{
  std::unordered_map<long long, std::vector<SupersededTarget> > out;
  if (!store || anchor_ids.empty ())
    {
      return out;
    }
  std::string placeholders;
  std::vector<std::any> params;
  params.reserve (anchor_ids.size () + 1);
  for (const long long id : anchor_ids)
    {
      if (!placeholders.empty ())
        placeholders += ", ";
      placeholders += "?";
      params.emplace_back (id);
    }
  // An as-of query sees no memory stored after its reference time.
  if (as_of_ts)
    params.emplace_back (*as_of_ts);
  params.emplace_back (per_anchor);
  const auto rows = store->Execute (
      "SELECT r.anchor_id, r.memory_id "
      "FROM (SELECT a.source_memory_id AS anchor_id, "
      "             a.target_memory_id AS memory_id, "
      "             ROW_NUMBER() OVER ("
      "               PARTITION BY a.source_memory_id "
      "               ORDER BY a.weight DESC, a.target_memory_id DESC) "
      "               AS edge_rank "
      "      FROM associations a "
      "      JOIN memories m ON m.memory_id = a.target_memory_id "
      "      WHERE a.source_memory_id IN (" + placeholders + ") "
      "        AND a.edge_type = 'supersedes' "
      "        AND m.kind = 'LONG_TERM' "
      "        AND m.modality IN ('text', 'mixed')"
      + std::string (as_of_ts ? " AND m.start_ts <= ?" : "") + ") r "
      "WHERE r.edge_rank <= ? "
      "ORDER BY r.anchor_id, r.edge_rank",
      params);
  for (const auto &row : rows)
    {
      out[RowLongLong (row, "anchor_id")].push_back (
          { RowLongLong (row, "memory_id") });
    }
  return out;
}

// Returns the number of superseded-predecessor blobs attached.
std::size_t
AttachSupersededPredecessors (
    Store *store, std::vector<Cortext::Context::Memory> &retrieved,
    const std::unordered_map<long long, AnchorRow> &anchor_rows,
    std::unordered_set<long long> excluded_ids,
    AttachmentHydrations &hydrations, std::optional<long long> as_of_ts)
{
  if (!store || retrieved.empty () || SourceBlobsDisabled ())
    {
      return 0;
    }
  for (const auto &memory : retrieved)
    {
      excluded_ids.insert (memory.id);
    }
  std::size_t budget = kSupersededRecallContextBudgetBytes;
  std::size_t attached = 0;
  const std::size_t anchors
      = std::min (kSupersededRecallAnchorLimit, retrieved.size ());
  try
    {
      // Which anchors recall predecessors does not depend on what earlier
      // anchors attached (an anchor's content only grows), so all their
      // edges are read at once.
      std::vector<std::size_t> anchor_indexes;
      std::vector<long long> anchor_ids;
      std::unordered_set<long long> seen_anchor_ids;
      for (std::size_t i = 0; i < anchors; ++i)
        {
          const auto &anchor = retrieved[i];
          if (anchor.id <= 0
              || !HasTextContent (anchor)
              || !HasHydratedContent (anchor)
              || anchor_rows.find (anchor.id) == anchor_rows.end ())
            {
              continue;
            }
          anchor_indexes.push_back (i);
          if (seen_anchor_ids.insert (anchor.id).second)
            {
              anchor_ids.push_back (anchor.id);
            }
        }
      // Over-fetch so already-packeted targets do not starve the anchor.
      const auto targets = LoadSupersededTargets (
          store, anchor_ids, kSupersededRecallPerAnchor * 2, as_of_ts);
      for (const std::size_t i : anchor_indexes)
        {
          if (budget == 0)
            {
              break;
            }
          auto &anchor = retrieved[i];
          const auto anchor_targets = targets.find (anchor.id);
          if (anchor_targets == targets.end ())
            {
              continue;
            }
          std::vector<std::vector<unsigned char> > recalled;
          std::vector<std::uint64_t> recalled_timestamps;
          for (const auto &target : anchor_targets->second)
            {
              if (static_cast<long long> (recalled.size ())
                  >= kSupersededRecallPerAnchor)
                {
                  break;
                }
              const long long target_id = target.memory_id;
              if (target_id <= 0 || !excluded_ids.insert (target_id).second)
                {
                  continue;
                }
              std::uint64_t text_timestamp = 0;
              auto text = hydrations.AttachedText (
                  target_id, kSupersededRecallMaxBytes, &text_timestamp);
              if (text.empty () || text.size () > budget)
                {
                  continue;
                }
              budget -= text.size ();
              recalled.push_back (std::move (text));
              recalled_timestamps.push_back (text_timestamp);
            }
          // Strongest edge first in rank, but oldest-to-newest in the packet:
          // the recalled predecessors precede the anchor's own content.
          AlignContentMimetypes (anchor);
          anchor.content_mimetypes.insert (anchor.content_mimetypes.begin (),
                                           recalled.size (), "text/plain");
          anchor.content.insert (
              anchor.content.begin (),
              std::make_move_iterator (recalled.rbegin ()),
              std::make_move_iterator (recalled.rend ()));
          anchor.content_timestamps.insert (anchor.content_timestamps.begin (),
                                            recalled_timestamps.rbegin (),
                                            recalled_timestamps.rend ());
          attached += recalled.size ();
        }
    }
  catch (const std::exception &e)
    {
      telemetry::LogWarn (
          "Failed to attach superseded predecessors",
          { telemetry::Attribute::String ("component", "cortext"),
            telemetry::Attribute::String ("error", e.what ()) });
    }
  return attached;
}

// Knowledge updates: a retrieved memory may be an older version of a fact
// that a later memory restates ('supersedes' edge from the later memory). An
// answer needs the latest version, which often ranks below the older one
// (the question is phrased like the first statement). For each of the top
// retrieved memories, its newest superseding memory is hydrated as an extra
// content blob placed after the anchor's own content, so the packet reads
// oldest to newest; clipped and under a per-context byte budget. One indexed
// lookup on idx_associations_target for all anchors.
constexpr std::size_t kSupersedingRecallAnchorLimit = 11;
constexpr std::size_t kSupersedingRecallMaxBytes = 512;
constexpr std::size_t kSupersedingRecallContextBudgetBytes = 4096;

// Returns the number of superseding-successor blobs attached.
std::size_t
AttachSupersedingSuccessors (Store *store,
                             std::vector<Cortext::Context::Memory> &retrieved,
                             const std::unordered_map<long long, AnchorRow> &anchor_rows,
                             std::unordered_set<long long> excluded_ids,
                             AttachmentHydrations &hydrations,
                             std::optional<long long> as_of_ts)
{
  if (!store || retrieved.empty () || SourceBlobsDisabled ())
    return 0;
  for (const auto &memory : retrieved)
    excluded_ids.insert (memory.id);
  std::size_t budget = kSupersedingRecallContextBudgetBytes;
  std::size_t attached = 0;
  const std::size_t anchors
      = std::min (kSupersedingRecallAnchorLimit, retrieved.size ());
  try
    {
      std::vector<std::size_t> anchor_indexes;
      std::string placeholders;
      std::vector<std::any> params;
      std::unordered_set<long long> seen_anchor_ids;
      for (std::size_t i = 0; i < anchors; ++i)
        {
          const auto &anchor = retrieved[i];
          if (anchor.id <= 0 || !HasTextContent (anchor)
              || !HasHydratedContent (anchor)
              || anchor_rows.find (anchor.id) == anchor_rows.end ())
            continue;
          anchor_indexes.push_back (i);
          if (seen_anchor_ids.insert (anchor.id).second)
            {
              if (!placeholders.empty ())
                placeholders += ", ";
              placeholders += "?";
              params.emplace_back (anchor.id);
            }
        }
      if (params.empty ())
        return 0;
      // An as-of query sees no successor stored after its reference time.
      if (as_of_ts)
        params.emplace_back (*as_of_ts);
      // Newest first per anchor; two per anchor so an already-packeted
      // successor does not starve it.
      const auto rows = store->Execute (
          "SELECT r.anchor_id, r.memory_id "
          "FROM (SELECT a.target_memory_id AS anchor_id, "
          "             a.source_memory_id AS memory_id, "
          "             ROW_NUMBER() OVER ("
          "               PARTITION BY a.target_memory_id "
          "               ORDER BY m.start_ts DESC, a.source_memory_id DESC) "
          "               AS edge_rank "
          "      FROM associations a "
          "      JOIN memories m ON m.memory_id = a.source_memory_id "
          "      WHERE a.target_memory_id IN (" + placeholders + ") "
          "        AND a.edge_type = 'supersedes' "
          "        AND m.kind = 'LONG_TERM' "
          "        AND m.modality IN ('text', 'mixed')"
          + std::string (as_of_ts ? " AND m.start_ts <= ?" : "") + ") r "
          "WHERE r.edge_rank <= 2 "
          "ORDER BY r.anchor_id, r.edge_rank",
          params);
      std::unordered_map<long long, std::vector<long long> > successors;
      for (const auto &row : rows)
        successors[RowLongLong (row, "anchor_id")].push_back (
            RowLongLong (row, "memory_id"));
      for (const std::size_t i : anchor_indexes)
        {
          if (budget == 0)
            break;
          auto &anchor = retrieved[i];
          const auto it = successors.find (anchor.id);
          if (it == successors.end ())
            continue;
          for (const long long successor_id : it->second)
            {
              if (successor_id <= 0 || !excluded_ids.insert (successor_id).second)
                continue;
              std::uint64_t text_timestamp = 0;
              auto text = hydrations.AttachedText (
                  successor_id, kSupersedingRecallMaxBytes, &text_timestamp);
              if (text.empty () || text.size () > budget)
                continue;
              budget -= text.size ();
              AlignContentMimetypes (anchor);
              anchor.content_mimetypes.push_back ("text/plain");
              anchor.content_timestamps.push_back (text_timestamp);
              anchor.content.push_back (std::move (text));
              ++attached;
              break;
            }
        }
    }
  catch (const std::exception &e)
    {
      telemetry::LogWarn (
          "Failed to attach superseding successors",
          { telemetry::Attribute::String ("component", "cortext"),
            telemetry::Attribute::String ("error", e.what ()) });
    }
  return attached;
}

void
HydrateWorkingMemoryFromDB (Store *store, ObjectStore *object_store,
                            std::vector<Cortext::Context::Memory> &out,
                            int max_fallback_signal_blobs,
                            std::size_t *loaded_fallback_signal_blobs)
{
  if (!store)
    return;

  try
    {
      // Query active WM slots from MEMORIES table with kind='WORKING'
      // Content blobs are loaded separately via LoadSignalBlobs
      auto rows = store->Execute (
          "SELECT m.memory_id, m.source_id, m.modality, m.start_ts, "
          "       m.strength, m.last_access, m.n_signals, "
          "       m.s_max, m.s_avg, m.s_arousal_avg, "
          "       m.retrieved_count, m.used_count, m.blob_id, "
          "       (SELECT mr.blob_id FROM memory_reconstructions mr "
          "        WHERE mr.memory_id = m.memory_id "
          "        ORDER BY mr.reconstruction_id DESC LIMIT 1) "
          "          AS reconstruction_blob_id, "
          "       COALESCE("
          "         (SELECT s.mime FROM signals s WHERE s.memory_id = m.memory_id "
          "            AND s.modality = m.modality "
          "          ORDER BY s.serial_position LIMIT 1), "
          "         (SELECT s.mime FROM signals s WHERE s.memory_id = m.memory_id "
          "          ORDER BY s.serial_position LIMIT 1)"
          "       ) AS signal_mime, "
          "       COALESCE("
          "         (SELECT s.timestamp FROM signals s WHERE s.memory_id = m.memory_id "
          "            AND s.modality = m.modality "
          "          ORDER BY s.serial_position LIMIT 1), "
          "         (SELECT s.timestamp FROM signals s WHERE s.memory_id = m.memory_id "
          "          ORDER BY s.serial_position LIMIT 1)"
          "       ) AS signal_ts "
          "FROM memories m "
          "WHERE m.kind = 'WORKING' AND m.end_ts IS NULL "
          "ORDER BY m.start_ts ASC, m.memory_id ASC");

      for (const auto &row : rows)
        {
          Cortext::Context::Memory m;

          // Helper lambdas (same pattern as HydrateMemory)
          auto get_s = [&row] (const char *k) -> std::string {
            auto it = row.find (k);
            if (it == row.end () || !it->second.has_value ())
              return {};
            if (it->second.type () == typeid (std::string))
              return std::any_cast<std::string> (it->second);
            return {};
          };

          auto get_ll = [&row] (const char *k) -> long long {
            auto it = row.find (k);
            if (it == row.end () || !it->second.has_value ())
              return 0LL;
            if (it->second.type () == typeid (long long))
              return std::any_cast<long long> (it->second);
            if (it->second.type () == typeid (int))
              return static_cast<long long> (std::any_cast<int> (it->second));
            return 0LL;
          };

          auto get_dbl = [&row] (const char *k, double def = 0.0) -> double {
            auto it = row.find (k);
            if (it == row.end () || !it->second.has_value ())
              return def;
            return store::AnyToDouble (it->second, def);
          };

          auto get_blob = [&row] (const char *k) {
            auto it = row.find (k);
            if (it == row.end () || !it->second.has_value ())
              return std::vector<unsigned char> ();
            return store::BlobFromAny (it->second);
          };

          // Populate Memory struct from WM slot row
          m.id = get_ll ("memory_id");
          m.source_id = get_s ("source_id");
          m.modality = get_s ("modality");
          m.mimetype = get_s ("signal_mime");
          // Preserve conversational/order semantics using the slot start time.
          // last_access can move when a slot is touched,
          // which should not reorder working-memory prompt reconstruction.
          m.timestamp = static_cast<std::uint64_t> (get_ll ("start_ts"));

          // Map WM metrics to Memory struct
          m.composite_score = get_dbl ("s_avg", 0.0);
          m.salience = get_dbl ("s_max", 0.0);
          m.arousal = get_dbl ("s_arousal_avg", 0.0);
          m.retrieved_count = get_ll ("retrieved_count");
          m.used_count = get_ll ("used_count");

          // A memory-level blob carries the time of the signal it starts
          // with, the one signal_mime comes from.
          const auto memory_blob_ts = static_cast<std::uint64_t> (
              std::max (0LL, get_ll ("signal_ts")));
          const auto reconstruction_blob_id
              = get_blob ("reconstruction_blob_id");
          if (!LoadContentBlob (store, object_store, reconstruction_blob_id,
                                m.modality, m.mimetype, memory_blob_ts,
                                m.content, m.content_mimetypes,
                                m.content_timestamps))
            {
              LoadContentBlob (store, object_store, get_blob ("blob_id"),
                               m.modality, m.mimetype, memory_blob_ts,
                               m.content, m.content_mimetypes,
                               m.content_timestamps);
            }
          if (m.content.empty ())
            {
              // v2 fallback: old databases may only have signal-level blobs.
              LoadSignalBlobs (store, m.id, object_store, m.modality,
                               m.mimetype, max_fallback_signal_blobs,
                               loaded_fallback_signal_blobs,
                               m.content, m.content_mimetypes,
                               m.content_timestamps);
            }
          StripSourceBlobsIfDisabled (m);

          out.push_back (std::move (m));
        }
    }
  catch (const std::exception &e)
    {
      telemetry::LogWarn (
          "Failed to hydrate working memory",
          { telemetry::Attribute::String ("component", "cortext"),
            telemetry::Attribute::String ("error", e.what ()) });
    }
  catch (...)
    {
      telemetry::LogWarn (
          "Failed to hydrate working memory (unknown error)",
          { telemetry::Attribute::String ("component", "cortext") });
    }
}

} // namespace

namespace
{

/// Maintenance signals (`force_consolidation`, sent by Consolidate()) run no
/// per-signal operations; the processor seals and recenters the sparse
/// retrieval route around them.
class SignalOperationDispatch final : public IOperation
{
public:
  explicit SignalOperationDispatch (
      std::unique_ptr<IOperation> signal_operation)
      : signal_operation_ (std::move (signal_operation))
  {
  }

  void
  Execute (OperationContext &context, Transaction &tx) const override
  {
    if (context.GetSignal ().force_consolidation)
      return;
    signal_operation_->Execute (context, tx);
  }

private:
  std::unique_ptr<IOperation> signal_operation_;
};

} // namespace

std::unique_ptr<IOperation>
BuildRootOperationSet (bool probe_mode)
{
  using cortext::operations::CheckSpikeBypass;
  using cortext::operations::CheckStreamingPacing;
  using cortext::operations::ComputeCoherence;
  using cortext::operations::ComputeCompositeScore;
  using cortext::operations::ComputeEffectiveFocus;
  using cortext::operations::ComputeFocusSpread;
  using cortext::operations::ComputeMetrics;
  using cortext::operations::ComputeMniGateDecision;
  using cortext::operations::ComputeWriteGate;
  using cortext::operations::DetectBoundary;
  using cortext::operations::DetectMemoryUsage;
  using cortext::operations::FitMetricWeightsRLS;
  using cortext::operations::GraphAugmentedRetrieveCandidates;
  using cortext::operations::InitializeEmbeddedCentroids;
  using cortext::operations::InitializeFocusPriors;
  using cortext::operations::InitializeSensitivityPriors;
  using cortext::operations::InitializeStabilityPriors;
  using cortext::operations::MemoryStorage;
  using cortext::operations::PersistSignalMetrics;
  using cortext::operations::ResetAccumulatorAfterFlush;
  using cortext::operations::ResetAccumulatorOnInterrupt;
  using cortext::operations::UpdateAccumulator;
  using cortext::operations::UpdateAccumulatorScores;
  using cortext::operations::UpdateDriftAccumulation;
  using cortext::operations::UpdateEmbeddingPredictionError;
  using cortext::operations::UpdateFocus;
  using cortext::operations::RecordMemoryUsage;
  using cortext::operations::UpdateMood;
  using cortext::operations::UpdateNeuromodulators;
  using cortext::operations::UpdatePrecisionDelta;
  using cortext::operations::UpdateRateState;
  using cortext::operations::UpdateRecentContext;
  using cortext::operations::UpdateSensitivity;
  using cortext::operations::UpdateThreshold;
  using cortext::operations::UpdateUncertainty;
  using cortext::operations::WorkingMemory;

  using cortext::OperationSet;

  using CoreStage = OperationSet<
      InitializeEmbeddedCentroids,

      InitializeFocusPriors, InitializeSensitivityPriors,
      InitializeStabilityPriors,

      ComputeCoherence, UpdateAccumulator, UpdateDriftAccumulation,
      ComputeFocusSpread, UpdateEmbeddingPredictionError, UpdateUncertainty,

      UpdateFocus, UpdateSensitivity, UpdateMood,

      ComputeEffectiveFocus, ComputeMetrics, UpdateNeuromodulators,
      FitMetricWeightsRLS, ComputeCompositeScore, UpdateAccumulatorScores,

      UpdatePrecisionDelta, UpdateThreshold, UpdateRecentContext,

      DetectBoundary, CheckSpikeBypass, ComputeWriteGate>;

  using StorageStage = OperationSet<MemoryStorage, PersistSignalMetrics>;

  using RetrievalStage
      = OperationSet<CheckStreamingPacing, GraphAugmentedRetrieveCandidates,
                     UpdateRateState, ComputeMniGateDecision,
                     DetectMemoryUsage>;

  using FeedbackStage = OperationSet<
      RecordMemoryUsage, WorkingMemory,
      ResetAccumulatorAfterFlush, ResetAccumulatorOnInterrupt>;

  using ProbeRoot = OperationSet<CoreStage, RetrievalStage>;
  using FullRoot
      = OperationSet<CoreStage, StorageStage, RetrievalStage, FeedbackStage>;

  // Every operation declares its Requires/Satisfies contract; the chain
  // aggregation proves at compile time that both root variants are
  // self-contained (no operation consumes a per-signal value that no
  // earlier operation produced). On failure, inspect the offending root's
  // Input alias: it lists the unsatisfied tags.
  static_assert (FullRoot::kContractsComplete,
                 "every operation must declare its contract");
  static_assert (IsSelfContained<ProbeRoot>,
                 "probe operation set consumes a value no operation produces");
  static_assert (IsSelfContained<FullRoot>,
                 "full operation set consumes a value no operation produces");

  if (probe_mode)
    {
      return operations::signal_record_rollback_internal::
          MarkEngineOwnedJournalAware (
              std::make_unique<ProbeRoot> ());
    }
  return operations::signal_record_rollback_internal::
      MarkEngineOwnedJournalAware (
          std::make_unique<SignalOperationDispatch> (
              std::make_unique<FullRoot> ()));
}

namespace
{

/// Enforce the embedding-model pin: a database that already contains
/// embeddings must carry the fingerprint of the encoder that produced them,
/// and it must match the active encoder. See models/embedding_model_pin.hpp.
void
VerifyEmbeddingModel (cortext::Store &store, const std::string &active)
{
  store.Execute ("CREATE TABLE IF NOT EXISTS cortext_embedding_model_pin ("
                 "  id INTEGER PRIMARY KEY CHECK (id = 1),"
                 "  fingerprint TEXT NOT NULL"
                 ")",
                 {});
  auto rows = store.Execute (
      "SELECT fingerprint FROM cortext_embedding_model_pin WHERE id = 1", {});
  if (!rows.empty ())
    {
      std::string stored;
      auto it = rows[0].find ("fingerprint");
      if (it != rows[0].end () && it->second.type () == typeid (std::string))
        {
          stored = std::any_cast<std::string> (it->second);
        }
      if (stored != active)
        {
          throw std::runtime_error (
              "embedding-model pin mismatch: database vectors were produced by '"
              + stored + "' but the active encoder is '" + active
              + "'. Precomputed embeddings are not comparable across "
                "encoder spaces; re-ingest with the active encoder or "
                "restore the original model.");
        }
      return;
    }

  long long existing = 0;
  try
    {
      auto count_rows
          = store.Execute ("SELECT COUNT(*) AS n FROM embeddings", {});
      if (!count_rows.empty ())
        {
          auto it = count_rows[0].find ("n");
          if (it != count_rows[0].end ())
            {
              if (it->second.type () == typeid (long long))
                {
                  existing = std::any_cast<long long> (it->second);
                }
              else if (it->second.type () == typeid (int))
                {
                  existing = std::any_cast<int> (it->second);
                }
            }
        }
    }
  catch (const std::exception &)
    {
      // Fresh database without the embeddings table yet.
    }
  if (existing > 0)
    {
      throw std::runtime_error (
          "embedding-model pin missing: database already contains "
          + std::to_string (existing)
          + " embeddings with no record of the model that produced them. "
            "There is no override: vectors from an unknown model cannot be "
            "compared against '"
          + active + "'. Re-ingest the data with the active encoder.");
    }
  store.Execute (
      "INSERT INTO cortext_embedding_model_pin (id, fingerprint) VALUES (1, ?)",
      { active });
}

} // namespace

struct Cortext::Impl
{
  Config cfg;
  internal::SignalFilter signal_filter;
  std::string db_path;

  std::unique_ptr<Encoder> encoder;
  std::shared_ptr<internal::DataTimeClock> clock;
  std::shared_ptr<cortext::Store> store;
  std::shared_ptr<cortext::ObjectStore> object_store;
  std::unique_ptr<cortext::IOperation> root_operations;
  std::unique_ptr<cortext::SignalProcessor> processor;

  std::string embedding_model_pin;
  /// Learned late-interaction reranker (required with the AIST encoder).
  std::unique_ptr<internal::LateInteractionModel> late_interaction;

  /// @brief Encodes a signal and builds its late-interaction set (query set
  /// for Ephemeral signals, stored codes otherwise) from the encoder's
  /// position features.
  void
  EncodeTextSignal (const std::string &text, Retention retention,
                    std::vector<float> &v, LateInteractionSet &li)
  {
    // A repeated query reuses its query set; the encoder's own semantic
    // cache then serves the embedding, as it did before position features.
    if (retention == Retention::Ephemeral)
      {
        if (const std::vector<float> *hit = query_set_cache.Find (text))
          {
            UnpackQuerySet (*hit, li);
            encoder->EncodeText (text, v);
            return;
          }
      }
    EncoderPositions positions;
    encoder->EncodeTextPositions (text, v, positions);
    li = BuildLateInteraction (positions, retention);
    if (retention == Retention::Ephemeral)
      {
        query_set_cache.Put (text, PackQuerySet (li));
      }
  }

  /// Query sets of recent text queries, packed as
  /// [count, dim, values (count x dim), gate (count)], byte-bounded LRU.
  internal::AistSemanticCache query_set_cache;

  static std::vector<float>
  PackQuerySet (const LateInteractionSet &set)
  {
    std::vector<float> out;
    out.reserve (2 + set.values.size () + set.gate.size ());
    out.push_back (static_cast<float> (set.count));
    out.push_back (static_cast<float> (set.dim));
    out.insert (out.end (), set.values.begin (), set.values.end ());
    out.insert (out.end (), set.gate.begin (), set.gate.end ());
    return out;
  }

  static void
  UnpackQuerySet (const std::vector<float> &packed, LateInteractionSet &set)
  {
    set = {};
    set.count = static_cast<int> (packed[0]);
    set.dim = static_cast<int> (packed[1]);
    const auto n = static_cast<std::ptrdiff_t> (set.count) * set.dim;
    set.values.assign (packed.begin () + 2, packed.begin () + 2 + n);
    set.gate.assign (packed.begin () + 2 + n, packed.end ());
  }

  void
  EncodeImageSignal (const std::uint8_t *data, int width, int height,
                     int channels, Retention retention, std::vector<float> &v,
                     LateInteractionSet &li)
  {
    EncoderPositions positions;
    encoder->EncodeImagePositions (data, width, height, channels, v,
                                   positions);
    li = BuildLateInteraction (positions, retention);
  }

  void
  EncodeAudioSignal (const float *pcm, std::size_t num_samples,
                     Retention retention, std::vector<float> &v,
                     LateInteractionSet &li)
  {
    EncoderPositions positions;
    encoder->EncodeAudioPositions (pcm, num_samples, v, positions);
    li = BuildLateInteraction (positions, retention);
  }

  LateInteractionSet
  BuildLateInteraction (const EncoderPositions &positions,
                        Retention retention) const
  {
    return retention == Retention::Ephemeral
               ? late_interaction->QuerySet (positions)
               : late_interaction->MemorySet (positions);
  }

  Impl (const Config &c, std::shared_ptr<cortext::Store> supplied_store,
        std::shared_ptr<cortext::ObjectStore> supplied_object_store,
        std::shared_ptr<cortext::Clock> supplied_clock)
      : cfg (c), signal_filter (MakeSignalFilterConfig (c)),
        clock (std::make_shared<internal::DataTimeClock> (
            supplied_clock ? std::move (supplied_clock)
                           : std::make_shared<cortext::SystemClock> ())),
        store (std::move (supplied_store)),
        object_store (std::move (supplied_object_store))
  {
    if (!store)
      {
        throw std::invalid_argument ("Cortext requires a non-null Store");
      }
    if (!object_store)
      {
        object_store = std::make_shared<cortext::SqlObjectStore> (store);
      }

    auto text_encoder = internal::CreatePreferredTextEncoder ();
    // Pin every precomputed vector to the encoder that produced it. The
    // runtime similarity space is the 256-d Matryoshka slice (see the vec0
    // schema in store/schema.cpp); comparing vectors across encoder spaces
    // is silently wrong, so mismatches are fatal rather than degraded.
    embedding_model_pin = models::ComputeEmbeddingModelPin (
        text_encoder.backend_name, text_encoder.resolved_path, 256);
    VerifyEmbeddingModel (*store, embedding_model_pin);
    encoder = std::move (text_encoder.encoder);
    // The late-interaction reranker ships with the AIST model (Git shards
    // linked into the library) and is required with it, like the encoder.
    const auto li_path = internal::ResolveLateInteractionModelPath (
        text_encoder.resolved_path);
    if (!li_path)
      {
        throw std::runtime_error (
            "AIST late-interaction reranker (AIST-87M-LI.gguf) not found beside "
            + text_encoder.resolved_path.string ()
            + " and not linked into this build; set CORTEXT_LI_MODEL_PATH "
              "to its .gguf file");
      }
    late_interaction = internal::LateInteractionModel::Load (*li_path);

    root_operations = BuildRootOperationSet (false);
    processor = std::make_unique<cortext::SignalProcessor> (
        MakeProcessorConfig (), store, std::move (root_operations),
        object_store);
    ObserveStoredSignalTime ();
  }

  // Engine time starts at the newest stored signal (indexed max) so a
  // reopened store is never in the future of its own clock.
  void
  ObserveStoredSignalTime ()
  {
    try
      {
        auto rows = store->Execute (
            "SELECT MAX(timestamp) AS newest FROM signals", {});
        if (!rows.empty ())
          {
            auto it = rows[0].find ("newest");
            if (it != rows[0].end () && it->second.has_value ()
                && it->second.type () == typeid (long long))
              {
                const auto newest = std::any_cast<long long> (it->second);
                if (newest > 0)
                  clock->Observe (static_cast<std::uint64_t> (newest));
              }
          }
      }
    catch (const std::exception &e)
      {
        telemetry::LogWarn ("cortext.clock.stored_time_unavailable",
                            { telemetry::Attribute::String ("error",
                                                            e.what ()) });
        return;
      }
    if (const auto lag = clock->LagMillis (); lag > 0)
      telemetry::LogInfo (
          "cortext.clock.behind_stored_signals",
          { telemetry::Attribute::Int64 ("lag_ms",
                                         static_cast<std::int64_t> (lag)) });
  }

  // Every signal goes through here: a stored signal's timestamp becomes
  // the floor of engine time before it is processed.
  cortext::SignalProcessor::Output
  Process (const cortext::Signal &signal)
  {
    if (signal.retention != Retention::Ephemeral)
      clock->Observe (signal.timestamp);
    return processor->Process (signal);
  }

  Impl (const Config &c, std::string db,
        std::shared_ptr<cortext::Clock> supplied_clock)
      : Impl (c, std::shared_ptr<cortext::Store> (
                     cortext::SQLiteStore::Create (db.c_str ())),
              nullptr, std::move (supplied_clock))
  {
    db_path = std::move (db);
  }

  std::uint64_t
  NowMillis () const
  {
    return clock->NowMillis ();
  }

  cortext::SignalProcessor::Config
  MakeProcessorConfig () const
  {
    cortext::SignalProcessor::Config pcfg;
    pcfg.focus = cfg.focus;
    pcfg.sensitivity = cfg.sensitivity;
    pcfg.stability = cfg.stability;
    pcfg.affect_interrupt = cfg.affect_interrupt;
    pcfg.encoder = encoder.get ();
    pcfg.clock = clock;
    pcfg.late_interaction = late_interaction.get ();

    return pcfg;
  }

  std::unique_ptr<cortext::SignalProcessor>
  MakeProbeProcessor () const
  {
    return std::make_unique<cortext::SignalProcessor> (
        MakeProcessorConfig (), store, BuildRootOperationSet (true),
        object_store);
  }

  void
  ResetProcessor ()
  {
    auto next_root = BuildRootOperationSet (false);
    auto next_processor = std::make_unique<cortext::SignalProcessor> (
        MakeProcessorConfig (), store, std::move (next_root), object_store);
    processor = std::move (next_processor);
    root_operations.reset ();
  }

  Cortext::Context
  ProcessTextEmbeddingAt (const Eigen::VectorXf &embedding,
                          std::uint64_t timestamp,
                          const std::string &source_id,
                          const std::string &text,
                          Retention retention = Retention::Natural)
  {
    telemetry::ScopedSpan span ("cortext.api.process_text_cached");
    const auto total_start = std::chrono::steady_clock::now ();
    const auto process_start = total_start;
    const auto encoded = ToStdVector (embedding);

    cortext::Signal s;
    s.embedding = ToEigen (RetrievalEmbeddingView (encoded));
    s.timestamp = timestamp != 0 ? timestamp : NowMillis ();
    s.source_id = source_id;
    s.retention = retention;
    s.payload = std::vector<unsigned char> (text.begin (), text.end ());
    s.modality = "text";
    s.mimetype = "text/plain";

    auto out = Process (s);
    const auto process_end = std::chrono::steady_clock::now ();
    span.SetAttribute (
        "cortext.candidate_memory_count",
        static_cast<std::int64_t> (out.candidate_memory_ids.size ()));
    span.SetAttribute (
        "cortext.used_memory_count",
        static_cast<std::int64_t> (out.used_memory_ids.size ()));
    span.SetStatusOk ();
    telemetry::ScopedSpan hydrate_span ("cortext.hydrate");
    const auto hydrate_start = std::chrono::steady_clock::now ();
    Cortext::Context result = HydrateContext (out, true, &s.embedding, true, &s);
    const auto hydrate_end = std::chrono::steady_clock::now ();
    hydrate_span.SetStatusOk ();
    result.encode_ms = 0.0;
    result.process_ms = ToMillis (process_end - process_start);
    result.hydrate_ms = ToMillis (hydrate_end - hydrate_start);
    result.total_ms = ToMillis (hydrate_end - total_start);
    return result;
  }

  cortext::SignalProcessor::Output
  ProcessEmbedding (const Eigen::VectorXf &embedding, std::uint64_t timestamp,
                    const std::string &source_id,
                    bool force_consolidation = false)
  {
    const auto encoded = ToStdVector (embedding);
    cortext::Signal s;
    s.embedding = ToEigen (RetrievalEmbeddingView (encoded));
    s.timestamp = timestamp;
    s.source_id = source_id;
    s.force_consolidation = force_consolidation;
    return Process (s);
  }

  Cortext::Context
  ConsolidateAt (StopToken stop_token, std::uint64_t timestamp)
  {
    internal::ScopedStopToken scoped_stop (stop_token);
    auto *sqlite_store = dynamic_cast<SQLiteStore *> (store.get ());
    std::unique_ptr<StopCallback<std::function<void ()>>> stop_callback;
    if (sqlite_store && stop_token.stop_possible ())
      {
        stop_callback
            = std::make_unique<StopCallback<std::function<void ()>>> (
                stop_token, [sqlite_store] {
                  internal::SQLiteStoreQueryInterrupter::Interrupt (
                      *sqlite_store);
                });
      }
    internal::ThrowIfStopRequested ();
    telemetry::ScopedSpan span ("cortext.api.consolidate");
    const auto total_start = std::chrono::steady_clock::now ();
    // A maintenance signal runs no operations; the processor seals and
    // recenters the sparse retrieval route around it.
    std::vector<float> v;
    telemetry::ScopedSpan encode_span ("cortext.encode");
    encoder->EncodeText (std::string (), v);
    encode_span.SetStatusOk ();
    const auto encode_end = std::chrono::steady_clock::now ();
    internal::ThrowIfStopRequested ();
    const auto process_start = std::chrono::steady_clock::now ();
    const Eigen::VectorXf consolidation_embedding
        = ToEigen (RetrievalEmbeddingView (v));
    auto out = ProcessEmbedding (consolidation_embedding,
                                 timestamp != 0 ? timestamp : NowMillis (),
                                 kMaintenanceSourceId, true);
    const auto process_end = std::chrono::steady_clock::now ();
    span.SetAttribute (
        "cortext.candidate_memory_count",
        static_cast<std::int64_t> (out.candidate_memory_ids.size ()));
    span.SetAttribute (
        "cortext.used_memory_count",
        static_cast<std::int64_t> (out.used_memory_ids.size ()));
    span.SetStatusOk ();
    telemetry::ScopedSpan hydrate_span ("cortext.hydrate");
    const auto hydrate_start = std::chrono::steady_clock::now ();
    Cortext::Context result
        = HydrateContext (out, true, &consolidation_embedding);
    const auto hydrate_end = std::chrono::steady_clock::now ();
    hydrate_span.SetStatusOk ();
    result.encode_ms = ToMillis (encode_end - total_start);
    result.process_ms = ToMillis (process_end - process_start);
    result.hydrate_ms = ToMillis (hydrate_end - hydrate_start);
    result.total_ms = ToMillis (hydrate_end - total_start);
    return result;
  }

  Cortext::Context
  HydrateContext (const cortext::SignalProcessor::Output &out,
                  bool hydrate_working_memory = true,
                  const Eigen::VectorXf *query_embedding = nullptr,
                  bool hydrate_retrieved_memory = true,
                  const cortext::Signal *query_signal = nullptr,
                  Cortext::EvidenceBudget evidence_budget = std::nullopt)
  {
    Cortext::Context result;
    if (query_embedding)
      {
        result.embedding = ToStdVector (*query_embedding);
      }
    result.should_interrupt = out.interrupt_allowed;
    result.interrupt_aborted = out.interrupt_aborted;
    result.at_boundary = out.at_boundary;
    result.consolidation_state = out.consolidation_state;
    result.interrupt_gate_has_candidates = out.interrupt_gate_has_candidates;
    result.interrupt_gate_blocked_no_store = out.interrupt_gate_blocked_no_store;
    result.interrupt_gate_rel_pass = out.interrupt_gate_rel_pass;
    result.interrupt_gate_novelty_pass = out.interrupt_gate_novelty_pass;
    result.interrupt_gate_mu_pass = out.interrupt_gate_mu_pass;
    result.interrupt_gate_novelty_mu_pass = out.interrupt_gate_novelty_mu_pass;
    result.interrupt_gate_dup_pass = out.interrupt_gate_dup_pass;
    result.interrupt_gate_boundary_mu_pass = out.interrupt_gate_boundary_mu_pass;
    result.interrupt_gate_rel_star = out.interrupt_gate_rel_star;
    result.interrupt_gate_retrieval_thresh =
        out.interrupt_gate_retrieval_thresh;
    result.interrupt_gate_boundary_mult_eff =
        out.interrupt_gate_boundary_mult_eff;
    result.interrupt_gate_affect_drive = out.interrupt_gate_affect_drive;
    result.boundary_score = out.boundary_score;
    result.boundary_type = out.boundary_type;

    // Populate output metrics
    result.output.composite_score = out.composite_score;
    result.output.threshold = out.threshold_T_dynamic;
    result.output.decision = out.write_decision;
    result.output.effective_focus = out.effective_focus;
    result.output.coherence = out.coherence;
    result.output.emotion_intensity = out.emotion_intensity;
    result.output.valence = out.valence;
    result.output.arousal = out.arousal;
    result.output.operation_ms = out.operation_ms;
    
    // Convert metrics map (enum -> int)
    for (const auto& [metric_enum, value] : out.metrics) {
    result.output.metrics[static_cast<int>(metric_enum)] = value;
  }

  // Wire stored_embedding_id from MemoryStorage operation
  result.output.stored_embedding_id = out.stored_embedding_id;
  result.output.stored_memory_id = out.stored_memory_id;
  result.output.stored_signal_id = out.stored_signal_id;
    
	    if (!store)
	      {
	        return result;
	      }

	    const int max_fallback_signal_blobs
	        = operations::sparse_retrieval_knobs_internal::
	            HydrationFallbackSignalLimit (
	            cfg.focus, cfg.sensitivity, cfg.stability);
	    std::size_t loaded_fallback_signal_blobs = 0;

	    if (hydrate_working_memory)
	      {
	        HydrateWorkingMemoryFromDB (store.get (), object_store.get (),
	                                    result.working_memory,
	                                    max_fallback_signal_blobs,
	                                    &loaded_fallback_signal_blobs);
	      }
	    result.output.operation_ms["Cortext.fallback_hydration_signal_limit"]
	        = static_cast<double> (max_fallback_signal_blobs);
	    result.output.operation_ms["Cortext.fallback_hydration_signal_rows"]
	        = static_cast<double> (loaded_fallback_signal_blobs);

    if (!hydrate_retrieved_memory)
      {
        return result;
      }

    auto resolve_candidate_memory_id = [&](long long candidate_id) -> long long {
      if (candidate_id <= 0)
        {
          return 0;
        }
      auto memory_rows = store->Execute (
          "SELECT memory_id FROM memories WHERE memory_id = ?",
          { candidate_id });
      if (!memory_rows.empty () && memory_rows[0].count ("memory_id") == 1)
        {
          return cortext::store::AnyToLongLong (
                     memory_rows[0].at ("memory_id"))
              .value_or (0);
        }
      auto rows = store->Execute (
          "SELECT memory_id FROM memories WHERE embedding_id = ?",
          { candidate_id });
      if (!rows.empty () && rows[0].count ("memory_id") == 1)
        {
          return cortext::store::AnyToLongLong (rows[0].at ("memory_id"))
              .value_or (0);
        }
      auto sig_rows = store->Execute (
          "SELECT memory_id FROM signals WHERE embedding_id = ? LIMIT 1",
          { candidate_id });
      if (!sig_rows.empty () && sig_rows[0].count ("memory_id") == 1)
        {
          return cortext::store::AnyToLongLong (
                     sig_rows[0].at ("memory_id"))
              .value_or (0);
        }
      return 0;
    };

    std::unordered_set<long long> seen_candidate_memory_ids;
    seen_candidate_memory_ids.reserve (out.candidate_memory_ids.size ());
    std::unordered_set<long long> seen_output_memory_ids;
    seen_output_memory_ids.reserve (out.candidate_memory_ids.size ());
    std::unordered_map<long long, operations::retrieval_trace::RankedCandidate>
        ranked_by_memory_id;
    for (const auto &candidate :
         operations::retrieval_trace::GetLastRankedCandidates ())
      {
        if (candidate.memory_id > 0)
          {
            ranked_by_memory_id.emplace (candidate.memory_id, candidate);
          }
      }

    std::vector<long long> ordered_candidate_memory_ids;
    ordered_candidate_memory_ids.reserve (out.candidate_memory_ids.size ());
    std::unordered_set<long long> seen_ordered_memory_ids;
    seen_ordered_memory_ids.reserve (out.candidate_memory_ids.size ());
    std::unordered_set<long long> output_candidate_memory_ids (
        out.candidate_memory_ids.begin (), out.candidate_memory_ids.end ());
    for (const auto &candidate :
         operations::retrieval_trace::GetLastRankedCandidates ())
      {
        if (candidate.memory_id > 0
            && output_candidate_memory_ids.count (candidate.memory_id) != 0
            && seen_ordered_memory_ids.insert (candidate.memory_id).second)
          {
            ordered_candidate_memory_ids.push_back (candidate.memory_id);
          }
      }
    for (const long long candidate_id : out.candidate_memory_ids)
      {
        const long long memory_id = resolve_candidate_memory_id (candidate_id);
        if (memory_id > 0 && seen_ordered_memory_ids.insert (memory_id).second)
          {
            ordered_candidate_memory_ids.push_back (memory_id);
          }
      }
    for (const long long memory_id : ordered_candidate_memory_ids)
      {
        if (memory_id <= 0
            || !seen_candidate_memory_ids.insert (memory_id).second)
          {
            continue;
          }
        Cortext::Context::Memory m;
        m.id = memory_id;
        HydrateMemory (store.get (), object_store.get (), memory_id, m,
                       max_fallback_signal_blobs,
                       &loaded_fallback_signal_blobs);
        ApplyRetrievalScore (m, ranked_by_memory_id);
        // Retrieved memories without hydrated content are skipped; nothing
        // writes the derived_from / has_label routes that used to resolve
        // them to linked source memories.
        if (HasHydratedContent (m)
            && seen_output_memory_ids.insert (memory_id).second)
          {
            StripSourceBlobsIfDisabled (m);
            result.retrieved_memory.push_back (std::move (m));
          }
      }

    // The attachment steps only read: one object-store transaction serves
    // all their payload reads instead of one transaction per read.
    std::unique_ptr<ObjectTransaction> attachment_object_tx;
    SharedObjectReads attachment_reads;
    if (!SourceBlobsDisabled () && object_store)
      {
        try
          {
            attachment_object_tx = object_store->Begin ();
            attachment_reads.object_tx = attachment_object_tx.get ();
          }
        catch (const std::exception &e)
          {
            attachment_object_tx.reset ();
            attachment_reads = {};
            telemetry::LogWarn (
                "Failed to begin attachment object reads",
                { telemetry::Attribute::String ("component", "cortext"),
                  telemetry::Attribute::String ("error", e.what ()) });
          }
      }
    AttachmentHydrations attachment_hydrations (
        store.get (), object_store.get (), &attachment_reads,
        max_fallback_signal_blobs, &loaded_fallback_signal_blobs,
        ClipTermsOf (query_signal));
    // The evidence budget fit (FitEvidenceToBudget) runs once the packet is
    // complete; it finds each memory's own content among its attached
    // neighbours.
    const auto own_content = evidence_budget
                                 ? OwnContentOf (result.retrieved_memory)
                                 : std::vector<OwnContent> ();
    const auto anchor_rows = SourceBlobsDisabled ()
        ? std::unordered_map<long long, AnchorRow> ()
        : LoadAnchorRows (store.get (), result.retrieved_memory,
                          std::max (kContiguityAnchorLimit,
                                    kSupersededRecallAnchorLimit));
    // An as-of query (Cortext::RetrievalTime) attaches no context stored
    // after its reference time.
    const std::optional<long long> as_of_ts
        = query_signal && query_signal->reference_ts
                  && query_signal->reference_as_of
              ? std::optional<long long> (static_cast<long long> (
                    std::min<std::uint64_t> (
                        *query_signal->reference_ts,
                        static_cast<std::uint64_t> (
                            std::numeric_limits<long long>::max ()))))
              : std::nullopt;
    std::unordered_set<long long> contiguous_turn_ids;
    const std::size_t contiguous_turns_attached = AttachContiguousTurns (
        store.get (), result.retrieved_memory, anchor_rows,
        attachment_hydrations, contiguous_turn_ids, as_of_ts);
    result.output.operation_ms["Cortext.contiguous_turns_attached"]
        = static_cast<double> (contiguous_turns_attached);
    const std::size_t superseding_successors_attached
        = AttachSupersedingSuccessors (store.get (), result.retrieved_memory,
                                       anchor_rows, contiguous_turn_ids,
                                       attachment_hydrations, as_of_ts);
    result.output.operation_ms["Cortext.superseding_successors_attached"]
        = static_cast<double> (superseding_successors_attached);
    const std::size_t superseded_predecessors_attached
        = AttachSupersededPredecessors (
            store.get (), result.retrieved_memory, anchor_rows,
            std::move (contiguous_turn_ids), attachment_hydrations, as_of_ts);
    result.output.operation_ms["Cortext.superseded_predecessors_attached"]
        = static_cast<double> (superseded_predecessors_attached);

    // Standing signals (operations/standing_memory_internal.hpp): short
    // lasting directives an answer must honour, chosen within their own set.
    // They form a block of their own (Context::standing_memory), outside the
    // retrieved ranking: each entry holds only the directive signals of one
    // memory, best first, within kStandingPacketBytes (and the evidence
    // budget), and the retrieved memories keep their ranks and content
    // whether or not they hold one.
    const std::size_t standing_limit
        = evidence_budget ? std::min (kStandingPacketBytes, *evidence_budget)
                          : kStandingPacketBytes;
    std::size_t standing_bytes = 0;
    for (const auto &pick :
         operations::retrieval_trace::GetLastStandingPicks ())
      {
        Cortext::Context::Memory directive;
        directive.id = pick.memory_id;
        const auto cost = LoadStandingSignal (store.get (), object_store.get (),
                                              pick.signal_id, directive);
        if (cost == 0)
          continue;
        if ((evidence_budget || !result.standing_memory.empty ())
            && standing_bytes + cost > standing_limit)
          continue;
        standing_bytes += cost;
        const auto joined = std::find_if (
            result.standing_memory.begin (), result.standing_memory.end (),
            [&] (const auto &m) { return m.id == pick.memory_id; });
        if (joined != result.standing_memory.end ())
          {
            // A second directive signal of a memory already in the block.
            AlignContentMimetypes (*joined);
            for (std::size_t i = 0; i < directive.content.size (); ++i)
              {
                joined->content.push_back (std::move (directive.content[i]));
                joined->content_mimetypes.push_back (
                    std::move (directive.content_mimetypes[i]));
                joined->content_timestamps.push_back (
                    directive.content_timestamps[i]);
              }
            continue;
          }
        Cortext::Context::Memory m;
        m.id = pick.memory_id;
        HydrateMemory (store.get (), object_store.get (), pick.memory_id, m,
                       max_fallback_signal_blobs,
                       &loaded_fallback_signal_blobs);
        if (!HasHydratedContent (m))
          continue;
        m.content = std::move (directive.content);
        m.content_mimetypes = std::move (directive.content_mimetypes);
        m.content_timestamps = std::move (directive.content_timestamps);
        StripSourceBlobsIfDisabled (m);
        result.standing_memory.push_back (std::move (m));
      }
    result.output.operation_ms["Cortext.standing_joined"]
        = static_cast<double> (result.standing_memory.size ());

    if (evidence_budget)
      {
        const auto fit = FitEvidenceToBudget (
            store.get (), object_store.get (), &attachment_reads,
            result.retrieved_memory, own_content, standing_bytes,
            query_embedding, *evidence_budget);
        result.output.operation_ms["Cortext.evidence_bytes"]
            = static_cast<double> (fit.bytes);
        result.output.operation_ms["Cortext.evidence_member_cap"]
            = static_cast<double> (fit.member_cap);
        result.output.operation_ms["Cortext.evidence_memories_trimmed"]
            = static_cast<double> (fit.trimmed);
        result.output.operation_ms["Cortext.evidence_neighbours_stripped"]
            = static_cast<double> (fit.stripped);
        result.output.operation_ms["Cortext.evidence_memories_dropped"]
            = static_cast<double> (fit.dropped);
        telemetry::AddCounter ("cortext.evidence_budget.fits_total", 1);
        telemetry::AddCounter ("cortext.evidence_budget.memories_trimmed_total",
                               fit.trimmed);
        telemetry::AddCounter ("cortext.evidence_budget.memories_dropped_total",
                               fit.dropped);
      }
    if (attachment_object_tx)
      {
        try
          {
            attachment_object_tx->Commit ();
          }
        catch (const std::exception &e)
          {
            telemetry::LogWarn (
                "Failed to end attachment object reads",
                { telemetry::Attribute::String ("component", "cortext"),
                  telemetry::Attribute::String ("error", e.what ()) });
          }
      }
    result.output.operation_ms["Cortext.fallback_hydration_signal_rows"]
        = static_cast<double> (loaded_fallback_signal_blobs);
    return result;
  }
};

std::unique_ptr<Cortext>
Cortext::Create (const Config &cfg, const std::string &db_path)
{
  return Cortext::Create (cfg, db_path, std::shared_ptr<Clock>{});
}

std::unique_ptr<Cortext>
Cortext::Create (const Config &cfg, const std::string &db_path,
                 std::shared_ptr<Clock> clock)
{
  return std::unique_ptr<Cortext> (
      new Cortext (cfg, db_path, std::move (clock)));
}

std::unique_ptr<Cortext>
Cortext::Create (const Config &cfg, const std::string &db_path,
                 std::shared_ptr<ObjectStore> object_store)
{
  return Cortext::Create (cfg, db_path, std::move (object_store),
                          std::shared_ptr<Clock>{});
}

std::unique_ptr<Cortext>
Cortext::Create (const Config &cfg, const std::string &db_path,
                 std::shared_ptr<ObjectStore> object_store,
                 std::shared_ptr<Clock> clock)
{
  return std::unique_ptr<Cortext> (
      new Cortext (cfg, db_path, std::move (object_store), std::move (clock)));
}

std::unique_ptr<Cortext>
Cortext::Create (const Config &cfg, std::shared_ptr<Store> store)
{
  return Cortext::Create (cfg, std::move (store), std::shared_ptr<Clock>{});
}

std::unique_ptr<Cortext>
Cortext::Create (const Config &cfg, std::shared_ptr<Store> store,
                 std::shared_ptr<Clock> clock)
{
  return std::unique_ptr<Cortext> (
      new Cortext (cfg, std::move (store), std::move (clock)));
}

std::unique_ptr<Cortext>
Cortext::Create (const Config &cfg, std::shared_ptr<Store> store,
                 std::shared_ptr<ObjectStore> object_store)
{
  return Cortext::Create (cfg, std::move (store), std::move (object_store),
                          std::shared_ptr<Clock>{});
}

std::unique_ptr<Cortext>
Cortext::Create (const Config &cfg, std::shared_ptr<Store> store,
                 std::shared_ptr<ObjectStore> object_store,
                 std::shared_ptr<Clock> clock)
{
  return std::unique_ptr<Cortext> (
      new Cortext (cfg, std::move (store), std::move (object_store),
                   std::move (clock)));
}

Cortext::Cortext (const Config &cfg, const std::string &db_path,
                  std::shared_ptr<Clock> clock)
    : impl_ (std::make_unique<Impl> (cfg, db_path, std::move (clock)))
{
}

Cortext::Cortext (const Config &cfg, const std::string &db_path,
                  std::shared_ptr<ObjectStore> object_store)
    : Cortext (cfg, db_path, std::move (object_store), nullptr)
{
}

Cortext::Cortext (const Config &cfg, const std::string &db_path,
                  std::shared_ptr<ObjectStore> object_store,
                  std::shared_ptr<Clock> clock)
    : impl_ (std::make_unique<Impl> (
          cfg,
          std::shared_ptr<cortext::Store> (
              cortext::SQLiteStore::Create (db_path.c_str ())),
          std::move (object_store), std::move (clock)))
{
}

Cortext::Cortext (const Config &cfg, std::shared_ptr<Store> store,
                  std::shared_ptr<Clock> clock)
    : impl_ (std::make_unique<Impl> (cfg, std::move (store), nullptr,
                                     std::move (clock)))
{
}

Cortext::Cortext (const Config &cfg, std::shared_ptr<Store> store,
                  std::shared_ptr<ObjectStore> object_store)
    : Cortext (cfg, std::move (store), std::move (object_store), nullptr)
{
}

Cortext::Cortext (const Config &cfg, std::shared_ptr<Store> store,
                  std::shared_ptr<ObjectStore> object_store,
                  std::shared_ptr<Clock> clock)
    : impl_ (std::make_unique<Impl> (cfg, std::move (store),
                                     std::move (object_store),
                                     std::move (clock)))
{
}

Cortext::~Cortext () = default;

Cortext::Context
Cortext::ProcessText (const std::string &text, const std::string &source_id,
                      Retention retention,
                      EvidenceBudget evidence_budget,
                      RetrievalTime retrieval_time)
{
  return ProcessText (text, source_id, Media{}, retention, evidence_budget,
                      retrieval_time);
}

Cortext::Context
Cortext::ProcessText (const std::string &text, const std::string &source_id,
                      const Media &media, Retention retention,
                      EvidenceBudget evidence_budget,
                      RetrievalTime retrieval_time)
{
  if (!impl_)
    {
      throw std::runtime_error ("Cortext not initialized");
    }
  ValidateMedia (media);
  telemetry::ScopedSpan span ("cortext.api.process_text");
  const auto total_start = std::chrono::steady_clock::now ();
  const auto filter_decision = impl_->signal_filter.EvaluateText (text);
  if (filter_decision.evaluated && !filter_decision.accepted)
    {
      span.SetStatusOk ();
      return MakeFilteredContext (filter_decision, total_start);
    }
  std::vector<float> v;
  telemetry::ScopedSpan encode_span ("cortext.encode");
  LateInteractionSet li;
  impl_->EncodeTextSignal (text, retention, v, li);
  encode_span.SetStatusOk ();
  const auto encode_end = std::chrono::steady_clock::now ();

  // Build signal with payload for MemoryStorage: the caller's Media when it
  // carries bytes, else the text itself. Text stays the encoded input.
  const auto process_start = std::chrono::steady_clock::now ();
  cortext::Signal s;
  s.embedding = ToEigen (RetrievalEmbeddingView (v));
  s.timestamp = impl_->NowMillis ();
  s.source_id = source_id;
  s.retention = retention;
  s.late_interaction = std::move (li);
  s.modality = "text";
  ApplyStoredMedia (s, media, text.data (), text.size (), "text/plain");

  ApplyRetrievalTime (s, retrieval_time);
  auto out = impl_->Process (s);
  const auto process_end = std::chrono::steady_clock::now ();
  span.SetAttribute ("cortext.candidate_memory_count",
                     static_cast<std::int64_t> (out.candidate_memory_ids.size ()));
  span.SetAttribute ("cortext.used_memory_count",
                     static_cast<std::int64_t> (out.used_memory_ids.size ()));
  span.SetStatusOk ();
  telemetry::ScopedSpan hydrate_span ("cortext.hydrate");
  const auto hydrate_start = std::chrono::steady_clock::now ();
  Cortext::Context result = impl_->HydrateContext (out, true, &s.embedding, true, &s,
                                                   evidence_budget);
  const auto hydrate_end = std::chrono::steady_clock::now ();
  hydrate_span.SetStatusOk ();
  result.encode_ms = ToMillis (encode_end - total_start);
  result.process_ms = ToMillis (process_end - process_start);
  result.hydrate_ms = ToMillis (hydrate_end - hydrate_start);
  result.total_ms = ToMillis (hydrate_end - total_start);
  ApplySignalFilterDecision (result.output, filter_decision);
  return result;
}

Cortext::Context
Cortext::ProcessTextAt (const std::string &text, const std::string &source_id,
                        std::uint64_t timestamp, Retention retention,
                        EvidenceBudget evidence_budget,
                        RetrievalTime retrieval_time)
{
  if (!impl_)
    {
      throw std::runtime_error ("Cortext not initialized");
    }
  telemetry::ScopedSpan span ("cortext.api.process_text");
  const auto total_start = std::chrono::steady_clock::now ();
  const auto filter_decision = impl_->signal_filter.EvaluateText (text);
  if (filter_decision.evaluated && !filter_decision.accepted)
    {
      span.SetStatusOk ();
      return MakeFilteredContext (filter_decision, total_start);
    }
  std::vector<float> v;
  telemetry::ScopedSpan encode_span ("cortext.encode");
  LateInteractionSet li;
  impl_->EncodeTextSignal (text, retention, v, li);
  encode_span.SetStatusOk ();
  const auto encode_end = std::chrono::steady_clock::now ();

  // Build signal with payload for MemoryStorage
  const auto process_start = std::chrono::steady_clock::now ();
  cortext::Signal s;
  s.embedding = ToEigen (RetrievalEmbeddingView (v));
  s.timestamp = timestamp;
  s.source_id = source_id;
  s.retention = retention;
  s.late_interaction = std::move (li);
  s.payload = std::vector<unsigned char> (text.begin (), text.end ());
  s.modality = "text";
  s.mimetype = "text/plain";

  ApplyRetrievalTime (s, retrieval_time);
  auto out = impl_->Process (s);
  const auto process_end = std::chrono::steady_clock::now ();
  span.SetAttribute ("cortext.candidate_memory_count",
                     static_cast<std::int64_t> (out.candidate_memory_ids.size ()));
  span.SetAttribute ("cortext.used_memory_count",
                     static_cast<std::int64_t> (out.used_memory_ids.size ()));
  span.SetStatusOk ();
  telemetry::ScopedSpan hydrate_span ("cortext.hydrate");
  const auto hydrate_start = std::chrono::steady_clock::now ();
  Cortext::Context result = impl_->HydrateContext (out, true, &s.embedding, true, &s,
                                                   evidence_budget);
  const auto hydrate_end = std::chrono::steady_clock::now ();
  hydrate_span.SetStatusOk ();
  result.encode_ms = ToMillis (encode_end - total_start);
  result.process_ms = ToMillis (process_end - process_start);
  result.hydrate_ms = ToMillis (hydrate_end - hydrate_start);
  result.total_ms = ToMillis (hydrate_end - total_start);
  ApplySignalFilterDecision (result.output, filter_decision);
  return result;
}

Cortext::Context
Cortext::ProcessAudio (const float *pcm, std::size_t num_samples,
                       const std::string &source_id, Retention retention,
                       EvidenceBudget evidence_budget,
                       RetrievalTime retrieval_time)
{
  if (!impl_)
    {
      throw std::runtime_error ("Cortext not initialized");
    }
#if defined(CORTEXT_DISABLE_AUDIO)
  (void)pcm;
  (void)num_samples;
  (void)source_id;
  (void)retention;
  ThrowAudioSupportDisabled ();
#else
  ValidateAudioInput (pcm);
  telemetry::ScopedSpan span ("cortext.api.process_audio");
  const auto total_start = std::chrono::steady_clock::now ();
  const auto filter_decision
      = impl_->signal_filter.EvaluateAudio (source_id, pcm, num_samples, 16000);
  if (filter_decision.evaluated && !filter_decision.accepted)
    {
      span.SetAttribute ("cortext.signal_filter.modality",
                         filter_decision.modality);
      span.SetAttribute ("cortext.signal_filter.reason",
                         filter_decision.reason);
      span.SetStatusOk ();
      return MakeFilteredContext (filter_decision, total_start);
    }
  std::vector<float> v;
  telemetry::ScopedSpan encode_span ("cortext.encode");
  LateInteractionSet li;
  impl_->EncodeAudioSignal (pcm, num_samples, retention, v, li);
  encode_span.SetStatusOk ();
  const auto encode_end = std::chrono::steady_clock::now ();

  // Build signal with payload for MemoryStorage
  const auto process_start = std::chrono::steady_clock::now ();
  cortext::Signal s;
  s.embedding = ToEigen (RetrievalEmbeddingView (v));
  s.timestamp = impl_->NowMillis ();
  s.source_id = source_id;
  s.retention = retention;
  s.late_interaction = std::move (li);
  s.modality = "audio";
  ApplyStoredMedia (s, Media{}, pcm, num_samples * sizeof (float),
                    MakeAudioMimePcmF32 ());
  s.sample_rate = 16000; // Public audio ingress contract is 16 kHz mono PCM.
  s.num_samples = num_samples;

  ApplyRetrievalTime (s, retrieval_time);
  auto out = impl_->Process (s);
  const auto process_end = std::chrono::steady_clock::now ();
  span.SetAttribute ("cortext.candidate_memory_count",
                     static_cast<std::int64_t> (out.candidate_memory_ids.size ()));
  span.SetAttribute ("cortext.used_memory_count",
                     static_cast<std::int64_t> (out.used_memory_ids.size ()));
  span.SetStatusOk ();
  telemetry::ScopedSpan hydrate_span ("cortext.hydrate");
  const auto hydrate_start = std::chrono::steady_clock::now ();
  Cortext::Context result = impl_->HydrateContext (out, true, &s.embedding, true, &s,
                                                   evidence_budget);
  const auto hydrate_end = std::chrono::steady_clock::now ();
  hydrate_span.SetStatusOk ();
  result.encode_ms = ToMillis (encode_end - total_start);
  result.process_ms = ToMillis (process_end - process_start);
  result.hydrate_ms = ToMillis (hydrate_end - hydrate_start);
  result.total_ms = ToMillis (hydrate_end - total_start);
  ApplySignalFilterDecision (result.output, filter_decision);
  return result;
#endif
}

Cortext::Context
Cortext::ProcessAudio (const float *pcm, std::size_t num_samples,
                       const std::string &source_id, const Media &media,
                       Retention retention,
                       EvidenceBudget evidence_budget,
                       RetrievalTime retrieval_time)
{
  if (!impl_)
    {
      throw std::runtime_error ("Cortext not initialized");
    }
#if defined(CORTEXT_DISABLE_AUDIO)
  (void)pcm;
  (void)num_samples;
  (void)source_id;
  (void)media;
  (void)retention;
  ThrowAudioSupportDisabled ();
#else
  ValidateAudioInput (pcm);
  ValidateMedia (media);
  telemetry::ScopedSpan span ("cortext.api.process_audio");
  const auto total_start = std::chrono::steady_clock::now ();
  const auto filter_decision
      = impl_->signal_filter.EvaluateAudio (source_id, pcm, num_samples, 16000);
  if (filter_decision.evaluated && !filter_decision.accepted)
    {
      span.SetAttribute ("cortext.signal_filter.modality",
                         filter_decision.modality);
      span.SetAttribute ("cortext.signal_filter.reason",
                         filter_decision.reason);
      span.SetStatusOk ();
      return MakeFilteredContext (filter_decision, total_start);
    }
  std::vector<float> v;
  telemetry::ScopedSpan encode_span ("cortext.encode");
  LateInteractionSet li;
  impl_->EncodeAudioSignal (pcm, num_samples, retention, v, li);
  encode_span.SetStatusOk ();
  const auto encode_end = std::chrono::steady_clock::now ();

  const auto process_start = std::chrono::steady_clock::now ();
  cortext::Signal s;
  s.embedding = ToEigen (RetrievalEmbeddingView (v));
  s.timestamp = impl_->NowMillis ();
  s.source_id = source_id;
  s.retention = retention;
  s.late_interaction = std::move (li);
  s.modality = "audio";
  ApplyStoredMedia (s, media, pcm, num_samples * sizeof (float),
                    MakeAudioMimePcmF32 ());
  s.sample_rate = 16000;
  s.num_samples = num_samples;

  ApplyRetrievalTime (s, retrieval_time);
  auto out = impl_->Process (s);
  const auto process_end = std::chrono::steady_clock::now ();
  span.SetAttribute ("cortext.candidate_memory_count",
                     static_cast<std::int64_t> (out.candidate_memory_ids.size ()));
  span.SetAttribute ("cortext.used_memory_count",
                     static_cast<std::int64_t> (out.used_memory_ids.size ()));
  span.SetStatusOk ();
  telemetry::ScopedSpan hydrate_span ("cortext.hydrate");
  const auto hydrate_start = std::chrono::steady_clock::now ();
  Cortext::Context result = impl_->HydrateContext (out, true, &s.embedding, true, &s,
                                                   evidence_budget);
  const auto hydrate_end = std::chrono::steady_clock::now ();
  hydrate_span.SetStatusOk ();
  result.encode_ms = ToMillis (encode_end - total_start);
  result.process_ms = ToMillis (process_end - process_start);
  result.hydrate_ms = ToMillis (hydrate_end - hydrate_start);
  result.total_ms = ToMillis (hydrate_end - total_start);
  ApplySignalFilterDecision (result.output, filter_decision);
  return result;
#endif
}

Cortext::Context
Cortext::ProcessImage (const std::uint8_t *data, int width, int height,
                       int channels, const std::string &source_id,
                       Retention retention,
                       EvidenceBudget evidence_budget,
                       RetrievalTime retrieval_time)
{
  if (!impl_)
    {
      throw std::runtime_error ("Cortext not initialized");
    }
#if defined(CORTEXT_DISABLE_IMAGE)
  (void)data;
  (void)width;
  (void)height;
  (void)channels;
  (void)source_id;
  (void)retention;
  ThrowImageSupportDisabled ();
#else
  ValidateImageInput (data, width, height, channels);
  telemetry::ScopedSpan span ("cortext.api.process_image");
  const auto total_start = std::chrono::steady_clock::now ();
  const auto filter_decision
      = impl_->signal_filter.EvaluateImage (source_id, data, width,
                                            height, channels);
  if (filter_decision.evaluated && !filter_decision.accepted)
    {
      span.SetStatusOk ();
      return MakeFilteredContext (filter_decision, total_start);
    }
  std::vector<float> v;
  telemetry::ScopedSpan encode_span ("cortext.encode");
  LateInteractionSet li;
  impl_->EncodeImageSignal (data, width, height, channels, retention, v, li);
  encode_span.SetStatusOk ();
  const auto encode_end = std::chrono::steady_clock::now ();

  // Build signal with payload for MemoryStorage
  const auto process_start = std::chrono::steady_clock::now ();
  cortext::Signal s;
  s.embedding = ToEigen (RetrievalEmbeddingView (v));
  s.timestamp = impl_->NowMillis ();
  s.source_id = source_id;
  s.retention = retention;
  s.late_interaction = std::move (li);
  s.modality = "image";
  ApplyStoredMedia (s, Media{}, data, RawImageBytes (width, height, channels),
                    RawImageMime (width, height, channels));
  s.width = width;
  s.height = height;
  s.channels = channels;

  ApplyRetrievalTime (s, retrieval_time);
  auto out = impl_->Process (s);
  const auto process_end = std::chrono::steady_clock::now ();
  span.SetAttribute ("cortext.candidate_memory_count",
                     static_cast<std::int64_t> (out.candidate_memory_ids.size ()));
  span.SetAttribute ("cortext.used_memory_count",
                     static_cast<std::int64_t> (out.used_memory_ids.size ()));
  span.SetStatusOk ();
  telemetry::ScopedSpan hydrate_span ("cortext.hydrate");
  const auto hydrate_start = std::chrono::steady_clock::now ();
  Cortext::Context result = impl_->HydrateContext (out, true, &s.embedding, true, &s,
                                                   evidence_budget);
  const auto hydrate_end = std::chrono::steady_clock::now ();
  hydrate_span.SetStatusOk ();
  result.encode_ms = ToMillis (encode_end - total_start);
  result.process_ms = ToMillis (process_end - process_start);
  result.hydrate_ms = ToMillis (hydrate_end - hydrate_start);
  result.total_ms = ToMillis (hydrate_end - total_start);
  ApplySignalFilterDecision (result.output, filter_decision);
  return result;
#endif
}

Cortext::Context
Cortext::ProcessImage (const std::uint8_t *data, int width, int height,
                       int channels, const std::string &source_id,
                       const Media &media, Retention retention,
                       EvidenceBudget evidence_budget,
                       RetrievalTime retrieval_time)
{
  if (!impl_)
    {
      throw std::runtime_error ("Cortext not initialized");
    }
#if defined(CORTEXT_DISABLE_IMAGE)
  (void)data;
  (void)width;
  (void)height;
  (void)channels;
  (void)source_id;
  (void)media;
  (void)retention;
  ThrowImageSupportDisabled ();
#else
  ValidateImageInput (data, width, height, channels);
  ValidateMedia (media);
  telemetry::ScopedSpan span ("cortext.api.process_image");
  const auto total_start = std::chrono::steady_clock::now ();
  const auto filter_decision
      = impl_->signal_filter.EvaluateImage (source_id, data, width,
                                            height, channels);
  if (filter_decision.evaluated && !filter_decision.accepted)
    {
      span.SetStatusOk ();
      return MakeFilteredContext (filter_decision, total_start);
    }
  std::vector<float> v;
  telemetry::ScopedSpan encode_span ("cortext.encode");
  LateInteractionSet li;
  impl_->EncodeImageSignal (data, width, height, channels, retention, v, li);
  encode_span.SetStatusOk ();
  const auto encode_end = std::chrono::steady_clock::now ();

  const auto process_start = std::chrono::steady_clock::now ();
  cortext::Signal s;
  s.embedding = ToEigen (RetrievalEmbeddingView (v));
  s.timestamp = impl_->NowMillis ();
  s.source_id = source_id;
  s.retention = retention;
  s.late_interaction = std::move (li);
  s.modality = "image";
  ApplyStoredMedia (s, media, data, RawImageBytes (width, height, channels),
                    RawImageMime (width, height, channels));
  s.width = width;
  s.height = height;
  s.channels = channels;

  ApplyRetrievalTime (s, retrieval_time);
  auto out = impl_->Process (s);
  const auto process_end = std::chrono::steady_clock::now ();
  span.SetAttribute ("cortext.candidate_memory_count",
                     static_cast<std::int64_t> (out.candidate_memory_ids.size ()));
  span.SetAttribute ("cortext.used_memory_count",
                     static_cast<std::int64_t> (out.used_memory_ids.size ()));
  span.SetStatusOk ();
  telemetry::ScopedSpan hydrate_span ("cortext.hydrate");
  const auto hydrate_start = std::chrono::steady_clock::now ();
  Cortext::Context result = impl_->HydrateContext (out, true, &s.embedding, true, &s,
                                                   evidence_budget);
  const auto hydrate_end = std::chrono::steady_clock::now ();
  hydrate_span.SetStatusOk ();
  result.encode_ms = ToMillis (encode_end - total_start);
  result.process_ms = ToMillis (process_end - process_start);
  result.hydrate_ms = ToMillis (hydrate_end - hydrate_start);
  result.total_ms = ToMillis (hydrate_end - total_start);
  ApplySignalFilterDecision (result.output, filter_decision);
  return result;
#endif
}

std::vector<float>
Cortext::EmbedText (const std::string &text) const
{
  if (!impl_)
    {
      throw std::runtime_error ("Cortext not initialized");
    }
  std::vector<float> embedding;
  telemetry::ScopedSpan span ("cortext.api.embed_text");
  impl_->encoder->EncodeText (text, embedding);
  span.SetAttribute ("cortext.embedding_dimension",
                     static_cast<std::int64_t> (embedding.size ()));
  span.SetStatusOk ();
  return embedding;
}

std::vector<float>
Cortext::EmbedAudio (const float *pcm, std::size_t num_samples) const
{
  if (!impl_)
    {
      throw std::runtime_error ("Cortext not initialized");
    }
#if defined(CORTEXT_DISABLE_AUDIO)
  (void)pcm;
  (void)num_samples;
  ThrowAudioSupportDisabled ();
#else
  ValidateAudioInput (pcm);
  std::vector<float> embedding;
  telemetry::ScopedSpan span ("cortext.api.embed_audio");
  impl_->encoder->EncodeAudio (pcm, num_samples, embedding);
  span.SetAttribute ("cortext.embedding_dimension",
                     static_cast<std::int64_t> (embedding.size ()));
  span.SetStatusOk ();
  return embedding;
#endif
}

std::vector<float>
Cortext::EmbedImage (const std::uint8_t *data, int width, int height,
                     int channels) const
{
  if (!impl_)
    {
      throw std::runtime_error ("Cortext not initialized");
    }
#if defined(CORTEXT_DISABLE_IMAGE)
  (void)data;
  (void)width;
  (void)height;
  (void)channels;
  ThrowImageSupportDisabled ();
#else
  ValidateImageInput (data, width, height, channels);
  std::vector<float> embedding;
  telemetry::ScopedSpan span ("cortext.api.embed_image");
  impl_->encoder->EncodeImage (data, width, height, channels, embedding);
  span.SetAttribute ("cortext.embedding_dimension",
                     static_cast<std::int64_t> (embedding.size ()));
  span.SetStatusOk ();
  return embedding;
#endif
}

Cortext::Context
internal::ReplayIngress::ProcessTextAt (Cortext &cortext,
                                        const std::string &text,
                                        const std::string &source_id,
                                        std::uint64_t timestamp,
                                        Retention retention,
                                        bool hydrate_context)
{
  if (!cortext.impl_)
    {
      throw std::runtime_error ("Cortext not initialized");
    }
  telemetry::ScopedSpan span ("cortext.internal.replay.process_text_at");
  const auto total_start = std::chrono::steady_clock::now ();
  const auto filter_decision = cortext.impl_->signal_filter.EvaluateText (text);
  if (filter_decision.evaluated && !filter_decision.accepted)
    {
      span.SetStatusOk ();
      return MakeFilteredContext (filter_decision, total_start);
    }
  std::vector<float> v;
  telemetry::ScopedSpan encode_span ("cortext.encode");
  LateInteractionSet li;
  cortext.impl_->EncodeTextSignal (text, retention, v, li);
  encode_span.SetStatusOk ();
  const auto encode_end = std::chrono::steady_clock::now ();

  const auto process_start = std::chrono::steady_clock::now ();
  cortext::Signal s;
  s.embedding = ToEigen (RetrievalEmbeddingView (v));
  s.timestamp = timestamp != 0 ? timestamp : cortext.impl_->NowMillis ();
  s.source_id = source_id;
  s.retention = retention;
  s.late_interaction = std::move (li);
  s.payload = std::vector<unsigned char> (text.begin (), text.end ());
  s.modality = "text";
  s.mimetype = "text/plain";

  operations::retrieval_trace::ScopedCapture retrieval_trace_capture (
      hydrate_context);
  auto out = cortext.impl_->Process (s);
  const auto process_end = std::chrono::steady_clock::now ();
  span.SetAttribute ("cortext.candidate_memory_count",
                     static_cast<std::int64_t> (
                         out.candidate_memory_ids.size ()));
  span.SetAttribute ("cortext.used_memory_count",
                     static_cast<std::int64_t> (out.used_memory_ids.size ()));
  span.SetStatusOk ();

  telemetry::ScopedSpan hydrate_span ("cortext.hydrate");
  const auto hydrate_start = std::chrono::steady_clock::now ();
  Cortext::Context result = cortext.impl_->HydrateContext (
      out, hydrate_context, &s.embedding, hydrate_context);
  const auto hydrate_end = std::chrono::steady_clock::now ();
  hydrate_span.SetStatusOk ();
  result.encode_ms = ToMillis (encode_end - total_start);
  result.process_ms = ToMillis (process_end - process_start);
  result.hydrate_ms = ToMillis (hydrate_end - hydrate_start);
  result.total_ms = ToMillis (hydrate_end - total_start);
  ApplySignalFilterDecision (result.output, filter_decision);
  return result;
}

Cortext::Context
internal::ReplayIngress::ProcessAudioAt (Cortext &cortext, const float *pcm,
                                         std::size_t num_samples,
                                         const std::string &source_id,
                                         std::uint64_t timestamp,
                                         Retention retention)
{
  if (!cortext.impl_)
    {
      throw std::runtime_error ("Cortext not initialized");
    }
#if defined(CORTEXT_DISABLE_AUDIO)
  (void)pcm;
  (void)num_samples;
  (void)source_id;
  (void)timestamp;
  (void)retention;
  ThrowAudioSupportDisabled ();
#else
  ValidateAudioInput (pcm);
  telemetry::ScopedSpan span ("cortext.internal.replay.process_audio_at");
  const auto total_start = std::chrono::steady_clock::now ();
  const auto filter_decision
      = cortext.impl_->signal_filter.EvaluateAudio (source_id, pcm, num_samples, 16000);
  if (filter_decision.evaluated && !filter_decision.accepted)
    {
      span.SetStatusOk ();
      return MakeFilteredContext (filter_decision, total_start);
    }
  std::vector<float> v;
  telemetry::ScopedSpan encode_span ("cortext.encode");
  LateInteractionSet li;
  cortext.impl_->EncodeAudioSignal (pcm, num_samples, retention, v, li);
  encode_span.SetStatusOk ();
  const auto encode_end = std::chrono::steady_clock::now ();

  const auto process_start = std::chrono::steady_clock::now ();
  cortext::Signal s;
  s.embedding = ToEigen (RetrievalEmbeddingView (v));
  s.timestamp = timestamp != 0 ? timestamp : cortext.impl_->NowMillis ();
  s.source_id = source_id;
  s.retention = retention;
  s.late_interaction = std::move (li);
  const std::size_t byte_len = num_samples * sizeof (float);
  s.payload = std::vector<unsigned char> (byte_len);
  std::memcpy (s.payload->data (), pcm, byte_len);
  s.modality = "audio";
  s.mimetype = "audio/pcm;format=f32";
  s.sample_rate = 16000;
  s.num_samples = num_samples;

  auto out = cortext.impl_->Process (s);
  const auto process_end = std::chrono::steady_clock::now ();
  span.SetAttribute ("cortext.candidate_memory_count",
                     static_cast<std::int64_t> (out.candidate_memory_ids.size ()));
  span.SetAttribute ("cortext.used_memory_count",
                     static_cast<std::int64_t> (out.used_memory_ids.size ()));
  span.SetStatusOk ();
  telemetry::ScopedSpan hydrate_span ("cortext.hydrate");
  const auto hydrate_start = std::chrono::steady_clock::now ();
  Cortext::Context result
      = cortext.impl_->HydrateContext (out, true, &s.embedding, true, &s);
  const auto hydrate_end = std::chrono::steady_clock::now ();
  hydrate_span.SetStatusOk ();
  result.encode_ms = ToMillis (encode_end - total_start);
  result.process_ms = ToMillis (process_end - process_start);
  result.hydrate_ms = ToMillis (hydrate_end - hydrate_start);
  result.total_ms = ToMillis (hydrate_end - total_start);
  ApplySignalFilterDecision (result.output, filter_decision);
  return result;
#endif
}

Cortext::Context
internal::ReplayIngress::ProcessImageAt (Cortext &cortext,
                                         const std::uint8_t *data, int width,
                                         int height, int channels,
                                         const std::string &source_id,
                                         std::uint64_t timestamp,
                                         Retention retention)
{
  if (!cortext.impl_)
    {
      throw std::runtime_error ("Cortext not initialized");
    }
#if defined(CORTEXT_DISABLE_IMAGE)
  (void)data;
  (void)width;
  (void)height;
  (void)channels;
  (void)source_id;
  (void)timestamp;
  (void)retention;
  ThrowImageSupportDisabled ();
#else
  ValidateImageInput (data, width, height, channels);
  telemetry::ScopedSpan span ("cortext.internal.replay.process_image_at");
  const auto total_start = std::chrono::steady_clock::now ();
  const auto filter_decision = cortext.impl_->signal_filter.EvaluateImage (
      source_id, data, width, height, channels);
  if (filter_decision.evaluated && !filter_decision.accepted)
    {
      span.SetStatusOk ();
      return MakeFilteredContext (filter_decision, total_start);
    }
  std::vector<float> v;
  telemetry::ScopedSpan encode_span ("cortext.encode");
  LateInteractionSet li;
  cortext.impl_->EncodeImageSignal (data, width, height, channels, retention, v, li);
  encode_span.SetStatusOk ();
  const auto encode_end = std::chrono::steady_clock::now ();

  const auto process_start = std::chrono::steady_clock::now ();
  cortext::Signal s;
  s.embedding = ToEigen (RetrievalEmbeddingView (v));
  s.timestamp = timestamp != 0 ? timestamp : cortext.impl_->NowMillis ();
  s.source_id = source_id;
  s.retention = retention;
  s.late_interaction = std::move (li);
  const std::size_t byte_len
      = static_cast<std::size_t> (width) * static_cast<std::size_t> (height)
        * static_cast<std::size_t> (channels);
  s.payload = std::vector<unsigned char> (byte_len);
  std::memcpy (s.payload->data (), data, byte_len);
  s.modality = "image";
  s.mimetype = "image/raw;width=" + std::to_string (width)
               + ";height=" + std::to_string (height)
               + ";channels=" + std::to_string (channels);
  s.width = width;
  s.height = height;
  s.channels = channels;

  auto out = cortext.impl_->Process (s);
  const auto process_end = std::chrono::steady_clock::now ();
  span.SetAttribute ("cortext.candidate_memory_count",
                     static_cast<std::int64_t> (out.candidate_memory_ids.size ()));
  span.SetAttribute ("cortext.used_memory_count",
                     static_cast<std::int64_t> (out.used_memory_ids.size ()));
  span.SetStatusOk ();
  telemetry::ScopedSpan hydrate_span ("cortext.hydrate");
  const auto hydrate_start = std::chrono::steady_clock::now ();
  Cortext::Context result
      = cortext.impl_->HydrateContext (out, true, &s.embedding, true, &s);
  const auto hydrate_end = std::chrono::steady_clock::now ();
  hydrate_span.SetStatusOk ();
  result.encode_ms = ToMillis (encode_end - total_start);
  result.process_ms = ToMillis (process_end - process_start);
  result.hydrate_ms = ToMillis (hydrate_end - hydrate_start);
  result.total_ms = ToMillis (hydrate_end - total_start);
  ApplySignalFilterDecision (result.output, filter_decision);
  return result;
#endif
}

Cortext::Context
internal::ReplayIngress::ConsolidateAt (Cortext &cortext,
                                        std::uint64_t timestamp)
{
  if (!cortext.impl_)
    {
      throw std::runtime_error ("Cortext not initialized");
    }
  return cortext.impl_->ConsolidateAt (StopToken {}, timestamp);
}

Cortext::Context
Cortext::Consolidate ()
{
  return Consolidate (StopToken {});
}

Cortext::Context
Cortext::Consolidate (StopToken stop_token)
{
  if (!impl_)
    {
      throw std::runtime_error ("Cortext not initialized");
    }
  return impl_->ConsolidateAt (stop_token, impl_->NowMillis ());
}

void
Cortext::Flush ()
{
  if (!impl_)
    {
      return; // Safe no-op if not initialized
    }
  if (impl_->processor)
    {
      telemetry::ScopedSpan span ("cortext.api.flush");
      impl_->processor->Flush ();
      span.SetStatusOk ();
    }
}

void
Cortext::Reset ()
{
  if (!impl_)
    {
      return;
    }
  telemetry::ScopedSpan span ("cortext.api.reset");
  impl_->ResetProcessor ();
  span.SetStatusOk ();
}

// Static helpers (simple canonical mime strings and content key format).
std::string
Cortext::MakeContentKey (long long embedding_id)
{
  return std::string ("mem/") + std::to_string (embedding_id);
}

std::string
Cortext::MakeAudioMimePcmF32 ()
{
  return "audio/pcm;format=f32";
}

std::string
Cortext::MakeImageMimePng ()
{
  return "image/png";
}

namespace internal
{
namespace
{

inline Eigen::VectorXf
NormalizeProbeEmbedding (const Eigen::VectorXf &embedding)
{
  if (embedding.size () == 0)
    {
      return embedding;
    }
  const float norm = embedding.norm ();
  if (norm > 0.0f)
    {
      return embedding / norm;
    }
  return embedding;
}

inline float
ProbeChunkWeight (const std::string &text)
{
  std::size_t count = 0;
  for (const char ch : text)
    {
      if (!std::isspace (static_cast<unsigned char> (ch)))
        {
          ++count;
        }
    }
  return static_cast<float> (std::max<std::size_t> (1, count));
}

} // namespace

struct StreamingTextProbeSession::Impl
{
  Cortext *cortext = nullptr;
  std::string source_id;
  std::unique_ptr<SignalProcessor> processor;
  Eigen::VectorXf aggregate_embedding_sum;
  bool has_cached_embedding = false;

  Impl (Cortext &ctx, std::string source)
      : cortext (&ctx), source_id (std::move (source))
  {
    Reset ();
  }

  Cortext::Context
  EncodeChunk (const std::string &text, std::uint64_t timestamp,
               bool run_probe)
  {
    if (!cortext || !cortext->impl_)
      {
        throw std::runtime_error ("Cortext not initialized");
      }
    if (!processor)
      {
        Reset ();
      }
    if (text.empty ())
      {
        return {};
      }

    const auto total_start = std::chrono::steady_clock::now ();
    std::vector<float> embedding;
    cortext->impl_->encoder->EncodeText (text, embedding);
    const auto encode_end = std::chrono::steady_clock::now ();
    const Eigen::VectorXf eigen_embedding = ToEigen (embedding);
    AccumulateFinalEmbedding (eigen_embedding, text);

    if (!run_probe)
      {
        Cortext::Context result;
        result.encode_ms = ToMillis (encode_end - total_start);
        result.process_ms = 0.0;
        result.hydrate_ms = 0.0;
        result.total_ms = result.encode_ms;
        return result;
      }

    cortext::Signal signal;
    signal.embedding = ToEigen (RetrievalEmbeddingView (embedding));
    signal.timestamp = timestamp != 0 ? timestamp
                                      : cortext->impl_->NowMillis ();
    signal.source_id = source_id;
    signal.payload = std::vector<unsigned char> (text.begin (), text.end ());
    signal.modality = "text";
    signal.mimetype = "text/plain";

    const auto process_start = std::chrono::steady_clock::now ();
    auto out = processor->Process (signal);
    const auto process_end = std::chrono::steady_clock::now ();

    const auto hydrate_start = std::chrono::steady_clock::now ();
    Cortext::Context result
        = cortext->impl_->HydrateContext (out, false, &signal.embedding, true, &signal);
    const auto hydrate_end = std::chrono::steady_clock::now ();

    result.encode_ms = ToMillis (encode_end - total_start);
    result.process_ms = ToMillis (process_end - process_start);
    result.hydrate_ms = ToMillis (hydrate_end - hydrate_start);
    result.total_ms = ToMillis (hydrate_end - total_start);
    return result;
  }

  Cortext::Context
  AppendTextChunkAt (const std::string &text, std::uint64_t timestamp)
  {
    return EncodeChunk (text, timestamp, true);
  }

  Cortext::Context
  CacheTextChunkAt (const std::string &text, std::uint64_t timestamp)
  {
    return EncodeChunk (text, timestamp, false);
  }

  Cortext::Context
  FinalizeTextAt (const std::string &text, std::uint64_t timestamp)
  {
    if (!cortext || !cortext->impl_)
      {
        throw std::runtime_error ("Cortext not initialized");
      }
    if (text.empty () || !has_cached_embedding
        || aggregate_embedding_sum.size () == 0)
      {
        return timestamp != 0
                   ? cortext->ProcessTextAt (text, source_id, timestamp,
                                              Retention::Durable)
                   : cortext->ProcessText (text, source_id,
                                           Retention::Durable);
      }
    const Eigen::VectorXf final_embedding
        = NormalizeProbeEmbedding (aggregate_embedding_sum);
    return cortext->impl_->ProcessTextEmbeddingAt (
        final_embedding, timestamp, source_id, text, Retention::Durable);
  }

  void
  AccumulateFinalEmbedding (const Eigen::VectorXf &embedding,
                            const std::string &text)
  {
    if (embedding.size () == 0)
      {
        return;
      }
    const Eigen::VectorXf weighted
        = NormalizeProbeEmbedding (embedding) * ProbeChunkWeight (text);
    if (!has_cached_embedding
        || aggregate_embedding_sum.size () != weighted.size ())
      {
        aggregate_embedding_sum = Eigen::VectorXf::Zero (weighted.size ());
        has_cached_embedding = true;
      }
    aggregate_embedding_sum += weighted;
  }

  void
  Reset ()
  {
    if (!cortext || !cortext->impl_)
      {
        throw std::runtime_error ("Cortext not initialized");
      }
    processor = cortext->impl_->MakeProbeProcessor ();
    aggregate_embedding_sum = Eigen::VectorXf ();
    has_cached_embedding = false;
  }
};

StreamingTextProbeSession::StreamingTextProbeSession (Cortext &cortext,
                                                      std::string source_id)
    : impl_ (std::make_unique<Impl> (cortext, std::move (source_id)))
{
}

StreamingTextProbeSession::~StreamingTextProbeSession () = default;

StreamingTextProbeSession::StreamingTextProbeSession (
    StreamingTextProbeSession &&) noexcept
    = default;

StreamingTextProbeSession &
StreamingTextProbeSession::operator= (StreamingTextProbeSession &&) noexcept
    = default;

Cortext::Context
StreamingTextProbeSession::AppendTextChunk (const std::string &text)
{
  return AppendTextChunkAt (text, 0);
}

Cortext::Context
StreamingTextProbeSession::AppendTextChunkAt (const std::string &text,
                                              std::uint64_t timestamp)
{
  return impl_->AppendTextChunkAt (text, timestamp);
}

Cortext::Context
StreamingTextProbeSession::CacheTextChunk (const std::string &text)
{
  return CacheTextChunkAt (text, 0);
}

Cortext::Context
StreamingTextProbeSession::CacheTextChunkAt (const std::string &text,
                                             std::uint64_t timestamp)
{
  return impl_->CacheTextChunkAt (text, timestamp);
}

Cortext::Context
StreamingTextProbeSession::FinalizeText (const std::string &text)
{
  return FinalizeTextAt (text, 0);
}

Cortext::Context
StreamingTextProbeSession::FinalizeTextAt (const std::string &text,
                                           std::uint64_t timestamp)
{
  return impl_->FinalizeTextAt (text, timestamp);
}

void
StreamingTextProbeSession::Reset ()
{
  impl_->Reset ();
}

} // namespace internal

#if defined(CORTEXT_TESTING)
Cortext::Context
Cortext::DebugHydrateForTest (const std::vector<long long> &candidate_ids,
                              const std::vector<long long> &used_ids)
{
  cortext::SignalProcessor::Output out;
  out.candidate_memory_ids = candidate_ids;
  out.used_memory_ids = used_ids;
  return impl_->HydrateContext (out);
}

Cortext::Context
Cortext::DebugHydrateForTest (const std::vector<long long> &candidate_ids,
                              const std::vector<long long> &used_ids,
                              const std::vector<float> &query_embedding)
{
  cortext::SignalProcessor::Output out;
  out.candidate_memory_ids = candidate_ids;
  out.used_memory_ids = used_ids;
  Eigen::VectorXf query;
  if (!query_embedding.empty ())
    {
      query = ToEigen (query_embedding);
    }
  return impl_->HydrateContext (out, true,
                                query.size () > 0 ? &query : nullptr);
}
#endif

} // namespace cortext
