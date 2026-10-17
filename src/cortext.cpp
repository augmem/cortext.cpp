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
#include "operations/constructive_recall_internal.hpp"
#include "operations/retrieval_trace_state.hpp"
#include "operations/signal_record_rollback_internal.hpp"
#include "operations/sparse_retrieval_knobs_internal.hpp"
#include "experimental_env.hpp"
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
#include "cortext/operations/soft_anchor.hpp"
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

double
Clamp01 (double value)
{
  return std::clamp (value, 0.0, 1.0);
}

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

#if !defined(CORTEXT_DISABLE_AUDIO) || !defined(CORTEXT_DISABLE_IMAGE)
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

void
ApplyStoredMedia (Signal &signal, const Cortext::Media &media)
{
  ValidateMedia (media);
  if (media.size == 0)
    {
      return;
    }

  signal.payload = std::vector<unsigned char> (media.data,
                                               media.data + media.size);
  signal.mimetype = media.mimetype;
}
#endif

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

std::optional<Eigen::VectorXf>
SoftAnchorEmbeddingView (const std::vector<float> &encoded)
{
  if (encoded.size () >= 1536)
    {
      std::vector<float> out (encoded.begin (), encoded.begin () + 1536);
      NormalizeVector (out);
      return ToEigen (out);
    }
  return std::nullopt;
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
  if (memory_modality == "text" || memory_mime == "text/plain"
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
/// @return true if any blobs were loaded.
bool
LoadSignalBlobs (Store *store, long long memory_id,
                 ObjectStore *object_store,
                 const std::string &memory_modality,
                 const std::string &memory_mime,
                 int max_signal_blobs,
                 std::size_t *loaded_signal_blobs,
                 std::vector<std::vector<unsigned char>> &out,
                 SharedObjectReads *shared = nullptr)
{
  if (!store || memory_id <= 0)
    return false;

  try
    {
      const long long limit = std::max (1, max_signal_blobs);
      long long cursor_position = std::numeric_limits<long long>::max ();
      long long cursor_signal_id = std::numeric_limits<long long>::max ();
      std::vector<std::vector<unsigned char>> matched_payloads;
      matched_payloads.reserve (static_cast<std::size_t> (limit));
      while (static_cast<long long> (matched_payloads.size ()) < limit)
        {
          auto rows = store->Execute (
              "SELECT signal_id, modality, mime, blob_id, "
              "       COALESCE(serial_position, signal_id) AS sort_position "
              "FROM signals "
              "WHERE memory_id = ? AND blob_id IS NOT NULL "
              "  AND (? = '' OR modality = ?) "
              "  AND (COALESCE(serial_position, signal_id) < ? "
              "       OR (COALESCE(serial_position, signal_id) = ? "
              "           AND signal_id < ?)) "
              "ORDER BY sort_position DESC, signal_id DESC "
              "LIMIT ?",
              { memory_id, memory_modality, memory_modality,
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
                                  memory_modality, memory_mime,
                                  signal_modality, signal_mime, payload))
                            {
                              matched_payloads.push_back (
                                  std::move (payload));
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
      if (loaded_signal_blobs)
        *loaded_signal_blobs += matched_payloads.size ();
      out.insert (out.end (),
                  std::make_move_iterator (matched_payloads.begin ()),
                  std::make_move_iterator (matched_payloads.end ()));
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
                 std::vector<std::vector<unsigned char>> &out,
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
      return true;
    }
  return false;
}

void
LoadSoftAnchors (
    Store *store, long long memory_id,
    std::vector<Cortext::Context::Memory::SoftAnchor> &out,
    int max_hydrated_soft_anchors)
{
  out.clear ();
  if (!store || memory_id <= 0)
    return;
  const int limit = std::max (1, max_hydrated_soft_anchors);

  try
    {
      auto rows = store->Execute (
          "SELECT anchor_id, anchor_strength, anchor_label, memory_tier, "
          "       score, margin, entropy "
          "FROM soft_anchor_links "
          "WHERE memory_id = ? "
          "  AND anchor_label NOT IN ('none', 'rejected') "
          "ORDER BY anchor_strength DESC, score DESC, updated_step DESC "
          "LIMIT ?",
          { memory_id, static_cast<long long> (limit) });

      out.reserve (rows.size ());
      for (const auto &row : rows)
        {
          auto get_s = [&row] (const char *k) -> std::string {
            auto it = row.find (k);
            if (it == row.end () || !it->second.has_value ())
              return {};
            if (it->second.type () == typeid (std::string))
              return std::any_cast<std::string> (it->second);
            return {};
          };

          auto get_dbl = [&row] (const char *k, double def = 0.0) -> double {
            auto it = row.find (k);
            if (it == row.end () || !it->second.has_value ())
              return def;
            return store::AnyToDouble (it->second, def);
          };

          Cortext::Context::Memory::SoftAnchor anchor;
          anchor.id = get_s ("anchor_id");
          anchor.strength = Clamp01 (get_dbl ("anchor_strength"));
          anchor.likelihood = anchor.strength;
          anchor.label = get_s ("anchor_label");
          anchor.tier = get_s ("memory_tier");
          anchor.score = Clamp01 (get_dbl ("score"));
          anchor.margin = get_dbl ("margin");
          anchor.entropy = Clamp01 (get_dbl ("entropy"));
          out.push_back (std::move (anchor));
        }
    }
  catch (const std::exception &e)
    {
      telemetry::LogWarn (
          "Failed to load soft anchors",
          { telemetry::Attribute::String ("component", "cortext"),
            telemetry::Attribute::Int64 ("memory_id", memory_id),
            telemetry::Attribute::String ("error", e.what ()) });
    }
  catch (...)
    {
      telemetry::LogWarn (
          "Failed to load soft anchors (unknown error)",
          { telemetry::Attribute::String ("component", "cortext"),
            telemetry::Attribute::Int64 ("memory_id", memory_id) });
    }
}

void
HydrateMemory (Store *store, ObjectStore *object_store, long long id,
               Cortext::Context::Memory &m,
               int max_hydrated_soft_anchors,
               int max_fallback_signal_blobs,
               std::size_t *loaded_fallback_signal_blobs,
               bool load_soft_anchors = true,
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
          "  ) AS signal_mime "
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
                }
            }

          if (m.content.empty ())
            {
              const auto blob_id = get_blob ("blob_id");
              LoadContentBlob (store, object_store, blob_id, m.modality,
                               m.mimetype, m.content, shared);
            }

          if (m.content.empty ())
            {
              // v2 fallback: old databases may only have signal-level blobs.
              LoadSignalBlobs (store, memory_id, object_store, m.modality,
                               m.mimetype, max_fallback_signal_blobs,
                               loaded_fallback_signal_blobs,
                               m.content, shared);
            }

          // v2: Counts are inline on memories table
          m.retrieved_count = get_ll ("retrieved_count");
          m.used_count = get_ll ("used_count");

          // v2: signal_metrics are stored on signals table, not readily available here
          // Set defaults for now - metrics come from signal processing context
          m.relevance = 0.0;
          m.mismatch = 0.0;
          m.surprise = 0.0;
          m.rarity = 0.0;
          m.drift = 0.0;
          m.contradiction = 0.0;
          m.utility = 0.0;
          m.periphery = 0.0;
          m.coverage = 0.0;
          m.salience = 0.0;
          m.valence = 0.5;
          m.arousal = 0.0;
          m.composite_score = 0.0;
          m.threshold_t = 0.0;
          if (load_soft_anchors)
            LoadSoftAnchors (store, memory_id, m.soft_anchors,
                             max_hydrated_soft_anchors);
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
    }
}

// Episodic contiguity completion for retrieved conversational turns.
//
// Recalling one moment of a conversation brings back what was said around it
// (the temporal-contiguity effect of episodic recall), and the effect fades
// with distance. A retrieved turn is rarely self-contained: a question's
// answer, the statement an acknowledgement refers to, or the user message an
// assistant reply was written for lives one or two turns away. For each
// retrieved text turn, the adjacent turns from the same source stream
// (adjacent LONG_TERM rows in write order, within a short time gap) are
// hydrated as extra content blobs of that same retrieved memory, clipped and
// under a per-context byte budget: first the immediate successor and
// predecessor of every anchor in rank order, then, with what budget is left,
// the turns two steps away. A 2026-10 sweep found the old 6 KiB budget and
// lag-2 cap (top 6 anchors) binding; 12 KiB with lag 2 for every anchor raised
// judged sufficiency on Eval 1 by +0.037 (budgets past 12 KiB change nothing). Cost is O(anchors): primary-key range lookups
// bounded by a fixed id span plus one indexed supersession check per
// neighbor, issued as two statements (lag 1, lag 2) for all anchors. Ranking
// and the number of retrieved memories are unchanged.
constexpr std::size_t kContiguityAnchorLimit = 11;
constexpr std::size_t kContiguityNeighborMaxBytes = 480;
constexpr std::size_t kContiguityContextBudgetBytes = 12288;
constexpr long long kContiguityIdScanSpan = 16;
constexpr long long kContiguityMaxGapMs = 30LL * 60LL * 1000LL;

bool
IsTextSurface (const std::string &modality, const std::string &mimetype)
{
  return modality == "text"
         && (mimetype.empty () || mimetype.rfind ("text/", 0) == 0);
}

std::string
SourceStreamOf (const std::string &source_id)
{
  const auto slash = source_id.rfind ('/');
  return slash == std::string::npos ? source_id : source_id.substr (0, slash);
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
                        std::size_t *loaded_fallback_signal_blobs)
      : store_ (store), object_store_ (object_store), shared_ (shared),
        max_fallback_signal_blobs_ (max_fallback_signal_blobs),
        loaded_fallback_signal_blobs_ (loaded_fallback_signal_blobs)
  {
  }

  // Text of a memory to attach to another memory's content: its content
  // blobs joined by spaces and clipped to max_bytes, or empty when the
  // memory has no text surface. Soft anchors are not loaded.
  std::vector<unsigned char>
  AttachedText (long long memory_id, std::size_t max_bytes)
  {
    const Hydrated &memory = Hydrate (memory_id);
    if (!IsTextSurface (memory.modality, memory.mimetype)
        || !memory.has_content)
      {
        return {};
      }
    std::vector<unsigned char> text;
    for (const auto &blob : memory.content)
      {
        if (blob.empty ())
          {
            continue;
          }
        if (!text.empty ())
          {
            text.push_back (' ');
          }
        text.insert (text.end (), blob.begin (), blob.end ());
        if (text.size () > max_bytes)
          {
            break;
          }
      }
    return ClipTextBlob (text, max_bytes);
  }

private:
  struct Hydrated
  {
    std::string modality;
    std::string mimetype;
    std::vector<std::vector<unsigned char> > content;
    bool has_content = false;
    std::size_t fallback_signal_blobs = 0;
  };

  const Hydrated &
  Hydrate (long long memory_id)
  {
    auto it = cache_.find (memory_id);
    if (it == cache_.end ())
      {
        Cortext::Context::Memory memory;
        std::size_t fallback_signal_blobs = 0;
        HydrateMemory (store_, object_store_, memory_id, memory, 1,
                       max_fallback_signal_blobs_, &fallback_signal_blobs,
                       false, shared_);
        Hydrated hydrated;
        hydrated.modality = std::move (memory.modality);
        hydrated.mimetype = std::move (memory.mimetype);
        hydrated.has_content = HasHydratedContent (memory);
        hydrated.content = std::move (memory.content);
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
  std::unordered_map<long long, Hydrated> cache_;
};

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
FindContiguousTurns (Store *store, const std::vector<ContiguityProbe> &probes)
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
      auto &turn = turns[static_cast<std::size_t> (k)];
      turn.position_id = neighbor_id;
      if (RowString (row, "modality") == "text"
          && RowLongLong (row, "corrected") == 0)
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
                       std::unordered_set<long long> &attached_ids)
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
    auto text = hydrations.AttachedText (turn.memory_id,
                                         kContiguityNeighborMaxBytes);
    if (!text.empty () && text.size () <= budget)
      {
        auto &anchor = retrieved[walk.index];
        budget -= text.size ();
        if (forward)
          anchor.content.push_back (std::move (text));
        else
          anchor.content.insert (anchor.content.begin (), std::move (text));
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
          if (anchor.id <= 0 || !IsTextSurface (anchor.modality, anchor.mimetype)
              || !HasHydratedContent (anchor))
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
      auto turns = FindContiguousTurns (store, probes);
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
          turns = FindContiguousTurns (store, probes);
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
                       long long per_anchor)
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
      "        AND m.kind = 'LONG_TERM' AND m.modality = 'text') r "
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
    AttachmentHydrations &hydrations)
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
              || !IsTextSurface (anchor.modality, anchor.mimetype)
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
          store, anchor_ids, kSupersededRecallPerAnchor * 2);
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
              auto text = hydrations.AttachedText (target_id,
                                                   kSupersededRecallMaxBytes);
              if (text.empty () || text.size () > budget)
                {
                  continue;
                }
              budget -= text.size ();
              recalled.push_back (std::move (text));
            }
          // Strongest edge first in rank, but oldest-to-newest in the packet:
          // the recalled predecessors precede the anchor's own content.
          anchor.content.insert (
              anchor.content.begin (),
              std::make_move_iterator (recalled.rbegin ()),
              std::make_move_iterator (recalled.rend ()));
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

void
HydrateWorkingMemoryFromDB (Store *store, ObjectStore *object_store,
                            std::vector<Cortext::Context::Memory> &out,
                            int max_hydrated_soft_anchors,
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
          "       ) AS signal_mime "
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

          const auto reconstruction_blob_id
              = get_blob ("reconstruction_blob_id");
          if (!LoadContentBlob (store, object_store, reconstruction_blob_id,
                                m.modality, m.mimetype, m.content))
            {
              LoadContentBlob (store, object_store, get_blob ("blob_id"),
                               m.modality, m.mimetype, m.content);
            }
          if (m.content.empty ())
            {
              // v2 fallback: old databases may only have signal-level blobs.
              LoadSignalBlobs (store, m.id, object_store, m.modality,
                               m.mimetype, max_fallback_signal_blobs,
                               loaded_fallback_signal_blobs,
                               m.content);
            }
          StripSourceBlobsIfDisabled (m);
          LoadSoftAnchors (store, m.id, m.soft_anchors,
                           max_hydrated_soft_anchors);

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
  using cortext::operations::UpdateSoftAnchor;
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

  using StorageStage = OperationSet<MemoryStorage, UpdateSoftAnchor,
                                    PersistSignalMetrics>;

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
  std::shared_ptr<cortext::Clock> clock;
  std::shared_ptr<cortext::Store> store;
  std::shared_ptr<cortext::ObjectStore> object_store;
  std::unique_ptr<cortext::IOperation> root_operations;
  std::unique_ptr<cortext::SignalProcessor> processor;

  std::string embedding_model_pin;

  Impl (const Config &c, std::shared_ptr<cortext::Store> supplied_store,
        std::shared_ptr<cortext::ObjectStore> supplied_object_store,
        std::shared_ptr<cortext::Clock> supplied_clock)
      : cfg (c), signal_filter (MakeSignalFilterConfig (c)),
        clock (std::move (supplied_clock)),
        store (std::move (supplied_store)),
        object_store (std::move (supplied_object_store))
  {
    if (!store)
      {
        throw std::invalid_argument ("Cortext requires a non-null Store");
      }
    if (!clock)
      {
        clock = std::make_shared<cortext::SystemClock> ();
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

    root_operations = BuildRootOperationSet (false);
    processor = std::make_unique<cortext::SignalProcessor> (
        MakeProcessorConfig (), store, std::move (root_operations),
        object_store);
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
    pcfg.affect_retrieval = cfg.affect_retrieval;
    pcfg.reinforcement_enabled = cfg.reinforcement_enabled;
    pcfg.procedural_enabled = cfg.procedural_enabled;
    pcfg.sequential_edges_enabled = cfg.sequential_edges_enabled;
    pcfg.encoder = encoder.get ();
    pcfg.clock = clock;

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
    s.soft_anchor_embedding = SoftAnchorEmbeddingView (encoded);
    s.timestamp = timestamp != 0 ? timestamp : NowMillis ();
    s.source_id = source_id;
    s.retention = retention;
    s.payload = std::vector<unsigned char> (text.begin (), text.end ());
    s.modality = "text";
    s.mimetype = "text/plain";

    auto out = processor->Process (s);
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
    Cortext::Context result = HydrateContext (out, true, &s.embedding);
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
    s.soft_anchor_embedding = SoftAnchorEmbeddingView (encoded);
    s.timestamp = timestamp;
    s.source_id = source_id;
    s.force_consolidation = force_consolidation;
    return processor->Process (s);
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
                  bool hydrate_retrieved_memory = true)
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
  result.output.soft_anchor_enabled = out.soft_anchor_enabled;
  result.output.soft_anchor_state_count = out.soft_anchor_state_count;
  result.output.soft_anchor_link_count = out.soft_anchor_link_count;
  result.output.soft_anchor_create_count = out.soft_anchor_create_count;
  result.output.soft_anchor_update_count = out.soft_anchor_update_count;
  result.output.soft_anchor_none_count = out.soft_anchor_none_count;
  result.output.soft_anchor_last_update_us = out.soft_anchor_last_update_us;
  result.output.soft_anchor_mean_update_us = out.soft_anchor_mean_update_us;
    
	    if (!store)
	      {
	        return result;
	      }

	    const int max_hydrated_soft_anchors = core::HydratedSoftAnchorLimit (
	        cfg.focus, cfg.sensitivity, cfg.stability);
	    const int max_fallback_signal_blobs
	        = operations::sparse_retrieval_knobs_internal::
	            HydrationFallbackSignalLimit (
	            cfg.focus, cfg.sensitivity, cfg.stability);
	    std::size_t loaded_fallback_signal_blobs = 0;

	    if (hydrate_working_memory)
	      {
	        HydrateWorkingMemoryFromDB (store.get (), object_store.get (),
	                                    result.working_memory,
	                                    max_hydrated_soft_anchors,
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
                       max_hydrated_soft_anchors,
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
        max_fallback_signal_blobs, &loaded_fallback_signal_blobs);
    const auto anchor_rows = SourceBlobsDisabled ()
        ? std::unordered_map<long long, AnchorRow> ()
        : LoadAnchorRows (store.get (), result.retrieved_memory,
                          std::max (kContiguityAnchorLimit,
                                    kSupersededRecallAnchorLimit));
    std::unordered_set<long long> contiguous_turn_ids;
    const std::size_t contiguous_turns_attached = AttachContiguousTurns (
        store.get (), result.retrieved_memory, anchor_rows,
        attachment_hydrations, contiguous_turn_ids);
    result.output.operation_ms["Cortext.contiguous_turns_attached"]
        = static_cast<double> (contiguous_turns_attached);
    const std::size_t superseded_predecessors_attached
        = AttachSupersededPredecessors (
            store.get (), result.retrieved_memory, anchor_rows,
            std::move (contiguous_turn_ids), attachment_hydrations);
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
    result.output.operation_ms["Cortext.superseded_predecessors_attached"]
        = static_cast<double> (superseded_predecessors_attached);
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
                      Retention retention)
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
  impl_->encoder->EncodeText (text, v);
  encode_span.SetStatusOk ();
  const auto encode_end = std::chrono::steady_clock::now ();

  // Build signal with payload for MemoryStorage
  const auto process_start = std::chrono::steady_clock::now ();
  cortext::Signal s;
  s.embedding = ToEigen (RetrievalEmbeddingView (v));
  s.soft_anchor_embedding = SoftAnchorEmbeddingView (v);
  s.timestamp = impl_->NowMillis ();
  s.source_id = source_id;
  s.retention = retention;
  s.payload = std::vector<unsigned char> (text.begin (), text.end ());
  s.modality = "text";
  s.mimetype = "text/plain";

  auto out = impl_->processor->Process (s);
  const auto process_end = std::chrono::steady_clock::now ();
  span.SetAttribute ("cortext.candidate_memory_count",
                     static_cast<std::int64_t> (out.candidate_memory_ids.size ()));
  span.SetAttribute ("cortext.used_memory_count",
                     static_cast<std::int64_t> (out.used_memory_ids.size ()));
  span.SetStatusOk ();
  telemetry::ScopedSpan hydrate_span ("cortext.hydrate");
  const auto hydrate_start = std::chrono::steady_clock::now ();
  Cortext::Context result = impl_->HydrateContext (out, true, &s.embedding);
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
                        std::uint64_t timestamp, Retention retention)
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
  impl_->encoder->EncodeText (text, v);
  encode_span.SetStatusOk ();
  const auto encode_end = std::chrono::steady_clock::now ();

  // Build signal with payload for MemoryStorage
  const auto process_start = std::chrono::steady_clock::now ();
  cortext::Signal s;
  s.embedding = ToEigen (RetrievalEmbeddingView (v));
  s.soft_anchor_embedding = SoftAnchorEmbeddingView (v);
  s.timestamp = timestamp;
  s.source_id = source_id;
  s.retention = retention;
  s.payload = std::vector<unsigned char> (text.begin (), text.end ());
  s.modality = "text";
  s.mimetype = "text/plain";

  auto out = impl_->processor->Process (s);
  const auto process_end = std::chrono::steady_clock::now ();
  span.SetAttribute ("cortext.candidate_memory_count",
                     static_cast<std::int64_t> (out.candidate_memory_ids.size ()));
  span.SetAttribute ("cortext.used_memory_count",
                     static_cast<std::int64_t> (out.used_memory_ids.size ()));
  span.SetStatusOk ();
  telemetry::ScopedSpan hydrate_span ("cortext.hydrate");
  const auto hydrate_start = std::chrono::steady_clock::now ();
  Cortext::Context result = impl_->HydrateContext (out, true, &s.embedding);
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
                       const std::string &source_id, Retention retention)
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
      = impl_->signal_filter.EvaluateAudio (pcm, num_samples, 16000);
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
  impl_->encoder->EncodeAudio (pcm, num_samples, v);
  encode_span.SetStatusOk ();
  const auto encode_end = std::chrono::steady_clock::now ();

  // Build signal with payload for MemoryStorage
  const auto process_start = std::chrono::steady_clock::now ();
  cortext::Signal s;
  s.embedding = ToEigen (RetrievalEmbeddingView (v));
  s.soft_anchor_embedding = SoftAnchorEmbeddingView (v);
  s.timestamp = impl_->NowMillis ();
  s.source_id = source_id;
  s.retention = retention;
  // Store raw PCM bytes (f32le) as payload
  const std::size_t byte_len = num_samples * sizeof (float);
  s.payload = std::vector<unsigned char> (byte_len);
  std::memcpy (s.payload->data (), pcm, byte_len);
  s.modality = "audio";
  s.mimetype = MakeAudioMimePcmF32 ();
  s.sample_rate = 16000; // Public audio ingress contract is 16 kHz mono PCM.
  s.num_samples = num_samples;

  auto out = impl_->processor->Process (s);
  const auto process_end = std::chrono::steady_clock::now ();
  span.SetAttribute ("cortext.candidate_memory_count",
                     static_cast<std::int64_t> (out.candidate_memory_ids.size ()));
  span.SetAttribute ("cortext.used_memory_count",
                     static_cast<std::int64_t> (out.used_memory_ids.size ()));
  span.SetStatusOk ();
  telemetry::ScopedSpan hydrate_span ("cortext.hydrate");
  const auto hydrate_start = std::chrono::steady_clock::now ();
  Cortext::Context result = impl_->HydrateContext (out, true, &s.embedding);
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
                       Retention retention)
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
      = impl_->signal_filter.EvaluateAudio (pcm, num_samples, 16000);
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
  impl_->encoder->EncodeAudio (pcm, num_samples, v);
  encode_span.SetStatusOk ();
  const auto encode_end = std::chrono::steady_clock::now ();

  const auto process_start = std::chrono::steady_clock::now ();
  cortext::Signal s;
  s.embedding = ToEigen (RetrievalEmbeddingView (v));
  s.soft_anchor_embedding = SoftAnchorEmbeddingView (v);
  s.timestamp = impl_->NowMillis ();
  s.source_id = source_id;
  s.retention = retention;
  s.modality = "audio";
  ApplyStoredMedia (s, media);
  s.sample_rate = 16000;
  s.num_samples = num_samples;

  auto out = impl_->processor->Process (s);
  const auto process_end = std::chrono::steady_clock::now ();
  span.SetAttribute ("cortext.candidate_memory_count",
                     static_cast<std::int64_t> (out.candidate_memory_ids.size ()));
  span.SetAttribute ("cortext.used_memory_count",
                     static_cast<std::int64_t> (out.used_memory_ids.size ()));
  span.SetStatusOk ();
  telemetry::ScopedSpan hydrate_span ("cortext.hydrate");
  const auto hydrate_start = std::chrono::steady_clock::now ();
  Cortext::Context result = impl_->HydrateContext (out, true, &s.embedding);
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
                       Retention retention)
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
      = impl_->signal_filter.EvaluateImage (data, width, height, channels);
  if (filter_decision.evaluated && !filter_decision.accepted)
    {
      span.SetStatusOk ();
      return MakeFilteredContext (filter_decision, total_start);
    }
  std::vector<float> v;
  telemetry::ScopedSpan encode_span ("cortext.encode");
  impl_->encoder->EncodeImage (data, width, height, channels, v);
  encode_span.SetStatusOk ();
  const auto encode_end = std::chrono::steady_clock::now ();

  // Build signal with payload for MemoryStorage
  const auto process_start = std::chrono::steady_clock::now ();
  cortext::Signal s;
  s.embedding = ToEigen (RetrievalEmbeddingView (v));
  s.soft_anchor_embedding = SoftAnchorEmbeddingView (v);
  s.timestamp = impl_->NowMillis ();
  s.source_id = source_id;
  s.retention = retention;
  // Store raw image bytes as payload
  const std::size_t byte_len
      = static_cast<std::size_t> (width) * static_cast<std::size_t> (height)
        * static_cast<std::size_t> (channels);
  s.payload = std::vector<unsigned char> (byte_len);
  std::memcpy (s.payload->data (), data, byte_len);
  s.modality = "image";
  s.mimetype = RawImageMime (width, height, channels);
  s.width = width;
  s.height = height;
  s.channels = channels;

  auto out = impl_->processor->Process (s);
  const auto process_end = std::chrono::steady_clock::now ();
  span.SetAttribute ("cortext.candidate_memory_count",
                     static_cast<std::int64_t> (out.candidate_memory_ids.size ()));
  span.SetAttribute ("cortext.used_memory_count",
                     static_cast<std::int64_t> (out.used_memory_ids.size ()));
  span.SetStatusOk ();
  telemetry::ScopedSpan hydrate_span ("cortext.hydrate");
  const auto hydrate_start = std::chrono::steady_clock::now ();
  Cortext::Context result = impl_->HydrateContext (out, true, &s.embedding);
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
                       const Media &media, Retention retention)
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
      = impl_->signal_filter.EvaluateImage (data, width, height, channels);
  if (filter_decision.evaluated && !filter_decision.accepted)
    {
      span.SetStatusOk ();
      return MakeFilteredContext (filter_decision, total_start);
    }
  std::vector<float> v;
  telemetry::ScopedSpan encode_span ("cortext.encode");
  impl_->encoder->EncodeImage (data, width, height, channels, v);
  encode_span.SetStatusOk ();
  const auto encode_end = std::chrono::steady_clock::now ();

  const auto process_start = std::chrono::steady_clock::now ();
  cortext::Signal s;
  s.embedding = ToEigen (RetrievalEmbeddingView (v));
  s.soft_anchor_embedding = SoftAnchorEmbeddingView (v);
  s.timestamp = impl_->NowMillis ();
  s.source_id = source_id;
  s.retention = retention;
  s.modality = "image";
  ApplyStoredMedia (s, media);
  s.width = width;
  s.height = height;
  s.channels = channels;

  auto out = impl_->processor->Process (s);
  const auto process_end = std::chrono::steady_clock::now ();
  span.SetAttribute ("cortext.candidate_memory_count",
                     static_cast<std::int64_t> (out.candidate_memory_ids.size ()));
  span.SetAttribute ("cortext.used_memory_count",
                     static_cast<std::int64_t> (out.used_memory_ids.size ()));
  span.SetStatusOk ();
  telemetry::ScopedSpan hydrate_span ("cortext.hydrate");
  const auto hydrate_start = std::chrono::steady_clock::now ();
  Cortext::Context result = impl_->HydrateContext (out, true, &s.embedding);
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
  cortext.impl_->encoder->EncodeText (text, v);
  encode_span.SetStatusOk ();
  const auto encode_end = std::chrono::steady_clock::now ();

  const auto process_start = std::chrono::steady_clock::now ();
  cortext::Signal s;
  s.embedding = ToEigen (RetrievalEmbeddingView (v));
  s.soft_anchor_embedding = SoftAnchorEmbeddingView (v);
  s.timestamp = timestamp != 0 ? timestamp : cortext.impl_->NowMillis ();
  s.source_id = source_id;
  s.retention = retention;
  s.payload = std::vector<unsigned char> (text.begin (), text.end ());
  s.modality = "text";
  s.mimetype = "text/plain";

  operations::retrieval_trace::ScopedCapture retrieval_trace_capture (
      hydrate_context);
  auto out = cortext.impl_->processor->Process (s);
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
      = cortext.impl_->signal_filter.EvaluateAudio (pcm, num_samples, 16000);
  if (filter_decision.evaluated && !filter_decision.accepted)
    {
      span.SetStatusOk ();
      return MakeFilteredContext (filter_decision, total_start);
    }
  std::vector<float> v;
  telemetry::ScopedSpan encode_span ("cortext.encode");
  cortext.impl_->encoder->EncodeAudio (pcm, num_samples, v);
  encode_span.SetStatusOk ();
  const auto encode_end = std::chrono::steady_clock::now ();

  const auto process_start = std::chrono::steady_clock::now ();
  cortext::Signal s;
  s.embedding = ToEigen (RetrievalEmbeddingView (v));
  s.soft_anchor_embedding = SoftAnchorEmbeddingView (v);
  s.timestamp = timestamp != 0 ? timestamp : cortext.impl_->NowMillis ();
  s.source_id = source_id;
  s.retention = retention;
  const std::size_t byte_len = num_samples * sizeof (float);
  s.payload = std::vector<unsigned char> (byte_len);
  std::memcpy (s.payload->data (), pcm, byte_len);
  s.modality = "audio";
  s.mimetype = "audio/pcm;format=f32";
  s.sample_rate = 16000;
  s.num_samples = num_samples;

  auto out = cortext.impl_->processor->Process (s);
  const auto process_end = std::chrono::steady_clock::now ();
  span.SetAttribute ("cortext.candidate_memory_count",
                     static_cast<std::int64_t> (out.candidate_memory_ids.size ()));
  span.SetAttribute ("cortext.used_memory_count",
                     static_cast<std::int64_t> (out.used_memory_ids.size ()));
  span.SetStatusOk ();
  telemetry::ScopedSpan hydrate_span ("cortext.hydrate");
  const auto hydrate_start = std::chrono::steady_clock::now ();
  Cortext::Context result
      = cortext.impl_->HydrateContext (out, true, &s.embedding);
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
      data, width, height, channels);
  if (filter_decision.evaluated && !filter_decision.accepted)
    {
      span.SetStatusOk ();
      return MakeFilteredContext (filter_decision, total_start);
    }
  std::vector<float> v;
  telemetry::ScopedSpan encode_span ("cortext.encode");
  cortext.impl_->encoder->EncodeImage (data, width, height, channels, v);
  encode_span.SetStatusOk ();
  const auto encode_end = std::chrono::steady_clock::now ();

  const auto process_start = std::chrono::steady_clock::now ();
  cortext::Signal s;
  s.embedding = ToEigen (RetrievalEmbeddingView (v));
  s.soft_anchor_embedding = SoftAnchorEmbeddingView (v);
  s.timestamp = timestamp != 0 ? timestamp : cortext.impl_->NowMillis ();
  s.source_id = source_id;
  s.retention = retention;
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

  auto out = cortext.impl_->processor->Process (s);
  const auto process_end = std::chrono::steady_clock::now ();
  span.SetAttribute ("cortext.candidate_memory_count",
                     static_cast<std::int64_t> (out.candidate_memory_ids.size ()));
  span.SetAttribute ("cortext.used_memory_count",
                     static_cast<std::int64_t> (out.used_memory_ids.size ()));
  span.SetStatusOk ();
  telemetry::ScopedSpan hydrate_span ("cortext.hydrate");
  const auto hydrate_start = std::chrono::steady_clock::now ();
  Cortext::Context result
      = cortext.impl_->HydrateContext (out, true, &s.embedding);
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
    signal.soft_anchor_embedding = SoftAnchorEmbeddingView (embedding);
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
        = cortext->impl_->HydrateContext (out, false, &signal.embedding);
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
