#include "cortext/core/knobs.hpp"
#include "cortext/processor.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/store.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include "cortext/store/schema.hpp"
#include "cortext/store/utils.hpp"
#include <algorithm>
#include <chrono>
#include <any>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

namespace cortext
{

namespace
{

void
AssembleOutputMemories (const OperationContext &op_context,
                        SignalProcessor::Output &out)
{
  const auto &cands = op_context.GetRetrievedMemoryEmbeddings ();
  out.candidate_memory_ids.reserve (cands.size ());
  for (const auto &kv : cands)
    {
      out.candidate_memory_ids.push_back (kv.first);
    }
  for (const auto &e : op_context.GetMemoryUsageEvents ())
    {
      if (e.used)
        {
          out.used_memory_ids.push_back (
              static_cast<long long> (e.embedding_id));
        }
    }
}

void
AssembleOutputFields (const OperationContext &op_context,
                      SignalProcessor::Output &out)
{
  out.interrupt_allowed = op_context.GetInterruptAllowed ();
  out.at_boundary = op_context.GetAtBoundary ();
  out.write_decision = op_context.GetWriteDecision ();
  out.threshold_T_dynamic = op_context.GetThresholdTDynamic ();
  out.threshold_hysteresis = op_context.GetThresholdHysteresis ();
  out.effective_focus = op_context.GetEffectiveFocus ();
  out.emotion_intensity = op_context.GetEmotionIntensity ();
  out.valence = op_context.GetValence ();
  out.arousal = op_context.GetArousal ();
  out.mni_jaccard = op_context.GetMniJaccard ();
  out.mni_best_mu = op_context.GetMniBestMu ();
  out.mni_dup_thresh = op_context.GetMniDupThresh ();
  out.mni_tau_jaccard_eff = op_context.GetMniTauJaccardEff ();
  out.mni_tau_mu_eff = op_context.GetMniTauMuEff ();
  out.composite_score = op_context.GetCompositeScore ();
  out.serial_position_multiplier = op_context.GetSerialPositionMultiplier ();
  out.metrics = op_context.GetAllMetrics ();
  out.stored_embedding_id = op_context.GetStoredEmbeddingId ();
}

const char *
GetMetricName (operations::Metric metric)
{
  switch (metric)
    {
    case operations::Metric::relevance:
      return "relevance";
    case operations::Metric::mismatch:
      return "mismatch";
    case operations::Metric::surprise:
      return "surprise";
    case operations::Metric::rarity:
      return "rarity";
    case operations::Metric::drift:
      return "drift";
    case operations::Metric::contradiction:
      return "contradiction";
    case operations::Metric::utility:
      return "utility";
    case operations::Metric::periphery:
      return "periphery";
    case operations::Metric::coverage:
      return "coverage";
    case operations::Metric::salience:
      return "salience";
    case operations::Metric::valence:
      return "valence";
    case operations::Metric::arousal:
      return "arousal";

    case operations::Metric::focus_spread:
      return "focus_spread";
    case operations::Metric::drift_mag:
      return "drift_mag";
    case operations::Metric::aw_prev:
      return "aw_prev";
    case operations::Metric::rate_prev:
      return "rate_prev";
    case operations::Metric::hys_prev:
      return "hys_prev";
    case operations::Metric::embedding_surprisal:
      return "embedding_surprisal";
    default:
      return "unknown";
    }
}

void
LogMetricTelemetry (const std::unordered_map<operations::Metric, double> &metrics)
{
  for (const auto &kv : metrics)
    {
      const char *metric_name = GetMetricName (kv.first);
      const std::string metric_full_name
          = std::string ("cortext.metric.") + metric_name;
      telemetry::RecordHistogram (metric_full_name, kv.second);
    }
}

void
LogProcessTelemetry (const OperationContext &op_context,
                     const SignalProcessor::Output &out)
{
  telemetry::RecordHistogram ("cortext.threshold_T_dynamic",
                              op_context.GetThresholdTDynamic ());
  telemetry::RecordHistogram ("cortext.threshold_hysteresis",
                              op_context.GetThresholdHysteresis ());
  telemetry::RecordHistogram ("cortext.effective_focus",
                              op_context.GetEffectiveFocus ());
  telemetry::RecordHistogram ("cortext.coherence", op_context.GetCoherence ());
  telemetry::RecordHistogram ("cortext.emotion_intensity",
                              op_context.GetEmotionIntensity ());
  telemetry::RecordHistogram ("cortext.mni_jaccard", op_context.GetMniJaccard ());
  telemetry::RecordHistogram ("cortext.mni_best_mu", op_context.GetMniBestMu ());
  telemetry::RecordHistogram ("cortext.mni_dup_thresh",
                              op_context.GetMniDupThresh ());
  telemetry::RecordHistogram ("cortext.last_weight_sum",
                              op_context.GetLastWeightSum ());
  telemetry::RecordHistogram ("cortext.last_effective_metric_count",
                              static_cast<double> (
                                  op_context.GetLastEffectiveMetricCount ()));
  LogMetricTelemetry (out.metrics);
}

// --- Helpers for vector/matrix conversion ---

inline std::vector<float>
ToFloatVector (const Eigen::VectorXf &v)
{
  return std::vector<float> (v.data (), v.data () + v.size ());
}

inline std::vector<char>
SerializeUint64Vector (const std::vector<uint64_t> &values)
{
  std::vector<char> blob (values.size () * sizeof (uint64_t));
  if (!values.empty ())
    {
      std::memcpy (blob.data (), values.data (), blob.size ());
    }
  return blob;
}

inline std::vector<uint64_t>
DeserializeUint64Vector (const std::any &blob)
{
  const uint64_t *data = nullptr;
  size_t count = 0;

  if (blob.type () == typeid (std::vector<char>))
    {
      const auto &vec = std::any_cast<const std::vector<char> &> (blob);
      data = reinterpret_cast<const uint64_t *> (vec.data ());
      count = vec.size () / sizeof (uint64_t);
    }
  else if (blob.type () == typeid (std::vector<unsigned char>))
    {
      const auto &vec
          = std::any_cast<const std::vector<unsigned char> &> (blob);
      data = reinterpret_cast<const uint64_t *> (vec.data ());
      count = vec.size () / sizeof (uint64_t);
    }

  if (!data || count == 0)
    return {};
  return std::vector<uint64_t> (data, data + count);
}

inline Eigen::VectorXf
BlobToEigen (const std::any &blob)
{
  const float *data = nullptr;
  size_t size = 0;

  if (blob.type () == typeid (std::vector<char>))
    {
      const auto &vec = std::any_cast<const std::vector<char> &> (blob);
      data = reinterpret_cast<const float *> (vec.data ());
      size = vec.size () / sizeof (float);
    }
  else if (blob.type () == typeid (std::vector<unsigned char>))
    {
      const auto &vec
          = std::any_cast<const std::vector<unsigned char> &> (blob);
      data = reinterpret_cast<const float *> (vec.data ());
      size = vec.size () / sizeof (float);
    }
  else
    {
      return Eigen::VectorXf ();
    }

  if (size == 0)
    return Eigen::VectorXf ();

  Eigen::VectorXf result (static_cast<Eigen::Index> (size));
  std::memcpy (result.data (), data, size * sizeof (float));
  return result;
}

inline std::vector<char>
SerializeEmbeddingWindow (const std::vector<Eigen::VectorXf> &window)
{
  std::uint32_t dim = 0;
  std::uint32_t count = 0;
  for (const auto &emb : window)
    {
      if (emb.size () > 0)
        {
          dim = static_cast<std::uint32_t> (emb.size ());
          break;
        }
    }
  if (dim == 0)
    {
      return {};
    }

  std::vector<float> data;
  data.reserve (window.size () * static_cast<size_t> (dim));
  for (const auto &emb : window)
    {
      if (emb.size () != static_cast<Eigen::Index> (dim))
        {
          continue;
        }
      data.insert (data.end (), emb.data (), emb.data () + emb.size ());
      ++count;
    }

  if (count == 0)
    {
      return {};
    }

  const size_t header_bytes = sizeof (std::uint32_t) * 2;
  const size_t payload_bytes = static_cast<size_t> (count) * dim * sizeof (float);
  std::vector<char> blob (header_bytes + payload_bytes);
  std::memcpy (blob.data (), &count, sizeof (std::uint32_t));
  std::memcpy (blob.data () + sizeof (std::uint32_t), &dim, sizeof (std::uint32_t));
  std::memcpy (blob.data () + header_bytes, data.data (), payload_bytes);
  return blob;
}

inline std::vector<Eigen::VectorXf>
DeserializeEmbeddingWindow (const std::any &blob)
{
  const char *data = nullptr;
  size_t size = 0;

  if (blob.type () == typeid (std::vector<char>))
    {
      const auto &vec = std::any_cast<const std::vector<char> &> (blob);
      data = vec.data ();
      size = vec.size ();
    }
  else if (blob.type () == typeid (std::vector<unsigned char>))
    {
      const auto &vec
          = std::any_cast<const std::vector<unsigned char> &> (blob);
      data = reinterpret_cast<const char *> (vec.data ());
      size = vec.size ();
    }

  if (!data || size < sizeof (std::uint32_t) * 2)
    {
      return {};
    }

  std::uint32_t count = 0;
  std::uint32_t dim = 0;
  std::memcpy (&count, data, sizeof (std::uint32_t));
  std::memcpy (&dim, data + sizeof (std::uint32_t), sizeof (std::uint32_t));

  if (count == 0 || dim == 0)
    {
      return {};
    }

  const size_t header_bytes = sizeof (std::uint32_t) * 2;
  const size_t expected = header_bytes + static_cast<size_t> (count) * dim * sizeof (float);
  if (size < header_bytes + sizeof (float) * dim)
    {
      return {};
    }

  if (size < expected)
    {
      const size_t available = size - header_bytes;
      count = static_cast<std::uint32_t> (available / (dim * sizeof (float)));
      if (count == 0)
        {
          return {};
        }
    }

  std::vector<Eigen::VectorXf> window;
  window.reserve (count);
  const char *cursor = data + header_bytes;
  for (std::uint32_t i = 0; i < count; ++i)
    {
      Eigen::VectorXf emb (static_cast<Eigen::Index> (dim));
      std::memcpy (emb.data (), cursor, static_cast<size_t> (dim) * sizeof (float));
      cursor += static_cast<size_t> (dim) * sizeof (float);
      window.push_back (std::move (emb));
    }
  return window;
}

// Serialize blender P matrix (2D vector of doubles) to flat float vector
inline std::vector<float>
SerializeMatrix (const std::vector<std::vector<double> > &mat)
{
  std::vector<float> out;
  for (const auto &row : mat)
    {
      for (double d : row)
        out.push_back (static_cast<float> (d));
    }
  return out;
}

// Deserialize flat float vector to 2D matrix
inline std::vector<std::vector<double> >
DeserializeMatrix (const std::any &blob, size_t n)
{
  const float *data = nullptr;
  size_t byte_size = 0;

  if (blob.type () == typeid (std::vector<char>))
    {
      const auto &vec = std::any_cast<const std::vector<char> &> (blob);
      data = reinterpret_cast<const float *> (vec.data ());
      byte_size = vec.size ();
    }
  else if (blob.type () == typeid (std::vector<unsigned char>))
    {
      const auto &vec
          = std::any_cast<const std::vector<unsigned char> &> (blob);
      data = reinterpret_cast<const float *> (vec.data ());
      byte_size = vec.size ();
    }
  else
    {
      return {};
    }

  const size_t expected = n * n * sizeof (float);
  if (byte_size != expected)
    return {};

  std::vector<std::vector<double> > result (n, std::vector<double> (n));
  size_t idx = 0;
  for (size_t i = 0; i < n; ++i)
    {
      for (size_t j = 0; j < n; ++j)
        {
          result[i][j] = static_cast<double> (data[idx++]);
        }
    }
  return result;
}

// Helper to extract int64 from std::any
inline int64_t
ExtractInt64 (const std::map<std::string, std::any> &row,
              const std::string &key, int64_t default_val = 0)
{
  auto it = row.find (key);
  if (it == row.end ())
    return default_val;
  const std::any &v = it->second;
  if (v.type () == typeid (long long))
    return static_cast<int64_t> (std::any_cast<long long> (v));
  if (v.type () == typeid (int64_t))
    return std::any_cast<int64_t> (v);
  if (v.type () == typeid (int))
    return static_cast<int64_t> (std::any_cast<int> (v));
  return default_val;
}

// Helper to extract double from std::any
inline double
ExtractDouble (const std::map<std::string, std::any> &row,
               const std::string &key, double default_val = 0.0)
{
  auto it = row.find (key);
  if (it == row.end ())
    return default_val;
  const std::any &v = it->second;
  if (v.type () == typeid (double))
    return std::any_cast<double> (v);
  if (v.type () == typeid (float))
    return static_cast<double> (std::any_cast<float> (v));
  if (v.type () == typeid (long long))
    return static_cast<double> (std::any_cast<long long> (v));
  if (v.type () == typeid (int))
    return static_cast<double> (std::any_cast<int> (v));
  return default_val;
}

// --- State Loading Functions ---
// v2 Schema: Unified state table replaces processor_state + blender tables

bool
LoadState (Store &store, ProcessorContext &ctx)
{
  try
    {
      auto rows = store.Execute ("SELECT * FROM state WHERE id = 1");
      if (rows.empty ())
        return false; // No persisted state, use defaults

      const auto &row = rows[0];

      // === Processor state fields ===
      ctx.signals_processed
          = static_cast<int> (ExtractInt64 (row, "signals_processed", 0));
      ctx.u_t = ExtractDouble (row, "u_uncertainty", 0.0);
      ctx.weight_relevance = ExtractDouble (row, "weight_relevance", 0.5);
      ctx.attention_width = ExtractDouble (row, "attention_width", 1.57);
      ctx.coverage_gain_floor
          = ExtractDouble (row, "coverage_gain_floor", 0.65);
      ctx.mismatch_weight = ExtractDouble (row, "mismatch_weight", 0.5);
      ctx.T_dynamic = ExtractDouble (row, "theta_dynamic", 0.2);
      ctx.T_target = ExtractDouble (row, "theta_target", ctx.T_dynamic);
      ctx.hysteresis = ExtractDouble (row, "hysteresis", 0.05);
      ctx.half_life = ExtractDouble (row, "half_life", 120.0);
      ctx.rate_target = ExtractDouble (row, "rate_target", 0.2);
      ctx.delta_half_life_adj
          = ExtractDouble (row, "delta_half_life_adj", 0.0);
      ctx.sustained_influence
          = ExtractDouble (row, "sustained_influence", 0.0);
      ctx.last_signal_timestamp
          = static_cast<uint64_t> (ExtractInt64 (row, "last_signal_timestamp", 0));
      ctx.episode_start_ts
          = static_cast<uint64_t> (ExtractInt64 (row, "episode_start_ts", 0));
      ctx.last_interrupt_tick
          = static_cast<int> (ExtractInt64 (row, "last_interrupt_tick",
                                            ctx.last_interrupt_tick));
      ctx.last_retrieval_ts
          = static_cast<uint64_t> (ExtractInt64 (row, "last_retrieval_ts", 0));
      ctx.last_consolidation_ts
          = static_cast<uint64_t> (ExtractInt64 (row, "last_consolidation_ts", 0));
      ctx.consolidation_count
          = static_cast<int> (ExtractInt64 (row, "consolidation_count", 0));
      ctx.is_processing_signal
          = ExtractInt64 (row, "is_processing_signal", 0) != 0;

      ctx.wm_last_accepted
          = ExtractInt64 (row, "wm_last_accepted", 0) != 0;
      ctx.wm_last_chunked
          = ExtractInt64 (row, "wm_last_chunked", 0) != 0;

      ctx.fok_state = ExtractDouble (row, "fok_state", 0.0);
      ctx.retrieval_strength
          = ExtractDouble (row, "retrieval_strength", 0.0);
      ctx.metacognitive_confidence
          = ExtractDouble (row, "metacognitive_confidence", 0.0);

      // Sensitivity state
      ctx.weight_novelty = ExtractDouble (row, "weight_novelty", 0.3);
      ctx.weight_surprise = ExtractDouble (row, "weight_surprise", 0.2);
      ctx.weight_valence = ExtractDouble (row, "weight_valence", 0.4);
      ctx.weight_arousal = ExtractDouble (row, "weight_arousal", 0.0);
      ctx.emotion_gain = ExtractDouble (row, "emotion_gain", 1.0);
      ctx.score_gain = ExtractDouble (row, "score_gain", 1.0);

      // Stability state
      ctx.rate_decay = ExtractDouble (row, "rate_decay", 0.60);
      ctx.periphery_half_life
          = ExtractDouble (row, "periphery_half_life", 120.0);
      ctx.salience_half_life = ExtractDouble (row, "salience_half_life", 120.0);
      ctx.drift_weight = ExtractDouble (row, "drift_weight", 0.5);
      ctx.retention_ema = ExtractDouble (row, "retention_ema", 0.0);

      // Rate control (Algorithm 8)
      ctx.m_rate = ExtractDouble (row, "m_rate", 0.0);
      ctx.rho_hat_prev = ExtractDouble (row, "rho_hat_prev", 0.0);
      ctx.rate_ticks = static_cast<int> (ExtractInt64 (row, "rate_ticks", 0));
      ctx.dt_ema = ExtractDouble (row, "dt_ema", 0.0);
      ctx.last_rate_timestamp
          = static_cast<uint64_t> (ExtractInt64 (row, "last_rate_timestamp", 0));
      ctx.reliability = ExtractDouble (row, "reliability", 1.0);

      // Restore write rate window timestamps if present.
      auto write_rate_it = row.find ("write_rate_timestamps");
      if (write_rate_it != row.end () && write_rate_it->second.has_value ())
        {
          const auto ts = DeserializeUint64Vector (write_rate_it->second);
          if (!ts.empty ())
            {
              ctx.write_rate_window_.SetTimestamps (ts);
            }
        }

      // Emotion state (Algorithm 4)
      ctx.emotion_intensity_ewma
          = ExtractDouble (row, "emotion_intensity", 0.0);
      ctx.valence_ewma = ExtractDouble (row, "valence", 0.5);
      ctx.arousal_ewma = ExtractDouble (row, "arousal", 0.0);

      // Mood state (Algorithm 4b) - stored as BLOB (48 bytes = 6 doubles)
      auto mood_it = row.find ("mood_vector");
      if (mood_it != row.end () && mood_it->second.has_value ())
        {
          const double *data = nullptr;
          size_t byte_size = 0;
          if (mood_it->second.type () == typeid (std::vector<char>))
            {
              const auto &vec
                  = std::any_cast<const std::vector<char> &> (mood_it->second);
              data = reinterpret_cast<const double *> (vec.data ());
              byte_size = vec.size ();
            }
          else if (mood_it->second.type () == typeid (std::vector<unsigned char>))
            {
              const auto &vec = std::any_cast<const std::vector<unsigned char> &> (
                  mood_it->second);
              data = reinterpret_cast<const double *> (vec.data ());
              byte_size = vec.size ();
            }
          if (data && byte_size >= 6 * sizeof (double))
            {
              for (size_t i = 0; i < 6; ++i)
                ctx.mood_vector[i] = data[i];
            }
        }
      ctx.last_mood_ts
          = static_cast<uint64_t> (ExtractInt64 (row, "last_mood_ts", 0));

      // Embedding prediction error state (Section 3.1.4)
      auto last_emb_it = row.find ("last_embedding");
      if (last_emb_it != row.end () && last_emb_it->second.has_value ())
        {
          Eigen::VectorXf emb = BlobToEigen (last_emb_it->second);
          if (emb.size () > 0)
            ctx.last_embedding = std::move (emb);
        }
      auto delta_trend_it = row.find ("delta_x_trend");
      if (delta_trend_it != row.end () && delta_trend_it->second.has_value ())
        {
          Eigen::VectorXf trend = BlobToEigen (delta_trend_it->second);
          if (trend.size () > 0)
            ctx.delta_x_trend = std::move (trend);
        }

      // === Blender weights (from unified state table) ===
      ctx.blender_state[operations::Metric::relevance]
          = ExtractDouble (row, "w_relevance", 0.5);
      ctx.blender_state[operations::Metric::mismatch]
          = ExtractDouble (row, "w_mismatch", 0.5);
      ctx.blender_state[operations::Metric::surprise]
          = ExtractDouble (row, "w_surprise", 0.5);
      ctx.blender_state[operations::Metric::rarity]
          = ExtractDouble (row, "w_rarity", 0.5);
      ctx.blender_state[operations::Metric::drift]
          = ExtractDouble (row, "w_drift", 0.5);
      ctx.blender_state[operations::Metric::contradiction]
          = ExtractDouble (row, "w_contradiction", 0.5);
      ctx.blender_state[operations::Metric::utility]
          = ExtractDouble (row, "w_utility", 0.5);
      ctx.blender_state[operations::Metric::periphery]
          = ExtractDouble (row, "w_periphery", 0.5);
      ctx.blender_state[operations::Metric::coverage]
          = ExtractDouble (row, "w_coverage", 0.5);
      ctx.blender_state[operations::Metric::salience]
          = ExtractDouble (row, "w_salience", 0.5);
      ctx.blender_state[operations::Metric::valence]
          = ExtractDouble (row, "w_valence", 0.5);
      ctx.blender_state[operations::Metric::arousal]
          = ExtractDouble (row, "w_arousal", 0.5);
      ctx.blender_ready = ExtractInt64 (row, "blender_ready", 0) != 0;
      ctx.blender_update_count
          = static_cast<int> (ExtractInt64 (row, "blender_update_count", 0));

      // === Blender covariance matrix (P_matrix) ===
      auto P_it = row.find ("blender_P_matrix");
      if (P_it != row.end () && P_it->second.has_value ())
        {
          ctx.blender_P = DeserializeMatrix (P_it->second, 12);
        }

      ctx.focus_priors_initialized = true;
      ctx.sensitivity_priors_initialized = true;
      ctx.stability_priors_initialized = true;
      return true;
    }
  catch (const std::exception &e)
    {
      telemetry::LogWarn (
          "Failed to load state",
          { telemetry::Attribute::String ("component", "signal_processor"),
            telemetry::Attribute::String ("error", e.what ()) });
    }
  return false;
}

void
LoadRecentContext (Store &store, ProcessorContext &ctx)
{
  try
    {
      auto rows = store.Execute (
          "SELECT embedding FROM recent_context ORDER BY timestamp ASC");
      for (const auto &row : rows)
        {
          auto it = row.find ("embedding");
          if (it != row.end () && it->second.has_value ())
            {
              Eigen::VectorXf emb = BlobToEigen (it->second);
              if (emb.size () > 0)
                {
                  ctx.recent_context_embeddings.push_back (std::move (emb));
                }
            }
        }
    }
  catch (const std::exception &e)
    {
      telemetry::LogWarn (
          "Failed to load recent context",
          { telemetry::Attribute::String ("component", "signal_processor"),
            telemetry::Attribute::String ("error", e.what ()) });
    }
}

void
LoadRecentScores (Store &store, ProcessorContext &ctx)
{
  try
    {
      auto rows = store.Execute (
          "SELECT score FROM recent_scores ORDER BY timestamp ASC");
      for (const auto &row : rows)
        {
          ctx.recent_scores.push_back (ExtractDouble (row, "score", 0.0));
        }
    }
  catch (const std::exception &e)
    {
      telemetry::LogWarn (
          "Failed to load recent scores",
          { telemetry::Attribute::String ("component", "signal_processor"),
            telemetry::Attribute::String ("error", e.what ()) });
    }
}

void
LoadObservedRetentionHistory (Store &store, ProcessorContext &ctx)
{
  try
    {
      const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds> (
                              std::chrono::system_clock::now ().time_since_epoch ())
                              .count ();
      auto rows = store.Execute (
          "SELECT COALESCE(last_used, last_access) AS last_used "
          "FROM memories "
          "WHERE COALESCE(last_used, last_access) IS NOT NULL "
          "ORDER BY last_used ASC "
          "LIMIT 256");
      for (const auto &row : rows)
        {
          const auto last_used
              = ExtractInt64 (row, "last_used", 0);
          if (last_used <= 0)
            continue;
          const double retention_sec
              = std::max (0.0, static_cast<double> (now_ms - last_used) / 1000.0);
          ctx.observed_retention_history.push_back (retention_sec);
        }
    }
  catch (const std::exception &e)
    {
      telemetry::LogWarn (
          "Failed to load observed retention history",
          { telemetry::Attribute::String ("component", "signal_processor"),
            telemetry::Attribute::String ("error", e.what ()) });
    }
}

// v2 Schema: Load working memory from MEMORIES table (kind='WORKING')
void
LoadWorkingMemory (Store &store, ProcessorContext &ctx, double sensitivity)
{
  try
    {
      const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds> (
                              std::chrono::system_clock::now ().time_since_epoch ())
                              .count ();
      const double cost_per_slot = core::WMMaintenanceCostPerSlot (sensitivity);

      // Load from MEMORIES table with kind='WORKING' and end_ts IS NULL (active)
      auto rows = store.Execute (
          "SELECT memory_id, embedding_id, source_id, strength, last_access, "
          "       n_signals, s_max, s_avg, s_emotion_max, s_arousal_avg, "
          "       drift_mag, modality, start_ts "
          "FROM memories "
          "WHERE kind = 'WORKING' AND end_ts IS NULL "
          "ORDER BY start_ts ASC");

      int pos = 0;
      for (const auto &row : rows)
        {
          ProcessorContext::WMSlot slot;
          slot.memory_id = ExtractInt64 (row, "memory_id", 0);
          slot.pos_index = pos++;
          slot.strength = ExtractDouble (row, "strength", 0.0);

          // Extract source_id
          auto source_it = row.find ("source_id");
          if (source_it != row.end () && source_it->second.has_value ()
              && source_it->second.type () == typeid (std::string))
            {
              slot.source_id = std::any_cast<std::string> (source_it->second);
            }

          // Extract modality
          auto mod_it = row.find ("modality");
          if (mod_it != row.end () && mod_it->second.has_value ()
              && mod_it->second.type () == typeid (std::string))
            {
              slot.modality = std::any_cast<std::string> (mod_it->second);
            }

          // Load embedding for slot
          const auto embedding_id = ExtractInt64 (row, "embedding_id", 0);
          if (embedding_id > 0)
            {
              auto emb_rows = store.Execute (
                  "SELECT embedding FROM embeddings WHERE embedding_id = ?",
                  { embedding_id });
              if (!emb_rows.empty ())
                {
                  auto emb_it = emb_rows[0].find ("embedding");
                  if (emb_it != emb_rows[0].end () && emb_it->second.has_value ())
                    {
                      slot.embedding = BlobToEigen (emb_it->second);
                    }
                }
            }

          // DB stores milliseconds, convert to seconds for last_ts
          const auto ts_ms = ExtractInt64 (row, "last_access", now_ms);
          slot.last_ts = static_cast<double> (ts_ms) / 1000.0;
          slot.start_ts = ExtractInt64 (row, "start_ts", 0);

          // Apply time-based decay
          const double now_s = static_cast<double> (now_ms) / 1000.0;
          const double elapsed = now_s - slot.last_ts;
          if (elapsed > 0)
            {
              slot.strength -= cost_per_slot * elapsed;
            }

          // Skip decayed slots
          if (slot.strength <= 0.0)
            continue;

          // Load extended metadata
          slot.n_signals = static_cast<int> (ExtractInt64 (row, "n_signals", 1));
          slot.s_max = ExtractDouble (row, "s_max", 0.0);
          slot.s_avg = ExtractDouble (row, "s_avg", 0.0);
          slot.s_emotion_max = ExtractDouble (row, "s_emotion_max", 0.0);
          slot.s_arousal_avg = ExtractDouble (row, "s_arousal_avg", 0.0);
          slot.drift_acc = ExtractDouble (row, "drift_mag", 0.0);

          // Load signal records for this WM slot (ordered)
          auto sig_rows = store.Execute (
              "SELECT embedding_id, timestamp, modality, mime, blob_id, "
              "       score, serial_position "
              "FROM signals WHERE memory_id = ? "
              "ORDER BY serial_position ASC",
              { slot.memory_id });
          for (const auto &sig_row : sig_rows)
            {
              SignalRecord rec;
              rec.timestamp
                  = static_cast<uint64_t> (ExtractInt64 (sig_row, "timestamp", 0));
              rec.modality = "";
              rec.mime = "";
              auto mod_it = sig_row.find ("modality");
              if (mod_it != sig_row.end () && mod_it->second.has_value ()
                  && mod_it->second.type () == typeid (std::string))
                {
                  rec.modality = std::any_cast<std::string> (mod_it->second);
                }
              auto mime_it = sig_row.find ("mime");
              if (mime_it != sig_row.end () && mime_it->second.has_value ()
                  && mime_it->second.type () == typeid (std::string))
                {
                  rec.mime = std::any_cast<std::string> (mime_it->second);
                }
              rec.score = ExtractDouble (sig_row, "score", 0.0);
              rec.serial_position
                  = static_cast<int> (ExtractInt64 (sig_row, "serial_position", 0));

              auto blob_it = sig_row.find ("blob_id");
              if (blob_it != sig_row.end () && blob_it->second.has_value ())
                {
                  rec.blob_id = store::BlobFromAny (blob_it->second);
                }
              if (!rec.blob_id.empty ())
                {
                  slot.blob_ids.push_back (rec.blob_id);
                }

              const auto sig_emb_id
                  = ExtractInt64 (sig_row, "embedding_id", 0);
              if (sig_emb_id > 0)
                {
                  auto sig_emb_rows = store.Execute (
                      "SELECT embedding FROM embeddings WHERE embedding_id = ?",
                      { sig_emb_id });
                  if (!sig_emb_rows.empty ())
                    {
                      auto it_emb = sig_emb_rows[0].find ("embedding");
                      if (it_emb != sig_emb_rows[0].end ()
                          && it_emb->second.has_value ())
                        {
                          rec.embedding = BlobToEigen (it_emb->second);
                        }
                    }
                }
              slot.signal_records.push_back (std::move (rec));
            }

          ctx.wm_slots.push_back (std::move (slot));
        }
    }
  catch (const std::exception &e)
    {
      telemetry::LogWarn (
          "Failed to load working memory",
          { telemetry::Attribute::String ("component", "signal_processor"),
            telemetry::Attribute::String ("error", e.what ()) });
    }
}

// v2 Schema: Load accumulators from ACCUMULATORS table (renamed from accumulator_state)
void
LoadAccumulators (Store &store, ProcessorContext &ctx)
{
  try
    {
      auto rows = store.Execute ("SELECT * FROM accumulators");
      for (const auto &row : rows)
        {
          auto source_id_it = row.find ("source_id");
          if (source_id_it == row.end () || !source_id_it->second.has_value ())
            continue;

          std::string source_id;
          if (source_id_it->second.type () == typeid (std::string))
            source_id = std::any_cast<std::string> (source_id_it->second);
          else
            continue;

          AccumulatorState state;

          // v2: Load episode_id (FK to episodes table)
          state.episode_id = ExtractInt64 (row, "episode_id", 0);

          // Load embeddings
          auto mu_it = row.find ("mu_acc");
          if (mu_it != row.end () && mu_it->second.has_value ())
            state.mu_acc = BlobToEigen (mu_it->second);

          auto peak_it = row.find ("e_peak");
          if (peak_it != row.end () && peak_it->second.has_value ())
            state.e_peak = BlobToEigen (peak_it->second);

          // Load scalars
          state.drift_acc = ExtractDouble (row, "drift_acc", 0.0);
          state.s_sum = ExtractDouble (row, "s_sum", 0.0);
          state.s_max = ExtractDouble (row, "s_max", 0.0);
          state.s_emotion_max = ExtractDouble (row, "emo_max", 0.0);
          state.s_arousal_sum = ExtractDouble (row, "arousal_sum", 0.0);
          state.n_signals
              = static_cast<int> (ExtractInt64 (row, "n", 0));
          state.t_start
              = static_cast<uint64_t> (ExtractInt64 (row, "t_start", 0));
          state.last_write_ts
              = static_cast<uint64_t> (ExtractInt64 (row, "last_write_ts", 0));
          state.last_signal_ts
              = static_cast<uint64_t> (ExtractInt64 (row, "last_signal_ts", 0));
          state.eta_acc = ExtractDouble (row, "eta_acc", 0.0);
          state.coherence_prev = ExtractDouble (row, "coherence_prev", 1.0);
          state.drift_accum = ExtractDouble (row, "drift_accum", 0.0);
          state.drift_at_last_interrupt
              = ExtractDouble (row, "drift_at_last_interrupt", 0.0);
          state.drift_acc_pacing
              = ExtractDouble (row, "drift_acc_pacing", 0.0);

          auto x_last_it = row.find ("x_last_check");
          if (x_last_it != row.end () && x_last_it->second.has_value ())
            state.x_last_check = BlobToEigen (x_last_it->second);

          auto prev_it = row.find ("prev_x");
          if (prev_it != row.end () && prev_it->second.has_value ())
            state.prev_x = BlobToEigen (prev_it->second);

          auto window_it = row.find ("acc_signals_window");
          if (window_it != row.end () && window_it->second.has_value ())
            {
              state.acc_signals_window
                  = DeserializeEmbeddingWindow (window_it->second);
            }

          ctx.accumulator_states[source_id] = std::move (state);
        }
    }
  catch (const std::exception &e)
    {
      telemetry::LogWarn (
          "Failed to load accumulators",
          { telemetry::Attribute::String ("component", "signal_processor"),
            telemetry::Attribute::String ("error", e.what ()) });
    }
}

} // namespace

SignalProcessor::SignalProcessor (const Config &config,
                                  std::shared_ptr<Store> store,
                                  std::unique_ptr<IOperation> root_operation)
    : config_ (config), store_ (std::move (store)),
      root_operation_ (std::move (root_operation)),
      context_ (std::make_unique<ProcessorContext> ())
{
  if (!config_.encoder)
    {
      throw std::invalid_argument (
          "SignalProcessor requires a non-null Encoder (embeddinggemma fallback expected)");
    }
  // Initialize rate observation window capacity derived from Stability knob.
  if (context_)
    {
      const double T = core::Clamp (config_.stability, 0.0, 1.0);
      const int cap = core::WRateSeconds (T);
      context_->write_rate_window_.SetCapacity (
          static_cast<size_t> (std::max (1, cap)));

      // Initialize LLM components from config
      context_->extractor = config_.extractor;
      context_->summarizer = config_.summarizer;
    }
  // Apply schema migrations exactly once during initialization.
  bool loaded_state = false;
  if (store_)
    {
      cortext::store::ApplyMigrations (*store_);

      // Load persisted state for algorithm resumption (v2 schema)
      loaded_state = LoadState (*store_, *context_);               // Unified state
      LoadRecentContext (*store_, *context_);                      // From views
      LoadRecentScores (*store_, *context_);                       // From views
      LoadObservedRetentionHistory (*store_, *context_);           // Derived from memories
      LoadWorkingMemory (*store_, *context_, config_.sensitivity); // From MEMORIES
      LoadAccumulators (*store_, *context_);                       // From ACCUMULATORS
    }

  if (!loaded_state && context_)
    {
      const double F = core::Clamp (config_.focus, 0.0, 1.0);
      const double S = core::Clamp (config_.sensitivity, 0.0, 1.0);
      const double T = core::Clamp (config_.stability, 0.0, 1.0);
      context_->T_dynamic = core::TPrior (F, S, T);
      context_->T_target = context_->T_dynamic;
      context_->hysteresis = core::BaseBandPrior (T);
      const auto now_ms
          = std::chrono::duration_cast<std::chrono::milliseconds> (
                std::chrono::system_clock::now ().time_since_epoch ())
                .count ();
      context_->last_rate_timestamp = static_cast<uint64_t> (now_ms);
    }

}

SignalProcessor::~SignalProcessor () = default;

SignalProcessor::Output
SignalProcessor::Process (const Signal &signal)
{
  const auto t0 = std::chrono::steady_clock::now ();
  telemetry::ScopedSpan span ("cortext.process");

  // Create transaction for this signal processing
  auto tx = store_ ? store_->Begin () : nullptr;

  OperationContext op_context (signal, *context_, config_, store_.get ());

  try
    {
      StartNewEpisode (tx.get (), signal.timestamp);
      if (tx)
        {
          root_operation_->Execute (op_context, *tx);
        }

      if (op_context.GetWriteDecision ())
        {
          context_->write_rate_window_.Record (signal.timestamp);
        }

      span.SetAttribute ("cortext.at_boundary", op_context.GetAtBoundary ());
      span.SetAttribute ("cortext.interrupt_allowed",
                         op_context.GetInterruptAllowed ());
      span.SetAttribute ("cortext.threshold_T_dynamic",
                         op_context.GetThresholdTDynamic ());
      span.SetAttribute ("cortext.threshold_hysteresis",
                         op_context.GetThresholdHysteresis ());
      span.SetAttribute ("cortext.effective_focus",
                         op_context.GetEffectiveFocus ());

      if (op_context.ShouldFinalizeEpisode ())
        {
          FinalizeEpisode (tx.get (), &op_context);
        }

      // Persist state within the same transaction (v2 schema)
      if (tx)
        {
          PersistState (*tx);           // Unified state
          PersistWorkingMemory (*tx);   // To MEMORIES
          tx->Commit ();
        }
    }
  catch (...)
    {
      if (tx)
        {
          tx->Rollback ();
        }
      throw;
    }

  context_->signals_processed += 1;
  Output out;
  AssembleOutputMemories (op_context, out);
  AssembleOutputFields (op_context, out);
  const auto t1 = std::chrono::steady_clock::now ();
  const double ms
      = std::chrono::duration_cast<std::chrono::duration<double, std::milli> > (
            t1 - t0)
            .count ();
  telemetry::RecordHistogram ("cortext.process_duration_ms", ms);
  telemetry::AddCounter ("cortext.signals_processed_total", 1);
  if (out.at_boundary)
    {
      telemetry::AddCounter ("cortext.at_boundary_total", 1);
    }
  if (out.interrupt_allowed)
    {
      telemetry::AddCounter ("cortext.interrupt_allowed_total", 1);
    }
  LogProcessTelemetry (op_context, out);
  span.SetStatusOk ();
  return out;
}

void
SignalProcessor::Flush ()
{
  telemetry::AddCounter ("cortext.flush_total", 1);
  if (!store_)
    {
      return;
    }
  auto tx = store_->Begin ();
  FinalizeEpisode (tx.get (), nullptr);
  PersistState (*tx);           // v2: Unified state
  PersistWorkingMemory (*tx);   // v2: To MEMORIES
  PersistAccumulators (*tx);    // v2: To ACCUMULATORS
  const auto now_ms
      = std::chrono::duration_cast<std::chrono::milliseconds> (
            std::chrono::system_clock::now ().time_since_epoch ())
            .count ();
  StartNewEpisode (tx.get (), static_cast<uint64_t> (now_ms));
  tx->Commit ();
}

void
SignalProcessor::StartNewEpisode (Transaction *tx, uint64_t start_ts)
{
  if (!context_ || start_ts == 0)
    {
      return;
    }
  if (context_->episode_start_ts != 0)
    {
      return;
    }
  context_->episode_start_ts = start_ts;

  if (tx)
    {
      tx->Execute (
          "INSERT OR IGNORE INTO episodes "
          "(episode_id, start_ts, end_ts, boundary_type, centroid, created_at) "
          "VALUES (?, ?, NULL, NULL, NULL, ?)",
          { static_cast<long long> (start_ts),
            static_cast<long long> (start_ts),
            static_cast<long long> (start_ts) });
    }
}

void
SignalProcessor::FinalizeEpisode (Transaction *tx,
                                  const OperationContext *op_context)
{
  telemetry::ScopedSpan span ("cortext.episode.finalize");

  if (context_ && context_->episode_start_ts != 0)
    {
      const uint64_t end_ts
          = op_context ? op_context->GetSignal ().timestamp
                       : static_cast<uint64_t> (
                             std::chrono::duration_cast<
                                 std::chrono::milliseconds> (
                                 std::chrono::system_clock::now ()
                                     .time_since_epoch ())
                                 .count ());
      std::optional<std::string> boundary_type;
      std::optional<Eigen::VectorXf> centroid;
      if (op_context)
        {
          boundary_type = op_context->GetBoundaryType ();
          centroid = op_context->GetBoundaryCentroid ();
        }
      else
        {
          boundary_type = std::string ("explicit");
        }

      std::any boundary_type_any
          = boundary_type.has_value () ? std::any (*boundary_type)
                                       : std::any ();
      std::any centroid_any;
      if (centroid.has_value () && centroid->size () > 0)
        {
          centroid_any = ToFloatVector (*centroid);
        }
      else
        {
          centroid_any = std::any ();
        }

      if (tx)
        {
          tx->Execute (
              "UPDATE episodes "
              "SET end_ts = ?, boundary_type = ?, centroid = ? "
              "WHERE episode_id = ?",
              { static_cast<long long> (end_ts), boundary_type_any, centroid_any,
                static_cast<long long> (context_->episode_start_ts) });
        }
    }

  if (context_)
    {
      context_->episode_start_ts = 0;
    }

  // v2 schema: recent_context and recent_scores are now VIEWs that derive
  // from signals/embeddings tables. No need to persist - data comes from
  // MemoryStorage operation which writes to signals table.
  //
  // Note: State, WM, and Accumulators persisted by caller (Flush/Process)

  telemetry::AddCounter ("cortext.episode_commit_total", 1);
  span.SetStatusOk ();

  // Maintain recent_context to last n_ctx(T) plus drift lag for drift metrics.
  if (context_)
    {
      const size_t keep
          = static_cast<size_t> (core::NCtx (config_.stability)
                                 + core::KCtx (config_.stability));
      auto &embs = context_->recent_context_embeddings;
      while (embs.size () > keep)
        {
          embs.pop_front ();
        }
    }
}

// v2 Schema: Unified state persistence to STATE table
void
SignalProcessor::PersistState (Transaction &tx)
{
  if (!context_)
    return;

  const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds> (
                          std::chrono::system_clock::now ().time_since_epoch ())
                          .count ();

  // Serialize mood_vector as raw binary BLOB (48 bytes = 6 doubles)
  std::vector<char> mood_blob (6 * sizeof (double));
  std::memcpy (mood_blob.data (), context_->mood_vector.data (),
               6 * sizeof (double));
  const std::vector<char> write_rate_blob = SerializeUint64Vector (
      context_->write_rate_window_.GetTimestamps ());
  const double wm_maintenance_cost
      = core::WMMaintenanceCostPerSlot (config_.sensitivity);
  const int wm_slot_count
      = static_cast<int> (context_->wm_slots.size ());

  // Get blender weights
  auto get_weight = [this] (operations::Metric m) {
    auto it = context_->blender_state.find (m);
    return (it != context_->blender_state.end ()) ? it->second : 0.5;
  };

  double w_relevance = get_weight (operations::Metric::relevance);
  double w_mismatch = get_weight (operations::Metric::mismatch);
  double w_surprise = get_weight (operations::Metric::surprise);
  double w_rarity = get_weight (operations::Metric::rarity);
  double w_drift = get_weight (operations::Metric::drift);
  double w_contradiction = get_weight (operations::Metric::contradiction);
  double w_utility = get_weight (operations::Metric::utility);
  double w_periphery = get_weight (operations::Metric::periphery);
  double w_coverage = get_weight (operations::Metric::coverage);
  double w_salience = get_weight (operations::Metric::salience);
  double w_valence = get_weight (operations::Metric::valence);
  double w_arousal = get_weight (operations::Metric::arousal);

  // Serialize matrices as BLOBs
  std::vector<float> P_blob;
  if (!context_->blender_P.empty ())
    P_blob = SerializeMatrix (context_->blender_P);

  // Insert unified state row
  tx.Execute (
      "INSERT OR REPLACE INTO state "
      "(id, signals_processed, "
      // Threshold state
      "theta_dynamic, theta_target, hysteresis, half_life, "
      // Focus state
      "weight_relevance, attention_width, coverage_gain_floor, mismatch_weight, "
      // Sensitivity state
      "weight_novelty, weight_surprise, weight_valence, weight_arousal, "
      "emotion_gain, score_gain, rate_target, "
      // Emotion state
      "emotion_intensity, valence, arousal, mood_vector, last_mood_ts, "
      // Stability state
      "rate_decay, periphery_half_life, salience_half_life, drift_weight, retention_ema, "
      // Rate control
      "m_rate, rho_hat_prev, dt_ema, rate_ticks, last_rate_timestamp, reliability, "
      // Uncertainty
      "u_uncertainty, "
      // Embedding prediction
      "last_embedding, delta_x_trend, delta_half_life_adj, sustained_influence, "
      // Working memory
      "wm_maintenance_cost, wm_slot_count, wm_last_accepted, wm_last_chunked, "
      // Metacognition
      "fok_state, retrieval_strength, metacognitive_confidence, "
      // Consolidation
      "last_consolidation_ts, consolidation_count, is_processing_signal, last_retrieval_ts, "
      // Episode tracking
      "episode_start_ts, last_interrupt_tick, last_signal_timestamp, updated_at, "
      "write_rate_timestamps, "
      // Blender weights
      "w_relevance, w_mismatch, w_surprise, w_rarity, w_drift, w_contradiction, "
      "w_utility, w_periphery, w_coverage, w_salience, w_valence, w_arousal, "
      "blender_ready, blender_update_count, "
      // Blender matrices
      "blender_P_matrix) "
      "VALUES (1, ?, "
      "?, ?, ?, ?, "  // Threshold
      "?, ?, ?, ?, "  // Focus
      "?, ?, ?, ?, ?, ?, ?, "  // Sensitivity
      "?, ?, ?, ?, ?, "  // Emotion
      "?, ?, ?, ?, ?, "  // Stability
      "?, ?, ?, ?, ?, ?, "  // Rate control
      "?, "  // Uncertainty
      "?, ?, ?, ?, "  // Embedding prediction
      "?, ?, ?, ?, "  // Working memory
      "?, ?, ?, "  // Metacognition
      "?, ?, ?, ?, "  // Consolidation
      "?, ?, ?, ?, ?, "  // Episode tracking + write_rate_timestamps
      "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "  // Blender weights (12)
      "?, ?, "  // Blender ready/count
      "?)",  // Blender matrices
      {
        context_->signals_processed,
        // Threshold state
        context_->T_dynamic, context_->T_target, context_->hysteresis,
        context_->half_life,
        // Focus state
        context_->weight_relevance, context_->attention_width,
        context_->coverage_gain_floor, context_->mismatch_weight,
        // Sensitivity state
        context_->weight_novelty, context_->weight_surprise,
        context_->weight_valence, context_->weight_arousal,
        context_->emotion_gain, context_->score_gain, context_->rate_target,
        // Emotion state
        context_->emotion_intensity_ewma, context_->valence_ewma,
        context_->arousal_ewma, mood_blob,
        static_cast<long long> (context_->last_mood_ts),
        // Stability state
        context_->rate_decay, context_->periphery_half_life,
        context_->salience_half_life, context_->drift_weight,
        context_->retention_ema,
        // Rate control
        context_->m_rate, context_->rho_hat_prev, context_->dt_ema,
        context_->rate_ticks,
        static_cast<long long> (context_->last_rate_timestamp),
        context_->reliability,
        // Uncertainty
        context_->u_t,
        // Embedding prediction
        context_->last_embedding.has_value ()
            ? std::any (ToFloatVector (*context_->last_embedding))
            : std::any (std::vector<float> ()),
        context_->delta_x_trend.has_value ()
            ? std::any (ToFloatVector (*context_->delta_x_trend))
            : std::any (std::vector<float> ()),
        context_->delta_half_life_adj, context_->sustained_influence,
        // Working memory
        wm_maintenance_cost,
        wm_slot_count,
        context_->wm_last_accepted ? 1 : 0,
        context_->wm_last_chunked ? 1 : 0,
        // Metacognition
        context_->fok_state,
        context_->retrieval_strength,
        context_->metacognitive_confidence,
        // Consolidation
        static_cast<long long> (context_->last_consolidation_ts),
        context_->consolidation_count,
        context_->is_processing_signal ? 1 : 0,
        static_cast<long long> (context_->last_retrieval_ts),
        // Episode tracking
        static_cast<long long> (context_->episode_start_ts),
        context_->last_interrupt_tick,
        static_cast<long long> (context_->last_signal_timestamp), now_ms,
        write_rate_blob.empty () ? std::any (std::vector<char> ())
                                 : std::any (write_rate_blob),
        // Blender weights
        w_relevance, w_mismatch, w_surprise, w_rarity, w_drift, w_contradiction,
        w_utility, w_periphery, w_coverage, w_salience, w_valence, w_arousal,
        // Blender ready/count
        context_->blender_ready ? 1 : 0, context_->blender_update_count,
        // Blender matrices
        P_blob.empty () ? std::any (std::vector<float> ()) : std::any (P_blob)
      });
}

// v2 Schema: Persist working memory to MEMORIES table (kind='WORKING')
void
SignalProcessor::PersistWorkingMemory (Transaction &tx)
{
  if (!context_)
    return;

  const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds> (
                          std::chrono::system_clock::now ().time_since_epoch ())
                          .count ();

  // Mark existing WM slots as ended (soft delete)
  tx.Execute (
      "UPDATE memories SET end_ts = ? WHERE kind = 'WORKING' AND end_ts IS NULL",
      { now_ms });

  // Insert current slots as MEMORIES with kind='WORKING'
  for (const auto &slot : context_->wm_slots)
    {
      if (slot.strength <= 0.0)
        continue;
      if (slot.embedding.size () == 0)
        continue;

      const auto ts_ms = static_cast<int64_t> (slot.last_ts * 1000.0);

      const std::vector<float> emb_vec = ToFloatVector (slot.embedding);
      tx.Execute (
          "INSERT INTO embeddings (embedding, created_at) VALUES (?, ?)",
          { emb_vec, now_ms });
      auto emb_rows
          = tx.Execute ("SELECT last_insert_rowid() AS id", {});
      const long long embedding_id
          = emb_rows.empty () ? 0 : ExtractInt64 (emb_rows[0], "id", 0);
      if (embedding_id == 0)
        {
          continue;
        }

      tx.Execute (
          "INSERT INTO memories "
          "(embedding_id, source_id, kind, modality, start_ts, n_signals, "
          " s_max, s_avg, s_emotion_max, s_arousal_avg, drift_mag, "
          " strength, last_access, created_at) "
          "VALUES (?, ?, 'WORKING', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
          { embedding_id,
            slot.source_id.empty () ? std::string ("unknown") : slot.source_id,
            slot.modality, slot.start_ts, slot.n_signals, slot.s_max, slot.s_avg,
            slot.s_emotion_max, slot.s_arousal_avg, slot.drift_acc,
            slot.strength, ts_ms, now_ms });

      auto mem_rows = tx.Execute ("SELECT last_insert_rowid() AS id", {});
      const long long memory_id
          = mem_rows.empty () ? 0 : ExtractInt64 (mem_rows[0], "id", 0);
      if (memory_id == 0)
        {
          continue;
        }

      for (const auto &rec : slot.signal_records)
        {
          long long signal_embedding_id = embedding_id;
          if (rec.embedding.size () > 0)
            {
              const std::vector<float> sig_vec = ToFloatVector (rec.embedding);
              tx.Execute (
                  "INSERT INTO embeddings (embedding, created_at) VALUES (?, ?)",
                  { sig_vec, static_cast<long long> (rec.timestamp) });
              auto sig_rows
                  = tx.Execute ("SELECT last_insert_rowid() AS id", {});
              if (!sig_rows.empty ())
                {
                  signal_embedding_id = ExtractInt64 (sig_rows[0], "id",
                                                      embedding_id);
                }
            }

          tx.Execute (
              "INSERT INTO signals "
              "(memory_id, source_id, embedding_id, timestamp, modality, "
              " mime, blob_id, serial_position, score, created_at) "
              "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
              { memory_id,
                slot.source_id.empty () ? std::string ("unknown") : slot.source_id,
                signal_embedding_id,
                static_cast<long long> (rec.timestamp), rec.modality, rec.mime,
                rec.blob_id.empty () ? std::any ()
                                     : std::any (rec.blob_id),
                static_cast<long long> (rec.serial_position), rec.score,
                static_cast<long long> (rec.timestamp) });
        }
    }
}

// v2 Schema: Persist accumulators to ACCUMULATORS table
void
SignalProcessor::PersistAccumulators (Transaction &tx)
{
  if (!context_)
    return;

  // Clear old entries
  tx.Execute ("DELETE FROM accumulators", {});

  // Insert current accumulator states
  for (const auto &[source_id, state] : context_->accumulator_states)
    {
      // Skip empty/reset accumulators
      if (state.n_signals == 0 && state.mu_acc.size () == 0)
        continue;

      std::vector<float> mu_blob;
      if (state.mu_acc.size () > 0)
        mu_blob = ToFloatVector (state.mu_acc);

      std::vector<float> peak_blob;
      if (state.e_peak.size () > 0)
        peak_blob = ToFloatVector (state.e_peak);

      std::vector<float> last_check_blob;
      if (state.x_last_check.size () > 0)
        last_check_blob = ToFloatVector (state.x_last_check);

      std::vector<float> prev_x_blob;
      if (state.prev_x.size () > 0)
        prev_x_blob = ToFloatVector (state.prev_x);

      std::vector<char> window_blob;
      if (!state.acc_signals_window.empty ())
        {
          window_blob = SerializeEmbeddingWindow (state.acc_signals_window);
        }

      tx.Execute (
          "INSERT INTO accumulators "
          "(source_id, episode_id, mu_acc, drift_acc, s_sum, s_max, n, "
          " e_peak, emo_max, arousal_sum, drift_accum, drift_at_last_interrupt, "
          " drift_acc_pacing, x_last_check, prev_x, acc_signals_window, "
          " t_start, last_write_ts, last_signal_ts, eta_acc, coherence_prev) "
          "VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, "
          "?15, ?16, ?17, ?18, ?19, ?20, ?21)",
          { source_id, state.episode_id, mu_blob, state.drift_acc, state.s_sum,
            state.s_max, static_cast<long long> (state.n_signals), peak_blob,
            state.s_emotion_max, state.s_arousal_sum, state.drift_accum,
            state.drift_at_last_interrupt, state.drift_acc_pacing,
            last_check_blob, prev_x_blob, window_blob,
            static_cast<long long> (state.t_start),
            static_cast<long long> (state.last_write_ts),
            static_cast<long long> (state.last_signal_ts), state.eta_acc,
            state.coherence_prev });
    }
}

} // namespace cortext
