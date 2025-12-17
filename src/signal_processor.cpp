#include "cortext/core/knobs.hpp"
#include "cortext/processor.hpp"
#include "cortext/processor/operation_context.hpp"
#include "cortext/store/store.hpp"
#include "cortext/telemetry/telemetry.hpp"
#include "cortext/store/schema.hpp"
#include <chrono>
#include <any>
#include <cstring>
#include <map>
#include <vector>

namespace cortext
{

namespace
{

void
ComputeObservedRetention (Store *store, const SignalProcessor::Config &config,
                          const Signal &signal, OperationContext &op_context)
{
  if (!store)
    {
      return;
    }
  const double cutoff = core::PeripheryCutoff (config.stability);
  const uint64_t now_ts = signal.timestamp;
  try
    {
      const std::vector<std::map<std::string, std::any> > rows
          = store->Execute ("SELECT mf.last_used FROM memory_feedback mf "
                            "JOIN embeddings e ON e.embedding_id = mf.embedding_id "
                            "WHERE e.strength >= ? AND mf.last_used > 0",
                            { cutoff });
      if (!rows.empty ())
        {
          double sum_age = 0.0;
          int count = 0;
          for (const auto &row : rows)
            {
              auto it = row.find ("last_used");
              if (it == row.end ())
                continue;
              const std::any &v = it->second;
              uint64_t last_used_ts = 0;
              if (v.type () == typeid (long long))
                {
                  last_used_ts = static_cast<uint64_t> (
                      std::any_cast<long long> (v));
                }
              else if (v.type () == typeid (int64_t))
                {
                  last_used_ts
                      = static_cast<uint64_t> (std::any_cast<int64_t> (v));
                }
              else if (v.type () == typeid (int))
                {
                  last_used_ts
                      = static_cast<uint64_t> (std::any_cast<int> (v));
                }
              else
                {
                  continue;
                }
              if (now_ts > last_used_ts)
                {
                  sum_age += static_cast<double> (now_ts - last_used_ts);
                  count += 1;
                }
            }
          if (count > 0)
            {
              op_context.SetObservedRetentionSeconds (
                  sum_age / static_cast<double> (count));
            }
        }
    }
  catch (...)
    {
      telemetry::LogWarn (
          "Observed retention query failed; skipping",
          { telemetry::Attribute::String ("component", "signal_processor"),
            telemetry::Attribute::String ("db.system", "sqlite"),
            telemetry::Attribute::String ("db.operation", "SELECT") });
    }
}

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
    case operations::Metric::goal_alignment:
      return "goal_alignment";
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

void
LoadProcessorState (Store &store, ProcessorContext &ctx)
{
  try
    {
      auto rows = store.Execute ("SELECT * FROM processor_state WHERE id = 1");
      if (rows.empty ())
        return; // No persisted state, use defaults

      const auto &row = rows[0];
      ctx.signals_processed
          = static_cast<int> (ExtractInt64 (row, "signals_processed", 0));
      ctx.u_t = ExtractDouble (row, "u_uncertainty", 0.0);
      ctx.weight_relevance_prior
          = ExtractDouble (row, "weight_relevance_prior", 0.5);
      ctx.weight_relevance = ExtractDouble (row, "weight_relevance", 0.5);
      ctx.attention_width = ExtractDouble (row, "attention_width", 1.57);
      ctx.T_dynamic = ExtractDouble (row, "theta_dynamic", 0.2);
      ctx.hysteresis = ExtractDouble (row, "hysteresis", 0.05);
      ctx.half_life = ExtractDouble (row, "half_life", 120.0);
      ctx.rate_target = ExtractDouble (row, "rate_target", 0.0);
      ctx.sustained_influence
          = ExtractDouble (row, "sustained_influence", 0.0);
      ctx.last_signal_timestamp
          = static_cast<uint64_t> (ExtractInt64 (row, "last_signal_timestamp", 0));

      // Focus priors (Algorithm 1)
      ctx.coverage_gain_floor_prior
          = ExtractDouble (row, "coverage_gain_floor_prior", 0.65);
      ctx.mismatch_weight_prior
          = ExtractDouble (row, "mismatch_weight_prior", 0.5);
      ctx.attention_width_prior
          = ExtractDouble (row, "attention_width_prior", 1.57);

      // Sensitivity priors (Algorithm 3)
      ctx.base_rate_prior = ExtractDouble (row, "base_rate_prior", 0.2);
      ctx.weight_novelty_prior
          = ExtractDouble (row, "weight_novelty_prior", 0.3);
      ctx.weight_surprise_prior
          = ExtractDouble (row, "weight_surprise_prior", 0.2);
      ctx.weight_valence_prior
          = ExtractDouble (row, "weight_valence_prior", 0.4);
      ctx.weight_arousal_prior
          = ExtractDouble (row, "weight_arousal_prior", 0.0);
      ctx.weight_emotion_prior
          = ExtractDouble (row, "weight_emotion_prior", 0.2);
      ctx.emotion_gain_prior = ExtractDouble (row, "emotion_gain_prior", 1.0);
      ctx.score_gain_prior = ExtractDouble (row, "score_gain_prior", 1.0);
      ctx.rate_target_prior = ExtractDouble (row, "rate_target_prior", 0.2);

      // Sensitivity dynamic (Algorithm 4)
      ctx.weight_novelty = ExtractDouble (row, "weight_novelty", 0.3);

      // Stability priors (Algorithm 5)
      ctx.hysteresis_band_prior
          = ExtractDouble (row, "hysteresis_band_prior", 0.02);
      ctx.half_life_prior = ExtractDouble (row, "half_life_prior", 120.0);
      ctx.rate_decay_prior = ExtractDouble (row, "rate_decay_prior", 0.60);
      ctx.periphery_half_life_prior
          = ExtractDouble (row, "periphery_half_life_prior", 120.0);
      ctx.salience_half_life_prior
          = ExtractDouble (row, "salience_half_life_prior", 120.0);
      ctx.drift_weight_prior = ExtractDouble (row, "drift_weight_prior", 0.5);

      // Stability dynamic (Algorithm 6)
      ctx.rate_decay = ExtractDouble (row, "rate_decay", 0.60);
      ctx.periphery_half_life
          = ExtractDouble (row, "periphery_half_life", 120.0);
      ctx.salience_half_life = ExtractDouble (row, "salience_half_life", 120.0);

      // Threshold state (Algorithm 8)
      ctx.m_rate = ExtractDouble (row, "m_rate", 0.0);
      ctx.rate_ticks = static_cast<int> (ExtractInt64 (row, "rate_ticks", 0));
      ctx.dt_ema = ExtractDouble (row, "dt_ema", 0.0);
      ctx.last_rate_timestamp
          = static_cast<uint64_t> (ExtractInt64 (row, "last_rate_timestamp", 0));

      // Emotion state (Algorithm 4) - column names without _ewma suffix per docs
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
    }
  catch (const std::exception &e)
    {
      telemetry::LogWarn (
          "Failed to load processor state",
          { telemetry::Attribute::String ("component", "signal_processor"),
            telemetry::Attribute::String ("error", e.what ()) });
    }
}

void
LoadBlenderState (Store &store, ProcessorContext &ctx)
{
  try
    {
      // Load weights
      auto weight_rows
          = store.Execute ("SELECT * FROM blender_weights WHERE id = 1");
      if (!weight_rows.empty ())
        {
          const auto &row = weight_rows[0];
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
          ctx.blender_ready
              = ExtractInt64 (row, "blender_ready", 0) != 0;
          ctx.blender_update_count
              = static_cast<int> (ExtractInt64 (row, "update_count", 0));
        }

      // Load covariance matrix
      auto cov_rows
          = store.Execute ("SELECT P_matrix FROM blender_covariance WHERE id = 1");
      if (!cov_rows.empty ())
        {
          auto it = cov_rows[0].find ("P_matrix");
          if (it != cov_rows[0].end () && it->second.has_value ())
            {
              ctx.blender_P = DeserializeMatrix (it->second, 12);
            }
        }
    }
  catch (const std::exception &e)
    {
      telemetry::LogWarn (
          "Failed to load blender state",
          { telemetry::Attribute::String ("component", "signal_processor"),
            telemetry::Attribute::String ("error", e.what ()) });
    }
}

void
LoadRecentContext (Store &store, ProcessorContext &ctx)
{
  try
    {
      auto rows = store.Execute (
          "SELECT embedding FROM recent_context ORDER BY seq_order ASC");
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
      auto rows = store.Execute (
          "SELECT retention_value FROM observed_retention_history "
          "ORDER BY timestamp ASC");
      for (const auto &row : rows)
        {
          ctx.observed_retention_history.push_back (
              ExtractDouble (row, "retention_value", 0.0));
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

// Deserialize 48 doubles from BLOB to rls_coefficients (12 x 4)
void
LoadRLSCoefficients (Store &store, ProcessorContext &ctx)
{
  try
    {
      auto rows = store.Execute (
          "SELECT coefficients FROM blender_coefficients WHERE id = 1");
      if (!rows.empty ())
        {
          auto it = rows[0].find ("coefficients");
          if (it != rows[0].end () && it->second.has_value ())
            {
              const float *data = nullptr;
              size_t byte_size = 0;

              if (it->second.type () == typeid (std::vector<char>))
                {
                  const auto &vec
                      = std::any_cast<const std::vector<char> &> (it->second);
                  data = reinterpret_cast<const float *> (vec.data ());
                  byte_size = vec.size ();
                }
              else if (it->second.type ()
                       == typeid (std::vector<unsigned char>))
                {
                  const auto &vec
                      = std::any_cast<const std::vector<unsigned char> &> (
                          it->second);
                  data = reinterpret_cast<const float *> (vec.data ());
                  byte_size = vec.size ();
                }

              constexpr size_t kExpected = 48 * sizeof (float);
              if (byte_size == kExpected && data != nullptr)
                {
                  ctx.rls_coefficients.resize (ProcessorContext::kNumMetrics);
                  size_t idx = 0;
                  for (size_t i = 0; i < ProcessorContext::kNumMetrics; ++i)
                    {
                      for (size_t j = 0; j < ProcessorContext::kCoeffsPerMetric;
                           ++j)
                        {
                          ctx.rls_coefficients[i][j]
                              = static_cast<double> (data[idx++]);
                        }
                    }
                  ctx.rls_coefficients_ready = true;
                }
            }
        }

      // Load covariance matrix
      auto cov_rows = store.Execute (
          "SELECT P_matrix FROM blender_coeff_covariance WHERE id = 1");
      if (!cov_rows.empty ())
        {
          auto cov_it = cov_rows[0].find ("P_matrix");
          if (cov_it != cov_rows[0].end () && cov_it->second.has_value ())
            {
              constexpr size_t kTotalCoeffs = 48;
              ctx.rls_coeff_P
                  = DeserializeMatrix (cov_it->second, kTotalCoeffs);
            }
        }
    }
  catch (const std::exception &e)
    {
      telemetry::LogWarn (
          "Failed to load RLS coefficients",
          { telemetry::Attribute::String ("component", "signal_processor"),
            telemetry::Attribute::String ("error", e.what ()) });
    }
}

void
LoadWorkingMemorySlots (Store &store, ProcessorContext &ctx, double sensitivity)
{
  try
    {
      const auto now = std::chrono::duration_cast<std::chrono::seconds> (
                           std::chrono::system_clock::now ().time_since_epoch ())
                           .count ();
      const double cost_per_slot = core::WMMaintenanceCostPerSlot (sensitivity);

      auto rows = store.Execute (
          "SELECT slot_index, strength, timestamp, embedding "
          "FROM working_memory_slots ORDER BY slot_index ASC");

      for (const auto &row : rows)
        {
          ProcessorContext::WMSlot slot;
          slot.pos_index
              = static_cast<int> (ExtractInt64 (row, "slot_index", 0));
          slot.strength = ExtractDouble (row, "strength", 0.0);
          slot.last_ts
              = static_cast<double> (ExtractInt64 (row, "timestamp", now));

          // Apply time-based decay (same formula as WorkingMemory::Execute
          // maintenance)
          const double elapsed = static_cast<double> (now) - slot.last_ts;
          if (elapsed > 0)
            {
              slot.strength -= cost_per_slot * elapsed;
            }

          // Skip decayed slots
          if (slot.strength <= 0.0)
            continue;

          auto emb_it = row.find ("embedding");
          if (emb_it != row.end () && emb_it->second.has_value ())
            {
              slot.embedding = BlobToEigen (emb_it->second);
              if (slot.embedding.size () > 0)
                {
                  ctx.wm_slots.push_back (std::move (slot));
                }
            }
        }
    }
  catch (const std::exception &e)
    {
      telemetry::LogWarn (
          "Failed to load working memory slots",
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
  // Initialize rate observation window capacity derived from Stability knob.
  if (context_)
    {
      const double T = core::Clamp (config_.stability, 0.0, 1.0);
      const int cap
          = static_cast<int> (std::round (core::Lerp (10.0, 60.0, T)));
      context_->write_rate_window_.SetCapacity (
          static_cast<size_t> (std::max (1, cap)));

      // Initialize LLM components from config
      context_->extractor = config_.extractor;
      context_->summarizer = config_.summarizer;
    }
  // Apply schema migrations exactly once during initialization.
  if (store_)
    {
      cortext::store::SchemaRegistry registry;
      if (root_operation_)
        {
          root_operation_->CollectSchema (registry);
        }
      cortext::store::ApplyMigrations (*store_, registry);

      // Load persisted state for algorithm resumption
      LoadProcessorState (*store_, *context_);
      LoadBlenderState (*store_, *context_);
      LoadRecentContext (*store_, *context_);
      LoadRecentScores (*store_, *context_);
      LoadObservedRetentionHistory (*store_, *context_);
      LoadRLSCoefficients (*store_, *context_);
      LoadWorkingMemorySlots (*store_, *context_, config_.sensitivity);
    }

  StartNewEpisode ();
}

SignalProcessor::~SignalProcessor () = default;

SignalProcessor::Output
SignalProcessor::Process (const Signal &signal)
{
  const auto t0 = std::chrono::steady_clock::now ();
  telemetry::ScopedSpan span ("cortext.process");
  OperationContext op_context (signal, *context_, config_, write_buffer_,
                               store_.get ());
  context_->write_rate_window_.Record (signal.timestamp);
  ComputeObservedRetention (store_.get (), config_, signal, op_context);
  root_operation_->Execute (op_context);
  span.SetAttribute ("cortext.at_boundary", op_context.GetAtBoundary ());
  span.SetAttribute ("cortext.interrupt_allowed",
                     op_context.GetInterruptAllowed ());
  span.SetAttribute ("cortext.threshold_T_dynamic",
                     op_context.GetThresholdTDynamic ());
  span.SetAttribute ("cortext.threshold_hysteresis",
                     op_context.GetThresholdHysteresis ());
  span.SetAttribute ("cortext.effective_focus", op_context.GetEffectiveFocus ());
  if (op_context.ShouldFinalizeEpisode ())
    {
      FinalizeEpisode ();
      StartNewEpisode ();
    }
  context_->signals_processed += 1;
  Output out;
  AssembleOutputMemories (op_context, out);
  AssembleOutputFields (op_context, out);
  const auto t1 = std::chrono::steady_clock::now ();
  const double ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli> > (t1 - t0).count ();
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
  FinalizeEpisode ();
  StartNewEpisode ();
}

void
SignalProcessor::StartNewEpisode ()
{
  // Keep the episode transaction closed during signal processing to avoid
  // holding a long-lived read transaction that blocks other writers. We open
  // a short-lived transaction only when finalizing buffered writes.
  episode_transaction_.reset ();
}

void
SignalProcessor::FinalizeEpisode ()
{
  if (!store_)
    {
      return;
    }

  telemetry::ScopedSpan span ("cortext.episode.finalize");

  episode_transaction_ = store_->Begin ();

  // Persist processor state (singleton tables)
  PersistProcessorState ();
  PersistBlenderState ();
  PersistRecentContext ();
  PersistRecentScores ();
  PersistObservedRetentionHistory ();
  PersistRLSCoefficients ();
  PersistWorkingMemorySlots ();

  // Execute buffered writes from operations
  for (const auto &instruction : write_buffer_)
    {
      episode_transaction_->Execute (instruction.query, instruction.params);
    }
  write_buffer_.clear ();

  episode_transaction_->Commit ();
  episode_transaction_.reset ();
  telemetry::AddCounter ("cortext.episode_commit_total", 1);
  span.SetStatusOk ();

  // Maintain recent_context to last n_ctx(T) after boundary (Alg 12).
  const size_t keep = static_cast<size_t> (core::NCtx (config_.stability));
  auto &embs = context_->recent_context_embeddings;
  while (embs.size () > keep)
    {
      embs.pop_front ();
    }
}

void
SignalProcessor::PersistProcessorState ()
{
  if (!episode_transaction_ || !context_)
    return;

  const auto now = std::chrono::duration_cast<std::chrono::seconds> (
                       std::chrono::system_clock::now ().time_since_epoch ())
                       .count ();

  // Serialize mood_vector as raw binary BLOB (48 bytes = 6 doubles)
  std::vector<char> mood_blob (6 * sizeof (double));
  std::memcpy (mood_blob.data (), context_->mood_vector.data (),
               6 * sizeof (double));

  episode_transaction_->Execute (
      "INSERT OR REPLACE INTO processor_state "
      "(id, signals_processed, u_uncertainty, weight_relevance_prior, "
      "weight_relevance, "
      "attention_width, theta_dynamic, hysteresis, half_life, rate_target, "
      "sustained_influence, last_signal_timestamp, updated_at, "
      // Focus priors (Algorithm 1)
      "coverage_gain_floor_prior, mismatch_weight_prior, attention_width_prior, "
      // Sensitivity priors (Algorithm 3)
      "base_rate_prior, weight_novelty_prior, weight_surprise_prior, "
      "weight_valence_prior, weight_arousal_prior, weight_emotion_prior, "
      "emotion_gain_prior, score_gain_prior, rate_target_prior, "
      // Sensitivity dynamic (Algorithm 4)
      "weight_novelty, "
      // Stability priors (Algorithm 5)
      "hysteresis_band_prior, half_life_prior, rate_decay_prior, "
      "periphery_half_life_prior, salience_half_life_prior, drift_weight_prior, "
      // Stability dynamic (Algorithm 6)
      "rate_decay, periphery_half_life, salience_half_life, "
      // Threshold state (Algorithm 8)
      "m_rate, rate_ticks, dt_ema, last_rate_timestamp, "
      // Emotion state (Algorithm 4) - no _ewma suffix per docs
      "emotion_intensity, valence, arousal, "
      // Mood state (Algorithm 4b) - single BLOB
      "mood_vector, "
      // Embedding prediction error state (Section 3.1.4)
      "last_embedding, delta_x_trend) "
      "VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
      "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
      "?, ?, ?, ?, ?, ?)",
      { // Original fields
        context_->signals_processed, context_->u_t,
        context_->weight_relevance_prior, context_->weight_relevance,
        context_->attention_width, context_->T_dynamic, context_->hysteresis,
        context_->half_life, context_->rate_target, context_->sustained_influence,
        static_cast<long long> (context_->last_signal_timestamp), now,
        // Focus priors (Algorithm 1)
        context_->coverage_gain_floor_prior, context_->mismatch_weight_prior,
        context_->attention_width_prior,
        // Sensitivity priors (Algorithm 3)
        context_->base_rate_prior, context_->weight_novelty_prior,
        context_->weight_surprise_prior, context_->weight_valence_prior,
        context_->weight_arousal_prior, context_->weight_emotion_prior,
        context_->emotion_gain_prior, context_->score_gain_prior,
        context_->rate_target_prior,
        // Sensitivity dynamic (Algorithm 4)
        context_->weight_novelty,
        // Stability priors (Algorithm 5)
        context_->hysteresis_band_prior, context_->half_life_prior,
        context_->rate_decay_prior, context_->periphery_half_life_prior,
        context_->salience_half_life_prior, context_->drift_weight_prior,
        // Stability dynamic (Algorithm 6)
        context_->rate_decay, context_->periphery_half_life,
        context_->salience_half_life,
        // Threshold state (Algorithm 8)
        context_->m_rate, context_->rate_ticks, context_->dt_ema,
        static_cast<long long> (context_->last_rate_timestamp),
        // Emotion state (Algorithm 4)
        context_->emotion_intensity_ewma, context_->valence_ewma,
        context_->arousal_ewma,
        // Mood state (Algorithm 4b) - as BLOB
        mood_blob,
        // Embedding prediction error state (Section 3.1.4)
        context_->last_embedding.has_value ()
            ? std::any (ToFloatVector (*context_->last_embedding))
            : std::any (std::vector<float> ()),
        context_->delta_x_trend.has_value ()
            ? std::any (ToFloatVector (*context_->delta_x_trend))
            : std::any (std::vector<float> ()) });
}

void
SignalProcessor::PersistBlenderState ()
{
  if (!episode_transaction_ || !context_)
    return;

  // Persist weights
  double w_relevance = 0.5, w_mismatch = 0.5, w_surprise = 0.5, w_rarity = 0.5;
  double w_drift = 0.5, w_contradiction = 0.5, w_utility = 0.5, w_periphery = 0.5;
  double w_coverage = 0.5, w_salience = 0.5, w_valence = 0.5, w_arousal = 0.5;

  auto get_weight = [this] (operations::Metric m) {
    auto it = context_->blender_state.find (m);
    return (it != context_->blender_state.end ()) ? it->second : 0.5;
  };

  w_relevance = get_weight (operations::Metric::relevance);
  w_mismatch = get_weight (operations::Metric::mismatch);
  w_surprise = get_weight (operations::Metric::surprise);
  w_rarity = get_weight (operations::Metric::rarity);
  w_drift = get_weight (operations::Metric::drift);
  w_contradiction = get_weight (operations::Metric::contradiction);
  w_utility = get_weight (operations::Metric::utility);
  w_periphery = get_weight (operations::Metric::periphery);
  w_coverage = get_weight (operations::Metric::coverage);
  w_salience = get_weight (operations::Metric::salience);
  w_valence = get_weight (operations::Metric::valence);
  w_arousal = get_weight (operations::Metric::arousal);

  episode_transaction_->Execute (
      "INSERT OR REPLACE INTO blender_weights "
      "(id, w_relevance, w_mismatch, w_surprise, w_rarity, w_drift, "
      "w_contradiction, w_utility, w_periphery, w_coverage, w_salience, "
      "w_valence, w_arousal, blender_ready, update_count) "
      "VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
      { w_relevance, w_mismatch, w_surprise, w_rarity, w_drift, w_contradiction,
        w_utility, w_periphery, w_coverage, w_salience, w_valence, w_arousal,
        context_->blender_ready ? 1 : 0, context_->blender_update_count });

  // Persist covariance matrix if populated
  if (!context_->blender_P.empty ())
    {
      std::vector<float> P_blob = SerializeMatrix (context_->blender_P);
      episode_transaction_->Execute (
          "INSERT OR REPLACE INTO blender_covariance (id, P_matrix) "
          "VALUES (1, ?)",
          { P_blob });
    }
}

void
SignalProcessor::PersistRecentContext ()
{
  if (!episode_transaction_ || !context_)
    return;

  // Clear old entries
  episode_transaction_->Execute ("DELETE FROM recent_context", {});

  // Insert current window
  int seq = 0;
  const auto now = std::chrono::duration_cast<std::chrono::seconds> (
                       std::chrono::system_clock::now ().time_since_epoch ())
                       .count ();

  for (const auto &emb : context_->recent_context_embeddings)
    {
      std::vector<float> emb_blob = ToFloatVector (emb);
      episode_transaction_->Execute (
          "INSERT INTO recent_context (embedding, timestamp, seq_order) "
          "VALUES (?, ?, ?)",
          { emb_blob, now, seq++ });
    }
}

void
SignalProcessor::PersistRecentScores ()
{
  if (!episode_transaction_ || !context_)
    return;

  // Clear old entries
  episode_transaction_->Execute ("DELETE FROM recent_scores", {});

  // Insert current window
  const auto now = std::chrono::duration_cast<std::chrono::seconds> (
                       std::chrono::system_clock::now ().time_since_epoch ())
                       .count ();

  for (const auto &score : context_->recent_scores)
    {
      episode_transaction_->Execute (
          "INSERT INTO recent_scores (score, timestamp) VALUES (?, ?)",
          { score, now });
    }
}

void
SignalProcessor::PersistObservedRetentionHistory ()
{
  if (!episode_transaction_ || !context_)
    return;

  // Clear old entries
  episode_transaction_->Execute ("DELETE FROM observed_retention_history", {});

  // Insert current window
  const auto now = std::chrono::duration_cast<std::chrono::seconds> (
                       std::chrono::system_clock::now ().time_since_epoch ())
                       .count ();

  for (const auto &retention : context_->observed_retention_history)
    {
      episode_transaction_->Execute (
          "INSERT INTO observed_retention_history "
          "(retention_seconds, timestamp) VALUES (?, ?)",
          { retention, now });
    }
}

void
SignalProcessor::PersistRLSCoefficients ()
{
  if (!episode_transaction_ || !context_)
    return;

  // Serialize rls_coefficients (12 x 4 = 48 doubles) to float BLOB
  if (!context_->rls_coefficients.empty ())
    {
      std::vector<float> coeff_blob;
      coeff_blob.reserve (ProcessorContext::kNumMetrics
                          * ProcessorContext::kCoeffsPerMetric);
      for (const auto &metric_coeffs : context_->rls_coefficients)
        {
          for (double c : metric_coeffs)
            {
              coeff_blob.push_back (static_cast<float> (c));
            }
        }
      episode_transaction_->Execute (
          "INSERT OR REPLACE INTO blender_coefficients (id, coefficients) "
          "VALUES (1, ?)",
          { coeff_blob });
    }

  // Persist coefficient covariance matrix if populated
  if (!context_->rls_coeff_P.empty ())
    {
      std::vector<float> P_blob = SerializeMatrix (context_->rls_coeff_P);
      episode_transaction_->Execute (
          "INSERT OR REPLACE INTO blender_coeff_covariance (id, P_matrix) "
          "VALUES (1, ?)",
          { P_blob });
    }
}

void
SignalProcessor::PersistWorkingMemorySlots ()
{
  if (!episode_transaction_ || !context_)
    return;

  // Clear old entries
  episode_transaction_->Execute ("DELETE FROM working_memory_slots", {});

  // Insert current slots
  const auto now = std::chrono::duration_cast<std::chrono::seconds> (
                       std::chrono::system_clock::now ().time_since_epoch ())
                       .count ();

  for (size_t i = 0; i < context_->wm_slots.size (); ++i)
    {
      const auto &slot = context_->wm_slots[i];
      std::vector<float> emb_blob = ToFloatVector (slot.embedding);
      episode_transaction_->Execute (
          "INSERT INTO working_memory_slots "
          "(slot_index, strength, timestamp, embedding) VALUES (?, ?, ?, ?)",
          { static_cast<int64_t> (i), slot.strength, now, emb_blob });
    }
}

} // namespace cortext
