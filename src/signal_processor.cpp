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
// v2 Schema: Unified state table replaces processor_state + blender tables

void
LoadState (Store &store, ProcessorContext &ctx)
{
  try
    {
      auto rows = store.Execute ("SELECT * FROM state WHERE id = 1");
      if (rows.empty ())
        return; // No persisted state, use defaults

      const auto &row = rows[0];

      // === Processor state fields ===
      ctx.signals_processed
          = static_cast<int> (ExtractInt64 (row, "signals_processed", 0));
      ctx.u_t = ExtractDouble (row, "u_uncertainty", 0.0);
      ctx.weight_relevance = ExtractDouble (row, "weight_relevance", 0.5);
      ctx.attention_width = ExtractDouble (row, "attention_width", 1.57);
      ctx.T_dynamic = ExtractDouble (row, "theta_dynamic", 0.2);
      ctx.hysteresis = ExtractDouble (row, "hysteresis", 0.05);
      ctx.half_life = ExtractDouble (row, "half_life", 120.0);
      ctx.rate_target = ExtractDouble (row, "rate_target", 0.2);
      ctx.sustained_influence
          = ExtractDouble (row, "sustained_influence", 0.0);
      ctx.last_signal_timestamp
          = static_cast<uint64_t> (ExtractInt64 (row, "last_signal_timestamp", 0));

      // Sensitivity state
      ctx.weight_novelty = ExtractDouble (row, "weight_novelty", 0.3);

      // Stability state
      ctx.rate_decay = ExtractDouble (row, "rate_decay", 0.60);
      ctx.periphery_half_life
          = ExtractDouble (row, "periphery_half_life", 120.0);
      ctx.salience_half_life = ExtractDouble (row, "salience_half_life", 120.0);

      // Rate control (Algorithm 8)
      ctx.m_rate = ExtractDouble (row, "m_rate", 0.0);
      ctx.rate_ticks = static_cast<int> (ExtractInt64 (row, "rate_ticks", 0));
      ctx.dt_ema = ExtractDouble (row, "dt_ema", 0.0);
      ctx.last_rate_timestamp
          = static_cast<uint64_t> (ExtractInt64 (row, "last_rate_timestamp", 0));

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

      // === RLS coefficients ===
      auto coeff_it = row.find ("blender_coefficients");
      if (coeff_it != row.end () && coeff_it->second.has_value ())
        {
          const float *data = nullptr;
          size_t byte_size = 0;

          if (coeff_it->second.type () == typeid (std::vector<char>))
            {
              const auto &vec
                  = std::any_cast<const std::vector<char> &> (coeff_it->second);
              data = reinterpret_cast<const float *> (vec.data ());
              byte_size = vec.size ();
            }
          else if (coeff_it->second.type ()
                   == typeid (std::vector<unsigned char>))
            {
              const auto &vec
                  = std::any_cast<const std::vector<unsigned char> &> (
                      coeff_it->second);
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
                  for (size_t j = 0; j < ProcessorContext::kCoeffsPerMetric; ++j)
                    {
                      ctx.rls_coefficients[i][j]
                          = static_cast<double> (data[idx++]);
                    }
                }
              ctx.rls_coefficients_ready = true;
            }
        }

      // === Coefficient covariance matrix ===
      auto coeff_P_it = row.find ("blender_coeff_P_matrix");
      if (coeff_P_it != row.end () && coeff_P_it->second.has_value ())
        {
          constexpr size_t kTotalCoeffs = 48;
          ctx.rls_coeff_P = DeserializeMatrix (coeff_P_it->second, kTotalCoeffs);
        }
    }
  catch (const std::exception &e)
    {
      telemetry::LogWarn (
          "Failed to load state",
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
      // Note: For now we use a simpler query without JOIN to embeddings
      // since we store the embedding directly. In a full implementation,
      // this would join with the embeddings table.
      auto rows = store.Execute (
          "SELECT memory_id, source_id, strength, last_access, n_signals, "
          "       s_max, s_avg, modality, start_ts "
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

          // Note: embedding is loaded separately if needed via embedding_id
          // For backward compatibility, we initialize an empty embedding
          // The full implementation would JOIN with embeddings table
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
          // Note: n_signals is tracked in-memory, not persisted in v2 schema
          state.n_signals = 0;
          state.t_start
              = static_cast<uint64_t> (ExtractInt64 (row, "t_start", 0));
          state.last_write_ts
              = static_cast<uint64_t> (ExtractInt64 (row, "last_write_ts", 0));
          state.last_signal_ts
              = static_cast<uint64_t> (ExtractInt64 (row, "last_signal_ts", 0));
          state.eta_acc = ExtractDouble (row, "eta_acc", 0.0);
          state.coherence_prev = ExtractDouble (row, "coherence_prev", 1.0);

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
      cortext::store::ApplyMigrations (*store_);

      // Load persisted state for algorithm resumption (v2 schema)
      LoadState (*store_, *context_);                              // Unified state
      LoadRecentContext (*store_, *context_);                      // From views
      LoadRecentScores (*store_, *context_);                       // From views
      LoadObservedRetentionHistory (*store_, *context_);           // Keep if used
      LoadWorkingMemory (*store_, *context_, config_.sensitivity); // From MEMORIES
      LoadAccumulators (*store_, *context_);                       // From ACCUMULATORS
    }

  StartNewEpisode ();
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
  context_->write_rate_window_.Record (signal.timestamp);

  try
    {
      if (tx)
        {
          root_operation_->Execute (op_context, *tx);
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

      if (op_context.ShouldFinalizeEpisode () && tx)
        {
          FinalizeEpisode (*tx);
          StartNewEpisode ();
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
  FinalizeEpisode (*tx);
  PersistState (*tx);           // v2: Unified state
  PersistWorkingMemory (*tx);   // v2: To MEMORIES
  PersistAccumulators (*tx);    // v2: To ACCUMULATORS
  tx->Commit ();
  StartNewEpisode ();
}

void
SignalProcessor::StartNewEpisode ()
{
  // No-op: transaction management is now per-signal in Process()
}

void
SignalProcessor::FinalizeEpisode (Transaction &tx)
{
  (void)tx;
  telemetry::ScopedSpan span ("cortext.episode.finalize");

  // v2 schema: recent_context and recent_scores are now VIEWs that derive
  // from signals/embeddings tables. No need to persist - data comes from
  // MemoryStorage operation which writes to signals table.
  //
  // Note: State, WM, and Accumulators persisted by caller (Flush/Process)

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

  std::vector<float> coeff_blob;
  if (!context_->rls_coefficients.empty ())
    {
      coeff_blob.reserve (ProcessorContext::kNumMetrics
                          * ProcessorContext::kCoeffsPerMetric);
      for (const auto &metric_coeffs : context_->rls_coefficients)
        {
          for (double c : metric_coeffs)
            coeff_blob.push_back (static_cast<float> (c));
        }
    }

  std::vector<float> coeff_P_blob;
  if (!context_->rls_coeff_P.empty ())
    coeff_P_blob = SerializeMatrix (context_->rls_coeff_P);

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
      "emotion_intensity, valence, arousal, mood_vector, "
      // Stability state
      "rate_decay, periphery_half_life, salience_half_life, drift_weight, retention_ema, "
      // Rate control
      "m_rate, dt_ema, rate_ticks, last_rate_timestamp, reliability, "
      // Uncertainty
      "u_uncertainty, "
      // Embedding prediction
      "last_embedding, delta_x_trend, delta_half_life_adj, sustained_influence, "
      // Episode tracking
      "last_signal_timestamp, updated_at, "
      // Blender weights
      "w_relevance, w_mismatch, w_surprise, w_rarity, w_drift, w_contradiction, "
      "w_utility, w_periphery, w_coverage, w_salience, w_valence, w_arousal, "
      "blender_ready, blender_update_count, "
      // Blender matrices
      "blender_P_matrix, blender_coefficients, blender_coeff_P_matrix) "
      "VALUES (1, ?, "
      "?, ?, ?, ?, "  // Threshold
      "?, ?, ?, ?, "  // Focus
      "?, ?, ?, ?, ?, ?, ?, "  // Sensitivity
      "?, ?, ?, ?, "  // Emotion
      "?, ?, ?, ?, ?, "  // Stability
      "?, ?, ?, ?, ?, "  // Rate control
      "?, "  // Uncertainty
      "?, ?, ?, ?, "  // Embedding prediction
      "?, ?, "  // Episode tracking
      "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "  // Blender weights (12)
      "?, ?, "  // Blender ready/count
      "?, ?, ?)",  // Blender matrices (3)
      {
        context_->signals_processed,
        // Threshold state
        context_->T_dynamic, context_->T_dynamic, context_->hysteresis,
        context_->half_life,
        // Focus state
        context_->weight_relevance, context_->attention_width,
        0.65, 0.5,  // coverage_gain_floor, mismatch_weight defaults
        // Sensitivity state
        context_->weight_novelty, 0.2, 0.4, 0.0,  // weight_surprise/valence/arousal
        1.0, 1.0, context_->rate_target,  // emotion_gain, score_gain, rate_target
        // Emotion state
        context_->emotion_intensity_ewma, context_->valence_ewma,
        context_->arousal_ewma, mood_blob,
        // Stability state
        context_->rate_decay, context_->periphery_half_life,
        context_->salience_half_life, 0.5, 0.0,  // drift_weight, retention_ema defaults
        // Rate control
        context_->m_rate, context_->dt_ema, context_->rate_ticks,
        static_cast<long long> (context_->last_rate_timestamp), 1.0,  // reliability
        // Uncertainty
        context_->u_t,
        // Embedding prediction
        context_->last_embedding.has_value ()
            ? std::any (ToFloatVector (*context_->last_embedding))
            : std::any (std::vector<float> ()),
        context_->delta_x_trend.has_value ()
            ? std::any (ToFloatVector (*context_->delta_x_trend))
            : std::any (std::vector<float> ()),
        0.0, context_->sustained_influence,  // delta_half_life_adj
        // Episode tracking
        static_cast<long long> (context_->last_signal_timestamp), now_ms,
        // Blender weights
        w_relevance, w_mismatch, w_surprise, w_rarity, w_drift, w_contradiction,
        w_utility, w_periphery, w_coverage, w_salience, w_valence, w_arousal,
        // Blender ready/count
        context_->blender_ready ? 1 : 0, context_->blender_update_count,
        // Blender matrices
        P_blob.empty () ? std::any (std::vector<float> ()) : std::any (P_blob),
        coeff_blob.empty () ? std::any (std::vector<float> ()) : std::any (coeff_blob),
        coeff_P_blob.empty () ? std::any (std::vector<float> ()) : std::any (coeff_P_blob)
      });
}

void
SignalProcessor::PersistRecentContext (Transaction & /*tx*/)
{
  // v2 schema: recent_context is now a VIEW derived from signals/embeddings.
  // Data is written by MemoryStorage operation when signals are stored.
  // This function is a no-op for backwards compatibility.
}

void
SignalProcessor::PersistRecentScores (Transaction & /*tx*/)
{
  // v2 schema: recent_scores is now a VIEW derived from signals.
  // Data is written by MemoryStorage operation when signals are stored.
  // This function is a no-op for backwards compatibility.
}

void
SignalProcessor::PersistObservedRetentionHistory (Transaction & /*tx*/)
{
  // v2 schema: observed_retention_history is derived from signals.
  // This function is a no-op for backwards compatibility.
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

      const auto ts_ms = static_cast<int64_t> (slot.last_ts * 1000.0);

      // For now, we don't have embedding_id - in full implementation this would
      // be set when the memory is created. Using 0 as placeholder.
      tx.Execute (
          "INSERT INTO memories "
          "(source_id, kind, modality, start_ts, n_signals, s_max, s_avg, "
          " strength, last_access, created_at) "
          "VALUES (?, 'WORKING', ?, ?, ?, ?, ?, ?, ?, ?)",
          { slot.source_id.empty () ? std::string ("unknown") : slot.source_id,
            slot.modality, slot.start_ts, slot.n_signals, slot.s_max, slot.s_avg,
            slot.strength, ts_ms, now_ms });
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

      tx.Execute (
          "INSERT INTO accumulators "
          "(source_id, episode_id, mu_acc, drift_acc, s_sum, s_max, "
          " e_peak, t_start, last_write_ts, last_signal_ts, eta_acc, "
          "coherence_prev) "
          "VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
          { source_id, state.episode_id, mu_blob, state.drift_acc, state.s_sum,
            state.s_max, peak_blob,
            static_cast<long long> (state.t_start),
            static_cast<long long> (state.last_write_ts),
            static_cast<long long> (state.last_signal_ts), state.eta_acc,
            state.coherence_prev });
    }
}

} // namespace cortext
