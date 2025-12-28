#pragma once

#include "cortext/data/centroids.hpp"
#include "cortext/operations/extraction.hpp"
#include "cortext/operations/metrics.hpp"
#include "cortext/processor/accumulator_state.hpp"
#include "cortext/signal.hpp"
#include <Eigen/Dense>
#include <array>
#include <deque>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cortext
{

// Forward declarations for LLM components
class Extractor;
class Summarizer;

/// @brief Holds the long-lived, evolving state of the SignalProcessor.
///
/// This includes all dynamic variables such as EWMAs, rolling windows,
/// dynamic thresholds, and counters that persist across multiple signals.
///
/// State is organized into logical groups:
/// - General: signals_processed, u_t, initialization flags
/// - Focus: weight_relevance, attention_width, recent_context (Alg 1, 2, 15)
/// - Sensitivity: base_rate, emotion weights/centroids, rate_target (Alg 3, 4, 16)
/// - Stability: half_lives, hysteresis, retention history (Alg 5, 6, 17)
/// - Threshold: T_dynamic, recent_scores, rate control (Alg 8)
/// - Blender: metric weights, RLS state (Alg 7)
/// - Consolidation: last_consolidation_ts, last_retrieval_ts (Alg 28)
/// - Influence: sustained_influence (Alg 19)
/// - Working Memory: wm_slots, acceptance state (Alg 24)
///
/// Note: This struct could be refactored into smaller sub-structs (e.g.,
/// FocusState, SensitivityState, StabilityState) to improve encapsulation,
/// but this would be a breaking change requiring updates to all operations.
struct ProcessorContext
{
  // --- Observed write-rate window ---
  class WriteRateWindow
  {
  public:
    WriteRateWindow () : capacity_ (30) {}
    explicit WriteRateWindow (size_t cap) : capacity_ (cap) {}
    void
    SetCapacity (size_t cap)
    {
      capacity_ = (cap == 0) ? 1 : cap;
      if (!timestamps_.empty ())
        {
          const uint64_t latest = timestamps_.back ();
          const uint64_t window_ms = static_cast<uint64_t> (capacity_) * 1000ULL;
          while (!timestamps_.empty ()
                 && latest > timestamps_.front ()
                        && (latest - timestamps_.front ()) > window_ms)
            {
              timestamps_.pop_front ();
            }
        }
      cached_rate_.reset ();
    }
    void
    Record (uint64_t ts)
    {
      if (!timestamps_.empty () && ts <= timestamps_.back ())
        {
          return; // ignore non-increasing timestamps
        }
      timestamps_.push_back (ts);
      const uint64_t window_ms = static_cast<uint64_t> (capacity_) * 1000ULL;
      while (!timestamps_.empty ()
             && ts > timestamps_.front ()
                    && (ts - timestamps_.front ()) > window_ms)
        {
          timestamps_.pop_front ();
        }
      cached_rate_.reset ();
    }
    double
    RatePerMinute (double alpha) const
    {
      if (cached_rate_.has_value ())
        {
          return *cached_rate_;
        }
      if (timestamps_.size () < 2)
        {
          cached_rate_ = 0.0;
          return 0.0;
        }
      // Compute EWMA of intervals (seconds)
      constexpr double kMillisToSeconds = 1e-3;
      constexpr double kMinInterval = 0.1;
      constexpr double kMaxInterval = 10.0;
      const double a = (alpha <= 0.0) ? 0.25 : (alpha >= 1.0 ? 1.0 : alpha);
      double ema = 0.0;
      bool first = true;
      for (size_t i = 1; i < timestamps_.size (); ++i)
        {
          double dt = static_cast<double> (timestamps_[i] - timestamps_[i - 1])
                      * kMillisToSeconds;
          if (dt < kMinInterval)
            dt = kMinInterval;
          if (dt > kMaxInterval)
            dt = kMaxInterval;
          if (first)
            {
              ema = dt;
              first = false;
            }
          else
            {
              ema = (1.0 - a) * ema + a * dt;
            }
        }
      const double eps = 1e-9;
      const double rate = (ema > eps) ? (60.0 / ema) : 0.0;
      cached_rate_ = std::isfinite (rate) ? rate : 0.0;
      return *cached_rate_;
    }
    std::vector<uint64_t>
    GetTimestamps () const
    {
      return std::vector<uint64_t> (timestamps_.begin (), timestamps_.end ());
    }
    void
    SetTimestamps (const std::vector<uint64_t> &timestamps)
    {
      timestamps_.assign (timestamps.begin (), timestamps.end ());
      if (!timestamps_.empty ())
        {
          const uint64_t latest = timestamps_.back ();
          const uint64_t window_ms = static_cast<uint64_t> (capacity_) * 1000ULL;
          while (!timestamps_.empty ()
                 && latest > timestamps_.front ()
                        && (latest - timestamps_.front ()) > window_ms)
            {
              timestamps_.pop_front ();
            }
        }
      cached_rate_.reset ();
    }

  private:
    std::deque<uint64_t> timestamps_;
    size_t capacity_;
    mutable std::optional<double> cached_rate_;
  };

  // --- Recent ID LRU for novelty (Algorithm 27 helper) ---
  class RecentIdsLru
  {
  public:
    explicit RecentIdsLru (size_t cap = 1024) : capacity_ (cap) {}
    void
    SetCapacity (size_t cap)
    {
      capacity_ = (cap == 0) ? 1 : cap;
      while (queue_.size () > capacity_)
        {
          const long long old = queue_.front ();
          queue_.pop_front ();
          set_.erase (old);
        }
    }
    void
    RecordIds (const std::vector<long long> &ids)
    {
      for (const long long id : ids)
        {
          if (set_.find (id) != set_.end ())
            {
              continue;
            }
          set_.insert (id);
          queue_.push_back (id);
          if (queue_.size () > capacity_)
            {
              const long long old = queue_.front ();
              queue_.pop_front ();
              set_.erase (old);
            }
        }
    }
    const std::unordered_set<long long> &
    GetIdSet () const
    {
      return set_;
    }
    size_t
    Size () const
    {
      return set_.size ();
    }

  private:
    std::unordered_set<long long> set_;
    std::deque<long long> queue_;
    size_t capacity_;
  };

  // ======================================================================
  // General State
  // ======================================================================
  int signals_processed = 0;
  double u_t = 0.0;
  double outcome_pred = 0.0;
  double delta_reward = 0.0;
  double neuromod_ach = 0.0;
  double neuromod_ne = 0.0;
  double neuromod_da = 0.0;
  double encode_bias = 0.5;
  double retrieval_bias = 0.5;
  double osc_phase = 0.0;
  double last_used_rate = 0.0;
  double last_used_flag = 0.0;
  bool focus_priors_initialized = false;
  bool sensitivity_priors_initialized = false;
  bool stability_priors_initialized = false;
  int last_interrupt_tick = -1000000;
  int blender_update_count = 0;
  WriteRateWindow write_rate_window_;
  RecentIdsLru recent_ids_lru_;

  // ======================================================================
  // Embedding Prediction Error State (Section 3.1.4)
  // ======================================================================
  /// @brief Previous signal embedding (optional utility)
  std::optional<Eigen::VectorXf> last_embedding;
  /// @brief EMA expectation state for prediction error
  std::optional<Eigen::VectorXf> x_pred_ema;
  /// @brief Prediction error SSE history for ΔSSE utility
  std::optional<double> prediction_error_sse;
  std::optional<double> prediction_error_sse_prev;

  // ======================================================================
  // Focus-Related State (Algorithms 1, 2, 15)
  // ======================================================================
  double weight_relevance_prior = 0.5;
  double coverage_gain_floor_prior = 0.65;
  double mismatch_weight_prior = 0.5;
  double attention_width_prior = 1.57;
  double weight_relevance = 0.5;
  double attention_width = 1.57;
  double coverage_gain_floor = 0.65;
  double mismatch_weight = 0.5;
  std::deque<Eigen::VectorXf> recent_context_embeddings;

  // ======================================================================
  // Sensitivity-Related State (Algorithms 3, 4, 16)
  // ======================================================================
  double base_rate_prior = 0.2;
  double weight_novelty_prior = 0.3;
  double weight_surprise_prior = 0.2;
  double weight_valence_prior = 0.4;
  double weight_arousal_prior = 0.0;
  double weight_emotion_prior = 0.2;
  double emotion_gain_prior = 1.0;
  double score_gain_prior = 1.0;
  double rate_target_prior = 0.2;
  std::optional<data::Centroids> centroids;
  double rate_target = 0.0;
  uint64_t last_signal_timestamp = 0;
  double weight_novelty = 0.3;
  double weight_surprise = 0.2;
  double weight_valence = 0.4;
  double weight_arousal = 0.0;
  double emotion_gain = 1.0;
  double score_gain = 1.0;

  // Emotion state (Algorithm 4, persisted EWMA values)
  double emotion_intensity_ewma = 0.0;
  double valence_ewma = 0.5;
  double arousal_ewma = 0.0;

  // Mood state (Algorithm 4b, persisted tonic state)
  // Order: [anger, fear, joy, love, sadness, surprise]
  std::array<double, 6> mood_vector = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  uint64_t last_mood_ts = 0;

  // ======================================================================
  // Stability-Related State (Algorithms 5, 6, 17)
  // ======================================================================
  double hysteresis_band_prior = 0.02;
  double half_life_prior = 120.0;
  double rate_decay_prior = 0.60;
  double periphery_half_life_prior = 120.0;
  double salience_half_life_prior = 120.0;
  double drift_weight_prior = 0.5;
  std::deque<double> observed_retention_history;
  double half_life = 120.0;
  double rate_decay = 0.60;
  double periphery_half_life = 120.0;
  double salience_half_life = 120.0;
  double drift_weight = 0.5;
  double retention_ema = 0.0;
  double delta_half_life_adj = 0.0;

  // ======================================================================
  // Threshold & Score Tracking State (Algorithm 8)
  // ======================================================================
  std::deque<double> recent_scores;
  double T_dynamic = 0.2;
  double T_target = 0.2;
  double hysteresis = 0.05;
  double dt_ema = 0.0;
  double m_rate = 0.0;
  double rho_hat_prev = 0.0;
  int rate_ticks = 0;
  uint64_t last_rate_timestamp = 0;
  double reliability = 1.0;  // ESS-based reliability measure

  // ======================================================================
  // Metric Weight Blending State (Algorithm 7)
  // ======================================================================
  std::unordered_map<operations::Metric, double> blender_state;
  std::vector<operations::Metric> blender_order;
  std::vector<std::vector<double> > blender_P;
  bool blender_ready = false;

  // ======================================================================
  // Consolidation State (Algorithms 28, 28b)
  // ======================================================================
  uint64_t last_consolidation_ts = 0;
  uint64_t last_retrieval_ts = 0;
  int consolidation_count = 0;
  bool is_processing_signal = false;

  // ======================================================================
  // Metacognitive State (Section 6.2)
  // ======================================================================
  double fok_state = 0.0;
  double retrieval_strength = 0.0;
  double metacognitive_confidence = 0.0;

  // ======================================================================
  // Extraction State (Section 7.4)
  // ======================================================================
  uint64_t last_extraction_ts = 0;
  std::vector<operations::ExtractionResult> pending_extraction_results;

  // ======================================================================
  // Episode Tracking State (Algorithm 12)
  // ======================================================================
  uint64_t episode_start_ts = 0;

  // ======================================================================
  // Influence Feedback State (Algorithm 19)
  // ======================================================================
  double sustained_influence = 0.0;

  // ======================================================================
  // Working Memory State (Algorithm 24, Section 6.1)
  // ======================================================================
  /// @brief Working memory slot holding a coherent memory unit (Section 6.1.1)
  struct WMSlot
  {
    // v2 persistence fields
    int64_t memory_id = 0;               ///< DB row ID (for updates)
    std::string source_id;               ///< Signal source (e.g., "chat/user")
    std::vector<std::vector<unsigned char>> blob_ids;  ///< Content refs for hydration
    std::string modality = "text";       ///< Content type ("text", "audio", "image")
    int64_t start_ts = 0;                ///< Memory start timestamp (ms)

    // Core slot state
    Eigen::VectorXf embedding;           ///< e_rep (representative embedding)
    double strength = 0.0;               ///< Slot activation strength
    double last_ts = 0.0;                ///< Last access timestamp (seconds)
    int pos_index = 0;                   ///< Position index

    // Memory-level metadata (Section 6.1.1)
    int n_signals = 0;                   ///< Number of signals in memory
    double s_max = 0.0;                  ///< Max signal score
    double s_avg = 0.0;                  ///< Average signal score
    double drift_acc = 0.0;              ///< Accumulated drift (D_acc)
    double mem_elapsed = 0.0;            ///< Memory duration in seconds
    double s_emotion_max = 0.0;          ///< Max emotion intensity
    double s_arousal_avg = 0.0;          ///< Average arousal

    std::vector<SignalRecord> signal_records;  ///< Ordered signal records
  };
  std::vector<WMSlot> wm_slots;
  bool wm_last_accepted = false;
  bool wm_last_chunked = false;

  // ======================================================================
  // Memory Accumulation State (Section 4.4)
  // ======================================================================
  /// @brief Per-source-stream accumulators for memory formation.
  ///
  /// Maps source_id → AccumulatorState. Each stream accumulates signals into
  /// coherent memories before making write decisions. This implements
  /// Event Segmentation Theory (Zacks & Swallow, 2007) for grouping
  /// signals into natural "thought units".
  std::unordered_map<std::string, AccumulatorState> accumulator_states;

  /// @brief Stream of written memory representatives (e_rep).
  std::deque<Eigen::VectorXf> memory_stream;

  /// @brief Recent memory centroids for interrupt gate context (Section 8.2)
  ///
  /// Stores μ_acc from recently written memories. Used by interrupt gate
  /// to compute context centroid for novelty and marginal utility.
  /// ctx_window ← recent_memory_centroids (not individual signals)
  std::deque<Eigen::VectorXf> recent_memory_centroids;

  // ======================================================================
  // Index and Procedural Stores (CLS extensions)
  // ======================================================================
  std::unordered_map<std::string, std::vector<long long>> index_store;
  std::unordered_map<long long, std::string> index_reverse;
  std::unordered_map<std::string, std::unordered_map<long long, double>> procedural_store;

  // ======================================================================
  // LLM Components (OGA/Phi-4)
  // ======================================================================
  /// @brief Extractor for label/relation extraction (optional, may be null)
  Extractor *extractor = nullptr;
  /// @brief Summarizer for text/audio summarization (optional, may be null)
  Summarizer *summarizer = nullptr;
};

} // namespace cortext
